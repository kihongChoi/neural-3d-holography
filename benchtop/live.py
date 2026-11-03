"""Append-only training snapshots and a read-only localhost dashboard.

The writer never waits for a browser. Images are immutable and the JSON index is
atomically replaced only after they are saved. Heartbeats are independent of
training iterations, so a stopped process cannot look like a live experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
import uuid

import numpy as np
from PIL import Image


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus can briefly hold a file without FILE_SHARE_DELETE.
    # Keep the previous complete JSON visible and retry the atomic rename.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def kernel_grid(value):
    value = array(value)
    # Conv weights: visualize each output/input kernel, keeping pixel boundaries.
    kernels = value.reshape(-1, *value.shape[-2:])
    rows = (len(kernels)+3)//4
    h, w = kernels.shape[-2:]
    grid = np.zeros((rows*(h+1)-1, 4*(w+1)-1), np.float32)
    for i, kernel in enumerate(kernels):
        y, x = (i//4)*(h+1), (i % 4)*(w+1)
        grid[y:y+h, x:x+w] = kernel
    return grid


class LiveMonitor:
    def __init__(self, root, command, source=None, max_streams=None, max_snapshots=None, max_history=None):
        self.root = Path(root).resolve()
        self.max_streams, self.max_snapshots, self.max_history = max_streams, max_snapshots, max_history
        self.retired_images = []
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
        self.directory = self.root/'live'/self.run_id
        (self.directory/'frames').mkdir(parents=True)
        now = time.time()
        self.state = dict(schema=1, run_id=self.run_id, command=command,
                          experiment=self.root.name, source=str(Path(source or root).resolve()),
                          status='running', started_at=now, updated_at=now,
                          message='실행 준비', active=None, streams={}, sequence=0)
        self.stop_event = threading.Event()
        self.disabled = False
        self._save()
        atomic_json(self.root/'live/current.json', {'run_id': self.run_id})
        self._heartbeat()
        self.worker = threading.Thread(target=self._heartbeats, daemon=True)
        self.worker.start()

    def _heartbeat(self):
        if not self.disabled:
            try:
                atomic_json(self.directory/'heartbeat.json', {'time': time.time(), 'pid': os.getpid()})
            except OSError as exc:
                self._disable(exc)

    def _disable(self, error):
        # Telemetry must never abort a hardware capture or optimizer update.
        self.disabled = True
        self.stop_event.set()
        print(f'Live monitor storage unavailable; training continues: {error}', file=sys.stderr, flush=True)

    def _heartbeats(self):
        while not self.stop_event.wait(2):
            self._heartbeat()

    def _save(self):
        if self.disabled:
            return
        self.state['sequence'] += 1
        self.state['updated_at'] = time.time()
        try:
            atomic_json(self.directory/'state.json', self.state)
        except OSError as exc:
            self._disable(exc)

    def message(self, message):
        self.state['message'] = message
        self._save()

    def start(self, stage, channel, total, mode, label=None):
        key = f'{stage}:{channel}:{self.state["sequence"]}'
        self.state['streams'][key] = dict(stage=stage, channel=channel, total=total,
                                         label=label or stage, mode=mode, history=[], snapshots=[])
        self.state['active'] = key
        self.state['message'] = label or stage
        if self.max_streams:
            while len(self.state['streams']) > self.max_streams:
                oldest = next(iter(self.state['streams']))
                removed = self.state['streams'].pop(oldest)
                self._retire(removed['snapshots'])
        self._save()
        self._prune_images()
        return key

    def _retire(self, snapshots):
        self.retired_images.extend((time.time()+5, image['file']) for snap in snapshots for image in snap['images'])

    def _prune_images(self):
        remaining = []
        frames = (self.directory/'frames').resolve()
        for deadline, relative in self.retired_images:
            if time.time() < deadline:
                remaining.append((deadline, relative))
                continue
            path = (self.root/relative).resolve()
            if path.parent != frames or path.suffix != '.png':
                raise ValueError('Refusing to prune anything outside this monitor session')
            try:
                path.unlink(missing_ok=True)
            except OSError:
                remaining.append((deadline, relative))
        self.retired_images = remaining

    def _image(self, name, values, vmin, vmax, cmap):
        from matplotlib import colormaps
        values = array(values)
        if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] != 3) or not np.isfinite(values).all():
            raise ValueError(f'Invalid live image {name}')
        clipped = float(np.mean((values < vmin) | (values > vmax)))
        normalized = np.clip((values-vmin)/(vmax-vmin), 0, 1)
        if values.ndim == 3:
            if cmap == 'srgb_amplitude':
                intensity = np.clip(values, 0, 1)**2
                normalized = np.where(intensity <= .0031308, 12.92*intensity,
                                      1.055*intensity**(1/2.4)-.055)
            pixels = (normalized*255).astype(np.uint8)
        else:
            pixels = (colormaps[cmap](normalized)[..., :3]*255).astype(np.uint8)
        preview = Image.fromarray(pixels)
        preview.thumbnail((640, 384), Image.Resampling.BILINEAR)
        if min(preview.size) < 64:
            preview = preview.resize((preview.width*12, preview.height*12), Image.Resampling.NEAREST)
        filename = f'frames/{self.state["sequence"]:06d}_{name}.png'
        preview.save(self.directory/filename)
        return dict(name=name, file=f'live/{self.run_id}/{filename}', vmin=vmin, vmax=vmax,
                    cmap=cmap, clipped_fraction=clipped)

    def update(self, key, step, metrics, images=None):
        if self.disabled:
            return
        stream = self.state['streams'][key]
        row = dict(step=int(step), time=time.time(), **metrics)
        # Images are a list of (label, array, minimum, maximum, colormap).
        if images:
            rendered = []
            for index, (label, values, lo, hi, cmap) in enumerate(images):
                try:
                    rendered.append(dict(self._image(f'{index:02d}', values, lo, hi, cmap), label=label))
                except OSError as exc:
                    self._disable(exc)
                    return
            stream['snapshots'].append(dict(step=int(step), metrics=row, images=rendered))
            if self.max_snapshots and len(stream['snapshots']) > self.max_snapshots:
                # Retain the initial snapshot and the most recent snapshots.
                self._retire([stream['snapshots'].pop(1)])
        stream['history'].append(row)
        if self.max_history and len(stream['history']) > self.max_history:
            stream['history'] = stream['history'][:1]+stream['history'][-(self.max_history-1):]
        self.state['active'] = key
        self._save()
        self._prune_images()

    def phase_trace(self, key, target, total, every=5):
        target = array(target)
        def trace(step, row, phase, amplitude, gradient):
            if self.disabled:
                return
            metrics = dict(row)
            metrics.pop('step', None)
            images = None
            if gradient is not None:
                metrics['phase_gradient_rms'] = float(gradient.square().mean().sqrt())
            if step % every == 0 or step == total:
                measured = array(amplitude)
                full_phase = array(phase)
                h, w = target.shape
                y, x = (full_phase.shape[0]-h)//2, (full_phase.shape[1]-w)//2
                images = [('Target amplitude', target, 0., 1.25, 'gray'),
                          ('Camera amplitude' if 'camera' in self.state['streams'][key]['mode'] else 'Model amplitude',
                           measured, 0., 1.25, 'gray'),
                          ('Amplitude error', measured-target, -.5, .5, 'RdBu_r'),
                          ('SLM phase [rad]', np.angle(np.exp(1j*full_phase)), -np.pi, np.pi, 'twilight'),
                          ('ROI phase [rad]', np.angle(np.exp(1j*full_phase[y:y+h, x:x+w])), -np.pi, np.pi, 'twilight')]
                if gradient is not None:
                    grad = array(gradient)[y:y+h, x:x+w]
                    images.append(('Phase gradient (fixed scale)', grad, -1e-5, 1e-5, 'RdBu_r'))
            self.update(key, step, metrics, images)
        return trace

    def model_epoch(self, key, ep, metrics, net, probe, predicted, multiplier):
        if self.disabled:
            return
        self.update(key, ep, metrics, [
            ('Recorded camera amplitude', probe[1], 0., 1.25, 'gray'),
            ('CNN prediction', predicted, 0., 1.25, 'gray'),
            ('Fixed ASM amplitude', probe[0], 0., 1.25, 'gray'),
            ('Prediction error', array(predicted)-array(probe[1]), -.5, .5, 'RdBu_r'),
            ('Learned amplitude multiplier', multiplier, .25, 2., 'viridis'),
            ('First layer kernels', kernel_grid(net.correction[0].weight), -.5, .5, 'RdBu_r'),
            ('Middle layer kernels', kernel_grid(net.correction[2].weight), -.5, .5, 'RdBu_r'),
            ('Last layer kernels', kernel_grid(net.correction[-1].weight), -.25, .25, 'RdBu_r')])

    def rgb_epoch(self, key, ep, metrics, nets, asm, measured, predicted, multiplier):
        images = [('Recorded camera · RGB', measured, 0., 1., 'srgb_amplitude'),
                  ('CNN prediction · RGB', predicted, 0., 1., 'srgb_amplitude'),
                  ('Fixed ASM · RGB', asm, 0., 1., 'srgb_amplitude'),
                  ('Absolute amplitude error · RGB', np.abs(array(predicted)-array(measured)), 0., .5, 'rgb'),
                  ('Learned multiplier · RGB', multiplier, .25, 2., 'rgb')]
        for ch, net in zip(('R','G','B'), nets):
            images.append((f'{ch} first-layer kernels', kernel_grid(net.correction[0].weight), -.5, .5, 'RdBu_r'))
        self.update(key, ep, metrics, images)

    def rgb_trace(self, key, target, every=20):
        target = array(target)
        def trace(step, row, phase, amplitude, gradient):
            if self.disabled:
                return
            metrics = {k:v for k,v in row.items() if k != 'step'}
            if gradient is not None:
                metrics['phase_gradient_rms'] = float(gradient.square().mean().sqrt())
                metrics['phase_gradient_rms_rgb'] = gradient.square().mean((0,1)).sqrt().tolist()
            images = None
            if step % every == 0 or 'stop_reason' in row:
                measured = array(amplitude)
                images = [('Target · RGB', target, 0., 1., 'srgb_amplitude'),
                          ('Camera · RGB' if 'camera' in self.state['streams'][key]['mode'] else 'Model · RGB',
                           measured, 0., 1., 'srgb_amplitude'),
                          ('Absolute amplitude error · RGB', np.abs(measured-target), 0., .5, 'rgb'),
                          ('SLM phase · RGB diagnostic', (array(phase)+np.pi)%(2*np.pi), 0., 2*np.pi, 'rgb')]
            self.update(key, step, metrics, images)
        return trace

    def end_stream(self, key, reason, details=None):
        stream = self.state['streams'][key]
        stream.update(status='stopped' if reason == 'user_stop' else 'completed',
                      stop_reason=reason, termination=details or {}, ended_at=time.time())
        self._save()

    def finish(self, error=None, reason=None):
        self.stop_event.set()
        self.worker.join(timeout=3)
        self.state['status'] = 'stopped' if isinstance(error, InterruptedError) or reason == 'verification_checkpoint' else ('failed' if error else 'completed')
        self.state['stop_reason'] = reason or ('user_stop' if isinstance(error, InterruptedError) else ('error' if error else 'completed'))
        active = self.state.get('active')
        if active and error and 'stop_reason' not in self.state['streams'][active]:
            self.end_stream(active, self.state['stop_reason'], {'error': str(error)})
        self.state['message'] = str(error) if error else '요청한 실행 완료 · 기록을 탐색할 수 있습니다'
        if reason == 'verification_checkpoint':
            self.state['message'] = '검증 checkpoint 저장 · 프로세스 재시작 대기'
        elif reason == 'short_verification':
            self.state['message'] = '짧은 RGB 검증 완료 · 장시간 학습은 중단 상태'
        self.state['ended_at'] = time.time()
        self._heartbeat()
        self._save()


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = Path(root).resolve()
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def send(self, data, kind, status=200):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def json(self, value, status=200):
        self.send(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8', status)

    def safe_path(self, relative):
        result = (self.root/relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError('Path outside results directory')
        return result

    def do_GET(self):
        try:
            url = urlparse(self.path)
            if url.path == '/':
                return self.send(Path(__file__).with_name('live.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path == '/paper/':
                return self.send(Path(__file__).with_name('paper_live.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path in ('/api/paper', '/api/paper-image'):
                from .paper_view import completed_data, summary, image_bytes
                query = parse_qs(url.query)
                data = completed_data(self.root, query.get('run', [''])[0])
                if url.path == '/api/paper':
                    return self.json(summary(data))
                return self.send(image_bytes(data, query.get('target', [''])[0],
                                             query.get('image', [''])[0]), 'image/webp')
            if url.path == '/learning/':
                guide = self.root.parent/'docs/learning-20261024/neural-3d-holography-learning.html'
                return self.send(guide.read_bytes(), 'text/html; charset=utf-8')
            if url.path == '/api/runs':
                runs = []
                for path in self.root.glob('*/live/current.json'):
                    try:
                        run_id = json.loads(path.read_text(encoding='utf-8'))['run_id']
                        state = json.loads((path.parent/run_id/'state.json').read_text(encoding='utf-8'))
                        runs.append(dict(name=path.parent.parent.name, run_id=run_id,
                                         started_at=state['started_at'], status=state['status']))
                    except (OSError, ValueError, KeyError):
                        continue
                return self.json(sorted(runs, key=lambda r: r['started_at'], reverse=True))
            if url.path == '/api/state':
                name = parse_qs(url.query).get('run', [''])[0]
                if not name or Path(name).name != name:
                    return self.json({'error': 'Unknown experiment'}, 404)
                live = self.safe_path(name+'/live')
                pointer = json.loads((live/'current.json').read_text(encoding='utf-8'))
                directory = self.safe_path(str(Path(name)/'live'/pointer['run_id']))
                state = json.loads((directory/'state.json').read_text(encoding='utf-8'))
                heartbeat = json.loads((directory/'heartbeat.json').read_text(encoding='utf-8'))
                from .paper_view import RUN
                if name == RUN and (self.root/name/'completed_all.json').exists():
                    completed = json.loads((self.root/name/'completed_all.json').read_text(encoding='utf-8'))
                    state.update(status=completed['status'], updated_at=completed['updated_at'],
                                 ended_at=completed['updated_at'], result_viewer='/paper/?run='+name,
                                 message='복소장 모델 학습 및 최종 재촬영 평가 완료')
                state['heartbeat_age_s'] = max(0, time.time()-heartbeat['time'])
                if state['status'] == 'running' and state['heartbeat_age_s'] > 10:
                    state['status'] = 'stale'
                return self.json(state)
            if url.path in ('/api/samples', '/api/preview'):
                query = parse_qs(url.query)
                name = query.get('run', [''])[0]
                if not name or Path(name).name != name:
                    return self.json({'error':'Unknown experiment'},404)
                from .preview import catalog, manager
                if url.path == '/api/samples':
                    return self.json(catalog(self.safe_path(name)))
                split = query.get('split', ['train'])[0]
                if split not in ('train','validation'):
                    return self.json({'error':'Unknown preview split'},400)
                serial = query.get('sample', ['auto'])[0]
                return self.json(manager(self.root).request(name,split,serial))
            if url.path.startswith('/data/'):
                relative = unquote(url.path[len('/data/'):])
                path = self.safe_path(relative)
                parts = Path(relative).parts
                if len(parts) < 5 or parts[1] not in ('live','preview_cache') or path.suffix != '.png':
                    return self.json({'error': 'Not a live image'}, 404)
                return self.send(path.read_bytes(), 'image/png')
            return self.json({'error': 'Not found'}, 404)
        except (ValueError, OSError, KeyError) as exc:
            self.json({'error': str(exc)}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results', help='Parent directory of experiment folders')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--host', action='append', help='Repeat to bind localhost and a specific LAN address; default: localhost only')
    args = parser.parse_args()
    servers = [ThreadingHTTPServer((host, args.port), partial(Handler, root=args.root))
               for host in (args.host or ['127.0.0.1'])]
    for server in servers:
        print(f'Live monitor: http://{server.server_address[0]}:{args.port}/ (read-only)', flush=True)
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == '__main__':
    main()
