"""Time-bounded RGB acquisition, forward-model fitting and physical evaluation.

Requires a successfully prepared wide-ROI experiment. Keeps a rolling training
pool, immutable validation/test identities, best/latest weights, small metric
logs and bounded live previews. No per-step full-resolution array dumps.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from PIL import Image

from .calibration import Rectifier
from .hardware import Bench, encode_phase
from .live import LiveMonitor, atomic_json
from .optics import crop_center, load_forward_model
from .joint import Convergence, optimize_rgb_phase
from .rgb_run import CHANNELS, capture_amplitude, channel_model, read_json, srgb_preview
from .run import memory, save_image
from .targets import prepare_image


def decode_phase(encoded, inverted=True):
    normalized = np.asarray(encoded, np.float32)/255.
    if inverted:
        normalized = 1-normalized
    return (normalized*2*np.pi-np.pi).astype(np.float32)


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def read_events(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def next_evaluation_id(directory, state):
    """An interrupted camera attempt must never share an id with a later attempt."""
    used = [row['evaluation'] for name in ('evaluation.jsonl', 'citl.jsonl', 'evaluation_attempts.jsonl')
            for row in read_events(Path(directory)/name)]
    return max(state.get('next_evaluation_id', 0), state['evaluations'], max(used, default=-1)+1)


def remove_owned(path, directory):
    """Only unlink a single generated file directly inside the named run folder."""
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if path.parent != directory or not path.is_file():
        raise ValueError(f'Not an owned generated file: {path}')
    path.unlink()


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


class StorageBudget:
    def __init__(self, root, max_gib=12., min_free_gib=50.):
        self.root = Path(root).resolve()
        self.maximum, self.minimum_free = max_gib*2**30, min_free_gib*2**30
        self.last_scan, self.used = 0., 0

    def check(self, force=False):
        free = shutil.disk_usage(self.root).free
        if force or time.monotonic()-self.last_scan > 30:
            self.used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            self.last_scan = time.monotonic()
        if free < self.minimum_free or self.used >= self.maximum:
            raise RuntimeError(f'Storage limit: run={self.used/2**30:.3f} GiB, free={free/2**30:.1f} GiB')
        return dict(run_gib=self.used/2**30, free_gib=free/2**30,
                    maximum_run_gib=self.maximum/2**30, minimum_free_gib=self.minimum_free/2**30)


@contextmanager
def exclusive_run(root):
    """OS-held byte lock is released even if the worker process crashes."""
    import msvcrt
    with (root/'campaign.lock').open('a+b') as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'1')
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def create_plan(dataset, seed=42, heldout=8):
    """Original-image identities never move between fitting/selection/evaluation."""
    dataset = Path(dataset).resolve()
    rng = np.random.default_rng(seed)
    train = list(rng.permutation(sorted((dataset/'DIV2K_train_HR').glob('*.png'))))
    valid = list(rng.permutation(sorted((dataset/'DIV2K_valid_HR').glob('*.png'))))
    if len(train) < 32 or len(valid) < heldout*4:
        raise ValueError('Insufficient DIV2K originals for the frozen campaign split')
    return dict(dataset_root=str(dataset), seed=seed,
                train=[str(p.relative_to(dataset)) for p in train],
                **{name: [str(p.relative_to(dataset)) for p in valid[i*heldout:(i+1)*heldout]]
                   for i, name in enumerate(('val', 'test', 'tracking', 'final'))})


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.out).resolve()
        self.configs = [read_json(self.root/ch/'config.json') for ch in CHANNELS]
        self.base = self.configs[1]
        if not all(c['roi_shape'] == [960, 1680] and c.get('wide_registration') for c in self.configs):
            raise ValueError('Campaign requires verified 1680x960 registration for all RGB channels')
        read_json(self.root/'completed_prepare.json')
        self.rectifiers = [Rectifier(self.root/ch/'calibration', c) for ch, c in zip(CHANNELS, self.configs)]
        self.directory = self.root/'campaign'
        self.directory.mkdir(exist_ok=True)
        for name in ('pool', 'models', 'weights', 'evaluations'):
            (self.directory/name).mkdir(exist_ok=True)
        self.budget = StorageBudget(self.root, args.max_gib, args.min_free_gib)
        self.budget.check(True)
        self.status_file = self.directory/'status.json'
        if self.status_file.exists():
            previous = read_json(self.status_file)
            if not args.resume:
                raise FileExistsError('Existing campaign: inspect its process, then use --resume only if stopped')
            if previous['status'] == 'completed':
                raise FileExistsError('This campaign already completed')
            if previous['requested_hours'] != args.hours:
                raise ValueError('Cannot change the time budget of an existing campaign')
            append_json(self.directory/'resume_history.jsonl', dict(resumed_at=time.time(), previous_status=previous))
            self.state = previous
        else:
            self.state = dict(status='starting', requested_hours=args.hours, active_seconds=0., cycle=0,
                              train_cursor=0, serial=0, created_at=time.time(), evaluations=0)
        self.elapsed_before = self.state['active_seconds']
        self.started = time.monotonic()
        for key in ('error', 'error_type', 'failed_at', 'stop_reason', 'finished_at'):
            self.state.pop(key, None)
        self.state.update(status='running', pid=os.getpid(), process_started_at=time.time())
        self.state['next_evaluation_id'] = next_evaluation_id(self.directory, self.state)
        plan_file = self.directory/'plan.json'
        if plan_file.exists():
            self.plan = read_json(plan_file)
            if str(Path(args.dataset).resolve()) != self.plan['dataset_root']:
                raise ValueError('Dataset path differs from the frozen plan')
        else:
            self.plan = create_plan(args.dataset)
            atomic_json(plan_file, self.plan)
        self.pool_file = self.directory/'pool.json'
        self.pool = read_json(self.pool_file) if self.pool_file.exists() else []
        self.monitor = LiveMonitor(self.root, '10-hour RGB campaign', source=args.dataset,
                                   max_streams=24, max_snapshots=12, max_history=500)
        self.session_id = self.monitor.run_id
        self.state['session_id'] = self.session_id
        self.monitor.state['retention'] = dict(max_streams=24, snapshots_per_stream=12,
                                               rolling_train_pairs=args.pool, archived_raw_frames=False)
        self.progress('준비 완료 · 1680×960 ROI · '+('짧은 RGB 검증' if getattr(args, 'verify_source', None) or getattr(args, 'verify_stage', None) == 'resume' else '장시간 학습 시작'))

    @property
    def elapsed(self):
        return self.elapsed_before+time.monotonic()-self.started

    def progress(self, message):
        storage = self.budget.check()
        self.state.update(active_seconds=self.elapsed, updated_at=time.time(), operation=message,
                          pool_size=len(self.pool), storage=storage, memory=memory())
        atomic_json(self.status_file, self.state)
        self.monitor.message(f'{self.elapsed/3600:.2f}/{self.args.hours:g} h · '+message)
        print(message, flush=True)

    def guard(self):
        self.budget.check()
        if (self.directory/'STOP_REQUESTED').exists():
            raise InterruptedError('Graceful stop requested by STOP_REQUESTED file')

    def target(self, relative):
        amplitude, metadata = prepare_image(Path(self.plan['dataset_root'])/relative, self.base['roi_shape'])
        return amplitude, dict(metadata, target_id=relative)

    def model(self, ci, learned=True):
        path = self.directory/'models'/f'{CHANNELS[ci]}_best.pt'
        if learned and path.exists():
            return load_forward_model(path, self.configs[ci], self.args.device)
        return channel_model(self.configs[ci], self.rectifiers[ci], self.args.device)

    def capture_rgb(self, bench, phase, frames=2):
        bench.show(phase)
        amplitudes = [capture_amplitude(bench, c, r, frames=frames) for c, r in zip(self.configs, self.rectifiers)]
        return np.stack(amplitudes, -1)

    def drain_frames(self, bench, label):
        append_json(self.directory/'frames.jsonl', dict(time=time.time(), label=label,
                                                       session_id=self.session_id, frames=bench.frame_metadata))
        bench.frame_metadata.clear()

    def acquire(self, bench, selections):
        # Hold only three propagation models; targets and measurements are streamed.
        models = [self.model(ci, learned=self.state['cycle'] % 2 == 1) for ci in range(3)]
        try:
            for index, (split, relative) in enumerate(selections):
                self.guard()
                serial = self.state['serial']
                target, source = self.target(relative)
                steps = (40, 60, 80)[serial % 3]
                self.progress(f'cycle {self.state["cycle"]} · 실제 RGB 샘플 {index+1}/{len(selections)} · {split} · {relative}')
                initial = []
                for ci, config in enumerate(self.configs):
                    torch.manual_seed(42000+serial*3+ci)
                    initial.append(.3*torch.randn(config['slm_shape'], device=self.args.device))
                key = self.monitor.start('generate', 'rgb', steps, 'simulation', f'cycle {self.state["cycle"]} · RGB · {relative}')
                phase, history, termination = optimize_rgb_phase(models, target, self.configs, torch.stack(initial,-1),
                    max_steps=steps, lr=.04, guard=self.guard, trace=self.monitor.rgb_trace(key, target, every=20))
                self.monitor.end_stream(key, termination['reason'], termination)
                phases = [phase[...,ci].cpu().numpy() for ci in range(3)]
                losses = history[-1]['amplitude_mse_rgb']
                encoded = encode_phase(np.stack(phases, -1), self.base['invert_phase'])
                # The two endpoint codes represent the same wrapped ideal phase.
                # Canonicalize before both capture and persistence so stored bytes
                # match the HDMI frame exactly, including phase-wrap endpoints.
                encoded = encode_phase(decode_phase(encoded, self.base['invert_phase']), self.base['invert_phase'])
                displayed = decode_phase(encoded, self.base['invert_phase'])
                if not np.array_equal(encode_phase(displayed, self.base['invert_phase']), encoded):
                    raise RuntimeError('Phase encoding round trip did not converge')
                measured = self.capture_rgb(bench, displayed)
                filename = f'sample_{serial:06d}.npz'
                path = self.directory/'pool'/filename
                if path.exists():
                    raise FileExistsError(f'Unindexed sample exists: {path}; inspect it before resuming')
                # Exact HDMI bytes + float16 sensor amplitude, not duplicate float32 phases.
                np.savez_compressed(path, phase_rgb_u8=encoded, amplitude_rgb=measured.astype(np.float16))
                record = dict(file=filename, serial=serial, split=split, source=source,
                              cycle=self.state['cycle'], steps=steps, seed=42000+serial*3,
                              measured_at=time.time(), generator='learned' if self.state['cycle'] % 2 else 'ASM',
                              simulated_final_mse_rgb=losses,
                              measured_target_mse_rgb=np.mean((measured-target)**2, axis=(0, 1)).tolist(),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                self.pool.append(record)
                self.state['serial'] += 1
                if split == 'train':
                    self.state['train_cursor'] += 1
                atomic_json(self.pool_file, self.pool)
                append_json(self.directory/'acquisition.jsonl', record)
                self.drain_frames(bench, filename)
                self.progress(f'실측 저장 완료 · {filename}')
                self.trim_pool()
        finally:
            del models
            if self.args.device == 'cuda':
                torch.cuda.empty_cache()

    def trim_pool(self):
        training = [r for r in self.pool if r['split'] == 'train']
        removed = training[:-self.args.pool] if len(training) > self.args.pool else []
        if removed:
            ids = {r['serial'] for r in removed}
            self.pool = [r for r in self.pool if r['serial'] not in ids]
            atomic_json(self.pool_file, self.pool)
            for row in removed:
                remove_owned(self.directory/'pool'/row['file'], self.directory/'pool')
                append_json(self.directory/'retention.jsonl', dict(time=time.time(), removed=row['file'],
                                                                 reason='rolling training pool', sha256=row['sha256']))

    def drift_check(self, bench):
        measured = self.capture_rgb(bench, np.zeros(tuple(self.base['slm_shape'])+(3,), np.float32), frames=3)
        whites = [float(np.median(measured[..., ci]**2)) for ci in range(3)]
        append_json(self.directory/'drift.jsonl', dict(time=time.time(), cycle=self.state['cycle'], median_white_rgb=whites))
        self.drain_frames(bench, 'flat_field_drift')
        if any(abs(white-1) > .25 for white in whites):
            raise RuntimeError(f'Flat illumination drift exceeds 25% of calibrated white: {whites}')

    def fit(self):
        from .rgb_training import fit_rgb
        return fit_rgb(self)

    def evaluate(self, bench, relative, final=False, baseline=False):
        evaluation_started = time.monotonic()
        self.guard()
        target, source = self.target(relative)
        eid = self.state['next_evaluation_id']
        self.state['next_evaluation_id'] += 1
        append_json(self.directory/'evaluation_attempts.jsonl', dict(evaluation=eid, cycle=self.state['cycle'],
                    target_id=relative, final=final, session_id=self.session_id, started_at=time.time()))
        self.progress(f'{"최종 미사용 타겟" if final else "주기적 실측"} 평가 {eid} · {relative}')
        variants = {}
        initial = []
        for ci, config in enumerate(self.configs):
            torch.manual_seed(91000+int(Path(relative).stem)*3+ci)
            initial.append(.3*torch.randn(config['slm_shape'], device=self.args.device))
        initial = torch.stack(initial, -1)
        for name in ('asm', 'cnn'):
            models = [self.model(ci, learned=name == 'cnn') for ci in range(3)]
            key = self.monitor.start('generate', 'rgb', 80, 'simulation', f'평가 {eid} · {name.upper()} RGB 위상')
            phase, _, termination = optimize_rgb_phase(models, target, self.configs, initial,
                max_steps=80, guard=self.guard, trace=self.monitor.rgb_trace(key, target, every=20))
            self.monitor.end_stream(key, termination['reason'], termination)
            variants[name] = phase.cpu().numpy()
            del models
        steps = getattr(self.args, 'citl_max_steps', 1000)
        policy = Convergence(minimum=getattr(self.args, 'citl_min_steps', 200),
                             window=getattr(self.args, 'citl_window', 20),
                             patience=getattr(self.args, 'citl_patience', 3),
                             relative=getattr(self.args, 'citl_relative', .002),
                             absolute=getattr(self.args, 'citl_absolute', 1e-5))
        models = [self.model(ci) for ci in range(3)]
        key = self.monitor.start('composite_citl', 'rgb', steps, 'live_camera_citl', f'평가 {eid} · RGB 공동 CITL · {relative}')
        self.monitor.state['streams'][key]['stopping_policy'] = dict(maximum=steps, minimum=policy.minimum,
            window=policy.window, patience=policy.patience, relative=policy.relative, absolute=policy.absolute)
        best, history, termination = optimize_rgb_phase(models, target, self.configs, variants['cnn'],
            max_steps=steps, capture=lambda phase:self.capture_rgb(bench, phase), convergence=policy,
            guard=self.guard, trace=self.monitor.rgb_trace(key, target, every=getattr(self.args, 'citl_preview_every', 20)))
        self.monitor.end_stream(key, termination['reason'], termination)
        append_json(self.directory/'citl.jsonl', dict(evaluation=eid, channel='rgb', target_id=relative,
            final=final, session_id=self.session_id, history=history, termination=termination))
        variants['cnn_citl'] = best.cpu().numpy()
        del models
        observations = {name:[] for name in variants}
        rows = []
        # Interleave conditions across repeat blocks, avoiding all-before/all-after drift bias.
        for repeat in range(3):
            for name in np.random.default_rng(500+eid*3+repeat).permutation(list(variants)):
                self.guard()
                amplitude = self.capture_rgb(bench, variants[name], frames=3)
                observations[name].append(amplitude)
                rows.append(dict(variant=name, repeat=repeat,
                                 amplitude_mse=float(np.mean((amplitude-target)**2)),
                                 amplitude_mse_rgb=np.mean((amplitude-target)**2, axis=(0,1)).tolist()))
        result = dict(evaluation=eid, cycle=self.state['cycle'], final=final, source=source,
                      session_id=self.session_id, baseline=baseline,
                      measured_at=time.time(), normalization='fixed per-channel calibration white',
                      roi_shape=self.base['roi_shape'], repeats=rows, citl_termination=termination, model_hashes={
                          ch:hashlib.sha256((self.directory/'models'/f'{ch}_best.pt').read_bytes()).hexdigest() for ch in CHANNELS})
        result['elapsed_seconds'] = time.monotonic()-evaluation_started
        self.state['estimated_evaluation_seconds'] = result['elapsed_seconds']
        append_json(self.directory/'evaluation.jsonl', result)
        key = self.monitor.start('evaluation', 'rgb', 1, 'recorded_camera_evaluation', f'평가 {eid} · 실제 RGB 비교 · 각 3회 평균')
        means = {name:np.mean(values, axis=0) for name,values in observations.items()}
        scores = {name:float(np.mean((mean-target)**2)) for name,mean in means.items()}
        self.monitor.update(key, 1, dict(amplitude_mse=scores['cnn_citl'], comparison_mse=scores,
                            amplitude_mse_rgb=np.mean((means['cnn_citl']-target)**2, axis=(0,1)).tolist()),
            [('Target · RGB', target, 0., 1., 'srgb_amplitude')]+[
             (name.upper()+' camera · RGB', means[name], 0., 1., 'srgb_amplitude') for name in variants])
        self.monitor.end_stream(key, 'evaluation_complete', dict(citl_termination=termination, comparison_mse=scores))
        prefix = ('final' if final else 'tracking')+f'_{eid:04d}'
        folder = self.directory/'evaluations'
        atomic_json(folder/f'{prefix}.json', result)
        save_image(folder/f'{prefix}_target.png', srgb_preview(target))
        for name in variants:
            mean = np.mean(observations[name], axis=0)
            save_image(folder/f'{prefix}_{name}_camera.png', srgb_preview(mean))
            if final:
                # Only final evaluations retain raw rectified RGB arrays and exact display bytes.
                np.savez_compressed(folder/f'{prefix}_{name}.npz', amplitude_rgb=mean.astype(np.float16),
                                    phase_rgb_u8=encode_phase(variants[name], self.base['invert_phase']))
        self.drain_frames(bench, prefix)
        self.state['evaluations'] += 1
        if not final:
            # Two most recent full preview sets; all scalar metrics remain in evaluation.jsonl.
            prefixes = [p.stem for p in sorted(folder.glob('tracking_*.json'))]
            for old in prefixes[:-2]:
                for path in folder.glob(old+'*'):
                    remove_owned(path, folder)
        self.progress(f'실측 평가 {eid} 완료 · ASM/CNN/CITL 각각 3회 측정')
        return result

    def sealed_test(self):
        rows = []
        for ci, ch in enumerate(CHANNELS):
            model = self.model(ci)
            with torch.no_grad():
                for row in self.pool:
                    if row['split'] != 'test':
                        continue
                    with np.load(self.directory/'pool'/row['file']) as pair:
                        phase = torch.from_numpy(decode_phase(pair['phase_rgb_u8'][...,ci], self.base['invert_phase'])).to(self.args.device)
                        camera = torch.from_numpy(pair['amplitude_rgb'][...,ci].astype(np.float32)).to(self.args.device)
                    prediction = model(phase)
                    asm = crop_center(model.asm(phase).abs(), self.base['roi_shape'])
                    rows.append(dict(channel=ch, target_id=row['source']['target_id'],
                                     cnn_mse=float((prediction-camera).square().mean()),
                                     asm_mse=float((asm-camera).square().mean())))
            del model
        atomic_json(self.directory/'sealed_test.json', rows)

    def run(self):
        import ctypes
        # Scoped execution request, restored on exit; does not unlock Windows.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        try:
            with Bench(self.base) as bench:
                existing = {(r['split'], r['source']['target_id']) for r in self.pool}
                pending = [(split, name) for split in ('val', 'test') for name in self.plan[split]
                           if (split, name) not in existing]
                needed = max(0, 32-sum(r['split']=='train' for r in self.pool))
                cursor = self.state['train_cursor']
                pending += [('train', self.plan['train'][(cursor+i) % len(self.plan['train'])]) for i in range(needed)]
                self.drift_check(bench)
                if pending:
                    self.acquire(bench, pending)
                if getattr(self.args, 'citl_first', False) and not (self.directory/'joint_citl_baseline.json').exists():
                    # A real convergence attempt precedes the first long RGB fit.
                    old_finished = {r['source']['target_id'] for r in read_events(self.directory/'evaluation.jsonl')
                                    if not r['final'] and r['cycle'] == self.state['cycle']}
                    targets = [self.plan['tracking'][(self.state['cycle']*2+i) % len(self.plan['tracking'])] for i in range(2)]
                    relative = next((name for name in targets if name not in old_finished), targets[0])
                    baseline = self.evaluate(bench, relative, baseline=True)
                    atomic_json(self.directory/'joint_citl_baseline.json', baseline)
                budget_seconds = self.args.hours*3600
                # Full 1000-step RGB evaluations can take much longer than the
                # old 30-step tests. Reserve time from actual measured duration.
                final_reserve = max(min(600., budget_seconds*.1),
                    self.state.get('estimated_evaluation_seconds', 0)*len(self.plan['final'])*1.2)
                while self.elapsed < budget_seconds-final_reserve or not (self.directory/'models/rgb_latest.pt').exists():
                    self.fit()
                    model_hashes = {ch:hashlib.sha256((self.directory/'models'/f'{ch}_best.pt').read_bytes()).hexdigest()
                                    for ch in CHANNELS}
                    finished = {r['source']['target_id'] for r in read_events(self.directory/'evaluation.jsonl')
                                if not r['final'] and not r.get('baseline') and r['cycle'] == self.state['cycle']
                                and r.get('model_hashes') == model_hashes}
                    for offset in range(2):
                        target_id = self.plan['tracking'][(self.state['cycle']*2+offset) % len(self.plan['tracking'])]
                        if target_id not in finished:
                            self.evaluate(bench, target_id)
                    self.state['cycle'] += 1
                    self.progress(f'cycle {self.state["cycle"]} 완료')
                    final_reserve = max(final_reserve,
                        self.state.get('estimated_evaluation_seconds', 0)*len(self.plan['final'])*1.2)
                    if self.elapsed >= budget_seconds-final_reserve:
                        break
                    self.drift_check(bench)
                    cursor = self.state['train_cursor']
                    self.acquire(bench, [('train', self.plan['train'][(cursor+i) % len(self.plan['train'])])
                                         for i in range(self.args.batch)])
                # Selection is now closed. These original images were never used earlier.
                self.sealed_test()
                final = [self.evaluate(bench, name, final=True) for name in self.plan['final']]
                atomic_json(self.directory/'final_evaluations.json', final)
                self.drain_frames(bench, 'end')
            self.state.update(status='completed', finished_at=time.time(), active_seconds=self.elapsed)
            self.progress('10시간 목표의 학습·평가 실행 종료 · 최종 평가 저장 완료')
            atomic_json(self.directory/'completed.json', dict(self.state, arguments=vars(self.args)))
            self.monitor.finish()
        except BaseException as exc:
            self.state.update(status='stopped' if isinstance(exc, InterruptedError) else 'failed', error=str(exc), error_type=type(exc).__name__, active_seconds=self.elapsed,
                              failed_at=time.time())
            atomic_json(self.status_file, self.state)
            self.monitor.finish(error=exc)
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dataset', default=r'C:\Users\khchoi_exp\Documents\DIV2K dataset')
    parser.add_argument('--hours', type=float, default=10.)
    parser.add_argument('--pool', type=int, default=192)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--min-epochs', type=int, default=50)
    parser.add_argument('--cnn-patience', type=int, default=30)
    parser.add_argument('--citl-max-steps', type=int, default=1000)
    parser.add_argument('--citl-min-steps', type=int, default=200)
    parser.add_argument('--citl-window', type=int, default=20)
    parser.add_argument('--citl-patience', type=int, default=3)
    parser.add_argument('--citl-relative', type=float, default=.002)
    parser.add_argument('--citl-absolute', type=float, default=1e-5)
    parser.add_argument('--citl-preview-every', type=int, default=20)
    parser.add_argument('--verify-source', help='Create an isolated short verification from a stopped measured run')
    parser.add_argument('--verify-stage', choices=('all', 'train', 'resume'), default='all')
    parser.add_argument('--max-gib', type=float, default=12.)
    parser.add_argument('--min-free-gib', type=float, default=50.)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--citl-first', action='store_true', help='Run one real RGB convergence attempt before fitting')
    args = parser.parse_args()
    if args.hours <= 0 or args.pool < 32 or args.batch < 1 or args.epochs < 1:
        parser.error('Positive hours/batch/epochs and pool >= 32 required')
    if not 1 <= args.citl_max_steps <= 1000 or not 0 <= args.citl_min_steps <= args.citl_max_steps:
        parser.error('Require 0 <= CITL min <= max <= 1000')
    if args.min_epochs < 0 or args.cnn_patience < 1 or args.citl_preview_every < 1:
        parser.error('Invalid epoch/patience/preview settings')
    Convergence(args.citl_min_steps, args.citl_window, args.citl_patience, args.citl_relative, args.citl_absolute)
    if args.verify_source:
        from .verify_joint import prepare_verification
        prepare_verification(Path(args.verify_source), Path(args.out))
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with exclusive_run(Path(args.out)):
        campaign = Campaign(args)
        if args.verify_source or args.verify_stage == 'resume':
            from .verify_joint import run_verification
            run_verification(campaign)
        else:
            campaign.run()


if __name__ == '__main__':
    main()
