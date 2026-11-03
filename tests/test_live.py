import json
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch

from benchtop.live import Handler, LiveMonitor, atomic_json
import os
from benchtop.optics import ASM, optimize_phase
from benchtop.rgb_run import train


def configuration():
    return dict(slm_shape=[16, 24], roi_shape=[8, 12], slm_pitch_m=6.4e-6,
                relay_magnification=1., distance_m=.015, wavelength_m=520e-9,
                fourier_aperture=.5, invert_phase=True, seed=42)


class LiveTests(unittest.TestCase):
    def test_windows_sharing_retry_and_telemetry_failure_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'atomic.json'
            replace = os.replace
            attempts = []
            def busy_once(a, b):
                attempts.append(1)
                if len(attempts) == 1:
                    raise PermissionError('Windows reader holds previous state')
                return replace(a, b)
            with patch('benchtop.live.os.replace', side_effect=busy_once):
                atomic_json(target, {'complete': True})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(target.read_text()), {'complete': True})
            live = LiveMonitor(Path(tmp)/'run', 'train')
            with patch('benchtop.live.atomic_json', side_effect=OSError('disk unavailable')):
                live.message('must not interrupt optimizer')
                live.finish()
            self.assertTrue(live.disabled)
            self.assertFalse(live.worker.is_alive())

    def test_citl_snapshots_match_actual_measurement_before_update(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            live = LiveMonitor(tmp, 'citl')
            config = configuration()
            target = torch.full((8, 12), .6)
            key = live.start('citl', 'green', 2, 'live_camera_citl')
            measurements = []
            def camera(phase):
                measurements.append(phase.copy())
                return np.full((8, 12), .3+len(measurements)*.05, np.float32)
            _, history = optimize_phase(ASM(config), target, config, iterations=2, capture=camera,
                                       fixed_scale=1., trace=live.phase_trace(key, target, 2, every=1))
            live.finish()
            stored = json.loads((live.directory/'state.json').read_text(encoding='utf-8'))
            self.assertEqual(stored['status'], 'completed')
            stream = stored['streams'][key]
            self.assertEqual([r['amplitude_mse'] for r in stream['history']], [r['amplitude_mse'] for r in history])
            self.assertEqual(len(stream['snapshots']), 3)
            files = [i['file'] for s in stream['snapshots'] for i in s['images']]
            self.assertEqual(len(set(files)), len(files))
            self.assertTrue(all((Path(tmp)/f).is_file() for f in files))
            self.assertFalse(any(p.suffix == '.tmp' for p in live.directory.rglob('*')))

    def test_server_reports_stopped_writer_and_rejects_outside_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = LiveMonitor(Path(tmp)/'run', 'train')
            live.stop_event.set()
            live.worker.join()
            atomic_json(live.directory/'heartbeat.json', {'time': 0, 'pid': 0})
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, root=tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            prefix = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(prefix+'/api/state?run=run', timeout=2) as response:
                    state = json.load(response)
                self.assertEqual(state['status'], 'stale')
                with self.assertRaises(HTTPError) as error:
                    urlopen(prefix+'/data/../../private.png', timeout=2)
                self.assertEqual(error.exception.code, 404)
                live.finish(error=RuntimeError('capture failed'))
                with urlopen(prefix+'/api/state?run=run', timeout=2) as response:
                    state = json.load(response)
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['message'], 'capture failed')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_retraining_preserves_source_and_streams_real_epoch_metrics(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            config = configuration()
            for channel in ['red', 'green', 'blue']:
                folder = source/channel
                (folder/'pairs').mkdir(parents=True)
                atomic_json(folder/'config.json', config)
                rows = []
                for i, split in enumerate(['train', 'train', 'val', 'test']):
                    name = f'{i}.npz'
                    np.savez(folder/'pairs'/name, phase=np.zeros(config['slm_shape'], np.float32),
                             amplitude=np.full(config['roi_shape'], .7+.03*i, np.float32))
                    rows.append(dict(file=name, split=split))
                atomic_json(folder/'pairs/manifest.json', dict(mode='REAL_CAMERA', config=config, pairs=rows))
                (folder/'forward_model.pt').write_bytes(b'preserve old checkpoint')
            before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            live = LiveMonitor(output, 'train', source)
            with patch('benchtop.rgb_run.Rectifier') as rectifier:
                rectifier.return_value.source_amplitude.return_value = np.ones(config['slm_shape'], np.float32)
                train(output, 'cpu', epochs=2, monitor=live, source_root=source)
            live.finish()
            self.assertEqual(before, {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()})
            for stream in live.state['streams'].values():
                saved = json.loads((output/stream['channel']/'forward_training.json').read_text())
                self.assertEqual([s['val_mse'] for s in stream['history']], [s['val_mse'] for s in saved['history']])
                self.assertEqual(len(stream['snapshots']), 3)
                self.assertEqual(len(stream['snapshots'][-1]['images']), 8)


if __name__ == '__main__':
    unittest.main()
