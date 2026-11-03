from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from benchtop.campaign import Campaign, StorageBudget, create_plan, decode_phase, remove_owned, next_evaluation_id
from benchtop.hardware import encode_phase
from benchtop.live import LiveMonitor
from benchtop.campaign_report import jsonl, evaluation_means


class CampaignTests(unittest.TestCase):
    def test_resume_allocates_past_failed_partial_camera_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'evaluation.jsonl').write_text('{"evaluation":92}\n')
            (root/'citl.jsonl').write_text('{"evaluation":93,"channel":"red"}\n')
            self.assertEqual(next_evaluation_id(root, {'evaluations':93}), 94)
            self.assertEqual(next_evaluation_id(root, {'evaluations':93,'next_evaluation_id':96}), 96)
    def test_report_handles_live_last_line_without_accepting_broken_completed_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metrics.jsonl'
            path.write_text('{"epoch":0}\n{"epoch":', encoding='utf-8')
            self.assertEqual(jsonl(path, allow_partial=True), [{'epoch':0}])
            with self.assertRaises(json.JSONDecodeError):
                jsonl(path)
            rows = [{'variant':name,'amplitude_mse':value} for name in ('asm','cnn','cnn_citl') for value in (1.,2.,3.)]
            self.assertEqual(evaluation_means({'repeats':rows}), {'asm':2.,'cnn':2.,'cnn_citl':2.})

    def test_all_hdmi_codes_have_stable_canonical_encoding(self):
        codes = np.arange(256, dtype=np.uint8)
        for inverted in (True, False):
            canonical = encode_phase(decode_phase(codes, inverted), inverted)
            np.testing.assert_array_equal(encode_phase(decode_phase(canonical, inverted), inverted), canonical)
            np.testing.assert_allclose(np.exp(1j*decode_phase(codes, inverted)),
                                       np.exp(1j*decode_phase(canonical, inverted)), atol=1e-6)

    def test_split_isolation_and_disk_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, count in [('train', 40), ('valid', 40)]:
                folder = root/f'DIV2K_{name}_HR'
                folder.mkdir()
                for i in range(count):
                    (folder/f'{i:04d}.png').write_bytes(b'test path only')
            plan = create_plan(root)
            identities = sum([plan[name] for name in ['train', 'val', 'test', 'tracking', 'final']], [])
            self.assertEqual(len(identities), len(set(identities)))
            self.assertEqual(len(plan['final']), 8)
            with self.assertRaises(RuntimeError):
                StorageBudget(root, max_gib=0., min_free_gib=0.).check(True)
            with self.assertRaises(ValueError):
                remove_owned(root/'DIV2K_train_HR/0000.png', root/'DIV2K_valid_HR')
            self.assertTrue((root/'DIV2K_train_HR/0000.png').is_file())

    def test_live_retention_keeps_initial_snapshot_and_deletes_only_owned_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = LiveMonitor(tmp, 'test', max_streams=2, max_snapshots=2, max_history=3)
            for i in range(4):
                key = live.start('cnn', 'red', 4, 'recorded_camera_cnn')
                for step in range(5):
                    live.update(key, step, dict(val_mse=.1), [('map', np.ones((4,4)), 0., 2., 'gray')])
                self.assertEqual([s['step'] for s in live.state['streams'][key]['snapshots']], [0,4])
                self.assertEqual([r['step'] for r in live.state['streams'][key]['history']], [0,3,4])
            self.assertEqual(len(live.state['streams']), 2)
            with patch('benchtop.live.time.time', return_value=10**12):
                live._prune_images()
            self.assertEqual(len(list((live.directory/'frames').glob('*.png'))), 4)
            live.finish()

    def test_rgb_fit_uses_train_and_validation_only_and_retains_reloadable_models(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = Campaign.__new__(Campaign)
            worker.directory = root
            worker.args = Namespace(device='cpu', epochs=2)
            worker.state = dict(cycle=0)
            worker.guard = lambda:None
            worker.progress = lambda message:None
            config = dict(slm_shape=[16,24], roi_shape=[8,12], slm_pitch_m=6.4e-6,
                          relay_magnification=1., distance_m=.015, wavelength_m=520e-9,
                          fourier_aperture=.5, invert_phase=True, seed=42)
            worker.configs = [dict(config, channel=ch) for ch in ('red','green','blue')]
            worker.rectifiers = [Mock(source_amplitude=lambda:np.ones((16,24), np.float32)) for _ in range(3)]
            for name in ('pool','models','weights'):
                (root/name).mkdir()
            worker.pool = []
            for i, split in enumerate(['train','train','val','test']):
                filename = f'{i}.npz'
                if split != 'test':
                    np.savez(root/'pool'/filename, phase_rgb_u8=np.full((16,24,3), 127, np.uint8),
                             amplitude_rgb=np.full((8,12,3), .7+.02*i, np.float16))
                # Missing test file deliberately proves fitting never reads sealed test.
                worker.pool.append(dict(file=filename, serial=i, split=split))
            worker.monitor = LiveMonitor(root, 'fit test')
            try:
                worker.fit()
                for ch in ('red','green','blue'):
                    ckpt = torch.load(root/'models'/f'{ch}_best.pt', weights_only=True)
                    self.assertEqual(ckpt['training_sample_serials'], [0,1])
                    self.assertLess(ckpt['validation_mse'], .3)
                    self.assertTrue((root/'models'/f'{ch}_latest.pt').is_file())
                before = (root/'training.jsonl').read_bytes()
                worker.fit()  # A restart must preserve all completed channel work.
                self.assertEqual((root/'training.jsonl').read_bytes(), before)
                first = torch.load(root/'models/rgb_latest.pt', weights_only=True)
                first_steps = [int(v['step']) for v in first['optimizer']['state'].values()]
                worker.state['cycle'] = 1
                worker.fit()  # Resume latest weights with matching optimizer moments.
                second = torch.load(root/'models/rgb_latest.pt', weights_only=True)
                second_steps = [int(v['step']) for v in second['optimizer']['state'].values()]
                self.assertTrue(all(b>a for a,b in zip(first_steps, second_steps)))
                self.assertEqual(second['global_epoch'], 4)
                self.assertEqual(second['samples_seen'], 8)
                self.assertTrue(second['optimizer']['state'])
                self.assertEqual(second['pool_fingerprint'], first['pool_fingerprint'])
                self.assertEqual(worker.monitor.state['streams'][worker.monitor.state['active']]['channel'], 'rgb')
                worker.state['cycle'] = 2
                steps = []
                original_step = torch.optim.Adam.step
                def counted_step(optimizer, *args, **kwargs):
                    value = original_step(optimizer, *args, **kwargs)
                    steps.append(1)
                    return value
                def interrupt_after_pair():
                    if steps:
                        raise InterruptedError('test user stop after one complete RGB pair')
                worker.guard = interrupt_after_pair
                with patch('torch.optim.Adam.step', new=counted_step):
                    with self.assertRaises(InterruptedError):
                        worker.fit()
                partial = torch.load(root/'models/rgb_latest.pt', weights_only=True)
                self.assertEqual(partial['sample_offset'], 1)
                self.assertEqual(partial['samples_seen'], 9)
                self.assertEqual(worker.monitor.state['streams'][worker.monitor.state['active']]['stop_reason'], 'user_stop')
                worker.guard = lambda:None
                worker.fit()
                resumed = torch.load(root/'models/rgb_latest.pt', weights_only=True)
                self.assertEqual(resumed['samples_seen'], 12)
                self.assertEqual(resumed['global_epoch'], 6)
                self.assertTrue(all(int(v['step']) == 12 for v in resumed['optimizer']['state'].values()))
                worker.state['cycle'] = 3
                steps.clear()
                def interrupt_during_validation():
                    if len(steps) == 2:
                        raise InterruptedError('test stop in epoch validation')
                worker.guard = interrupt_during_validation
                with patch('torch.optim.Adam.step', new=counted_step):
                    with self.assertRaises(InterruptedError):
                        worker.fit()
                partial = torch.load(root/'models/rgb_latest.pt', weights_only=True)
                self.assertEqual(partial['sample_offset'], 2)
                self.assertEqual(partial['global_epoch'], 6)
                worker.guard = lambda:None
                worker.fit()
                resumed = torch.load(root/'models/rgb_latest.pt', weights_only=True)
                self.assertEqual(resumed['samples_seen'], 16)
                self.assertEqual(resumed['global_epoch'], 8)
                self.assertTrue(all(int(v['step']) == 16 for v in resumed['optimizer']['state'].values()))
            finally:
                worker.monitor.finish()


if __name__ == '__main__':
    unittest.main()
