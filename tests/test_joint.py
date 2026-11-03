import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchtop.joint import Convergence, optimize_rgb_phase
from benchtop.live import LiveMonitor
from benchtop.optics import ASM


def config():
    return dict(slm_shape=[16,24], roi_shape=[8,12], slm_pitch_m=6.4e-6,
                relay_magnification=1., distance_m=.015, wavelength_m=520e-9,
                fourier_aperture=.5, invert_phase=True, seed=42)


class JointTests(unittest.TestCase):
    def test_convergence_requires_stable_windows_and_does_not_stop_descent_or_growth(self):
        flat = Convergence(minimum=20,window=5,patience=3,relative=.002)
        stopped = [i for i in range(100) if flat.update(i, .1+(-1)**i*1e-6)]
        self.assertTrue(stopped)
        self.assertGreaterEqual(stopped[0], 30)
        for direction in (-1,1):
            gate = Convergence(minimum=20,window=5,patience=3,relative=.002)
            self.assertFalse(any(gate.update(i, 1+direction*.005*i) for i in range(100)))

    def test_camera_rgb_phase_best_measurement_and_reason_are_consistent(self):
        torch.set_num_threads(2)
        configurations = [config() for _ in range(3)]
        models = [ASM(c) for c in configurations]
        target = np.full((8,12,3), .6, np.float32)
        initial = torch.randn(16,24,3)*.1
        displayed = []
        def camera(phase):
            displayed.append(phase.copy())
            value = .4 if len(displayed) == 3 else .2
            return np.full_like(target, value)
        best, history, termination = optimize_rgb_phase(models, target, configurations, initial,
            max_steps=5, capture=camera)
        self.assertEqual(termination['reason'], 'iteration_limit')
        self.assertEqual(termination['best_step'], 2)
        np.testing.assert_allclose(best.numpy(), displayed[2])
        self.assertEqual(len(displayed), 6)
        self.assertFalse(np.array_equal(displayed[0], displayed[1]))
        self.assertEqual(len(history[-1]['amplitude_mse_rgb']), 3)

    def test_convergence_ends_before_cap_and_saves_final_rgb_snapshot(self):
        torch.set_num_threads(2)
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            live = LiveMonitor(tmp, 'RGB convergence test')
            key = live.start('composite_citl','rgb',40,'live_camera_citl')
            target = np.full((8,12,3), .6, np.float32)
            try:
                _, history, end = optimize_rgb_phase([ASM(c) for _ in range(3)], target, [c]*3,
                    torch.zeros(16,24,3), max_steps=40, capture=lambda p:target*.5,
                    convergence=Convergence(minimum=10,window=4,patience=2),
                    trace=live.rgb_trace(key,target,every=20))
                live.end_stream(key,end['reason'],end)
                self.assertEqual(end['reason'],'converged')
                self.assertLess(end['steps'],40)
                snapshots = live.state['streams'][key]['snapshots']
                self.assertEqual(snapshots[-1]['step'], history[-1]['step'])
                self.assertEqual(snapshots[-1]['images'][0]['cmap'], 'srgb_amplitude')
                self.assertEqual(live.state['streams'][key]['stop_reason'], 'converged')
            finally:
                live.finish()


if __name__ == '__main__':
    unittest.main()
