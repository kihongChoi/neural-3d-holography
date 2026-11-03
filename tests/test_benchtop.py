"""Numerical and acquisition-contract tests, without connected hardware."""
import json
import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
import torch

from benchtop.hardware import Bench, DisplayUnavailableError, encode_phase
from benchtop.optics import ASM, crop_center, optimize_phase, make_target, CompactForwardModel, load_forward_model
from benchtop.calibration import find_focus


class BenchtopTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'configs/benchtop.json').read_text())
        self.config.update(slm_shape=[32, 48], roi_shape=[24, 32], fourier_aperture=1.0)
        torch.manual_seed(42)

    def test_zero_distance_and_center_crop(self):
        config = dict(self.config, distance_m=0)
        model = ASM(config)
        phase = torch.rand(config['slm_shape'])
        torch.testing.assert_close(model(phase, quantize=False), torch.exp(1j*phase), atol=5e-6, rtol=5e-6)
        self.assertEqual(tuple(crop_center(phase, (24, 32)).shape), (24, 32))

    def test_complex_fft_gradient_against_finite_difference(self):
        model = ASM(self.config)
        phase = (torch.randn(self.config['slm_shape'])*0.2).requires_grad_(True)
        weight = torch.rand(self.config['slm_shape'])
        loss = (model(phase, quantize=False).abs().square()*weight).sum()
        loss.backward()
        direction = torch.randn_like(phase)
        analytic = (phase.grad*direction).sum().item()
        with torch.no_grad():
            plus = (model(phase+0.002*direction, quantize=False).abs().square()*weight).sum()
            minus = (model(phase-0.002*direction, quantize=False).abs().square()*weight).sum()
        numeric = ((plus-minus)/0.004).item()
        self.assertAlmostEqual(analytic, numeric, delta=max(0.03, abs(numeric)*0.005))

    def test_phase_lut_round_trip(self):
        phase = np.linspace(-np.pi, np.pi, 1001, endpoint=False, dtype=np.float32)
        for inverted in (False, True):
            encoded = encode_phase(phase, inverted)
            normalized = encoded.astype(np.float32)/255
            if inverted:
                normalized = 1-normalized
            reconstructed = normalized*2*np.pi-np.pi
            error = np.angle(np.exp(1j*(reconstructed-phase)))
            self.assertLess(np.abs(error).max(), np.pi/255+1e-6)

    def test_citl_uses_camera_values_and_preserves_best_phase(self):
        model = ASM(self.config)
        target = make_target(self.config['roi_shape'])
        initial = torch.zeros(self.config['slm_shape'])
        calls = []
        def camera(phase):
            calls.append(phase.copy())
            # First measured image is perfect; later measurements are worse.
            return target.numpy() if len(calls) == 1 else np.ones(target.shape, np.float32)
        best, history = optimize_phase(model, target, self.config, 3, initial=initial, capture=camera)
        self.assertEqual(len(calls), 4)
        self.assertLess(history[0]['amplitude_mse'], 1e-10)
        torch.testing.assert_close(best, initial)
        self.assertGreater(history[-1]['amplitude_mse'], history[0]['amplitude_mse'])

    def test_no_focus_is_rejected_and_local_focus_is_found(self):
        ref = np.full((100, 120), 0.1, np.float32)
        with self.assertRaises(RuntimeError):
            find_focus(ref, ref)
        y, x = np.indices(ref.shape)
        image = ref+0.4*np.exp(-((x-71)**2+(y-43)**2)/12)
        center, _ = find_focus(image, ref)
        np.testing.assert_allclose(center, [71, 43], atol=0.2)

    def test_compact_model_can_fit_distinct_train_and_validation_pairs(self):
        model = CompactForwardModel(ASM(self.config), self.config['roi_shape'])
        phases = [torch.randn(self.config['slm_shape'])*0.2 for _ in range(4)]
        with torch.no_grad():
            targets = [model(p).detach()*0.8 for p in phases]
            initial = ((model(phases[-1])-targets[-1])**2).mean().item()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
        for _ in range(8):
            for phase, target in zip(phases[:3], targets[:3]):
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(phase)-target)**2).mean()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            final = ((model(phases[-1])-targets[-1])**2).mean().item()
        self.assertLess(final, initial*0.3)

    def test_lock_during_capture_discards_data_and_releases_acquisition(self):
        bench = Bench(dict(self.config, average_frames=2))
        bench.camera = Mock()
        bench.app = Mock()
        bench.verify_display = Mock(side_effect=[None, DisplayUnavailableError('locked during exposure')])
        frame = Mock()
        frame.IsIncomplete.return_value = False
        frame.GetNDArray.return_value = np.full((4, 6), 1000, np.uint16)
        frame.GetFrameID.return_value = 10
        frame.GetTimeStamp.return_value = 10000
        bench.camera.GetNextImage.return_value = frame
        with self.assertRaises(DisplayUnavailableError):
            bench.capture()
        bench.camera.EndAcquisition.assert_called_once()
        self.assertEqual(frame.Release.call_count, 3)
        self.assertFalse(bench.acquiring)
        self.assertEqual(bench.frame_metadata, [])

    def test_primary_display_error_survives_sdk_cleanup_failure(self):
        bench = Bench(self.config)
        bench.connect = Mock()
        bench.close = Mock(side_effect=RuntimeError('SDK release failed'))
        with self.assertWarnsRegex(UserWarning, 'Cleanup after DisplayUnavailableError'):
            with self.assertRaisesRegex(DisplayUnavailableError, 'original display failure'):
                with bench:
                    raise DisplayUnavailableError('original display failure')
        bench.connect.side_effect = DisplayUnavailableError('connect display failure')
        with self.assertWarnsRegex(UserWarning, 'Cleanup after connection failure'):
            with self.assertRaisesRegex(DisplayUnavailableError, 'connect display failure'):
                with bench:
                    pass

    def test_release_failure_still_closes_slm_window(self):
        bench = Bench(self.config)
        bench.system = Mock()
        bench.system.ReleaseInstance.side_effect = RuntimeError('SDK camera reference')
        label = bench.label = Mock()
        bench.app = Mock()
        with self.assertRaisesRegex(RuntimeError, 'SDK release'):
            bench.close()
        label.close.assert_called_once()
        self.assertIsNone(bench.label)
        self.assertIsNone(bench.system)

    def test_capture_failure_traceback_does_not_retain_sdk_camera_alias(self):
        bench = Bench(dict(self.config, average_frames=1))
        bench.camera = Mock()
        bench.camera.GetNextImage.side_effect = RuntimeError('capture timeout')
        bench.app = Mock()
        bench.verify_display = Mock()
        try:
            bench.capture()
        except RuntimeError as exc:
            traceback = exc.__traceback__
            while traceback and traceback.tb_frame.f_code.co_name != 'capture':
                traceback = traceback.tb_next
            self.assertIsNotNone(traceback)
            self.assertNotIn('c', traceback.tb_frame.f_locals)
            self.assertNotIn('frame', traceback.tb_frame.f_locals)
        else:
            self.fail('Expected camera failure')
        bench.camera.EndAcquisition.assert_called_once()

    def test_display_wait_is_bounded_without_connecting_camera(self):
        bench = Bench(self.config, display_wait_s=1)
        bench.app = Mock()
        bench.verify_display = Mock(side_effect=DisplayUnavailableError('locked'))
        with patch('benchtop.hardware.time.monotonic', side_effect=[0., 1.]):
            with self.assertRaises(DisplayUnavailableError):
                bench.wait_for_display()
        self.assertIsNone(bench.camera)

    def test_display_wait_rechecks_after_settling(self):
        bench = Bench(self.config, display_wait_s=1)
        bench.app = Mock()
        bench.verify_display = Mock()
        bench.settle = Mock()
        bench.wait_for_display()
        self.assertEqual(bench.verify_display.call_count, 2)
        bench.settle.assert_called_once()
        self.assertIsNone(bench.camera)

    def test_checkpoint_roundtrip_and_phase_gradient(self):
        original = CompactForwardModel(ASM(self.config), self.config['roi_shape'])
        with torch.no_grad():
            original.log_gain.fill_(0.12)
            original.correction[-1].bias.fill_(0.03)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'forward_model.pt'
            torch.save({'architecture': 'single_plane_asm_residual_cnn_8', 'config': self.config,
                        'correction': original.correction.state_dict(), 'log_gain': original.log_gain.detach(),
                        'illumination': original.asm.illumination}, path)
            restored = load_forward_model(path, self.config)
            phase = torch.randn(self.config['slm_shape']).requires_grad_(True)
            torch.testing.assert_close(restored(phase), original(phase))
            restored(phase).square().mean().backward()
            self.assertTrue(torch.isfinite(phase.grad).all())
            self.assertGreater(float(phase.grad.abs().max()), 0)
            self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
            with self.assertRaises(ValueError):
                load_forward_model(path, dict(self.config, distance_m=0.02))

    def test_rgb_phase_encoding_keeps_channel_identity(self):
        phase = np.zeros((4, 6, 3), np.float32)
        phase[..., 0], phase[..., 1], phase[..., 2] = -2., .2, 1.8
        encoded = encode_phase(phase)
        for i in range(3):
            np.testing.assert_array_equal(encoded[..., i], encode_phase(phase[..., i]))
        self.assertEqual(len(np.unique(encoded)), 3)

    def test_fixed_white_loss_and_trace_use_measured_phase_before_update(self):
        model = ASM(self.config)
        target = make_target(self.config['roi_shape'])
        phases, traces = [], []
        def camera(phase):
            phases.append(phase.copy())
            return 2*target.numpy()
        def trace(step, row, phase, amplitude, gradient):
            traces.append((phase.cpu().numpy().copy(), float(row['scale'])))
            np.testing.assert_allclose(amplitude.numpy(), 2*target.numpy(), atol=1e-6)
            if step < 2:
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
        _, history = optimize_phase(model, target, self.config, iterations=2,
                                     capture=camera, trace=trace, fixed_scale=1.)
        for measured_phase, (traced_phase, scale) in zip(phases, traces):
            np.testing.assert_array_equal(measured_phase, traced_phase)
            self.assertEqual(scale, 1.)
        self.assertAlmostEqual(history[0]['amplitude_mse'], float(target.square().mean()), places=6)

    def test_rgb_targets_are_reproducible_and_different_by_identity(self):
        from benchtop.rgb_run import target_rgb
        a, b = target_rgb((24, 32), 1000), target_rgb((24, 32), 2000)
        np.testing.assert_array_equal(a, target_rgb((24, 32), 1000))
        self.assertEqual(a.shape, (24, 32, 3))
        self.assertGreater(np.mean(np.abs(a-b)), .05)
        self.assertTrue(np.all((a >= .12) & (a <= .95)))

    def test_rgb_display_verification_rejects_swapped_channels(self):
        from PySide6.QtGui import QImage
        bench = Bench(self.config)
        h, w = self.config['slm_shape']
        arr = np.zeros((h, w, 3), np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = 30, 100, 220
        bench.last_encoded = arr
        bench.label = Mock()
        bench.label.screen.return_value.name.return_value = self.config['slm_screen']
        image = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888).copy()
        bench.label.screen.return_value.grabWindow.return_value.toImage.return_value = image
        bench.verify_display()
        swapped = np.ascontiguousarray(arr[..., ::-1])
        image = QImage(swapped.data, w, h, 3*w, QImage.Format_RGB888).copy()
        bench.label.screen.return_value.grabWindow.return_value.toImage.return_value = image
        with self.assertRaises(DisplayUnavailableError):
            bench.verify_display()

    def test_rectifier_subtracts_background_but_checks_raw_saturation(self):
        from benchtop.calibration import Rectifier
        rectifier = Rectifier.__new__(Rectifier)
        rectifier.config = dict(self.config, channel='green')
        rectifier.dark = np.full((4, 6), .02, np.float32)
        rectifier.intensity_scale = .2
        rectifier.roi = (slice(None), slice(None))
        rectifier.warp = lambda image: image
        rgb = np.full((4, 6, 3), .22, np.float32)
        np.testing.assert_allclose(rectifier.amplitude(rgb), 1., atol=1e-6)
        rgb[..., 1] = .99
        with self.assertRaises(RuntimeError):
            rectifier.amplitude(rgb)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
