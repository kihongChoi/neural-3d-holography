import ctypes as ct
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from benchtop.viewer_devices import HomingParameters, ViewerStage, step_target, validate_speed
from benchtop.viewer_slm import (compose_frame, direct_codes, target_amplitude, load_lut,
                                 encode_complex, read_complex, complex_dpac, prepare)
from benchtop.viewer_service import (CameraWorker, DemoCamera, DemoLaser, DemoStage,
                                     LaserWorker, Shared, SnapshotWorker, StageWorker)


def until(condition, timeout=4):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(.02)
    raise AssertionError('Condition did not complete before timeout')


class ViewerSafety(unittest.TestCase):
    def test_complex_dpac_recovers_amplitude_phase_and_rgb_ratios(self):
        # Constant adjacent samples: the mean of the two unit phasors must
        # equal the normalized input, including its nonzero complex phase.
        u = np.ones((3, 2, 4), np.complex128) * np.exp(.7j)
        u *= np.array([1., .5, .1])[:, None, None]
        theta, info = encode_complex(u)
        reconstructed = (np.exp(1j*theta[..., ::2]) + np.exp(1j*theta[..., 1::2]))/2
        np.testing.assert_allclose(reconstructed, u[..., ::2], atol=3e-7)
        self.assertEqual(theta.dtype, np.float32)
        self.assertEqual(info['clipped'], [0., 0., 0.])
        self.assertAlmostEqual(info['amp_max'], 1.)
        _, info = encode_complex(u, gain=2.)
        self.assertEqual(info['clipped'], [1., 0., 0.])
        # All-zero fields are valid and encode cancelling checkerboard phases.
        zero, info = encode_complex(np.zeros((3, 2, 4), np.complex64))
        self.assertTrue(np.isfinite(zero).all())
        np.testing.assert_allclose(np.exp(1j*zero).mean(-1), 0, atol=1e-7)
        self.assertEqual(info['amp_max'], 0.)
        for gain in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                encode_complex(u, gain=gain)
        for amax in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                encode_complex(u, amp_max=amax)

    def test_complex_npy_layout_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'field.npy'
            u = np.arange(24).reshape(3, 2, 4).astype(np.complex64)*(1+2j)
            for value in (u, np.moveaxis(u, 0, -1)):
                np.save(path, value)
                loaded, meta = read_complex(path)
                np.testing.assert_array_equal(loaded, u)
                self.assertFalse(meta['single_channel_repeated'])
            for value in (u[0], u[:1], np.moveaxis(u[:1], 0, -1)):
                np.save(path, value)
                loaded, meta = read_complex(path)
                np.testing.assert_array_equal(loaded, np.repeat(u[:1], 3, 0))
                self.assertTrue(meta['single_channel_repeated'])
            for value in (u.real, np.array([object()], dtype=object),
                          np.zeros((3, 4, 3), np.complex64),
                          np.zeros((2, 4, 4), np.complex64),
                          np.zeros((0, 4), np.complex64), u*np.nan):
                np.save(path, value)
                with self.assertRaises(ValueError):
                    read_complex(path)

    def test_complex_prepare_saves_native_frame_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp)/'field.npy', Path(tmp)/'patterns/0001'
            np.save(path, np.full((2, 4), .5+.25j, np.complex64))
            codes, meta = prepare(path, 'complex', 3., output)
            self.assertFalse(codes.flags.writeable)
            self.assertEqual(meta['mode'], 'complex_dpac')
            self.assertFalse(meta['propagation'] or meta['lowpass'] or meta['lut_applied'])
            self.assertTrue(meta['single_channel_repeated'])
            self.assertEqual(meta['output_directory'], str(output.resolve()))
            with Image.open(output/'dpac_slm.png') as image:
                self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.size, (1920, 1080))
                np.testing.assert_array_equal(np.asarray(image), compose_frame(codes))
            self.assertEqual(json.loads((output/'pattern.json').read_text(encoding='utf-8')), meta)
            # Padding represents zero complex amplitude, not uniform phase.
            self.assertFalse(np.array_equal(codes[0, 0], codes[0, 1]))
            with self.assertRaises(FileExistsError):
                prepare(path, 'complex', 3., output)

    def test_complex_full_frame_crop_preserves_source_scale(self):
        from benchtop.hardware import encode_phase
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'field.npy'
            u = np.full((3, 1080, 1920), .25j, np.complex64)
            u[:, 0, 0] = 2  # Outside the ROI, still the normalization reference.
            np.save(path, u)
            codes, meta = complex_dpac(path)
            theta, _ = encode_complex(u[:, 60:1020, 96:1824], amp_max=2.)
            np.testing.assert_array_equal(codes, np.moveaxis(encode_phase(theta), 0, -1))
            self.assertEqual(meta['amp_max'], 2.)
            np.save(path, np.ones((1000, 1800), np.complex64))
            with self.assertRaisesRegex(ValueError, '자동 축소'):
                complex_dpac(path)

    def test_speed_limits_and_motion_interlock(self):
        for value in [.049, 2.01, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                validate_speed(value)
        stage = ViewerStage()
        stage.status = Mock(return_value=dict(moving=True))
        stage.dll = Mock()
        with self.assertRaises(RuntimeError):
            stage.set_speed(1.)
        stage.dll.ISC_SetVelParams.assert_not_called()

    def test_speed_readback_and_move_deadline(self):
        stage = ViewerStage()
        stage.status = Mock(return_value=dict(position_mm=100., homed=True, enabled=True, moving=False))
        stage.arm = Mock()
        stage.backlash_mm = .05
        stage.units = lambda v, kind=0: int(v*1000)
        stage.real = lambda v, kind=0: v/1000
        stage.dll = Mock()
        stage.dll.ISC_SetVelParams.return_value = 0
        stage.dll.ISC_RequestVelParams.return_value = 0
        stage.dll.ISC_MoveToPosition.return_value = 0
        def readback(serial, acceleration, velocity):
            acceleration._obj.value = 1000
            velocity._obj.value = 50
            return 0
        stage.dll.ISC_GetVelParams.side_effect = readback
        self.assertAlmostEqual(stage.set_speed(.05), .05)
        before = time.monotonic()
        stage.start_step(10, -1)
        self.assertGreaterEqual(stage.motion['deadline']-before, 210)
        stage.dll.ISC_MoveToPosition.assert_called_once_with(stage.serial, 90000)
        stage.motion = None
        with self.assertRaises(RuntimeError):
            stage.set_speed(2.)  # Readback still says .05: reject mismatch.

    def test_slm_80_percent_and_toggle_does_not_cascade(self):
        table = load_lut()
        original = np.tile(np.arange(1728, dtype=np.uint16) % 256, (960, 1)).astype(np.uint8)
        original = np.stack([original]*3, -1)
        original.setflags(write=False)
        off = compose_frame(original)
        on = compose_frame(original, table)
        self.assertAlmostEqual(1728*960/(1920*1080), .8)
        np.testing.assert_array_equal(off[60:1020, 96:1824], original)
        np.testing.assert_array_equal(on[60:1020, 96:1824, 0], table[0, original[..., 0]])
        self.assertTrue(np.all(on[:60] == 128))
        self.assertTrue(np.all(on[:, :96] == 128))
        np.testing.assert_array_equal(compose_frame(original), off)
        np.testing.assert_array_equal(compose_frame(original, table), on)

    def test_photo_contain_and_native_codes(self):
        rgb = np.full((800, 400, 3), 255, np.uint8)
        amplitude = target_amplitude(rgb)
        self.assertEqual(amplitude.shape, (960, 1728, 3))
        self.assertTrue(np.all(amplitude[:, 624:1104] == 1))
        self.assertTrue(np.all(amplitude[:, :624] == 0))
        code, _ = direct_codes(rgb)
        np.testing.assert_array_equal(code[80:880, 664:1064], rgb)
        with self.assertRaises(ValueError):
            direct_codes(np.zeros((1000, 1800, 3), np.uint8))

    def test_steps_require_homed_stationary_stage_and_backlash_clearance(self):
        state = dict(position_mm=100, homed=True, enabled=True, moving=False)
        self.assertEqual(step_target(state, .5, -1), 99.5)
        self.assertEqual(step_target(state, 10, 1), 110)
        for step in (.49, 10.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                step_target(state, step, 1)
        for change in (dict(homed=False), dict(enabled=False), dict(moving=True)):
            with self.assertRaises(RuntimeError):
                step_target(dict(state, **change), .5, 1)
        for position, direction in [(123.5, 1), (.1, -1)]:
            with self.assertRaises(ValueError):
                step_target(dict(state, position_mm=position), .5, direction)

    def test_home_refuses_unverified_direction_and_caps_speed(self):
        self.assertEqual(ct.sizeof(HomingParameters), 12)
        for direction, switch, allowed in [(1, 1, False), (2, 4, False), (2, 1, True)]:
            stage = ViewerStage()
            stage.status = Mock(return_value=dict(moving=False))
            stage.arm = Mock()
            stage.units = lambda v, kind=0: int(v*1000)
            stage.real = lambda v, kind=0: v/1000
            params = HomingParameters(direction, switch, 5000, 1000)
            def get_params(serial, pointer):
                ct.memmove(pointer, ct.byref(params), ct.sizeof(params))
                return 0
            def set_params(serial, pointer):
                ct.memmove(ct.byref(params), pointer, ct.sizeof(params))
                return 0
            stage.dll = Mock()
            for name in ('ISC_RequestHomingParams', 'ISC_EnableChannel', 'ISC_Home'):
                getattr(stage.dll, name).return_value = 0
            stage.dll.ISC_CanHome.return_value = True
            stage.dll.ISC_GetHomingParamsBlock.side_effect = get_params
            stage.dll.ISC_SetHomingParamsBlock.side_effect = set_params
            if allowed:
                stage.start_home()
                self.assertEqual(params.velocity, 500)
                stage.dll.ISC_Home.assert_called_once()
            else:
                with self.assertRaises(RuntimeError):
                    stage.start_home()
                stage.dll.ISC_Home.assert_not_called()

    def test_motion_failure_stops_stage(self):
        stage = ViewerStage()
        stage.motion = dict(kind='move', deadline=time.monotonic()+5, target=101)
        stage.status = Mock(return_value=dict(position_mm=125, enabled=True, moving=True, homed=True))
        stage.stop = Mock()
        with self.assertRaises(RuntimeError):
            stage.poll()
        stage.stop.assert_called_once()

    def test_stream_continues_during_motion_and_exposure_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Shared(tmp)
            camera = CameraWorker(shared, DemoCamera)
            stage = StageWorker(shared, DemoStage)
            camera.start()
            stage.start()
            try:
                until(lambda: shared.sequence > 1 and shared.snapshot()['stage']['connected'])
                start = shared.sequence
                stage.submit('step', 10, -1)
                camera.submit('exposure', 7000)
                until(lambda: shared.snapshot()['stage'].get('moving') and
                      shared.snapshot()['camera'].get('exposure_us') == 7000 and shared.sequence > start+2)
                stage.stop()
                until(lambda: not shared.snapshot()['stage'].get('moving'))
                self.assertGreater(shared.snapshot()['stage']['position_mm'], 90)
            finally:
                camera.quit.set()
                stage.quit.set()
                camera.join(4)
                stage.join(4)
            self.assertFalse(camera.is_alive() or stage.is_alive())

    def test_laser_standby_on_stale_camera_and_worker_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Shared(tmp)
            shared.update('camera', connected=True, last_frame=time.monotonic())
            worker = LaserWorker(shared, DemoLaser)
            worker.start()
            try:
                until(lambda: shared.snapshot()['laser']['connected'])
                worker.submit('on')
                until(lambda: shared.snapshot()['laser'].get('on'))
                shared.update('camera', last_frame=time.monotonic()-3)
                until(lambda: shared.snapshot()['laser'].get('on') is False)
            finally:
                worker.quit.set()
                worker.join(4)
            self.assertEqual(shared.snapshot()['laser']['readback']['mode'], 'Digital')

    def test_latest_only_and_full_frame_png_saved_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Shared(tmp)
            rgb = np.zeros((60, 80, 3), np.uint8)
            rgb[0, 0] = [200, 12, 43]
            for number in range(100):
                shared.frame(rgb, dict(frame_id=number, received=time.monotonic()))
            self.assertEqual(shared.get_frame()[0], 100)
            self.assertEqual(shared.get_frame()[2]['frame_id'], 99)
            writer = SnapshotWorker(shared)
            writer.start()
            writer.submit('save', rgb, shared.get_frame()[2])
            writer.quit.set()
            writer.join(4)
            files = list(Path(tmp).rglob('*.png'))
            self.assertEqual(len(files), 1)
            with Image.open(files[0]) as image:
                self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.size, (80, 60))
                np.testing.assert_array_equal(np.asarray(image), rgb)


if __name__ == '__main__':
    unittest.main()
