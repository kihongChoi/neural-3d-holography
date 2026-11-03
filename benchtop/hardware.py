"""Actual LETO HDMI output and linear Bayer captures. No simulated fallback."""
from __future__ import annotations

import time
import warnings
import numpy as np


class DisplayUnavailableError(RuntimeError):
    """The physical display is covered, locked, or showing another image."""


def encode_phase(phase, inverted=True):
    """Same [-pi, pi] / inverted LUT convention as BenchtopDisplay."""
    normalized = (np.asarray(phase) + np.pi) % (2 * np.pi) / (2 * np.pi)
    if inverted:
        normalized = 1 - normalized
    return np.round(normalized * 255).clip(0, 255).astype(np.uint8)


class Bench:
    def __init__(self, config, display_wait_s=0):
        self.config = config
        self.display_wait_s = max(0., float(display_wait_s))
        self.app = self.label = self.system = self.cameras = self.camera = None
        self.last_encoded = None
        self.camera_initialized = False
        self.acquiring = False
        self.original = {}
        self.frame_metadata = []
        self.active_timing = None

    def __enter__(self):
        try:
            self.connect()
            return self
        except BaseException:
            try:
                self.close()
            except Exception as cleanup_error:
                warnings.warn(f'Cleanup after connection failure: {cleanup_error}')
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            warnings.warn(f'Cleanup after {exc_type.__name__}: {cleanup_error}')
        return False

    def connect_display(self):
        from PySide6.QtWidgets import QApplication, QLabel
        from PySide6.QtCore import Qt
        self.app = QApplication.instance() or QApplication([])
        screens = [s for s in self.app.screens() if s.name() == self.config['slm_screen']]
        if len(screens) != 1:
            raise RuntimeError(f"Expected SLM {self.config['slm_screen']}; found {[s.name() for s in self.app.screens()]}")
        screen = screens[0]
        h, w = self.config['slm_shape']
        if screen.size().toTuple() != (w, h) or screen.devicePixelRatio() != 1:
            raise RuntimeError('SLM requires native 1920x1080 and Windows display scaling 100%.')
        self.label = QLabel()
        self.label.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.label.setCursor(Qt.BlankCursor)
        self.label.setScaledContents(False)
        self.label.setStyleSheet('background-color: black;')
        self.label.setGeometry(screen.geometry())
        self.label.winId()
        self.label.windowHandle().setScreen(screen)
        self.label.showFullScreen()
        self.label.setGeometry(screen.geometry())
        self.label.raise_()
        self.label.activateWindow()
        try:
            self.show(np.zeros((h, w), np.float32))
        except DisplayUnavailableError:
            if not self.display_wait_s:
                raise
            self.wait_for_display()

    def connect(self):
        import PySpin
        self.spin = PySpin
        self.connect_display()
        self.system = PySpin.System.GetInstance()
        self.cameras = self.system.GetCameras()
        for index in range(self.cameras.GetSize()):
            candidate = self.cameras.GetByIndex(index)
            if candidate.TLDevice.DeviceSerialNumber.GetValue() == self.config['camera_serial']:
                self.camera = candidate
                break
            del candidate
        if self.camera is None:
            raise RuntimeError(f"Camera {self.config['camera_serial']} not detected.")
        self.camera.Init()
        self.camera_initialized = True
        names = ['PixelFormat', 'ExposureAuto', 'ExposureTime', 'GainAuto', 'Gain',
                 'GammaEnable', 'AcquisitionMode', 'TriggerMode', 'TriggerSelector',
                 'TriggerSource', 'TriggerActivation', 'TriggerOverlap', 'TriggerDelay',
                 'BalanceWhiteAuto', 'BlackLevel', 'AdcBitDepth']
        nm = self.camera.GetNodeMap()
        for name in names:
            node = nm.GetNode(name)
            if PySpin.IsReadable(node):
                self.original[name] = PySpin.CValuePtr(node).ToString()
        self.original_stream = self.camera.TLStream.StreamBufferHandlingMode.GetValue()
        c = self.camera
        c.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        c.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        c.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        c.GainAuto.SetValue(PySpin.GainAuto_Off)
        c.GammaEnable.SetValue(False)
        c.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG16)
        c.ExposureTime.SetValue(float(self.config['exposure_us']))
        c.Gain.SetValue(float(self.config['gain_db']))
        c.BlackLevel.SetValue(0.0)
        c.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
        c.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)
        if self.config['trigger'] != 'off':
            c.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            c.TriggerSource.SetValue(getattr(PySpin, 'TriggerSource_' + self.config['trigger']))
            c.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            c.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
            c.TriggerDelay.SetValue(float(self.config['trigger_delay_us']))
            c.TriggerMode.SetValue(PySpin.TriggerMode_On)
        print(f"Connected {self.label.screen().name()}, camera {self.config['camera_serial']}, linear BayerRG16, exposure {c.ExposureTime.GetValue():.1f} us", flush=True)

    def settle(self):
        until = time.monotonic() + self.config['settle_s']
        while time.monotonic() < until:
            self.app.processEvents()
            time.sleep(0.005)

    def show(self, phase):
        self.show_codes(encode_phase(phase, self.config['invert_phase']))

    def show_codes(self, arr):
        """Present pre-encoded HDMI pixels with the same physical display checks."""
        self.update_display(arr)
        self.settle()
        self.verify_display()

    def update_display(self, arr):
        """Refresh an optimization preview without the acquisition settling delay."""
        from PySide6.QtGui import QImage, QPixmap
        if arr.dtype != np.uint8:
            raise ValueError('Expected uint8 HDMI pixels')
        shape = tuple(self.config['slm_shape'])
        if arr.shape not in (shape, shape+(3,)):
            raise ValueError(f'Phase must use physical SLM resolution, got {arr.shape}')
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        channels = 3 if arr.ndim == 3 else 1
        fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_Grayscale8
        image = QImage(arr.data, w, h, channels*w, fmt).copy()
        self.label.setPixmap(QPixmap.fromImage(image))
        # Reassert the dedicated output window after unrelated desktop activity.
        # Windows system overlays can still win: verify_display remains mandatory.
        self.label.raise_()
        self.last_encoded = arr
        self.app.processEvents()

    def verify_display(self):
        """Inspect the actual monitor; a widget's offscreen render is insufficient."""
        if self.config.get('verify_display', True):
            from PySide6.QtGui import QImage
            if self.last_encoded is None:
                raise DisplayUnavailableError('No phase has been presented on the SLM.')
            if self.label.screen().name() != self.config['slm_screen']:
                raise DisplayUnavailableError('SLM window moved to another monitor; acquisition stopped.')
            arr = self.last_encoded
            h, w = arr.shape[:2]
            channels = 3 if arr.ndim == 3 else 1
            fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_Grayscale8
            actual = self.label.screen().grabWindow(0).toImage().convertToFormat(fmt)
            if actual.width() != w or actual.height() != h:
                raise DisplayUnavailableError('SLM desktop capture has the wrong dimensions.')
            pixels = np.frombuffer(actual.constBits(), dtype=np.uint8).reshape(h, actual.bytesPerLine())[:, :w*channels]
            pixels = pixels.reshape(arr.shape)
            error = float(np.abs(pixels.astype(np.float32)-arr).mean())
            if error > 1.0:
                raise DisplayUnavailableError(f'SLM desktop differs from requested pattern (mean error {error:.1f}/255). '
                                              'Unlock Windows and keep the SLM display unobstructed. Acquisition was stopped.')

    def set_timing(self, exposure_us, delay_us):
        """Change CFS acquisition slot only between acquisitions; retain original settings."""
        if self.acquiring:
            raise RuntimeError('Cannot change timing during acquisition')
        c = self.camera
        mode = c.TriggerMode.GetValue()
        c.TriggerMode.SetValue(self.spin.TriggerMode_Off)
        try:
            c.ExposureTime.SetValue(float(exposure_us))
            c.TriggerDelay.SetValue(float(delay_us))
        finally:
            c.TriggerMode.SetValue(mode)
        self.active_timing = {'exposure_us': float(c.ExposureTime.GetValue()),
                              'trigger_delay_us': float(c.TriggerDelay.GetValue())}
        return self.active_timing.copy()

    def wait_for_display(self):
        """Bounded wait only: do not attempt to unlock or alter Windows security."""
        deadline = time.monotonic()+self.display_wait_s
        print(f'Waiting up to {self.display_wait_s:.0f}s for the SLM display to become available; camera is not connected.', flush=True)
        while True:
            self.app.processEvents()
            try:
                self.verify_display()
                self.settle()
                self.verify_display()
                print('SLM framebuffer verified; continuing hardware setup.', flush=True)
                return
            except DisplayUnavailableError:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.2, remaining))

    def capture(self, phase=None, frames=None):
        """Start acquisition only AFTER display settles, so old frames cannot leak in.

        Average Bayer samples per 2x2 cell, avoiding white balance, gamma and
        demosaicing. Returned sensor coordinates are half the native resolution.
        BayerRG16 uses MSB-aligned ADC samples, divided by 65535.
        """
        if phase is not None:
            self.show(phase)
        self.verify_display()
        frames = frames or self.config['average_frames']
        c = self.camera
        c.BeginAcquisition()
        self.acquiring = True
        accumulated = None
        metadata = []
        try:
            for _ in range(frames + 1):
                frame = c.GetNextImage(3000)
                try:
                    if frame.IsIncomplete():
                        raise RuntimeError(f'Incomplete camera frame: {frame.GetImageStatus()}')
                    raw = frame.GetNDArray().astype(np.float32) / 65535.0
                    rgb = np.stack((raw[0::2, 0::2], (raw[0::2, 1::2] + raw[1::2, 0::2]) / 2,
                                    raw[1::2, 1::2]), axis=-1)
                    info = {'frame_id': int(frame.GetFrameID()), 'timestamp_ns': int(frame.GetTimeStamp()),
                            'saturated_fraction': float((raw >= 0.98).mean()),
                            'max_raw_normalized': float(raw.max())}
                    if self.active_timing is not None:
                        info.update(self.active_timing)
                    if metadata:
                        accumulated = rgb if accumulated is None else accumulated + rgb
                    metadata.append(info)
                finally:
                    frame.Release()
                    del frame
                self.app.processEvents()
        finally:
            try:
                c.EndAcquisition()
                self.acquiring = False
            finally:
                # An exception traceback retains local variables. Keeping this
                # SDK camera reference alive prevents ReleaseInstance on exit.
                del c
        # If Windows locks during acquisition, discard the entire measurement.
        self.verify_display()
        self.frame_metadata.extend(metadata[1:])
        return accumulated / frames

    def close(self):
        # Restore runtime settings without persisting a camera user set or
        # touching laser enable, laser power, or the SLM controller firmware.
        errors = []
        nm = node = None
        if self.camera is not None and self.camera_initialized:
            try:
                if self.acquiring:
                    self.camera.EndAcquisition()
                    self.acquiring = False
                self.camera.TriggerMode.SetValue(self.spin.TriggerMode_Off)
                nm = self.camera.GetNodeMap()
                for name, value in self.original.items():
                    if name == 'TriggerMode':
                        continue
                    node = nm.GetNode(name)
                    if self.spin.IsWritable(node):
                        self.spin.CValuePtr(node).FromString(value)
                if self.original:
                    self.camera.TLStream.StreamBufferHandlingMode.SetValue(self.original_stream)
                    self.spin.CValuePtr(nm.GetNode('TriggerMode')).FromString(self.original['TriggerMode'])
            except Exception as exc:
                errors.append(('camera restore', exc))
            finally:
                nm = node = None
                try:
                    self.camera.DeInit()
                except Exception as exc:
                    errors.append(('camera deinitialization', exc))
                self.camera_initialized = False
        self.camera = None
        if self.cameras is not None:
            try:
                self.cameras.Clear()
            except Exception as exc:
                errors.append(('camera list release', exc))
            finally:
                self.cameras = None
        if self.system is not None:
            try:
                self.system.ReleaseInstance()
            except Exception as exc:
                errors.append(('SDK release', exc))
            finally:
                self.system = None
        if self.label is not None:
            try:
                self.label.close()
                self.app.processEvents()
            except Exception as exc:
                errors.append(('SLM window close', exc))
            finally:
                self.label = None
        if errors:
            raise RuntimeError('; '.join(f'{stage}: {error}' for stage, error in errors)) from errors[0][1]
