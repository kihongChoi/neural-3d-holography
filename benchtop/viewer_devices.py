"""Device adapters for the external-SLM viewer; no display ownership or motion on connect."""
from __future__ import annotations

import ctypes as ct
import math
import time

import numpy as np

from .stage import Stage, GUARD_MM, validate_target


def validate_speed(value):
    value = float(value)
    if not math.isfinite(value) or not .05 <= value <= 2.:
        raise ValueError('이동 속도는 0.05–2.00 mm/s여야 합니다.')
    return value


def step_target(state, step, direction, backlash=.05):
    if not math.isfinite(step) or not .5 <= step <= 10 or direction not in (-1, 1):
        raise ValueError('이동 간격은 0.5–10 mm여야 합니다.')
    Stage.require_stationary(state)
    validate_target(state['position_mm'], (0, GUARD_MM), backlash)
    return validate_target(state['position_mm'] + direction * step, (0, GUARD_MM), backlash)


class HomingParameters(ct.Structure):
    # Installed Kinesis header: two short enums followed by two uint32 fields.
    _fields_ = [('direction', ct.c_short), ('limitSwitch', ct.c_short),
                ('velocity', ct.c_uint32), ('offsetDistance', ct.c_uint32)]


class ViewerStage(Stage):
    """Opt-in manual controls, separate from the experiment's restricted Stage API."""
    def __init__(self):
        super().__init__(interval=(0, GUARD_MM), monitor_only=True)
        self.motion = None
        self.speed_mm_s = .5

    def __enter__(self):
        super().__enter__()
        try:
            for name, result, args in [
                ('ISC_Home', ct.c_short, [ct.c_char_p]),
                ('ISC_EnableChannel', ct.c_short, [ct.c_char_p]),
                ('ISC_CanHome', ct.c_bool, [ct.c_char_p]),
                ('ISC_RequestVelParams', ct.c_short, [ct.c_char_p]),
                ('ISC_GetVelParams', ct.c_short, [ct.c_char_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]),
                ('ISC_RequestHomingParams', ct.c_short, [ct.c_char_p]),
                ('ISC_GetHomingParamsBlock', ct.c_short, [ct.c_char_p, ct.POINTER(HomingParameters)]),
                ('ISC_SetHomingParamsBlock', ct.c_short, [ct.c_char_p, ct.POINTER(HomingParameters)]),
            ]:
                f = getattr(self.dll, name)
                f.restype, f.argtypes = result, args
            return self
        except BaseException:
            self.close()
            raise

    def set_speed(self, value):
        value = validate_speed(value)
        if self.motion or self.status()['moving']:
            raise RuntimeError('이동 중에는 속도를 변경할 수 없습니다.')
        self.check(self.dll.ISC_SetVelParams(self.serial, self.units(1., 2), self.units(value, 1)))
        self.check(self.dll.ISC_RequestVelParams(self.serial))
        time.sleep(.15)
        acceleration, velocity = ct.c_int(), ct.c_int()
        self.check(self.dll.ISC_GetVelParams(self.serial, ct.byref(acceleration), ct.byref(velocity)))
        if abs(velocity.value-self.units(value, 1)) > 1 or abs(acceleration.value-self.units(1., 2)) > 1:
            raise RuntimeError('속도/가속도 readback 불일치')
        # Keep endpoint requests valid despite conversion to integer device units.
        self.speed_mm_s = value
        self.speed_readback_mm_s = self.real(velocity.value, 1)
        return self.speed_mm_s

    def start_step(self, step, direction):
        if self.motion:
            raise RuntimeError('스테이지 이동이 진행 중입니다.')
        before = self.status()
        target = step_target(before, step, direction, self.backlash_mm)
        self.arm()  # Recheck controller limits immediately before issuing motion.
        self.set_speed(self.speed_mm_s)
        self.check(self.dll.ISC_MoveToPosition(self.serial, self.units(target)))
        self.motion = dict(kind='move', target=target, deadline=time.monotonic()+step/self.speed_mm_s+10)
        return target

    def start_home(self):
        before = self.status()
        if self.motion or before['moving']:
            raise RuntimeError('이동 중에는 호밍할 수 없습니다.')
        if not self.dll.ISC_CanHome(self.serial):
            raise RuntimeError('이 컨트롤러는 호밍을 지원하지 않습니다.')
        self.arm()
        self.check(self.dll.ISC_RequestHomingParams(self.serial))
        time.sleep(.2)
        params = HomingParameters()
        self.check(self.dll.ISC_GetHomingParamsBlock(self.serial, ct.byref(params)))
        # Never guess/reverse the physical direction or reset coordinates.
        if params.direction != 2 or params.limitSwitch != 1:
            raise RuntimeError('역방향 홈/역방향 리밋 설정이 아닙니다. Kinesis 설정을 확인하세요.')
        if not 0 <= self.real(params.offsetDistance) <= 5:
            raise RuntimeError('예상 밖 홈 오프셋(0–5 mm 범위 초과)입니다.')
        params.velocity = min(params.velocity, self.units(.5, 1))
        if not params.velocity:
            raise RuntimeError('호밍 속도가 0입니다.')
        self.check(self.dll.ISC_SetHomingParamsBlock(self.serial, ct.byref(params)))
        actual = HomingParameters()
        self.check(self.dll.ISC_GetHomingParamsBlock(self.serial, ct.byref(actual)))
        if bytes(actual) != bytes(params):
            raise RuntimeError('호밍 설정 readback 불일치')
        self.check(self.dll.ISC_EnableChannel(self.serial))
        self.check(self.dll.ISC_Home(self.serial))
        self.motion = dict(kind='home', seen_active=False, deadline=time.monotonic()+660)

    def poll(self):
        state = self.status()
        if self.motion:
            try:
                if not state['enabled'] or time.monotonic() > self.motion['deadline']:
                    raise RuntimeError('스테이지 비활성화 또는 이동 시간 초과')
                if self.motion['kind'] == 'home':
                    self.motion['seen_active'] |= state['moving'] or not state['homed']
                    done = self.motion['seen_active'] and state['homed'] and not state['moving']
                    if done:
                        validate_target(state['position_mm'], (0, GUARD_MM), self.backlash_mm)
                else:
                    if not state['homed'] or not 0 <= state['position_mm'] < GUARD_MM:
                        raise RuntimeError(f'스테이지 위치/호밍 상태 오류: {state}')
                    done = not state['moving'] and abs(state['position_mm']-self.motion['target']) < .002
                if done:
                    self.motion = None
            except BaseException:
                self.stop()
                raise
        state['operation'] = self.motion['kind'] if self.motion else None
        state['speed_mm_s'] = self.speed_mm_s
        state['speed_readback_mm_s'] = getattr(self, 'speed_readback_mm_s', .5)
        return state

    def stop(self):
        if self.opened:
            self.check(self.dll.ISC_StopImmediate(self.serial))
        self.motion = None

    def close(self):
        try:
            if self.opened and (self.motion or self.status()['moving']):
                self.stop()
        finally:
            super().close()
            if getattr(self, 'dll_dir', None):
                self.dll_dir.close()
                self.dll_dir = None


class StreamCamera:
    """Full sensor, continuous Line0-triggered capture with a bounded SDK buffer."""
    def __init__(self, exposure=15000):
        self.exposure = exposure
        self.system = self.cameras = self.camera = self.processor = None
        self.initialized = self.acquiring = False
        self.original = {}
        self.stream_mode = None

    def connect(self):
        import PySpin as spin
        self.spin = spin
        try:
            self.system = spin.System.GetInstance()
            self.cameras = self.system.GetCameras()
            for i in range(self.cameras.GetSize()):
                candidate = self.cameras.GetByIndex(i)
                if candidate.TLDevice.DeviceSerialNumber.GetValue() == '25222112':
                    self.camera = candidate
                    del candidate
                    break
                del candidate
            if self.camera is None:
                raise RuntimeError('FLIR 카메라 25222112를 찾을 수 없습니다.')
            c = self.camera
            c.Init()
            self.initialized = True
            names = ['AcquisitionMode', 'ExposureMode', 'ExposureAuto', 'ExposureTime',
                     'GainAuto', 'Gain', 'GammaEnable', 'PixelFormat', 'BalanceWhiteAuto',
                     'BlackLevel', 'TriggerSelector', 'TriggerSource', 'TriggerActivation',
                     'TriggerOverlap', 'TriggerDelay', 'TriggerMode', 'Width', 'Height',
                     'OffsetX', 'OffsetY', 'AcquisitionFrameRateEnable']
            nm = c.GetNodeMap()
            for name in names:
                node = nm.GetNode(name)
                if spin.IsReadable(node):
                    self.original[name] = spin.CValuePtr(node).ToString()
            self.stream_mode = c.TLStream.StreamBufferHandlingMode.GetValue()
            c.TriggerMode.SetValue(spin.TriggerMode_Off)
            c.OffsetX.SetValue(0)
            c.OffsetY.SetValue(0)
            c.Width.SetValue(c.Width.GetMax())
            c.Height.SetValue(c.Height.GetMax())
            if (c.Width.GetValue(), c.Height.GetValue()) != (4096, 3000):
                raise RuntimeError('4096 × 3000 전체 센서 설정을 확인할 수 없습니다.')
            c.AcquisitionMode.SetValue(spin.AcquisitionMode_Continuous)
            c.ExposureMode.SetValue(spin.ExposureMode_Timed)
            c.ExposureAuto.SetValue(spin.ExposureAuto_Off)
            c.GainAuto.SetValue(spin.GainAuto_Off)
            c.Gain.SetValue(0.)
            c.GammaEnable.SetValue(False)
            c.BlackLevel.SetValue(0.)
            c.BalanceWhiteAuto.SetValue(spin.BalanceWhiteAuto_Off)
            c.PixelFormat.SetValue(spin.PixelFormat_BayerRG8)
            if spin.IsWritable(c.AcquisitionFrameRateEnable):
                c.AcquisitionFrameRateEnable.SetValue(False)
            c.TLStream.StreamBufferHandlingMode.SetValue(spin.StreamBufferHandlingMode_NewestOnly)
            c.TriggerSelector.SetValue(spin.TriggerSelector_FrameStart)
            c.TriggerSource.SetValue(spin.TriggerSource_Line0)
            c.TriggerActivation.SetValue(spin.TriggerActivation_RisingEdge)
            c.TriggerOverlap.SetValue(spin.TriggerOverlap_Off)
            c.TriggerDelay.SetValue(22.)
            self.set_exposure(self.exposure)
            c.TriggerMode.SetValue(spin.TriggerMode_On)
            self.processor = spin.ImageProcessor()
            self.processor.SetColorProcessing(spin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
            c.BeginAcquisition()
            self.acquiring = True
            return self.settings()
        except BaseException:
            # Avoid SDK objects held by traceback frames during ReleaseInstance.
            c = nm = node = None
            self.close()
            raise

    def settings(self):
        c = self.camera
        return dict(exposure_us=c.ExposureTime.GetValue(),
                    exposure_min=c.ExposureTime.GetMin(),
                    exposure_max=min(100000., c.ExposureTime.GetMax()),
                    trigger='Line0 · RisingEdge', delay_us=c.TriggerDelay.GetValue(),
                    width=c.Width.GetValue(), height=c.Height.GetValue())

    def set_exposure(self, value):
        c = self.camera
        if not math.isfinite(value) or not c.ExposureTime.GetMin() <= value <= min(100000, c.ExposureTime.GetMax()):
            raise ValueError('노출값이 허용 범위를 벗어났습니다.')
        was_acquiring = self.acquiring
        if was_acquiring:
            c.EndAcquisition()
            self.acquiring = False
        try:
            c.ExposureTime.SetValue(float(value))
            self.exposure = float(c.ExposureTime.GetValue())
        finally:
            if was_acquiring:
                c.BeginAcquisition()
                self.acquiring = True
        return self.settings()

    def grab(self):
        try:
            frame = self.camera.GetNextImage(250)
        except self.spin.SpinnakerException as exc:
            if exc.errorcode == self.spin.SPINNAKER_ERR_TIMEOUT:
                return None  # Missing V-sync is reported as waiting, never free-run.
            raise
        converted = None
        try:
            if frame.IsIncomplete():
                raise RuntimeError(f'불완전 카메라 프레임: {frame.GetImageStatus()}')
            converted = self.processor.Convert(frame, self.spin.PixelFormat_RGB8)
            rgb = converted.GetNDArray().copy()
            return rgb, dict(frame_id=int(frame.GetFrameID()), timestamp_ns=int(frame.GetTimeStamp()),
                             received=time.monotonic(), exposure_us=self.exposure)
        finally:
            converted = None
            frame.Release()

    def close(self):
        errors = []
        c = self.camera
        if c is not None and self.initialized:
            try:
                if self.acquiring:
                    c.EndAcquisition()
                    self.acquiring = False
                c.TriggerMode.SetValue(self.spin.TriggerMode_Off)
            except Exception as exc:
                errors.append(str(exc))
            # Restore dependent nodes in a valid order, and try all on failure.
            order = ['OffsetX', 'OffsetY', 'Width', 'Height']
            for name in order[:2]:
                try:
                    getattr(c, name).SetValue(0)
                except Exception as exc:
                    errors.append(str(exc))
            last = ['ExposureAuto', 'GainAuto', 'BalanceWhiteAuto', 'TriggerMode']
            order = ['Width', 'Height', 'OffsetX', 'OffsetY'] + [
                n for n in self.original if n not in order and n not in last] + last
            for name in order:
                try:
                    if name in self.original:
                        node = c.GetNodeMap().GetNode(name)
                        if not self.spin.IsWritable(node):
                            raise RuntimeError(f'{name}: restore not writable')
                        self.spin.CValuePtr(node).FromString(self.original[name])
                except Exception as exc:
                    errors.append(str(exc))
            node = None
            try:
                if self.stream_mode is not None:
                    c.TLStream.StreamBufferHandlingMode.SetValue(self.stream_mode)
                c.DeInit()
            except Exception as exc:
                errors.append(str(exc))
        self.initialized = False
        self.camera = c = self.processor = None
        if self.cameras is not None:
            self.cameras.Clear()
            self.cameras = None
        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None
        if errors:
            raise RuntimeError('카메라 종료/복원: ' + '; '.join(errors))
