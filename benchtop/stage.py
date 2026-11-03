"""Restricted LTS300/M control. No homing, jog, relative move, or zero reset API.

125 mm is a physical collision boundary on this bench, NOT the 300 mm travel.
Only absolute moves in the caller's smaller measurement interval are exposed.
"""
from __future__ import annotations

import ctypes as ct
import math
import os
import time
from pathlib import Path


COLLISION_MM = 125.0
GUARD_MM = 124.0


def validate_target(target, interval=(99.0, 103.0), backlash_mm=0.05):
    target = float(target)
    lo, hi = interval
    if not all(math.isfinite(v) for v in (target, lo, hi, backlash_mm)):
        raise ValueError('Nonfinite motion request')
    if not 0 <= lo <= hi <= GUARD_MM or backlash_mm < 0:
        raise ValueError('Unsafe measurement interval')
    if not lo <= target <= hi or target + backlash_mm > GUARD_MM:
        raise ValueError(f'Unsafe target {target} mm; interval {interval}, guard {GUARD_MM}')
    return target


class Stage:
    def __init__(self, serial='45897069', interval=(99.0, 103.0), *, monitor_only=False):
        validate_target(sum(interval)/2, interval)
        self.serial = serial.encode('ascii')
        self.interval = interval
        self.monitor_only = monitor_only
        self.opened = self.polling = False
        self.moves = []

    @staticmethod
    def check(rc):
        if rc != 0:
            raise RuntimeError(f'Kinesis error {rc}')

    def __enter__(self):
        path = Path(r'C:\Program Files\Thorlabs\Kinesis')
        self.dll_dir = os.add_dll_directory(str(path))
        self.dll = ct.CDLL(str(path/'Thorlabs.MotionControl.IntegratedStepperMotors.dll'))
        signatures = {
            'TLI_BuildDeviceList': (ct.c_short, []),
            'ISC_Open': (ct.c_short, [ct.c_char_p]),
            'ISC_Close': (None, [ct.c_char_p]),
            'ISC_LoadSettings': (ct.c_bool, [ct.c_char_p]),
            'ISC_StartPolling': (ct.c_bool, [ct.c_char_p, ct.c_int]),
            'ISC_StopPolling': (None, [ct.c_char_p]),
            'ISC_GetStatusBits': (ct.c_uint32, [ct.c_char_p]),
            'ISC_GetPosition': (ct.c_int, [ct.c_char_p]),
            'ISC_GetBacklash': (ct.c_long, [ct.c_char_p]),
            'ISC_GetSoftLimitMode': (ct.c_short, [ct.c_char_p]),
            'ISC_SetLimitsSoftwareApproachPolicy': (None, [ct.c_char_p, ct.c_short]),
            'ISC_GetRealValueFromDeviceUnit': (ct.c_short, [ct.c_char_p, ct.c_int, ct.POINTER(ct.c_double), ct.c_int]),
            'ISC_GetDeviceUnitFromRealValue': (ct.c_short, [ct.c_char_p, ct.c_double, ct.POINTER(ct.c_int), ct.c_int]),
            'ISC_GetLimitSwitchParams': (ct.c_short, [ct.c_char_p, ct.POINTER(ct.c_ushort), ct.POINTER(ct.c_ushort), ct.POINTER(ct.c_uint), ct.POINTER(ct.c_uint), ct.POINTER(ct.c_ushort)]),
            'ISC_SetLimitSwitchParams': (ct.c_short, [ct.c_char_p, ct.c_ushort, ct.c_ushort, ct.c_uint, ct.c_uint, ct.c_ushort]),
            'ISC_SetVelParams': (ct.c_short, [ct.c_char_p, ct.c_int, ct.c_int]),
            'ISC_MoveToPosition': (ct.c_short, [ct.c_char_p, ct.c_int]),
            'ISC_StopImmediate': (ct.c_short, [ct.c_char_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            f = getattr(self.dll, name)
            f.restype, f.argtypes = restype, argtypes
        try:
            self.check(self.dll.TLI_BuildDeviceList())
            self.check(self.dll.ISC_Open(self.serial))
            self.opened = True
            if not self.dll.ISC_LoadSettings(self.serial):
                raise RuntimeError('No validated device settings')
            if not self.dll.ISC_StartPolling(self.serial, 100):
                raise RuntimeError('Polling failed')
            self.polling = True
            time.sleep(.5)
            self.initial = self.status()
            if self.initial['moving']:
                raise RuntimeError(f'Stage moving during connection: {self.initial}')
            self.backlash_mm = abs(self.real(self.dll.ISC_GetBacklash(self.serial)))
            # Cross-check the model-specific unit scaling before any move.
            if abs(self.units(1.0)-409600) > 1:
                raise RuntimeError('Unexpected LTS300 coordinate conversion')
            self.original_limits = self.limits()
            # LoadSettings also reapplies the manufacturer's ignored soft-limit
            # defaults. Always re-arm immediately, including diagnostic opens.
            self.arm()
            if not self.monitor_only:
                validate_target(self.initial['position_mm'], self.interval, self.backlash_mm)
            return self
        except BaseException:
            self.close()
            raise

    def real(self, units, kind=0):
        v = ct.c_double()
        self.check(self.dll.ISC_GetRealValueFromDeviceUnit(self.serial, units, ct.byref(v), kind))
        return v.value

    def units(self, value, kind=0):
        v = ct.c_int()
        self.check(self.dll.ISC_GetDeviceUnitFromRealValue(self.serial, value, ct.byref(v), kind))
        return v.value

    def status(self):
        self.check(self.dll.ISC_RequestStatus(self.serial))
        time.sleep(.12)
        bits = self.dll.ISC_GetStatusBits(self.serial)
        return dict(position_mm=self.real(self.dll.ISC_GetPosition(self.serial)),
                    status_hex=hex(bits), homed=bool(bits & 0x400),
                    moving=bool(bits & 0x2f0), enabled=bool(bits & 0x80000000))

    @staticmethod
    def require_stationary(status):
        if not status['homed'] or status['moving'] or not status['enabled']:
            raise RuntimeError(f'Stage is not ready: {status}')

    def limits(self):
        self.check(self.dll.ISC_RequestLimitSwitchParams(self.serial))
        time.sleep(.15)
        v = [ct.c_ushort(), ct.c_ushort(), ct.c_uint(), ct.c_uint(), ct.c_ushort()]
        self.check(self.dll.ISC_GetLimitSwitchParams(self.serial, *[ct.byref(x) for x in v]))
        return [x.value for x in v]

    def arm(self):
        """Apply runtime safety limits, preserving hardware switch modes. No motion."""
        if self.status()['moving']:
            raise RuntimeError('Cannot change runtime limits while stage is moving')
        cw, ccw, *_ = self.original_limits
        self.check(self.dll.ISC_SetLimitSwitchParams(self.serial, cw, ccw,
                   self.units(GUARD_MM), self.units(0.), 2))
        self.dll.ISC_SetLimitsSoftwareApproachPolicy(self.serial, 0)
        expected = [cw, ccw, self.units(GUARD_MM), 0, 2]
        if self.limits() != expected or self.dll.ISC_GetSoftLimitMode(self.serial) != 0:
            raise RuntimeError('Safety limit readback mismatch')
        # 0.5 mm/s maximum velocity and 1 mm/s^2 acceleration.
        self.check(self.dll.ISC_SetVelParams(self.serial, self.units(1., 2), self.units(.5, 1)))
        self.armed = False
        if not self.monitor_only:
            self.require_stationary(self.status())
            self.armed = True

    def move(self, target, timeout=25):
        if self.monitor_only or not self.opened or not getattr(self, 'armed', False):
            raise RuntimeError('Stage safety limits have not been armed')
        target = validate_target(target, self.interval, self.backlash_mm)
        before = self.status()
        self.require_stationary(before)
        validate_target(before['position_mm'], self.interval, self.backlash_mm)
        self.check(self.dll.ISC_MoveToPosition(self.serial, self.units(target)))
        self.moves.append(dict(target_mm=target, from_mm=before['position_mm'], time=time.time()))
        deadline = time.monotonic()+timeout
        try:
            while time.monotonic() < deadline:
                s = self.status()
                if not s['homed'] or not s['enabled'] or not 0 <= s['position_mm'] < GUARD_MM:
                    raise RuntimeError(f'Unsafe moving state: {s}')
                if not s['moving'] and abs(s['position_mm']-target) < .002:
                    time.sleep(.35)
                    return self.status()
            raise TimeoutError(f'Stage did not reach {target} mm')
        except BaseException:
            self.dll.ISC_StopImmediate(self.serial)
            raise

    def close(self):
        self.armed = False
        if self.opened:
            if self.polling:
                self.dll.ISC_StopPolling(self.serial)
            self.dll.ISC_Close(self.serial)
            self.opened = self.polling = False
        # Intentionally do not restore the previous ignored soft limits or fast speed.
        # No device EEPROM write is made; rerun arm() after settings reload/power cycle.

    def __exit__(self, *_):
        self.close()
