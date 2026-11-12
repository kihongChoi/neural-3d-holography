"""Bounded camera mailbox and independent, serialized hardware workers."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
from PIL import Image

from .viewer_devices import StreamCamera, ViewerStage, step_target


def kst_now():
    return datetime.now(timezone(timedelta(hours=9)))


def new_session(results, demo=False):
    path = Path(results) / (kst_now().strftime('%Y%m%d-%H%M%S') +
                            ('_camera-live-viewer-demo' if demo else '_camera-live-viewer'))
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


class Shared:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.lock = threading.Lock()
        self.states = {name: dict(connected=False, message='연결 안 됨') for name in ('camera', 'stage', 'laser')}
        self.latest = None
        self.sequence = 0
        self.messages = queue.Queue(maxsize=100)

    def update(self, name, **state):
        with self.lock:
            self.states[name].update(state)

    def snapshot(self):
        with self.lock:
            return {k: dict(v) for k, v in self.states.items()}

    def frame(self, rgb, meta):
        with self.lock:
            self.sequence += 1
            self.latest = (self.sequence, rgb, meta)

    def get_frame(self):
        with self.lock:
            return self.latest

    def clear_frame(self):
        with self.lock:
            self.latest = None

    def log(self, name, message, **details):
        row = dict(time_kst=kst_now().isoformat(), device=name, message=message, **details)
        # The same lock serializes log writes from all device threads.
        with self.lock:
            with (self.folder/'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        try:
            self.messages.put_nowait(row)
        except queue.Full:
            pass


class Worker(threading.Thread):
    def __init__(self, shared, name):
        super().__init__(name=f'viewer-{name}', daemon=False)
        self.shared, self.device_name = shared, name
        self.quit = threading.Event()
        self.commands = queue.Queue(maxsize=1)

    def submit(self, *command):
        if not self.is_alive() or self.quit.is_set():
            return False
        try:
            self.commands.put_nowait(command)
            return True
        except queue.Full:
            return False

    def error(self, exc):
        self.shared.update(self.device_name, message=str(exc), error=str(exc))
        self.shared.log(self.device_name, str(exc), error=True)


class CameraWorker(Worker):
    def __init__(self, shared, factory=StreamCamera, exposure=15000):
        super().__init__(shared, 'camera')
        self.factory, self.exposure = factory, exposure

    def run(self):
        camera = self.factory(self.exposure)
        started, count, last_frame = time.monotonic(), 0, time.monotonic()
        self.shared.clear_frame()
        self.shared.update('camera', connected=False, error=None, message='카메라 연결 중…', fps=0)
        try:
            settings = camera.connect()
            self.shared.update('camera', connected=True, message='외부 트리거 대기', **settings)
            self.shared.log('camera', 'connected', **settings)
            while not self.quit.is_set():
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    settings = camera.set_exposure(command[1])
                    self.shared.update('camera', **settings)
                    self.shared.log('camera', 'exposure', **settings)
                result = camera.grab()
                if result is None:
                    if time.monotonic()-last_frame > 1:
                        self.shared.update('camera', message='V-Sync 트리거 대기 · 영상 갱신 없음', fps=0)
                    continue
                rgb, meta = result
                last_frame = time.monotonic()
                count += 1
                self.shared.frame(rgb, meta)
                self.shared.update('camera', message='실시간 수신 중', frame_id=meta['frame_id'],
                                   timestamp_ns=meta['timestamp_ns'], last_frame=last_frame)
                elapsed = time.monotonic()-started
                if elapsed >= 1:
                    self.shared.update('camera', fps=count/elapsed)
                    started, count = time.monotonic(), 0
        except Exception as exc:
            self.error(exc)
        finally:
            try:
                camera.close()
            except Exception as exc:
                self.error(exc)
            self.shared.update('camera', connected=False, fps=0)
            self.shared.log('camera', 'disconnected')


class StageWorker(Worker):
    def __init__(self, shared, factory=ViewerStage):
        super().__init__(shared, 'stage')
        self.factory = factory
        self.stop_motion = threading.Event()

    def stop(self):
        self.stop_motion.set()

    def run(self):
        stage = self.factory()
        self.shared.update('stage', message='스테이지 연결 중…', error=None)
        try:
            stage.__enter__()
            self.shared.update('stage', connected=True, **stage.poll())
            self.shared.log('stage', 'connected', **stage.initial)
            while not self.quit.is_set():
                if self.stop_motion.is_set():
                    stage.stop()
                    while not self.commands.empty():
                        self.commands.get_nowait()
                    self.stop_motion.clear()
                    self.shared.log('stage', 'stop')
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    try:
                        if command[0] == 'home':
                            stage.start_home()
                            self.shared.log('stage', 'home_requested')
                        elif command[0] == 'step':
                            target = stage.start_step(command[1], command[2])
                            self.shared.log('stage', 'step_requested', target_mm=target, speed_mm_s=stage.speed_mm_s)
                        elif command[0] == 'speed':
                            speed = stage.set_speed(command[1])
                            self.shared.log('stage', 'speed_applied', speed_mm_s=speed)
                        self.shared.update('stage', error=None)
                    except Exception as exc:
                        self.error(exc)
                state = stage.poll()
                self.shared.update('stage', connected=True, **state,
                                   message='호밍 중' if state['operation'] == 'home' else
                                   ('이동 중' if state['moving'] or state['operation'] else '정지'))
                self.quit.wait(.08)
        except Exception as exc:
            self.error(exc)
        finally:
            try:
                stage.close()
            except Exception as exc:
                self.error(exc)
            self.shared.update('stage', connected=False)
            self.shared.log('stage', 'disconnected')


class LaserWorker(Worker):
    def __init__(self, shared, factory=None, guard=None):
        super().__init__(shared, 'laser')
        self.factory, self.guard = factory, guard
        self.off = threading.Event()

    def standby(self):
        self.off.set()

    def run(self):
        if self.factory is None:
            from .readybeam import ReadyBeam, exclusive
            factory, guard = ReadyBeam, exclusive
        else:
            factory, guard = self.factory, self.guard or nullcontext
        self.shared.update('laser', message='레이저 연결 중…', error=None, on=False)
        device = None
        try:
            with guard():
                try:
                    device = factory()
                    state = device.configure('Digital')
                    self.report(state)
                    last_poll = time.monotonic()
                    while not self.quit.is_set():
                        command = None
                        if self.off.is_set():
                            while not self.commands.empty():
                                self.commands.get_nowait()
                            self.off.clear()
                            command = ('off',)
                        else:
                            try:
                                command = self.commands.get_nowait()
                            except queue.Empty:
                                pass
                        camera = self.shared.snapshot()['camera']
                        fresh = camera['connected'] and time.monotonic()-camera.get('last_frame', 0) < 2
                        on = state['mode'] == 'Analog'
                        if command or (on and not fresh):
                            mode = 'Analog' if command and command[0] == 'on' and fresh else 'Digital'
                            state = device.configure(mode)
                            self.report(state)
                            last_poll = time.monotonic()
                        elif time.monotonic()-last_poll > 2:
                            state = device.read()
                            if state['error'] != 'No Error':
                                raise RuntimeError(str(state))
                            self.report(state, log=False)
                            last_poll = time.monotonic()
                        self.quit.wait(.1)
                finally:
                    # Keep ownership until final standby has been read back.
                    if device is not None:
                        self.report(device.configure('Digital'))
        except Exception as exc:
            self.error(exc)
            self.shared.update('laser', on=None, message=f'레이저 상태 확인 필요: {exc}')
        finally:
            self.shared.update('laser', connected=False)

    def report(self, state, log=True):
        on = state['mode'] == 'Analog'
        self.shared.update('laser', connected=True, on=on, error=None, readback=state,
                           message='ON · 외부 동기' if on else 'OFF · Digital / 0%')
        if log:
            self.shared.log('laser', 'readback', **state)


class SnapshotWorker(Worker):
    def __init__(self, shared):
        super().__init__(shared, 'snapshot')

    def run(self):
        while not self.quit.is_set() or not self.commands.empty():
            try:
                _, rgb, meta = self.commands.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                folder = self.shared.folder/'camera'
                folder.mkdir(exist_ok=True)
                path = folder/(kst_now().strftime('%Y%m%d-%H%M%S-%f') + f"_frame-{meta['frame_id']}.png")
                with path.open('xb') as f:
                    Image.fromarray(rgb).save(f, format='PNG')
                self.shared.log('snapshot', 'PNG 저장 완료', path=str(path), **meta)
            except Exception as exc:
                self.shared.log('snapshot', str(exc), error=True)


class DemoCamera:
    """Explicit demo only; never a fallback for disconnected hardware."""
    def __init__(self, exposure):
        self.exposure, self.number = exposure, 0

    def settings(self):
        return dict(exposure_us=self.exposure, exposure_min=20, exposure_max=100000,
                    width=1024, height=750, trigger='DEMO', delay_us=22)

    def connect(self):
        return self.settings()

    def set_exposure(self, value):
        self.exposure = value
        return self.settings()

    def grab(self):
        time.sleep(.065)
        self.number += 1
        y, x = np.mgrid[:750, :1024]
        rgb = np.stack(((x+self.number*3) % 256, y % 256, (x//3+y//3) % 256), -1)
        rgb = np.clip(rgb*(self.exposure/15000), 0, 255).astype(np.uint8)
        return rgb, dict(frame_id=self.number, timestamp_ns=time.monotonic_ns(),
                         received=time.monotonic(), exposure_us=self.exposure)

    def close(self):
        pass


class DemoStage:
    def __init__(self):
        self.position, self.target, self.motion = 100., 100., None
        self.speed_mm_s = .5
        self.last_poll = time.monotonic()
        self.initial = dict(position_mm=100., homed=True, moving=False, enabled=True)

    def __enter__(self):
        return self

    def poll(self):
        elapsed = time.monotonic()-self.last_poll
        self.last_poll = time.monotonic()
        if self.motion:
            distance = (.5 if self.motion == 'home' else self.speed_mm_s)*elapsed
            self.position += float(np.clip(self.target-self.position, -distance, distance))
            if abs(self.position-self.target) < .001:
                self.motion = None
        return dict(position_mm=self.position, homed=True, moving=bool(self.motion),
                    enabled=True, operation=self.motion, status_hex='DEMO', speed_mm_s=self.speed_mm_s)

    def set_speed(self, value):
        from .viewer_devices import validate_speed
        if self.motion:
            raise RuntimeError('이동 중에는 속도를 변경할 수 없습니다.')
        self.speed_mm_s = validate_speed(value)
        return self.speed_mm_s

    def start_step(self, step, direction):
        self.target = step_target(self.poll(), step, direction)
        self.motion = 'move'
        return self.target

    def start_home(self):
        self.target, self.motion = 0., 'home'

    def stop(self):
        self.target, self.motion = self.position, None

    def close(self):
        self.stop()


class DemoLaser:
    def configure(self, mode):
        self.mode = mode
        return self.read()

    def read(self):
        return dict(mode=self.mode, enable_BGR=['Off']*3, percent_BGR=[0]*3, error='No Error', demo=True)
