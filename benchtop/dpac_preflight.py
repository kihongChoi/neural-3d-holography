"""Record independent, stationary hardware checks for a fresh DPAC run."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import traceback


def run(out):
    from .readybeam import ReadyBeam, exclusive
    from .stage import Stage
    report = {'time_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
              'stage_motion_requested': False, 'checks': {}}
    destination = out / 'preflight.json'
    if destination.exists():
        raise FileExistsError(destination)

    def check(name, action):
        try:
            report['checks'][name] = {'ok': True, 'result': action()}
        except Exception as exc:
            report['checks'][name] = {'ok': False, 'error': repr(exc),
                                      'traceback': traceback.format_exc()}
        destination.write_text(json.dumps(report, indent=2), encoding='utf8')
        print(name, json.dumps(report['checks'][name]), flush=True)

    def stage():
        with Stage(monitor_only=True) as s:
            state = s.status()
            result = dict(state=state, limits=s.limits(),
                          upper_limit_mm=s.real(s.limits()[2]),
                          soft_limit_mode=s.dll.ISC_GetSoftLimitMode(s.serial))
            result['ready_for_motion'] = state['homed'] and state['enabled'] and not state['moving']
            return result

    def laser():
        device = ReadyBeam()
        before = device.read()
        return dict(before=before, standby=device.configure('Digital'))

    def camera():
        import PySpin
        system = PySpin.System.GetInstance()
        cameras = system.GetCameras()
        found = []
        try:
            for i in range(cameras.GetSize()):
                cam = cameras.GetByIndex(i)
                try:
                    serial = cam.TLDevice.DeviceSerialNumber.GetValue()
                    cam.Init()
                    try:
                        found.append(dict(serial=serial, width=cam.Width.GetValue(),
                                          height=cam.Height.GetValue(), exposure_us=cam.ExposureTime.GetValue()))
                    finally:
                        cam.DeInit()
                finally:
                    del cam
        finally:
            cameras.Clear()
            system.ReleaseInstance()
        if not any(x['serial'] == '25222112' for x in found):
            raise RuntimeError(f'Expected camera not found: {found}')
        return found

    def display():
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        screens = [dict(name=s.name(), size=list(s.size().toTuple()), scale=s.devicePixelRatio())
                   for s in app.screens()]
        return dict(screens=screens, expected_present=any(
            s['name'] == 'LETO-3-0088' and s['size'] == [1920, 1080] and s['scale'] == 1
            for s in screens))

    def compute():
        import torch
        return dict(cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)

    with exclusive():
        # Load torch before vendor DLLs; their libraries can otherwise shadow
        # torch dependencies in this process on Windows.
        for name, action in [('compute', compute), ('stage', stage), ('laser', laser),
                             ('camera', camera), ('display', display)]:
            check(name, action)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
