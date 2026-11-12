"""Fresh DPAC calibration and measured LUT selection; no historical data defaults."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import torch  # Must precede vendor camera/motion DLL loading on Windows.
import cv2
import numpy as np

from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .rgb_roi_calibration import roi_mask, stats, routing
from .rgb_study_block import rectify
from .stage import Stage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


class Session:
    def __init__(self, root, label, slot_offset_us=0):
        self.root = root
        self.out = root / label
        self.out.mkdir(exist_ok=False)
        self.camera = self.out / 'camera'
        self.camera.mkdir()
        self.report = dict(version='v3.0-dpac', started_kst=now(), state='preparing',
                           root=str(root.resolve()), source_sha256=sha(__file__),
                           nominal_zero_mm=grid.ZERO, nd_od_planned=.4,
                           records=[], calibrations=[], routing={})
        v = np.arange(65536) / 65535
        self.enc = np.uint8(np.rint(255*np.where(v <= .0031308, 12.92*v,
                                                1.055*v**(1/2.4)-.055)))
        self.flat = np.full((1080, 1920, 3), 128, np.uint8)
        self.modes = dict(R=(175, 22), G=(350, 6022+slot_offset_us),
                          B=(350, 12022+slot_offset_us), RGB=(15000, 22))
        self.report['requested_modes'] = self.modes
        self.save()

    def save(self):
        self.report['updated_kst'] = now()
        tmp = self.out / 'status.tmp'
        tmp.write_text(json.dumps(self.report, indent=2), encoding='utf8')
        # Windows readers and scanners can briefly deny replacing an open file.
        # Keep the previous complete JSON until an atomic replacement succeeds.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                tmp.replace(self.out / 'status.json')
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)

    def capture(self, name):
        if (self.root / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED: stopped between captures')
        b = self.b
        b.verify_display()
        cam = b.camera
        cam.BeginAcquisition()
        b.acquiring = True
        try:
            for index in range(2):
                frame = cam.GetNextImage(4000)
                try:
                    if frame.IsIncomplete():
                        raise RuntimeError('Incomplete camera frame')
                    if index:
                        raw = frame.GetNDArray().copy()
                        meta = dict(frame_id=int(frame.GetFrameID()),
                                    timestamp_ns=int(frame.GetTimeStamp()))
                finally:
                    frame.Release()
        finally:
            try:
                cam.EndAcquisition()
            finally:
                b.acquiring = False
                del cam
        b.verify_display()
        if raw.shape != (3000, 4096) or raw.dtype != np.uint16:
            raise RuntimeError(f'Expected full BayerRG16 frame: {raw.shape}, {raw.dtype}')
        path = self.camera / (name + '.png')
        grid.png(path, self.enc[cv2.cvtColor(raw, cv2.COLOR_BayerRGGB2BGR)])
        self.report['records'].append(dict(name=name, path=str(path.relative_to(self.root)),
            png_sha256=sha(path), time_kst=now(), timing=b.active_timing.copy(),
            full_frame_near_clip=float((raw >= .98*65535).mean()), **meta))
        self.save()
        return raw

    def calibrate(self, z, frames, route=False):
        self.report['state'] = f'calibrating_{z:g}mm'
        self.save()
        self.laser.configure('Digital')
        self.stage.move(round(grid.ZERO-z, 6))
        self.b.show_codes(self.flat)
        dark = {}
        for mode, timing in self.modes.items():
            self.b.set_timing(*timing)
            dark[mode] = self.capture(f'z{z:g}_{mode}_dark')
        self.report['last_laser_external'] = self.laser.configure('Analog')
        calibrated = {}
        for mode, timing in self.modes.items():
            prefix = f'z{z:g}_{mode}'
            self.b.set_timing(*timing)
            raw = {}
            for kind in ['blank', *geo.FIT, geo.CHECK]:
                self.b.show_codes(frames[kind])
                raw[kind] = self.capture(f'{prefix}_geometry_{kind}')
            registration = geo.analyze(raw, z)
            entry = dict(depth_mm=z, mode=mode, stage=self.stage.status(),
                         timing=self.b.active_timing.copy(), registration=registration)
            self.report['calibrations'].append(entry)
            self.save()
            if not registration['accepted']:
                raise RuntimeError(f'{prefix}: independent geometry gate: {registration["scores"]["pooled"]}')
            H = registration['models']['pooled']
            mask, _ = roi_mask(H, (3000, 4096))
            self.b.show_codes(self.flat)
            white = self.capture(f'{prefix}_white')
            entry['white_roi'] = stats(white, mask)
            self.save()
            channels = range(3) if mode == 'RGB' else ['RGB'.index(mode)]
            for ch in channels:
                tag = 'RGB'[ch]
                q = entry['white_roi'][tag]
                if q['near_clip'] > .0001 or q['p998'] >= .95:
                    raise RuntimeError(f'{prefix}: white exposure rejected: {q}')
            calibrated[mode] = dict(H=H, mask=mask, dark=dark[mode], white=white)
            if route:
                # The routing helper saves full-frame 8-bit PNGs only.
                result = routing(self.b, self.flat, mask, self.camera,
                                 prefix, self.enc, mode)
                self.report['routing'][mode] = result
                self.save()
                if not result['accepted']:
                    raise RuntimeError(f'{prefix}: HDMI slot routing rejected: {result}')
            print(prefix, 'geometry/exposure accepted', flush=True)
            del raw
        self.laser.configure('Digital')
        return calibrated

    def run(self, depths):
        frames = {z: geo.cached_patterns(z, self.root / 'calibration_patterns') for z in depths}
        self.report['pattern_hashes'] = {p.name: sha(p) for p in
                                       (self.root / 'calibration_patterns').glob('*.png')}
        self.save()
        with exclusive():
            self.laser = ReadyBeam()
            try:
                self.report['standby'] = self.laser.configure('Digital')
                with Stage(interval=(90.780, 101.000)) as self.stage:
                    start = self.stage.initial['position_mm']
                    self.report['stage_start'] = self.stage.status()
                    try:
                        with Bench(dict(grid.C, settle_s=.25)) as self.b:
                            if (self.b.camera.Width.GetValue(), self.b.camera.Height.GetValue(),
                                self.b.camera.OffsetX.GetValue(), self.b.camera.OffsetY.GetValue()) != (4096, 3000, 0, 0):
                                raise RuntimeError('Full sensor dimensions/offset gate failed')
                            self.report['camera_original'] = self.b.original
                            for z in depths:
                                self.calibrate(z, frames[z], route=(z == 0))
                    finally:
                        self.report['standby_before_return'] = self.laser.configure('Digital')
                        if self.stage.moves:
                            self.stage.move(start)
                        self.report['stage_end'] = self.stage.status()
                        self.report['moves'] = self.stage.moves
                self.report['state'] = 'calibration_complete'
            except BaseException as exc:
                self.report.update(state='interrupted', error=repr(exc))
                raise
            finally:
                try:
                    self.report['final_standby'] = self.laser.configure('Digital')
                finally:
                    self.save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--label', default='calibration_01')
    p.add_argument('--depths', type=float, nargs='+', default=[0, 3, 7, 9])
    p.add_argument('--slot-offset-us', type=float, default=0)
    p.add_argument('--blue-delay-us', type=float)
    p.add_argument('--blue-exposure-us', type=float, default=350)
    args = p.parse_args()
    session = Session(args.root, args.label, args.slot_offset_us)
    if args.blue_delay_us is not None:
        session.modes['B'] = (args.blue_exposure_us, args.blue_delay_us)
        session.save()
    session.run(args.depths)
