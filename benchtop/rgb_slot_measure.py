"""Measure slot routing and integrated exposure without moving the stage."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from . import step01_dot_grid as grid
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .stage import Stage


def measure(out):
    out.mkdir(parents=True, exist_ok=False)
    levels = np.arange(65536) / 65535
    lut = np.uint8(np.rint(255 * np.where(levels <= .0031308, 12.92*levels,
                                         1.055*levels**(1/2.4)-.055)))
    blank = np.full((1080, 1920, 3), 128, np.uint8)
    checker = np.uint8(64 + 128*(np.indices((960, 1728)).sum(0) % 2))
    records = []
    report = dict(plan='v2.1', revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        camera_serial=grid.C['camera_serial'], records=records)

    def save():
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(report, indent=2),encoding='utf8')
        tmp.replace(out/'status.json')

    def capture(name):
        raw = grid.full_capture(b)
        if raw.shape != (3000,4096) or raw.dtype != np.uint16:
            raise ValueError('Expected native full-frame BayerRG16')
        grid.png(out/(name+'.png'), lut[cv2.cvtColor(raw, cv2.COLOR_BayerRGGB2BGR)])
        # Use Bayer samples directly for quantitative metrics.
        rgb = np.stack([raw[::2,::2], (raw[::2,1::2].astype('f')+raw[1::2,::2])/2,
                        raw[1::2,1::2]], axis=-1)/65535
        records.append(dict(name=name, timing=b.active_timing, rgb_mean=rgb.mean((0,1)).tolist(),
                            near_clip=float((raw>=.98*65535).mean()),
                            p998=np.quantile(rgb[::4,::4],.998,axis=(0,1)).tolist()))
        save()
        print(name, records[-1]['rgb_mean'],records[-1]['near_clip'],flush=True)
        return rgb

    save()
    with exclusive():
        laser=ReadyBeam()
        try:
            report['standby']=laser.configure('Digital')
            with Stage(serial='45897069',interval=grid.SCAN_RANGE,monitor_only=True) as stage:
                report['stage']=stage.status()
                stage.require_stationary(report['stage'])
                with Bench(dict(grid.C,settle_s=.22)) as b:
                    b.show_codes(blank)
                    b.set_timing(15000,22)
                    capture('standby_dark')
                    report['external_trigger']=laser.configure('Analog')
                    for ch,delay in enumerate((22,6022,12022)):
                        tag='RGB'[ch]
                        b.show_codes(blank)
                        for requested in (350,175,85,40):
                            b.set_timing(requested,delay)
                            ref=capture(f'{tag}_white_{requested}')
                            if records[-1]['near_clip']<=.0001 and records[-1]['p998'][ch]<.8:
                                break
                        else:
                            raise RuntimeError(f'{tag}: exposure candidates saturated')
                        responses=[]
                        for k in range(3):
                            frame=blank.copy()
                            frame[grid.ROI][...,k]=checker
                            b.show_codes(frame)
                            test=capture(f'{tag}_change_{"RGB"[k]}')
                            responses.append(float(np.abs(test[...,ch]-ref[...,ch]).mean()))
                        b.show_codes(blank)
                        repeat=capture(f'{tag}_repeat')
                        noise=float(np.abs(repeat[...,ch]-ref[...,ch]).mean())
                        report[tag]=dict(timing=b.active_timing,repeat_mae=noise,response=responses)
                        save()
                    b.show_codes(blank)
                    b.set_timing(15000,22)
                    capture('RGB_integrated_white')
                    capture('RGB_integrated_repeat')
                    report['stage_end']=stage.status()
        except BaseException as exc:
            report['error']=repr(exc)
            raise
        finally:
            try:
                report['final_standby']=laser.configure('Digital')
            finally:
                save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    measure(parser.parse_args().out)
