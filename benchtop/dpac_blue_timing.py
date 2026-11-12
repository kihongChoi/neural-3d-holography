"""Check earlier blue windows for channel isolation and flat-field stability."""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from .dpac_lut_session import Session, sha
from . import step01_dot_grid as grid
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .rgb_roi_calibration import roi_mask, routing, stats
from .rgb_study_block import rectify
from .stage import Stage


def run(root, channel='B', *, geometry_source=None, settings_override=None, repetitions=1, stability=False):
    label = 'blue_timing_01' if channel == 'B' else 'green_timing_01'
    s = Session(root, label)
    prior = Path(geometry_source) if geometry_source is not None else root/'calibration_04'/'status.json'
    previous = json.loads(prior.read_text())
    ch = 'RGB'.index(channel)
    blue = next(x for x in previous['calibrations'] if x['mode'] == channel and x['depth_mm'] == 0 and x['registration']['accepted'])
    H = blue['registration']['models']['pooled']
    mask, _ = roi_mask(H, (3000, 4096))
    s.report.update(geometry_source=str(prior), geometry_sha256=sha(prior), trials=[])
    with exclusive():
        laser = ReadyBeam()
        try:
            s.report['standby'] = laser.configure('Digital')
            with Stage(interval=(90.780, 101.000)) as stage:
                start = stage.initial['position_mm']
                try:
                    stage.move(grid.ZERO)
                    with Bench(dict(grid.C, settle_s=.25)) as s.b:
                        s.b.show_codes(s.flat)
                        s.report['external'] = laser.configure('Analog')
                        settings = ([(175, 12222), (85, 12222), (175, 12172)] if channel == 'B'
                                    else [(175, 6422), (175, 6322), (175, 6222)])
                        if settings_override is not None:
                            settings = list(settings_override)
                        schedule = [(repeat, timing) for repeat in range(repetitions)
                                    for timing in (settings if repeat % 2 == 0 else reversed(settings))]
                        for repeat, (exposure, delay) in schedule:
                            key = f'{channel}_{exposure}_{delay}' + (f'_r{repeat}' if repetitions > 1 else '')
                            timing = s.b.set_timing(exposure, delay)
                            s.b.show_codes(s.flat)
                            before = s.capture(key+'_white_start')
                            route = routing(s.b, s.flat, mask, s.camera, key, s.enc, channel)
                            s.b.show_codes(s.flat)
                            after = s.capture(key+'_white_end')
                            drift = float(rectify(after, ch, H).mean()/rectify(before, ch, H).mean()-1)
                            trial = dict(requested=[exposure, delay], repeat=repeat, timing=timing, routing=route,
                                         white_roi=stats(before, mask), white_drift_fraction=drift)
                            trial['accepted'] = bool(route['accepted'] and abs(drift) <= .03 and
                                                     trial['white_roi'][channel]['near_clip'] <= .0001)
                            s.report['trials'].append(trial)
                            s.save()
                            print(json.dumps(trial), flush=True)
                        if channel == 'G' or stability:
                            accepted = []
                            for setting in settings:
                                trials = [t for t in s.report['trials'] if t['requested']==list(setting)]
                                if len(trials)==repetitions and all(t['accepted'] for t in trials):
                                    accepted.append((setting,float(np.mean([t['white_roi'][channel]['mean'] for t in trials]))))
                            if not accepted:
                                raise RuntimeError(f'No repeatably isolated {channel} timing candidate')
                            baseline = tuple(settings[0])
                            chosen_setting = (baseline if settings_override is not None and baseline in dict(accepted)
                                              else max(accepted,key=lambda item:item[1])[0])
                            chosen = next(t for t in s.report['trials'] if t['requested']==list(chosen_setting))
                            s.report['selected_timing'] = list(chosen_setting)
                            s.b.set_timing(*chosen['requested'])
                            s.b.show_codes(s.flat)
                            samples = []
                            start_time = time.monotonic()
                            for index in range(9):
                                while time.monotonic() < start_time+15*index:
                                    s.b.app.processEvents()
                                    time.sleep(.05)
                                raw = s.capture(f'{channel}_stability_{index:02}')
                                samples.append(float(rectify(raw, ch, H).mean()))
                                s.report['stability'] = dict(requested=chosen['requested'],
                                    elapsed_s=time.monotonic()-start_time, means=samples,
                                    max_fraction_from_first=max(abs(v/samples[0]-1) for v in samples))
                                s.save()
                                print(channel, 'stability', index, samples[-1]/samples[0]-1, flush=True)
                            s.report['stability']['accepted'] = s.report['stability']['max_fraction_from_first'] <= .03
                            s.save()
                            if not s.report['stability']['accepted']:
                                raise RuntimeError(f'Two-minute {channel} stability gate failed')
                finally:
                    s.report['standby_before_return'] = laser.configure('Digital')
                    if stage.moves:
                        stage.move(start)
                    s.report['stage_end'] = stage.status()
            s.report['state'] = 'timing_trials_complete'
        except BaseException as exc:
            s.report.update(state='interrupted', error=repr(exc))
            raise
        finally:
            try:
                s.report['final_standby'] = laser.configure('Digital')
            finally:
                s.save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--channel', choices=['B', 'G'], default='B')
    a = p.parse_args()
    run(a.root, a.channel)
