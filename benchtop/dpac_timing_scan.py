"""Fresh full-frame flat-field trigger scan, with no stage motion."""
import argparse
import numpy as np
from .dpac_lut_session import Session
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .stage import Stage
from . import step01_dot_grid as grid


def run(root):
    session = Session(root, 'timing_scan_01')
    with exclusive():
        laser = ReadyBeam()
        try:
            session.report['standby'] = laser.configure('Digital')
            with Stage(monitor_only=True) as stage:
                session.report['stage'] = stage.status()
                stage.require_stationary(session.report['stage'])
                with Bench(dict(grid.C, settle_s=.25)) as session.b:
                    session.b.show_codes(session.flat)
                    session.b.set_timing(175, 22)
                    session.capture('dark')
                    session.report['external'] = laser.configure('Analog')
                    for delay in [22, *range(1000, 18001, 1000)]:
                        session.b.set_timing(175, delay)
                        raw = session.capture(f'delay_{delay:05}')
                        means = [float(raw[::2, ::2].mean()/65535),
                                 float((raw[::2, 1::2].mean()+raw[1::2, ::2].mean())/131070),
                                 float(raw[1::2, 1::2].mean()/65535)]
                        session.report['records'][-1]['mean_rgb'] = means
                        session.save()
                        print(delay, means, flush=True)
                    session.report['state'] = 'scan_complete'
        except BaseException as exc:
            session.report.update(state='interrupted', error=repr(exc))
            raise
        finally:
            try:
                session.report['final_standby'] = laser.configure('Digital')
            finally:
                session.save()


if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    run(p.parse_args().root)
