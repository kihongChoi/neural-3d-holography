"""Infer propagation direction from opposite carrier tilts relative to optical zero."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench,encode_phase
from .readybeam import ReadyBeam,exclusive
from .rgb_study_block import CHANNELS,MODES,rectify
from .stage import Stage


def run(out,cal,route):
    out.mkdir(parents=True,exist_ok=False)
    h,w=grid.HEIGHT,grid.WIDTH
    y,x=np.indices((h,w),dtype=np.float32)
    target=cv2.resize(np.random.default_rng(731).uniform(-1,1,(h//9,w//9)).astype('f'),(w,h),interpolation=cv2.INTER_CUBIC).clip(-1,1)
    checker=np.where((x+y)%2,1.,-1.).astype('f')
    levels=np.arange(65536)/65535
    enc=np.uint8(np.rint(255*np.where(levels<=.0031308,12.92*levels,1.055*levels**(1/2.4)-.055)))
    windows={}
    period=8
    status=dict(plan='v2.1',added_nd_od=.4,z=1.25,calibration=str(cal),routing=str(route),slots={},
                reference_depth_mm=0.,period_slm_px=period,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2))
        tmp.replace(out/'status.json')
    def emit(name,raw):
        grid.png(out/(name+'.png'),enc[cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)])
    save()
    with exclusive():
        laser=ReadyBeam()
        try:
            status['standby']=laser.configure('Digital')
            with Stage(serial='45897069',interval=(grid.SCAN_RANGE[0],101.)) as s:
                initial=s.initial['position_mm']
                try:
                    with Bench(dict(grid.C,settle_s=.22)) as b:
                        status['external']=laser.configure('Analog')
                        b.show_codes(np.full((1080,1920,3),128,np.uint8))
                        for z in (0.,1.25):
                            frames=geo.cached_patterns(z,grid.ROOT/'results/step03_geometry/pattern_cache')
                            for ch,tag in enumerate(CHANNELS):
                                slot=status['slots'].setdefault(tag,{'geometry':{},'trials':{}})
                                b.set_timing(*MODES[ch])
                                reg=geo.move_and_align(s,b,z,frames,save=lambda images:
                                    [emit(f'z{z}_{tag}_{key}',im) for key,im in images.items()])
                                slot['geometry'][z]=reg
                                save()
                                if not reg['accepted']:raise RuntimeError(f'Geometry failed {tag} {z}')
                                for axis,coord in [('x',x),('y',y)]:
                                    shifts=[]
                                    for polarity in (1,-1):
                                        captured=[]
                                        for contrast in (1,-1):
                                            amplitude=.55+.36*contrast*target
                                            phase=polarity*2*np.pi*coord/period+checker*np.arccos(amplitude)
                                            f=np.full((1080,1920,3),128,np.uint8)
                                            f[grid.ROI][...,ch]=encode_phase(phase)
                                            b.show_codes(f)
                                            raw=grid.full_capture(b)
                                            emit(f'z{z}_{tag}_{axis}_{polarity}_{contrast}',raw)
                                            captured.append(rectify(raw,ch,reg['models']['pooled']))
                                        diff=(captured[0]-captured[1])/np.maximum(300,captured[0]+captured[1])
                                        feature=cv2.GaussianBlur(diff,(0,0),1.2)-cv2.GaussianBlur(diff,(0,0),13.)
                                        feature=np.ascontiguousarray(feature[96:-96,96:-96])
                                        key=(tag,axis,polarity)
                                        if not z:
                                            windows[key]=feature
                                        else:
                                            win=cv2.createHanningWindow(feature.shape[::-1],cv2.CV_32F)
                                            shift,peak=cv2.phaseCorrelate(windows[key].copy(),feature,win)
                                            shifts.append(dict(dx=shift[0],dy=shift[1],peak=peak))
                                    if z:
                                        dim='d'+axis
                                        observed=shifts[0][dim]-shifts[1][dim]
                                        pitch=grid.C['slm_pitch_m']*abs(grid.C['relay_magnification'])
                                        expected=2*z*.001*grid.WAVELENGTHS[ch]/(period*pitch*pitch)
                                        ratio=observed/expected
                                        slot['trials'][axis]=dict(shifts=shifts,observed_px=observed,nominal_px=expected,
                                            ratio=ratio,accepted=bool(.45<=abs(ratio)<=1.8 and min(t['peak'] for t in shifts)>.20))
                                        save()
                                if z:
                                    trials=list(slot['trials'].values())
                                    ok=all(t['accepted'] for t in trials) and len(set(np.sign(t['ratio']) for t in trials))==1
                                    slot['carrier_sign']=int(np.sign(trials[0]['ratio'])) if ok else 0
                                    save()
                finally:
                    if s.moves:
                        s.require_stationary(s.status())
                        s.move(initial)
                    status['stage_end']=s.status()
            if not all(status['slots'][tag]['carrier_sign'] for tag in CHANNELS):
                raise RuntimeError('Carrier displacement does not establish a sign for all channels')
        except BaseException as e:
            status['error']=repr(e)
            raise
        finally:
            try: status['final_standby']=laser.configure('Digital')
            finally: save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--routing',type=Path,required=True)
    args=p.parse_args()
    run(args.out,args.calibration,args.routing)
