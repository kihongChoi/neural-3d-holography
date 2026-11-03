"""Use the geometry-only registration hook with different natural scenes at a stage stop."""
import json,time,argparse
from pathlib import Path
import cv2,numpy as np
from . import step01_dot_grid as bench
from . import step02_div2k as div
from . import geometry_registration as geo
from .step03_geometry import ROOT,save_result
from .step02_alignment import landmarks,PTS,transform,bootstrap
from .hardware import Bench
from .readybeam import ReadyBeam,exclusive
from .stage import Stage
ALT=ROOT/'div2k_validation'
NEW=[(0.,804),(5.,805),(10.,806)]
def design():
    disp=Bench(dict(bench.C,settle_s=.15))
    try:disp.connect_display();div.design(disp,NEW,ALT)
    finally:disp.close()
def run():
    out=ALT/'camera_png';out.mkdir(exist_ok=False)
    banks={z:geo.cached_patterns(z,ROOT/'pattern_cache') for z,_ in NEW};data=[]
    v=np.arange(65536,dtype='d')/65535
    lut=np.rint(255*np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)).astype('uint8')
    old=bench.ROOT/'results'/'step02_div2k_3planes'
    (_,flip,seed),_=bootstrap(cv2.imread(str(old/'targets'/'DIV2K_0802_z05mm.png'),0),cv2.imread(str(old/'camera_png'/'02_z05.000mm.png'),0))
    assert flip
    with exclusive():
        laser=ReadyBeam();laser.configure('Digital')
        with Stage(serial='45897069',interval=bench.SCAN_RANGE) as s:
            start=s.initial['position_mm']
            if abs(start-bench.ZERO)>.002 or s.real(s.limits()[2])!=124 or s.dll.ISC_GetSoftLimitMode(s.serial)!=0:raise RuntimeError('Stage preflight')
            try:
                with Bench(dict(bench.C,exposure_us=15000.,gain_db=0.,trigger_delay_us=22.,settle_s=.22)) as b:
                    b.show_codes(banks[0.]['blank'])
                    tm=b.set_timing(15000.,22.)
                    if abs(tm['exposure_us']-15000)>10 or (b.camera.Width.GetValue(),b.camera.Height.GetValue(),b.camera.OffsetX.GetValue(),b.camera.OffsetY.GetValue())!=(4096,3000,0,0):raise RuntimeError('Camera configuration mismatch')
                    try:
                        print('Laser',laser.configure('Analog')['mode'],tm,flush=True)
                        for (z,old_id),(_,new_id) in zip(div.PAIRS,NEW):
                            reg=geo.move_and_align(s,b,z,banks[z])
                            if not reg['accepted']:raise RuntimeError(f'Independent cross check failed: {reg["scores"]["pooled"]}')
                            reg['scenes']=[]
                            for folder,n in [(div.OUT,old_id),(ALT,new_id)]:
                                name=div.stem(z,n);bench.present_phase(b,folder/'phases'/(name+'.png'))
                                raw=bench.full_capture(b);rgb=lut[cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)]
                                bench.png(out/(name+'.png'),rgb)
                                t=cv2.imread(str(folder/'targets'/(name+'.png')),0)
                                # Previous-session scene registration supplies only a common search window.
                                # No H is fitted to either validation image here.
                                dst=landmarks(cv2.flip(t,1),cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),seed)
                                src=PTS.copy();src[:,0]=geo.W-1-src[:,0]
                                valid=np.isfinite(dst[:,0]);q={}
                                for key,h in reg['models'].items():
                                    if valid.any():
                                        err=np.linalg.norm(transform(np.array(h),src[valid])-dst[valid],axis=1)
                                        q[key]=dict(rms=float(np.sqrt(np.mean(err**2))),p95=float(np.percentile(err,95)))
                                reg['scenes'].append(dict(id=n,n=int(valid.sum()),quadrants=len(np.unique((src[valid,0]>geo.W/2).astype(int)+2*(src[valid,1]>geo.H/2))),scores=q,src=src[valid].tolist(),dst=dst[valid].tolist()))
                                print(z,n,valid.sum(),q.get('pooled'),flush=True)
                            data.append(reg);save_result(ALT,data)
                    finally:print('Standby',laser.configure('Digital'),flush=True)
            finally:
                if s.moves:s.require_stationary(s.status());s.move(start)
                print('Final stage',s.status(),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['design','capture']);a=p.parse_args();design() if a.step=='design' else run()
