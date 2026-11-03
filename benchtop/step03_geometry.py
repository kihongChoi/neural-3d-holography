"""Bench validation for the post-move geometric calibration, with no scene-based fitting."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from . import geometry_registration as geo
from . import step01_dot_grid as bench
from .readybeam import ReadyBeam,exclusive
from .stage import Stage
from .hardware import Bench
ROOT=bench.ROOT/'results'/'step03_geometry'
def save_result(folder,records):
    tmp=folder/'registration.tmp'
    tmp.write_text(json.dumps(records,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(folder/'registration.json')
def run(depths,out):
    if not depths or len(set(depths))!=len(depths) or any(not 0<=z<=10 for z in depths):raise ValueError('Unique depths in 0..10 mm required')
    out.mkdir(parents=True,exist_ok=False);root=out/'camera_png';root.mkdir()
    t=time.perf_counter()
    banks={z:geo.cached_patterns(z,ROOT/'pattern_cache') for z in depths}
    print('Preparation, outside motion loop:',time.perf_counter()-t,flush=True)
    g=np.arange(65536,dtype='d')/65535
    lut=np.rint(255*np.where(g<=.0031308,12.92*g,1.055*g**(1/2.4)-.055)).astype('uint8')
    rec=[]
    with exclusive():
        laser=ReadyBeam();laser.configure('Digital')
        with Stage(serial='45897069',interval=bench.SCAN_RANGE) as stage:
            start=stage.initial['position_mm']
            if abs(start-bench.ZERO)>.002 or stage.real(stage.limits()[2])!=124 or stage.dll.ISC_GetSoftLimitMode(stage.serial)!=0:raise RuntimeError('Stage preflight failed')
            try:
                with Bench(dict(bench.C,exposure_us=15000.,gain_db=0.,trigger_delay_us=22.,settle_s=.22)) as b:
                    b.show_codes(banks[depths[0]]['blank'])
                    tm=b.set_timing(15000.,22.)
                    if abs(tm['exposure_us']-15000)>10 or (b.camera.Width.GetValue(),b.camera.Height.GetValue(),b.camera.OffsetX.GetValue(),b.camera.OffsetY.GetValue())!=(4096,3000,0,0):raise RuntimeError('Camera preflight failed')
                    try:
                        print('Light',laser.configure('Analog')['mode'],'timing',tm,flush=True)
                        for rep,path in enumerate([depths,depths[::-1]]):
                            for z in path:
                                def save(raw):
                                    for name,a in raw.items():
                                        bench.png(root/f'r{rep}_z{z:g}_{name}.png',lut[cv2.cvtColor(a,cv2.COLOR_BayerRGGB2BGR)])
                                item=geo.move_and_align(stage,b,z,banks[z],reverse=bool(rep),save=save)
                                item['repeat']=rep;rec.append(item);save_result(out,rec)
                                print(z,rep,item['seconds'],item['scores'],flush=True)
                    finally:print('Standby',laser.configure('Digital'),flush=True)
            finally:
                if stage.moves:stage.require_stationary(stage.status());stage.move(start)
                print('Final stage',stage.status(),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--depths',nargs='+',type=float,default=[0.,5.,10.]);ap.add_argument('--run',default=time.strftime('run_%Y%m%d_%H%M%S'))
    a=ap.parse_args();run(a.depths,ROOT/a.run)
