"""Compare 0159 iterations and loss choices optically on its 3 mm design plane."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench
from .paper_model import image_metrics
from .readybeam import ReadyBeam,exclusive
from .rgb_roi_calibration import roi_mask,stats
from .rgb_study_block import CHANNELS,MODES,rectify
from .stage import Stage
from .targets import prepare_image


def compare(a):
    if json.loads((a.campaign/'status.json').read_text())['state']!='stopped_between_blocks':
        raise RuntimeError('Wait for bulk capture to reach the safe block boundary')
    fit=json.loads((a.fit/'quality.json').read_text())
    fixed=json.loads((a.fixed/'quality.json').read_text())
    meta=json.loads(a.original.with_suffix('.json').read_text())
    if (fit['recipe']!=meta['recipe'] or fixed['recipe']!=fit['recipe']
            or fit['checks']!=fixed['checks'] or not fixed['fixed_white_objective']
            or not fit.get('summary') or not fixed.get('summary')):
        raise ValueError('Mismatched convergence runs')
    candidates=dict(original=a.original,fit=a.fit/'drive.png',fixed=a.fixed/'drive.png')
    hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in candidates.items()}
    assert hashes['original']==meta['png_sha256']
    amp,source=prepare_image(meta['target']['source_path'],(grid.HEIGHT,grid.WIDTH))
    assert source['source_sha256']==meta['target']['source_sha256']
    z=meta['recipe']['design_mm']
    frames=geo.cached_patterns(z,grid.ROOT/'results/step03_geometry/pattern_cache')
    a.out.mkdir(parents=True,exist_ok=False)
    levels=np.arange(65536)/65535
    lut=np.uint8(np.rint(255*np.where(levels<=.0031308,12.92*levels,1.055*levels**(1/2.4)-.055)))
    report=dict(recipe=meta['recipe'],hashes=hashes,slots={},records=[])
    def save():
        temp=a.out/'status.tmp'
        temp.write_text(json.dumps(report,indent=2))
        temp.replace(a.out/'status.json')
    def emit(name,raw):
        grid.png(a.out/(name+'.png'),lut[cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)])
    save()
    with exclusive():
        laser=ReadyBeam()
        try:
            report['standby']=laser.configure('Digital')
            with Stage(serial='45897069',interval=(grid.SCAN_RANGE[0],101.)) as stage:
                initial=stage.initial['position_mm']
                try:
                    with Bench(dict(grid.C,settle_s=.22),display_wait_s=20) as b:
                        dark=[]
                        for ch,mode in enumerate(MODES):
                            b.set_timing(*mode);raw=grid.full_capture(b)
                            emit('dark_'+CHANNELS[ch],raw);dark.append(raw)
                        report['external']=laser.configure('Analog')
                        b.show_codes(np.full((1080,1920,3),128,np.uint8))
                        for ch,tag in enumerate(CHANNELS):
                            b.set_timing(*MODES[ch])
                            reg=geo.move_and_align(stage,b,z,frames,save=lambda imgs:
                                [emit('reg_'+tag+'_'+k,raw) for k,raw in imgs.items()])
                            report['slots'][tag]=dict(geometry=reg)
                            save()
                            if not reg['accepted']:raise RuntimeError('Rejected geometry '+tag)
                            H=reg['models']['pooled']
                            b.show_codes(np.full((1080,1920,3),128,np.uint8))
                            raw=grid.full_capture(b);emit('white_'+tag,raw)
                            black=rectify(dark[ch],ch,H)
                            white=float((rectify(raw,ch,H)-black).mean())
                            if white<=0:raise RuntimeError('Invalid white')
                            mask,_=roi_mask(H,raw.shape)
                            keys=list(candidates)
                            for rep in range(3):
                                for k in np.roll(keys,rep+ch):
                                    grid.present_phase(b,candidates[k])
                                    raw=grid.full_capture(b);emit(f'{tag}_{rep}_{k}',raw)
                                    clip=stats(raw,mask)[tag]['near_clip']
                                    measured=torch.from_numpy(np.sqrt(np.maximum((rectify(raw,ch,H)-black)/white,0)))
                                    target=torch.from_numpy(amp[...,ch].copy())
                                    metrics=image_metrics(measured,target,fit_brightness=False)
                                    report['records'].append(dict(channel=tag,candidate=k,repeat=rep,near_clip=clip,**metrics))
                                    save()
                                    if clip>.0001:raise RuntimeError(f'ROI near clipping {tag} {k}')
                            b.show_codes(np.full((1080,1920,3),128,np.uint8))
                            raw=grid.full_capture(b);emit('white_end_'+tag,raw)
                            drift=float((rectify(raw,ch,H)-black).mean()/white-1)
                            report['slots'][tag]['white_drift']=drift;save()
                            if abs(drift)>.03:raise RuntimeError('White drift '+tag)
                finally:
                    if stage.moves:
                        stage.require_stationary(stage.status());stage.move(initial)
                    report['stage_end']=stage.status()
            report['summary']={}
            for k in candidates:
                repeats=[]
                for rep in range(3):
                    rows=[r for r in report['records'] if r['candidate']==k and r['repeat']==rep]
                    if len(rows)!=3:raise ValueError('Missing RGB repeat')
                    repeats.append({metric:float(np.mean([r[metric] for r in rows])) for metric in
                                   ('fixed_white_mse','fixed_clipped_srgb_psnr','fixed_clipped_srgb_ssim')})
                report['summary'][k]=dict(repeats=repeats,
                    mean={m:float(np.mean([r[m] for r in repeats])) for m in repeats[0]})
        except BaseException as e:
            report['error']=repr(e)
            raise
        finally:
            try:report['final_standby']=laser.configure('Digital')
            finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('original','fit','fixed','campaign','out'):p.add_argument(field,type=Path)
    compare(p.parse_args())
