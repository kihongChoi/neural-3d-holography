"""Bracket the 1.25 mm plane with two zero visits and measure the phase of interference fringes."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench, encode_phase
from .readybeam import ReadyBeam, exclusive
from .rgb_roi_calibration import roi_mask
from .rgb_study_block import CHANNELS, MODES, rectify
from .stage import Stage


def layout():
    h,w=grid.HEIGHT,grid.WIDTH
    y,x=np.indices((h,w),dtype='f')
    wave=np.empty_like(x)
    regions={}
    for row,period in enumerate((8,12)):
        for col,(axis,coord) in enumerate([('x',x),('y',y)]):
            ys=row*(h//2); xs=col*(w//2)
            area=np.s_[ys:ys+h//2,xs:xs+w//2]
            wave[area]=coord[area]*(2*np.pi/period)
            inner=np.s_[ys+64:ys+h//2-64,xs+64:xs+w//2-64]
            regions[f'{axis}{period}']=(inner,np.exp(-1j*wave[inner]))
    return wave,(x+y)%2==1,regions


def decide(visits,wl):
    rows={}
    for key in visits[0]:
        def relative(i,j):
            ratios=[]
            for direction in ('1','-1'):
                a=visits[i][key][direction]; b=visits[j][key][direction]
                ratios.append(np.exp(1j*(a['phase']-b['phase'])))
            return np.angle(ratios[0]*ratios[1])/2
        nominal=-np.pi*wl*.00125/(int(key[1:])*grid.C['slm_pitch_m'])**2
        before=relative(1,0); after=relative(1,2); drift=abs(relative(2,0))
        ratio=(before+after)/(2*nominal)
        mod=min(v[key][d]['modulation'] for v in visits for d in ('1','-1'))
        ok=(mod>=.02 and min(abs(before),abs(after))>max(3*drift,.08)
            and before*after>0 and .45<=abs(ratio)<=1.65)
        rows[key]=dict(observed_before_rad=before,observed_after_rad=after,zero_drift_rad=drift,
                       nominal_positive_rad=nominal,ratio=ratio,min_modulation=mod,accepted=bool(ok))
    signs=[int(np.sign(r['ratio'])) for r in rows.values() if r['accepted']]
    return dict(trials=rows,sign=signs[0] if len(signs)>=3 and len(set(signs))==1 else 0)


def run(out,from_100=False):
    out.mkdir(parents=True,exist_ok=False)
    wave,odd,regions=layout()
    levels=np.arange(65536)/65535
    enc=np.uint8(np.rint(255*np.where(levels<=.0031308,12.92*levels,1.055*levels**(1/2.4)-.055)))
    report=dict(plan='v2.1',added_nd_od=.4,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                visits=[],decision={},gate='per channel >=3/4 agreeing axes/period cases; min modulation .02; both bracketed changes exceed .08 rad and 3x return drift; absolute observed/ideal .45..1.65; ROI near-clip <=.0001')
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(report,indent=2),encoding='utf8')
        tmp.replace(out/'status.json')
    def emit(name,raw):
        grid.png(out/(name+'.png'),enc[cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)])
    save()
    with exclusive():
        laser=ReadyBeam()
        try:
            report['standby']=laser.configure('Digital')
            with Stage(serial='45897069',interval=(grid.ZERO-1.25,grid.ZERO)) as stage:
                report['stage_start']=stage.initial
                save()
                initial=stage.initial['position_mm']
                if from_100:
                    stage.require_stationary(stage.initial)
                    if abs(initial-100.)>.002:raise RuntimeError('Expected operator-reported 100 mm start')
                    report['optical_zero']=stage.move(grid.ZERO)
                    save()
                    initial=report['optical_zero']['position_mm']
                if abs(initial-grid.ZERO)>.002:raise RuntimeError('Start at optical zero')
                try:
                    with Bench(dict(grid.C,settle_s=.22)) as b:
                        flat=np.full((1080,1920,3),128,np.uint8)
                        b.show_codes(flat)
                        report['external']=laser.configure('Analog')
                        for n,z in enumerate((0.,1.25,0.)):
                            visit=dict(z=z,slots={});report['visits'].append(visit)
                            patterns=geo.cached_patterns(z,grid.ROOT/'results/step03_geometry/pattern_cache')
                            for ch,tag in enumerate(CHANNELS):
                                prefix=f'v{n}_{tag}'
                                timing=b.set_timing(*MODES[ch])
                                reg=geo.move_and_align(stage,b,z,patterns,save=lambda images:
                                    [emit(prefix+'_'+k,im) for k,im in images.items()])
                                slot=dict(geometry=reg,timing=timing,fringes={},samples=[])
                                visit['slots'][tag]=slot; save()
                                if not reg['accepted']:raise RuntimeError('ROI registration failed: '+prefix)
                                H=reg['models']['pooled']
                                b.show_codes(flat)
                                ref=grid.full_capture(b);emit(prefix+'_flat',ref)
                                mask,_=roi_mask(H,ref.shape)
                                white=rectify(ref,ch,H)
                                floor=max(100.,float(np.quantile(white,.1)))
                                for polarity in (1,-1):
                                    phasors={k:0j for k in regions}
                                    averages={k:0. for k in regions}
                                    for step in [0,4,2,6,1,5,3,7]:
                                        alpha=step*np.pi/4
                                        phase=np.where(odd,polarity*wave+alpha,0.)
                                        frame=flat.copy()
                                        frame[grid.ROI][...,ch]=encode_phase(phase)
                                        b.show_codes(frame)
                                        raw=grid.full_capture(b)
                                        name=f'{prefix}_{polarity}_{step}'
                                        emit(name,raw)
                                        offsets=[(0,0)] if ch==0 else [(1,1)] if ch==2 else [(0,1),(1,0)]
                                        count=sum(np.count_nonzero((raw[y::2,x::2]>=.98*65535)&mask[y::2,x::2]) for y,x in offsets)
                                        size=sum(np.count_nonzero(mask[y::2,x::2]) for y,x in offsets)
                                        slot['samples'].append(dict(name=name,roi_near_clip=count/size))
                                        if count/size>.0001:raise RuntimeError('Fringe ROI near clipping: '+name)
                                        signal=rectify(raw,ch,H)/np.maximum(white,floor)
                                        for key,(roi,basis) in regions.items():
                                            patch=signal[roi]
                                            averages[key]+=float(patch.mean())/8
                                            projection=basis if polarity==1 else basis.conj()
                                            phasors[key]+=np.mean((patch-patch.mean())*projection)*np.exp(-1j*alpha)/8
                                    for k,value in phasors.items():
                                        slot['fringes'].setdefault(k,{})[str(polarity)]=dict(phase=float(np.angle(value)),
                                            modulation=float(2*abs(value)/max(averages[k],1e-8)))
                                    save()
                                print(prefix,'measured',flush=True)
                finally:
                    if stage.moves and abs(stage.status()['position_mm']-initial)>.002:
                        stage.require_stationary(stage.status());stage.move(initial)
                    report['stage_end']=stage.status()
            for ch,tag in enumerate(CHANNELS):
                report['decision'][tag]=decide([v['slots'][tag]['fringes'] for v in report['visits']],grid.WAVELENGTHS[ch])
        except BaseException as exc:
            report['error']=repr(exc)
            raise
        finally:
            try:report['final_standby']=laser.configure('Digital')
            finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--from-100',action='store_true')
    args=p.parse_args()
    run(args.out,args.from_100)
