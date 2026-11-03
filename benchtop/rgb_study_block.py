"""Acquire a matched RGB block with current geometry, sign probes and references."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .rgb_study_manifest import VERSION, DEPTHS, TRAIN_DEPTHS, HELD_OUT, PATTERNS
from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench, encode_phase
from .optics import ASM
from .readybeam import ReadyBeam, exclusive
from .rgb_roi_calibration import roi_mask
from .stage import Stage

CHANNELS='RGB'
MODES=[(175,22),(350,6022),(350,12022)]


def publish_status(folder, report):
    temp=folder/'status.tmp'
    temp.write_text(json.dumps(report,indent=2),encoding='utf8')
    for attempt in range(21):
        try:
            temp.replace(folder/'status.json')
            return
        except PermissionError:
            if attempt==20: raise
            time.sleep(.1)


def sensor_channel(im,ch):
    if ch==1:
        return .5*(im[::2,1::2].astype('f')+im[1::2,::2])
    offset=ch//2
    return im[offset::2,offset::2].astype('f')


def rectify(im,ch,H):
    half=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])@np.array(H)
    return cv2.warpPerspective(sensor_channel(im,ch),half,(grid.WIDTH,grid.HEIGHT),
                               flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)


def probe(b,z,ch,H,flat,emit,scale_probe=False):
    h,w=grid.HEIGHT,grid.WIDTH
    # Keep lambda*z/(feature pitch)^2 comparable to the validated 10 mm probe.
    spacing=max(2,round(8*np.sqrt(z/10))) if scale_probe else 8
    phase=cv2.resize(np.random.default_rng(401).normal(0,.75,(h//spacing,w//spacing)).astype('f'),
                     (w,h),interpolation=cv2.INTER_CUBIC)
    model=ASM(dict(grid.C,slm_shape=[h,w],wavelength_m=grid.WAVELENGTHS[ch],distance_m=z/1000),'cuda')
    measured=[]
    predicted=[]
    for sign in [1,-1]:
        frame=np.full((1080,1920,3),128,np.uint8)
        frame[grid.ROI][...,ch]=encode_phase(sign*phase)
        b.show_codes(frame)
        raw=grid.full_capture(b)
        emit(f'probe_{CHANNELS[ch]}_{sign}',raw)
        measured.append(rectify(raw,ch,H))
        predicted.append(model(torch.from_numpy(sign*phase).cuda()).abs().square().cpu().numpy())
    valid=(flat>np.quantile(flat,.10))&(flat<64000)
    valid[:32]=False; valid[-32:]=False; valid[:,:32]=False; valid[:,-32:]=False
    actual=(measured[0]-measured[1])/np.maximum(flat,128)
    ideal=predicted[0]-predicted[1]
    return float(np.corrcoef(actual[valid][::4],ideal[valid][::4])[0,1])


def grating_probe(b,z,ch,H,flat,emit):
    """Measure the signed Fourier component at two injected spatial frequencies per axis."""
    h,w=flat.shape
    y,x=np.indices(flat.shape,dtype='f')
    model=ASM(dict(grid.C,slm_shape=[h,w],wavelength_m=grid.WAVELENGTHS[ch],distance_m=z/1000),'cuda')
    keep=(flat>np.quantile(flat,.2))&(flat<64000)
    keep[:48]=False; keep[-48:]=False; keep[:,:48]=False; keep[:,-48:]=False
    trials=[]
    for axis,coord in [('x',x),('y',y)]:
        for period in ((16,32) if z<2 else (32,48)):
            angle=2*np.pi*coord/period
            basis=np.exp(-1j*angle[keep])
            screen=.8*np.cos(angle)
            measured=[]; predicted=[]
            for polarity in (1,-1):
                frame=np.full((1080,1920,3),128,np.uint8)
                phase=polarity*screen
                frame[grid.ROI][...,ch]=encode_phase(phase)
                b.show_codes(frame)
                raw=grid.full_capture(b)
                emit(f'grating_{CHANNELS[ch]}_{axis}{period}_{polarity}',raw)
                measured.append(rectify(raw,ch,H))
                predicted.append(model(torch.from_numpy(phase).cuda()).abs().square().cpu().numpy())
            def component(diff):
                d=diff[keep]
                return 2*np.mean((d-d.mean())*basis)
            a=component((measured[0]-measured[1])/np.maximum(flat,128))
            p=component(predicted[0]-predicted[1])
            alignment=float(np.real(a*np.conj(p))/max(abs(a)*abs(p),1e-10))
            trials.append(dict(axis=axis,period=period,measured_amplitude=float(abs(a)),
                               predicted_amplitude=float(abs(p)),alignment=alignment))
    valid=[v['alignment'] for v in trials if v['measured_amplitude']>=.01 and abs(v['alignment'])>=.5]
    consistent=len(valid)>=3 and len(set(np.sign(valid)))==1
    return trials,float(np.median(valid)) if consistent else 0.


def acquire(patterns,out,z,start,stop,cal,route,only_calibrate,sign_runs,grating=False,scale_probe=False):
    with (route/'status.json').open() as f: routing=json.load(f)
    with (cal/'status.json').open() as f: calibration=json.load(f)
    if (not all(routing.get('routing',{}).get(c,{}).get('accepted') for c in (*CHANNELS,'RGB'))
            or calibration['plan']!=VERSION
            or {(round(g['depth_mm'],2),g['mode']) for g in calibration['geometries']}
               != {(d,m) for d in (0.,*DEPTHS) for m in ('RGB',*CHANNELS)} or not all(x.get('accepted') for x in calibration['geometries'])
            or any('error' in run or run.get('final_standby',{}).get('mode')!='Digital' for run in (routing,calibration))
            or calibration['added_nd_od']!=.4 or routing['added_nd_od']!=.4):
        raise ValueError('Require valid ND 0.4 calibration and four verified HDMI modes')
    if not 0<=start<stop<=PATTERNS or z not in DEPTHS: raise ValueError('Unknown block')
    signs={}
    if not only_calibrate:
        for source in sign_runs:
            check=json.loads((source/'status.json').read_text())
            if check.get('error') or check.get('final_standby',{}).get('mode')!='Digital' or check['routing']!=str(route) or check.get('calibration')!=str(cal):
                raise ValueError(f'Invalid sign calibration: {source}')
            values=[]
            for tag in CHANNELS:
                slot=check['slots'][tag]
                if 'carrier_sign' in slot:
                    if slot['carrier_sign'] not in (-1,1): raise ValueError(f'Unresolved carrier sign in {source}')
                    values.append(slot['carrier_sign'])
                else:
                    corr=slot['positive_sign_correlation']
                    if not np.isfinite(corr) or abs(corr)<.15: raise ValueError(f'Unresolved correlation in {source}')
                    values.append(int(np.sign(corr)))
            signs[check['z']]=values
        if set(signs)!=set(TRAIN_DEPTHS) or len({tuple(v) for v in signs.values()})!=1:
            raise ValueError('Require consistent signs at all training depths')
    order=np.random.default_rng(20261103+start).permutation(np.arange(start,stop)).tolist()
    entries={}
    if not only_calibrate:
        if json.loads((patterns/'settings.json').read_text())['plan']!=VERSION:
            raise ValueError('Pattern set does not match study version')
        for i in order:
            p=patterns/f'{i:04}.png'
            m=json.loads(p.with_suffix('.json').read_text())
            if m['png_sha256']!=hashlib.sha256(p.read_bytes()).hexdigest():raise IOError(p)
            entries[i]=m
    out.mkdir(parents=True,exist_ok=False)
    level=np.arange(65536)/65535
    enc=np.uint8(np.rint(255*np.where(level<=.0031308,12.92*level,1.055*level**(1/2.4)-.055)))
    def emit(name,raw):
        grid.png(out/(name+'.png'),enc[cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)])
    flat=np.full((1080,1920,3),128,np.uint8)
    report=dict(plan=VERSION,added_nd_od=.4,z=z,order=order,slots={},calibration=str(cal),
                routing=str(route),propagation_signs=signs,records=[],sealed=z==HELD_OUT,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        publish_status(out,report)
    frames=geo.cached_patterns(z,grid.ROOT/'results/step03_geometry/pattern_cache')
    save()
    with exclusive():
        laser=ReadyBeam()
        try:
            report['standby']=laser.configure('Digital')
            with Stage(serial='45897069',interval=(grid.SCAN_RANGE[0],101.)) as stage:
                initial=stage.initial['position_mm']
                try:
                    with Bench(dict(grid.C,settle_s=.22)) as b:
                        for ch,(exp,delay) in enumerate(MODES):
                            b.set_timing(exp,delay)
                            emit('dark_'+CHANNELS[ch],grid.full_capture(b))
                        report['external']=laser.configure('Analog')
                        stage.move(grid.ZERO-z)
                        b.show_codes(flat)
                        for ch in [(k+start//32)%3 for k in range(3)]:
                            tag=CHANNELS[ch]
                            timing=b.set_timing(*MODES[ch])
                            reg=geo.move_and_align(stage,b,z,frames,save=lambda images:
                                [emit(f'geometry_{tag}_{k}',im) for k,im in images.items()])
                            report['slots'][tag]=dict(timing=timing,geometry=reg)
                            save()
                            if not reg['accepted']:raise RuntimeError(f'{tag} geometry gate')
                            H=reg['models']['pooled']
                            b.show_codes(flat)
                            white=grid.full_capture(b)
                            emit('white_'+tag,white)
                            ref=rectify(white,ch,H)
                            if only_calibrate:
                                if grating:
                                    trials,corr=grating_probe(b,z,ch,H,ref,emit)
                                    report['slots'][tag]['grating_trials']=trials
                                else:
                                    report['slots'][tag]['probe_mesh_px']=max(2,round(8*np.sqrt(z/10))) if scale_probe else 8
                                    corr=probe(b,z,ch,H,ref,emit,scale_probe)
                                report['slots'][tag]['positive_sign_correlation']=corr
                                save()
                                report['slots'][tag]['sign_accepted']=bool(np.isfinite(corr) and abs(corr)>=.15)
                                continue
                            mask,_=roi_mask(H,white.shape)
                            for i in order:
                                tic=time.perf_counter()
                                grid.present_phase(b,patterns/f'{i:04}.png')
                                raw=grid.full_capture(b)
                                photosites=([raw[::2,::2][mask[::2,::2]]] if ch==0 else
                                           [raw[1::2,1::2][mask[1::2,1::2]]] if ch==2 else
                                           [raw[::2,1::2][mask[::2,1::2]],raw[1::2,::2][mask[1::2,::2]]])
                                clip=sum(np.count_nonzero(x>=.98*65535) for x in photosites)/sum(x.size for x in photosites)
                                capname=f'{i:04}_{tag}'
                                emit(capname,raw)
                                qc={}
                                if i==order[0]:
                                    pngfile=out/(capname+'.png')
                                    png=cv2.imread(str(pngfile))
                                    if png is None or png.shape!=(*raw.shape,3): raise IOError(pngfile)
                                    signal=png[...,2-ch].astype('f')/255
                                    signal=np.where(signal<=.04045,signal/12.92,((signal+.055)/1.055)**2.4)
                                    sites=[(0,0)] if ch==0 else [(1,1)] if ch==2 else [(0,1),(1,0)]
                                    diffs=[]
                                    for y,x in sites:
                                        valid=mask[y::2,x::2]
                                        diffs.append(np.abs(signal[y::2,x::2][valid]-raw[y::2,x::2][valid]/65535))
                                    p99=float(np.quantile(np.concatenate(diffs),.99))
                                    qc=dict(png_linear_abs_error_p99=p99,png_bytes=pngfile.stat().st_size)
                                    if p99>.006:raise RuntimeError(f'{pngfile}: radiometry/encoding mismatch {p99}')
                                report['records'].append(dict(id=i,scene=entries[i]['recipe']['scene'],channel=tag,
                                    rgb_sha256=entries[i]['rgb_sha256'],near_clip=clip,accepted=clip<=.0001,seconds=time.perf_counter()-tic,**qc))
                                save()
                                if clip>.0001: raise RuntimeError(f'{tag} {i}: measured ROI near clipping {clip:.6f}')
                            b.show_codes(flat)
                            end=grid.full_capture(b)
                            emit('white_end_'+tag,end)
                            fraction=float(rectify(end,ch,H).mean()/ref.mean()-1)
                            report['slots'][tag]['white_drift_fraction']=fraction
                            save()
                            if abs(fraction)>.03:raise RuntimeError(f'{tag}: white drift exceeds 3%')
                        if only_calibrate:
                            unresolved=[tag for tag in CHANNELS if not report['slots'][tag]['sign_accepted']]
                            if unresolved:raise RuntimeError(f'propagation sign unresolved: {unresolved}')
                finally:
                    if stage.moves:
                        stage.require_stationary(stage.status())
                        stage.move(initial)
                    report['stage_end']=stage.status()
        except BaseException as e:
            report['error']=repr(e)
            raise
        finally:
            try: report['final_standby']=laser.configure('Digital')
            finally: save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('patterns',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('z',type=float)
    p.add_argument('start',type=int)
    p.add_argument('stop',type=int)
    p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--routing',required=True,type=Path)
    p.add_argument('--only-calibrate',action='store_true')
    p.add_argument('--sign-runs',type=Path,nargs='*',default=[])
    p.add_argument('--grating-probe',action='store_true')
    p.add_argument('--scale-probe',action='store_true')
    a=p.parse_args()
    acquire(a.patterns,a.out,a.z,a.start,a.stop,a.calibration,a.routing,a.only_calibrate,a.sign_runs,a.grating_probe,a.scale_probe)
