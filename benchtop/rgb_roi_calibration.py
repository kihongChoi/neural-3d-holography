"""Register the physical active canvas and evaluate exposure on native ROI photosites."""
import argparse
import json
from pathlib import Path
import hashlib
import subprocess

import cv2
import numpy as np

from .rgb_study_manifest import DEPTHS, VERSION
from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .stage import Stage


def roi_mask(h,shape):
    corners=cv2.perspectiveTransform(geo.CORNERS[None],np.asarray(h,dtype='d'))[0]
    if (corners<0).any() or (corners>=np.array(shape[::-1])).any():
        raise ValueError('Active canvas extends outside the camera')
    mask=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(mask,np.rint(corners*16).astype('i'),1,shift=4)
    return mask.astype(bool),corners.tolist()


def stats(raw,mask):
    output={}
    for ch,indices in [('R',[(0,0)]),('G',[(0,1),(1,0)]),('B',[(1,1)])]:
        v=np.concatenate([raw[y::2,x::2][mask[y::2,x::2]] for y,x in indices])
        output[ch]=dict(n=v.size,mean=float(np.mean(v)/65535),
                        near_clip=float(np.mean(v>=.98*65535)),
                        p998=float(np.quantile(v,.998)/65535))
    return output


def routing(b,flat,mask,out,prefix,lut,mode):
    """Compare each changed HDMI slot to four interleaved flat references."""
    checker=np.uint8(64+128*(np.indices((grid.HEIGHT,grid.WIDTH)).sum(0)%2))
    tiles={}
    chosen=cv2.resize(mask.astype('f'),(128,94),interpolation=cv2.INTER_AREA)>.999
    for index,key in enumerate(['flat','R','G','B','flat','B','R','G','flat','G','B','R','flat']):
        frame=flat.copy()
        if key!='flat':
            frame[grid.ROI][...,'RGB'.index(key)]=checker
        b.show_codes(frame)
        im=grid.full_capture(b)
        grid.png(out/f'{prefix}_routing_{index:02}_{key}.png',lut[cv2.cvtColor(im,cv2.COLOR_BayerRGGB2BGR)])
        values=[]
        for samples in ((im[::2,::2],),(im[::2,1::2],im[1::2,::2]),(im[1::2,1::2],)):
            value=np.mean([cv2.resize(x.astype('f'),(128,94),interpolation=cv2.INTER_AREA)[chosen]
                           for x in samples],axis=0)/65535
            values.append(value)
        tiles.setdefault(key,[]).append(np.stack(values))
    ref=np.mean(tiles['flat'],axis=0)
    noise=np.max([np.mean(np.abs(a-d),axis=1) for i,a in enumerate(tiles['flat']) for d in tiles['flat'][:i]],axis=0)
    response={key:np.mean(np.abs(np.mean(tiles[key],axis=0)-ref),axis=1) for key in 'RGB'}
    delta=np.stack([(ref-np.mean(tiles[key],axis=0)).mean(axis=1) for key in 'RGB'],axis=1)
    limit=np.maximum(noise,1/4095)
    if mode=='RGB':
        accepted=all(response[key][i]>5*limit[i] for i,key in enumerate('RGB'))
    else:
        i='RGB'.index(mode)
        accepted=response[mode][i]>5*limit[i] and all(response[k][i]<=limit[i] for k in 'RGB' if k!=mode)
    return dict(noise_ceiling=limit.tolist(),mae={k:v.tolist() for k,v in response.items()},
                signed_sensor_by_drive=delta.tolist(),accepted=bool(accepted))


def run(out,stationary,scan_exposure=False,added_nd_od=0.,route=False,rgb_timing=(15000,22)):
    out.mkdir(parents=True,exist_ok=False)
    cache=grid.ROOT/'results/step03_geometry/pattern_cache'
    depths=[0.] if stationary else [0.,*DEPTHS]
    patterns={z:geo.cached_patterns(z,cache) for z in depths}
    level=np.arange(65536)/65535
    lut=np.uint8(np.rint(255*np.where(level<=.0031308,12.92*level,1.055*level**(1/2.4)-.055)))
    flat=np.full((1080,1920,3),128,np.uint8)
    report=dict(plan=VERSION,added_nd_od=added_nd_od,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                clip_threshold='native >= 0.98*65535 inside registered active canvas',
                geometries=[],records=[])
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(report,indent=2),encoding='utf8')
        tmp.replace(out/'status.json')
    save()
    with exclusive():
        laser=ReadyBeam()
        try:
            report['standby']=laser.configure('Digital')
            # The additional 0.22 mm covers the verified initial parked position.
            with Stage(serial='45897069',interval=(grid.SCAN_RANGE[0],101.),monitor_only=stationary) as s:
                start=s.initial['position_mm']
                s.require_stationary(s.initial)
                try:
                    with Bench(dict(grid.C,settle_s=.22)) as b:
                        modes=[('RGB',*rgb_timing),('R',175,22),('G',350,6022),('B',350,12022)]
                        if route:
                            b.show_codes(flat)
                            for mode,exp,delay in modes:
                                b.set_timing(exp,delay)
                                dark=grid.full_capture(b)
                                grid.png(out/f'{mode}_dark.png',lut[cv2.cvtColor(dark,cv2.COLOR_BayerRGGB2BGR)])
                        report['external']=laser.configure('Analog')
                        for z in depths:
                            if not stationary and abs(s.status()['position_mm']-(grid.ZERO-z))>.002:
                                s.move(grid.ZERO-z)
                            position=s.status()
                            s.require_stationary(position)
                            actual_z=grid.ZERO-position['position_mm']
                            for mode,exp,delay in modes[:1] if scan_exposure else modes:
                                prefix=f'z{actual_z:.3f}_{mode}'
                                timing=b.set_timing(exp,delay)
                                raw={}
                                for k in ['blank',*geo.FIT,geo.CHECK]:
                                    b.show_codes(patterns[z][k])
                                    raw[k]=grid.full_capture(b)
                                    grid.png(out/f'{prefix}_{k}.png',lut[cv2.cvtColor(raw[k],cv2.COLOR_BayerRGGB2BGR)])
                                reg=geo.analyze(raw,actual_z)
                                report['geometries'].append(dict(mode=mode,stage=position,timing=timing,**reg))
                                save()
                                if not reg['accepted']:
                                    raise RuntimeError(f'{prefix} independent geometry rejected: {reg["scores"]["pooled"]}')
                                mask,corners=roi_mask(reg['models']['pooled'],raw['blank'].shape)
                                report['geometries'][-1]['reference_pattern_roi']={
                                    k:stats(raw[k],mask) for k in ('blank',geo.CHECK)}
                                del raw
                                b.show_codes(flat)
                                for rep in range(2):
                                    im=grid.full_capture(b)
                                    grid.png(out/f'{prefix}_white_{rep}.png',lut[cv2.cvtColor(im,cv2.COLOR_BayerRGGB2BGR)])
                                    row=dict(mode=mode,z=actual_z,rep=rep,timing=timing,corners=corners,roi=stats(im,mask))
                                    report['records'].append(row)
                                    save()
                                    print(prefix,rep,{c:round(x['near_clip'],6) for c,x in row['roi'].items()},flush=True)
                                if scan_exposure:
                                    # Vary both ends of the integration window. A low clip rate
                                    # only qualifies a candidate for a later three-slot routing test.
                                    for offset in (22,5722,11222):
                                        for tail in (0,400,800):
                                            for width in (11000,11600):
                                                setting=b.set_timing(width,offset+tail)
                                                im=grid.full_capture(b)
                                                name=f'scan_{offset+tail}_{width}'
                                                grid.png(out/(name+'.png'),lut[cv2.cvtColor(im,cv2.COLOR_BayerRGGB2BGR)])
                                                report.setdefault('scan',[]).append(dict(name=name,timing=setting,roi=stats(im,mask)))
                                                save()
                                if route:
                                    report.setdefault('routing',{})[mode]=routing(b,flat,mask,out,prefix,lut,mode)
                                    save()
                finally:
                    if s.moves and abs(s.status()['position_mm']-start)>.002:
                        s.require_stationary(s.status())
                        s.move(start)
                    report['stage_end']=s.status()
        except BaseException as exc:
            report['error']=repr(exc)
            raise
        finally:
            try:
                report['final_standby']=laser.configure('Digital')
            finally:
                save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--stationary',action='store_true')
    p.add_argument('--exposure-scan',action='store_true')
    p.add_argument('--added-nd-od',type=float,default=0.,help='Additional filter OD reported by the operator')
    p.add_argument('--routing',action='store_true')
    p.add_argument('--rgb-timing',nargs=2,type=float,default=(15000,22),metavar=('EXPOSURE_US','DELAY_US'))
    a=p.parse_args()
    if a.exposure_scan and not a.stationary:
        p.error('Exposure scan requires a stationary stage')
    if a.routing and (not a.stationary or a.exposure_scan):
        p.error('Routing requires a separate stationary measurement')
    run(a.out,a.stationary,a.exposure_scan,a.added_nd_od,a.routing,a.rgb_timing)
