"""Scene-independent planar registration on the 1728x960 SLM canvas."""
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from . import step01_dot_grid as bench
from .optics import ASM, crop_center

FIT=('circle','square','diamond')
CHECK='cross'
SHAPES=(*FIT,CHECK)
W,H=bench.WIDTH,bench.HEIGHT
CORNERS=np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])

def layout(kind):
    rng=np.random.default_rng(143 if kind==CHECK else 725)
    p=np.float32([(x,y) for y in np.linspace(88,H-89,5) for x in np.linspace(92,W-93,7)])
    return np.rint(p+rng.uniform([-40,-36],[40,36],p.shape)).astype('float32')

def target(kind):
    im=np.zeros((H,W),np.float32)
    for x,y in layout(kind).astype(int):
        if kind=='circle':cv2.circle(im,(x,y),17,1,-1)
        elif kind=='square':cv2.rectangle(im,(x-15,y-15),(x+15,y+15),1,-1)
        elif kind=='diamond':cv2.fillConvexPoly(im,np.int32([[x,y-21],[x+21,y],[x,y+21],[x-21,y]]),1)
        else:
            cv2.rectangle(im,(x-7,y-22),(x+7,y+22),1,-1);cv2.rectangle(im,(x-22,y-7),(x+22,y+7),1,-1)
    return cv2.GaussianBlur(im,(0,0),.9)

def pack(rgb):
    f=np.full((1080,1920,3),128,np.uint8)
    f[bench.ROI]=rgb
    return f

def cached_patterns(depth,cache):
    """Compute each phase-only CGH once per depth and load it before moving."""
    cache.mkdir(parents=True,exist_ok=True)
    paths={k:cache/f'{k}_{float(depth):.6f}mm.png' for k in SHAPES}
    y,x=np.indices((H,W));checker=np.where((x+y)%2==0,1.,-1.).astype('f')
    pedestal=.30
    if not all(p.exists() for p in paths.values()):
        fields={k:torch.from_numpy(target(k)).cuda() for k in SHAPES}
        chk=torch.as_tensor(checker,device='cuda')
        phases={k:[] for k in SHAPES}
        for wl in bench.WAVELENGTHS:
            k=ASM(dict(bench.C,slm_shape=[H,W],wavelength_m=wl,distance_m=depth*.001),'cuda')
            for name,mark in fields.items():
                spectrum=torch.fft.fft2(F.pad(mark*(.84-pedestal),k.padding))
                field=pedestal+crop_center(torch.fft.ifft2(spectrum*k.kernel.conj()),(H,W))
                phase=field.angle()+chk*torch.acos(field.abs().clamp(0,1))
                phases[name].append(bench.encode_phase(phase.cpu().numpy()))
        for name,p in paths.items():
            if not p.exists():bench.png(p,np.stack(phases[name][::-1],axis=-1))
    frames={k:pack(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)) for k,p in paths.items()}
    plane=bench.encode_phase(checker*np.arccos(pedestal))
    frames['blank']=pack(plane[...,None])
    return frames

def project(h,p):
    return cv2.perspectiveTransform(np.float32(p)[None],h)[0]

def bayer_half(raw):
    # One value per RGGB cell; linear, full field and identical across patterns.
    return cv2.resize(raw,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA).astype('f')/65535.

def contrast(raw,blank):
    diff=bayer_half(raw)-blank
    # Suppress slow changes in illumination before finding marker centers.
    return np.abs(cv2.GaussianBlur(diff,(0,0),2)-cv2.GaussianBlur(diff,(0,0),24))

def seed_geometry(signal,t):
    # No natural-image priors. Search for the coded constellation on a small sensor thumbnail.
    shrink=4;small=cv2.resize(signal,None,fx=1/shrink,fy=1/shrink,interpolation=cv2.INTER_AREA)
    best=(-np.inf,None);runner=[]
    for sx,sy in [(1,1),(-1,1),(1,-1),(-1,-1)]:
        T=np.ascontiguousarray(t[::sy,::sx]);direction=(-np.inf,None)
        for scale in np.arange(.84,1.025,.015):
            stamp=cv2.resize(T,None,fx=scale/shrink,fy=scale/shrink,interpolation=cv2.INTER_AREA)
            scores=cv2.matchTemplate(small,stamp,cv2.TM_CCOEFF_NORMED);_,v,_,(x,y)=cv2.minMaxLoc(scores)
            if v>direction[0]:
                h=np.array([[sx*scale,0,x*shrink+(W-1)*scale*(sx<0)],[0,sy*scale,y*shrink+(H-1)*scale*(sy<0)],[0,0,1.]],dtype='d')
                direction=(v,h)
        runner.append(direction[0])
        if direction[0]>best[0]:best=direction
    if best[0]<.17 or best[0]-sorted(runner)[-2]<.035:raise RuntimeError(f'Ambiguous constellation orientation: {runner}')
    return best[1]

def detect(signal,kind,seed):
    t=target(kind);src=layout(kind);dst=np.full_like(src,np.nan)
    rect=cv2.warpPerspective(signal,seed,(W,H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    for i,(x,y) in enumerate(src.astype(int)):
        # The differential optical response is strongest on the marker body or rim.
        ref=t[y-29:y+30,x-29:x+30].copy()
        area=rect[y-48:y+49,x-48:x+49]
        if area.shape!=(97,97):continue
        score=cv2.matchTemplate(area,ref,cv2.TM_CCOEFF_NORMED)
        iy,ix=np.unravel_index(score.argmax(),score.shape)
        if score[iy,ix]<.35 or not (0<ix<38 and 0<iy<38):continue
        def sub(v):
            a,b,c=v;return float(np.clip(.5*(a-c)/min(-1e-7,a+c-2*b),-.6,.6))
        loc=[x+ix-19+sub(score[iy,ix-1:ix+2]),y+iy-19+sub(score[iy-1:iy+2,ix])]
        dst[i]=project(seed,[loc])[0]
    return dst

def solve(src,dst):
    ok=np.isfinite(dst[:,0])
    if ok.sum()<12:raise RuntimeError(f'Only {ok.sum()} usable marks')
    h,inliers=cv2.findHomography(src[ok],dst[ok],cv2.RANSAC,3.0)
    if h is None:raise RuntimeError('Singular geometry')
    keep=ok.copy();keep[ok]=inliers.ravel().astype(bool)
    if keep.sum()<12 or np.ptp(src[keep,0])<W*.65 or np.ptp(src[keep,1])<H*.65:raise RuntimeError('Insufficient inlier field coverage')
    h,_=cv2.findHomography(src[keep],dst[keep],0)
    return h,keep

def errors(h,src,dst):
    valid=np.isfinite(dst[:,0]);v=src[valid]
    d=(project(h,v)-dst[valid])*2 # full-resolution sensor pixels
    r=np.linalg.norm(d,axis=1)
    return dict(n=len(r),rms=float(np.sqrt(np.mean(r*r))) if len(r) else None,p95=float(np.percentile(r,95)) if len(r) else None,
                quadrants=len(np.unique((v[:,0]>W/2).astype(int)+2*(v[:,1]>H/2))))

def move_and_align(stage,b,depth,frames,reverse=False,save=None):
    """After a settled move return a checked mapping. Always restore the incoming SLM frame."""
    dest=round(bench.ZERO-depth,6)
    if abs(stage.status()['position_mm']-dest)>.002:stage.move(dest)
    state=stage.status()
    if state['moving'] or abs(state['position_mm']-dest)>.002:raise RuntimeError('Stage not settled')
    previous=b.last_encoded
    if previous is None:raise ValueError('Present an application frame before calibration')
    raw={};tim=time.perf_counter()
    try:
        for k in ['blank',*(FIT[::-1] if reverse else FIT),CHECK]:
            b.show_codes(frames[k]);raw[k]=bench.full_capture(b)
    finally:b.show_codes(previous)
    exposed=time.perf_counter()
    try:
        result=analyze(raw,depth)
        result.update(acquire_restore_s=exposed-tim,seconds=time.perf_counter()-tim)
    finally:
        if save:save(raw)
    return result

def analyze(raw,depth):
    blank=bayer_half(raw['blank']);marks={};hs={};keeps={}
    sig=contrast(raw['circle'],blank);seed=seed_geometry(sig,target('circle'))
    for k in FIT:
        a=detect(sig if k=='circle' else contrast(raw[k],blank),k,seed)
        hs[k],keeps[k]=solve(layout(k),a);marks[k]=a
        if k=='circle':seed=hs[k]
    # Average the actual measured coordinates, not the homography coefficients.
    combined=np.stack([np.where(keeps[k][:,None],marks[k],np.nan) for k in FIT])
    n=np.isfinite(combined[:,:,0]).sum(0)
    mean=np.full_like(marks['circle'],np.nan);ok=n>=2
    mean[ok]=np.nansum(combined[:,ok],axis=0)/n[ok,None]
    h,keep=solve(layout('circle'),mean);hs['pooled']=h
    # Identical validation measurements for all candidates. Seed from circle only.
    test=detect(contrast(raw[CHECK],blank),CHECK,seed)
    scores={k:errors(v,layout(CHECK),test) for k,v in hs.items()};q=scores['pooled']
    accepted=q['n']>=20 and q['quadrants']==4 and q['rms']<=3 and q['p95']<=6
    result=dict(depth_mm=depth,accepted=bool(accepted),scores=scores,
        pooled_sites=int(keep.sum()),pixel_coordinates='original unmirrored 1728x960 SLM canvas -> full sensor, index pixel centers',
        models={k:(np.array([[2,0,.5],[0,2,.5],[0,0,1]])@v).tolist() for k,v in hs.items()},
        validation_src=layout(CHECK)[np.isfinite(test[:,0])].tolist(),validation_full_sensor=(2*test[np.isfinite(test[:,0])]+.5).tolist())
    return result
