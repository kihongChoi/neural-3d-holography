"""Controlled 100 vs 1000 optimizer updates at fixed bench depths."""
import argparse
import json
import numpy as np
import cv2
import torch
from . import step01_dot_grid as bench
from . import step02_div2k as div
from .targets import srgb_to_linear
from .optics import ASM

OUT=bench.ROOT/'results'/'step04_div2k_1000'
PAIRS=[(5.,802),(10.,803)]
# Alternate which drive is shown first at each depth. Two independent exposures per drive.
ORDERS=[[100,1000,1000,100],[1000,100,100,1000]]
def design():
    display=bench.Bench(dict(bench.C,settle_s=.15))
    try:
        display.connect_display()
        div.design(display,PAIRS,OUT,steps=1000)
    finally:display.close()

def inputs():
    for (z,n),order in zip(PAIRS,ORDERS):
        name=div.stem(z,n)+'.png'
        assert np.array_equal(cv2.imread(str(div.OUT/'targets'/name)),cv2.imread(str(OUT/'targets'/name)))
        for steps in order:
            yield z,n,steps,(OUT if steps==1000 else div.OUT)/'phases'/name

def capture():
    items=list(inputs())
    bench.capture(OUT,[i[0] for i in items],[i[3] for i in items],register_on_move=True)
    for i,(z,n,steps,_) in enumerate(items):
        path=OUT/'camera_png'/f'{i+1:02d}_z{z:06.3f}mm.png'
        path.rename(path.with_name(f'{i+1:02d}_DIV2K_{n:04d}_{z:g}mm_{steps}steps.png'))
def evaluate():
    regs=json.loads((OUT/'geometry.json').read_text())
    result=[]
    for i,(z,n,steps,phasepath) in enumerate(inputs()):
        target=div.target_image(n)
        img=cv2.imread(str(OUT/'camera_png'/f'{i+1:02d}_DIV2K_{n:04d}_{z:g}mm_{steps}steps.png'))
        assert img.shape==(3000,4096,3)
        H=np.array(next(r for r in regs if r['depth_mm']==z and r['accepted'])['models']['pooled'])
        # Use the same geometry and entire native target ROI for both step counts.
        measured=cv2.warpPerspective(srgb_to_linear(img[...,::-1].astype('f')/255),H,(bench.WIDTH,bench.HEIGHT),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        target_amp=np.sqrt(target)
        amp=np.sqrt(np.clip(measured,0,1))
        def mse(a,b):
            scale=(a*b).mean(axis=(0,1))/np.maximum((a*a).mean(axis=(0,1)),1e-12)
            return float(((a*scale-b)**2).mean())
        ms=mse(amp,target_amp)
        weights=np.float32([.2126,.7152,.0722])
        def hp(a):
            y=a@weights
            return (y-cv2.GaussianBlur(y,(0,0),3)).ravel()
        detail=float(np.corrcoef(hp(target),hp(measured))[0,1])
        result.append(dict(z=z,image=n,steps=steps,mse=ms,detail=detail,roi_any_channel_at_255=float((measured.max(2)>=1-1e-6).mean())))
    sim=[]
    for z,n in PAIRS:
        target=np.sqrt(div.target_image(n))
        for steps,folder in [(100,div.OUT),(1000,OUT)]:
            codes=cv2.imread(str(folder/'phases'/(div.stem(z,n)+'.png')))[...,::-1]
            err=[]
            for ch,wl in enumerate(bench.WAVELENGTHS):
                model=ASM(dict(bench.C,slm_shape=[bench.HEIGHT,bench.WIDTH],wavelength_m=wl,distance_m=z*.001),'cuda')
                phase=torch.from_numpy((1-codes[...,ch].astype('f')/255)*2*np.pi-np.pi).cuda()
                a=model(phase,quantize=False).abs().cpu().numpy()
                err.append(mse(a[...,None],target[...,ch,None]))
            sim.append(dict(z=z,steps=steps,mse=float(np.mean(err))))
    (OUT/'comparison.json').write_text(json.dumps(dict(camera=result,simulation=sim),indent=2))
    print('Camera',result,'ASM',sim,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['design','capture','evaluate']);q=ap.parse_args()
    {'design':design,'capture':capture,'evaluate':evaluate}[q.action]()
