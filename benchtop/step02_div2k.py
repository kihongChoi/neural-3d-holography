"""Three independently optimized, full-canvas RGB DIV2K holograms."""
from pathlib import Path
import argparse
import io
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

from . import step01_dot_grid as bench
from .optics import ASM
from .paper_model import dpac
from .targets import srgb_to_linear

OUT = bench.ROOT/'results'/'step02_div2k_center_crop'
DATA = Path.home()/'Documents'/'DIV2K dataset'/'DIV2K_valid_HR'
PAIRS = [(0.,801),(5.,802),(10.,803)]
STEPS = 100


def target_image(number):
    with Image.open(DATA/f'{number:04d}.png') as src:
        im = ImageOps.exif_transpose(src)
        icc = im.info.get('icc_profile')
        if icc:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                           ImageCms.createProfile('sRGB'), outputMode='RGB')
        linear = srgb_to_linear(np.asarray(im.convert('RGB'),np.float32)/255)
    h,w = linear.shape[:2]
    if w < bench.WIDTH or h < bench.HEIGHT:
        raise ValueError(f'Source too small for a native center crop: {w}x{h}')
    x,y=(w-bench.WIDTH)//2,(h-bench.HEIGHT)//2
    return linear[y:y+bench.HEIGHT,x:x+bench.WIDTH].copy()



def stem(depth,number):
    return f'DIV2K_{number:04d}_z{depth:02.0f}mm'


def design(display, pairs=PAIRS, output=OUT, steps=STEPS):
    total=len(pairs)*3*steps
    completed=0
    last_refresh=0.
    refresh_count=0
    preview_seconds=0.
    lettering_seconds=0.
    start_time=time.perf_counter()
    preview=np.full((1080,1920,3),128,np.uint8)
    font=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),baseline=cv2.getTextSize('update 0000/0000',font,.55,1)
    x,y=1920-tw-14,1080-baseline-10
    assert y-th-4>=bench.ROI[0].stop
    output.mkdir(exist_ok=False)
    for folder in ['targets','phases']:
        (output/folder).mkdir()
    for depth,number in pairs:
        name=stem(depth,number)
        preview.fill(128)
        intensity=target_image(number)
        srgb=np.where(intensity<=.0031308,12.92*intensity,1.055*intensity**(1/2.4)-.055)
        bench.png(output/'targets'/f'{name}.png',cv2.cvtColor(np.uint8(np.rint(255*srgb)),cv2.COLOR_RGB2BGR))
        codes=[];before=[];after=[]
        for ch,wl in enumerate(bench.WAVELENGTHS):
            config=dict(bench.C,slm_shape=[bench.HEIGHT,bench.WIDTH],distance_m=depth*.001,wavelength_m=wl)
            model=ASM(config,'cuda')
            target=torch.from_numpy(np.ascontiguousarray(np.sqrt(intensity[...,ch]))).cuda()
            phase=dpac(model,target).detach().requires_grad_()
            optimizer=torch.optim.Adam([phase],lr=.025)
            best=float('inf')
            for step in range(steps+1):
                amp=model(phase).abs()
                with torch.no_grad():
                    scale=(amp*target).sum()/amp.square().sum().clamp_min(1e-8)
                loss=(scale*amp-target).square().mean()
                value=float(loss.detach())
                if step==0:before.append(value)
                if value<best:
                    best=value
                    best_phase=phase.detach().clone()
                if step<steps:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    completed+=1
                    now=time.perf_counter()
                    if now-last_refresh>=.125 or (step+1)==steps:
                        native=bench.encode_phase(phase.detach().cpu().numpy())
                        preview[bench.ROI][...,ch]=native
                        lettering_start=time.perf_counter()
                        preview[y-th-4:y+baseline+4,x-4:x+tw+4]=0
                        cv2.putText(preview,f'update {completed:04d}/{total:04d}',(x,y),font,.55,(255,255,0),1,cv2.LINE_AA)
                        lettering_seconds+=time.perf_counter()-lettering_start
                        display.update_display(preview)
                        if refresh_count==0 or completed==total:
                            display.settle()
                            display.verify_display()
                            from PySide6.QtGui import QImage
                            actual=display.label.screen().grabWindow(0).toImage().convertToFormat(QImage.Format_RGB888)
                            pixels=np.frombuffer(actual.constBits(),np.uint8).reshape(1080,actual.bytesPerLine())[:,:1920*3].reshape(1080,1920,3)
                            tile=np.s_[y-th-4:y+baseline+4,x-4:x+tw+4]
                            if not np.array_equal(pixels[tile],preview[tile]):raise RuntimeError('Progress overlay readback failed')
                        last_refresh=now
                        refresh_count+=1
                        preview_seconds+=time.perf_counter()-now
            assert np.isfinite(best) and best<before[-1]
            after.append(best)
            codes.append(bench.encode_phase(best_phase.cpu().numpy()))
            preview[bench.ROI][...,ch]=codes[-1]
        frame=np.stack(codes,axis=-1)
        path=output/'phases'/f'{name}.png'
        bench.png(path,cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        assert np.array_equal(cv2.imread(str(path))[...,::-1],frame)
        print(f'{name}: scaled ASM amplitude MSE, mean of RGB {np.mean(before):.7f} -> {np.mean(after):.7f}',flush=True)


    print(f'Preview {refresh_count} displayed frames, text {1000*lettering_seconds/refresh_count:.3f} ms/frame, full updates {preview_seconds:.2f} s / {time.perf_counter()-start_time:.2f} s total',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('step',choices=['design','capture'])
    args=parser.parse_args()
    if args.step=='design':
        display=bench.Bench(dict(bench.C,settle_s=.15))
        try:
            display.connect_display()
            design(display)
        finally:
            display.close()
    else:
        destination=OUT
        if (destination/'camera_png').exists():
            destination=OUT/time.strftime('rescan_%Y%m%d_%H%M%S')
            destination.mkdir(exist_ok=False)
        bench.capture(destination,[z for z,_ in PAIRS],[OUT/'phases'/f'{stem(z,n)}.png' for z,n in PAIRS],register_on_move=True)
