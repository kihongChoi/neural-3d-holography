"""Independent green drive diagnostics and a three-slot camera comparison."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import multiplane_green as mp
from .evaluate_green_lut import PATCH, drive
from .green_drive_lut import GreenDriveLUT
from .hardware import Bench, encode_phase
from .measure_slm_plane import write
from .rgb_run import DELAYS, DARK_DELAYS
from .stage import Stage,validate_target

def grab(bench, frame, channel):
    bench.show_codes(frame)
    raw=bench.capture()[...,channel].copy()
    if max(m["saturated_fraction"] for m in bench.frame_metadata[-2:])>.0001:
        raise RuntimeError("Saturated sensor")
    return raw

def rectification(bench, ch, out, prior):
    # Compare a grating confined to the selected HDMI channel with an inactive
    # slot and flat repeats. This also checks that RGB timing is not permuted.
    src=prior["src"]
    y,x=np.indices((1080,1920))
    mask=np.zeros(x.shape,bool)
    for cx,cy in src:
        mask |= (x-cx)**2+(y-cy)**2<35**2
    codes=encode_phase((mask*(1-2*((x+y)%2))*np.pi/2).astype("f"))
    frame=np.full((1080,1920,3),128,np.uint8)
    bench.set_timing(100.,DARK_DELAYS[ch])
    off=grab(bench,frame,ch)
    timing=bench.set_timing(100.,DELAYS[ch])
    flat=grab(bench,frame,ch)
    shifted=frame.copy()
    shifted[...,(ch+1)%3]=codes
    wrong=grab(bench,shifted,ch)
    frame[...,ch]=codes
    dark=grab(bench,frame,ch)
    frame.fill(128)
    ref=grab(bench,frame,ch)
    meanflat=(flat+ref)*.5

    response=cv2.GaussianBlur((meanflat-dark)/np.maximum(meanflat,.003),(0,0),4)
    centers=[];quality=[]
    for px,py in np.rint(prior["centers"]).astype(int):
        win=response[py-65:py+66,px-65:px+66]
        w=np.maximum(win-.3*win.max(),0)
        if w.sum()<30: raise RuntimeError(f"Channel {ch} missing optical disk")
        yy,xx=np.indices(win.shape)
        cx=px-65+(xx*w).sum()/w.sum();cy=py-65+(yy*w).sum()/w.sum()
        centers.append((cx,cy))
        roi=np.s_[round(cy)-10:round(cy)+11,round(cx)-10:round(cx)+11]
        f=meanflat[roi].mean()
        quality.append([float((f-a[roi].mean())/max(f,.003)) for a in (dark,wrong,ref)]+[float(off[roi].mean()/max(f,.003))])
    corners=[0,2,6,8]
    H=cv2.getPerspectiveTransform(src[corners],np.float32(centers)[corners])
    err=np.linalg.norm(cv2.perspectiveTransform(src[None],H)[0]-centers,axis=1)
    q=np.array(quality)
    write(out/f"slot{ch}.json",dict(timing=timing,error_px=err.tolist(),disk_own_wrong_repeat_offratio=quality))
    if max(err)>3 or q[:,0].min()<.15 or np.abs(q[:,1:3]).max()>.12 or q[:,3].max()>.25:
        raise RuntimeError(f"Channel {ch} optical slot validation failed")
    yy,xx=np.indices((256,256),dtype="f")
    xy=cv2.perspectiveTransform(np.stack((xx+832,yy+412),axis=-1),H)
    def rect(a):return cv2.remap(a,xy[...,0],xy[...,1],cv2.INTER_LINEAR)
    flat_roi=rect(meanflat)
    np.savez(out/f"refs{ch}.npz",flat=flat_roi,off=rect(off),H=H)
    return rect,flat_roi

def compare(bench,lut,root,out,prior):
    maps=[rectification(bench,ch,out,prior) for ch in range(3)]
    scenes=sorted((root/"targets_v1").glob("div2k_*.npz"))+sorted((root/"targets_v1").glob("video_*.npz"))
    for i,file in enumerate(scenes):
        with np.load(file) as data:
            for method in ("ABC" if i%2==0 else "CBA"):
                frame=drive(lut,data["drive"],data["learned"],method)
                bench.show_codes(frame)
                channels=[]
                for ch,(rect,_) in enumerate(maps):
                    bench.set_timing(100.,DELAYS[ch])
                    a=bench.capture()[...,ch]
                    if max(m["saturated_fraction"] for m in bench.frame_metadata[-2:])>.0001:
                        raise RuntimeError("Saturation in RGB capture")
                    channels.append(rect(a))
                np.save(out/f"{file.stem}_{method}.npy",np.stack(channels,-1))

def levels(bench,out,prior):
    # Nine physical field locations, repeated and shuffled symmetric binary
    # phase commands, different checker orientation reserved for validation.
    yy,xx=np.indices((1080,1920))
    offsets=np.float32([[x,y] for y in (230,540,850) for x in (375,960,1545)])
    field=cv2.perspectiveTransform(offsets[None],prior["H"])[0]
    points=np.concatenate([prior["centers"],field],axis=0)
    write(out/"drive_probe_locations.json",dict(sensor_half_px=points.tolist(),slm_xy=np.concatenate([prior["src"],offsets],axis=0).tolist(),note="First nine are registered disks. Last nine are diagnostic samples projected from central H; peripheral registration is approximate."))
    contrasts=[]
    for repeat in range(2):
        for n in np.random.default_rng(55+repeat).permutation(9):
            phase=(1-2*((xx+yy)%2))*float(n)*np.pi/8
            frame=encode_phase(phase.astype("f"))
            a=grab(bench,frame,1)
            vals=[]
            for x,y in np.rint(points).astype(int):
                vals.append(float(a[y-12:y+13,x-12:x+13].mean()))
            contrasts.append(dict(repeat=repeat,axis="checker",half_difference_rad=n.item()*np.pi/8,mean_intensity=vals))
    for n in (0,2,4,6,8):
        frame=encode_phase(((1-2*(xx%2))*n*np.pi/8).astype("f"))
        a=grab(bench,frame,1)
        vals=[float(a[y-12:y+13,x-12:x+13].mean()) for x,y in np.rint(points).astype(int)]
        contrasts.append(dict(repeat=0,axis="horizontal carrier",half_difference_rad=n*np.pi/8,mean_intensity=vals))
    write(out/"levels.json",contrasts)

def main(root,out):
    out.mkdir(exist_ok=False)
    lut=GreenDriveLUT(root/"green_drive_v1.npz",json.loads((root/"freeze.json").read_text())["sha256"])
    with Stage(interval=(92.7,101.1)) as stage:
        start=stage.initial["position_mm"]
        validate_target(start,stage.interval,stage.backlash_mm)
        write(out/"preflight.json",dict(initial=stage.initial,serial=stage.serial.decode(),limit=stage.real(stage.limits()[2]),policy=stage.dll.ISC_GetSoftLimitMode(stage.serial),backlash_mm=stage.backlash_mm,units_mm=stage.units(1.)))
        try:
            stage.move(mp.ZERO-6.25)
            with Bench(mp.C) as bench:
                mp.OUT=out
                mp.calibrate(bench,6.25)
                with np.load(out/"cal_6.25.npz") as prior:
                    levels(bench,out,prior)
                    compare(bench,lut,root,out,prior)
                    from .green_optical_timing import timed_replay
                    timed_replay(bench,lut,prior,Path.home()/"Downloads/mov_org_back_z8_alq13.mp4",out)
                from .green_focus_check import scan
                scan(bench,stage,out)
                write(out/"frames.json",bench.frame_metadata)
        finally:
            if stage.moves:stage.move(start)
            write(out/"final_stage.json",dict(state=stage.status(),moves=stage.moves))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    main(a.root,a.out)






