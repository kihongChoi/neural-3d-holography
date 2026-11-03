"""Prepare fixed A/B/C drives and acquire green measurements in the 0..10 mm range."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import multiplane_green as mp
from .green_drive_lut import GreenDriveLUT
from .hardware import Bench, encode_phase
from .measure_slm_plane import write
from .paper_model import dpac, image_metrics
from .stage import Stage,validate_target
from .targets import prepare_image,srgb_to_linear

BASE=Path("results/multiplane_green_20261103_100356")
PATCH=np.s_[284:796,704:1216]

def targets(video):
    root=mp.DATA/"DIV2K_valid_HR"
    for idx in (871,872,873,874,875,876):
        yield f"div2k_{idx}",prepare_image(root/f"{idx:04}.png",(256,256))[0]
    y,x=np.indices((256,256))
    line=(.12+.75*((x//4+y//32)%2)).astype("f")
    yield "lines",np.repeat(line[...,None],3,axis=-1)
    for name,idx,scale,offset in [("bright",877,.45,.52),("dark",878,.38,0)]:
        yield name,(prepare_image(root/f"{idx:04}.png",(256,256))[0]*scale+offset).astype("f")
    cap=cv2.VideoCapture(str(video))
    try:
        for i in range(8):
            ok,frame=cap.read()
            if not ok: raise ValueError("Video sequence too short")
            rgb=cv2.resize(frame,(256,256),interpolation=cv2.INTER_AREA)[...,::-1].copy()
            yield f"video_{i:03}",np.sqrt(srgb_to_linear(rgb.astype("f")/255))
    finally:
        cap.release()

def prepare(root,video):
    plans=root/"targets_v1"
    plans.mkdir(exist_ok=False)
    model=mp.Model().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(root/"best.pt",weights_only=True))
    z=6.25
    kernels=[mp.ASM(dict(mp.C,slm_shape=[512,512],distance_m=z*.001,wavelength_m=w),"cuda") for w in (638e-9,520e-9,450e-9)]
    for name,target in targets(video):
        phase=[dpac(k,mp.t(target[...,i].copy())) for i,k in enumerate(kernels)]
        tar=mp.t(target[...,1].copy())
        mask=torch.ones_like(tar)
        # Preset before evaluation. Same initialization and iteration count for A and C.
        asm=mp.synth(None,[kernels[1]],[tar],[mask],phase[1],32)
        learned=mp.synth(model,[kernels[1]],[tar],[mask],phase[1],32)
        drive=np.stack([encode_phase(p.cpu().numpy()) for p in (phase[0],asm,phase[2])],axis=-1)
        np.savez(plans/f"{name}.npz",target=target,drive=drive,learned=encode_phase(learned.cpu().numpy()),depth=z)
    print("Prepared",len(list(plans.glob("*.npz"))),"fixed targets")

def drive(lut,base,learned,method):
    packed=base.copy()
    if method=="B": packed[...,1]=lut.apply(np.ascontiguousarray(base[...,1]))
    elif method=="C": packed[...,1]=learned
    frame=np.full((1080,1920,3),128,np.uint8)
    frame[PATCH]=packed
    return frame

def acquire(root,out):
    smoke=json.loads((root/"out1/result.json").read_text())
    if not smoke.get("optical_pass"): raise RuntimeError("Optical smoke gate failed")
    freeze=json.loads((root/"freeze.json").read_text())
    lut=GreenDriveLUT(root/"green_drive_v1.npz",freeze["sha256"])
    out.mkdir(exist_ok=False)
    mp.OUT=out
    rows=[]
    with Stage(interval=(92.7,101.1)) as stage:
        start=stage.initial["position_mm"]
        for d in (*mp.DEPTHS,):
            validate_target(mp.ZERO-d,stage.interval,stage.backlash_mm)
        write(out/"preflight.json",dict(start=stage.initial,serial=stage.serial.decode(),guard=stage.real(stage.limits()[2]),policy=stage.dll.ISC_GetSoftLimitMode(stage.serial),units_mm=stage.units(1.),backlash=stage.backlash_mm,sha256=freeze["sha256"]))
        try:
            with Bench(mp.C) as bench:
                for j,d in enumerate(mp.DEPTHS):
                    stage.move(mp.ZERO-d)
                    # New per-depth fixed white and registration only. LUT frozen beforehand.
                    mp.calibrate(bench,d)
                    cal=np.load(out/f"cal_{d}.npz")
                    yy,xx=np.indices((256,256),dtype="f")
                    xy=cv2.perspectiveTransform(np.stack([xx+832,yy+412],-1),cal["H"])
                    def warp(a): return cv2.remap(a,xy[...,0],xy[...,1],cv2.INTER_LINEAR)
                    white=cv2.GaussianBlur(warp(cal["flat"]),(0,0),12)
                    def measure(key,baseline,learned,target,roi):
                        for rep in range(2):
                            for method in ("ABC" if (len(key)+rep)%2==0 else "CBA"):
                                frame=drive(lut,baseline,learned,method)
                                bench.show_codes(frame)
                                raw=bench.capture()[...,1]
                                sat=max(f["saturated_fraction"] for f in bench.frame_metadata[-2:])
                                if sat>.0001: raise RuntimeError("Saturated capture")
                                intensity=warp(raw)
                                amp=np.sqrt(np.maximum(intensity,0)/white)
                                np.save(out/f"{d}_{key}_{rep}_{method}.npy",intensity)
                                metrics=image_metrics(torch.from_numpy(amp[roi]),torch.from_numpy(target[roi].copy()))
                                rows.append(dict(depth=d,scene=key,repeat=rep,method=method,**metrics))
                                write(out/"evaluation.json",rows)
                    for i in range(6):
                        a=encode_phase(np.load(BASE/f"final_phase_{i}_asm.npy"))
                        c=encode_phase(np.load(BASE/f"final_phase_{i}_learned.npy"))
                        baseline=np.repeat(a[...,None],3,axis=2)
                        tar=mp.image("valid",841+4*i+j).cpu().numpy()
                        yp=j//2*128;xp=j%2*128
                        measure(f"historical_{i}",baseline,c,tar,np.s_[yp:yp+128,xp:xp+128])
                    if d==6.25:
                        for path in sorted((root/"targets_v1").glob("*.npz")):
                            with np.load(path) as p:
                                measure(path.stem,p["drive"],p["learned"],p["target"][...,1],np.s_[:,:])
        finally:
            if stage.moves and abs(stage.status()["position_mm"]-start)>.002: stage.move(start)
            write(out/"final_stage.json",dict(state=stage.status(),moves=stage.moves))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["prepare","acquire"]);p.add_argument("root",type=Path)
    p.add_argument("path",type=Path,help="video for prepare, new output directory for acquire")
    args=p.parse_args()
    (prepare if args.command=="prepare" else acquire)(args.root,args.path)
