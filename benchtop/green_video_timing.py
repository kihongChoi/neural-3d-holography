"""Benchmark green video on the verified LETO HDMI output, without stage motion."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtGui import QImage

from .green_drive_lut import GreenDriveLUT
from .hardware import Bench, encode_phase
from .multiplane_green import C, kernel
from .paper_model import dpac
from .targets import srgb_to_linear

def benchmark(out, video):
    freeze=json.loads((out/"freeze.json").read_text())
    lut=GreenDriveLUT(out/"green_drive_v1.npz",freeze["sha256"])
    cap=cv2.VideoCapture(str(video))
    frames=[]
    try:
        for _ in range(80):
            ok,frame=cap.read()
            if not ok: break
            small=cv2.resize(frame,(256,256),interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(small[...,1]))
    finally:
        cap.release()
    if len(frames)<40: raise ValueError("Need >=40 sequential video frames")
    asm=kernel(6.25)
    def generate(frame):
        target=torch.from_numpy(np.sqrt(srgb_to_linear(frame.astype("f")/255))).cuda()
        return encode_phase(dpac(asm,target).cpu().numpy())
    times=[]
    with Bench(dict(C,settle_s=.025)) as b:
        for block,use in enumerate((False,True,True,False)):
            for i,frame in enumerate(frames):
                start=time.perf_counter_ns()
                codes=generate(frame)
                ready=time.perf_counter_ns()
                patch=lut.apply(codes) if use else codes
                looked=time.perf_counter_ns()
                full=np.full((1080,1920),128,np.uint8)
                full[284:796,704:1216]=patch
                b.show_codes(full)
                shown=time.perf_counter_ns()
                # Test the patch itself for stale frames; full-frame mean alone is insensitive.
                im=b.label.screen().grabWindow(0).toImage().convertToFormat(QImage.Format_Grayscale8)
                fb=np.frombuffer(im.constBits(),dtype=np.uint8).reshape(1080,im.bytesPerLine())[284:796,704:1216]
                if i>=10:
                    times.append(dict(block=block,frame=i,on=use,generate_ms=(ready-start)/1e6,lut_ms=(looked-ready)/1e6,
                                      compute_ms=(looked-start)/1e6,hdmi_ms=(shown-start)/1e6,
                                      mismatch_pixels=int(np.count_nonzero(fb!=patch))))
    result=dict(video=str(video),original_fps=cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS),
                mode="LETO-3-0088 1920x1080 60 Hz; grayscale HDMI for green slot, 25 ms settle and framebuffer readback; no optical timing",
                input="decoded 256x256 green sRGB8 -> amplitude -> single-depth GPU DPAC -> native 512x512 codes",
                gpu=torch.cuda.get_device_name(),sha256=freeze["sha256"],samples=times)
    for use in (False,True):
        rows=[r for r in times if r["on"]==use]
        result[str(use)]={k:{"mean":float(np.mean(v)),"p95":float(np.percentile(v,95))} for k in ("generate_ms","lut_ms","compute_ms","hdmi_ms") for v in [[r[k] for r in rows]]}
        result[str(use)]["framebuffer_mismatches"]=sum(bool(r["mismatch_pixels"]) for r in rows)
    (out/"video_timing.json").write_text(json.dumps(result,indent=2))
    print(result["True"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path);p.add_argument("video",type=Path)
    a=p.parse_args()
    benchmark(a.out,a.video)
