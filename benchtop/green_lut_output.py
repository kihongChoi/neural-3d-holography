"""Play images, video or prepared central RGB drive codes using the frozen G LUT."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .green_drive_lut import GreenDriveLUT
from .hardware import Bench, encode_phase
from .multiplane_green import C, ZERO, synth
from .optics import ASM
from .paper_model import dpac
from .targets import prepare_image, srgb_to_linear
from .stage import Stage

V1_SHA256 = "d2f5b69379d43a339546a8d3d10f6a7e117c98518df6daf869a7a213def59f7c"

def inputs(path):
    if path.suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
        if a.dtype != np.uint8 or a.shape != (512,512,3):
            raise ValueError("Expected inverted uint8 HDMI codes, shape (512,512,3)")
        yield a
    elif path.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tiff",".webp"):
        yield prepare_image(path,(256,256))[0]
    else:
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode {path}")
            while ok:
                rgb = cv2.resize(frame,(256,256),interpolation=cv2.INTER_AREA)[...,::-1].copy()
                yield np.sqrt(srgb_to_linear(rgb.astype("f")/255))
                ok, frame = cap.read()
        finally:
            cap.release()

def nominal_codes(target, kernels, steps):
    channels = []
    for ch,kernel in enumerate(kernels):
        amplitude = torch.from_numpy(target[...,ch].copy()).to(kernel.kernel.device)
        phase = dpac(kernel,amplitude)
        if ch==1 and steps:
            phase = synth(None,[kernel],[amplitude],[torch.ones_like(amplitude)],phase,steps)
        channels.append(encode_phase(phase.cpu().numpy()))
    return np.stack(channels,axis=2)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--lut",type=Path)
    parser.add_argument("--off",action="store_true",help="Bypass the green drive correction")
    parser.add_argument("--depth-mm",type=float,default=6.25)
    parser.add_argument("--steps",type=int,choices=[0,32],default=32,help="32 matches the new-scene camera comparison; 0 uses single-pass DPAC")
    parser.add_argument("--compute-only",action="store_true",help="Generate frames without opening the hardware")
    parser.add_argument("--frames",type=int,help="Limit the input sequence length")
    parser.add_argument("--timing",type=Path,help="Write per-frame millisecond measurements")
    args=parser.parse_args()
    if not 0<=args.depth_mm<=10 or (args.frames is not None and args.frames < 1):
        parser.error("Invalid depth or frame count")
    if not args.off and args.lut is None:
        parser.error("Provide --lut or --off")
    lut = None if args.off else GreenDriveLUT(args.lut,V1_SHA256)
    kernels = [] if args.input.suffix.lower()==".npy" else [
        ASM(dict(C,slm_shape=[512,512],distance_m=args.depth_mm*.001,wavelength_m=w),"cuda")
        for w in (638e-9,520e-9,450e-9)]
    records = []
    frames = inputs(args.input)
    try:
        with (nullcontext() if args.compute_only else Stage(interval=(0.,101.1),monitor_only=True)) as stage:
            if stage:
                state=stage.status()
                stage.require_stationary(state)
                if abs(state["position_mm"]-(ZERO-args.depth_mm))>.002:
                    raise ValueError(f"Stage must already be at {ZERO-args.depth_mm:.3f} mm; read {state}")
            with (nullcontext() if args.compute_only else Bench(dict(C,settle_s=.025))) as bench:
                while args.frames is None or len(records)<args.frames:
                    start=time.perf_counter_ns()
                    try: value=next(frames)
                    except StopIteration: break
                    decoded=time.perf_counter_ns()
                    codes = nominal_codes(value,kernels,args.steps) if kernels else value.copy()
                    generated=time.perf_counter_ns()
                    if lut: codes[...,1]=lut.apply(np.ascontiguousarray(codes[...,1]))
                    corrected=time.perf_counter_ns()
                    full=np.full((1080,1920,3),128,np.uint8)
                    full[284:796,704:1216]=codes
                    packed=time.perf_counter_ns()
                    if bench:bench.show_codes(full)
                    done=time.perf_counter_ns()
                    records.append(dict(input_ms=(decoded-start)/1e6,generate_ms=(generated-decoded)/1e6,
                                        apply_ms=(corrected-generated)/1e6,packed_ms=(packed-start)/1e6,
                                        total_ms=(done-start)/1e6))
                if bench and len(records)==1:
                    time.sleep(C["settle_s"])
    finally:
        frames.close()
    if args.timing:args.timing.write_text(json.dumps(dict(steps=args.steps,depth=args.depth_mm,compute_only=args.compute_only,on=bool(lut),records=records),indent=2))
    print(f"Processed {len(records)} frames")

if __name__ == "__main__":
    main()
