"""Distil a pointwise drive-code correction on recorded training phase IDs only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from benchtop import multiplane_green as mp
from benchtop.hardware import encode_phase

BASE = Path("results/multiplane_green_20261103_100356")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fit(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out/"green_drive_v1.npz"
    if path.exists():
        raise FileExistsError(path)
    teacher = mp.Model().cuda().eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(BASE/"best.pt", weights_only=True))
    kernels = {z: mp.kernel(z) for z in mp.TRAIN}
    # Every case uses its original target, phase and nominal synthesis depth.
    samples = []
    for i in range(40):
        phase = np.load(BASE/f"phase_{i}.npy")
        index = torch.as_tensor(255-encode_phase(phase), device="cuda").long()
        samples.append((index, mp.image("train", i+1), kernels[mp.TRAIN[i%3]]))
    # Piecewise-linear nominal phase -> drive phase. Positive slopes permit
    # phase expansion around the carrier. Endpoints are clipped before encoding.
    axis = torch.linspace(-torch.pi, torch.pi, 256, device="cuda")
    knots = torch.zeros(16, device="cuda", requires_grad=True)
    bias = torch.zeros((), device="cuda", requires_grad=True)
    opt = torch.optim.Adam([knots, bias], lr=.035)
    rng = np.random.default_rng(11303)
    def curve():
        inc = torch.exp(knots) * (2*torch.pi/16)
        edges = torch.cat([inc.new_zeros(1), inc.cumsum(0)])
        edges = edges - edges[8] + bias
        coords = torch.arange(256,device="cuda")*16/255
        lo = coords.long().clamp(max=15)
        return torch.lerp(edges[lo],edges[lo+1],coords-lo).clamp(-torch.pi,torch.pi-1e-5)
    history = []
    for epoch in range(12):
        total = []
        for i in rng.permutation(40):
            idx, target, kernel = samples[i]
            prediction = teacher(curve()[idx], kernel)
            mse = (prediction-target).square().mean()
            opt.zero_grad()
            (mse + .0005*knots.diff().square().mean()).backward()
            opt.step()
            total.append(float(mse.detach()))
        history.append(float(np.mean(total)))
    phase = curve().detach().cpu().numpy()
    table = encode_phase(phase)[::-1].copy()
    meta = dict(version="green-drive-v1", wavelength_nm=520, input="inverted nominal uint8 HDMI code",
                output="uint8 HDMI drive code", axes=["nominal_code_0_to_255"], units="digital counts",
                order="table[input]", interpolation="exact discrete lookup",phase_convention="nominal radians wrap [-pi,pi), invert, round to 0..255; optimized drive radians clip before inverted encoding",
                patch_xywh=[704,284,512,512],origin="upper left of 1920x1080 HDMI frame, y down",
                fit_depth_mm=list(mp.TRAIN),fit_samples="DIV2K train 1..40, archived phase IDs 0..39",exposure_us=103,
                fit="distillation from frozen camera-trained CNN; effective output correction, physical phase response unidentified",
                reference_model_sha256=digest(BASE/"best.pt"), reference_source_sha256=digest(BASE/"experiment_source.py"),
                validation="requires independent camera A/B", conditions="LETO central patch, external CFS green, central 256x256 camera-aligned ROI; optical power in mW unknown")
    np.savez(path, table=table, metadata=json.dumps(meta))
    (out/"freeze.json").write_text(json.dumps(dict(sha256=digest(path),metadata=meta,history=history),indent=2))
    print(digest(path),history,flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    fit(p.parse_args().out)
