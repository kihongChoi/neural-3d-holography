"""Opposite phase sinusoid contrast, constrained to the allowed side of z=0."""
import cv2
import numpy as np

from . import multiplane_green as mp
from .hardware import encode_phase
from .measure_slm_plane import write

def scan(bench,stage,out):
    bench.set_timing(100.,6022.)
    y,x=np.indices((1080,1920),dtype="f")
    signals={name:np.cos(2*np.pi*coord/24) for name,coord in
             [("x",x),("y",y),("diagonal",(x+y)/np.sqrt(2))]}
    records=[]
    for rep,depths in enumerate(((.04,.16,.32,.60),(.60,.32,.16,.04))):
        for z in depths:
            pos=stage.move(mp.ZERO-z)
            tag=f"scan_{rep}_{z}"
            mp.calibrate(bench,tag)
            with np.load(out/f"cal_{tag}.npz") as cal:
                H=cal["H"]
                yy,xx=np.indices((256,256),dtype="f")
                xy=cv2.perspectiveTransform(np.stack((xx+832,yy+412),-1),H)
                def warp(a):return cv2.remap(a,xy[...,0],xy[...,1],cv2.INTER_LINEAR)
                white=warp(cal["flat"])
            for name,wave in signals.items():
                pair=[]
                for sign in (1,-1):
                    bench.show_codes(encode_phase(sign*.65*wave))
                    a=bench.capture()[...,1]
                    if max(m["saturated_fraction"] for m in bench.frame_metadata[-2:])>.0001:
                        raise RuntimeError("Focus probe saturated")
                    pair.append(warp(a))
                diff=(pair[0]-pair[1])/(2*white.mean())
                np.savez(out/f"{tag}_{name}.npz",plus=pair[0],minus=pair[1])
                coord={"x":xx+832,"y":yy+412,"diagonal":(xx+832+yy+412)/np.sqrt(2)}[name]
                phase=2*np.pi*coord/24
                mat=np.stack([np.sin(phase).ravel(),np.cos(phase).ravel(),np.ones(phase.size)],1)
                coeff=np.linalg.lstsq(mat,diff.ravel(),rcond=None)[0][:2]
                records.append(dict(z=z,repeat=rep,axis=name,position_mm=pos["position_mm"],coeff=coeff.tolist(),power=float(coeff@coeff)))
            write(out/"focus_scan.json",records)
