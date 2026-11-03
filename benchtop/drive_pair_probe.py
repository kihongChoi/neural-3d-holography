"""Measure code-pair interference in independent random tile layouts and carriers."""
import itertools
from pathlib import Path

import cv2
import numpy as np

from .measure_slm_plane import write
from .rgb_run import DELAYS
from .green_rgb_recheck import rectification

def acquire(bench,prior,out):
    for ch in range(3):
        rectification(bench,ch,out,prior)
    xy=np.float32([(x,y) for y in (348,476,604,732) for x in (768,896,1024,1152)])
    centers=[]
    for ch in range(3):
        H=np.load(out/f"refs{ch}.npz")["H"]
        centers.append(np.rint(cv2.perspectiveTransform(xy[None],H)[0]).astype(int))
    codes=np.rint(np.linspace(0,255,13)).astype("uint8")
    pairs=list(itertools.combinations_with_replacement(range(len(codes)),2))
    rows=[]; mats=[]
    def capture(frame,rep,batch,pairs):
        bench.show_codes(frame)
        for ch,delay in enumerate(DELAYS):
            bench.set_timing(100.,delay)
            im=bench.capture()[...,ch]
            if max(m["saturated_fraction"] for m in bench.frame_metadata[-2:])>.0001:
                raise RuntimeError("Saturated drive-pair probe")
            # Store intensity cutouts in native half-sensor coordinates.
            sites=np.stack([im[y-12:y+13,x-12:x+13].copy() for x,y in centers[ch]])
            mats.append(sites)
            rows.append(dict(rep=rep,batch=batch,ch=ch,pairs=pairs,mean=sites.mean(axis=(1,2),dtype="d").tolist()))
    yy,xx=np.indices((128,128))
    blank=np.full((1080,1920),128,np.uint8)
    for rep in range(3):
        # Validation includes a change of grating orientation.
        carrier=(xx+yy)%2==0 if rep<2 else xx%2==0
        shuffled=list(pairs)
        np.random.default_rng(13030+rep).shuffle(shuffled)
        capture(blank,rep,-1,[])
        for batch in range(6):
            layout=shuffled[16*batch:16*batch+16]
            frame=blank.copy()
            for (cx,cy),(i,j) in zip(xy.astype(int),layout):
                frame[cy-64:cy+64,cx-64:cx+64]=np.where(carrier,codes[i],codes[j])
            capture(frame,rep,batch,layout)
        capture(blank,rep,6,[])
        write(out/"pair_data.json",dict(codes=codes.tolist(),centers_slm=xy.tolist(),centers_sensor=np.array(centers).tolist(),rows=rows))
        np.save(out/"pair_cutouts.npy",np.array(mats))
