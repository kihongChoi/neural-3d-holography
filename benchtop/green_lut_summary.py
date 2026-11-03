"""Summarize actual acquisition rows; average repeats before ranking scenes."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from benchtop.rgb_run import srgb_preview
from benchtop import multiplane_green as mp

def render(root,run,rgb=None):
    rows=json.loads((run/"evaluation.json").read_text())
    by_scene={}
    for row in rows:
        by_scene.setdefault((row['depth'],row['scene']),{}).setdefault(row['method'],[]).append(row)
    def summary(keys):
        cases=[]
        for key in keys:
            gr=by_scene[key]
            if any(len(gr.get(n,[]))!=2 for n in "ABC"):continue
            m={n:float(np.mean([r["fixed_white_mse"] for r in gr[n]])) for n in "ABC"}
            cases.append(dict(z=key[0],scene=key[1],**m,improvement=m["A"]-m["B"],
                              repeat_abs_B=abs(gr["B"][0]["fixed_white_mse"]-gr["B"][1]["fixed_white_mse"])))
        if not cases:return {}
        mean={n:float(np.mean([r[n] for r in cases])) for n in "ABC"}
        return dict(mean=mean,relative_drop=1-mean["B"]/mean["A"],cnn_gap_fraction=(mean["A"]-mean["B"])/(mean["A"]-mean["C"]),n=len(cases),worse=sum(r['B']>r['A'] for r in cases),worst=min(cases,key=lambda r:r["improvement"]),mean_repeat_abs_B=float(np.mean([r["repeat_abs_B"] for r in cases])),cases=cases)
    groups={name:summary([k for k in by_scene if filt(k)]) for name,filt in {
        "archive_targets":lambda k:k[1].startswith("historical"),
        "new_natural":lambda k:k[1].startswith("div2k"),
        "video":lambda k:k[1].startswith("video"),
        "stress":lambda k:k[1] in ("dark","bright","lines")}.items()}
    for d in mp.DEPTHS:groups[str(d)]=summary([k for k in by_scene if k[0]==d and k[1].startswith("historical")])
    (run/"summary.json").write_text(json.dumps(groups,indent=2))
    for key,g in groups.items():
        if g:print(key,g["mean"],g["cnn_gap_fraction"],g["worst"]["scene"],g["worse"])
    if not rgb:return
    whites=np.stack([np.load(rgb/f"refs{ch}.npz")["flat"] for ch in range(3)],-1)
    # Same fixed per-channel scalar for every displayed panel; preserve illumination shading.
    display_white=np.median(whites,axis=(0,1))
    stats={}
    for file in sorted(rgb.glob("div2k_*_A.npy")):
        name=file.stem.rsplit("_",1)[0]
        target=np.load(root/"targets_v1"/(name+".npz"))["target"]
        stats[name]={}
        for method in "ABC":
            intensity=np.load(rgb/f"{name}_{method}.npy")
            a=np.sqrt(intensity[...,1]/cv2.GaussianBlur(whites[...,1],(0,0),12))
            stats[name][method]=float(np.mean((a-target[...,1])**2))
    (rgb/"reconnect_mse.json").write_text(json.dumps(stats,indent=2))
    ranked=sorted(stats,key=lambda n:stats[n]['A']-stats[n]['B'])
    fig,axs=plt.subplots(2,4,figsize=(11,5.9),layout='constrained')
    for row,name in enumerate([ranked[len(ranked)//2],ranked[0]]):
        target=np.load(root/"targets_v1"/(name+".npz"))["target"]
        images=[srgb_preview(target)]
        for method in "ABC":
            im=np.load(rgb/f"{name}_{method}.npy")
            images.append(srgb_preview(np.sqrt(im/display_white)))
        for col,(title,im) in enumerate(zip(["RGB target","ASM (camera)","Frozen G LUT (camera)","Learned G (camera)"],images)):
            ax=axs[row,col];ax.imshow(np.clip(im,0,1));ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(title)
            if col==0:ax.set_ylabel(("Representative" if row==0 else "Least gain")+"\n"+name)
            if col:ax.set_xlabel(f"G amplitude MSE   {stats[name]['ABC'[col-1]]:.4f}")
    fig.suptitle("6.25 mm  |  three camera slots  |  G only corrected, R/B drives fixed")
    fig.savefig(rgb/"rgb_montage.png",dpi=140);plt.close(fig)
    print("RGB reconnect",stats)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("run",type=Path);p.add_argument("rgb",type=Path,nargs="?")
    a=p.parse_args();render(a.root,a.run,a.rgb)
