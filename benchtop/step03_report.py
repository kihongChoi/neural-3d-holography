"""Report independent validation residuals without refitting the camera transforms."""
import json
from pathlib import Path
import cv2,numpy as np
import matplotlib.pyplot as plt
from .geometry_registration import FIT,CORNERS,layout,project
from .step03_geometry import ROOT
from .step03_scene_check import ALT
from .step02_div2k import stem,OUT

def main():
    runs=json.loads((ROOT/'validation/registration.json').read_text());fresh=json.loads((ALT/'registration.json').read_text())
    assert len(runs)==6 and len(fresh)==3
    def rms(r,k):return r['scores'][k]['rms']
    rows=[]
    for f in fresh:
        d=f['depth_mm'];ab=[r for r in runs if r['depth_mm']==d];assert len(ab)==2
        stats={}
        for k in [*FIT,'pooled']:
            first,second=[project(np.array(r['models'][k]),layout('circle')) for r in ab]
            stats[k]=dict(holdout_cross_rms=float(np.sqrt(np.mean([rms(v,k)**2 for v in ab]))),
                          reverse_repeat_shift_rms=float(np.sqrt(np.mean(np.sum((first-second)**2,axis=1)))))
        # The first photograph trains the single-image reference; only the second is scored.
        a,b=f['scenes']
        h,good=cv2.findHomography(np.float32(a['src']),np.float32(a['dst']),cv2.RANSAC,4.)
        if h is None or good.sum()<8:
            control=None
        else:
            err=np.linalg.norm(project(h,np.float32(b['src']))-np.array(b['dst']),axis=1)
            control=dict(fitted_scene=a['id'],tested_scene=b['id'],fit_inliers=int(good.sum()),test_n=len(err),rms=float(np.sqrt(np.mean(err**2))),p95=float(np.percentile(err,95)))
        rows.append(dict(depth=d,methods=stats,one_photo_transfer=control,seconds=[r['seconds'] for r in ab],scene_check=[{k:s[k] for k in ['id','n','quadrants','scores']} for s in f['scenes']]))
    fig,axs=plt.subplots(3,3,figsize=(14,9),layout='constrained')
    for row,rec in enumerate(fresh):
        d=rec['depth_mm'];check=next(r for r in runs if r['repeat']==0 and r['depth_mm']==d)
        file=ROOT/'validation/camera_png'/f'r0_z{d:g}_cross.png'
        cam=cv2.imread(str(file),0);ax=axs[row,0]
        ax.imshow(cam[::4,::4],extent=(0,4096,3000,0),cmap='gray',vmin=0,vmax=255)
        h=np.array(check['models']['pooled']);pred=project(h,np.float32(check['validation_src']));obs=np.array(check['validation_full_sensor']);diff=obs-pred
        ax.scatter(*pred.T,facecolors='none',edgecolors='#60ffbe',s=18,lw=.8)
        ax.quiver(pred[:,0],pred[:,1],diff[:,0]*28,diff[:,1]*28,angles='xy',scale_units='xy',scale=1,color='#ff663c',width=.003)
        ax.set_title(f'{d:g} mm | independent crosses | {check["scores"]["pooled"]["rms"]:.2f} px')
        ax.axis('off')
        for j,scene in enumerate(rec['scenes'],1):
            n=scene['id'];name=stem(d,n);base=OUT if n<=803 else ALT
            t=cv2.imread(str(base/'targets'/(name+'.png')),0);c=cv2.imread(str(ALT/'camera_png'/(name+'.png')),0)
            h=np.array(rec['models']['pooled']);registered=cv2.warpPerspective(c,h,(t.shape[1],t.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            edges=cv2.Canny(cv2.GaussianBlur(t,(0,0),1.6),38,80)>0
            overlay=np.repeat(registered[...,None],3,axis=2);overlay[edges]=[0,229,249]
            ax=axs[row,j];ax.imshow(overlay[::2,::2]);ax.axis('off');ax.set_title(f'DIV2K {n:04d} | no refit | {scene["scores"]["pooled"]["rms"]:.2f} px')
    fig.suptitle('Per-move geometry | validation centers (arrows x28) | target edges in cyan over actual camera images')
    fig.savefig(ROOT/'verification.png',dpi=145);plt.close(fig)
    (ROOT/'comparison.json').write_text(json.dumps(rows,indent=2))
    for r in rows:print(r['depth'],[(k,round(v['holdout_cross_rms'],3),round(v['reverse_repeat_shift_rms'],3)) for k,v in r['methods'].items()],[(q['id'],q['n'],q['quadrants'],round(q['scores']['pooled']['rms'],2),round(q['scores']['pooled']['p95'],2)) for q in r['scene_check']])
    print('Single image A to B',[(r['depth'],r['one_photo_transfer']) for r in rows]);print('Total times',min(v['seconds'] for v in runs+fresh),max(v['seconds'] for v in runs+fresh))
if __name__=='__main__':main()
