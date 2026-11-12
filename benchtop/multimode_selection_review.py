"""Fixed selection-only residual panels from already frozen main checkpoints."""
import argparse,gc,json,os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .integrated_rgb_train import Frames,JointRGB,masked_errors
from .integrated_rgb_data import sha,write_json,now
from .multimode_train import validate_manifest,MANIFEST_SHA
from .multimode_model import MultimodeRGB


@torch.no_grad()
def run(a):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(exist_ok=False)
    assert sha(a.cache/'manifest.json')==MANIFEST_SHA
    data=json.loads((a.cache/'manifest.json').read_text());validate_manifest(data)
    rows=[r for r in data['records'] if r['split']=='selection'];ds=Frames(a.cache,rows)
    parents=sorted({r['scene'] for r in rows})[:3]
    panel=[]
    for parent in parents:
        for depth in (3.,7.,9.):
            choices=[i for i,r in enumerate(rows) if r['scene']==parent and r['depth_mm']==depth and r['condition']=='current_lut']
            panel.append(min(choices,key=lambda i:rows[i]['id']))
    masks={}
    for key,r in data['references'].items():
        assert sha(a.cache/r['file'])==r['sha256']
        with np.load(a.cache/r['file']) as f:masks[key]=torch.from_numpy(f['valid'].copy()).cuda()
    write_json(a.out/'panel_manifest.json',dict(created_kst=now(),rule='lexicographic first3 parents, depths3/7/9, current_lut first ID',
        records=[rows[i] for i in panel],test_pixels_decoded=False,display_gamma=2.2,residual_abs_mean_rgb_range=[0,.15],full_frame=True))
    targets={};predictions={};residuals={};summaries=[]
    thumb=lambda x:F.interpolate(x[None],size=(180,324),mode='area')[0].permute(1,2,0).cpu().numpy()
    for seed in (117,229):
        for rank in (1,3,6,0):
            name=f'k{rank}_s{seed}';p=a.main/name
            result=json.loads((p/'result.json').read_text())
            assert result['state']=='complete' and result['samples']==75000 and not result['test_pixels_decoded']
            model=JointRGB(a.initial) if rank==0 else MultimodeRGB(rank,'cuda')
            ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False);model.load_weights(ck['weights']);del ck
            model.eval();records=[]
            for i,r in enumerate(rows):
                _,code,y=ds[i];y=y.cuda();pred=model(code.cuda(),r['depth_mm']);mask=masks[r['reference_key']].bool()
                error=pred-y;vals=[]
                for axis in (-1,-2):
                    d=error.diff(dim=axis)
                    m=mask[...,1:]&mask[...,:-1] if axis==-1 else mask[...,1:,:]&mask[...,:-1,:]
                    vals.append((d.square()*m).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1))
                edge=(vals[0]+vals[1])/2
                mse,_=masked_errors(pred,y,mask)
                assert torch.isfinite(edge).all() and torch.isfinite(mse).all()
                records.append(dict(id=r['id'],scene=r['scene'],depth_mm=r['depth_mm'],edge_residual_mse_rgb=edge.tolist(),intensity_mse=float(mse.mean())))
                if i in panel:
                    targets[i]=thumb(y);predictions[(name,i)]=thumb(pred)
                    residuals[(name,i)]=thumb(error.abs()).mean(-1)
                if (i+1)%50==0:write_json(a.out/'status.json',dict(state='running',model=name,selection_count=i+1,updated_kst=now(),test_pixels_decoded=False))
            epoch=result['best_progress']['epoch']+1
            original=json.loads((p/f'selection_epoch_{epoch:03d}.json').read_text())
            assert [r['id'] for r in records]==[r['id'] for r in original['records']]
            maxdiff=max(abs(r['intensity_mse']-old['intensity_mse']) for r,old in zip(records,original['records']))
            assert maxdiff<1e-7,('frozen prediction drift',name,maxdiff)
            grouped=defaultdict(list)
            for r in records:grouped[r['scene']].append(r['edge_residual_mse_rgb'])
            entry=dict(name=name,checkpoint_sha256=sha(p/'best.pt'),max_mse_difference_from_original=maxdiff,
                parent_balanced_edge_residual_mse_rgb=np.mean([np.mean(v,axis=0) for v in grouped.values()],axis=0).tolist(),records=records)
            write_json(a.out/(name+'.json'),entry);summaries.append({k:v for k,v in entry.items() if k!='records'})
            del model;gc.collect();torch.cuda.empty_cache()
        for i in panel:
            r=rows[i];fig,axes=plt.subplots(2,5,figsize=(15,4.6),layout='constrained')
            axes[0,0].imshow(np.clip(targets[i],0,1)**(1/2.2));axes[0,0].set_title('Measured RGB')
            axes[1,0].text(.02,.8,'Absolute residual\nRGB mean, range 0–0.15\nFull frame, downsampled\nDisplay gamma 2.2',transform=axes[1,0].transAxes,fontsize=10)
            for j,rank in enumerate((1,3,6,0),1):
                name=f'k{rank}_s{seed}';axes[0,j].imshow(np.clip(predictions[(name,i)],0,1)**(1/2.2))
                axes[0,j].set_title('CNN' if rank==0 else f'K{rank}')
                axes[1,j].imshow(residuals[(name,i)],vmin=0,vmax=.15,cmap='magma')
            for ax in axes.flat:ax.axis('off')
            fig.suptitle(f'Selection parent {r["scene"]} | {r["depth_mm"]:g} mm | current LUT | seed {seed}')
            fig.savefig(a.out/f'panel_{r["scene"]}_{int(r["depth_mm"])}mm_s{seed}.png',dpi=120);plt.close(fig)
    write_json(a.out/'summary.json',dict(state='complete',created_kst=now(),models=summaries,panels=18,
        edge_definition='mean squared horizontal/vertical first differences of prediction-minus-target over valid adjacent pixels; parent balanced',test_pixels_decoded=False))
    write_json(a.out/'status.json',dict(state='complete',updated_kst=now(),test_pixels_decoded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','initial','main','out'):p.add_argument(name,type=Path)
    run(p.parse_args())
