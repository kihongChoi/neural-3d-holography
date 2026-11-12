"""Compare old and updated teacher responses on the same newly measured probes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .nh3d_update_train import ReplayData
from .nh3d_lut_train import probe_check
from .nh3d_lut_model import DepthSharedNH3D,optics,sha,write_json

def run(root):
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    data=ReplayData(root);out=data.root/'model_update_diagnostics';out.mkdir(exist_ok=False)
    trained=data.root/'updated_training_01'
    with np.load(data.plan['baseline_candidate'],allow_pickle=False) as f:parents=json.loads(str(f['metadata']));baseline=f['table'].copy()
    with np.load(trained/'rgb_lut_candidate.npz',allow_pickle=False) as f:table=f['table'].copy()
    summaries=[];fig,axes=plt.subplots(3,2,figsize=(11,13))
    for i,tag in enumerate('RGB'):
        old_path=Path(parents[i].get('checkpoint_path',str(Path(data.plan['baseline_candidate']).parent/tag/'best.pt')))
        if sha(old_path)!=parents[i]['teacher_sha256']:raise ValueError('Original teacher changed')
        net=DepthSharedNH3D(optics(tag),'cuda')
        net.load_learnable(torch.load(old_path,weights_only=True,map_location='cpu')['weights'])
        rows=[dict(r,variant='identity' if r['variant']=='current_lut' else r['variant']) for r in data.rows if r['_bank']==1 and r['channel']==tag and r['split']=='selection']
        before=probe_check(net,data,rows)
        checks=json.loads((trained/tag/'forward_checks.json').read_text())
        after=checks['probes'];write_json(out/f'{tag}_initial_probe_response.json',before)
        history=json.loads((trained/tag/'history.json').read_text())
        summaries.append(dict(channel=tag,epochs_executed=len(history),best_epoch=checks['best_epoch'],optimizer_steps=sum(h['updates'] for h in history),initial_old_mse=checks['initial_old_mse'],updated_old_mse=checks['final_old']['mean_mse'],initial_new_mse=checks['initial_new_mse'],updated_new_mse=checks['final_new']['mean_mse'],new_mse_reduction=1-checks['final_new']['mean_mse']/checks['initial_new_mse'],teacher_accepted=checks['accepted_for_distillation'],changed_lut_entries=int(np.sum(table[i]!=baseline[i])),initial_response={k:v for k,v in before.items() if k!='records'},updated_response={k:v for k,v in after.items() if k!='records'},initial_teacher_sha256=sha(old_path),updated_teacher_sha256=checks['teacher_sha256']))
        extent=max(abs(v) for probes in [before,after] for r in probes['records'] for v in [r['measured_target_mse_delta'],r['predicted_target_mse_delta']])*1.1
        for j,(title,probes) in enumerate([('Previous teacher',before),('Retained teacher (epoch 0)' if checks['best_epoch']==0 else f'Selected teacher (epoch {checks["best_epoch"]})',after)]):
            ax=axes[i,j]
            for z,color in [(3.,'#386cb0'),(7.,'#f7a541'),(9.,'#5e9c61')]:
                rows=[r for r in probes['records'] if r['depth_mm']==z]
                ax.scatter([r['measured_target_mse_delta'] for r in rows],[r['predicted_target_mse_delta'] for r in rows],label=f'{z:g} mm',color=color,alpha=.8)
            ax.plot([-extent,extent],[-extent,extent],color='#555555',linestyle='--',linewidth=1);ax.axhline(0,color='#cccccc');ax.axvline(0,color='#cccccc');ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent)
            ax.set_title(f'{tag}: {title} | response cosine {probes["median_response_cosine"]:.3f}');ax.set_xlabel('Measured change in target amplitude MSE');ax.set_ylabel('Predicted change');ax.legend(fontsize=8)
        del net;torch.cuda.empty_cache();print('Response diagnostic',tag,summaries[-1],flush=True)
    fig.suptitle('Same newly captured probes, before and after model update',fontsize=14);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'response_before_after.png',dpi=150);plt.close(fig)
    write_json(out/'summary.json',dict(dataset_sha256=data.fingerprint,candidate_sha256=sha(trained/'rgb_lut_candidate.npz'),baseline_sha256=data.plan['baseline_candidate_sha256'],channels=summaries,note='Model prediction diagnostic on selection data; not a camera quality A/B test. Response cosine compares spatial amplitude changes; scatter compares their target-MSE changes.'))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);print(run(p.parse_args().root))
