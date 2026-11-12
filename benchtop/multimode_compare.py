"""Parent-paired selection analysis. Never reads image or sealed test files."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .integrated_rgb_data import write_json,now


def parent_errors(evaluation):
    groups=defaultdict(list)
    for r in evaluation['records']:groups[r['scene']].append(r['intensity_mse'])
    return {k:float(np.mean(v)) for k,v in groups.items()}


def compare(run):
    experiments={}
    for rank in (0,1,3,6):
        for seed in (117,229):
            p=run/f'k{rank}_s{seed}';res=json.loads((p/'result.json').read_text())
            if res['state']!='complete' or res['test_pixels_decoded']:raise ValueError('Incomplete or unsealed experiment')
            progress=res['best_progress'];epoch=progress['epoch']+1
            ev=json.loads((p/f'selection_epoch_{epoch:03d}.json').read_text())
            experiments[(rank,seed)]=(res,ev,parent_errors(ev))
    rng=np.random.default_rng(80117);pairs=[]
    for reference in (1,0):
        for rank in (1,3,6):
            if reference==rank:continue
            for seed in (117,229):
                a,ae,ap=experiments[(reference,seed)];b,be,bp=experiments[(rank,seed)]
                if set(ap)!=set(bp):raise ValueError('Parent set mismatch')
                keys=sorted(ap);diff=np.array([bp[k]-ap[k] for k in keys])
                boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
                ratio=be['parent_balanced_intensity_mse']/ae['parent_balanced_intensity_mse']
                depth_ratios={z:be['depth_intensity_mse'][z]/ae['depth_intensity_mse'][z] for z in ae['depth_intensity_mse']}
                pairs.append(dict(reference_rank=reference,rank=rank,seed=seed,parent_count=len(keys),
                                  mse_ratio=ratio,parent_mean_difference=float(diff.mean()),
                                  parent_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                                  depth_ratios=depth_ratios,passes_rank_criterion=ratio<1 and max(depth_ratios.values())<=1.05,
                                  within_cnn_practical_5percent=ratio<=1.05 if reference==0 else None,
                                  matched_full_sample_budget=a['config']['epochs']==b['config']['epochs'] and a['samples']==b['samples']==a['config']['epochs']*7500))
    supported=[]
    for rank in (3,6):
        values=[p for p in pairs if p['reference_rank']==1 and p['rank']==rank]
        if len(values)==2 and all(p['passes_rank_criterion'] and p['matched_full_sample_budget'] for p in values):supported.append(rank)
    summary=dict(created_kst=now(),state='selection_comparison_complete',test_pixels_decoded=False,
                 supported_ranks_vs_1=supported,smallest_supported_rank=min(supported) if supported else None,
                 comparisons=pairs,
                 models=[dict(rank=k[0],seed=k[1],best_mse=v[0]['best_selection_mse'],
                              response=v[0]['response_summary'],samples=v[0]['samples'],
                              parameters=v[0]['config']['parameters'],elapsed_seconds=v[0]['elapsed_seconds'],
                              peak_allocated_GiB=v[0]['peak_allocated_GiB']) for k,v in experiments.items()],
                 interpretation='Selection analysis, not held-out test or optical LUT A/B. CNN initialization differs. Bootstrap is parent-paired, not proof of physical coherence identification.',
                 next_action='Review response and residual evidence, pre-register optional controls, then freeze all candidates before test decoding.')
    write_json(run/'selection_comparison.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('comparisons','models')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);compare(p.parse_args().run)
