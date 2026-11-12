"""Report measured A/B results and build a contact sheet of full camera frames."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .nh3d_lut_model import sha, write_json


def summarize(rows):
    before = np.asarray([row['A'] for row in rows])
    after = np.asarray([row['B'] for row in rows])
    return dict(pairs=len(rows), before=float(before.mean()), after=float(after.mean()),
                relative_mse_reduction=float(1-after.mean()/before.mean()),
                improved=int((after < before).sum()), worsened=int((after > before).sum()))


def repeatability(pairs):
    groups = {}
    for pair in pairs:
        sensors = list(pair['images']['A']['metrics'])
        components = sensors + (['equal_mean'] if pair['mode'] == 'RGB' else [])
        for component in components:
            selected = sensors if component == 'equal_mean' else [component]
            values = {method:float(np.mean([pair['images'][method]['metrics'][tag]['fixed_white_amplitude_mse']
                                             for tag in selected])) for method in ('A','B')}
            key = (pair['scene'],pair['depth_mm'],pair['mode'],component)
            groups.setdefault(key,[]).append(dict(repeat=pair['repeat'],order=pair['order'],improvement=values['A']-values['B']))
    data = []
    for (scene,z,mode,component), observations in groups.items():
        improvements = [o['improvement'] for o in observations]
        if len({tuple(o['order']) for o in observations}) != 2:
            raise ValueError('Expected both A/B orders')
        data.append(dict(scene=scene,depth_mm=z,mode=mode,component=component,repeats=observations,
            mean_improvement=float(np.mean(improvements)),sample_sd=float(np.std(improvements,ddof=1)),
            range=float(np.ptp(improvements)),all_repeats_improved=bool(np.all(np.asarray(improvements)>0))))
    return dict(metric='baseline A minus candidate B fixed-white amplitude MSE; SD across repeats of the same scene and depth',groups=data)


def camera_montage(root,plan,rgb_pairs,destination):
    # One row per depth. Preserve the camera framing and the saved pixel values.
    scenes = plan['scenes'][:2]
    depths = plan['depths_mm']
    fig,axes = plt.subplots(len(depths),2*len(scenes),figsize=(3.45*2*len(scenes),2.75*len(depths)+.65),
                            gridspec_kw={'wspace':.04,'hspace':.20},squeeze=False)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.057,right=.985,bottom=.035,top=.92)
    fig.suptitle('BeamSplat   |   RGB camera A/B' if plan['split']=='transfer' else 'Camera A/B   |   RGB integration',x=.055,y=.984,ha='left',fontsize=16,fontweight='semibold')
    for i,depth in enumerate(depths):
        for j,scene in enumerate(scenes):
            obs=next(p for p in rgb_pairs if p['scene']==scene and p['depth_mm']==depth and p['repeat']==0)
            for k,method in enumerate(('A','B')):
                ax=axes[i,2*j+k]
                entry=obs['images'][method]
                with Image.open(root/entry['relative_path']) as im:
                    im.thumbnail((720,528),Image.Resampling.LANCZOS)
                    ax.imshow(im)
                value=np.mean([entry['metrics'][tag]['fixed_white_amplitude_mse'] for tag in 'RGB'])
                label=plan.get('baseline_label','identity') if method=='A' else ('updated LUT' if plan.get('baseline_candidate') else 'distilled LUT')
                ax.set_title(f'{scene}   {method}   {label}',fontsize=9,loc='left',pad=3)
                ax.set_xlabel(f'Fixed-white amplitude MSE   {value:.4f}',fontsize=8,labelpad=2)
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
        axes[i,0].set_ylabel(f'{depth:g} mm',fontsize=11,fontweight='semibold')
    fig.savefig(destination,dpi=150,facecolor='white')
    plt.close(fig)


def report(root):
    root = Path(root).resolve()
    plan = json.loads((root/'ab_plan.json').read_text())
    state = json.loads((root/'ab/status.json').read_text())
    if state['state'] not in ('pilot_selection_ab_complete','sealed_test_ab_complete','beamsplat_transfer_ab_complete'):
        raise ValueError('A/B session is incomplete')
    if plan.get('baseline_candidate') and sha(root/'baseline_lut.npz')!=plan['baseline_sha256']:
        raise ValueError('Previous-LUT baseline changed')
    if sha(root/'rgb_lut_candidate.npz') != plan['candidate_sha256']:
        raise ValueError('Candidate changed after the experiment')
    if state['final_standby']['mode'] != 'Digital' or state['stage_end']['moving']:
        raise ValueError('Final standby or stage state mismatch')
    if not all(state['routing'].get(mode,{}).get('accepted') for mode in ('R','G','B','RGB')):
        raise ValueError('Slot isolation gate incomplete')
    if len(state['calibrations']) != 4*(1+len(plan['depths_mm'])):
        raise ValueError('Incomplete depth calibration')
    if not all(entry['registration']['accepted'] for entry in state['calibrations']):
        raise ValueError('Rejected geometry')
    peak_drift = max(abs(d['fraction']) for d in state['white_drift'])
    if peak_drift > .03:
        raise ValueError('White drift exceeded threshold')
    records = {r['name']: r for r in state['records']}
    if len(records) != len(state['records']):
        raise ValueError('Duplicate camera record')
    expected = {(scene, depth, mode, repeat) for scene in plan['scenes']
                for depth in plan['depths_mm'] for mode in ('R','G','B','RGB')
                for repeat in range(plan['repeats'])}
    actual = set()
    groups = {}; pairs = []; rgb_pairs = []
    for observation in state['comparisons']:
        scene,depth,mode,repeat = (observation[k] for k in ('scene','depth_mm','mode','repeat'))
        key = (scene,depth,mode,repeat)
        if key in actual:
            raise ValueError('Duplicate comparison')
        actual.add(key)
        pair = observation['pair']
        if pair['A']['base_rgb_sha256'] != pair['B']['base_rgb_sha256']:
            raise ValueError('Different base DPAC code within pair')
        recipe = json.loads((root/'patterns'/f'{observation["id"]:04}.json').read_text())
        if recipe['rgb_sha256'] != pair['A']['base_rgb_sha256'] or recipe['recipe']['split'] != plan['split']:
            raise ValueError('Pattern provenance mismatch')
        data = dict(scene=scene,depth_mm=depth,mode=mode,repeat=repeat,order=observation['order'],images={})
        for method in ('A','B'):
            rec = records[pair[method]['name']]
            path = root/rec['path']
            if sha(path) != rec['png_sha256']:
                raise ValueError('Camera PNG hash mismatch')
            with Image.open(path) as photo:
                if photo.size != (4096,3000) or photo.mode != 'RGB':
                    raise ValueError('Ordinary camera capture format mismatch')
            data['images'][method] = dict(relative_path=rec['path'],sha256=rec['png_sha256'],
                                          metrics=pair[method]['metrics'])
        for tag in mode:
            label = f'{mode}/{tag}'
            groups.setdefault(label,[]).append({m:pair[m]['metrics'][tag]['fixed_white_amplitude_mse'] for m in ('A','B')})
        if mode=='RGB':
            rgb_pairs.append(data)
            groups.setdefault('RGB/equal_mean',[]).append({m:float(np.mean([pair[m]['metrics'][tag]['fixed_white_amplitude_mse'] for tag in 'RGB'])) for m in ('A','B')})
        pairs.append(data)
    if actual != expected or len(pairs)*2 != plan['expected_scene_photos']:
        raise ValueError('Incomplete or unexpected measurement coverage')
    out = root/'measured_report'
    out.mkdir(exist_ok=False)
    summaries = {label:summarize(rows) for label,rows in groups.items()}
    for label, values in summaries.items():
        mode, component = label.split('/')
        observations = [row for row in pairs if row['mode'] == mode]
        for metric in ('fixed_clipped_srgb_psnr_db','fixed_clipped_srgb_ssim','mean_normalized_intensity'):
            by_method = {}
            for method in ('A','B'):
                per_observation = []
                for row in observations:
                    metrics = row['images'][method]['metrics']
                    channels = list(metrics) if component == 'equal_mean' else [component]
                    per_observation.append(np.mean([metrics[channel][metric] for channel in channels]))
                by_method[method] = float(np.mean(per_observation))
            values[metric] = by_method
    result = dict(source_run=root.name,candidate_sha256=plan['candidate_sha256'],
        baseline_label=plan.get('baseline_label','identity'),baseline_sha256=plan.get('baseline_sha256'),
        split=plan['split'],scenes=plan['scenes'],depths_mm=plan['depths_mm'],repeats=plan['repeats'],
        scene_photos=len(pairs)*2,summary=summaries,max_abs_white_drift=peak_drift,
        calibration_count=len(state['calibrations']),stage_end=state['stage_end'],laser_end=state['final_standby'],
        status_sha256=sha(root/'ab/status.json'),pairs=pairs)
    write_json(out/'results.json',result)
    camera_montage(root,plan,rgb_pairs,out/'rgb_camera_before_after.png')
    if plan['repeats'] >= 2:
        write_json(out/'repeatability.json',repeatability(pairs))
    print(json.dumps(summaries),flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
