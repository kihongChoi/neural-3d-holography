"""Independent recaptures after a flagged first evaluation; no phase changes."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .hardware import Bench
from .calibration import Rectifier, channel_intensity
from .paper_campaign import sha
from .paper_model import image_metrics
from .rgb_run import CHANNELS, read_json, select_channel, srgb_preview

root = Path('results/beamsplat_20261026_eval01')
out = root/'validation_repeat01'
out.mkdir(exist_ok=False)
protocol = read_json(root/'protocol.json')
configs = protocol['configs']
trained = Path(protocol['trained'])
rects = [Rectifier(trained/ch/'calibration', c) for ch,c in zip(CHANNELS, configs)]
base = dict(configs[1], settle_s=1.0)
names = ['dpac', 'asm', 'cnn', 'citl', 'raw_dph']
phases = {name:np.load(root/f'phase_{name}.npy') for name in names}
target = np.load(root/'target_amplitude.npy')
for file in ['target.png','target_amplitude.npy']+[f'history_{ch}_citl.json' for ch in CHANNELS]:
    shutil.copy2(root/file, out/file)
protocol.update(validation_policy='Independent 3-repeat evaluation of the same frozen phases after initial evaluation flagged CNN outlier and end-reference drift. 1s display dwell, 1s after each color timing change, 5-frame camera averages. White reference brackets each randomized round. No post-hoc renormalization.',
    frozen_phase_sha256={name:sha(root/f'phase_{name}.npy') for name in names},
    original_evaluation=str(root.resolve()), acquisition_config=base)
(out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
records, captures, drift_records = [], {k:[] for k in names}, []
initial = (np.load(root/'drift_start.npz')['amplitude']**2).mean((0,1))
def save(path, value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def progress(**data):
    row=dict(time=time.time(), **data)
    save(out/'status.json',row)
    print(json.dumps(row),flush=True)
with Bench(base) as bench:
    def capture(p, label):
        bench.show(p)
        amps, sensors = [], []
        for c,r in zip(configs,rects):
            select_channel(bench,c)
            bench.settle()
            rgb = bench.capture(frames=5)
            amps.append(r.amplitude(rgb))
            sensors.append(channel_intensity(rgb,c))
        with (out/'frames.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(label=label,frames=bench.frame_metadata))+'\n')
        bench.frame_metadata.clear()
        return np.stack(amps,-1),np.stack(sensors,-1)
    def white(label):
        a,s = capture(np.zeros_like(phases['citl']),label)
        value=(a*a).mean((0,1))
        relative=value/initial-1
        np.savez_compressed(out/f'{label}.npz',amplitude=a,sensor=s)
        row=dict(label=label,mean_intensity_rgb=value.tolist(),change_from_original=relative.tolist())
        drift_records.append(row)
        save(out/'white_checks.json',drift_records)
        progress(stage='white_check',**row)
        if np.any(abs(relative)>.2):
            raise RuntimeError('White reference differs more than 20%; preserve and investigate')
        return value
    rng=np.random.default_rng(2026102602)
    for repeat in range(3):
        start=white(f'white_{repeat}_before')
        for name in rng.permutation(names):
            progress(stage='capture',method=str(name),repeat=repeat)
            a,s=capture(phases[name],f'{name}_{repeat}')
            np.savez_compressed(out/f'capture_{name}_{repeat}.npz',amplitude=a,sensor=s)
            captures[name].append(a)
            metrics=[image_metrics(torch.from_numpy(a[...,ci]),torch.from_numpy(target[...,ci])) for ci in range(3)]
            records.append(dict(method=str(name),repeat=repeat,channels=metrics))
            save(out/'metrics.json',dict(records=records))
        end=white(f'white_{repeat}_after')
        if np.any(abs(end/start-1)>.1):
            raise RuntimeError('Within-round white reference changed more than 10%')
    summary={}
    for name,arrays in captures.items():
        selected=[r for r in records if r['method']==name]
        summary[name]={key:float(np.mean([c[key] for r in selected for c in r['channels']]))
                       for key in selected[0]['channels'][0] if key!='fitted_amplitude_scale'}
        summary[name]['fixed_white_mse_repeat_sd']=float(np.std([
            np.mean([c['fixed_white_mse'] for c in r['channels']]) for r in selected],ddof=1))
        mean=np.mean(arrays,axis=0)
        np.save(out/f'mean_{name}.npy',mean)
        Image.fromarray(np.uint8(np.clip(srgb_preview(mean)*255,0,255))).save(out/f'fixed_{name}.png')
        scales=np.array([(mean[...,ci]*target[...,ci]).mean()/max((mean[...,ci]**2).mean(),1e-12) for ci in range(3)])
        Image.fromarray(np.uint8(np.clip(srgb_preview(mean*scales)*255,0,255))).save(out/f'fitted_{name}.png')
    best=min(('dpac','asm','cnn','citl'),key=lambda k:summary[k]['fixed_white_mse'])
    bench.show(phases[best])
    save(out/'summary.json',dict(methods=summary,best_fixed_white_method=best,
        drift_fraction=(end/initial-1).tolist(),excessive_drift=False,
        white_checks=drift_records,hashes_unchanged=all(sha(Path(p))==h for p,h in protocol['hashes'].items()),
        phase_hashes_unchanged=all(sha(root/f'phase_{k}.npy')==h for k,h in protocol['frozen_phase_sha256'].items())))
    progress(stage='completed',best=best,summary=summary)
