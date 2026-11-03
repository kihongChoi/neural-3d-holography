"""Read-only numerical audit of the completed experiment; writes report assets only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import base64
import hashlib
import io
import json
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image
from benchtop.targets import prepare_image

RUN = ROOT/'results/paper_cnnpropcnn_20261024'
OUT = ROOT/'docs/learning-20261024'
METHODS = ['dpac', 'sgd_asm', 'sgd_cnnpropcnn']
COLORS = ['red', 'green', 'blue']
sources = []
def read(p, lines=False):
    sources.append(p)
    s = p.read_text(encoding='utf-8')
    return [json.loads(x) for x in s.splitlines() if x.strip()] if lines else json.loads(s)
def preview(a):
    intensity = np.clip(a*a, 0, 1)
    srgb = np.where(intensity <= .0031308, 12.92*intensity, 1.055*intensity**(1/2.4)-.055)
    im = Image.fromarray(np.uint8(np.clip(srgb, 0, 1)*255))
    im.thumbnail((720, 720), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    im.save(stream, format='WEBP', quality=85)
    return 'data:image/webp;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')

status = read(RUN/'completed_all.json')
assert status['status'] == 'completed'
plan = read(RUN/'plan.json')
protocol = read(RUN/'protocol.json')
summary = read(RUN/'evaluation/summary.json')
splits = plan['sources']
assert all(set(splits[a]).isdisjoint(splits[b]) for i,a in enumerate(splits) for b in list(splits)[i+1:])
assert len(splits['final']) == 8
training = read(RUN/'training.jsonl', True)
models = {}
for c in COLORS:
    rows = [r for r in training if r['channel']==c]
    best = min(rows, key=lambda r:r['val_l1'])
    models[c] = dict(**read(RUN/'models'/f'{c}_complete.json'), best_epoch=best['epoch'],
                     curve=[[r['epoch'], r['val_l1']] for r in rows])

targets = []
max_errors = {'fixed_white_mse':0., 'fitted_amplitude_mse':0., 'fitted_amplitude_scale':0.}
for source in splits['final']:
    folder = RUN/'evaluation'/Path(source).stem
    report = read(folder/'metrics.json')
    target, meta = prepare_image(Path(plan['dataset_root'])/source, protocol['configs'][0]['roi_shape'])
    assert report['source'] == source and len(report['records']) == 9
    with np.load(folder/'phases.npz') as phases:
        assert set(phases.files)==set(METHODS)
        assert all(phases[m].shape==(1080,1920,3) and phases[m].dtype==np.uint8 for m in METHODS)
    item = dict(id=Path(source).stem, metadata=meta, means={}, repeats={}, images={'target':preview(target)})
    for method in METHODS:
        rows = sorted([r for r in report['records'] if r['method']==method], key=lambda r:r['repeat'])
        assert [r['repeat'] for r in rows]==[0,1,2]
        item['repeats'][method] = []
        for row in rows:
            p = folder/f'{method}_repeat{row["repeat"]}.npz'
            with np.load(p) as f:
                a = f['amplitude_rgb'].astype(np.float64)
                assert a.shape==target.shape and np.isfinite(a).all()
                assert np.isfinite(f['sensor_intensity_rgb']).all()
            t = target.astype(np.float64)
            scale = np.sum(a*t, axis=(0,1))/np.maximum(np.sum(a*a, axis=(0,1)),1e-12)
            computed = dict(fixed_white_mse=np.mean((a-t)**2,axis=(0,1)),
                            fitted_amplitude_mse=np.mean((a*scale-t)**2,axis=(0,1)),
                            fitted_amplitude_scale=scale)
            for key, values in computed.items():
                error = float(np.max(np.abs(values-np.array([c[key] for c in row['channels']]))))
                max_errors[key] = max(max_errors[key],error)
                assert error<2e-6, (source,method,row['repeat'],key,error)
            item['repeats'][method].append({k:float(np.mean([c[k] for c in row['channels']])) for k in row['channels'][0]})
            if row['repeat']==0:
                item['images'][method] = preview(a)
                item['images'][method+'_fitted'] = preview(a*scale)
        item['means'][method] = {k:float(np.mean([r[k] for r in item['repeats'][method]])) for k in item['repeats'][method][0]}
        saved = next(r for r in summary['per_target'] if r['source']==source and r['method']==method)
        assert all(abs(saved[k]-v)<1e-12 for k,v in item['means'][method].items())
        if method!='dpac':
            for c in COLORS:
                hist = read(folder/f'{method}_{c}_optimization.json')
                assert len(hist)==1001 and hist[0]['step']==0 and hist[-1]['step']==1000
    targets.append(item)
    print('Audited',item['id'],flush=True)

forward = []
for c in read(RUN/'evaluation/forward_test.json')['results']:
    assert len(c['records'])==22
    forward.append(dict(channel=c['channel'],l1=c['l1'],
        cnn_mse=float(np.mean([r['fixed_white_mse'] for r in c['records']])),
        asm_mse=float(np.mean([r['asm']['fixed_white_mse'] for r in c['records']])),
        cnn_fitted_mse=float(np.mean([r['fitted_amplitude_mse'] for r in c['records']])),
        asm_fitted_mse=float(np.mean([r['asm']['fitted_amplitude_mse'] for r in c['records']]))))
probe=read(RUN/'evaluation/figure6_probe.json')
probe_values={}
for name in ['zero','pi_3x3']:
    probe_values[name]=[]
    for rep in range(3):
        with np.load(RUN/'evaluation'/f'probe_{name}_{rep}.npz') as f:
            a=f['amplitude_rgb']
            assert a.shape==target.shape and np.isfinite(a).all()
            probe_values[name].append(a)
assert len(probe['results'])==12
drift=read(RUN/'drift.jsonl',True)
baseline=np.array(drift[0]['mean_intensity_rgb'])
evaldrift=[r for r in drift if r['label'].startswith('evaluation_')]
assert len(evaldrift)==16
ratios=np.array([r['mean_intensity_rgb'] for r in evaldrift])/baseline-1
assert np.max(abs(ratios))<.20
means={m:{k:float(np.mean([t['means'][m][k] for t in targets])) for k in targets[0]['means'][m]} for m in METHODS}
assert all(abs(means[m][k]-summary['means'][m][k])<1e-12 for m in METHODS for k in means[m])
wins={base:{k:sum(t['means']['sgd_cnnpropcnn'][k]<t['means'][base][k] for t in targets)
             for k in ['fixed_white_mse','fitted_amplitude_mse']} for base in METHODS[:2]}
repeat_sd={m:{k:max(float(np.std([r[k] for r in t['repeats'][m]],ddof=1)) for t in targets)
              for k in ['fixed_white_mse','fitted_amplitude_mse']} for m in METHODS}
data=dict(targets=targets,means=means,models=models,forward=forward,probe=probe,wins=wins,
    repeat_sd_max=repeat_sd,drift_min_percent=(100*ratios.min(0)).tolist(),drift_max_percent=(100*ratios.max(0)).tolist(),
    completed_kst=datetime.fromtimestamp(status['updated_at'],timezone(timedelta(hours=9))).isoformat(),
    elapsed_hours=(status['updated_at']-status['started_at'])/3600,
    audit=dict(raw_capture_files=72,probe_capture_files=6,phase_sets=8,optimization_histories=48,
               independent_float64_max_abs_error=max_errors,all_final_splits_disjoint=True),
    sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dict.fromkeys(sources)])
(OUT/'paper-results-data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
evidence=dict(data)
evidence['targets']=[{k:v for k,v in t.items() if k!='images'} for t in targets]
(OUT/'paper-results-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in data.items() if k not in ('sources','targets','models')},ensure_ascii=False,indent=2))
