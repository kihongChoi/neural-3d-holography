"""Build a portable teaching guide from existing measurements; no hardware access."""
import base64
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from PIL import Image
from paper_render import render as render_paper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = ROOT / 'results/rgb80_10h_20261022/campaign'
D = ROOT / 'results/citl_diagnosis_20261023_run02'
SOURCES = []

def read(path, lines=False):
    path = Path(path)
    if path not in SOURCES:
        SOURCES.append(path)
    text = path.read_text(encoding='utf-8')
    return [json.loads(s) for s in text.splitlines() if s.strip()] if lines else json.loads(text)

def embedded(path, width=840):
    SOURCES.append(path)
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((width, width), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        im.save(stream, format='WEBP', quality=88)
    return 'data:image/webp;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')

final = []
histories = {row['evaluation']: row['history'] for row in read(C/'citl.jsonl', True) if row['evaluation'] >= 100}
for p in sorted((C/'evaluations').glob('final_*.json')):
    row = read(p)
    item = dict(id=Path(row['source']['target_id']).stem, evaluation=row['evaluation'],
                source_size=row['source']['source_size_wh'], crop=row['source']['crop_xywh'],
                termination=row['citl_termination'], means={}, repeats={}, images={}, history=[])
    for method in ('asm', 'cnn', 'cnn_citl'):
        values = [x['amplitude_mse'] for x in row['repeats'] if x['variant'] == method]
        assert len(values) == 3
        item['repeats'][method] = values
        item['means'][method] = float(np.mean(values))
        item['images'][method] = embedded(p.with_name(p.stem+f'_{method}_camera.png'))
    item['images']['target'] = embedded(p.with_name(p.stem+'_target.png'))
    item['history'] = [[r['step'], r['amplitude_mse']] for r in histories[row['evaluation']]]
    final.append(item)
assert len(final) == 8
means = {m:float(np.mean([x['means'][m] for x in final])) for m in ('asm', 'cnn', 'cnn_citl')}
test = read(C/'sealed_test.json')
prediction = {ch:{m:float(np.mean([x[m+'_mse'] for x in test if x['channel']==ch])) for m in ('asm','cnn')} for ch in ('red','green','blue')}
gradient = read(D/'gradient_summary.json')
gradstats = {m:{base:float(np.median([x['models'][m]['cosine'] for x in gradient if base=='all' or x['base']==base])) for base in ('all','initial','old_best')} for m in ('asm','cnn')}
diag = read(D/'citl_summary.json')
for x in diag:
    folder = D/f"citl_{x['id']}_lr{x['lr']:g}"
    x['history'] = [[r['step'],r['amplitude_mse']] for r in read(folder/'history.jsonl',True)]
    x['images'] = {key:embedded(folder/f'{key}.png') for key in ('initial','best_remeasured','last_remeasured')}
small = read(ROOT/'results/div2k_rgb_20261022/fullcolor/metrics.json')
small_means = {m:float(np.mean([r['amplitude_mse'] for r in small['repeat_validation'] if r['variant']=='new_'+m])) for m in ('asm','cnn','cnn_citl')}
acq = read(C/'acquisition.jsonl',True)
plan = read(C/'plan.json')
training = read(C/'training.jsonl',True)
complete = read(C/'completed.json')
data = dict(final=final,means=means,prediction=prediction,gradient=gradient,gradient_stats=gradstats,
            diagnosis=diag,stability=read(D/'stability_summary.json'),diagnostic_report=read(D/'report/summary.json'),
            small_means=small_means,plan={k:v for k,v in plan.items() if k!='dataset_root'},
            acquisition_counts=dict(Counter(r['split'] for r in acq)),
            training=[{k:r.get(k) for k in ('epoch','train_mse','val_mse')} for r in training if r['channel']=='rgb'],
            active_hours=complete['active_seconds']/3600,
            completed_kst=datetime.fromtimestamp(complete['finished_at'],timezone(timedelta(hours=9))).isoformat(),
            conditions=read(ROOT/'results/rgb80_10h_20261022/acquisition.json')['channels'])
assert data['acquisition_counts']=={'val':8,'test':8,'train':784}
assert all(set(plan[a]).isdisjoint(plan[b]) for i,a in enumerate(('train','val','test','tracking','final')) for b in ('train','val','test','tracking','final')[i+1:])
assert sum(x['means']['cnn']<x['means']['asm'] for x in final)==2
assert all(x['means']['cnn_citl'] < x['means']['asm'] for x in final)
assert sum(x['termination']['reason']=='iteration_limit' for x in final)==7
paper_data, paper_section = render_paper()
data['paper'] = paper_data
evidence = {k:v for k,v in data.items() if k not in ('final','diagnosis','conditions')}
evidence['paper'] = json.loads((HERE/'paper-results-evidence.json').read_text(encoding='utf-8'))
evidence['final'] = [{k:v for k,v in x.items() if k!='images'} for x in final]
evidence['diagnosis'] = [{k:v for k,v in x.items() if k!='images'} for x in diag]
evidence['source_files'] = [{'path':str(p.relative_to(ROOT)).replace('\\','/'), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(SOURCES)]
(HERE/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
template=(HERE/'template.html').read_text(encoding='utf-8').replace('__PAPER_SECTION__',paper_section).replace('__STYLE__',(HERE/'guide.css').read_text(encoding='utf-8')).replace('__SCRIPT__',(HERE/'guide.js').read_text(encoding='utf-8')+'\n'+(HERE/'paper-guide.js').read_text(encoding='utf-8'))
html=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
(HERE/'neural-3d-holography-learning.html').write_text(html,encoding='utf-8')
print(json.dumps(dict(output=str(HERE/'neural-3d-holography-learning.html'),bytes=len(html.encode()),means=means,
                     small_means=small_means,prediction=prediction,gradient_stats=gradstats,
                     active_hours=data['active_hours'],completed=data['completed_kst']),ensure_ascii=False,indent=2))
