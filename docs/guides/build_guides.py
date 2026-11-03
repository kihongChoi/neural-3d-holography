"""Build two offline Korean guides from preserved measurements. No hardware access."""
import base64
import hashlib
import io
import json
import statistics as st
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMP = ROOT / 'results/rgb80_10h_20261022/campaign'
DIAG = ROOT / 'results/citl_diagnosis_20261023_run02'
SOURCES = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def lines(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(raw).hexdigest()
    return [json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]


def image_data(path, width=560):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
    im = Image.open(path).convert('RGB')
    im.thumbnail((width, width))
    output = io.BytesIO()
    im.save(output, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


completed = read(CAMP / 'completed.json')
final = read(CAMP / 'final_evaluations.json')
sealed = read(CAMP / 'sealed_test.json')
plan = read(CAMP / 'plan.json')
pool = read(CAMP / 'pool.json')
acquisition = lines(CAMP / 'acquisition.jsonl')
training = lines(CAMP / 'training.jsonl')
termination = lines(CAMP / 'training_termination.jsonl')
diagnosis = read(DIAG / 'report/summary.json')
verification = read(DIAG / 'verification.json')
citl = lines(CAMP / 'citl.jsonl')
config = [read(CAMP.parent / c / 'config.json') for c in ['red', 'green', 'blue']]
variants = ['asm', 'cnn', 'cnn_citl']
items = []
for record in final:
    eid = record['evaluation']
    curve = next(x for x in citl if x.get('evaluation') == eid and x.get('final'))
    stop = record['citl_termination']
    points = [x for x in curve['history'] if x['step'] % 10 == 0 or x['step'] in (stop['best_step'], stop['steps'])]
    items.append(dict(id=Path(record['source']['target_id']).stem,
        # Windows path identity is intentionally parsed explicitly for portability.
        target_id=record['source']['target_id'], evaluation=eid,
        mse={v:st.mean(x['amplitude_mse'] for x in record['repeats'] if x['variant'] == v) for v in variants},
        repeats=record['repeats'], stop={k:stop[k] for k in ['reason','steps','best_step','best_mse']},
        curve=[[x['step'], x['amplitude_mse']] for x in points],
        images={v:image_data(CAMP / 'evaluations' / f'final_{eid:04d}_{"target" if v == "target" else v+"_camera"}.png') for v in ['target']+variants}))
    items[-1]['id'] = record['source']['target_id'].replace('\\','/').split('/')[-1].split('.')[0]
mean = {v:st.mean(x['mse'][v] for x in items) for v in variants}
sealed_mean = {v:st.mean(x[v+'_mse'] for x in sealed) for v in ['asm','cnn']}
diag_items = []
for d in diagnosis['conditions']:
    history = lines(DIAG / f"citl_{d['id']}_lr{d['lr']}" / 'history.jsonl')
    diag_items.append(dict(id=d['id'], lr=d['lr'], best_step=d['best_step'],
        mse={v:st.mean(x['amplitude_mse'] for x in d['repeat_rows'] if x['variant']==v) for v in ['best','last']},
        curve=[[x['step'],x['amplitude_mse']] for x in history]))

splits = ['train','val','test','tracking','final']
overlaps = {f'{a}/{b}':len(set(plan[a]) & set(plan[b])) for i,a in enumerate(splits) for b in splits[i+1:]}
assert all(v == 0 for v in overlaps.values())
assert len(items) == 8 and len(sealed) == 24
assert all(Counter(x['variant'] for x in i['repeats']) == dict.fromkeys(variants,3) for i in items)

data = dict(as_of='2026-10-24 KST', mean=mean, sealed_mean=sealed_mean, items=items,
    diagnosis=diag_items, diagnostic_summary=diagnosis, config=config,
    finish_kst=datetime.fromtimestamp(completed['finished_at'],timezone(timedelta(hours=9))).isoformat(),
    active_hours=completed['active_seconds']/3600,
    samples=dict(acquired=len(acquisition), counts=dict(Counter(x['split'] for x in acquisition)), retained=len(pool)),
    split_counts={s:len(plan[s]) for s in splits}, split_overlaps=overlaps,
    joint_training_termination=termination, source_verification=verification,
    aggregation='Per capture amplitude MSE, averaged over 3 captures per image and 8 images. Not MSE of the averaged image.',
    limits=['8 final image identities, 3 recaptures per method; not 24 independent targets.',
            'Sealed test measures forward prediction against stored captures, not target reconstruction.',
            '0846 and 0898 became diagnostic targets after the original final evaluation.'])
audit = {k:v for k,v in data.items() if k != 'items'}
audit['items'] = [{k:v for k,v in x.items() if k != 'images'} for x in items]
for path in ['benchtop/optics.py','benchtop/joint.py','benchtop/campaign.py','benchtop/rgb_training.py','prop_model.py','algorithms.py','BENCHTOP.md','docs/guides/sources/NeuralHolography3D.pdf']:
    p=ROOT/path
    SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest()
audit['sources_sha256']=SOURCES
(HERE/'evidence-snapshot.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

css = (HERE/'guide.css').read_text(encoding='utf-8')
js = (HERE/'guide.js').read_text(encoding='utf-8')
for slug,title,subtitle,other in [
    ('paper-vs-results','논문과 현재 실험의 차이','실험 근거 · 결과 해석 · 다음 검증','paper-study'),
    ('paper-study','Neural 3D Holography 학습 가이드','복소장 → 전파 모델 → RGBD → 검증','paper-vs-results')]:
    body=(HERE/(slug+'.body.html')).read_text(encoding='utf-8')
    embedded = json.dumps(data if slug == 'paper-vs-results' else {},ensure_ascii=False).replace('</', '<\\/')
    out=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head>
<body data-guide="{slug}"><header><div class="eyebrow">NEURAL 3D HOLOGRAPHY / RESEARCH NOTES</div><h1>{title}</h1><p>{subtitle}</p><div class="meta">작성 기준 2026.10.24 KST · 실측 기록 10.22–10.23 · 한국어 학습 문서</div>
<nav class="toolbar" aria-label="문서 도구"><a href="{other}.html">다른 가이드 열기 ↗</a><button type="button" id="mode" aria-pressed="false">비유 함께 보기</button><button type="button" id="expand">풀이 모두 펼치기</button><button type="button" id="print">인쇄 / PDF 저장</button></nav>
<p class="small">인터넷 없이 설명·수식·실습이 작동합니다. 논문 원문과 저장소 근거 링크는 별도 파일 또는 인터넷 연결이 필요합니다. 비유는 이해를 돕는 설명이며 물리적 등가 모델이 아닙니다.</p></header>
<main>{body}</main><footer>Choi, Gopakumar, Peng, Kim &amp; Wetzstein · ACM TOG 40(6), Article 240, 2021 · <a href="https://doi.org/10.1145/3478513.3480542">DOI</a><br>이 문서는 원 논문을 대체하지 않는 독립적인 학습·실험 해설입니다. 후속 진단 수치는 별도 세션이며 원본 최종 평가와 합산하지 않았습니다.<br><a href="evidence-snapshot.json">집계값·계산 조건·원본 SHA-256</a> · <a href="sources/NeuralHolography3D.pdf">보관한 논문 PDF</a></footer>
<script id="evidence-data" type="application/json">{embedded}</script><script>{js}</script></body></html>'''
    (HERE/(slug+'.html')).write_text(out,encoding='utf-8')
print(json.dumps(dict(outputs=['paper-vs-results.html','paper-study.html'],mean=mean,sealed_mean=sealed_mean,
    cnn_change_percent=100*(mean['cnn']/mean['asm']-1),citl_reduction_percent=100*(1-mean['cnn_citl']/mean['asm']),
    prediction_reduction_percent=100*(1-sealed_mean['cnn']/sealed_mean['asm'])),ensure_ascii=False))
