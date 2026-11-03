"""Read-only report snapshot; never controls hardware or modifies experiment sources."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
import csv
import hashlib
import html
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
start = datetime.now(KST)
name = 'neural3d_reports_' + start.strftime('%Y%m%d_%H%M%S_KST')
OUT = ROOT / 'exports' / name
OUT.mkdir(exist_ok=False)
CURRENT = 'paper_cnnpropcnn_20261024'
status_bytes = (ROOT / 'results' / CURRENT / 'status.json').read_bytes()
status = json.loads(status_bytes)
cutoff = status.get('sample', 0)
records, excluded = [], []

def include(p):
    rel = p.relative_to(ROOT)
    parts = rel.parts
    ext = p.suffix.lower()
    if '__pycache__' in parts:
        return False, 'Python cache'
    if parts[0] == 'docs':
        return True, '논문 분석·학습자료·참조 논문·제작 원본'
    if len(parts) == 1:
        return True, '실험 방법·저자 코드·환경 명세'
    if parts[0] in {'benchtop', 'configs', 'scripts', 'tests'}:
        return ext in {'.py', '.json', '.html', '.ps1', '.js', '.cjs', '.css', '.md'}, '실험 구현·설정·검증 코드'
    if parts[0] != 'results':
        return False, '보고서 범위 밖'
    if ext in {'.npz', '.npy', '.pt', '.pth', '.lock', '.pid', '.previous'}:
        return False, '원시 배열·모델 가중치·실행 상태 제어 파일'
    if {'live', 'preview_cache', 'models'}.intersection(parts):
        return False, '중복 실시간 캐시 또는 모델'
    if parts[1] == CURRENT and {'pairs', 'patterns'}.intersection(parts):
        if ext != '.json' or not p.stem.isdigit() or int(p.stem) >= cutoff:
            return False, '현재 실행의 스냅샷 이후 또는 대용량 원시 데이터'
    if ext in {'.json', '.jsonl', '.csv', '.log', '.txt', '.md', '.pdf', '.py', '.css', '.js', '.cjs'}:
        return True, '실험 수치·계획·보정·진행 및 검증 기록'
    if ext == '.html':
        return True, '작성된 실험 리포트 또는 모니터 원본'
    if ext in {'.png', '.jpg', '.jpeg', '.svg'}:
        low = p.name.lower()
        if parts[1] in {'probe', 'display_check', 'goal_display_recheck_20261022', 'citl_diagnosis_20261023'}:
            return False, '화면 점검 이미지 또는 미완료 초기 진단의 중복 이미지'
        if any(x in low for x in ['slm', 'display_', 'computed_', 'screen', 'monitor_', 'preflight']):
            return False, '화면 점검 또는 위상 원시 이미지'
        if 'visualizations' in parts and re.match(r'(red|green|blue)_(epoch|step)_', low):
            return False, 'HTML 리포트에 이미 내장된 단계별 이미지의 별도 원본'
        if 'fullcolor' in parts and re.match(r'.*step_\d+', low):
            return False, 'HTML 리포트에 이미 내장된 단계별 이미지의 별도 원본'
        if {'patterns', 'pairs'}.intersection(parts):
            return False, '학습 입력 원시 이미지'
        return True, '실측 비교·최종 평가·진단 그래프'
    return False, '보고서 범위 밖의 파일 형식'

paths = [p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {'.py', '.md', '.yml', '.toml', '.lock'} or p.name in {'LICENSE', '.python-version'})]
for folder in ['docs', 'benchtop', 'configs', 'scripts', 'tests', 'results']:
    paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())

for p in sorted(paths):
    rel = p.relative_to(ROOT).as_posix()
    selected, reason = include(p)
    if not selected:
        excluded.append((rel, reason))
        continue
    if p.name in {'notification_banner_backup.json', 'notification_smartoptout_backup.json'}:
        # Configuration provenance, not a request to apply it on another machine.
        reason = '실험대 설정 변경의 복구용 기록'
    size = p.stat().st_size
    if rel == f'results/{CURRENT}/status.json':
        data = status_bytes
    else:
        with p.open('rb') as f:
            data = f.read(size)
    if p.suffix == '.jsonl' and data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    if rel == f'results/{CURRENT}/acquisition.jsonl':
        data = b''.join(line + b'\n' for line in data.splitlines() if json.loads(line)['id'] < cutoff)
    if p.suffix == '.json':
        json.loads(data)
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    records.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'category': reason})

finished = datetime.now(KST)
final = json.loads((OUT / 'results/rgb80_10h_20261022/campaign/final_evaluations.json').read_text(encoding='utf-8'))
methods = ['asm', 'cnn', 'cnn_citl']
means = {m: sum(x['amplitude_mse'] for row in final for x in row['repeats'] if x['variant'] == m) / sum(x['variant'] == m for row in final for x in row['repeats']) for m in methods}
with (OUT / 'FINAL_8_TARGETS.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['target_id', 'ASM_fixed_white_amplitude_MSE', 'CNN_fixed_white_amplitude_MSE', 'CNN_CITL_fixed_white_amplitude_MSE', 'repeats_per_method'])
    for row in final:
        vals = [sum(x['amplitude_mse'] for x in row['repeats'] if x['variant'] == m) / sum(x['variant'] == m for x in row['repeats']) for m in methods]
        w.writerow([row['source']['target_id'], *vals, 3])
    w.writerow(['MEAN', *[means[m] for m in methods], 3])

def a(path, text):
    return '<a href="' + html.escape(path, quote=True) + '">' + html.escape(text) + '</a>'

links = [
    ('docs/learning-20261024/neural-3d-holography-learning.html', '통합 인터랙티브 학습자료 — 논문·실험·후속 진단 해설'),
    ('docs/guides/paper-study.html', '원 논문의 방법론 분석'),
    ('docs/guides/paper-vs-results.html', '원 논문과 현재 실험의 비교·해석'),
    ('BENCHTOP.md', '전체 실험 방법·장비·절차·재현 조건 기록'),
    ('docs/guides/sources/NeuralHolography3D.pdf', '참조 원 논문 PDF'),
    ('results/rgb_15mm_20261022/report.html', '10월 22일 초기 단일 평면 RGB 실험 리포트'),
    ('results/div2k_rgb_20261022/report.html', '10월 22일 DIV2K 영상 실험 리포트'),
    ('results/rgb80_10h_20261022/campaign/report/rgb_measured_montage_20261023.png', '10월 22–23일 장시간 실험의 실측 비교'),
    ('results/rgb80_10h_20261022/campaign/final_evaluations.json', '장시간 실험 최종 8장 평가의 원 수치'),
    ('results/citl_diagnosis_20261023_run02/report/summary.json', '10월 23일 안정성·유한차분·학습률 진단 결과'),
    ('results/citl_diagnosis_20261023_run02/report/stability_montage.png', '고정 위상 반복 측정의 안정성'),
    ('results/citl_diagnosis_20261023_run02/report/gradient_montage.png', '실측 응답과 모델 응답의 비교'),
    ('results/citl_diagnosis_20261023_run02/report/learning_rate_curves.png', '학습률별 최적화 곡선'),
    (f'results/{CURRENT}/protocol.json', '10월 24일 새 CNNpropCNN 재현 실험의 고정 프로토콜'),
    (f'results/{CURRENT}/status.json', '새 실험의 수집 시점 상태'),
    (f'results/{CURRENT}/verification/training_resume.json', '새 모델의 실제 측정 데이터 기반 학습·복원 기능 검증'),
    ('FINAL_8_TARGETS.csv', '완료된 장시간 실험의 최종 8장 수치 요약표'),
    ('FILE_MANIFEST.csv', '포함 파일 전체 목록·크기·SHA-256'),
    ('EXCLUDED_FILES.csv', '별도 원시 데이터·중복 이미지 등 제외 목록'),
]
for path, _ in links:
    if path not in {'FILE_MANIFEST.csv', 'EXCLUDED_FILES.csv'}:
        assert (OUT / path).is_file(), path

summary = f'''Neural 3D Holography 실험 자료 통합 묶음
기준 시점: {start:%Y-%m-%d %H:%M:%S} KST
파일 읽기 완료: {finished:%Y-%m-%d %H:%M:%S} KST

압축을 모두 해제한 후 00_START_HERE.html을 여세요.
문서·그림·수식 원문은 로컬 파일로 열 수 있습니다. 인터랙티브 학습자료의 수식 조판은 MathJax CDN을 사용하므로 완전한 수식 렌더링에는 인터넷 연결이 필요할 수 있습니다.

실험 범위: 고정 +15 mm 단일 평면. 8개 깊이의 3D 재현, 깊이 일반화 또는 ADMM 성능을 입증한 자료가 아닙니다.

완료된 장시간 실험: 새 평가 원본 8장, 방식별 3회 실측, 고정 white 정규화 진폭 MSE.
ASM {means['asm']:.9f} / 소형 CNN {means['cnn']:.9f} / CNN+CITL {means['cnn_citl']:.9f}.
CNN 단독은 ASM보다 평균 {(means['cnn']/means['asm']-1)*100:.2f}% 높고, CNN+CITL은 ASM보다 {(1-means['cnn_citl']/means['asm'])*100:.2f}% 낮습니다. 해당 소형 CNN은 원 논문 크기의 복소장 CNNpropCNN과 다릅니다.
후속 진단은 측정 안정성·광량 변화·모델 응답·학습률을 분리해 해석합니다. 학습률 비교는 2개 타깃·200회 예산이며, 수렴이나 일반화를 입증하지 않습니다.

새 CNNpropCNN 실험: status={status.get('status')}, stage={status.get('stage')}, RGB 패턴 {cutoff}/{status.get('total')}개 시점의 기록입니다. 색별 100 DPAC+1,000 SGD 학습 쌍, 별도 검증/테스트 각각 22쌍 계획입니다. 각 RGB 패턴은 세 색의 측정을 묶은 수입니다.
이 시점은 수집 단계이며 새 모델의 최종 성능 결과는 아직 없습니다. 기능 검증 수치를 최종 성능으로 해석하지 마세요. 학습 완료 후 미사용 8장 DPAC/SGD-ASM/SGD-CNNpropCNN 재촬영과 유한차분 응답 평가가 예정되어 있습니다. 고정 밝기 지표와 스칼라 보정 밝기 지표는 따로 평가합니다.

원본 리포트는 생성 당시 내용을 보존했습니다. 특히 results/rgb80_10h_20261022/campaign/report/index.html과 summary.json에는 초기 진행 상태가 남아 있습니다. 이 실험의 최종 상태와 수치는 campaign/completed.json, final_evaluations.json 및 통합 학습자료를 기준으로 읽어 주세요. 과거 오류·중단 로그는 이력이며 현재 실패를 뜻하지 않습니다.

이 압축은 문서·리포트·증거 자료의 읽기 전용 사본입니다. 진행 중인 실험을 멈추거나 소스·조건을 변경하지 않았습니다. 파일들은 위 시간 구간에 순차적으로 읽었으므로 전체 실행 상태의 원자적 백업은 아닙니다. 현재 실행의 status 및 수집 영수증/목록은 위 패턴 수 기준으로 잘랐으며 나머지 로그는 파일별 읽기 시점의 완성된 기록입니다. 이 사본에서 실험을 재시작하지 마세요.

포함: 작성된 HTML/Markdown 리포트와 학습자료, 원 논문 PDF, 분석 근거 JSON, 실측 비교 이미지·그래프, 실험 계획/설정/보정 메타데이터/로그, 참조 구현과 소스 스냅샷, 검증 기록.
제외: 대용량 NPY/NPZ 원시 배열, PT 모델 가중치, 별도 DIV2K 원본 데이터셋, 실시간 미리보기 캐시, 화면 점검 스크린샷, HTML 리포트에 이미 내장된 단계별 이미지의 일부 별도 원본. 문서의 데이터 파일 경로는 원래 실험대의 출처 기록으로 남아 있습니다. 전체 원시 실험 백업은 아닙니다.
HTML 리포트의 사진과 단계별 재생은 내장된 자료를 사용합니다. 실행 모니터 HTML은 서버·실시간 데이터에 의존하므로 이 묶음에서는 status.json과 로그를 읽어 주세요.
'''
(OUT / 'README.txt').write_text(summary, encoding='utf-8-sig')
body = '\n'.join('<li>' + a(path, title) + '</li>' for path, title in links)
(OUT / '00_START_HERE.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Neural 3D Holography · 실험 자료 모음</title><style>body{font:17px/1.75 system-ui,sans-serif;color:#172c3b;background:#f4f7fa;margin:0}main{max-width:1000px;margin:40px auto;padding:28px;background:white;border-radius:12px}a{color:#075da5}li{margin:10px 0}pre{white-space:pre-wrap;font:inherit;background:#edf3f8;padding:22px;border-radius:8px}h1{font-size:30px}h2{font-size:22px}@media(max-width:650px){main{margin:0;padding:18px}h1{font-size:24px}}</style><main><h1>Neural 3D Holography 실험 자료 모음</h1><p>논문 분석 · 실험 방법 · 실측 리포트 · 진행 상황</p><h2>자료 열기</h2><ol>''' + body + '</ol><h2>읽기 안내와 확인된 결과</h2><pre>' + html.escape(summary) + '</pre></main></html>', encoding='utf-8')
snapshot = {'snapshot_start_kst': start.isoformat(), 'read_finished_kst': finished.isoformat(), 'current_run': CURRENT, 'current_status': status, 'raw_data_and_weights_included': False, 'scope': 'fixed +15 mm single plane', 'source_files': len(records), 'source_bytes': sum(r['bytes'] for r in records)}
(OUT / 'SNAPSHOT.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT / 'EXCLUDED_FILES.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['path', 'reason']); w.writerows(excluded)

# Verify all relative links in the authored guides and index against the package.
missing = []
for p in [OUT / '00_START_HERE.html', *OUT.glob('docs/**/*.html')]:
    for link in re.findall(r'''(?:src|href)=["']([^"']+)["']''', p.read_text(encoding='utf-8')):
        if link.startswith(('#', 'data:', 'http:', 'https:', 'mailto:', 'chatgpt-conversation:')):
            continue
        target = link.split('#')[0].split('?')[0]
        if target in {'FILE_MANIFEST.csv'}:
            continue
        if target and not (p.parent / target).is_file():
            missing.append([str(p.relative_to(OUT)), link])
assert not missing, missing
(OUT / 'PACKAGE_VERIFICATION.json').write_text(json.dumps({'relative_guide_links_missing': missing, 'json_parse_check': 'passed', 'source_files': len(records), 'excluded_files': len(excluded)}, ensure_ascii=False, indent=2), encoding='utf-8')
known = {r['path'] for r in records}
for p in OUT.iterdir():
    if p.is_file() and p.name not in known:
        data = p.read_bytes()
        records.append({'path': p.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'category': '이번 묶음의 안내·요약·검증'})
with (OUT / 'FILE_MANIFEST.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['path', 'bytes', 'sha256', 'category']); w.writeheader(); w.writerows(records)

archive = OUT.with_suffix('.zip')
print(json.dumps({'phase': 'compressing', 'folder': str(OUT), 'files': len(records)+1, 'uncompressed_mib': sum(r['bytes'] for r in records)/1024**2, 'current_patterns': cutoff}, ensure_ascii=False), flush=True)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, str(Path(name) / p.relative_to(OUT)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for record in records:
        payload = z.read(name + '/' + record['path'])
        assert hashlib.sha256(payload).hexdigest() == record['sha256'], record['path']
h = hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda: f.read(1024*1024), b''):
        h.update(block)
archive.with_suffix('.zip.sha256').write_text(h.hexdigest() + '  ' + archive.name + '\n', encoding='ascii')
result = {'archive': str(archive), 'bytes': archive.stat().st_size, 'mib': archive.stat().st_size/1024**2, 'sha256': h.hexdigest(), 'files': len(records)+1, 'verified': True, 'snapshot_kst': start.isoformat(), 'current_patterns': cutoff}
(ROOT / 'exports/latest_package.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False), flush=True)
