from pathlib import Path
import csv
import hashlib
import html
import json
import re
import zipfile

root = Path(__file__).resolve().parents[3]
src = root / 'exports/neural3d_reports_20261024_201821_KST'
dst = root / 'exports/neural3d_learning_essentials_20261024'
dst.mkdir(exist_ok=False)
files = '''BENCHTOP.md
FINAL_8_TARGETS.csv
docs/learning-20261024/neural-3d-holography-learning.html
docs/learning-20261024/evidence.json
docs/guides/paper-study.html
docs/guides/paper-vs-results.html
docs/guides/evidence-snapshot.json
docs/guides/sources/NeuralHolography3D.pdf
docs/guides/sources/NeuralHolography3D.txt
configs/benchtop_rgb80.json
results/div2k_rgb_20261022/fullcolor/metrics.json
results/div2k_rgb_20261022/visualizations/summary.json
results/rgb_15mm_20261022/fullcolor/metrics.json
results/rgb_15mm_20261022/visualizations/summary.json
results/rgb80_10h_20261022/campaign/sealed_test.json
results/rgb80_10h_20261022/campaign/final_evaluations.json
results/rgb80_10h_20261022/campaign/plan.json
results/rgb80_10h_20261022/campaign/completed.json
results/rgb80_10h_20261022/acquisition.json
results/citl_diagnosis_20261023_run02/protocol.json
results/citl_diagnosis_20261023_run02/stability_summary.json
results/citl_diagnosis_20261023_run02/gradient_summary.json
results/citl_diagnosis_20261023_run02/citl_summary.json
results/citl_diagnosis_20261023_run02/report/summary.json
results/citl_diagnosis_20261023_run02/interpretation_notes.json
results/citl_diagnosis_20261023_run02/verification.json
results/paper_cnnpropcnn_20261024/protocol.json
results/paper_cnnpropcnn_20261024/status.json
results/paper_cnnpropcnn_20261024/verification/training_resume.json'''.splitlines()
for rel in files:
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((src / rel).read_bytes())

# Preserve every paragraph, equation, image and interaction. Only unavailable
# source/log links become plain text, so the portable copy has no broken links.
disabled = []
for p in dst.glob('docs/**/*.html'):
    original = p.read_text(encoding='utf-8')
    def rewrite(m):
        attrs, link, label = m.group(1), m.group(2), m.group(3)
        if link.startswith(('#', 'http:', 'https:', 'mailto:', 'data:')):
            return m.group(0)
        local = link.split('#')[0].split('?')[0]
        if not link.startswith('chatgpt-conversation:') and (p.parent / local).is_file():
            return m.group(0)
        disabled.append({'document': p.relative_to(dst).as_posix(), 'omitted_link': link})
        return '<span title="경량 묶음에서 제외한 실행 코드·로그·제작 원본">' + label + '</span>'
    modified = re.sub(r'<a\b([^>]*?)href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', rewrite, original, flags=re.S|re.I)
    # The only changes are anchor tags; all visible content and embedded data stay exact.
    strip_links = lambda t: re.sub(r'</?(?:a|span)\b[^>]*>', '', t, flags=re.I)
    assert strip_links(original) == strip_links(modified)
    p.write_text(modified, encoding='utf-8')

readme = '''Neural 3D Holography — 학습자료 제작용 핵심 자료

압축을 모두 풀고 00_START_HERE.html을 여세요.

이 묶음은 2026년 10월 24일 20:18 KST 자료 묶음에서 학습자료 제작에 필요한 항목만 추린 경량본입니다.
원 논문 PDF와 추출 텍스트, 논문 분석, 기존 통합 학습자료, 실험 방법, 핵심 결과와 근거 수치를 담았습니다.
기존 학습자료에 내장된 실측 비교 이미지·그래프·인터랙션은 그대로 유지했습니다. 수식 조판은 외부 MathJax CDN을 사용하므로 인터넷 연결이 필요할 수 있습니다.

읽기 순서
1. docs/learning-20261024/neural-3d-holography-learning.html — 전체 학습자료
2. docs/guides/paper-study.html — 원 논문 방법론
3. docs/guides/paper-vs-results.html — 원 논문과 실험 결과의 비교
4. BENCHTOP.md — 실제 장비와 실험 방법, 새 재현 실험 계획
5. FINAL_8_TARGETS.csv 및 results 아래 JSON — 실험 수치의 근거
6. docs/guides/sources/NeuralHolography3D.pdf — 원 논문

범위와 현재 결과
실험 장비는 고정 +15 mm 단일 평면입니다. 8개 깊이의 3D 재현 결과로 해석하면 안 됩니다.
완료된 소형 CNN 실험의 미사용 8장·각 방식 3회 측정에서 고정 white 정규화 진폭 MSE는 ASM 0.110674672, CNN 0.115092968, CNN+CITL 0.055738657입니다. 소형 CNN 단독은 ASM보다 나빴으며, 이 모델은 원 논문의 CNNpropCNN과 다릅니다.
새 논문 크기 CNNpropCNN 실험은 이 자료의 기준 시점에 RGB 패턴 354/1,144개 수집 단계였습니다. 학습 완료 후 최종 성능은 아직 포함되지 않았습니다. verification/training_resume.json은 학습·복원 기능 검사이며 최종 성능 결과가 아닙니다.
실험의 고정 밝기 지표와 스칼라 보정 밝기 지표는 혼용하지 마세요. 10월 23일 후속 진단은 2개 타깃·200회 학습률 비교와 국소 응답 검사이며 수렴이나 전체 Jacobian을 입증하지 않습니다.

용량을 줄이기 위해 제외한 항목
- 수천 개의 촬영·훈련·실행 로그와 개별 영수증
- 학습자료에 이미 내장된 사진·그래프의 별도 중복 파일
- 과거 실험의 단계별 재생을 모두 내장한 대형 HTML 리포트 2개
- 실행 코드, 환경 파일, 제작·검증 코드, 원시 촬영 배열, 모델 가중치
제외된 코드·로그로 연결되던 링크는 경량 사본에서 설명 텍스트로 바꿨습니다. 학습자료의 본문·수식·내장 그림·계산 동작·실험 수치는 바꾸지 않았습니다.
이 묶음은 가이드 제작과 분석용입니다. 전체 원시 데이터 백업 또는 실험 재시작용 사본은 아닙니다. 원본 파일과 진행 중인 실험은 변경하지 않았습니다.
'''
(dst / 'README.txt').write_text(readme, encoding='utf-8-sig')
nav = [
    ('docs/learning-20261024/neural-3d-holography-learning.html', '통합 인터랙티브 학습자료'),
    ('docs/guides/paper-study.html', '원 논문의 방법론 분석'),
    ('docs/guides/paper-vs-results.html', '논문과 실험 결과의 비교'),
    ('BENCHTOP.md', '실험 방법과 새 재현 실험 계획'),
    ('FINAL_8_TARGETS.csv', '완료된 실험의 최종 8장 평가 수치'),
    ('docs/guides/sources/NeuralHolography3D.pdf', '원 논문 PDF'),
    ('docs/guides/sources/NeuralHolography3D.txt', '원 논문 추출 텍스트'),
    ('results/paper_cnnpropcnn_20261024/protocol.json', '새 CNNpropCNN 실험 프로토콜'),
    ('README.txt', '포함 범위와 읽기 안내')
]
(dst / '00_START_HERE.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>학습자료 제작용 핵심 자료</title><style>body{max-width:960px;margin:36px auto;padding:20px;font:17px/1.75 system-ui,sans-serif;color:#173249}a{color:#075da5}li{margin:10px 0}pre{white-space:pre-wrap;font:inherit;background:#f0f5f8;padding:22px}h1{font-size:28px}</style><h1>학습자료 제작용 핵심 자료</h1><p>논문 · 실험 방법 · 근거 수치 · 완성된 학습자료</p><ol>''' + ''.join(f'<li><a href="{html.escape(path,quote=True)}">{html.escape(title)}</a></li>' for path,title in nav) + '</ol><pre>' + html.escape(readme) + '</pre></html>', encoding='utf-8')

missing=[]
for p in dst.rglob('*.html'):
    for link in re.findall(r'''(?:src|href)=["']([^"']+)["']''',p.read_text(encoding='utf-8')):
        if link.startswith(('#','http:','https:','data:','mailto:')): continue
        if not (p.parent / link.split('#')[0].split('?')[0]).is_file(): missing.append((str(p),link))
assert not missing, missing
(dst / 'PACKAGE_NOTES.json').write_text(json.dumps({'source_snapshot_kst':'2026-10-24T20:18:21+09:00','guide_visible_content_and_embedded_images_unchanged':True,'missing_local_links':missing,'omitted_source_links':disabled},ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[]
for p in sorted(dst.rglob('*')):
    if p.is_file():
        data=p.read_bytes();manifest.append({'path':p.relative_to(dst).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
with (dst/'FILE_MANIFEST.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(manifest)
archive=dst.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(dst.rglob('*')):
        if p.is_file():z.write(p,str(Path(dst.name)/p.relative_to(dst)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for row in manifest:
        assert hashlib.sha256(z.read(dst.name+'/'+row['path'])).hexdigest()==row['sha256']
result={'archive':str(archive),'bytes':archive.stat().st_size,'megabytes':archive.stat().st_size/1e6,'reduction_percent':100*(1-archive.stat().st_size/451926399),'files':len(manifest)+1,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified':True}
(root/'exports/latest_learning_package.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
