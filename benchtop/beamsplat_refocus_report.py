"""Compare real-camera original/background focus at the unchanged bench plane."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

root=Path('results/beamsplat_refocus_20261026/capture_background')
target=np.load(root/'target_amplitude.npy').astype(np.float64)
rois={'left_plant':(26,42,481,550),'rear_right':(1155,77,1645,471)}
results={}
for method in ('original_focus','background_focus'):
    rows=[]
    for repeat in range(3):
        a=np.load(root/f'capture_{method}_{repeat}.npz')['amplitude'].astype(np.float64)
        rows.append({name:float(np.mean((a[y0:y1,x0:x1]-target[y0:y1,x0:x1])**2))
                     for name,(x0,y0,x1,y1) in rois.items()})
    results[method]={name:float(np.mean([row[name] for row in rows])) for name in rois}
metrics=json.loads((root/'metrics.json').read_text())
results['white_change_final_comparison']=(np.array(metrics['white_end'])/np.array(metrics['white_before_comparison'])-1).tolist()
results['hashes_unchanged']=metrics['hashes_unchanged']
(root/'focus_quality.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
sheet=Image.new('RGB',(1200,1160),'#111722')
d=ImageDraw.Draw(sheet)
for ri,(name,roi) in enumerate(rois.items()):
    for ci,method in enumerate(('original_focus','background_focus')):
        im=Image.open(root/f'camera_{method}.png').crop(roi)
        im.thumbnail((580,520))
        x,y=ci*600,ri*580
        sheet.paste(im,(x+10,y+40))
        d.text((x+10,y+6),f'{name}: {method}',font=font,fill='white')
sheet.save(root/'background_detail_comparison.png')
rows=''.join(f'<tr><td>{name}</td><td>{results["original_focus"][name]:.6f}</td><td>{results["background_focus"][name]:.6f}</td></tr>' for name in rois)
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>뒤쪽 식물 재초점</title>
<style>body{background:#111722;color:#e5edf8;font:16px system-ui;max-width:1400px;margin:32px auto;padding:20px}p{line-height:1.7}img{max-width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #4c5e75}a{color:#9ed3ff}</style>
<h1>뒤쪽 식물·텃밭으로 수치 재초점</h1>
<p>원본 U.npy를 원래 3 µm 격자에서 zero-padding ASM으로 전파했다. 탐색 결과 −2.2 mm 부근에서 뒤쪽 잎과 지지대 경계가 가장 선명해져 이 값을 선택했다. 가상 센서 위치는 15 → 12.8 mm, 얇은 렌즈 식에 따른 근사 초점 깊이는 4.02 → 5.75 PLY이다. 이 깊이 표시는 원본 렌즈 설정에서 계산한 근사값이다.</p>
<p><strong>물리 SLM–카메라 관측면은 +15 mm로 유지했다.</strong> 원본 복소장을 수치 재초점한 진폭을 새로운 목표로 삼아 기존 RGB CNN과 실제 카메라 CITL로 위상을 다시 최적화했다. 따라서 기존 보정이 적용된 관측면의 초점 선택 영상이며, 물리 카메라 이동이나 3D 복소장 보존을 검증한 결과는 아니다.</p>
<img src="focus_before_after.png" alt="수치 재초점과 실촬영 비교">
<p>위 행은 목표 영상, 아래 행은 실제 카메라 촬영이다. 좌측이 기존 초점, 우측이 뒤쪽 초점이다. 기존/새 초점을 번갈아 각각 3회 촬영했다. 노출·gain·등록·고정 white와 RGB 공통 진폭 정규화는 유지했다. 표시 후 1초 및 색별 타이밍 전환 후 1초를 기다리고 5개 프레임을 평균했다.</p>
<h2>뒤쪽 영역 확대</h2><img src="background_detail_comparison.png">
<p>동일한 새 배경 목표에 대한 영역별 진폭 MSE(3회 촬영 평균, 낮을수록 일치):</p><table><tr><th>영역</th><th>기존 초점</th><th>뒤쪽 초점</th></tr>ROWS</table>
<p>수치 재초점에서는 배경이 선명해져도 실제 재생은 색 스펙클·등록 오차·광학계 대역폭의 영향을 받는다. 영역 오차는 초점만을 분리한 광학 해상도 측정이 아니다.</p>
<p><a href="../focus_sweep.png">전체 거리 탐색</a> · <a href="slm_background.png">뒤쪽 초점 SLM PNG</a> · <a href="phase_background.npy">위상 배열</a> · <a href="U_refocused.npy">재초점 복소장</a> · <a href="focus_quality.json">영역별 평가</a> · <a href="protocol.json">실험 조건</a></p></html>'''.replace('ROWS',rows)
(root/'report.html').write_text(doc,encoding='utf-8')
print(json.dumps(results,indent=2))
