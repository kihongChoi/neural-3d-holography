"""Numerical audit and local visual report of the real BeamSplat experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def report(root):
    root = Path(root).resolve()
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    target = np.load(root/'target_amplitude.npy').astype(np.float64)
    audit = {}
    for method, metrics in summary['methods'].items():
        values = []
        for repeat in range(3):
            p = np.load(root/f'capture_{method}_{repeat}.npz')['amplitude'].astype(np.float64)
            values.append(float(np.mean((p-target)**2)))
        audit[method] = dict(float64_fixed_mse=np.mean(values), repeat_values=values,
                             summary_abs_difference=abs(np.mean(values)-metrics['fixed_white_mse']))
    assert max(a['summary_abs_difference'] for a in audit.values()) < 1e-6
    (root/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    rows = []
    names = dict(dpac='DPAC (adapted)', asm='SGD ASM', cnn='Trained CNNpropCNN', citl='CNN + camera CITL', raw_dph='Original DPH (unmatched)')
    for method in ('dpac', 'asm', 'cnn', 'citl', 'raw_dph'):
        m = summary['methods'][method]
        rows.append(f"<tr><td>{names[method]}</td><td>{m['fixed_white_mse']:.6f} ± {m['fixed_white_mse_repeat_sd']:.6f}</td><td>{m['fitted_amplitude_psnr']:.2f}</td><td>{m['fitted_clipped_srgb_ssim']:.4f}</td></tr>")
    best = summary['best_fixed_white_method']
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 25)
    for fitted in (False, True):
        sheet = Image.new('RGB', (2520, 540), '#101723')
        d = ImageDraw.Draw(sheet)
        items = [('Target', root/'target.png'), ('Before: SGD ASM', root/f'{"fitted" if fitted else "fixed"}_asm.png'),
                 (f'After: {names[best]}', root/f'{"fitted" if fitted else "fixed"}_{best}.png')]
        for i, (label, file) in enumerate(items):
            im = Image.open(file)
            im.thumbnail((820, 470))
            x = i*840+10
            sheet.paste(im, (x, 50))
            d.text((x, 10), label, fill='white', font=font)
        sheet.save(root/f'before_after_{"fitted" if fitted else "fixed"}.png')
    methods = ''.join(f'<option value="{m}">{names[m]}</option>' for m in names)
    document = '''<!DOCTYPE html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BeamSplat 실측 비교</title>
<style>body{font:16px system-ui;background:#101723;color:#e3eaf5;max-width:1320px;margin:40px auto;padding:20px}h1{font-size:30px}p{line-height:1.7}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #344154;text-align:left}img{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}select{font:inherit;padding:8px;margin:8px;background:#253249;color:white}a{color:#8dd3ff}.note{color:#b7c8dd}</style>
<h1>BeamSplat → phase-only SLM: 실제 카메라 비교</h1>
<p>LETO-3 1920×1080, 6.4 µm, RGB 638/520/450 nm, +15 mm 단일 관측면. 고정 노출·gain·등록·white 정규화. 각 방법 3회, 반복마다 촬영 순서를 섞었다. 영상은 색별 CFS 슬롯의 촬영을 합성한 RGB이다.</p>
<img src="before_after_fixed.png" alt="Target, before, after">
<table><thead><tr><th>방법</th><th>고정 white 진폭 MSE (평균 ± 반복 SD)</th><th>사후 배율 보정 진폭 PSNR (dB)</th><th>보정·클리핑 sRGB SSIM</th></tr></thead><tbody>ROWS</tbody></table>
<p class="note">PSNR/SSIM은 채널별 사후 스칼라 보정 결과이며 실측 광량 증가를 뜻하지 않는다. 고정 white MSE를 우선 지표로 사용했다. 원본 DPH는 격자·관측면·영상 크기 조건이 달라 공정한 기준선에서 제외한다.</p>
<p>보기: <select id="brightness"><option value="fixed">고정 white</option><option value="fitted">사후 채널 배율 보정</option></select><select id="method">OPTIONS</select></p>
<div class="grid"><div><p>목표</p><img src="target.png"></div><div><p id="caption">카메라 촬영 평균</p><img id="capture" src="fixed_citl.png"></div></div>
<p>U.npy는 RGB complex64 (3,1080,1920)이다. 제공된 설명에 따라 센서면 장으로 해석했다. 현재 평면의 목표 진폭은 |U|²를 면적 보간해 전체 장면을 1680×945로 맞추고 1680×960 ROI에 넣은 뒤 제곱근을 취했다. RGB 공통 최대 진폭으로 정규화했다. 이 화면 맞춤의 물리적 가로 배율은 약 2.33이다. U의 공간 위상을 최적화 목표로 사용하지 않았으므로 깊이 정보·초점 이동·복소장 일치를 검증한 결과로 해석하면 안 된다.</p>
<p>ASM과 CNN은 같은 DPAC 초기값, Adam 0.01, 최대 1000회로 최적화했다. CNN 모델은 기존 색별 최적 checkpoint를 동결해 사용했다. 실제 카메라 CITL은 CNN 위상부터 색별 최대 200회 수행하고 측정 loss가 가장 작았던 업데이트 이전 위상을 보존했다. 전역 수렴을 의미하지 않는다.</p>
<p>학습 모델과 입력 및 보정 원본의 해시 불변: HASH. 세션 시작 대비 종료 광량 변화(R/G/B): DRIFT.</p>
<p><a href="summary.json">요약 수치</a> · <a href="metrics.json">반복/색별 원시 지표</a> · <a href="protocol.json">실험 조건·해시</a> · <a href="audit.json">float64 독립 재계산</a></p>
<script>const m=document.querySelector('#method'), b=document.querySelector('#brightness');m.value='BEST';function update(){document.querySelector('#capture').src=b.value+'_'+m.value+'.png'}m.onchange=b.onchange=update;update();</script></html>'''
    document = document.replace('ROWS', ''.join(rows)).replace('OPTIONS', methods).replace('BEST', best)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    saved_steps = []
    for ch in ('red', 'green', 'blue'):
        history = json.loads((root/f'history_{ch}_citl.json').read_text(encoding='utf-8'))
        key = 'mse' if 'mse' in history[0] else 'amplitude_mse'
        best_row = min(history, key=lambda row: row[key])
        saved_steps.append(dict(channel=ch, last_saved_step=history[-1]['step'],
                                best_saved_step=best_row['step'], best_in_loop_mse=best_row[key]))
    (root/'optimization_checkpoints.json').write_text(json.dumps(saved_steps, indent=2), encoding='utf-8')
    step_text = ' / '.join(f"{row['channel']} {row['last_saved_step']}회(최적 {row['best_saved_step']}회)" for row in saved_steps)
    note = f'<p>실제 CITL 저장 기록: {step_text}. 녹색은 Windows 잠금으로 중단되어 저장된 최저 오차 위상을 복원했다. 재개 시 고정 white 광량을 다시 확인했다. 최종 지표는 보정 중의 최저값이 아니라 별도 3회 재촬영에서 계산했다.</p>'
    document = document.replace('<p>학습 모델과 입력', note+'<p>학습 모델과 입력')
    phase_prefix = '' if (root/f'slm_{best}.png').exists() else '../'
    document = document.replace('<a href="summary.json">', f'<a href="{phase_prefix}slm_{best}.png">최종 선택 SLM 위상 PNG</a> · <a href="{phase_prefix}phase_{best}.npy">위상 배열</a> · <a href="summary.json">')
    if summary['excessive_drift']:
        warning = '<p style="background:#6a2831;padding:18px"><strong>기준광 검증 미통과.</strong> 종료 시 적색 기준광이 20% 이상 변했고 CNN 단독의 한 촬영에서 청색 이상값이 관측되었다. 이 첫 촬영 수치는 보존용이며 최종 성능의 근거에서 제외한다. <a href="validation_repeat01/report.html">별도 검증 촬영 보고서</a>를 확인한다.</p>'
        document = document.replace('<h1>', warning+'<h1>', 1)
    if 'validation_policy' in protocol:
        note = '<p>첫 평가에서 기준광 이상 및 단발성 이상값이 발견되어 위상·노출·보정값을 그대로 유지한 별도 재촬영이다. 화면 표시 후 1초, 색별 카메라 타이밍 전환 후 1초를 기다리고 5개 프레임을 평균했다. 매 촬영 묶음 전후 기준광을 측정했으며 사후 광량 재정규화는 하지 않았다. <a href="white_checks.json">모든 기준광 측정</a> · <a href="../report.html">보존된 첫 평가</a>.</p>'
        document = document.replace('<h1>', note+'<h1>', 1)
        m = summary['methods']
        drop = 1-m['citl']['fixed_white_mse']/m['asm']['fixed_white_mse']
        conclusion = f'<p><strong>판정:</strong> CNN+CITL의 고정 white 진폭 MSE는 ASM 대비 {drop:.1%} 낮아 전체 밝기·대비 재현이 크게 개선됐다. 그러나 사후 밝기 보정 SSIM은 ASM {m["asm"]["fitted_clipped_srgb_ssim"]:.3f}, CNN {m["cnn"]["fitted_clipped_srgb_ssim"]:.3f}, CNN+CITL {m["citl"]["fitted_clipped_srgb_ssim"]:.3f}으로 CITL이 낮다. 색 스펙클과 일부 원형 얼룩이 남으며, 모든 질감·구조 지표가 함께 개선된 결과는 아니다.</p>'
        document = document.replace('<img src="before_after_fixed.png"', conclusion+'<img src="before_after_fixed.png"', 1)
    document = document.replace('약 2.33', f"약 {protocol['physical_scene_magnification']:.2f}")
    document = document.replace('HASH', str(summary['hashes_unchanged'])).replace('DRIFT', ', '.join(f'{x:+.2%}' for x in summary['drift_fraction']))
    (root/'report.html').write_text(document, encoding='utf-8')
    print(json.dumps(dict(audit=audit, summary=summary), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    report(parser.parse_args().root)
