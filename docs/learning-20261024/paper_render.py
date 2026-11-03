"""Render the audited completed-run data into the existing guide."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
METHODS=['dpac','sgd_asm','sgd_cnnpropcnn']
LABELS=['DPAC','SGD-ASM','SGD-CNNpropCNN']
def table(head, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def render():
    data=json.loads((HERE/'paper-results-data.json').read_text(encoding='utf-8'))
    html=(HERE/'paper-section.html').read_text(encoding='utf-8')
    rows=[]
    for c,label in zip(['red','green','blue'],['적색','녹색','청색']):
        m=data['models'][c];f=next(r for r in data['forward'] if r['channel']==c)
        rows.append([label,f"{m['epoch']} / {m['best_epoch']}",f"{m['best_l1']:.6f}",f"{f['asm_mse']:.6f}",f"{f['cnn_mse']:.6f}"])
    html=html.replace('__PAPER_TRAIN_TABLE__',table(['채널','종료 / 선택 epoch','최저 검증 L1','시험 ASM 고정 MSE','시험 CNN 고정 MSE'],rows))
    keys=['fixed_white_mse','fitted_amplitude_mse','fitted_amplitude_psnr','fitted_clipped_srgb_psnr','fitted_clipped_srgb_ssim']
    html=html.replace('__PAPER_MEANS_TABLE__',table(['방법','고정 진폭 MSE ↓','보정 진폭 MSE ↓','보정 진폭 PSNR ↑ (dB)','보정 sRGB PSNR ↑ (dB)','보정 sRGB SSIM ↑'],[[label]+[f"{data['means'][m][k]:.{3 if 'psnr' in k else 6}f}" for k in keys] for m,label in zip(METHODS,LABELS)]))
    html=html.replace('__PAPER_TARGET_TABLE__',table(['DIV2K ID','DPAC 고정 MSE','ASM 고정 MSE','CNN 고정 MSE','DPAC 보정 MSE','ASM 보정 MSE','CNN 보정 MSE'],[[t['id']]+[f"{t['means'][m][k]:.6f}" for k in ['fixed_white_mse','fitted_amplitude_mse'] for m in METHODS] for t in data['targets']]))
    rows=[]
    for c,label in zip(['red','green','blue'],['적색','녹색','청색']):
        v={m:next(r for r in data['probe']['results'] if r['channel']==c and r['model']==m and r['domain']=='amplitude') for m in ['asm','cnnpropcnn']}
        rows.append([label]+[f"{v[m][k]:.{3 if k=='cosine' else 6}f}" for k in ['cosine','rmse'] for m in ['asm','cnnpropcnn']])
    html=html.replace('__PAPER_PROBE_TABLE__',table(['채널','ASM 진폭 코사인 ↑','CNN 진폭 코사인 ↑','ASM 진폭 RMSE ↓','CNN 진폭 RMSE ↓'],rows))
    html=html.replace('__PAPER_DRIFT__','최종 평가 시작·종료의 고정 영 위상 촬영 16회에서, 데이터 획득 시작 기준 평균 광강도 변화 범위는 '+', '.join(f"{c} {lo:+.2f}~{hi:+.2f}%" for c,lo,hi in zip(['R','G','B'],data['drift_min_percent'],data['drift_max_percent']))+'였습니다. 백색 보정은 새로 맞추지 않았습니다.')
    html=html.replace('__PAPER_REPEAT__','8개 영상 중 고정 진폭 MSE의 재촬영 3회 표본 표준편차 최댓값은 '+', '.join(f"{label} {data['repeat_sd_max'][m]['fixed_white_mse']:.6f}" for m,label in zip(METHODS,LABELS))+'였습니다. 색 평균 후 반복 사이의 표준편차입니다.')
    return data,html
