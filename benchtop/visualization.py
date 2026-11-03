"""Scientific parameter snapshots plus a local, self-contained HTML timeline."""
from __future__ import annotations
import base64
import io
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .optics import crop_center
from .rgb_run import CHANNELS, read_json
from .run import save_json


def experiment_summary(root):
    rows = read_json(Path(root)/'fullcolor/metrics.json')['repeat_validation']
    means = {name: float(np.mean([r['amplitude_mse'] for r in rows if r['variant'] == name]))
             for name in ['new_asm', 'new_cnn', 'new_cnn_citl']}
    per_channel = {name: np.mean([r['amplitude_mse_rgb'] for r in rows if r['variant'] == name], axis=0)
                   for name in ['new_asm', 'new_cnn']}
    worse = [ch.upper() for i, ch in enumerate(CHANNELS) if per_channel['new_cnn'][i] > per_channel['new_asm'][i]]
    text = (f'새 타깃 3회 재촬영 평균 진폭 MSE: ASM {means["new_asm"]:.6f} → '
            f'CNN {means["new_cnn"]:.6f} → CNN + CITL {means["new_cnn_citl"]:.6f}. '
            f'ASM 대비 최종 오차 감소는 {100*(1-means["new_cnn_citl"]/means["new_asm"]):.2f}%입니다. ')
    if worse:
        text += f'CNN 단독에서는 {" / ".join(worse)} 오차가 ASM보다 증가했습니다. '
    return text+'저장한 위상 분포에서의 예측 정확도가 새 위상 최적화 성능을 보장하지는 않습니다.'


def heat(ax, value, title, vmin, vmax, cmap='viridis'):
    im = ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    low, high = np.nanmin(value) < vmin, np.nanmax(value) > vmax
    extend = 'both' if low and high else 'min' if low else 'max' if high else 'neither'
    plt.colorbar(im, ax=ax, fraction=.035, pad=.02, extend=extend)


def tile_filters(weights):
    filters = weights[:, 0] if weights.shape[1] == 1 else weights[0]
    n, h, w = filters.shape
    grid = np.full((2*h+1, 4*w+3), np.nan)
    for i in range(min(n, 8)):
        y, x = (i//4)*(h+1), (i%4)*(w+1)
        grid[y:y+h, x:x+w] = filters[i]
    return grid


def phase_figures(path, traces, history, target, name, output):
    snapshots = sorted(traces.glob('step_*.npz'))
    if not snapshots:
        return []
    with np.load(snapshots[0]) as first:
        initial = first['phase'].copy()
    gradient_limit = 1e-9
    for file in snapshots:
        with np.load(file) as values:
            gradient_limit = max(gradient_limit, float(np.quantile(np.abs(values['gradient_roi']), .99)))
    files = []
    for file in snapshots:
        step = int(file.stem.split('_')[-1])
        with np.load(file) as values:
            phase, amplitude, gradient = values['phase'], values['amplitude'], values['gradient_roi']
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
            heat(axes[0, 0], target, 'Target amplitude / fixed flat white', 0, 1.5)
            heat(axes[0, 1], amplitude, 'Measured amplitude', 0, 1.5)
            heat(axes[0, 2], amplitude-target, 'Measured minus target', -1, 1, 'RdBu_r')
            heat(axes[0, 3], gradient, 'Phase gradient (final step: not computed)' if step == history[-1]['step']
                 else 'Phase gradient in ROI', -gradient_limit, gradient_limit, 'RdBu_r')
            heat(axes[1, 0], np.angle(np.exp(1j*phase))[::3, ::3], 'Learned phase: entire physical SLM [rad]', -np.pi, np.pi, 'twilight')
            heat(axes[1, 1], crop_center(np.angle(np.exp(1j*phase)), target.shape), 'Learned phase: ROI [rad]', -np.pi, np.pi, 'twilight')
            heat(axes[1, 2], crop_center(np.angle(np.exp(1j*(phase-initial))), target.shape),
                 'Wrapped change since step 0 [rad]', -np.pi, np.pi, 'twilight')
            ax = axes[1, 3]
            ax.plot([r['step'] for r in history], [r['amplitude_mse'] for r in history], color='#246d9d')
            ax.axvline(step, color='#d66133', lw=2)
            ax.set(xlabel='CITL iteration', ylabel='Fixed-white amplitude MSE', title='Camera loss (no fitted image gain)')
            ax.grid(alpha=.2)
            fig.suptitle(f'{name} | CITL step {step} | trainable: SLM phase | all heatmap scales fixed over time', fontsize=14)
            destination = output/f'{name.lower().replace(" ", "_")}_step_{step:04d}.png'
            fig.savefig(destination, dpi=110)
            plt.close(fig)
        files.append({'step': step, 'file': destination.name})
    return files


def model_figures(path, output, ch):
    report = read_json(path/'forward_training.json')
    history = report['history']
    weights = sorted((path/'model_trace').glob('epoch_*.pt'))
    first_limit, last_limit, multiplier_limit = 1e-6, 1e-6, .05
    for file in weights:
        state = torch.load(file, weights_only=True, map_location='cpu')['correction']
        first_limit = max(first_limit, float(state['0.weight'].abs().max()))
        last_limit = max(last_limit, float(state['4.weight'].abs().max()))
        with np.load(file.with_suffix('.npz')) as data:
            multiplier_limit = max(multiplier_limit, float(np.abs(data['amplitude_multiplier']-1).max()))
    files = []
    for file in weights:
        epoch = int(file.stem.split('_')[-1])
        state = torch.load(file, weights_only=True, map_location='cpu')
        with np.load(file.with_suffix('.npz')) as data:
            fig, axes = plt.subplots(3, 3, figsize=(14, 11), layout='constrained')
            heat(axes[0, 0], data['asm_amplitude'], 'Fixed validation probe: ASM amplitude', 0, 1.5)
            heat(axes[0, 1], data['camera_amplitude'], 'Same probe: real camera amplitude', 0, 1.5)
            heat(axes[0, 2], data['prediction'], 'Learned model prediction', 0, 1.5)
            heat(axes[1, 0], data['amplitude_multiplier'], 'Learned amplitude multiplier (CNN + gain)',
                 1-multiplier_limit, 1+multiplier_limit, 'RdBu_r')
            heat(axes[1, 1], data['prediction']-data['camera_amplitude'], 'Prediction minus measurement', -.5, .5, 'RdBu_r')
            ax = axes[1, 2]
            for split in ('train', 'val'):
                ax.plot([r['epoch'] for r in history], [r[split+'_mse'] for r in history], label=split)
            ax.axvline(epoch, color='#d66133')
            ax.set(xlabel='Epoch', ylabel='Amplitude MSE', title='Loss: fixed radiometric units')
            ax.legend(); ax.grid(alpha=.2)
            heat(axes[2, 0], tile_filters(state['correction']['0.weight'].numpy()),
                 'Actual learned first-layer filters (8 x 5 x 5)', -first_limit, first_limit, 'RdBu_r')
            heat(axes[2, 1], tile_filters(state['correction']['4.weight'].numpy()),
                 'Actual learned output filters (8 x 3 x 3)', -last_limit, last_limit, 'RdBu_r')
            ax = axes[2, 2]
            for key in history[0]['parameter_l2']:
                if key == 'log_gain':
                    continue
                ax.plot([r['epoch'] for r in history], [r['parameter_l2'][key] for r in history], label=key.replace('correction.', ''))
            ax.axvline(epoch, color='#d66133')
            ax.set(xlabel='Epoch', ylabel='Parameter L2 norm', title='All CNN weights and biases')
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
            gain = float(state['log_gain'].exp())
            fig.suptitle(f'{ch.upper()} | epoch {epoch} | learned amplitude gain {gain:.5f} | {report["parameter_count"]} parameters\n'
                         'Fixed distance / aperture / LUT / illumination; maps and filters retain the same scales', fontsize=13)
            destination = output/f'{ch}_epoch_{epoch:03d}.png'
            fig.savefig(destination, dpi=110)
            plt.close(fig)
        files.append({'epoch': epoch, 'file': destination.name})
    return files


def create_report(root):
    root = Path(root)
    output = root/'visualizations'
    output.mkdir(exist_ok=True)
    target_plan = read_json(root/'targets/manifest.json') if (root/'targets/manifest.json').exists() else None
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    index, summary = {}, {}
    for ci, ch in enumerate(CHANNELS):
        path = root/ch
        target = np.load(path/'target_amplitude.npy')
        history = read_json(path/'citl_metrics.json')['history']
        index[ch] = {'citl': phase_figures(path, path/'citl_trace', history, target, ch.upper(), output),
                     'cnn': model_figures(path, output, ch)}
        joint_history = read_json(root/'fullcolor'/f'citl_{ch}_metrics.json')
        joint_target = np.load(root/'fullcolor/new_target_amplitude_rgb.npy')[..., ci]
        index[ch]['composite'] = phase_figures(path, root/'fullcolor'/f'citl_{ch}', joint_history,
                                              joint_target, f'Composite {ch.upper()}', output)
        training = read_json(path/'forward_training.json')
        test = [r for r in training['evaluation'] if r['split'] == 'test']
        summary[ch] = {'best_epoch': training['best_epoch'],
                       'test': {k: float(np.mean([r[k] for r in test])) for k in ['asm_mse', 'gain_only_mse', 'cnn_mse']}}
        print(f'Generated {ch} parameter timelines', flush=True)
    # Comparison of actual composite displays; not synthesized propagation images.
    variants = ['demo_target', 'demo_before_camera', 'demo_citl_camera',
                'new_target', 'new_asm_camera', 'new_cnn_camera', 'new_cnn_citl_camera']
    labels = ['Target A', 'ASM warm start / camera', 'Channel CITL / camera',
              'Unseen target B', 'ASM / camera', 'Trained CNN / camera', 'CNN + composite CITL / camera']
    if target_plan:
        labels[0] += ' / DIV2K '+Path(next(r['target_id'] for r in target_plan['targets'] if r['slot'] == 1000)).stem
        labels[3] += ' / DIV2K '+Path(next(r['target_id'] for r in target_plan['targets'] if r['slot'] == 2000)).stem
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), layout='constrained')
    for ax in axes.flat:
        ax.set_axis_off()
    for i, (variant, label) in enumerate(zip(variants, labels)):
        position = i if i < 3 else i+1
        axes.flat[position].imshow(Image.open(root/'fullcolor'/f'{variant}_rgb.png'))
        axes.flat[position].set_title(label)
    fig.suptitle('Actual RGB888 SLM frames | time-gated camera channels | fixed flat-white normalization\n'
                 'Gamma-encoded sensor RGB; not CIE color calibration. No per-image intensity rescaling.', fontsize=13)
    fig.savefig(output/'fullcolor_comparison.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, ch in zip(axes, CHANNELS):
        values = summary[ch]['test']
        ax.bar(['ASM', 'Gain only', 'CNN'], list(values.values()), color=['#9099a2', '#edaa54', '#3c849e'])
        ax.set(title=f'{ch.upper()} / untouched test targets', ylabel='Amplitude MSE', ylim=(0, max(values.values())*1.2))
    fig.savefig(output/'heldout_model_comparison.png', dpi=130)
    plt.close(fig)
    save_json(output/'index.json', index)
    save_json(output/'summary.json', summary)
    # Embed the image assets so the timeline remains usable as one portable file.
    images = {}
    if (root/'targets/contact_sheet.png').exists():
        shutil.copy2(root/'targets/contact_sheet.png', output/'div2k_targets.png')
    for file in output.glob('*.png'):
        buffer = io.BytesIO()
        with Image.open(file) as source:
            source.convert('RGB').save(buffer, format='JPEG', quality=88)
        images[file.name] = 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RGB CITL · 학습 파라미터 기록</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#18242e;font:16px/1.65 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:28px;margin:0}h2{font-size:21px;margin:30px 0 8px}p{max-width:1050px}.note{color:#536573}section{background:white;border:1px solid #dce2e6;border-radius:12px;padding:18px;margin:20px 0}img{width:100%;height:auto;display:block}nav{display:flex;flex-wrap:wrap;align-items:center;gap:15px;padding:12px 0;position:sticky;top:0;background:#fff;z-index:1}select,button{font:inherit;padding:6px 12px;border:1px solid #bbc7ce;border-radius:6px;background:white}input{min-width:180px;flex:1;accent-color:#287896}output{min-width:130px;font-variant-numeric:tabular-nums}code{background:#eaf0f4;padding:2px 5px;border-radius:4px}li{margin:4px 0}
</style><main><h1>RGB CITL · 중간 학습 파라미터</h1>
<p>15 mm 단일 평면 · LETO-3 1920×1080 · FLIR 25222112 · R 638 / G 520 / B 450 nm</p>
<p>채널별 실측 보정 → 위상–카메라 쌍 확보 → CNN 전파 모델 학습 → 미사용 컬러 타깃 생성 → 실제 RGB 프레임에서 CITL 보정 순서로 진행했습니다. CITL은 SLM 위상, CNN 학습은 필터·bias·gain을 업데이트합니다.</p>
<section><h2>학습 중 무엇이 바뀌었나</h2><nav><label>채널 <select id="channel"><option value="red">Red</option><option value="green">Green</option><option value="blue">Blue</option></select></label>
<label>과정 <select id="kind"><option value="citl">채널 CITL · SLM 위상</option><option value="cnn">CNN · 필터와 보정 맵</option><option value="composite">합성 RGB 프레임 CITL</option></select></label>
<button id="play" type="button">재생</button><input id="step" type="range" min="0" value="0" aria-label="학습 단계"><output id="label"></output></nav>
<img id="timeline" alt="선택한 단계의 실제 학습 파라미터와 측정 결과">
<p class="note">색 범위는 시간에 따라 고정됩니다. CNN 보정 맵은 매 epoch 동일한 검증 입력으로 계산했습니다. 필터는 실제 checkpoint의 가중치입니다. 마지막 CITL 측정에서는 gradient를 계산하지 않습니다.</p></section>
<section><h2>실제 컬러 출력 비교</h2><p>__RESULT_TEXT__</p><img id="comparison" alt="타깃과 실측 컬러 비교"><p class="note">각 결과는 한 RGB888 프레임을 SLM에 출력한 상태에서 얻었습니다. 미리보기는 R/G/B 발광 구간을 따로 촬영한 센서 채널을 정합해 합성한 영상입니다. 채널별 flat white로 정규화했으며, 색도계 기반 색 보정이나 한 번의 장노출 촬영은 아닙니다.</p></section>
<section><h2>학습에 사용하지 않은 시험 타깃</h2><img id="heldout" alt="ASM, 밝기 보정, CNN의 시험 오차"><p>Gain-only는 학습 데이터로만 구한 단일 밝기 계수입니다. CNN이 이 기준보다 개선됐는지 함께 확인해야 합니다. 모델 선택에는 validation만 사용했고 test는 선택 후 평가했습니다.</p></section>
<section><h2>해석 범위</h2><ul><li>학습 변수: 각 SLM 픽셀 위상, 채널별 CNN 필터·bias와 scalar log gain.</li><li>고정값: 거리 15 mm, 파장, 픽셀 피치, Fourier aperture 0.5, 선형 phase LUT, 정합, flat 기반 조명 추정치. 이 값들의 수차 보정 학습을 완료했다는 의미는 아닙니다.</li><li>Compact 모델은 채널별 866개 파라미터를 사용합니다. 논문의 대규모 CNNpropCNN 재현 실험과 구분해야 합니다.</li><li>서로 다른 슬롯 사이의 작은 영향과 고정 광학 무늬가 남습니다. 합성 프레임 CITL은 다른 채널 위상을 유지한 채 각 채널을 한 차례씩 순서대로 보정했습니다.</li></ul></section></main>
<script>const data=__INDEX__, images=__IMAGES__; const channel=document.querySelector('#channel'),kind=document.querySelector('#kind'),step=document.querySelector('#step'),label=document.querySelector('#label');let timer=null;
function update(reset=false){const rows=data[channel.value][kind.value];if(reset)step.value=0;step.max=rows.length-1;const row=rows[+step.value];label.textContent=(row.epoch===undefined?'Iteration '+row.step:'Epoch '+row.epoch)+' / '+(rows.at(-1).epoch??rows.at(-1).step);document.querySelector('#timeline').src=images[row.file];}
channel.onchange=kind.onchange=()=>update(true);step.oninput=()=>update();document.querySelector('#play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='재생';}else{this.textContent='정지';timer=setInterval(()=>{step.value=(+step.value+1)%(+step.max+1);update();},800);}};
document.querySelector('#comparison').src=images['fullcolor_comparison.png'];document.querySelector('#heldout').src=images['heldout_model_comparison.png'];update();</script></html>'''
    document = document.replace('__INDEX__', json.dumps(index)).replace('__IMAGES__', json.dumps(images))
    document = document.replace('__RESULT_TEXT__', experiment_summary(root))
    if target_plan:
        text = (f'<p>타깃 입력: DIV2K 원본 {target_plan["available_originals"]["train"]} train / '
                f'{target_plan["available_originals"]["valid"]} validation 중 서로 다른 {len(target_plan["targets"])}장. '
                'sRGB를 선형 광강도로 변환해 종횡비 유지 리사이즈와 중앙 crop 후 진폭 타깃을 만들었습니다. '
                '학습·검증·시험·CITL 데모·새 타깃은 원본 이미지 단위로 분리했습니다. '
                f'{target_plan["pairs"]}개 학습·평가용 RGB 패턴과 최종 검증 패턴은 patterns/ 및 fullcolor/ 폴더에 있습니다.</p>')
        document = document.replace('<section><h2>학습 중 무엇이 바뀌었나</h2>', text+'<section><h2>학습 중 무엇이 바뀌었나</h2>', 1)
        if 'div2k_targets.png' in images:
            gallery = ('<section><h2>이번에 선택한 DIV2K 타깃</h2><img alt="원본 이미지별 데이터 분리와 타깃 crop" src="'
                       +images['div2k_targets.png']+'"><p class="note">표시된 번호는 DIV2K 원본 파일 번호입니다. '
                       '각 crop은 RGB 진폭 타깃으로 변환되어 동일한 SLM 격자에서 최적화됩니다.</p></section>')
            document = document.replace('<section><h2>해석 범위</h2>', gallery+'<section><h2>해석 범위</h2>', 1)
    (root/'report.html').write_text(document, encoding='utf-8')
    print(f'Interactive report saved: {root/"report.html"}', flush=True)
