"""Small, replaceable scientific report and independent completion checks."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .campaign import decode_phase
from .live import atomic_json
from .optics import crop_center, load_forward_model
from .rgb_run import CHANNELS, read_json


def jsonl(path, allow_partial=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    if allow_partial and lines and not lines[-1].endswith('\n'):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def evaluation_means(result):
    return {name: float(np.mean([r['amplitude_mse'] for r in result['repeats'] if r['variant'] == name]))
            for name in ('asm', 'cnn', 'cnn_citl')}


def audit(root, device='cuda'):
    root = Path(root).resolve()
    directory = root/'campaign'
    state = read_json(directory/'status.json')
    completed = read_json(directory/'completed.json')
    assert state['status'] == completed['status'] == 'completed'
    assert completed['requested_hours'] == 10., 'This audit is for the requested ten-hour run'
    assert completed['active_seconds'] >= 9.5*3600, 'Less than approximately ten hours of actual run time'
    assert completed['cycle'] >= 2
    plan = read_json(directory/'plan.json')
    all_ids = sum([plan[key] for key in ('train','val','test','tracking','final')], [])
    assert len(all_ids) == len(set(all_ids)), 'Source-image split leakage'
    assert all('DIV2K_train_HR' in name for name in plan['train'])
    assert all('DIV2K_valid_HR' in name for key in ('val','test','tracking','final') for name in plan[key])
    ledger = {r['serial']:r for r in jsonl(directory/'acquisition.jsonl')}
    assert len(ledger) == state['serial']
    for row in ledger.values():
        assert row['source']['target_id'] in plan[row['split']]
    pool = read_json(directory/'pool.json')
    counts = Counter(r['split'] for r in pool)
    assert counts['train'] <= completed['arguments']['pool']
    assert counts['val'] == counts['test'] == 8
    for row in pool:
        path = directory/'pool'/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        with np.load(path) as values:
            assert values['phase_rgb_u8'].shape == (1080,1920,3)
            assert values['phase_rgb_u8'].dtype == np.uint8
            assert values['amplitude_rgb'].shape == (960,1680,3)
            assert np.isfinite(values['amplitude_rgb']).all()
    histories = jsonl(directory/'training.jsonl')
    assert {r['channel'] for r in histories}.issubset(set(CHANNELS)|{'rgb'})
    assert any(r['channel'] == 'rgb' for r in histories) or {r['channel'] for r in histories} == set(CHANNELS)
    model_hashes = {}
    for ch in CHANNELS:
        calibration = read_json(root/ch/'calibration/calibration.json')
        assert calibration['config']['roi_shape'] == [960,1680]
        assert calibration['rms_half_sensor_px'] <= 2.
        assert len(calibration['validation']) == 4
        assert max(v['error_px'] for v in calibration['validation']) <= 3.
        path = directory/'models'/f'{ch}_best.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert all(ledger[s]['split'] == 'train' for s in checkpoint['training_sample_serials'])
        model_hashes[ch] = hashlib.sha256(path.read_bytes()).hexdigest()
    evaluations = jsonl(directory/'evaluation.jsonl')
    finals = [r for r in evaluations if r['final']]
    assert {r['source']['target_id'] for r in finals} == set(plan['final'])
    assert len(finals) == 8
    assert len(evaluations) == state['evaluations']
    for row in evaluations:
        assert row['source']['target_id'] in plan['final' if row['final'] else 'tracking']
        assert Counter(r['variant'] for r in row['repeats']) == {'asm':3,'cnn':3,'cnn_citl':3}
        assert all(np.isfinite(r['amplitude_mse']) and r['amplitude_mse'] >= 0 for r in row['repeats'])
    for row in finals:
        assert row['model_hashes'] == model_hashes, 'Final evaluation changed selected weights'
        for variant in ('asm','cnn','cnn_citl'):
            path = directory/'evaluations'/f'final_{row["evaluation"]:04d}_{variant}.npz'
            with np.load(path) as values:
                assert values['phase_rgb_u8'].shape == (1080,1920,3)
                assert values['amplitude_rgb'].shape == (960,1680,3)
                assert np.isfinite(values['amplitude_rgb']).all()
    # Recompute the sealed test from exact saved display bytes and camera arrays.
    sealed = read_json(directory/'sealed_test.json')
    assert len(sealed) == 24
    for ci, ch in enumerate(CHANNELS):
        config = read_json(root/ch/'config.json')
        model = load_forward_model(directory/'models'/f'{ch}_best.pt', config, device)
        with torch.no_grad():
            for row in pool:
                if row['split'] != 'test':
                    continue
                with np.load(directory/'pool'/row['file']) as values:
                    phase = torch.from_numpy(decode_phase(values['phase_rgb_u8'][...,ci], config['invert_phase'])).to(device)
                    amplitude = torch.from_numpy(values['amplitude_rgb'][...,ci].astype(np.float32)).to(device)
                reference = next(r for r in sealed if r['channel']==ch and r['target_id']==row['source']['target_id'])
                for name, prediction in [('cnn',model(phase)),('asm',crop_center(model.asm(phase).abs(),config['roi_shape']))]:
                    value = float((prediction-amplitude).square().mean())
                    assert abs(value-reference[name+'_mse']) < 2e-6
        del model
    frames = jsonl(directory/'frames.jsonl')
    timestamps = [f['timestamp_ns'] for event in frames for f in event['frames']]
    assert len(timestamps) > 1000
    assert all(b > a for a,b in zip(timestamps,timestamps[1:])), 'Camera timestamps not strictly increasing'
    assert all(event['frames'] for event in frames if event['label'] != 'end')
    provenance = read_json(directory/'runtime_provenance.json')
    for row in provenance['sources']:
        archived = directory/'source_snapshot'/Path(row['file']).name
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == row['sha256']
    total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert total < completed['arguments']['max_gib']*2**30
    report = dict(status='passed', active_hours=completed['active_seconds']/3600, cycles=completed['cycle'],
                  captured_rgb_samples=len(ledger), remaining_pool=dict(counts), final_targets=8,
                  physical_evaluations=len(evaluations), sealed_test_recomputed=24,
                  camera_frames=len(timestamps), total_run_gib=total/2**30,
                  caveat='Completion checks do not imply reconstruction improves on every target.')
    atomic_json(directory/'completion_audit.json', report)
    return report


def render(root):
    root = Path(root)
    directory = root/'campaign'
    state = read_json(directory/'status.json')
    training = jsonl(directory/'training.jsonl', allow_partial=state['status']=='running')
    evaluations = jsonl(directory/'evaluation.jsonl', allow_partial=state['status']=='running')
    drift = jsonl(directory/'drift.jsonl', allow_partial=state['status']=='running')
    output = directory/'report'
    output.mkdir(exist_ok=True)
    started = state['created_at']
    fig, axes = plt.subplots(2, 2, figsize=(13,8), constrained_layout=True)
    colors = dict(red='#d74450',green='#299661',blue='#397cd9')
    for ch in CHANNELS:
        rows = [dict(r, val_mse=r['val_mse_rgb'][CHANNELS.index(ch)],
                     amplitude_gain=r['amplitude_gain_rgb'][CHANNELS.index(ch)]) if r['channel']=='rgb' else r
                for r in training if r['channel'] in (ch, 'rgb')]
        hours = [(r['time']-started)/3600 for r in rows]
        axes[0,0].plot(hours, [r['val_mse'] for r in rows], label=ch, color=colors[ch])
        axes[0,1].plot(hours, [r['amplitude_gain'] for r in rows], label=ch, color=colors[ch])
        axes[1,1].plot([(r['time']-started)/3600 for r in drift], [r['median_white_rgb'][CHANNELS.index(ch)] for r in drift], '.-', color=colors[ch], label=ch)
    tracking = [r for r in evaluations if not r['final']]
    for name, color in [('asm','#777777'),('cnn','#397cd9'),('cnn_citl','#299661')]:
        axes[1,0].plot([(r['measured_at']-started)/3600 for r in tracking], [evaluation_means(r)[name] for r in tracking], '.-', color=color, label=name)
    for ax, title, ylabel in zip(axes.flat, ['Fixed validation pairs','Learned amplitude gain','Physical tracking (rotating targets)','Flat-white drift'],['Amplitude MSE','Gain','Amplitude MSE','Median intensity / calibration white']):
        ax.set(title=title, xlabel='Wall-clock hours since launch', ylabel=ylabel)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    fig.suptitle(f'RGB / 1680 x 960 ROI / {state["status"]} / {state["active_seconds"]/3600:.2f} active hours')
    fig.savefig(output/'learning_curves.png', dpi=140)
    plt.close(fig)
    finals = [r for r in evaluations if r['final']]
    summary = dict(status=state['status'], active_hours=state['active_seconds']/3600,
                   cycle=state['cycle'], acquired_rgb_samples=state['serial'], evaluations=state['evaluations'],
                   storage=state['storage'], memory=state['memory'], final_target_count=len(finals))
    if finals:
        summary['final_mean_mse'] = {name:float(np.mean([evaluation_means(r)[name] for r in finals]))
                                     for name in ('asm','cnn','cnn_citl')}
        summary['final_per_target'] = [dict(target_id=r['source']['target_id'], **evaluation_means(r)) for r in finals]
    atomic_json(output/'summary.json', summary)
    gallery = ''.join('<article><h3>'+html.escape(row['source']['target_id'])+'</h3><div class="grid">'+''.join(
        f'<figure><img src="../evaluations/final_{row["evaluation"]:04d}_{suffix}.png"><figcaption>{label}</figcaption></figure>'
        for suffix,label in [('target','Target'),('asm_camera','ASM camera'),('cnn_camera','CNN camera'),('cnn_citl_camera','CNN + CITL camera')])+'</div></article>' for row in finals)
    page = '<!doctype html><meta charset="utf-8"><title>10-hour RGB experiment</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px system-ui;background:#111923;color:#edf2f7}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}figure{margin:0}pre{white-space:pre-wrap;color:#a8dccc}</style><h1>RGB 10시간 학습·실측 평가</h1><p>SLM 1920×1080 · 복원 ROI 1680×960 · +15 mm</p><p>Tracking 영상은 원본 8장을 순환하므로 곡선의 인접 점은 서로 다른 타겟일 수 있습니다. 최종 평가는 별도 미사용 원본 8장, 각 방식 3회 측정입니다. 미리보기는 fixed-white 정규화 sensor RGB이며 CIE 색 보정을 뜻하지 않습니다.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="learning_curves.png" alt="Learning, gain, physical evaluation and drift curves">'+gallery
    (output/'index.html').write_text(page,encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args()
    if args.audit:
        print(json.dumps(audit(args.out,args.device),indent=2))
    print(json.dumps(render(args.out),indent=2))


if __name__=='__main__':
    main()
