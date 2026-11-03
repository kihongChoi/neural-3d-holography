"""Apply frozen measured RGB models to a BeamSplat sensor-plane amplitude.

Single +15 mm plane only. Does not claim complex-phase or depth fidelity.
All hardware captures use the existing framebuffer, saturation and SDK guards.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .calibration import Rectifier, channel_intensity
from .campaign import decode_phase
from .hardware import Bench
from .optics import crop_center
from .paper_campaign import canonical, sha
from .paper_model import PaperCNNpropCNN, dpac, image_metrics
from .rgb_run import CHANNELS, channel_model, read_json, select_channel, srgb_preview


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    source, trained = Path(args.source), Path(args.trained)
    configs = [read_json(trained/ch/'config.json') for ch in CHANNELS]
    base = configs[1]
    rects = [Rectifier(trained/ch/'calibration', c) for ch, c in zip(CHANNELS, configs)]
    u = np.load(source/'field'/'U.npy', allow_pickle=False)
    assert u.shape == (3, 1080, 1920) and np.iscomplexobj(u) and np.isfinite(u).all()
    rh, rw = base['roi_shape']
    # Preserve the complete scene and aspect ratio; zero letterbox in the ROI.
    scale = min(rw/u.shape[2], rh/u.shape[1])
    nw, nh = round(u.shape[2]*scale), round(u.shape[1]*scale)
    peak = float(np.abs(u).max())
    target = np.zeros((rh, rw, 3), np.float32)
    intensity = np.moveaxis(np.abs(u)**2, 0, -1)/peak**2
    target[(rh-nh)//2:(rh-nh)//2+nh, (rw-nw)//2:(rw-nw)//2+nw] = np.sqrt(
        np.maximum(cv2.resize(intensity, (nw, nh), interpolation=cv2.INTER_AREA), 0))
    np.save(out/'target_amplitude.npy', target)
    Image.fromarray(np.uint8(np.clip(srgb_preview(target)*255, 0, 255))).save(out/'target.png')
    watched = [source/'field'/'U.npy', source/'summary.json']
    watched += [trained/'models'/f'{ch}_best.pt' for ch in CHANNELS]
    watched += [p for ch in CHANNELS for p in (trained/ch/'calibration').iterdir() if p.is_file()]
    hashes = {str(p.resolve()): sha(p) for p in watched}
    protocol = dict(source=str(source.resolve()), trained=str(trained.resolve()), hashes=hashes,
        configs=configs, source_optics=read_json(source/'summary.json')['optics'],
        target_policy='Sensor-plane amplitude, full scene fitted in ROI, area resize of linear intensity, common RGB peak amplitude normalization. Single +15mm target. Input phase retained only in raw saved DPH reference.',
        input_shape=list(u.shape), resized_wh=[nw, nh], common_peak_amplitude=peak,
        physical_scene_magnification=(nw*base['slm_pitch_m'])/(u.shape[2]*3e-6),
        comparison='raw saved DPH (different geometry; diagnostic only), adapted DPAC, SGD ASM, frozen CNNpropCNN, frozen CNNpropCNN + actual camera CITL',
        model_steps=args.model_steps, citl_steps=args.citl_steps, repeats=3,
        optimization='Adam .01, quantized forward / straight-through backward, fixed white amplitude MSE, same DPAC initialization for ASM and CNN; CITL starts from CNN. Best measured pre-update phase retained.',
        limitation='Single plane intensity comparison. Neither 3D reconstruction nor input complex phase preservation validated. Raw DPH is not a matched baseline.')
    if args.resume:
        original = read_json(out/'protocol.json')
        if original['hashes'] != hashes or original['configs'] != configs:
            raise RuntimeError('Resume source/model/calibration mismatch')
        if list(out.glob('capture_*_0.npz')):
            raise RuntimeError('Evaluation captures already exist; audit before resuming')
        save_json(out/f'resume_{int(time.time())}.json', dict(time=time.time(), source_sha256=sha(Path(__file__)),
            policy='Reuse all saved best phases, including interrupted green CITL best. Do not claim exact Adam continuation. New initial drift check.'))
    else:
        save_json(out/'protocol.json', protocol)
    device = 'cuda'
    phases = {k: np.zeros(tuple(base['slm_shape'])+(3,), np.float32)
              for k in ('dpac', 'asm', 'cnn', 'citl')}
    phases['raw_dph'] = np.moveaxis(np.load(source/'field'/'dph_phase.npy'), 0, -1)
    frames_log = out/'frames.jsonl'

    def progress(stage, **kw):
        row = dict(stage=stage, time=time.time(), **kw)
        save_json(out/'status.json', row)
        print(json.dumps(row), flush=True)

    with Bench(base, display_wait_s=60 if args.resume else 0) as bench:
        def drain(label):
            with frames_log.open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(label=label, frames=bench.frame_metadata))+'\n')
            bench.frame_metadata.clear()

        def capture(p, label):
            bench.show(p)
            amps, sensors = [], []
            for c, r in zip(configs, rects):
                select_channel(bench, c)
                rgb = bench.capture(frames=3)
                amps.append(r.amplitude(rgb))
                sensors.append(channel_intensity(rgb, c))
            drain(label)
            return np.stack(amps, -1), np.stack(sensors, -1)

        def drift(label):
            a, sensor = capture(np.zeros_like(phases['dpac']), label)
            np.savez_compressed(out/f'drift_{label}.npz', amplitude=a, sensor=sensor)
            return (a*a).mean((0, 1))

        if args.resume:
            a = np.load(out/'drift_start.npz')['amplitude']
            reference = (a*a).mean((0, 1))
            current = drift('resume_start')
            if np.any(np.abs(current/reference-1) > .2):
                raise RuntimeError('White intensity changed more than 20% since initial capture')
        else:
            reference = drift('start')
        prior_ref = np.load(trained/'drift_reference.npz')
        progress('hardware_connected', current_white_intensity=reference.tolist(),
                 previous_drift_reference_keys=prior_ref.files)

        def optimize(model, t, c, initial, steps, label, ci, camera=False):
            phase = initial.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([phase], lr=.01)
            best, best_phase, history = float('inf'), phase.detach().clone(), []
            plateau = 0
            for step in range(steps+1):
                bench.app.processEvents()
                if (out/'STOP_REQUESTED').exists():
                    raise InterruptedError('STOP_REQUESTED')
                opt.zero_grad(set_to_none=True)
                pred = crop_center(model(phase).abs(), t.shape)
                if camera:
                    p = phases['cnn'].copy()
                    p[..., ci] = phase.detach().cpu().numpy()
                    bench.show(p)
                    select_channel(bench, c)
                    rgb = bench.capture(frames=3)
                    measured = rects[ci].amplitude(rgb)
                    pred = pred+(torch.from_numpy(measured).to(device)-pred).detach()
                    drain(f'{label}_{step}')
                loss = (pred-t).square().mean()
                value = float(loss.detach())
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite loss')
                if value < best:
                    best, best_phase = value, phase.detach().clone()
                history.append(dict(step=step, mse=value, best=best))
                if step % 20 == 0 or step == steps:
                    save_json(out/f'history_{label}.json', history)
                    np.save(out/f'best_{label}.npy', best_phase.cpu().numpy())
                    progress('optimize', method=label, step=step, limit=steps, mse=value, best=best)
                if step >= 200 and step % 20 == 0:
                    previous = min(x['mse'] for x in history[:-20])
                    plateau = plateau+1 if previous-best < max(1e-6, previous*.001) else 0
                    if plateau >= 5:
                        break
                if step < steps:
                    loss.backward()
                    opt.step()
            save_json(out/f'history_{label}.json', history)
            return best_phase

        # Short real capture before expensive optimization, with complete source phase.
        if not args.resume:
            raw, sensor = capture(phases['raw_dph'], 'raw_dph_preflight')
            np.savez_compressed(out/'raw_dph_preflight.npz', amplitude=raw, sensor=sensor)
            Image.fromarray(np.uint8(np.clip(srgb_preview(raw)*255, 0, 255))).save(out/'raw_dph_preflight.png')
        for ci, (c, r, ch) in enumerate(zip(configs, rects, CHANNELS)):
            t = torch.from_numpy(target[..., ci]).to(device)
            asm = channel_model(c, r, device)
            with torch.no_grad():
                initial = dpac(asm, t)
            phases['dpac'][..., ci] = initial.cpu().numpy()
            path = out/f'best_{ch}_asm.npy'
            p = (torch.from_numpy(np.load(path)).to(device) if args.resume and path.exists() else
                 optimize(asm, t, c, initial, args.model_steps, f'{ch}_asm', ci))
            phases['asm'][..., ci] = p.cpu().numpy()
            saved = torch.load(trained/'models'/f'{ch}_best.pt', map_location='cpu', weights_only=False)
            if saved['config'] != c or saved['plan_sha256'] != read_json(trained/'protocol.json')['plan_sha256']:
                raise ValueError('Model provenance mismatch')
            model = PaperCNNpropCNN(c, device)
            model.load_learnable(saved['weights'])
            model.eval().requires_grad_(False)
            del saved
            path = out/f'best_{ch}_cnn.npy'
            p = (torch.from_numpy(np.load(path)).to(device) if args.resume and path.exists() else
                 optimize(model, t, c, initial, args.model_steps, f'{ch}_cnn', ci))
            phases['cnn'][..., ci] = p.cpu().numpy()
            path = out/f'best_{ch}_citl.npy'
            p = (torch.from_numpy(np.load(path)).to(device) if args.resume and path.exists() else
                 optimize(model, t, c, p, args.citl_steps, f'{ch}_citl', ci, camera=True))
            phases['citl'][..., ci] = p.cpu().numpy()
            for name in phases:
                encoded, restored = canonical(phases[name], base)
                np.save(out/f'phase_{name}.npy', restored)
                Image.fromarray(encoded).save(out/f'slm_{name}.png')
            del model, asm, p, initial, t
            gc.collect()
            torch.cuda.empty_cache()

        encoded_phases = {k: canonical(v, base)[1] for k, v in phases.items()}
        records, captures = [], {k: [] for k in phases}
        rng = np.random.default_rng(20261026)
        for repeat in range(3):
            for name in rng.permutation(list(phases)):
                progress('evaluate_capture', method=str(name), repeat=repeat)
                a, sensor = capture(encoded_phases[name], f'{name}_{repeat}')
                captures[name].append(a)
                np.savez_compressed(out/f'capture_{name}_{repeat}.npz', amplitude=a, sensor=sensor)
                metrics = [image_metrics(torch.from_numpy(a[..., ci]), torch.from_numpy(target[..., ci])) for ci in range(3)]
                records.append(dict(method=str(name), repeat=repeat, channels=metrics))
        end = drift('end')
        drift_fraction = end/reference-1
        save_json(out/'metrics.json', dict(records=records, drift_fraction=drift_fraction.tolist()))
        # Final displayed pattern is the measured best candidate, selected below.
        summary = {}
        for name, arrays in captures.items():
            selected = [r for r in records if r['method'] == name]
            summary[name] = {key: float(np.mean([c[key] for r in selected for c in r['channels']]))
                             for key in selected[0]['channels'][0] if key != 'fitted_amplitude_scale'}
            summary[name]['fixed_white_mse_repeat_sd'] = float(np.std([
                np.mean([c['fixed_white_mse'] for c in r['channels']]) for r in selected], ddof=1))
            mean = np.mean(arrays, axis=0)
            np.save(out/f'mean_{name}.npy', mean)
            Image.fromarray(np.uint8(np.clip(srgb_preview(mean)*255, 0, 255))).save(out/f'fixed_{name}.png')
            scales = np.array([(mean[..., ci]*target[..., ci]).mean()/max((mean[..., ci]**2).mean(), 1e-12) for ci in range(3)])
            Image.fromarray(np.uint8(np.clip(srgb_preview(mean*scales)*255, 0, 255))).save(out/f'fitted_{name}.png')
        best = min(('dpac', 'asm', 'cnn', 'citl'), key=lambda k: summary[k]['fixed_white_mse'])
        bench.show(encoded_phases[best])
        save_json(out/'summary.json', dict(methods=summary, best_fixed_white_method=best,
                  drift_fraction=drift_fraction.tolist(), excessive_drift=bool(np.any(abs(drift_fraction)>.2)),
                  hashes_unchanged=all(sha(Path(p)) == h for p,h in hashes.items())))
        progress('completed', best=best, summary=summary)

    thumbs = [('Target', out/'target.png')]+[(k, out/f'fixed_{k}.png') for k in phases]
    sheet = Image.new('RGB', (840*3, 510*2), '#151a22')
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(thumbs):
        im = Image.open(path)
        im.thumbnail((820, 470))
        x, y = (i%3)*840+10, (i//3)*510+30
        sheet.paste(im, (x, y))
        draw.text((x, y-22), label, fill='white')
    sheet.save(out/'comparison_fixed.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--trained', default='results/paper_cnnpropcnn_20261024')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model-steps', type=int, default=1000)
    parser.add_argument('--citl-steps', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        import traceback
        out = Path(args.out)
        if out.exists():
            failure = out/'failure.txt'
            if failure.exists():
                failure = out/f'failure_{int(time.time())}.txt'
            failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise
