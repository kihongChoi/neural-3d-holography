"""Run with .venv/Scripts/python.exe -m benchtop.run --help."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .hardware import Bench, encode_phase
from .optics import ASM, CompactForwardModel, make_target, crop_center, optimize_phase, load_forward_model
from .calibration import calibrate, Rectifier, channel_intensity


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def save_image(path, value):
    Image.fromarray(np.round(np.clip(value, 0, 1)*255).astype(np.uint8)).save(path)


def memory():
    if not torch.cuda.is_available():
        return {'device': 'cpu'}
    return {'device': torch.cuda.get_device_name(0),
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}


def preview(path, target, before, after, labels=('Target amplitude', 'Before', 'After')):
    arrays = [target, before, after]
    h, w = target.shape
    canvas = Image.new('RGB', (w*3, h+36), '#141b27')
    draw = ImageDraw.Draw(canvas)
    for i, (array, label) in enumerate(zip(arrays, labels)):
        canvas.paste(Image.fromarray((np.clip(array, 0, 1)*255).astype('uint8')).convert('RGB'), (i*w, 36))
        draw.text((i*w+10, 10), label, fill='white')
    canvas.save(path)


def initialize(config, device):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return ASM(config, device)


def preflight(bench, config, output):
    h, w = config['slm_shape']
    flat_phase = np.zeros((h, w), np.float32)
    flat = bench.capture(flat_phase)
    repeat = bench.capture(flat_phase)
    random = np.random.default_rng(config['seed']).uniform(-np.pi, np.pi, (h, w)).astype(np.float32)
    changed = bench.capture(random)
    noise = float(np.mean(np.abs(repeat-flat)))
    response = float(np.mean(np.abs(changed-flat)))
    result = {'repeat_mae': noise, 'pattern_change_mae': response,
              'response_to_noise': response/max(noise, 1e-8),
              'flat_mean_rgb': flat.mean((0, 1)).tolist(), 'frames': bench.frame_metadata.copy()}
    np.save(output/'preflight_flat_rgb.npy', flat)
    np.save(output/'preflight_changed_rgb.npy', changed)
    save_json(output/'preflight.json', result)
    if response < max(noise*4, 0.005):
        raise RuntimeError(f'Optical response too small ({response:.4g}, noise {noise:.4g}); '
                           'check that SLM phase modulation reaches the camera.')
    print(f'Preflight optical response/noise={result["response_to_noise"]:.1f}', flush=True)
    return result


def timing_scan(bench, config, output):
    rows = []
    original_delay = float(bench.camera.TriggerDelay.GetValue())
    try:
        for delay in range(22, 16023, 1000):
            bench.camera.TriggerDelay.SetValue(float(delay))
            rgb = bench.capture()
            h, w = rgb.shape[:2]
            values = rgb[h//4:3*h//4, w//4:3*w//4].mean((0, 1))
            row = {'delay_us': delay, 'mean_rgb': values.tolist(),
                   'saturated_fraction': bench.frame_metadata[-1]['saturated_fraction']}
            rows.append(row)
            print(row, flush=True)
    finally:
        bench.camera.TriggerDelay.SetValue(original_delay)
    save_json(output/'timing_scan.json', rows)


def offline(config, output, device, steps, checkpoint=None):
    model = initialize(config, device)
    if checkpoint:
        model = load_forward_model(checkpoint, config, device)
    target = make_target(config['roi_shape'], device=device)
    initial = 0.3*torch.randn(config['slm_shape'], device=device)
    def log(step, row, image):
        if step % 10 == 0:
            print(f'Simulation {step}: MSE={row["amplitude_mse"]:.6f}', flush=True)
    best, history = optimize_phase(model, target, config, iterations=steps, initial=initial, callback=log)
    save_json(output/'simulation_metrics.json', {'mode': 'SIMULATION_ONLY', 'history': history, 'memory': memory()})
    np.save(output/'simulation_phase.npy', best.cpu().numpy())
    with torch.no_grad():
        before = crop_center(model(initial).abs(), config['roi_shape']).cpu().numpy()*history[0]['scale']
        after = crop_center(model(best).abs(), config['roi_shape']).cpu().numpy()
        t = target.cpu().numpy()
        after *= np.mean(after*t)/np.mean(after**2)
    preview(output/'simulation_preview.png', t, before, after)
    print(json.dumps({'initial_mse': history[0]['amplitude_mse'],
                      'best_mse': min(x['amplitude_mse'] for x in history), 'memory': memory()}), flush=True)


def citl(bench, rectifier, model, config, output, steps):
    device = model.kernel.device
    target = make_target(config['roi_shape'], device=device)
    initial = 0.3*torch.randn(config['slm_shape'], device=device)
    print('Simulation warm start (20 iterations)', flush=True)
    initial, _ = optimize_phase(model, target, config, iterations=20, initial=initial)
    np.save(output/'initial_phase.npy', initial.cpu().numpy())
    def capture(phase):
        return rectifier.amplitude(bench.capture(phase))
    first = capture(initial.cpu().numpy())
    np.save(output/'initial_camera_amplitude.npy', first)
    images = {}
    def log(step, row, image):
        print(f'CITL {step}/{steps}: camera MSE={row["amplitude_mse"]:.6f}, PSNR={row["psnr_db"]:.2f} dB', flush=True)
        if step == 0:
            images['first'] = image
        save_json(output/'citl_progress.json', row)
    best, history = optimize_phase(model, target, config, iterations=steps, initial=initial,
                                  capture=capture, callback=log, lr=0.02)
    # Repeated independent before/after captures reduce best-of-noise bias.
    validation = []
    for _ in range(3):
        for label, phase in [('before', initial), ('after', best)]:
            amp = capture(phase.cpu().numpy())
            t = target.cpu().numpy()
            scale = float(np.mean(amp*t)/max(np.mean(amp**2), 1e-8))
            loss = float(np.mean((scale*amp-t)**2))
            validation.append({'label': label, 'mse': loss, 'psnr_db': -10*math.log10(max(loss, 1e-12))})
            images[label] = amp*scale
    final = capture(best.cpu().numpy())
    np.save(output/'best_phase.npy', best.cpu().numpy())
    np.save(output/'best_camera_amplitude.npy', final)
    np.save(output/'target_amplitude.npy', target.cpu().numpy())
    Image.fromarray(encode_phase(best.cpu().numpy(), config['invert_phase'])).save(output/'best_phase_slm.png')
    report = {'mode': 'REAL_CAMERA_CITL', 'config': config, 'history': history,
              'repeat_validation': validation, 'memory': memory(), 'frames': bench.frame_metadata.copy()}
    save_json(output/'citl_metrics.json', report)
    preview(output/'citl_preview.png', target.cpu().numpy(), images['before'], images['after'],
            ('Target amplitude', 'Camera: simulation warm start', 'Camera: CITL best'))
    return best


def capture_dataset(bench, rectifier, model, config, output, count):
    if count < 6:
        raise ValueError('Use at least 6 pairs for a train/validation split.')
    dataset = output/'pairs'
    dataset.mkdir(exist_ok=True)
    entries = []
    for i in range(count):
        target = make_target(config['roi_shape'], i, model.kernel.device)
        initial = (0.25+0.1*(i % 4))*torch.randn(config['slm_shape'], device=model.kernel.device)
        phase, _ = optimize_phase(model, target, config, iterations=5+i % 8, initial=initial)
        amplitude = rectifier.amplitude(bench.capture(phase.cpu().numpy()))
        filename = f'pair_{i:03d}.npz'
        np.savez_compressed(dataset/filename, phase=phase.cpu().numpy().astype(np.float32), amplitude=amplitude)
        entries.append({'file': filename, 'split': 'val' if i % 4 == 3 else 'train'})
        print(f'Captured real training pair {i+1}/{count}', flush=True)
    save_json(dataset/'manifest.json', {'mode': 'REAL_CAMERA', 'config': config, 'pairs': entries,
                                       'calibration': rectifier.report, 'intensity_scale': rectifier.intensity_scale,
                                       'frame_metadata': bench.frame_metadata.copy()})
    return dataset


def train_forward(model, config, dataset, output, epochs):
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest['mode'] != 'REAL_CAMERA' or manifest['config'] != config:
        raise ValueError('Training requires matching real camera data and config.')
    device = model.kernel.device
    network = CompactForwardModel(model, config['roi_shape']).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
    best_val = float('inf')
    history = []
    def epoch(split, training):
        total, n = 0., 0
        for row in manifest['pairs']:
            if row['split'] != split:
                continue
            with np.load(dataset/row['file']) as pair:
                phase = torch.from_numpy(pair['phase']).to(device)
                target = torch.from_numpy(pair['amplitude']).to(device)
            with torch.set_grad_enabled(training):
                prediction = network(phase)
                loss = (prediction-target).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite forward-model loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            total += float(loss.detach())
            n += 1
        if not n:
            raise ValueError(f'Empty {split} dataset')
        return total/n
    baseline = {'epoch': 0, 'train_mse': epoch('train', False), 'val_mse': epoch('val', False)}
    history.append(baseline)
    for ep in range(epochs+1):
        if ep:
            row = {'epoch': ep, 'train_mse': epoch('train', True), 'val_mse': epoch('val', False)}
            history.append(row)
        else:
            row = baseline
        print(f'Forward model {ep}: train={row["train_mse"]:.6f}, val={row["val_mse"]:.6f}', flush=True)
        if row['val_mse'] < best_val:
            best_val = row['val_mse']
            torch.save({'architecture': 'single_plane_asm_residual_cnn_8', 'config': config,
                        'correction': {k: v.detach().cpu() for k, v in network.correction.state_dict().items()},
                        'log_gain': network.log_gain.detach().cpu(),
                        'illumination': model.illumination.detach().cpu(), 'epoch': ep,
                        'validation_mse': best_val}, output/'forward_model.pt')
    save_json(output/'forward_training.json', {'mode': 'REAL_CAMERA_PAIRS', 'history': history,
                                              'best_val_mse': best_val, 'memory': memory(),
                                              'train_pairs': sum(r['split']=='train' for r in manifest['pairs']),
                                              'val_pairs': sum(r['split']=='val' for r in manifest['pairs'])})


def evaluate_checkpoint(config, dataset, checkpoint, output, device):
    """Independent checkpoint reload and held-out prediction, without hardware."""
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest['mode'] != 'REAL_CAMERA' or manifest['config'] != config:
        raise ValueError('Evaluation requires matching real camera pairs.')
    network = load_forward_model(checkpoint, config, device)
    rows = []
    with torch.no_grad():
        for row in manifest['pairs']:
            if row['split'] != 'val':
                continue
            with np.load(dataset/row['file']) as pair:
                phase = torch.from_numpy(pair['phase']).to(device)
                measured = torch.from_numpy(pair['amplitude']).to(device)
            baseline = crop_center(network.asm(phase).abs(), config['roi_shape'])
            predicted = network(phase)
            rows.append({'file': row['file'], 'asm_amplitude_mse': float((baseline-measured).square().mean()),
                         'trained_amplitude_mse': float((predicted-measured).square().mean())})
            if len(rows) == 1:
                preview(output/'model_validation_preview.png', measured.cpu().numpy(),
                        baseline.cpu().numpy(), predicted.cpu().numpy(),
                        ('Held-out camera amplitude', 'ASM prediction', 'Reloaded trained model prediction'))
    if not rows:
        raise ValueError('No validation pairs available.')
    result = {'mode': 'RELOADED_CHECKPOINT_REAL_CAMERA_VALIDATION', 'pairs': rows,
              'mean_asm_mse': float(np.mean([r['asm_amplitude_mse'] for r in rows])),
              'mean_trained_mse': float(np.mean([r['trained_amplitude_mse'] for r in rows]))}
    save_json(output/'checkpoint_validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['simulate', 'scan-timing', 'preflight', 'calibrate', 'citl', 'train', 'evaluate', 'smoke'])
    p.add_argument('--config', default='configs/benchtop.json')
    p.add_argument('--out', default=None)
    p.add_argument('--calibration', default='calibration/benchtop/green_15mm')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--pairs', type=int, default=12)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--checkpoint', help='Measured forward_model.pt to use for simulate, citl, or evaluate.')
    p.add_argument('--wait-for-display', type=float, default=0,
                   help='Wait this many seconds for an unlocked, unobstructed SLM display before connecting the camera.')
    args = p.parse_args()
    if args.checkpoint and args.command not in ['simulate', 'citl', 'evaluate']:
        p.error('--checkpoint is supported for simulate, citl, and evaluate.')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['distance_m'] <= 0 or config['relay_magnification'] == 0:
        raise ValueError('Use positive propagation distance after the relayed plane and nonzero magnification.')
    output = Path(args.out or f'results/benchtop_{time.strftime("%Y%m%d_%H%M%S")}_{args.command}')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'config.json', config)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2*2**30 else 'cpu'
    try:
        if args.command == 'evaluate':
            evaluate_checkpoint(config, output/'pairs', args.checkpoint or output/'forward_model.pt', output, device)
            return
        if args.command == 'simulate':
            offline(config, output, device, args.steps, args.checkpoint)
            return
        if args.command == 'train':
            rectifier = Rectifier(args.calibration, config)
            model = initialize(config, device)
            model.illumination.copy_(torch.from_numpy(rectifier.source_amplitude()).to(device))
            train_forward(model, config, output/'pairs', output, args.epochs)
            return
        with Bench(config, display_wait_s=args.wait_for_display) as bench:
            if args.command == 'scan-timing':
                timing_scan(bench, config, output)
                return
            preflight(bench, config, output)
            if args.command == 'preflight':
                return
            if args.command in ['calibrate', 'smoke']:
                calibrate(bench, config, args.calibration)
            if args.command == 'calibrate':
                return
            rectifier = Rectifier(args.calibration, config)
            model = initialize(config, device)
            model.illumination.copy_(torch.from_numpy(rectifier.source_amplitude()).to(device))
            if args.checkpoint:
                model = load_forward_model(args.checkpoint, config, device)
            best = citl(bench, rectifier, model, config, output, args.steps)
            if args.command == 'smoke':
                dataset = capture_dataset(bench, rectifier, model, config, output, args.pairs)
                bench.show(best.cpu().numpy())
        # Release the hardware before the offline forward-model fitting.
        if args.command == 'smoke':
            train_forward(model, config, dataset, output, args.epochs)
        save_json(output/'completed.json', {'command': args.command, 'status': 'complete', 'memory': memory()})
    except Exception as exc:
        save_json(output/'failure.json', {'command': args.command, 'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
