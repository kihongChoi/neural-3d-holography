"""RGB CFS acquisition, fixed-white CITL, and measured forward-model fitting.

Run: python -m benchtop.rgb_run all --out results/rgb_15mm
Stages are restartable; existing measurements are never silently overwritten.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .hardware import Bench, encode_phase
from .calibration import calibrate, Rectifier, channel_intensity
from .optics import ASM, CompactForwardModel, crop_center, optimize_phase, load_forward_model
from .run import save_json, save_image, memory
from .targets import create_div2k_plan, load_target, target_entry

CHANNELS = ('red', 'green', 'blue')
WAVELENGTHS = (638e-9, 520e-9, 450e-9)
DELAYS = (22., 6022., 12022.)
DARK_DELAYS = (2022., 8022., 14022.)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def target_rgb(shape, target_id):
    """Reproducible extended targets with disjoint identities across splits.

    Values are amplitude in units of each channel's fixed flat-field white.
    No per-image brightness fit is used in the RGB experiment.
    """
    rng = np.random.default_rng(47011+int(target_id))
    h, w = shape
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    a = np.full((h, w, 3), .12, np.float32)
    for k in range(4):
        cx, cy = rng.uniform(-.65, .65, 2)
        sx, sy = rng.uniform(.18, .4, 2)
        if k % 2:
            r = np.sqrt(((x-cx)/sx)**2+((y-cy)/sy)**2)
            blob = np.exp(-((r-.8)/.3)**2)
        else:
            blob = np.exp(-(((x-cx)/sx)**4+((y-cy)/sy)**4))
        color = rng.uniform(.15, .9, 3)
        color[k % 3] = .95
        a += (blob[..., None]*color*.65).astype(np.float32)
    return np.minimum(a, .95)


def srgb_preview(amplitude):
    intensity = np.clip(np.asarray(amplitude)**2, 0, 1)
    return np.where(intensity <= .0031308, 12.92*intensity, 1.055*intensity**(1/2.4)-.055)


def channel_model(config, rectifier, device):
    model = ASM(config, device)
    model.illumination.copy_(torch.from_numpy(rectifier.source_amplitude()).to(device))
    return model


def select_channel(bench, config):
    bench.set_timing(config['exposure_us'], config['trigger_delay_us'])


def prepare(root, base, monitor=None):
    """Verify actual CFS routing, exposure, off-slot background, and geometry."""
    destination = root/'acquisition.json'
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; use acquire/train/final to continue.')
    rows = []
    with Bench(base) as bench:
        for ch, wavelength, delay, dark_delay in zip(CHANNELS, WAVELENGTHS, DELAYS, DARK_DELAYS):
            if monitor:
                monitor.message(f'{ch}: CFS 응답, 노출 및 기하 보정 측정 중')
            path = root/ch
            path.mkdir(parents=True, exist_ok=True)
            config = dict(base, channel=ch, wavelength_m=wavelength, trigger_delay_us=delay)
            exposure = 700.
            attempts = []
            for _ in range(5):
                timing = bench.set_timing(exposure, delay)
                flat = bench.capture(np.zeros(base['slm_shape'], np.float32), frames=3)
                signal = channel_intensity(flat, config)
                q = float(np.quantile(signal, .998))
                attempts.append(dict(timing, p998=q, mean_rgb=flat.mean((0, 1)).tolist()))
                if q < .65:
                    break
                exposure *= .5
            else:
                raise RuntimeError(f'{ch}: cannot avoid saturation with bounded exposure reduction')
            config.update(timing)
            bench.set_timing(config['exposure_us'], dark_delay)
            dark_rgb = bench.capture(frames=5)
            dark = channel_intensity(dark_rgb, config)
            if float(dark.mean()) > max(.01, float(signal.mean())*.1):
                raise RuntimeError(f'{ch}: selected off-slot contains too much light')
            select_channel(bench, config)
            repeat = channel_intensity(bench.capture(frames=3), config)
            noise = float(np.abs(repeat-signal).mean())
            rng = np.random.default_rng(218)
            random_phase = rng.uniform(-np.pi, np.pi, base['slm_shape']).astype(np.float32)
            responses = []
            for slot in range(3):
                phase = np.zeros(tuple(base['slm_shape'])+(3,), np.float32)
                phase[..., slot] = random_phase
                measured = channel_intensity(bench.capture(phase, frames=3), config)
                responses.append(float(np.abs(measured-signal).mean()))
            own = responses[CHANNELS.index(ch)]
            other = max(v for j, v in enumerate(responses) if j != CHANNELS.index(ch))
            diagnostic = {'channel': ch, 'config': config, 'exposure_attempts': attempts,
                          'off_slot_delay_us': dark_delay, 'off_slot_mean': float(dark.mean()),
                          'repeat_mae': noise, 'response_by_hdmi_rgb_slot': responses,
                          'response_to_noise': own/max(noise, 1e-8),
                          'off_slot_response_ratio': other/max(own, 1e-8)}
            save_json(path/'preflight.json', diagnostic)
            print(f'RGB preflight {ch}: {diagnostic}', flush=True)
            if own < max(.005, 4*noise) or other > max(4*noise, own*.35):
                raise RuntimeError(f'{ch}: CFS phase slot isolation failed; inspect preflight.json')
            cal = path/'calibration'
            cal.mkdir(exist_ok=True)
            np.save(cal/'dark_intensity.npy', dark)
            save_json(path/'config.json', config)
            report = calibrate(bench, config, cal)
            rectifier = Rectifier(cal, config)
            diagnostic.update(rms_native_sensor_px=report['rms_half_sensor_px']*2,
                              flat_intensity_scale=rectifier.intensity_scale)
            rows.append(diagnostic)
            save_json(root/'preparation_progress.json', rows)
        save_json(root/'acquisition_frames.json', bench.frame_metadata)
    save_json(destination, {'mode': 'MEASURED_RGB_CFS', 'channels': rows,
                           'normalization': 'Per-channel fixed flat-field white; not colorimetric calibration',
                           'background': 'Same-exposure off-slot background, not shutter-closed dark current'})


def capture_amplitude(bench, config, rectifier, phase=None, frames=None):
    select_channel(bench, config)
    return rectifier.amplitude(bench.capture(phase, frames=frames))


def generate_div2k_patterns(root, base, device, pairs, iterations=80, monitor=None):
    """Offline generation using the measured bench profile; no hardware connection."""
    plan = read_json(root/'targets/manifest.json')
    if plan['pairs'] != pairs:
        raise ValueError('Pattern count differs from frozen DIV2K plan')
    configurations = [read_json(root/ch/'config.json') for ch in CHANNELS]
    output = root/'patterns'
    manifest_path = output/'manifest.json'
    specification = {'iterations': iterations, 'seed': base['seed'], 'configurations': configurations}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get('generation') != specification:
            raise ValueError('Existing pattern bank uses different settings; preserve it and use a fresh run.')
        return manifest
    output.mkdir(exist_ok=True)
    models = [channel_model(c, Rectifier(root/ch/'calibration', c), device) for ch, c in zip(CHANNELS, configurations)]
    entries = []
    for slot in range(pairs):
        source = target_entry(root, slot)
        target = load_target(root, base['roi_shape'], slot)
        name = f'{source["split"]}_{slot:04d}_{Path(source["target_id"]).stem}'
        phase_path = output/f'{name}_phase_rgb.npy'
        if phase_path.exists():
            raise FileExistsError(f'Incomplete pattern bank exists: {phase_path}; preserve it and use a fresh run.')
        phases, histories = [], {}
        for ci, (ch, model, config) in enumerate(zip(CHANNELS, models, configurations)):
            torch.manual_seed(base['seed']+slot*3+ci)
            initial = .3*torch.randn(config['slm_shape'], device=device)
            live_trace = None
            if monitor:
                key = monitor.start('generate', ch, iterations, 'simulation',
                                    f'패턴 {slot+1}/{pairs} · {source["target_id"]}')
                live_trace = monitor.phase_trace(key, target[..., ci], iterations, every=20)
            phase, history = optimize_phase(model, torch.from_numpy(target[..., ci]).to(device), config,
                                           iterations=iterations, initial=initial, fixed_scale=1., trace=live_trace)
            phases.append(phase.cpu().numpy())
            histories[ch] = history
        stacked = np.stack(phases, axis=-1)
        np.save(phase_path, stacked)
        Image.fromarray(encode_phase(stacked, base['invert_phase'])).save(output/f'{name}_slm_rgb.png')
        save_image(output/f'{name}_target_rgb.png', srgb_preview(target))
        save_json(output/f'{name}_optimization.json', histories)
        entries.append({'file': f'{name}_slm_rgb.png', 'phase_file': phase_path.name,
                        'target_preview': f'{name}_target_rgb.png', 'target_slot': slot,
                        'target_id': source['target_id'], 'split': source['split'], 'source_image': source})
        print(f'DIV2K RGB phase {slot+1}/{pairs}: {source["target_id"]} -> {name}_slm_rgb.png', flush=True)
    manifest = {'mode': 'RGB888_CFS_PHASE_PATTERNS', 'shape_hwc': list(stacked.shape),
                'generation': specification, 'encoding': 'Inverted linear 8-bit phase LUT, RGB slot order',
                'phase_source': 'Measured-profile ASM; full RGB frame used for subsequent camera capture',
                'patterns': entries}
    save_json(manifest_path, manifest)
    return manifest


def acquire(root, base, device, steps, pairs, pattern_steps=80, monitor=None):
    if pairs < 12:
        raise ValueError('At least 12 targets needed for disjoint train/val/test subsets.')
    read_json(root/'acquisition.json')
    if (root/'targets/manifest.json').exists() and read_json(root/'targets/manifest.json')['pairs'] != pairs:
        raise ValueError('--pairs differs from the frozen DIV2K target plan')
    pattern_bank = (generate_div2k_patterns(root, base, device, pairs, pattern_steps, monitor)
                    if (root/'targets/manifest.json').exists() else None)
    with Bench(base) as bench:
        for ci, ch in enumerate(CHANNELS):
            path = root/ch
            if (path/'pairs/manifest.json').exists():
                print(f'{ch}: acquisition already complete, preserving it.', flush=True)
                continue
            if (path/'initial_phase.npy').exists():
                raise FileExistsError(f'{path}: partial capture exists; preserve it and use a fresh run directory.')
            config = read_json(path/'config.json')
            rectifier = Rectifier(path/'calibration', config)
            model = channel_model(config, rectifier, device)
            select_channel(bench, config)
            target = torch.from_numpy(load_target(root, config['roi_shape'], 1000)[..., ci]).to(device)
            trace_dir = path/'citl_trace'
            trace_dir.mkdir(exist_ok=True)
            torch.manual_seed(base['seed']+ci)
            initial = .3*torch.randn(config['slm_shape'], device=device)
            initial, _ = optimize_phase(model, target, config, iterations=30, initial=initial, fixed_scale=1.)
            np.save(path/'initial_phase.npy', initial.cpu().numpy())
            np.save(path/'target_amplitude.npy', target.cpu().numpy())
            live_trace = None
            if monitor:
                key = monitor.start('citl', ch, steps, 'live_camera_citl', '실측 CITL · demo target')
                live_trace = monitor.phase_trace(key, target, steps)
            def capture(phase):
                return capture_amplitude(bench, config, rectifier, phase)
            def trace(step, row, phase, amplitude, gradient):
                if step % 5 == 0 or step == steps:
                    np.savez_compressed(trace_dir/f'step_{step:04d}.npz',
                                        phase=phase.cpu().numpy(), amplitude=amplitude.cpu().numpy(),
                                        gradient_roi=(np.zeros(config['roi_shape'], np.float32) if gradient is None
                                                      else crop_center(gradient, config['roi_shape']).cpu().numpy()))
                if gradient is not None:
                    row['phase_gradient_rms'] = float(gradient.square().mean().sqrt())
                print(f'{ch} CITL {step}/{steps}: fixed-white amplitude MSE {row["amplitude_mse"]:.6f}', flush=True)
                save_json(path/'citl_progress.json', row)
                if live_trace:
                    live_trace(step, row, phase, amplitude, gradient)
            best, history = optimize_phase(model, target, config, iterations=steps, lr=.025,
                                           initial=initial, capture=capture, trace=trace, fixed_scale=1.)
            np.save(path/'best_phase.npy', best.cpu().numpy())
            validation = []
            for repeat in range(3):
                for label, phase in [('before', initial), ('after', best)]:
                    measured = capture(phase.cpu().numpy())
                    np.save(path/f'{label}_camera_amplitude.npy', measured)
                    validation.append({'repeat': repeat, 'label': label,
                                       'amplitude_mse': float(np.mean((measured-target.cpu().numpy())**2))})
            save_json(path/'citl_metrics.json', {'mode': 'REAL_CAMERA_FIXED_WHITE_CITL', 'history': history,
                                                'repeat_validation': validation, 'target_id': 1000,
                                                'source_image': target_entry(root, 1000),
                                                'memory': memory()})
            dataset = path/'pairs'
            dataset.mkdir(exist_ok=True)
            entries = []
            # Every target identity belongs to exactly one split. CITL demo 1000 is excluded.
            for i in range(pairs):
                if monitor:
                    monitor.message(f'{ch}: 실제 RGB 패턴 데이터 획득 {i+1}/{pairs}')
                t = torch.from_numpy(load_target(root, config['roi_shape'], i)[..., ci]).to(device)
                if pattern_bank:
                    pattern = pattern_bank['patterns'][i]
                    rgb_phase = np.load(root/'patterns'/pattern['phase_file'])
                    phase_array = rgb_phase[..., ci].copy()
                    amplitude = capture(rgb_phase)
                else:
                    start = (.2+.1*(i % 4))*torch.randn(config['slm_shape'], device=device)
                    phase, _ = optimize_phase(model, t, config, iterations=12+i % 9,
                                              initial=start, fixed_scale=1.)
                    phase_array = phase.cpu().numpy()
                    amplitude = capture(phase_array)
                filename = f'pair_{i:03d}.npz'
                np.savez_compressed(dataset/filename, phase=phase_array, amplitude=amplitude,
                                    target=t.cpu().numpy())
                split = 'test' if i % 6 == 5 else 'val' if i % 6 == 4 else 'train'
                source = target_entry(root, i)
                if source and source['split'] != split:
                    raise ValueError('Dataset split differs from frozen target plan')
                entries.append({'file': filename, 'split': split, 'target_id': source['target_id'] if source else i,
                                'target_slot': i, 'source_image': source,
                                'rgb_pattern_file': pattern['phase_file'] if pattern_bank else None})
                print(f'{ch}: captured pair {i+1}/{pairs} [{split}]', flush=True)
            save_json(dataset/'manifest.json', {'mode': 'REAL_CAMERA', 'config': config, 'pairs': entries,
                                               'intensity_scale': rectifier.intensity_scale,
                                               'frame_metadata': bench.frame_metadata.copy()})
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
    if not pattern_bank:
        export_pattern_bank(root)


def export_pattern_bank(root):
    """Join the wavelength-specific phases into native RGB888 CFS files."""
    output = root/'patterns'
    output.mkdir(exist_ok=True)
    manifests = [read_json(root/ch/'pairs/manifest.json') for ch in CHANNELS]
    entries = []
    for rows in zip(*(m['pairs'] for m in manifests)):
        if len({(r['target_id'], r['split']) for r in rows}) != 1:
            raise ValueError('RGB channel target identities/splits do not align')
        phases = []
        for ch, row in zip(CHANNELS, rows):
            with np.load(root/ch/'pairs'/row['file']) as pair:
                phases.append(pair['phase'])
        phase = np.stack(phases, axis=-1)
        slot = rows[0].get('target_slot', len(entries))
        source = rows[0].get('source_image')
        stem = Path(str(rows[0]['target_id'])).stem
        name = f'{rows[0]["split"]}_{slot:04d}_{stem}_slm_rgb.png'
        Image.fromarray(encode_phase(phase, manifests[0]['config']['invert_phase'])).save(output/name)
        entries.append({'file': name, 'target_id': rows[0]['target_id'], 'target_slot': slot,
                        'split': rows[0]['split'], 'source_image': source})
    save_json(output/'manifest.json', {'mode': 'RGB888_CFS_PHASE_PATTERNS',
              'shape_hwc': list(phase.shape), 'encoding': 'Inverted linear 8-bit phase LUT, RGB slot order',
              'phase_source': 'ASM-generated per-channel patterns, real-camera pairs captured before RGB packing',
              'patterns': entries})


def train(root, device, epochs, monitor=None, source_root=None):
    """Cache fixed ASM predictions; only CNN weights and scalar log gain are learned."""
    source_root = Path(source_root or root)
    for ch in CHANNELS:
        path = root/ch
        source = source_root/ch
        path.mkdir(parents=True, exist_ok=True)
        if (path/'forward_training.json').exists():
            print(f'{ch}: training already complete, preserving it.', flush=True)
            continue
        if (path/'model_trace').exists() or (path/'forward_model.pt').exists():
            raise FileExistsError(f'{path}: partial training exists; use a fresh output directory.')
        config = read_json(source/'config.json')
        manifest = read_json(source/'pairs/manifest.json')
        if manifest['mode'] != 'REAL_CAMERA' or manifest['config'] != config:
            raise ValueError('Measured dataset configuration mismatch')
        if source.resolve() != path.resolve():
            save_json(path/'config.json', config)
            save_json(path/'training_source.json', {'source': str(source.resolve()), 'manifest': manifest})
        rectifier = Rectifier(source/'calibration', config)
        if monitor:
            monitor.message(f'{ch}: 저장된 실측 데이터로 고정 ASM 예측 준비 중 (카메라 촬영 없음)')
        torch.manual_seed(config['seed'])
        model = channel_model(config, rectifier, device)
        net = CompactForwardModel(model, config['roi_shape']).to(device)
        cached = []
        with torch.no_grad():
            for row in manifest['pairs']:
                with np.load(source/'pairs'/row['file']) as pair:
                    phase = torch.from_numpy(pair['phase']).to(device)
                    amp = crop_center(model(phase).abs(), config['roi_shape']).detach()
                    measured = torch.from_numpy(pair['amplitude']).to(device)
                cached.append((row, amp, measured))
        numerator = sum(float((a*b).sum()) for r, a, b in cached if r['split'] == 'train')
        denominator = sum(float(a.square().sum()) for r, a, b in cached if r['split'] == 'train')
        gain_baseline = numerator/denominator
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        trace_dir = path/'model_trace'
        trace_dir.mkdir(exist_ok=True)
        probe = next((a, b) for r, a, b in cached if r['split'] == 'val')
        history, best_val, best_epoch = [], float('inf'), 0
        live_key = (monitor.start('cnn', ch, epochs, 'recorded_camera_cnn', '실측 쌍 → CNN forward model')
                    if monitor else None)
        def predict(a):
            correction = net.correction(a[None, None])[0, 0]
            return a*torch.exp(net.log_gain+.5*torch.tanh(correction))
        def checkpoint(ep, val):
            return {'architecture': 'single_plane_asm_residual_cnn_8', 'config': config,
                    'correction': {k: v.detach().cpu().clone() for k, v in net.correction.state_dict().items()},
                    'log_gain': net.log_gain.detach().cpu().clone(),
                    'illumination': model.illumination.detach().cpu(), 'epoch': ep, 'validation_mse': val}
        for ep in range(epochs+1):
            gradients = []
            if ep:
                for index in np.random.default_rng(config['seed']+ep).permutation(len(cached)):
                    row, a, measured = cached[index]
                    if row['split'] != 'train':
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = (predict(a)-measured).square().mean()
                    loss.backward()
                    gradients.append(float(sum(p.grad.square().sum() for p in net.parameters() if p.grad is not None).sqrt()))
                    optimizer.step()
            with torch.no_grad():
                metrics = {'epoch': ep, 'amplitude_gain': float(net.log_gain.exp()),
                           'gradient_l2_mean': float(np.mean(gradients)) if gradients else 0.,
                           'parameter_l2': {k: float(v.norm()) for k, v in net.named_parameters()}}
                for split in ('train', 'val'):
                    metrics[split+'_mse'] = float(np.mean([float((predict(a)-b).square().mean())
                                                         for r, a, b in cached if r['split'] == split]))
                if not np.isfinite(metrics['val_mse']):
                    raise RuntimeError('Non-finite RGB model validation')
                history.append(metrics)
                # Save every epoch's actual weights; ASM buffers are identical and saved only in best checkpoint.
                torch.save({'epoch': ep, 'correction': checkpoint(ep, metrics['val_mse'])['correction'],
                            'log_gain': net.log_gain.detach().cpu().clone()}, trace_dir/f'epoch_{ep:03d}.pt')
                predicted = predict(probe[0])
                correction = net.correction(probe[0][None, None])[0, 0]
                np.savez_compressed(trace_dir/f'epoch_{ep:03d}.npz', asm_amplitude=probe[0].cpu().numpy(),
                                    camera_amplitude=probe[1].cpu().numpy(), prediction=predicted.cpu().numpy(),
                                    amplitude_multiplier=torch.exp(net.log_gain+.5*torch.tanh(correction)).cpu().numpy())
                if metrics['val_mse'] < best_val:
                    best_val, best_epoch = metrics['val_mse'], ep
                    torch.save(checkpoint(ep, best_val), path/'forward_model.pt')
            print(f'{ch} CNN epoch {ep}: train={metrics["train_mse"]:.6f}, val={metrics["val_mse"]:.6f}', flush=True)
            if monitor:
                monitor.model_epoch(live_key, ep, metrics, net, probe, predicted,
                                    torch.exp(net.log_gain+.5*torch.tanh(correction)))
        # Independent best-checkpoint reload and a test split never used for selection.
        restored = load_forward_model(path/'forward_model.pt', config, device)
        evaluation = []
        with torch.no_grad():
            for row, a, b in cached:
                if row['split'] not in ('val', 'test'):
                    continue
                with np.load(source/'pairs'/row['file']) as pair:
                    prediction = restored(torch.from_numpy(pair['phase']).to(device))
                evaluation.append(dict(row, asm_mse=float((a-b).square().mean()),
                                       gain_only_mse=float((a*gain_baseline-b).square().mean()),
                                       cnn_mse=float((prediction-b).square().mean())))
        save_json(path/'forward_training.json', {'mode': 'REAL_CAMERA_RGB_PAIRS', 'history': history,
                  'best_epoch': best_epoch, 'best_val_mse': best_val, 'evaluation': evaluation,
                  'train_fitted_amplitude_gain_baseline': gain_baseline,
                  'learnable_parameters': {k: list(v.shape) for k, v in net.named_parameters()},
                  'fixed_parameters': ['distance_m', 'wavelength_m', 'SLM pitch', 'aperture',
                                       'registration', 'source illumination estimate', 'linear phase LUT'],
                  'parameter_count': sum(p.numel() for p in net.parameters()), 'memory': memory()})
        del net, restored, model, cached
        if device == 'cuda':
            torch.cuda.empty_cache()


def final_capture(root, base, device, steps, monitor=None):
    """One RGB888 SLM hologram; gated RGB measurements of that same hologram."""
    output = root/'fullcolor'
    if (output/'metrics.json').exists():
        raise FileExistsError(f'{output}: completed measurements already exist.')
    output.mkdir(exist_ok=True)
    configurations = [read_json(root/ch/'config.json') for ch in CHANNELS]
    rectifiers = [Rectifier(root/ch/'calibration', c) for ch, c in zip(CHANNELS, configurations)]
    target = load_target(root, base['roi_shape'], 2000)  # Never used in fitting or selection.
    phase_sets = {'demo_before': [], 'demo_citl': [], 'new_asm': [], 'new_cnn': []}
    for ci, (ch, config, rectifier) in enumerate(zip(CHANNELS, configurations, rectifiers)):
        phase_sets['demo_before'].append(np.load(root/ch/'initial_phase.npy'))
        phase_sets['demo_citl'].append(np.load(root/ch/'best_phase.npy'))
        asm = channel_model(config, rectifier, device)
        cnn = load_forward_model(root/ch/'forward_model.pt', config, device)
        torch.manual_seed(config['seed']+600+ci)
        initial = .3*torch.randn(config['slm_shape'], device=device)
        t = torch.from_numpy(target[..., ci]).to(device)
        for name, model in [('new_asm', asm), ('new_cnn', cnn)]:
            live_trace = None
            if monitor:
                key = monitor.start(name, ch, steps, 'simulation', f'{name} · 새 타겟 위상 계산')
                live_trace = monitor.phase_trace(key, t, steps, every=10)
            phase, history = optimize_phase(model, t, config, iterations=steps, initial=initial,
                                            fixed_scale=1., trace=live_trace)
            phase_sets[name].append(phase.cpu().numpy())
            save_json(output/f'{name}_{ch}_optimization.json', history)
        del asm, cnn
    rows = []
    with Bench(base) as bench:
        # Refine a genuinely composite RGB888 frame with the learned gradients.
        # Other phase slots remain on their current holograms, so cross-slot effects
        # are present in every camera measurement used for these updates.
        joint = np.stack(phase_sets['new_cnn'], axis=-1)
        refinement_steps = min(40, steps)
        for ci, (ch, config, rectifier) in enumerate(zip(CHANNELS, configurations, rectifiers)):
            cnn = load_forward_model(root/ch/'forward_model.pt', config, device)
            trace_dir = output/f'citl_{ch}'
            trace_dir.mkdir(exist_ok=True)
            live_trace = None
            if monitor:
                key = monitor.start('composite_citl', ch, refinement_steps, 'live_camera_citl',
                                    'RGB 합성 프레임 · 실측 CITL')
                live_trace = monitor.phase_trace(key, target[..., ci], refinement_steps)
            def capture(phase):
                joint[..., ci] = phase
                return capture_amplitude(bench, config, rectifier, joint)
            def trace(step, row, phase, amplitude, gradient):
                if step % 5 == 0 or step == refinement_steps:
                    np.savez_compressed(trace_dir/f'step_{step:04d}.npz', phase=phase.cpu().numpy(),
                                        amplitude=amplitude.cpu().numpy(),
                                        gradient_roi=(np.zeros(config['roi_shape'], np.float32) if gradient is None
                                                      else crop_center(gradient, config['roi_shape']).cpu().numpy()))
                print(f'Composite RGB {ch} CITL {step}/{refinement_steps}: {row["amplitude_mse"]:.6f}', flush=True)
                if live_trace:
                    live_trace(step, row, phase, amplitude, gradient)
            best, history = optimize_phase(cnn, torch.from_numpy(target[..., ci]).to(device), config,
                                           initial=torch.from_numpy(joint[..., ci].copy()).to(device),
                                           iterations=refinement_steps, lr=.02, capture=capture,
                                           trace=trace, fixed_scale=1.)
            joint[..., ci] = best.cpu().numpy()
            save_json(output/f'citl_{ch}_metrics.json', history)
            del cnn
        phase_sets['new_cnn_citl'] = [joint[..., ci].copy() for ci in range(3)]
        for name, phases in phase_sets.items():
            if monitor:
                monitor.message(f'최종 RGB 실측 반복 평가 · {name}')
            stacked = np.stack(phases, axis=-1)
            np.save(output/f'{name}_phase_rgb.npy', stacked)
            Image.fromarray(encode_phase(stacked, base['invert_phase'])).save(output/f'{name}_slm_rgb.png')
            bench.show(stacked)
            measured_target = load_target(root, base['roi_shape'], 1000) if name.startswith('demo') else target
            repeats = []
            for repeat in range(3):
                amplitudes = [capture_amplitude(bench, c, r, frames=3) for c, r in zip(configurations, rectifiers)]
                measured = np.stack(amplitudes, axis=-1)
                repeats.append(measured)
                rows.append({'variant': name, 'repeat': repeat,
                             'amplitude_mse_rgb': ((measured-measured_target)**2).mean((0, 1)).tolist(),
                             'amplitude_mse': float(np.mean((measured-measured_target)**2))})
            mean = np.mean(repeats, axis=0)
            np.save(output/f'{name}_camera_amplitude_rgb.npy', mean)
            save_image(output/f'{name}_camera_rgb.png', srgb_preview(mean))
            print(f'Actual RGB composite {name}: {np.mean([r["amplitude_mse"] for r in rows if r["variant"] == name]):.6f}', flush=True)
        save_json(output/'frame_metadata.json', bench.frame_metadata)
    for name, identity in [('demo_target', 1000), ('new_target', 2000)]:
        t = load_target(root, base['roi_shape'], identity)
        np.save(output/f'{name}_amplitude_rgb.npy', t)
        save_image(output/f'{name}_rgb.png', srgb_preview(t))
    save_json(output/'metrics.json', {'mode': 'REAL_RGB888_CFS_DISPLAY_GATED_CAMERA_RECONSTRUCTION',
              'normalization': 'Fixed per-channel flat white; no per-image scale optimization',
              'preview': 'Gamma-encoded normalized sensor RGB; not CIE-calibrated color or a single integrated exposure',
              'new_target_id': 2000, 'new_target_source': target_entry(root, 2000),
              'demo_target_source': target_entry(root, 1000), 'repeat_validation': rows, 'memory': memory()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'targets', 'prepare', 'generate', 'acquire', 'train', 'final', 'report', 'patterns'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/benchtop.json')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--pairs', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--pattern-steps', type=int, default=80, help='Offline DIV2K phase optimization iterations.')
    parser.add_argument('--div2k-root', help='Directory containing DIV2K_train_HR and DIV2K_valid_HR; freezes target provenance.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--data-root', help='train only: reuse measured pairs in this folder; save new weights under --out.')
    args = parser.parse_args()
    if args.data_root and args.stage != 'train':
        parser.error('--data-root is supported only for train')
    torch.set_num_threads(4)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    base = read_json(args.config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    from .live import LiveMonitor
    monitor = LiveMonitor(root, args.stage, source=args.data_root)
    try:
        if args.div2k_root:
            plan = create_div2k_plan(root, args.div2k_root, base['roi_shape'], args.pairs, base['seed'])
            print(f'DIV2K: {plan["available_originals"]}; frozen {len(plan["targets"])} distinct original images.', flush=True)
        if args.stage == 'targets' and not (root/'targets/manifest.json').exists():
            raise ValueError('targets requires --div2k-root')
        if args.stage == 'patterns':
            if (root/'targets/manifest.json').exists():
                generate_div2k_patterns(root, base, args.device, args.pairs, args.pattern_steps, monitor)
            else:
                export_pattern_bank(root)
        if args.stage in ('all', 'prepare'):
            prepare(root, base, monitor)
        if args.stage == 'generate':
            generate_div2k_patterns(root, base, args.device, args.pairs, args.pattern_steps, monitor)
        if args.stage in ('all', 'acquire'):
            acquire(root, base, args.device, args.steps, args.pairs, args.pattern_steps, monitor)
        if args.stage in ('all', 'train'):
            train(root, args.device, args.epochs, monitor, source_root=args.data_root)
        if args.stage in ('all', 'final'):
            final_capture(root, base, args.device, args.steps, monitor)
        if args.stage in ('all', 'report'):
            from .visualization import create_report
            create_report(root)
        save_json(root/f'completed_{args.stage}.json', {'status': 'complete', 'seconds': time.time()-started,
                  'arguments': vars(args), 'memory': memory()})
    except BaseException as exc:
        monitor.finish(error=exc)
        save_json(root/f'failure_{args.stage}.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
    else:
        monitor.finish()


if __name__ == '__main__':
    main()
