"""Persistent joint RGB fitting. Only training pairs contribute gradients."""
import copy
import hashlib
import json
import time

import numpy as np
import torch

from .optics import CompactForwardModel, crop_center, load_forward_model
from .rgb_run import CHANNELS, channel_model


def fit_rgb(worker):
    # Imported lazily because Campaign delegates to this module.
    from .campaign import append_json, decode_phase, remove_owned, save_checkpoint
    w = worker
    device = w.args.device
    cycle = w.state['cycle']
    marker = w.directory/'weights'/f'rgb_{cycle:04d}.pt'
    if marker.exists():
        w.progress(f'cycle {cycle} · RGB 공동 학습 완료 기록 보존')
        return
    latest_path, best_path = [w.directory/'models'/f'rgb_{kind}.pt' for kind in ('latest', 'best')]
    saved = torch.load(latest_path, map_location='cpu', weights_only=True) if latest_path.exists() else None
    if saved and saved['configs'] != w.configs:
        raise ValueError('RGB continuation acquisition/optics configuration differs')
    nets = []
    for ch, config, rectifier in zip(CHANNELS, w.configs, w.rectifiers):
        legacy = w.directory/'models'/f'{ch}_best.pt'
        net = (load_forward_model(legacy, config, device) if legacy.exists() else
               CompactForwardModel(channel_model(config, rectifier, device), config['roi_shape']).to(device))
        if saved:
            branch = saved['branches'][ch]
            net.correction.load_state_dict(branch['correction'])
            with torch.no_grad():
                net.log_gain.copy_(branch['log_gain'])
                net.asm.illumination.copy_(branch['illumination'])
        net.correction.requires_grad_(True)
        net.log_gain.requires_grad_(True)
        nets.append(net.train())
    parameters = [p for net in nets for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, min_lr=1e-6)
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
    global_epoch = saved['global_epoch'] if saved else 0
    samples_seen = saved['samples_seen'] if saved else 0
    patience = saved['patience'] if saved else 0
    best_val = saved['best_validation_mse'] if saved else float('inf')
    same_cycle = saved is not None and saved['cycle'] == cycle
    local_epoch = saved['local_epoch'] if same_cycle else 0
    offset = saved.get('sample_offset', 0) if same_cycle else 0
    records = [r for r in w.pool if r['split'] in ('train', 'val')]
    fingerprint = hashlib.sha256(json.dumps([(r['serial'], r['split']) for r in records]).encode()).hexdigest()
    if same_cycle and saved['pool_fingerprint'] != fingerprint:
        raise ValueError('Cannot change a partially trained RGB cycle pool')
    train_ids = [i for i, r in enumerate(records) if r['split'] == 'train']
    val_ids = [i for i, r in enumerate(records) if r['split'] == 'val']
    if not train_ids or not val_ids:
        raise ValueError('Joint RGB fitting requires training and validation pairs')
    w.progress(f'cycle {cycle} · RGB 공동 학습 준비 · optimizer 복원={saved is not None}')
    # Half precision CPU caches; physics FFTs and CNN optimization remain float32.
    cached = []
    with torch.no_grad():
        for row in records:
            w.guard()
            with np.load(w.directory/'pool'/row['file']) as pair:
                asm = []
                for ci, (net, config) in enumerate(zip(nets, w.configs)):
                    phase = torch.from_numpy(decode_phase(pair['phase_rgb_u8'][..., ci], config['invert_phase'])).to(device)
                    asm.append(crop_center(net.asm(phase).abs(), config['roi_shape']).cpu().half())
                cached.append((asm, torch.from_numpy(pair['amplitude_rgb'].astype(np.float16))))
    key = w.monitor.start('cnn', 'rgb', w.args.epochs, 'recorded_camera_cnn', f'cycle {cycle} · RGB 공동 CNN')
    stream = w.monitor.state['streams'][key]
    stream['optimizer_restored'] = saved is not None
    stream['objective'] = 'mean RGB amplitude MSE; one Adam step per RGB pair'
    minimum = min(getattr(w.args, 'min_epochs', 50), w.args.epochs)
    limit_patience = getattr(w.args, 'cnn_patience', 30)
    stream['stopping_policy'] = dict(max_epochs=w.args.epochs, min_epochs=minimum, patience=limit_patience)

    def predict(ci, a):
        a = a.to(device=device, dtype=torch.float32)
        net = nets[ci]
        multiplier = torch.exp(net.log_gain+.5*torch.tanh(net.correction(a[None, None])[0, 0]))
        return a*multiplier, multiplier

    def checkpoint():
        branches = {}
        for ch, config, net in zip(CHANNELS, w.configs, nets):
            branches[ch] = dict(architecture='single_plane_asm_residual_cnn_8', config=config,
                correction={k:v.detach().cpu().clone() for k,v in net.correction.state_dict().items()},
                log_gain=net.log_gain.detach().cpu().clone(), illumination=net.asm.illumination.detach().cpu().clone(),
                cycle=cycle, epoch=local_epoch, global_epoch=global_epoch,
                validation_mse=metrics['val_mse_rgb'][CHANNELS.index(ch)],
                training_sample_serials=[records[i]['serial'] for i in train_ids])
        return dict(schema=2, architecture='joint_rgb_independent_physics_residual_cnn_8', configs=w.configs,
                    branches=branches, optimizer=copy.deepcopy(optimizer.state_dict()),
                    scheduler=copy.deepcopy(scheduler.state_dict()), global_epoch=global_epoch,
                    samples_seen=samples_seen, cycle=cycle, local_epoch=local_epoch,
                    sample_offset=offset, pool_fingerprint=fingerprint, patience=patience,
                    best_validation_mse=best_val, metrics=metrics,
                    optimizer_origin='restored_rgb' if saved else 'new; legacy weights have no saved Adam moments')

    def measure(ep):
        result = dict(epoch=ep, cycle=cycle, channel='rgb', global_epoch=global_epoch,
                      amplitude_gain=float(np.mean([float(n.log_gain.exp().detach()) for n in nets])),
                      amplitude_gain_rgb=[float(n.log_gain.exp().detach()) for n in nets],
                      parameter_l2={f'{ch}.{k}':float(p.detach().norm()) for ch,n in zip(CHANNELS,nets)
                                    for k,p in n.named_parameters() if p.requires_grad},
                      learning_rate=optimizer.param_groups[0]['lr'], samples_seen=samples_seen,
                      optimizer_restored=saved is not None)
        with torch.no_grad():
            for split, ids in [('train', train_ids), ('val', val_ids)]:
                losses = []
                for index in ids:
                    w.guard()
                    a, b = cached[index]
                    losses.append([float((predict(ci,a[ci])[0]-b[...,ci].to(device=device,dtype=torch.float32)).square().mean())
                                   for ci in range(3)])
                result[split+'_mse_rgb'] = np.mean(losses, axis=0).tolist()
                result[split+'_mse'] = float(np.mean(losses))
        if not np.isfinite(result['val_mse']):
            raise RuntimeError('Non-finite joint RGB validation loss')
        return result

    def publish():
        a, b = cached[val_ids[0]]
        with torch.no_grad():
            predicted, multipliers = zip(*(predict(ci, a[ci]) for ci in range(3)))
        w.monitor.rgb_epoch(key, local_epoch, metrics, nets,
                            torch.stack(a,-1), b, torch.stack(predicted,-1), torch.stack(multipliers,-1))
        append_json(w.directory/'training.jsonl', dict(time=time.time(), **metrics))
        w.progress(f'cycle {cycle} · RGB epoch {local_epoch}/{w.args.epochs} · val={metrics["val_mse"]:.6f}')

    metrics = measure(local_epoch)
    metrics['gradient_l2_mean'] = 0.
    if metrics['val_mse'] < best_val:
        best_val = metrics['val_mse']
        value = checkpoint()
        save_checkpoint(best_path, value)
        for ch in CHANNELS:
            save_checkpoint(w.directory/'models'/f'{ch}_best.pt', value['branches'][ch])
    publish()
    reason = 'epoch_limit'
    try:
        while local_epoch < w.args.epochs:
            gradient_norms = []
            order = np.random.default_rng(42+global_epoch).permutation(train_ids)
            for position in range(offset, len(order)):
                w.guard()
                a, b = cached[int(order[position])]
                optimizer.zero_grad(set_to_none=True)
                for ci in range(3):
                    predicted, _ = predict(ci, a[ci])
                    ((predicted-b[...,ci].to(device=device,dtype=torch.float32)).square().mean()/3).backward()
                gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, 1.)))
                optimizer.step()
                samples_seen += 1
                offset = position+1
            # Keep the completed sample offset until validation finishes. A stop
            # during validation resumes its measurement without repeating updates.
            metrics = measure(local_epoch+1)
            offset = 0
            local_epoch += 1
            global_epoch += 1
            metrics.update(global_epoch=global_epoch, gradient_l2_mean=float(np.mean(gradient_norms)) if gradient_norms else 0.,
                           gradient_sample_count=len(gradient_norms))
            scheduler.step(metrics['val_mse'])
            if metrics['val_mse'] < best_val:
                best_val, patience = metrics['val_mse'], 0
                value = checkpoint()
                save_checkpoint(best_path, value)
                for ch in CHANNELS:
                    save_checkpoint(w.directory/'models'/f'{ch}_best.pt', value['branches'][ch])
            else:
                patience += 1
            metrics.update(patience=patience, best_val_mse=best_val,
                           learning_rate=optimizer.param_groups[0]['lr'])
            value = checkpoint()
            save_checkpoint(latest_path, value)
            for ch in CHANNELS:
                save_checkpoint(w.directory/'models'/f'{ch}_latest.pt', value['branches'][ch])
            publish()
            if local_epoch >= minimum and patience >= limit_patience:
                reason = 'validation_plateau'
                break
    except InterruptedError:
        save_checkpoint(latest_path, checkpoint())
        w.monitor.end_stream(key, 'user_stop', dict(epoch=local_epoch, sample_offset=offset))
        raise
    termination = dict(reason=reason, epoch=local_epoch, maximum=w.args.epochs,
                       minimum=minimum, patience=patience, global_epoch=global_epoch,
                       best_validation_mse=best_val, optimizer_restored=saved is not None)
    w.monitor.end_stream(key, reason, termination)
    append_json(w.directory/'training_termination.jsonl', dict(time=time.time(), cycle=cycle, **termination))
    learnable = {ch:dict(correction={k:v.detach().cpu().clone() for k,v in net.correction.state_dict().items()},
                        log_gain=net.log_gain.detach().cpu().clone()) for ch,net in zip(CHANNELS,nets)}
    save_checkpoint(marker, dict(termination, learnable=learnable))
    for old in sorted((w.directory/'weights').glob('rgb_*.pt'))[:-8]:
        remove_owned(old, w.directory/'weights')
    del nets, cached, optimizer, scheduler, parameters
    if device == 'cuda':
        torch.cuda.empty_cache()
