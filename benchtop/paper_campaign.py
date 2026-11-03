"""Resumable paper-method single-plane experiment; real camera, no fallback.

1100 training phase/camera pairs per color (100 DPAC + 1000 SGD), fixed bank,
authors' two complex-field U-Nets, Adam 5e-4, batch 1, amplitude L1, patience 10.
All departures from the 8-plane paper are frozen in protocol.json.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import numpy as np
import torch
from PIL import Image
from .calibration import Rectifier, channel_intensity
from .campaign import append_json, decode_phase, save_checkpoint, exclusive_run, StorageBudget
from .hardware import Bench, encode_phase
from .live import atomic_json
from .optics import crop_center, optimize_phase, amplitude_loss
from .paper_model import PaperCNNpropCNN, dpac, mse_metrics, image_metrics
from .rgb_run import CHANNELS, read_json, channel_model, select_channel, srgb_preview
from .targets import prepare_image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_suffix('.pending')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def canonical(phase, config):
    encoded = encode_phase(phase, config['invert_phase'])
    encoded = encode_phase(decode_phase(encoded, config['invert_phase']), config['invert_phase'])
    restored = decode_phase(encoded, config['invert_phase'])
    if not np.array_equal(encode_phase(restored, config['invert_phase']), encoded):
        raise RuntimeError('SLM phase encoding does not round trip')
    return encoded, restored


def prior_valid_ids(results, exclude):
    """Conservatively exclude every DIV2K validation identity in older records."""
    found = set()
    for path in Path(results).rglob('*'):
        if exclude in path.parents or path.suffix not in ('.json', '.jsonl'):
            continue
        if path.stat().st_size > 40*2**20:
            continue
        text = path.read_text(encoding='utf-8-sig', errors='replace')
        found.update(re.findall(r'DIV2K_valid_HR[^a-zA-Z0-9]{1,8}(0[89]\d\d)\.png', text))
    return sorted(found)


def make_plan(root, dataset, seed=20261024):
    rng = np.random.default_rng(seed)
    dataset = Path(dataset).resolve()
    train = sorted((dataset/'DIV2K_train_HR').glob('*.png'))
    excluded = prior_valid_ids(root.parent, root)
    valid = [p for p in sorted((dataset/'DIV2K_valid_HR').glob('*.png')) if p.stem not in excluded]
    if len(train) != 800 or len(valid) < 32:
        raise ValueError(f'Need 800 train and 32 previously unused validation identities, found {len(train)}, {len(valid)}')
    valid = list(rng.permutation(valid))
    splits = dict(train=train, val=valid[:12], test=valid[12:24], final=valid[24:32])
    records = []
    for split, count, n_dpac in [('train', 1100, 100), ('val', 22, 2), ('test', 22, 2)]:
        sources = []
        while len(sources) < count:
            sources.extend(rng.permutation(splits[split]))
        methods = rng.permutation(['dpac']*n_dpac+['sgd']*(count-n_dpac))
        for source, method in zip(sources[:count], methods):
            records.append(dict(id=len(records), split=split, method=str(method),
                source=str(source.relative_to(dataset)), source_sha256=sha(source),
                seed=int(rng.integers(1, 2**31-1)), steps=0 if method == 'dpac' else int(rng.integers(50, 201))))
    return dict(schema=1, seed=seed, dataset_root=str(dataset), previously_used_validation_ids=excluded,
                sources={k:[str(p.relative_to(dataset)) for p in v] for k,v in splits.items()},
                records=records)


class Worker:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.out).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for d in ('pairs', 'patterns', 'models', 'evaluation', 'source_snapshot'):
            (self.root/d).mkdir(exist_ok=True)
        self.budget = StorageBudget(self.root, max_gib=150, min_free_gib=100)
        self.configs = [read_json(self.root/ch/'config.json') for ch in CHANNELS]
        if not (self.root/'completed_prepare.json').exists():
            raise ValueError('Measured RGB preparation must complete first')
        self.base = self.configs[1]
        self.rectifiers = [Rectifier(self.root/ch/'calibration', c) for ch,c in zip(CHANNELS, self.configs)]
        self.device = args.device
        plan_path = self.root/'plan.json'
        if not plan_path.exists():
            atomic_json(plan_path, make_plan(self.root, args.dataset))
        self.plan = read_json(plan_path)
        self.fingerprint = sha(plan_path)
        self.state = dict(status='running', pid=os.getpid(), started_at=time.time(), stage=args.stage)
        protocol = dict(
            paper='Choi et al., Neural 3D Holography, TOG 40(6), 240, 2021; section 3.3',
            scope='Fixed +15 mm single plane, explicitly confirmed by the user; not an 8-plane 3D reproduction',
            architecture=PaperCNNpropCNN.architecture, parameters_per_color=66746156,
            cnn_source='unchanged authors unet.py; 2-channel real/imag input/output; upstream rect format',
            training=dict(per_color_pairs=1100, dpac=100, sgd=1000, loss='L1 amplitude',
                          adam_lr=5e-4, batch_size=1, early_stopping_patience=10, max_epochs=200,
                          validation_pairs=22, validation_originals=12, independent_color_models=True,
                          precision='float32 convolutions and complex64 FFT; no mixed precision',
                          fixed_bank=True, checkpoint_interval_samples=25),
            pattern_generation=dict(sgd_iterations='uniform integer 50..200 inclusive (local choice; range unspecified in main paper)',
                initial_phase='uniform [-pi,pi] independently per pixel', phase_adam_lr=.01,
                quantization='Continuous phase optimization as released code; canonical inverted 8-bit encoding only on display and persistence',
                amplitude_scale='analytic detached least-squares scale, as released algorithms.py',
                dpac='back-propagated zero-phase target; normalized amplitude and checkerboard +/- arccos encoding'),
            evaluation=dict(test_pairs=22, test_originals=12, untouched_final_originals=8,
                phase_steps=1000, phase_lr=.01, methods=['dpac','sgd_asm','sgd_cnnpropcnn'],
                capture_repeats=3, primary='independent captured quality after phase optimization',
                metrics=['fixed-white amplitude MSE','scalar-fitted amplitude MSE and PSNR'],
                gradient_probe='paper Figure 6: central 3x3 pi perturbation at zero phase, three paired repeats'),
            limitations=['no depth actuator or RGBD targets; no 3D/held-out-depth/ADMM claim',
                '15 mm, 638/520/450 nm, 6.4 um pitch and 0.5 aperture belong to this bench',
                'homography and fixed-white calibration differ from authors camera calibration',
                'local generation iteration distribution and split sizes; no exact numerical paper-score reproduction',
                'paper-sized original network, but propagation uses numerically stable local calibrated ASM'],
            configs=self.configs, plan_sha256=self.fingerprint,
            stopping='200 epochs safety limit or validation fails to decrease for 10 epochs; not a convergence proof')
        protocol['calibration_sha256'] = {str(p.relative_to(self.root)):sha(p)
            for ch in CHANNELS for p in (self.root/ch/'calibration').iterdir() if p.is_file()}
        protocol['evaluation']['additional_metrics']='Per-channel fitted amplitude -> clipped linear intensity -> sRGB PSNR and Gaussian SSIM; average channels/images equally. Not numerically interchangeable with paper scores.'
        protocol_path = self.root/'protocol.json'
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError('Protocol differs: do not silently change a running experiment')
        atomic_json(protocol_path, protocol)
        self.progress('ready')

    def snapshot(self):
        manifest = {}
        files = [Path('unet.py')]+list(Path('benchtop').glob('*.py'))
        for p in files:
            relative = str(p).replace('\\','/')
            destination = self.root/'source_snapshot'/p
            if destination.exists() and sha(destination) != sha(p):
                raise RuntimeError(f'Execution source changed since start: {relative}; audit before resume')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(p, destination)
            manifest[relative] = sha(p)
        atomic_json(self.root/'source_snapshot'/'manifest.json', manifest)

    def guard(self):
        self.budget.check()
        if getattr(self,'bench',None) is not None:
            self.bench.app.processEvents()
        if (self.root/'STOP_REQUESTED').exists():
            raise InterruptedError('STOP_REQUESTED: safely stopped at next checkpoint')

    def progress(self, stage, **data):
        self.state.update(status='running', stage=stage, updated_at=time.time(), **data)
        atomic_json(self.root/'status.json', self.state)
        print(json.dumps(dict(stage=stage, **data), ensure_ascii=False), flush=True)

    def target(self, source):
        return prepare_image(Path(self.plan['dataset_root'])/source, self.base['roi_shape'])[0]

    def generate_one(self, row, asms):
        dest = self.root/'patterns'/f'{row["id"]:04d}.npz'
        if dest.exists():
            return dest
        self.guard()
        target = self.target(row['source'])
        phases, history = [], []
        for ci, (model, c) in enumerate(zip(asms,self.configs)):
            t = torch.from_numpy(target[...,ci]).to(self.device)
            torch.manual_seed(row['seed']+ci)
            if row['method'] == 'dpac':
                with torch.no_grad():
                    p = dpac(model, t)
                hist = []
            else:
                initial = (torch.rand(c['slm_shape'], device=self.device)*2-1)*torch.pi
                p, hist = optimize_phase(lambda phase:model(phase,quantize=False), t, c, row['steps'], .01, initial=initial,
                                         trace=lambda *a: self.guard())
            phases.append(p.detach().cpu().numpy())
            history.append(hist)
        encoded, _ = canonical(np.stack(phases,-1), self.base)
        atomic_npz(dest, phase_rgb_u8=encoded)
        atomic_json(dest.with_suffix('.json'),dict(**row, phase_sha256=sha(dest), histories=history))
        return dest

    def capture_rgb(self, bench, phase):
        bench.show(phase)
        amplitudes, sensor = [], []
        for c,r in zip(self.configs,self.rectifiers):
            select_channel(bench,c)
            rgb = bench.capture(frames=3)
            amplitudes.append(r.amplitude(rgb))
            sensor.append(channel_intensity(rgb,c))
        return np.stack(amplitudes,-1), np.stack(sensor,-1)

    def drain(self, bench, label):
        append_json(self.root/'frames.jsonl', dict(time=time.time(), label=label, frames=bench.frame_metadata))
        bench.frame_metadata.clear()

    def drift(self, bench, label):
        measured, sensor = self.capture_rgb(bench,np.zeros(tuple(self.base['slm_shape'])+(3,),np.float32))
        stats = dict(time=time.time(),label=label,mean_amplitude_rgb=measured.mean((0,1)).tolist(),
                     mean_intensity_rgb=(measured**2).mean((0,1)).tolist())
        append_json(self.root/'drift.jsonl',stats)
        self.drain(bench,label)
        reference = self.root/'drift_reference.npz'
        if not reference.exists():
            atomic_npz(reference,amplitude_rgb=measured,sensor_intensity_rgb=sensor)
        else:
            with np.load(reference) as f:
                baseline = (f['amplitude_rgb']**2).mean((0,1))
            ratios = (measured**2).mean((0,1))/baseline
            if np.max(np.abs(ratios-1)) > .20:
                raise RuntimeError(f'Fixed-white reference intensity drift exceeds 20%: {ratios.tolist()}; inspect without renormalizing old data')

    def acquire(self):
        asms = [channel_model(c,r,self.device) for c,r in zip(self.configs,self.rectifiers)]
        with Bench(self.base,display_wait_s=60) as bench:
            self.bench = bench
            self.drift(bench,'acquisition_start')
            for row in self.plan['records']:
                self.guard()
                dest = self.root/'pairs'/f'{row["id"]:04d}.npz'
                receipt = dest.with_suffix('.json')
                if dest.exists() and receipt.exists():
                    saved = read_json(receipt)
                    if saved['sha256'] != sha(dest) or saved['plan_sha256'] != self.fingerprint:
                        raise RuntimeError(f'Acquisition receipt mismatch: {dest}')
                    continue
                if dest.exists():
                    raise RuntimeError(f'Uncommitted capture exists: {dest}; preserve and audit before resume')
                self.progress('acquire',sample=row['id'],total=len(self.plan['records']),split=row['split'],method=row['method'])
                pattern = self.generate_one(row,asms)
                with np.load(pattern) as f:
                    encoded = f['phase_rgb_u8']
                measured, sensor = self.capture_rgb(bench,decode_phase(encoded,self.base['invert_phase']))
                if not np.isfinite(measured).all():
                    raise RuntimeError('Non-finite camera amplitude')
                atomic_npz(dest,phase_rgb_u8=encoded,amplitude_rgb=measured,sensor_intensity_rgb=sensor)
                record = dict(**row,sha256=sha(dest),plan_sha256=self.fingerprint,captured_at=time.time())
                atomic_json(receipt,record)
                append_json(self.root/'acquisition.jsonl',record)
                self.drain(bench,f'sample_{row["id"]:04d}')
                if row['id']%10 == 0:
                    for label,amp in [('target',self.target(row['source'])),('camera',measured)]:
                        image=Image.fromarray(np.uint8(srgb_preview(amp)*255))
                        image.thumbnail((840,480))
                        image.save(self.root/f'latest_{label}.png')
                    atomic_json(self.root/'preview.json',dict(source=row['source'],sample=row['id'],split=row['split'],
                        time=time.time(),meaning='Actual measured fixed-bank acquisition, not CITL'))
                if (row['id']+1)%50 == 0:
                    self.drift(bench,f'after_{row["id"]+1}')
            self.drift(bench,'acquisition_end')
        self.bench = None
        atomic_json(self.root/'completed_acquire.json',dict(time=time.time(),pairs=len(self.plan['records'])))

    def load_pair(self, row, ci):
        with np.load(self.root/'pairs'/f'{row["id"]:04d}.npz') as f:
            phase = torch.from_numpy(decode_phase(f['phase_rgb_u8'][...,ci], self.configs[ci]['invert_phase'])).to(self.device)
            amplitude = torch.from_numpy(f['amplitude_rgb'][...,ci].copy()).to(self.device)
        return phase,amplitude

    def validate(self, net, ci, split='val'):
        rows = []
        net.eval()
        with torch.no_grad():
            for row in self.plan['records']:
                if row['split'] != split:
                    continue
                self.guard()
                p,t = self.load_pair(row,ci)
                prediction = crop_center(net(p).abs(),t.shape)
                rows.append(dict(id=row['id'],l1=float((prediction-t).abs().mean()),**mse_metrics(prediction,t)))
        return float(np.mean([r['l1'] for r in rows])),rows

    def train_color(self, ci):
        ch,c = CHANNELS[ci],self.configs[ci]
        marker = self.root/'models'/f'{ch}_complete.json'
        if marker.exists():
            return
        latest,best = [self.root/'models'/f'{ch}_{kind}.pt' for kind in ('latest','best')]
        torch.manual_seed(self.plan['seed']+ci)
        net = PaperCNNpropCNN(c,self.device)
        optimizer = torch.optim.Adam(net.parameters(),lr=5e-4)
        epoch,offset,patience,seen,best_l1 = 0,0,0,0,float('inf')
        if latest.exists():
            saved = torch.load(latest,map_location='cpu',weights_only=False)
            if saved['plan_sha256'] != self.fingerprint or saved['config'] != c:
                raise ValueError('Model checkpoint configuration differs')
            net.load_learnable(saved['weights'])
            optimizer.load_state_dict(saved['optimizer'])
            epoch,offset,patience,seen,best_l1 = [saved[k] for k in ('epoch','offset','patience','seen','best_l1')]
            del saved
        train = [r for r in self.plan['records'] if r['split']=='train']
        def save():
            save_checkpoint(latest,dict(config=c,plan_sha256=self.fingerprint,architecture=net.architecture,
                weights=net.learnable_state(),optimizer=optimizer.state_dict(),epoch=epoch,offset=offset,
                patience=patience,seen=seen,best_l1=best_l1))
        try:
            while epoch < 200 and patience < 10:
                order = np.random.default_rng(self.plan['seed']+epoch).permutation(len(train))
                net.train()
                for position in range(offset,len(order)):
                    self.guard()
                    row = train[int(order[position])]
                    p,t = self.load_pair(row,ci)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = crop_center(net(p).abs(),t.shape)
                    loss = (prediction-t).abs().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite L1 training loss')
                    loss.backward()
                    for parameter in net.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError('Non-finite CNN gradient')
                    optimizer.step()
                    seen += 1
                    offset = position+1
                    if offset%25 == 0 or offset == len(order):
                        save()
                        self.progress('train',channel=ch,epoch=epoch+1,offset=offset,train_pairs=len(train),
                                      loss_l1=float(loss),best_validation_l1=best_l1 if np.isfinite(best_l1) else None,
                                      gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                val,rows = self.validate(net,ci)
                epoch += 1
                offset = 0
                if val < best_l1:
                    best_l1,patience=val,0
                    save_checkpoint(best,dict(config=c,plan_sha256=self.fingerprint,weights=net.learnable_state(),
                                             architecture=net.architecture,epoch=epoch,validation_l1=val))
                else:
                    patience += 1
                append_json(self.root/'training.jsonl',dict(time=time.time(),channel=ch,epoch=epoch,val_l1=val,
                    best_l1=best_l1,patience=patience,validation=rows,seen=seen))
                save()
        except BaseException:
            # Only completed Adam steps are counted. Unexpected failed steps are
            # resumed from the previous atomic checkpoint, never partially saved.
            raise
        atomic_json(marker,dict(time=time.time(),epoch=epoch,patience=patience,best_l1=best_l1,
                               reason='validation_patience_10' if patience>=10 else 'epoch_limit_200'))
        del net,optimizer
        torch.cuda.empty_cache()

    def load_model(self,ci):
        saved = torch.load(self.root/'models'/f'{CHANNELS[ci]}_best.pt',map_location='cpu',weights_only=False)
        if saved['plan_sha256'] != self.fingerprint or saved['config'] != self.configs[ci]:
            raise ValueError('Evaluation model does not match frozen configuration')
        net = PaperCNNpropCNN(self.configs[ci],self.device)
        net.load_learnable(saved['weights'])
        return net.eval().requires_grad_(False)

    def sealed_forward(self):
        dest=self.root/'evaluation'/'forward_test.json'
        if dest.exists():
            return
        results=[]
        for ci in range(3):
            net=self.load_model(ci)
            val,rows=self.validate(net,ci,'test')
            asm=channel_model(self.configs[ci],self.rectifiers[ci],self.device)
            with torch.no_grad():
                for row,result in zip([r for r in self.plan['records'] if r['split']=='test'],rows):
                    p,t=self.load_pair(row,ci)
                    result['asm']=mse_metrics(crop_center(asm(p).abs(),t.shape),t)
            results.append(dict(channel=CHANNELS[ci],l1=val,records=rows))
            del net,asm
            torch.cuda.empty_cache()
        atomic_json(dest,dict(time=time.time(),results=results))

    def evaluate(self):
        self.sealed_forward()
        for index,source in enumerate(self.plan['sources']['final']):
            dest=self.root/'evaluation'/Path(source).stem
            dest.mkdir(exist_ok=True)
            if (dest/'metrics.json').exists():
                continue
            target=self.target(source)
            Image.fromarray(np.uint8(srgb_preview(target)*255)).save(dest/'target.png')
            phases={name:[] for name in ('dpac','sgd_asm','sgd_cnnpropcnn')}
            for ci,c in enumerate(self.configs):
                self.progress('evaluate_optimize',target=source,channel=CHANNELS[ci],target_index=index)
                t=torch.from_numpy(target[...,ci]).to(self.device)
                asm=channel_model(c,self.rectifiers[ci],self.device)
                torch.manual_seed(self.plan['seed']+10000+index*3+ci)
                initial=(torch.rand(c['slm_shape'],device=self.device)*2-1)*torch.pi
                with torch.no_grad():
                    phases['dpac'].append(dpac(asm,t).cpu().numpy())
                for name,model in [('sgd_asm',asm),('sgd_cnnpropcnn',self.load_model(ci))]:
                    p,hist=optimize_phase(lambda phase:model(phase,quantize=False),t,c,iterations=1000,lr=.01,initial=initial,
                        trace=lambda step,*a: self.guard())
                    phases[name].append(p.cpu().numpy())
                    atomic_json(dest/f'{name}_{CHANNELS[ci]}_optimization.json',hist)
                    if name=='sgd_cnnpropcnn':
                        del model
                del asm
                torch.cuda.empty_cache()
            encoded={name:canonical(np.stack(values,-1),self.base)[0] for name,values in phases.items()}
            atomic_npz(dest/'phases.npz',**encoded)
            rows=[]
            rng=np.random.default_rng(self.plan['seed']+index)
            with Bench(self.base,display_wait_s=60) as bench:
                self.drift(bench,f'evaluation_{index}_start')
                for repeat in range(3):
                    for name in rng.permutation(list(encoded)):
                        self.guard()
                        self.progress('evaluate_capture',target=source,method=str(name),repeat=repeat)
                        measured,sensor=self.capture_rgb(bench,decode_phase(encoded[name],self.base['invert_phase']))
                        atomic_npz(dest/f'{name}_repeat{repeat}.npz',amplitude_rgb=measured,sensor_intensity_rgb=sensor)
                        if repeat==0:
                            Image.fromarray(np.uint8(srgb_preview(measured)*255)).save(dest/f'{name}.png')
                        metrics=[image_metrics(torch.from_numpy(measured[...,ci]),torch.from_numpy(target[...,ci])) for ci in range(3)]
                        rows.append(dict(method=str(name),repeat=repeat,channels=metrics))
                        self.drain(bench,f'{index}_{name}_{repeat}')
                self.drift(bench,f'evaluation_{index}_end')
            atomic_json(dest/'metrics.json',dict(source=source,time=time.time(),records=rows))
        self.gradient_probe()

    def gradient_probe(self):
        dest=self.root/'evaluation'/'figure6_probe.json'
        if dest.exists():
            return
        base=np.zeros(tuple(self.base['slm_shape'])+(3,),np.float32)
        changed=base.copy()
        h,w=self.base['slm_shape']
        changed[h//2-1:h//2+2,w//2-1:w//2+2,:]=np.pi
        actual={}
        with Bench(self.base,display_wait_s=60) as bench:
            for rep in range(3):
                for name,p in ([('zero',base),('pi_3x3',changed)] if rep%2==0 else [('pi_3x3',changed),('zero',base)]):
                    measured,sensor=self.capture_rgb(bench,p)
                    actual.setdefault(name,[]).append(measured)
                    atomic_npz(self.root/'evaluation'/f'probe_{name}_{rep}.npz',amplitude_rgb=measured)
                    self.drain(bench,f'probe_{name}_{rep}')
        measured={domain:np.mean([a**power for a in actual['pi_3x3']],0)-np.mean([a**power for a in actual['zero']],0)
                  for domain,power in [('amplitude',1),('intensity',2)]}
        results=[]
        for ci,c in enumerate(self.configs):
            net=self.load_model(ci)
            asm=channel_model(c,self.rectifiers[ci],self.device)
            for name,model in [('asm',asm),('cnnpropcnn',net)]:
                with torch.no_grad():
                    ap=model(torch.from_numpy(changed[...,ci]).to(self.device)).abs()
                    az=model(torch.from_numpy(base[...,ci]).to(self.device)).abs()
                    for domain,power in [('amplitude',1),('intensity',2)]:
                        response=crop_center(ap**power-az**power,c['roi_shape'])
                        reference=torch.from_numpy(measured[domain][...,ci]).to(self.device)
                        cosine=float((response*reference).sum()/(response.norm()*reference.norm()).clamp_min(1e-12))
                        results.append(dict(channel=CHANNELS[ci],model=name,domain=domain,cosine=cosine,
                                            rmse=float((response-reference).square().mean().sqrt())))
            del net,asm
            torch.cuda.empty_cache()
        atomic_json(dest,dict(results=results,warning='Finite pi perturbation, not an infinitesimal full Jacobian; qualitative Figure 6 analogue'))

    def run(self):
        with exclusive_run(self.root):
            self.snapshot()
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
            try:
                if self.args.stage in ('all','acquire'):
                    self.acquire()
                if self.args.stage in ('all','train'):
                    if not (self.root/'completed_acquire.json').exists():
                        raise RuntimeError('Complete fixed acquisition bank before training')
                    for ci in range(3):
                        self.train_color(ci)
                if self.args.stage in ('all','evaluate'):
                    self.evaluate()
                    summaries=[]
                    for p in (self.root/'evaluation').glob('*/metrics.json'):
                        report=read_json(p)
                        for method in ('dpac','sgd_asm','sgd_cnnpropcnn'):
                            values=[m for row in report['records'] if row['method']==method for m in row['channels']]
                            summaries.append(dict(source=report['source'],method=method,
                                **{k:float(np.mean([m[k] for m in values])) for k in values[0]}))
                    atomic_json(self.root/'evaluation'/'summary.json',dict(per_target=summaries,
                        means={method:{k:float(np.mean([m[k] for m in summaries if m['method']==method]))
                            for k in summaries[0] if k not in ('source','method')}
                            for method in ('dpac','sgd_asm','sgd_cnnpropcnn')}))
                self.state.update(status='completed',updated_at=time.time())
                atomic_json(self.root/'status.json',self.state)
                atomic_json(self.root/f'completed_{self.args.stage}.json',self.state)
            except BaseException as exc:
                self.state.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',
                                  updated_at=time.time(),error=str(exc),traceback=traceback.format_exc())
                atomic_json(self.root/'status.json',self.state)
                append_json(self.root/'failures.jsonl',self.state)
                raise
            finally:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--dataset',default=r'C:\Users\khchoi_exp\Documents\DIV2K dataset')
    p.add_argument('--stage',choices=['all','acquire','train','evaluate','plan'],default='all')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=p.parse_args()
    torch.set_num_threads(4)
    worker=Worker(args)
    if args.stage!='plan':
        worker.run()


if __name__=='__main__':
    main()
