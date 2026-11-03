"""Isolated physical CITL diagnostics. Never modify the source experiment.

Fixed patterns, central finite differences, and matched-budget learning-rate
trials use the original optics/exposures and frozen best propagation model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from .calibration import Rectifier
from .campaign import append_json, decode_phase, exclusive_run
from .hardware import Bench, encode_phase
from .joint import optimize_rgb_phase
from .live import LiveMonitor, atomic_json
from .optics import crop_center, load_forward_model
from .rgb_run import CHANNELS, capture_amplitude, channel_model, read_json, srgb_preview
from .targets import prepare_image


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64)-b)**2))


def save_preview(path, amplitude):
    Image.fromarray(np.round(srgb_preview(amplitude)*255).clip(0,255).astype(np.uint8)).save(path)


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denom) if denom > 1e-20 else None


def fd_metrics(camera_replicates, model):
    """Replicated central differences; noise of mean of two independent pairs."""
    camera_replicates = np.asarray(camera_replicates, np.float64)
    if len(camera_replicates) != 2:
        raise ValueError('Exactly two independent finite-difference pairs required')
    measured = camera_replicates.mean(0)
    noise_rms = float(np.sqrt(np.mean(((camera_replicates[0]-camera_replicates[1])/2)**2)))
    response_rms = float(np.sqrt(np.mean(measured**2)))
    return dict(cosine=cosine(measured, model),
                cosine_rgb=[cosine(measured[...,i],model[...,i]) for i in range(3)],
                response_rms=response_rms, estimated_noise_of_mean_rms=noise_rms,
                response_to_noise=response_rms/max(noise_rms,1e-12),
                relative_rmse=float(np.sqrt(mse(measured,model)))/max(response_rms,1e-12))


def source_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and not any(part in ('live','preview_cache','report','__pycache__') for part in p.relative_to(root).parts))


def fingerprint(root):
    rows = {}
    for path in source_files(root):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(4*1024*1024), b''):
                digest.update(chunk)
        rows[path.relative_to(root).as_posix()] = dict(size=path.stat().st_size,sha256=digest.hexdigest())
    return rows


class Diagnosis:
    def __init__(self,args):
        self.args=args
        self.source=Path(args.source).resolve(); self.out=Path(args.out).resolve()
        if self.out==self.source or self.out.is_relative_to(self.source):
            raise ValueError('Diagnostics must be outside the source experiment')
        self.out.mkdir(exist_ok=True,parents=True)
        if (self.out/'protocol.json').exists():
            raise FileExistsError('Use a fresh output folder; completed measurements are immutable')
        state=read_json(self.source/'campaign/status.json')
        if state['status']!='completed':
            raise RuntimeError('The source campaign must finish before taking hardware ownership')
        self.before=fingerprint(self.source)
        atomic_json(self.out/'source_manifest_before.json',self.before)
        self.configs=[read_json(self.source/ch/'config.json') for ch in CHANNELS]
        # Independent model/calibration snapshot; no writes back to source.
        for ch in CHANNELS:
            shutil.copytree(self.source/ch/'calibration',self.out/ch/'calibration')
            shutil.copy2(self.source/ch/'config.json',self.out/ch/'config.json')
        shutil.copytree(self.source/'campaign/models',self.out/'models')
        self.rectifiers=[Rectifier(self.out/ch/'calibration',c) for ch,c in zip(CHANNELS,self.configs)]
        self.models=[load_forward_model(self.out/'models'/f'{ch}_best.pt',c,args.device) for ch,c in zip(CHANNELS,self.configs)]
        self.asm=[channel_model(c,r,args.device) for c,r in zip(self.configs,self.rectifiers)]
        plan=read_json(self.source/'campaign/plan.json')
        self.cases=[]
        for eid in (100,101):
            record=read_json(self.source/'campaign/evaluations'/f'final_{eid:04d}.json')
            relative=record['source']['target_id']
            target,_=prepare_image(Path(plan['dataset_root'])/relative,self.configs[1]['roi_shape'])
            phases={}
            for key,suffix in [('initial','cnn'),('old_best','cnn_citl')]:
                with np.load(self.source/'campaign/evaluations'/f'final_{eid:04d}_{suffix}.npz') as pair:
                    phases[key]=decode_phase(pair['phase_rgb_u8'],self.configs[1]['invert_phase'])
            self.cases.append(dict(id=Path(relative.replace('\\','/')).stem,target_id=relative,target=target,phases=phases))
        self.protocol=dict(schema=1,source=str(self.source),started_at=time.time(),
            targets=[c['target_id'] for c in self.cases],device=args.device,
            stability_rounds=args.rounds,fd_epsilons=[.05,.1],fd_directions=3,fd_independent_pairs=2,
            rates=[.02,.005],steps=args.steps,frames_per_channel=2,
            reference_every_steps=50,reference='0846 initial CNN phase throughout session',
            order=[['0846',.02],['0898',.005],['0846',.005],['0898',.02]],
            notes=['Same initial encoded phase, same frozen model and fresh Adam per condition.',
                   'Fixed calibration white; no exposure, LUT, optics or laser adjustment.',
                   'Matched iteration count is not a convergence claim; two targets are exploratory.',
                   'Finite differences compare models and hardware using the same +/- perturbations.',
                   'Autodiff directional gradients use the same real camera residual as CITL.'])
        atomic_json(self.out/'protocol.json',self.protocol)
        self.monitor=LiveMonitor(self.out,'CITL stability / gradients / learning rates',max_streams=20,max_snapshots=20,max_history=1000)
        self.bench=None;self.reference_rows=[];self.started=time.monotonic()

    def guard(self):
        if (self.out/'STOP_REQUESTED').exists():
            raise InterruptedError('Diagnostic stop requested')
        if self.bench is not None:
            self.bench.app.processEvents()

    def capture(self,phase,label):
        self.guard();self.bench.show(phase)
        result=np.stack([capture_amplitude(self.bench,c,r,frames=2) for c,r in zip(self.configs,self.rectifiers)],-1)
        append_json(self.out/'frames.jsonl',dict(time=time.time(),label=label,frames=self.bench.frame_metadata))
        self.bench.frame_metadata.clear()
        return result

    def row(self,amplitude,target,**extra):
        return dict(time=time.time(),elapsed=time.monotonic()-self.started,amplitude_mse=mse(amplitude,target),
                    amplitude_mse_rgb=[mse(amplitude[...,ci],target[...,ci]) for ci in range(3)],
                    mean_intensity_rgb=np.mean(amplitude**2,axis=(0,1),dtype=np.float64).tolist(),**extra)

    def reference(self,label):
        c=self.cases[0]
        amplitude=self.capture(c['phases']['initial'],'reference_'+label)
        row=self.row(amplitude,c['target'],label=label)
        self.reference_rows.append(row);append_json(self.out/'references.jsonl',row)
        return row

    def stability(self):
        probes=[(c,name) for c in self.cases for name in ('initial','old_best')]
        accum={}; rows=[]
        key=self.monitor.start('stability','rgb',self.args.rounds*4,'live_camera_stability','고정 위상 반복 촬영 · RGB')
        for repeat in range(self.args.rounds):
            for idx in np.random.default_rng(300+repeat).permutation(4):
                c,name=probes[idx];identity=c['id']+'_'+name
                a=self.capture(c['phases'][name],f'stability_{identity}_{repeat}')
                row=self.row(a,c['target'],id=c['id'],phase=name,repeat=repeat)
                if identity not in accum:
                    accum[identity]=dict(count=0,mean=np.zeros_like(a,dtype=np.float64),m2=np.zeros_like(a,dtype=np.float64))
                    save_preview(self.out/f'stability_{identity}_first.png',a)
                acc=accum[identity];acc['count']+=1;delta=a-acc['mean'];acc['mean']+=delta/acc['count'];acc['m2']+=delta*(a-acc['mean'])
                rows.append(row);append_json(self.out/'stability.jsonl',row)
                self.monitor.update(key,len(rows),{k:v for k,v in row.items() if k!='time'},
                    [('Target · RGB',c['target'],0.,1.,'srgb_amplitude'),('Camera · RGB',a,0.,1.,'srgb_amplitude')])
                print(f'stability {len(rows)}/{self.args.rounds*4}: {identity} MSE={row["amplitude_mse"]:.6f}',flush=True)
        summary={}
        for identity,acc in accum.items():
            selected=[r for r in rows if r['id']+'_'+r['phase']==identity]
            vals=np.array([r['mean_intensity_rgb'] for r in selected]);m=np.array([r['amplitude_mse'] for r in selected])
            std=np.sqrt(acc['m2']/(acc['count']-1))
            np.savez_compressed(self.out/f'stability_{identity}.npz',mean_amplitude=acc['mean'].astype(np.float16),std_amplitude=std.astype(np.float16))
            save_preview(self.out/f'stability_{identity}_mean.png',acc['mean'])
            summary[identity]=dict(count=acc['count'],intensity_cv_rgb=(vals.std(0,ddof=1)/vals.mean(0)).tolist(),
                mse_mean=float(m.mean()),mse_std=float(m.std(ddof=1)),mse_range=[float(m.min()),float(m.max())],
                amplitude_noise_rms=float(np.sqrt(np.mean(std**2))),duration_s=selected[-1]['time']-selected[0]['time'])
        atomic_json(self.out/'stability_summary.json',summary);self.monitor.end_stream(key,'measurement_complete',summary)

    def predict(self,models,phase):
        with torch.no_grad():
            return np.stack([crop_center(model(torch.as_tensor(phase[...,ci],device=self.args.device)).abs(),
                              self.configs[ci]['roi_shape']).cpu().numpy() for ci,model in enumerate(models)],-1)

    def autodiff(self,models,phase,camera,target):
        gradients=[]
        for ci,model in enumerate(models):
            p=torch.tensor(phase[...,ci],device=self.args.device,requires_grad=True)
            pred=crop_center(model(p).abs(),target.shape[:2])
            real=torch.as_tensor(camera[...,ci],device=self.args.device)
            desired=torch.as_tensor(target[...,ci],device=self.args.device)
            surrogate=pred+(real-pred).detach()
            loss=(surrogate-desired).square().mean()/3
            grad,=torch.autograd.grad(loss,p)
            gradients.append(grad.detach().cpu().numpy())
        return np.stack(gradients,-1)

    def gradients(self):
        c=self.cases[0];rows=[]
        key=self.monitor.start('gradient_check','rgb',12,'live_camera_gradient','실측 / 모델 gradient 비교 · 0846.png')
        for base_name in ('initial','old_best'):
            phase=c['phases'][base_name]
            base_camera=np.mean([self.capture(phase,f'fd_base_{base_name}_{r}') for r in range(2)],axis=0)
            grads={name:self.autodiff(models,phase,base_camera,c['target']) for name,models in [('asm',self.asm),('cnn',self.models)]}
            for direction_index in range(3):
                rng=np.random.default_rng(900+direction_index)
                h,w=phase.shape[:2]
                coarse=rng.choice([-1.,1.],size=(16,28,3)).astype(np.float32)
                direction=np.repeat(np.repeat(coarse,int(np.ceil(h/16)),axis=0),int(np.ceil(w/28)),axis=1)[:h,:w]
                direction-=direction.mean((0,1),keepdims=True);direction/=np.sqrt(np.mean(direction**2))
                for epsilon in (.05,.1):
                    name=f'fd_{base_name}_d{direction_index}_e{epsilon:g}'
                    measurements={1:[],-1:[]}
                    # Reverse sign order in second repeat to reduce linear time-drift bias.
                    for repeat in range(2):
                        for sign in ((1,-1) if repeat==0 else (-1,1)):
                            measurements[sign].append(self.capture(phase+sign*epsilon*direction,f'{name}_r{repeat}_{sign}'))
                    plus=np.stack(measurements[1]);minus=np.stack(measurements[-1])
                    measured=(plus-minus)/(2*epsilon)
                    losses_plus=[mse(a,c['target']) for a in plus];losses_minus=[mse(a,c['target']) for a in minus]
                    scalar=(np.array(losses_plus)-losses_minus)/(2*epsilon)
                    row=dict(base=base_name,direction=direction_index,epsilon=epsilon,
                        physical_loss_derivative_replicates=scalar.tolist(),physical_loss_derivative=float(scalar.mean()),
                        physical_loss_derivative_repeat_se=float(scalar.std(ddof=1)/np.sqrt(2)),models={})
                    arrays={'camera_fd_mean':measured.mean(0).astype(np.float16),
                            'camera_fd_noise':((measured[0]-measured[1])/2).astype(np.float16)}
                    for model_name,models in [('asm',self.asm),('cnn',self.models)]:
                        fd=(self.predict(models,phase+epsilon*direction)-self.predict(models,phase-epsilon*direction))/(2*epsilon)
                        metrics=fd_metrics(measured,fd)
                        metrics['autodiff_loss_derivative']=float(np.sum(grads[model_name].astype(np.float64)*direction))
                        metrics['fd_surrogate_loss_derivative']=float(2*np.mean((base_camera-c['target'])*fd))
                        metrics['autodiff_sign_agrees']=bool(metrics['autodiff_loss_derivative']*float(scalar.mean())>0)
                        row['models'][model_name]=metrics;arrays[model_name+'_fd']=fd.astype(np.float16)
                    np.savez_compressed(self.out/(name+'.npz'),**arrays)
                    rows.append(row);append_json(self.out/'gradients.jsonl',row)
                    self.monitor.update(key,len(rows),dict(amplitude_mse=mse(base_camera,c['target']),**row))
                    print(f'gradient {len(rows)}/12: {base_name} eps={epsilon} CNN cosine={row["models"]["cnn"]["cosine"]:.3f} SNR={row["models"]["cnn"]["response_to_noise"]:.2f}',flush=True)
        atomic_json(self.out/'gradient_summary.json',rows);self.monitor.end_stream(key,'measurement_complete')

    def learning_rates(self):
        summaries=[]
        for case_id,lr in self.protocol['order']:
            self.guard();c=next(c for c in self.cases if c['id']==case_id)
            folder=self.out/f'citl_{case_id}_lr{lr:g}';folder.mkdir()
            key=self.monitor.start('composite_citl','rgb',self.args.steps,'live_camera_citl',f'진단 CITL · {c["target_id"]} · lr={lr:g}')
            self.monitor.state['streams'][key]['objective']='Matched fixed iteration budget; no convergence early stopping'
            callback=self.monitor.rgb_trace(key,c['target'],every=20)
            first=None;last=None;best_camera=None
            self.reference(f'{case_id}_{lr:g}_before')
            def capture(phase):
                return self.capture(phase,f'CITL_{case_id}_{lr:g}')
            def trace(step,row,phase,amplitude,gradient):
                nonlocal first,last,best_camera
                camera=amplitude.detach().cpu().numpy()
                if first is None:
                    first=camera.copy()
                if row['best_step']==step:
                    best_camera=camera.copy()
                if step==self.args.steps:
                    last=camera.copy()
                    np.savez_compressed(folder/'last.npz',phase_rgb_u8=encode_phase(phase.detach().cpu().numpy(),self.configs[1]['invert_phase']),amplitude_rgb=camera.astype(np.float16))
                append_json(folder/'history.jsonl',dict(time=time.time(),**row))
                callback(step,row,phase,amplitude,gradient)
                if step%20==0:
                    print(f'CITL {case_id} lr={lr:g} step {step}/{self.args.steps}: mse={row["amplitude_mse"]:.6f} best={row["best_mse"]:.6f}',flush=True)
                if step and step%50==0:
                    self.reference(f'{case_id}_{lr:g}_step{step}')
            best,history,termination=optimize_rgb_phase(self.models,c['target'],self.configs,c['phases']['initial'],
                max_steps=self.args.steps,lr=lr,capture=capture,trace=trace,guard=self.guard)
            self.monitor.end_stream(key,termination['reason'],termination)
            best_phase=best.cpu().numpy();del best
            np.savez_compressed(folder/'best.npz',phase_rgb_u8=encode_phase(best_phase,self.configs[1]['invert_phase']),amplitude_rgb=best_camera.astype(np.float16))
            for name,a in [('target',c['target']),('initial',first),('best_measured',best_camera),('last',last)]:
                save_preview(folder/(name+'.png'),a)
            # Independently remeasure adopted best and final phase in interleaved order.
            with np.load(folder/'last.npz') as saved:
                last_phase=decode_phase(saved['phase_rgb_u8'],self.configs[1]['invert_phase'])
            repeats={'best':[],'last':[]};repeat_rows=[]
            for repeat in range(3):
                for name in (('best','last') if repeat%2==0 else ('last','best')):
                    a=self.capture(best_phase if name=='best' else last_phase,f'remeasure_{case_id}_{lr:g}_{name}_{repeat}')
                    repeats[name].append(a);repeat_rows.append(self.row(a,c['target'],variant=name,repeat=repeat))
            for name,values in repeats.items():
                a=np.mean(values,axis=0)
                save_preview(folder/f'{name}_remeasured.png',a)
                np.savez_compressed(folder/f'{name}_remeasured.npz',amplitude_rgb=a.astype(np.float16))
            summary=dict(id=case_id,target_id=c['target_id'],lr=lr,termination=termination,
                initial_mse=history[0]['amplitude_mse'],last_mse=history[-1]['amplitude_mse'],
                best_mse=termination['best_mse'],best_step=termination['best_step'],
                repeat_rows=repeat_rows,
                remeasured_mean_mse={name:float(np.mean([r['amplitude_mse'] for r in repeat_rows if r['variant']==name])) for name in repeats})
            atomic_json(folder/'summary.json',summary);summaries.append(summary)
            atomic_json(self.out/'citl_summary.json',summaries)
            self.reference(f'{case_id}_{lr:g}_after')

    def run(self):
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        error=None
        try:
            with Bench(self.configs[1],display_wait_s=10) as bench:
                self.bench=bench
                self.reference('session_start')
                self.stability();self.gradients();self.learning_rates()
                self.reference('session_end')
            self.bench=None
            atomic_json(self.out/'completed.json',dict(completed_at=time.time(),elapsed_seconds=time.monotonic()-self.started))
        except BaseException as exc:
            error=exc;atomic_json(self.out/'failure.json',dict(time=time.time(),error=str(exc),type=type(exc).__name__))
            raise
        finally:
            self.bench=None
            after=fingerprint(self.source)
            atomic_json(self.out/'source_manifest_after.json',after)
            atomic_json(self.out/'source_preservation.json',dict(unchanged=self.before==after,
                file_count=len(self.before),changed=[name for name in set(self.before)|set(after) if self.before.get(name)!=after.get(name)]))
            self.monitor.finish(error=error)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='results/rgb80_10h_20261022')
    parser.add_argument('--out',required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--rounds',type=int,default=8)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    if not 1<=args.steps<=1000 or args.rounds<2:
        parser.error('Require 1<=steps<=1000 and rounds>=2')
    Path(args.out).mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    with exclusive_run(Path(args.out)):
        diagnosis=Diagnosis(args);diagnosis.run()


if __name__=='__main__':
    main()
