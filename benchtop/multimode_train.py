"""Isolated multimode study, with sealed test guard and resumable training."""
import argparse
import json
import math
import os
import random
import socket
import time
import traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .integrated_rgb_train import Frames, JointRGB, evaluate, masked_errors, ordered_indices, lr_at
from .integrated_rgb_data import now,sha,write_json
from .multimode_model import MultimodeRGB


MANIFEST_SHA='1db1bd6a3efc5be745dea62b6b81cbb246609718feb14e72b2bb810a2cb3f32c'


def validate_manifest(data):
    if not data['complete'] or data['pilot_only'] or data['test_pixels_decoded']:
        raise ValueError('Incomplete or unsealed cache')
    rows=data['records']
    if Counter(r['split'] for r in rows)!={'fit':7500,'selection':300}:
        raise ValueError('Unexpected split counts')
    if len({r['id'] for r in rows})!=7800 or any(r['depth_mm'] not in (3.,7.,9.) for r in rows):
        raise ValueError('Duplicate ID or sealed depth')
    parents={}
    for r in rows:
        if parents.setdefault(r['scene'],r['split'])!=r['split']:raise ValueError('Parent leakage')
        for name in ('codes_path','intensity_path'):
            p=Path(r[name])
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe cache path')
    if Counter(parents.values())!={'fit':742,'selection':25}:raise ValueError('Parent counts')


@torch.no_grad()
def response_evaluation(model,ds,rows,masks):
    model.eval();groups=defaultdict(dict)
    for i,r in enumerate(rows):
        if r['condition'] in ('current_lut','probe_plus','probe_minus'):
            groups[(r['scene'],r['depth_mm'],r['pattern_sha256'],r['reference_key'])][r['condition']]=i
    result=[]
    for key,group in groups.items():
        if set(group)!={'current_lut','probe_plus','probe_minus'}:raise ValueError('Incomplete response triple')
        vals={}
        for name,i in group.items():
            _,code,y=ds[i];vals[name]=(model(code.cuda(),key[1]),y.cuda())
        pred0,y0=vals['current_lut'];mask=masks[key[3]].bool()
        for name in ('probe_plus','probe_minus'):
            pred,y=vals[name];dp=(pred-pred0)[mask];dy=(y-y0)[mask]
            denom=dp.norm()*dy.norm()
            result.append(dict(scene=key[0],depth_mm=key[1],condition=name,
                               delta_mse=float((dp-dy).square().mean()),
                               measured_rms=float(dy.square().mean().sqrt()),
                               predicted_rms=float(dp.square().mean().sqrt()),
                               cosine=float((dp*dy).sum()/denom) if float(denom)>1e-12 else None,
                               mean_change_sign_match=bool(dp.mean()*dy.mean()>0)))
    model.train()
    return dict(pairs=len(result),cosine_median=float(np.median([r['cosine'] for r in result if r['cosine'] is not None])),
                delta_mse_mean=float(np.mean([r['delta_mse'] for r in result])),
                mean_change_sign_fraction=float(np.mean([r['mean_change_sign_match'] for r in result])),
                noise_qualified=False,interpretation='Descriptive only: no independently established repeat-noise floor; not a LUT adoption gate',records=result)


def run(args):
    if args.mode not in ('pilot','train'):raise ValueError('Unsupported stage')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.out.resolve()
    if args.resume:
        if args.mode!='train' or not (out/'last.pt').exists():raise ValueError('No resumable checkpoint')
    else:out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(state='running',stage=args.mode,rank=args.rank,seed=args.seed,
                                      started_kst=now(),job_id=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),
                                      test_pixels_decoded=False,epoch=0,samples=0)
    def update(**kw):
        status.update(kw,updated_kst=now(),invocation_seconds=time.monotonic()-start)
        write_json(out/'status.json',status);print(json.dumps(status),flush=True)
    update(phase='initializing')
    try:
        manifest=args.cache/'manifest.json'
        if sha(manifest)!=MANIFEST_SHA:raise ValueError('Approved manifest SHA mismatch')
        data=json.loads(manifest.read_text());validate_manifest(data)
        fit=[r for r in data['records'] if r['split']=='fit'];sel=[r for r in data['records'] if r['split']=='selection']
        masks={}
        for key,r in data['references'].items():
            p=args.cache/r['file']
            if sha(p)!=r['sha256']:raise ValueError('Calibration changed')
            with np.load(p) as f:masks[key]=torch.from_numpy(f['valid'].copy()).cuda()
        ds=Frames(args.cache,fit);sd=Frames(args.cache,sel)
        counts=Counter(r['scene'] for r in fit);balance={k:len(fit)/(len(counts)*v) for k,v in counts.items()}
        def loader(order):return DataLoader(ds,batch_size=None,sampler=order,num_workers=2,pin_memory=True,prefetch_factor=2)
        sl=DataLoader(sd,batch_size=None,num_workers=2,pin_memory=True,prefetch_factor=2)
        model=JointRGB(args.initial) if args.rank==0 else MultimodeRGB(args.rank,'cuda')
        source_lr=3e-5 if args.rank==0 else 1e-3
        head_lr=1e-4 if args.rank==0 else 1e-3
        groups=[dict(params=list(model.branches.parameters()),lr=source_lr),dict(params=list(model.head.parameters()),lr=head_lr)]
        opt=torch.optim.Adam(groups,foreach=False)
        config=dict(rank=args.rank,seed=args.seed,mode=args.mode,epochs=args.epochs,hours=args.hours,
                    branch_lr=source_lr,head_lr=head_lr,manifest_sha=MANIFEST_SHA,
                    parameters=sum(p.numel() for p in model.parameters()),shape=[960,1728],precision='float32_complex64',
                    fit=7500,selection=300,sealed_test=400,loss='parent_balanced_linear_intensity_mse',
                    mode_grid=[60,108],initial_tilt_cycles=24,pupil_grid=[15,27],dc_grid=[15,27],phase_knots=33,amplitude_knots=17,
                    checkpoint_every_steps=1000,initialization='pretrained_single_slot_teachers_fresh_camera_head' if args.rank==0 else 'near_coherent_compact_grid',
                    device=torch.cuda.get_device_name(),torch=torch.__version__)
        if args.resume:
            original=json.loads((out/'config.json').read_text())
            for k in ('rank','seed','epochs','hours','manifest_sha','branch_lr','head_lr'):
                if original[k]!=config[k]:raise ValueError('Resume configuration changed: '+k)
        else:write_json(out/'config.json',config)
        def step(i,codes,y,factor=1.):
            for g,lr in zip(opt.param_groups,(source_lr,head_lr)):g['lr']=lr*factor
            opt.zero_grad(set_to_none=True);r=fit[i];codes=codes.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            pred=model(codes,r['depth_mm']);per,_=masked_errors(pred,y,masks[r['reference_key']]);loss=per.mean()*balance[r['scene']]
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();return float(loss.detach()),float(norm)
        probe_order=ordered_indices(fit,991)[:200]
        # Fixed fit-only panel measures a comparable pre/post pilot quantity.
        @torch.no_grad()
        def fit_panel():
            values=[];model.eval()
            for i in probe_order[:12]:
                _,code,y=ds[i];r=fit[i]
                per,_=masked_errors(model(code.cuda(),r['depth_mm']),y.cuda(),masks[r['reference_key']]);values.append(float(per.mean()))
            model.train();return float(np.mean(values))
        if args.mode=='pilot':
            # File integrity check on actual pilot inputs, never on sealed pixels.
            for i in probe_order:
                for name in ('codes','intensity'):
                    r=fit[i]
                    if sha(args.cache/r[name+'_path'])!=r[name+'_sha256']:raise ValueError('Pilot cache hash mismatch')
            first=fit_panel();before=[next(b.parameters()).detach().clone() for b in model.branches]
            head_before=model.head.raw.detach().clone();times=[];losses=[]
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();last=time.perf_counter()
            for n,(i,codes,y) in enumerate(loader(probe_order),1):
                loss,norm=step(i,codes,y);torch.cuda.synchronize();t=time.perf_counter();times.append(t-last);last=t;losses.append(loss)
                if n%25==0:update(phase='fit_only_pilot',samples=n,gradient_norm=norm,loss=loss)
            final=fit_panel();changed=[not torch.equal(old,next(b.parameters())) for old,b in zip(before,model.branches)]
            head_changed=not torch.equal(head_before,model.head.raw)
            if not all(changed) or not head_changed:raise RuntimeError('Pilot parameters unchanged')
            pilot=dict(passed=True,steps=200,seconds_per_step=float(np.mean(times[5:])),
                       fixed12_fit_mse_before=first,fixed12_fit_mse_after=final,
                       branch_parameters_changed=changed,head_changed=head_changed,
                       finite_gradients=True,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,selection_read=False,test_pixels_decoded=False,
                       disposable_weights=True,config=config)
            write_json(out/'pilot.json',pilot);update(state='complete',phase='pilot_complete',samples=200);return
        history=[];best=float('inf');samples=0;start_epoch=0;start_pos=0;elapsed_before=0.;limited=False
        carried_loss=0.
        def checkpoint_save(name,epoch,position,loss_sum):
            progress=dict(epoch=epoch,position=position,samples=samples,best=best,loss_sum=loss_sum,
                          elapsed_seconds=elapsed_before+time.monotonic()-start)
            value=dict(weights=model.weights(),optimizer=opt.state_dict(),progress=progress,config=config,
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate())
            tmp=out/(name+'.tmp');torch.save(value,tmp);tmp.replace(out/(name+'.pt'))
        if args.resume:
            ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            model.load_weights(ck['weights']);opt.load_state_dict(ck['optimizer']);p=ck['progress']
            samples=p['samples'];best=p['best'];start_epoch=p['epoch'];start_pos=p['position'];elapsed_before=p['elapsed_seconds'];carried_loss=p['loss_sum']
            torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng'])
            if (out/'history.json').exists():history=json.loads((out/'history.json').read_text())
            if start_pos==7500:start_epoch+=1;start_pos=0;carried_loss=0.
        else:checkpoint_save('initial',0,0,0.)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start_epoch,args.epochs):
            epoch_start=time.monotonic();pos0=start_pos if epoch==start_epoch else 0
            loss_sum=carried_loss if epoch==start_epoch else 0.;limited=False;position=pos0
            for position,(i,codes,y) in enumerate(loader(ordered_indices(fit,args.seed+epoch)[pos0:]),pos0+1):
                factor=lr_at(epoch+(position-1)/7500,args.epochs)
                loss,norm=step(i,codes,y,factor);loss_sum+=loss;samples+=1
                if position%50==0:update(phase='training',epoch=epoch+position/7500,samples=samples,loss=loss_sum/position,gradient_norm=norm,lr=[g['lr'] for g in opt.param_groups])
                if position%1000==0:checkpoint_save('last',epoch,position,loss_sum)
                if elapsed_before+time.monotonic()-start>=args.hours*3600:limited=True;break
            update(phase='selection',epoch=epoch+position/7500,samples=samples)
            ev=evaluate(model,sl,sel,masks);value=ev['parent_balanced_intensity_mse']
            write_json(out/f'selection_epoch_{epoch+1:03d}.json',ev)
            if value<best:best=value;checkpoint_save('best',epoch,position,loss_sum)
            record=dict(epoch=epoch+position/7500,samples=samples,selection_parent_mse=value,
                        selection_depth_mse=ev['depth_intensity_mse'],selection_amplitude_mse_rgb=ev['amplitude_mse_rgb'],
                        selection_ssim_rgb=ev['ssim_rgb'],train_mse=loss_sum/position,best=best,
                        epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=elapsed_before+time.monotonic()-start,
                        lr=[g['lr'] for g in opt.param_groups],kst=now())
            history.append(record);write_json(out/'history.json',history);checkpoint_save('last',epoch,position,loss_sum)
            if sha(manifest)!=MANIFEST_SHA:raise ValueError('Cache metadata changed')
            if limited:break
        update(phase='best_response_evaluation')
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_weights(ck['weights'])
        response=response_evaluation(model,sd,sel,masks);write_json(out/'response.json',response)
        if args.rank:write_json(out/'mode_spectrum.json',model.diagnostics())
        write_json(out/'result.json',dict(state='complete',best_selection_mse=best,best_progress=ck['progress'],
                   full_epochs_completed=sum(r['epoch'].is_integer() for r in history),samples=samples,
                   stop_reason='time_limit' if limited else 'requested_epochs',response_summary={k:v for k,v in response.items() if k!='records'},
                   peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                   elapsed_seconds=elapsed_before+time.monotonic()-start,test_pixels_decoded=False,config=config))
        update(state='complete',phase='selection_and_response_complete',best_selection_mse=best,samples=samples)
    except Exception as e:
        update(state='failed',error=repr(e),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('initial',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--mode',choices=['pilot','train'],required=True);p.add_argument('--rank',type=int,choices=[0,1,3,6],required=True)
    p.add_argument('--seed',type=int,default=117);p.add_argument('--epochs',type=int,default=12);p.add_argument('--hours',type=float,default=24)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.epochs<2 or not 0<args.hours<=24:raise ValueError('Study budget exceeded')
    run(args)


if __name__=='__main__':main()
