"""Offline joint RGB teacher training; no test decoding or hardware control."""
import argparse, json, math, os, random, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from .nh3d_lut_model import DepthSharedNH3D, optics
from .integrated_rgb_data import now, sha, write_json


class CameraHead(nn.Module):
    """Positive intensity mixture. No arbitrary per-image exposure adjustment."""
    def __init__(self):
        super().__init__()
        matrix=torch.eye(3)+.01*(1-torch.eye(3))
        self.raw=nn.Parameter(torch.log(torch.expm1(matrix)))
    def forward(self,intensity):
        return torch.einsum('ij,jhw->ihw',F.softplus(self.raw),intensity)


class JointRGB(nn.Module):
    def __init__(self,initial):
        super().__init__()
        self.branches=nn.ModuleList([DepthSharedNH3D(optics(c),'cuda',depths=(3.,5.,7.,9.)) for c in 'RGB'])
        self.head=CameraHead().cuda()
        self.initial=Path(initial)
        self.reset()
    def reset(self):
        for c,m in zip('RGB',self.branches):
            state=torch.load(self.initial/c/'best.pt',map_location='cpu',weights_only=True)
            m.load_learnable(state['weights'])
        self.head.load_state_dict(CameraHead().state_dict())
    def forward(self,codes,z,ideal=False):
        values=[(m.ideal_amplitude(codes[c],z) if ideal else m.amplitude(codes[c],z)).square() for c,m in enumerate(self.branches)]
        return self.head(torch.stack(values))
    def weights(self):
        return dict(branches=[m.learnable_state() for m in self.branches],head={k:v.detach().cpu().clone() for k,v in self.head.state_dict().items()})
    def load_weights(self,w):
        for m,state in zip(self.branches,w['branches']):m.load_learnable(state)
        self.head.load_state_dict(w['head'])


class Frames(Dataset):
    def __init__(self,root,rows):self.root=Path(root);self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        if r['split'] not in ('fit','selection') or r['depth_mm'] not in (3.,7.,9.):raise ValueError('Sealed data access')
        # Full physical grid retained. NumPy file caches avoid repeated decompression.
        codes=np.load(self.root/r['codes_path'],allow_pickle=False)
        y=np.load(self.root/r['intensity_path'],allow_pickle=False)
        if codes.shape!=(3,960,1728) or codes.dtype!=np.uint8 or y.shape!=codes.shape or y.dtype!=np.float32:raise ValueError('Cache shape/dtype')
        return i,torch.from_numpy(codes),torch.from_numpy(y)


def ordered_indices(rows,seed):
    rng=random.Random(seed);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['depth_mm']].append(i)
    for v in groups.values():rng.shuffle(v)
    return [i for group in zip(*(groups[z] for z in sorted(groups))) for i in group]


def lr_at(progress,total):
    if progress<1:return .1+.9*progress
    return .1+.9*.5*(1+math.cos(math.pi*min(1.,(progress-1)/(total-1))))


def masked_errors(pred,target,mask):
    denom=mask.sum(dim=(-2,-1)).clamp_min(1)
    per=((pred-target).square()*mask).sum(dim=(-2,-1))/denom
    amp=((torch.sqrt(pred.clamp_min(1e-12))-torch.sqrt(target.clamp_min(1e-12))).square()*mask).sum(dim=(-2,-1))/denom
    return per,amp


def masked_ssim(pred,target,mask):
    # Explicit fixed-range, 11x11 uniform-window SSIM on valid neighborhoods.
    x=pred[None];y=target[None];pool=lambda t:F.avg_pool2d(t,11,stride=1)
    a=pool(x);b=pool(y);va=(pool(x*x)-a*a).clamp_min(0);vb=(pool(y*y)-b*b).clamp_min(0);cov=pool(x*y)-a*b
    scores=((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(va+vb+.03**2))
    valid=pool(mask[None].float())>.999
    return (scores*valid).sum(dim=(-2,-1))[0]/valid.sum(dim=(-2,-1))[0].clamp_min(1)


def save_checkpoint(path,model,opt,progress,manifest_sha):
    data=dict(version='joint-rgb-v1',weights=model.weights(),optimizer=opt.state_dict(),progress=progress,
              cache_manifest_sha256=manifest_sha,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
              python_rng=random.getstate(),numpy_rng=np.random.get_state(),precision='float32_complex64',
              scheduler=dict(kind='one_epoch_warmup_cosine',branch_lr=3e-5,head_lr=1e-4,min_factor=.1),saved_kst=now())
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)


@torch.no_grad()
def evaluate(model,loader,rows,masks,ideal=False):
    model.eval();records=[]
    for i,codes,y in loader:
        r=rows[i];codes=codes.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        pred=model(codes,r['depth_mm'],ideal);mask=masks[r['reference_key']]
        per,amp=masked_errors(pred,y,mask)
        if not torch.isfinite(per).all():raise RuntimeError('Nonfinite selection prediction')
        records.append(dict(id=r['id'],scene=r['scene'],depth_mm=r['depth_mm'],condition=r['condition'],
                            intensity_mse_rgb=per.tolist(),amplitude_mse_rgb=amp.tolist(),intensity_mse=float(per.mean()),
                            ssim_rgb=masked_ssim(pred,y,mask).tolist(),
                            psnr_db=float(-10*torch.log10(per.clamp_min(1e-12)).mean())))
    parent=defaultdict(list);depth=defaultdict(list)
    for r in records:parent[r['scene']].append(r['intensity_mse']);depth[str(r['depth_mm'])].append(r['intensity_mse'])
    result=dict(parent_balanced_intensity_mse=float(np.mean([np.mean(v) for v in parent.values()])),
                depth_intensity_mse={z:float(np.mean(v)) for z,v in depth.items()},
                amplitude_mse_rgb=np.mean([r['amplitude_mse_rgb'] for r in records],axis=0).tolist(),
                ssim_rgb=np.mean([r['ssim_rgb'] for r in records],axis=0).tolist(),ssim_definition='uniform11 window, data_range1, valid neighborhoods',
                psnr_db=float(np.mean([r['psnr_db'] for r in records])),records=records)
    model.train();return result


def run(args):
    torch.set_num_threads(8);torch.manual_seed(117);random.seed(117);np.random.seed(117)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(state='initializing',started_kst=now(),phase='initialization',actual_dataset_training_started=False,
                                     epoch=0,samples=0,optimizer_steps=0,test_pixels_decoded=False)
    def update(**kw):
        status.update(kw,updated_kst=now(),elapsed_seconds=time.monotonic()-start)
        write_json(out/'status.json',status);print(json.dumps(status),flush=True)
    update()
    try:
        manifest=args.cache/'manifest.json';mh=sha(manifest);data=json.loads(manifest.read_text())
        if not data['complete'] or data['pilot_only'] or data['counts']!={'fit':7500,'selection':300} or data['test_pixels_decoded']:raise ValueError('Unapproved cache')
        fit=[r for r in data['records'] if r['split']=='fit'];sel=[r for r in data['records'] if r['split']=='selection']
        if {r['scene'] for r in fit}&{r['scene'] for r in sel}:raise ValueError('Parent leakage')
        masks={}
        for k,r in data['references'].items():
            path=args.cache/r['file']
            if sha(path)!=r['sha256']:raise ValueError('Reference changed')
            with np.load(path) as f:masks[k]=torch.from_numpy(f['valid'].copy()).cuda()
        parent_counts=Counter(r['scene'] for r in fit);sample_weight={k:len(fit)/(len(parent_counts)*n) for k,n in parent_counts.items()}
        fitds=Frames(args.cache,fit);selds=Frames(args.cache,sel)
        def loader(order):return DataLoader(fitds,batch_size=None,sampler=order,num_workers=2,pin_memory=True,prefetch_factor=2)
        selloader=DataLoader(selds,batch_size=None,num_workers=2,pin_memory=True,prefetch_factor=2,persistent_workers=True)
        initial_hashes={c:sha(args.initial/c/'best.pt') for c in 'RGB'}
        model=JointRGB(args.initial)
        config=dict(started_kst=now(),seed=117,epochs=args.epochs,phase_limit_hours=args.hours,cache_manifest_sha256=mh,
                    initial_teacher_sha256=initial_hashes,device=torch.cuda.get_device_name(),torch=torch.__version__,
                    fit=7500,selection=300,sealed_test=400,head='positive3x3 intensity mixture, no spatial residual',
                    branch_lr=3e-5,head_lr=1e-4,optimizer='Adam foreach=False',gradient_clip_norm=1.,parent_balanced=True,
                    pilot_steps=200,baseline_head_full_epochs=1,main_sample_forwards_per_epoch=7500,paired_forwards=0,
                    code='native1728x960; decoded pi-code*2pi/255; three separate complex fields; intensity mixture')
        write_json(out/'config.json',config)
        def optimizer(joint=True):
            for m in model.branches:m.requires_grad_(joint)
            groups=[dict(params=list(model.head.parameters()),lr=1e-4)]
            if joint:groups.insert(0,dict(params=[p for m in model.branches for p in m.parameters()],lr=3e-5))
            return torch.optim.Adam(groups,foreach=False)
        def step(opt,i,codes,y,ideal=False):
            r=fit[i];opt.zero_grad(set_to_none=True)
            codes=codes.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            pred=model(codes,r['depth_mm'],ideal);per,_=masked_errors(pred,y,masks[r['reference_key']])
            loss=per.mean()*sample_weight[r['scene']]
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.,error_if_nonfinite=True)
            opt.step();return float(loss.detach()),float(norm)
        # Actual measured data pilot, then restore the pre-pilot initialization.
        opt=optimizer();times=[];losses=[];torch.cuda.reset_peak_memory_stats();update(phase='actual_data_pilot',actual_dataset_training_started=True)
        before=model.head.raw.detach().clone();last=time.perf_counter()
        for k,(i,codes,y) in enumerate(loader(ordered_indices(fit,991)[:200]),1):
            loss,norm=step(opt,i,codes,y);torch.cuda.synchronize();t=time.perf_counter();times.append(t-last);last=t;losses.append(loss)
            if k%25==0:update(phase_samples=k,pilot_optimizer_steps=k,train_mse_weighted=loss,gradient_norm=norm,lr=[g['lr'] for g in opt.param_groups])
        if torch.equal(before,model.head.raw):raise RuntimeError('No pilot parameter update')
        pilot=dict(kind='actual_integrated_rgb_fit_training_pilot',steps=200,seconds_per_step=float(np.mean(times[5:])),
                   peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                   loss_first20=float(np.mean(losses[:20])),loss_last20=float(np.mean(losses[-20:])),finite_gradients=True,
                   parameters_updated=True,weights_restored_before_main=True,test_pixels_decoded=False)
        write_json(out/'pilot.json',pilot);del opt;model.reset();torch.cuda.empty_cache()
        # Equal full-fit head calibration budgets for both frozen baselines.
        for label,ideal in [('ideal_asm',True),('frozen_teacher',False)]:
            model.reset();opt=optimizer(False);update(phase='baseline_'+label,phase_samples=0)
            loss_sum=0.
            for k,(i,codes,y) in enumerate(loader(ordered_indices(fit,700)),1):
                loss,norm=step(opt,i,codes,y,ideal);loss_sum+=loss
                if k%50==0:update(phase_samples=k,train_mse_weighted=loss_sum/k,lr=[g['lr'] for g in opt.param_groups])
            ev=evaluate(model,selloader,sel,masks,ideal);write_json(out/('baseline_'+label+'.json'),ev)
            write_json(out/('baseline_'+label+'_head.json'),dict(matrix=F.softplus(model.head.raw).detach().cpu().tolist(),samples=7500,optimizer_steps=7500))
            del opt
        # Start from the original wavelength branches plus calibrated teacher head.
        opt=optimizer(True);best=float('inf');samples=0;main_start=time.monotonic();history=[]
        update(phase='joint_training',main_started_kst=now(),phase_samples=0,epoch=0,samples=0,optimizer_steps=0)
        save_checkpoint(out/'initial.pt',model,opt,dict(epoch=0,samples=0,optimizer_steps=0),mh)
        for epoch in range(args.epochs):
            epoch_start=time.monotonic();loss_sum=0.;count=0;time_limited=False
            for k,(i,codes,y) in enumerate(loader(ordered_indices(fit,117+epoch)),1):
                factor=lr_at(epoch+(k-1)/7500,args.epochs)
                for g,base in zip(opt.param_groups,(3e-5,1e-4)):g['lr']=base*factor
                loss,norm=step(opt,i,codes,y);loss_sum+=loss;count=k;samples+=1
                progress=dict(epoch=epoch,position=k,samples=samples,optimizer_steps=samples,forward_samples=samples,
                              epoch_loss_sum=loss_sum,main_elapsed_seconds=time.monotonic()-main_start,
                              requested_epochs=args.epochs,phase_limit_hours=args.hours,
                              seed=117+epoch,lr=[g['lr'] for g in opt.param_groups],best_selection_mse=best if math.isfinite(best) else None)
                if k%25==0:update(phase='joint_training',epoch=epoch+(k/7500),samples=samples,optimizer_steps=samples,phase_samples=k,
                                  train_mse_weighted=loss_sum/k,gradient_norm=norm,lr=progress['lr'],
                                  main_elapsed_seconds=time.monotonic()-main_start)
                if k%500==0:save_checkpoint(out/'last.pt',model,opt,progress,mh)
                if time.monotonic()-main_start>=args.hours*3600:time_limited=True;break
            ev_start=time.monotonic();update(phase='selection',epoch=epoch+count/7500)
            ev=evaluate(model,selloader,sel,masks);value=ev['parent_balanced_intensity_mse']
            ev_path=out/f'selection_epoch_{epoch+1:03d}.json';write_json(ev_path,ev)
            if value<best:
                best=value;progress['best_selection_mse']=best;save_checkpoint(out/'best.pt',model,opt,progress,mh)
            save_checkpoint(out/'last.pt',model,opt,progress,mh)
            item=dict(epoch=epoch+count/7500,samples=samples,optimizer_steps=samples,train_mse_weighted=loss_sum/count,
                      selection_parent_mse=value,selection_depth_mse=ev['depth_intensity_mse'],selection_amplitude_mse_rgb=ev['amplitude_mse_rgb'],
                      selection_ssim_rgb=ev['ssim_rgb'],
                      lr=progress['lr'],epoch_seconds=time.monotonic()-epoch_start,selection_seconds=time.monotonic()-ev_start,
                      elapsed_seconds=time.monotonic()-start,kst=now(),best_selection_mse=best)
            history.append(item);write_json(out/'history.json',history)
            if sha(manifest)!=mh:raise RuntimeError('Cache metadata changed during training')
            if time_limited:break
        update(state='complete',phase='joint_training_complete',epoch=history[-1]['epoch'],samples=samples,optimizer_steps=samples,
               best_selection_mse=best,stop_reason='phase_time_budget' if time_limited else 'requested_full_epochs',
               next_phase='response fine-tuning and controls require separate execution; LUT/test not run')
    except Exception as e:
        update(state='failed',error=repr(e),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('initial',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--hours',type=float,default=50);a=p.parse_args()
    if a.epochs<2 or not 0<a.hours<=50:raise ValueError('Invalid main training budget')
    run(a)
if __name__=='__main__':main()
