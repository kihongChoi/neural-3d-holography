"""Green, native-pitch central-patch multi-plane CNNpropCNN bench experiment."""
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hardware import Bench
from .stage import Stage
from .measure_slm_plane import write
from .optics import ASM, crop_center, quantized_phase
from .paper_model import dpac, image_metrics
from .targets import prepare_image

OUT = Path("results/multiplane_green_20261102")
TRAIN = (1., 4.5, 8.)
DEPTHS = (*TRAIN, 6.25)
ZERO = 100.78
C = json.loads(Path("configs/benchtop.json").read_text())
C.update(exposure_us=100., settle_s=.45, average_frames=2)
DATA = Path.home()/"Documents/DIV2K dataset"
H0 = np.array(json.loads(Path("results/slm_plane_20261102_clean/registration.json").read_text())["homography_slm_to_half_sensor"])
DEVICE = "cuda"


def log(stage, **info):
    row = dict(stage=stage, time=time.time(), **info)
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    write(OUT/'status.json', row)
    print(row, flush=True)


def display_patch(patch):
    full = np.zeros((1080, 1920), np.float32)
    full[284:796, 704:1216] = patch
    return full


def raw(bench, full):
    image = bench.capture(full)[..., 1].copy()
    if max(f['saturated_fraction'] for f in bench.frame_metadata[-2:]) > .0001:
        raise RuntimeError('Saturated image')
    return image


def calibrate(bench, depth):
    zero = np.zeros((1080,1920), np.float32)
    flat = raw(bench, zero)
    y,x = np.indices(zero.shape)
    src = np.float32([(cx,cy) for cy in (345,540,735) for cx in (755,960,1165)])
    mask = np.zeros_like(zero, bool)
    for cx,cy in src:
        mask |= (x-cx)**2+(y-cy)**2 < 35**2
    dark = raw(bench, (mask*(1-2*((x+y)%2))*np.pi/2).astype(np.float32))
    delta = cv2.GaussianBlur((flat-dark)/np.maximum(flat,.003),(0,0),4)
    np.savez(OUT/f'diagnostic_{depth}.npz',flat=flat,dark=dark)
    predicted=cv2.perspectiveTransform(src[None],H0)[0]
    # Find the whole disk constellation even if the camera carriage was reseated.
    stamp=np.zeros((int(np.ptp(predicted[:,1]))+91,int(np.ptp(predicted[:,0]))+91),np.float32)
    anchor=predicted.min(0)-45
    for pt in predicted:
        cv2.circle(stamp,tuple(np.rint(pt-anchor).astype(int)),30,1.,-1)
    score=cv2.matchTemplate(delta,stamp,cv2.TM_CCOEFF_NORMED)
    _,strength,_,corner=cv2.minMaxLoc(score)
    print('Calibration reference', depth, float(flat.mean()), float(flat.max()),strength,corner,flush=True)
    centers = []
    for px,py in np.rint(predicted-anchor+corner).astype(int):
        window = delta[py-80:py+81,px-80:px+81]
        weights = np.maximum(window - .25*window.max(), 0)
        if window.max()<.15 or weights.sum()<30:
            raise RuntimeError(f'Missing calibration disk at {px},{py}, peak={window.max()}, weight={weights.sum()}')
        yy,xx = np.indices(weights.shape)
        centers.append([px-80+(xx*weights).sum()/weights.sum(),py-80+(yy*weights).sum()/weights.sum()])
    fit=[0,2,6,8]
    centers=np.float32(centers)
    H=cv2.getPerspectiveTransform(src[fit],centers[fit])
    err=np.linalg.norm(cv2.perspectiveTransform(src[None],H)[0]-centers,axis=1)
    np.savez(OUT/f'cal_{depth}.npz',H=H,flat=flat,dark_disks=dark,src=src,centers=centers,error=err)
    if max(err)>3:
        raise RuntimeError(f'Registration failed at {depth}: {err}')
    # SLM -> half-camera sampling coordinates on the central 256x256 native grid.
    yy,xx=np.indices((256,256),dtype=np.float32)
    xy=cv2.perspectiveTransform(np.stack((xx+832,yy+412),axis=-1),H)
    def warp(a):
        return cv2.remap(a,xy[...,0],xy[...,1],cv2.INTER_LINEAR)
    white=cv2.GaussianBlur(warp(flat),(0,0),12)
    if white.min()<.003:
        raise RuntimeError('Insufficient green white reference')
    def capture(phase):
        return np.sqrt(warp(raw(bench,display_patch(phase)))/white)
    return capture, float(max(err))


def kernel(z):
    return ASM(dict(C,slm_shape=[512,512],distance_m=z*.001), DEVICE)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        def correction():
            net=nn.Sequential(nn.Conv2d(2,12,5,padding=2),nn.SiLU(),nn.Conv2d(12,2,3,padding=1))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net
        self.slm=correction()
        self.sensor=correction()

    def forward(self,p,asm):
        u=torch.exp(1j*quantized_phase(p))
        xy=torch.view_as_real(u).permute(2,0,1)[None]
        u=u+torch.view_as_complex(self.slm(xy)[0].permute(1,2,0).contiguous())
        u=crop_center(torch.fft.ifft2(torch.fft.fft2(F.pad(u,asm.padding))*asm.kernel),(512,512))
        xy=torch.view_as_real(u).permute(2,0,1)[None]
        u=u+torch.view_as_complex(self.sensor(xy)[0].permute(1,2,0).contiguous())
        return crop_center(u.abs(),(256,256))


def t(a):
    return torch.as_tensor(a,device=DEVICE,dtype=torch.float32)


def image(split,index):
    path=DATA/f'DIV2K_{split}_HR'/f'{index:04d}.png'
    return t(prepare_image(path,(256,256))[0][...,1].copy())


def synth(model, kernels, targets, masks, initial, steps):
    p=initial.clone().detach().requires_grad_()
    opt=torch.optim.Adam([p],lr=.035)
    for _ in range(steps):
        opt.zero_grad()
        loss=0
        for k,tar,mask in zip(kernels,targets,masks):
            a=model(p,k) if model else crop_center(k(p).abs(),(256,256))
            loss=loss+((a-tar).square()*mask).sum()/mask.sum()
        loss.backward()
        opt.step()
    return p.detach()


def run():
    torch.manual_seed(1102)
    rng=np.random.default_rng(1102)
    kernels={d:kernel(d) for d in DEPTHS}
    # An asymmetric phase screen gives distinct +z and -z intensities.
    probe=t(cv2.resize(rng.normal(size=(64,64)).astype('f'),(512,512),interpolation=cv2.INTER_CUBIC))
    model=Model().to(DEVICE)
    phases={}
    samples={}
    cal={}
    with Stage(interval=(92.,102.)) as stage, Bench(C,display_wait_s=60) as bench:
        start=stage.initial['position_mm']
        try:
            for d in DEPTHS:
                pos=stage.move(ZERO-d)
                cal[d],err=calibrate(bench,d)
                actual=cal[d](probe.cpu().numpy())
                correlations=[]
                for sign in (1,-1):
                    pred=crop_center(kernel(sign*d)(probe).abs(),(256,256)).cpu().numpy()
                    correlations.append(float(np.corrcoef(actual.ravel(),pred.ravel())[0,1]))
                write(OUT/f'smoke_{d}.json',dict(position=pos,registration_max_px=err,correlations=correlations))
                log('smoke_capture',depth=d,correlations=correlations)
            votes=np.array([json.loads((OUT/f'smoke_{d}.json').read_text())['correlations'] for d in TRAIN])
            if not np.isfinite(votes).all() or abs(np.diff(votes.mean(0))[0])<.1 or votes.mean(0).max()<.25:
                raise RuntimeError(f'Ambiguous propagation sign {votes}')
            sign=(1,-1)[votes.mean(0).argmax()]
            kernels={d:kernel(sign*d) for d in DEPTHS}
            # Verify a real gradient step and camera A/B path on separate smoke target.
            smoke=Model().to(DEVICE)
            opt=torch.optim.Adam(smoke.parameters(),lr=.001)
            loss=F.l1_loss(smoke(probe,kernels[d]),t(actual))
            loss.backward()
            opt.step()
            target=image('train',200)
            trials=[]
            for m in [None,smoke]:
                p=synth(m,[kernels[d]],[target],[torch.ones_like(target)],probe,3)
                trials.append(float((t(cal[d](p.cpu().numpy()))-target).square().mean()))
            if not np.isfinite([float(loss.detach()),*trials]).all():
                raise RuntimeError('Nonfinite smoke test')
            write(OUT/'smoke_pass.json',dict(sign=sign,l1=float(loss.detach()),trial_mse=trials))
            log('smoke_pass_main_started',sign=sign)
            # Same phase IDs across planes. Split by original, never by frame.
            for i in range(48):
                target=image('train',i+1) if i<40 else image('valid',801+i-40)
                k=kernels[TRAIN[i%3]]
                p=dpac(k,target)
                if i%2:
                    p=synth(None,[k],[target],[torch.ones_like(target)],p,25)
                phases[i]=p.detach()
                np.save(OUT/f'phase_{i}.npy',p.cpu().numpy())
            for d in DEPTHS:
                stage.move(ZERO-d)
                ids=range(48) if d in TRAIN else range(40,48)
                for i in ids:
                    a=cal[d](phases[i].cpu().numpy())
                    samples[d,i]=t(a)
                    np.save(OUT/f'amp_{d}_{i}.npy',a)
                    if i%8==7: log('main_capture',depth=d,last_id=i)
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            best=float('inf')
            order=[(d,i) for d in TRAIN for i in range(40)]
            for epoch in range(18):
                rng.shuffle(order)
                for d,i in order:
                    opt.zero_grad()
                    loss=F.l1_loss(model(phases[i],kernels[d]),samples[d,i])
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    val=np.mean([float(F.l1_loss(model(phases[i],kernels[d]),samples[d,i])) for d in TRAIN for i in range(40,48)])
                if val<best:
                    best=val
                    torch.save(model.state_dict(),OUT/'best.pt')
                log('train',epoch=epoch+1,validation_l1=val,best=best)
            model.load_state_dict(torch.load(OUT/'best.pt',weights_only=True))
            model.requires_grad_(False)
            with torch.no_grad():
                write(OUT/'prediction.json',{str(d):{name:float(np.mean([float(F.l1_loss(model(phases[i],kernels[d]) if name=='learned' else crop_center(kernels[d](phases[i]).abs(),(256,256)),samples[d,i])) for i in range(40,48)])) for name in ['asm','learned']} for d in DEPTHS})
            # One phase per scene, four disjoint target regions at four physical planes.
            holograms={}
            scenes={}
            yy,xx=torch.meshgrid(torch.arange(256,device=DEVICE),torch.arange(256,device=DEVICE),indexing='ij')
            masks=[((xx>=128).int()+2*(yy>=128).int()==j).float() for j in range(4)]
            for scene in range(6):
                targets=[image('valid',841+4*scene+j) for j in range(4)]
                scenes[scene]=targets
                initial=dpac(kernels[4.5],sum(a*b for a,b in zip(targets,masks)))
                for name,m in [('asm',None),('learned',model)]:
                    p=synth(m,[kernels[d] for d in DEPTHS],targets,masks,initial,100)
                    holograms[scene,name]=p.cpu().numpy()
                    np.save(OUT/f'final_phase_{scene}_{name}.npy',holograms[scene,name])
            results=[]
            for j,d in enumerate(DEPTHS):
                stage.move(ZERO-d)
                for scene in range(6):
                    for repeat in range(2):
                        for name in (['asm','learned'] if (repeat+scene)%2==0 else ['learned','asm']):
                            a=t(cal[d](holograms[scene,name]))
                            np.save(OUT/f'final_amp_{d}_{scene}_{repeat}_{name}.npy',a.cpu().numpy())
                            roi=masks[j].bool()
                            metrics=image_metrics(a[roi].reshape(128,128),scenes[scene][j][roi].reshape(128,128))
                            results.append(dict(depth=d,scene=scene,repeat=repeat,method=name,**metrics))
                    write(OUT/'evaluation.json',results)
                log('evaluation',depth=d,completed=len(results))
            summary={name:float(np.mean([r['fixed_white_mse'] for r in results if r['method']==name])) for name in ['asm','learned']}
            write(OUT/'result.json',summary)
        finally:
            stage.move(start)
            write(OUT/'final_stage.json',stage.status())
    log('completed',**summary)


if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    try:
        run()
    except BaseException:
        log('failed',error=traceback.format_exc())
        raise

