"""Warm-start frozen CNN/CITL for a numerically background-refocused target."""
import gc
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .hardware import Bench
from .calibration import Rectifier,channel_intensity
from .optics import crop_center
from .paper_model import PaperCNNpropCNN,image_metrics
from .paper_campaign import canonical,sha
from .rgb_run import CHANNELS,read_json,select_channel,srgb_preview

root=Path('results/beamsplat_refocus_20261026')
out=root/'capture_background'
out.mkdir(exist_ok=False)
prior=Path('results/beamsplat_20261026_eval01')
protocol=read_json(prior/'protocol.json')
configs=protocol['configs']
base=dict(configs[1],settle_s=1.)
trained=Path(protocol['trained'])
rects=[Rectifier(trained/ch/'calibration',c) for ch,c in zip(CHANNELS,configs)]
source=Path(protocol['source'])
u=np.load(source/'field/U.npy')
meta=read_json(source/'summary.json')
dz_mm=-2.2
h,w=u.shape[-2:]
pitch=meta['optics']['pixel_pitch_um']*1e-6
fy=np.fft.fftfreq(2*h,pitch)[:,None]
fx=np.fft.fftfreq(2*w,pitch)[None,:]
r2=fx*fx+fy*fy
fields=[]
for ci,wave in enumerate(meta['optics']['wavelengths_nm']):
    wave*=1e-9
    k=-2*np.pi*r2/(np.sqrt(1/wave**2-r2)+1/wave)
    kernel=torch.from_numpy(np.exp(1j*k*dz_mm*1e-3).astype(np.complex64)).cuda()
    spec=torch.fft.fft2(F.pad(torch.from_numpy(u[ci]).cuda(),(w//2,w-w//2,h//2,h-h//2)))
    fields.append(crop_center(torch.fft.ifft2(spec*kernel),(h,w)).cpu().numpy())
field=np.stack(fields)
np.save(out/'U_refocused.npy',field)
del fields,spec,kernel
gc.collect()
torch.cuda.empty_cache()
rh,rw=configs[0]['roi_shape']
nh,nw=945,1680
peak=protocol['common_peak_amplitude']
target=np.zeros((rh,rw,3),np.float32)
intensity=np.moveaxis(abs(field)**2,0,-1)/peak**2
target[(rh-nh)//2:(rh-nh)//2+nh]=np.sqrt(np.maximum(cv2.resize(intensity,(nw,nh),interpolation=cv2.INTER_AREA),0))
old_target=np.load(prior/'target_amplitude.npy')
np.save(out/'target_amplitude.npy',target)
def image(path,a):
    Image.fromarray(np.uint8(np.clip(srgb_preview(a)*255,0,255))).save(path)
image(out/'target_background.png',target)
image(out/'target_original.png',old_target)
old_phase=np.load(prior/'phase_citl.npy')
phase=old_phase.copy()
model_phase=old_phase.copy()
watched=dict(protocol['hashes'])
watched[str((prior/'phase_citl.npy').resolve())]=sha(prior/'phase_citl.npy')
def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def progress(**data):
    row=dict(time=time.time(),**data)
    save(out/'status.json',row)
    print(json.dumps(row),flush=True)
save(out/'protocol.json',dict(dz_mm=dz_mm,virtual_sensor_distance_mm=15+dz_mm,
    physical_camera_plane_m=.015,policy='Numerical refocus of ORIGINAL 3um RGB complex field, followed by amplitude-only synthesis at unchanged calibrated +15mm physical plane. Same original common RGB amplitude peak and aspect-preserving 1680x945 target fit.',
    selected_from=str((root/'sweep.json').resolve()),configs=configs,hashes=watched,
    optimization='Warm start from previous best actual-camera phase. Frozen RGB CNN, Adam .005, 400 model steps then 60 actual-camera steps per color; retain best measured phase, no new model training.',
    capture='1s display dwell and 1s after timing change for final comparisons; 5-frame averages; before/after 3 repeats, alternating order; same camera exposure, gain, registration, white normalization.'))

with Bench(base) as bench:
    def drain(label):
        with (out/'frames.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(label=label,frames=bench.frame_metadata))+'\n')
        bench.frame_metadata.clear()
    def capture(p,label):
        bench.show(p)
        amps,sensors=[],[]
        for c,r in zip(configs,rects):
            select_channel(bench,c)
            bench.settle()
            rgb=bench.capture(frames=5)
            amps.append(r.amplitude(rgb))
            sensors.append(channel_intensity(rgb,c))
        drain(label)
        return np.stack(amps,-1),np.stack(sensors,-1)
    def white(label):
        a,s=capture(np.zeros_like(phase),label)
        np.savez_compressed(out/f'white_{label}.npz',amplitude=a,sensor=s)
        values=(a*a).mean((0,1))
        original=(np.load(prior/'drift_start.npz')['amplitude']**2).mean((0,1))
        if np.any(abs(values/original-1)>.2):
            raise RuntimeError('White reference differs >20%')
        return values
    start=white('start')
    for ci,(ch,c,r) in enumerate(zip(CHANNELS,configs,rects)):
        checkpoint=torch.load(trained/'models'/f'{ch}_best.pt',map_location='cpu',weights_only=False)
        assert checkpoint['config']==c
        model=PaperCNNpropCNN(c,'cuda')
        model.load_learnable(checkpoint['weights'])
        model.eval().requires_grad_(False)
        del checkpoint
        t=torch.from_numpy(target[...,ci]).cuda()
        for camera,steps in ((False,400),(True,60)):
            name=ch+('_citl' if camera else '_cnn')
            p=torch.from_numpy(phase[...,ci]).cuda().clone().requires_grad_(True)
            opt=torch.optim.Adam([p],lr=.005)
            best=float('inf')
            best_phase=p.detach().clone()
            history=[]
            for step in range(steps+1):
                bench.app.processEvents()
                if (out/'STOP_REQUESTED').exists():
                    raise InterruptedError('STOP_REQUESTED')
                opt.zero_grad(set_to_none=True)
                pred=crop_center(model(p).abs(),t.shape)
                if camera:
                    shown=phase.copy()
                    shown[...,ci]=p.detach().cpu().numpy()
                    bench.show(shown)
                    select_channel(bench,c)
                    measured=r.amplitude(bench.capture(frames=3))
                    pred=pred+(torch.from_numpy(measured).cuda()-pred).detach()
                    drain(f'{name}_{step}')
                loss=(pred-t).square().mean()
                value=float(loss.detach())
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite loss')
                if value<best:
                    best,best_phase=value,p.detach().clone()
                history.append(dict(step=step,mse=value,best=best))
                if step%20==0 or step==steps:
                    progress(stage='optimize',method=name,step=step,limit=steps,mse=value,best=best)
                    save(out/f'history_{name}.json',history)
                    np.save(out/f'best_{name}.npy',best_phase.cpu().numpy())
                if step<steps:
                    loss.backward()
                    opt.step()
            phase[...,ci]=best_phase.cpu().numpy()
            if not camera:
                model_phase[...,ci]=phase[...,ci]
        np.save(out/'partial_phase.npy',phase)
        del model,p,best_phase,t,opt
        gc.collect()
        torch.cuda.empty_cache()
    patterns={'original_focus':old_phase,'background_focus':canonical(phase,base)[1]}
    encoded,restored=canonical(phase,base)
    np.save(out/'phase_background.npy',restored)
    Image.fromarray(encoded).save(out/'slm_background.png')
    np.save(out/'phase_model_only.npy',model_phase)
    before_eval=white('before_comparison')
    records=[]
    measured_all={name:[] for name in patterns}
    for repeat in range(3):
        order=list(patterns) if repeat%2==0 else list(reversed(patterns))
        for name in order:
            progress(stage='comparison',method=name,repeat=repeat)
            a,s=capture(patterns[name],f'{name}_{repeat}')
            np.savez_compressed(out/f'capture_{name}_{repeat}.npz',amplitude=a,sensor=s)
            measured_all[name].append(a)
            reference=old_target if name=='original_focus' else target
            metrics=[image_metrics(torch.from_numpy(a[...,ci]),torch.from_numpy(reference[...,ci])) for ci in range(3)]
            records.append(dict(method=name,repeat=repeat,channels=metrics))
    end=white('end')
    if np.any(abs(end/before_eval-1)>.1):
        raise RuntimeError('White reference changed >10% across final comparisons')
    for name,arrays in measured_all.items():
        mean=np.mean(arrays,axis=0)
        np.save(out/f'mean_{name}.npy',mean)
        image(out/f'camera_{name}.png',mean)
    save(out/'metrics.json',dict(records=records,white_start=start.tolist(),white_before_comparison=before_eval.tolist(),
        white_end=end.tolist(),hashes_unchanged=all(sha(Path(p))==value for p,value in watched.items())))
    bench.show(patterns['background_focus'])
    progress(stage='completed',dz_mm=dz_mm)

font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
sheet=Image.new('RGB',(1680,1040),'#111722')
draw=ImageDraw.Draw(sheet)
for i,(file,label) in enumerate([('target_original.png','Original numerical focus'),('target_background.png','Background numerical focus: dz -2.2 mm'),
        ('camera_original_focus.png','Camera: original focus (CITL)'),('camera_background_focus.png','Camera: background focus (CITL)')]):
    im=Image.open(out/file)
    im.thumbnail((820,470))
    x,y=(i%2)*840,(i//2)*520
    sheet.paste(im,(x+10,y+40))
    draw.text((x+10,y+8),label,font=font,fill='white')
sheet.save(out/'focus_before_after.png')
