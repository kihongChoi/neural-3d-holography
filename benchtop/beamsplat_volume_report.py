"""Physical A/B and local depth-focus diagnostics, kept distinct from simulation."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .dpac_lut_session import sha,now
from .rgb_study_block import rectify
from .nh3d_lut_ab import fixed_quality
from .nh3d_aligned_quality import srgb
from .beamsplat_volume_lut import ROI

REGIONS={'left foreground':(14,19,258,291),'rear right':(619,37,881,248),'table':(305,192,577,305)}
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')
def png(p,v):Image.fromarray(np.uint8(np.rint(255*srgb(v)))).save(p)
def focus(v):
    lum=(v*np.array([.2126,.7152,.0722],np.float32)).sum(-1)
    smooth=cv2.GaussianBlur(lum,(0,0),1.2);gx=cv2.Sobel(smooth,cv2.CV_32F,1,0);gy=cv2.Sobel(smooth,cv2.CV_32F,0,1)
    return {name:float((gx[y0:y1,x0:x1]**2+gy[y0:y1,x0:x1]**2).mean()/(smooth[y0:y1,x0:x1].mean()**2+1e-12)) for name,(x0,y0,x1,y1) in REGIONS.items()}

def run(root):
    plan=json.loads((root/'ab_plan.json').read_text());state=json.loads((root/'ab/status.json').read_text())
    assert state['state']=='physical_ab_complete' and len(state['comparisons'])==54
    assert state['final_standby']['mode']=='Digital' and state['final_standby']['enable_BGR']==['Off']*3
    assert state['final_standby']['percent_BGR']==[0]*3 and abs(state['stage_end']['position_mm']-state['stage_start']['position_mm'])<.002
    assert all(c['registration']['accepted'] for c in state['calibrations'])
    assert all(abs(d['fraction'])<=.03 for d in state['white_drift'])
    out=root/'report';out.mkdir(exist_ok=False);(out/'aligned').mkdir()
    records={r['name']:r for r in state['records']};assert sum('method' in r for r in records.values())==108
    x,y,w,h=ROI;rows=[];cal=[]
    def read(name,H):
        r=records[name];p=root/r['raw_path'];assert sha(p)==r['raw_sha256'] and sha(root/r['path'])==r['png_sha256']
        with np.load(p) as f:raw=f['raw']
        return np.stack([rectify(raw,c,H) for c in range(3)],-1)
    for z in plan['depths_mm']:
        c=next(c for c in state['calibrations'] if c['depth_mm']==z and c['mode']=='RGB');H=c['registration']['models']['pooled']
        dark=cv2.GaussianBlur(read(f'z{z}_RGB_dark',H),(0,0),12)
        white=cv2.GaussianBlur(read(f'z{z}_RGB_white',H),(0,0),12)-dark
        valid=np.stack([white[...,i]>max(32.,.1*float(np.median(white[...,i]))) for i in range(3)],-1)[y:y+h,x:x+w]
        assert valid.mean()>.98
        cal.append(dict(depth_mm=z,H=H,scores=c['registration']['scores']['pooled']))
        for s in plan['scenes']:
            scene=s['scene'];targets=np.load(root/'numerical'/f'{scene}_z{z}.npy')[y:y+h,x:x+w]
            png(out/'aligned'/f'{scene}_z{z}_ideal.png',targets)
            for repeat in range(2):
                pair=next(p for p in state['comparisons'] if p['scene']==scene and p['depth_mm']==z and p['repeat']==repeat)
                for method in 'AB':
                    rec=pair['pair'][method];assert rec['base_rgb_sha256']==s['rgb_sha256']
                    values=(np.maximum(read(rec['name'],H)-dark,0)/np.maximum(white,1))[y:y+h,x:x+w]
                    channel=[fixed_quality(np.sqrt(values[...,i]),np.sqrt(targets[...,i]),valid[...,i]) for i in range(3)]
                    mse=float(((values-targets)**2)[valid].mean(dtype=np.float64))
                    srgbmse=float(((srgb(values)-srgb(targets))**2)[valid].mean(dtype=np.float64))
                    q=dict(amplitude_mse=float(np.mean([v['fixed_white_amplitude_mse'] for v in channel])),
                           linear_psnr_db=-10*np.log10(max(mse,1e-15)),srgb_psnr_db=-10*np.log10(max(srgbmse,1e-15)),
                           ssim=float(np.mean([v['fixed_clipped_srgb_ssim'] for v in channel])),mean_intensity=float(values[valid].mean()))
                    file=f'aligned/{scene}_z{z}_r{repeat}_{method}.png';png(out/file,values)
                    rows.append(dict(scene=scene,depth_mm=z,repeat=repeat,method=method,metrics=q,focus=focus(values),ideal_focus=focus(targets),file=file))
        print('Analyzed physical plane',z,flush=True)
    aggregate=lambda rs:{m:{k:float(np.mean([r['metrics'][k]for r in rs if r['method']==m])) for k in rows[0]['metrics']}for m in 'AB'}
    total=aggregate(rows);summary=dict(kst=now(),scene_photos=108,aggregate=total,
        by_sample={s['scene']:aggregate([r for r in rows if r['scene']==s['scene']]) for s in plan['scenes']},
        by_depth={str(z):aggregate([r for r in rows if r['depth_mm']==z])for z in plan['depths_mm']},
        amplitude_mse_reduction=1-total['B']['amplitude_mse']/total['A']['amplitude_mse'],
        calibration=cal,max_white_drift=max(abs(d['fraction'])for d in state['white_drift']),
        caveat='Reference is ideal propagated resampled complex field, not a photograph. Local normalized gradient also responds to speckle and static optical artifacts; peak alone does not prove volumetric fidelity.',
        fixed_drive_hashes={s['scene']:s['rgb_sha256']for s in plan['scenes']})
    peaks=[]
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for i,s in enumerate(plan['scenes']):
        for j,name in enumerate(REGIONS):
            ax=axes[i,j]
            for m in ['ideal','A','B']:
                vals=[]
                for z in plan['depths_mm']:
                    selected=[r for r in rows if r['scene']==s['scene'] and r['depth_mm']==z and (r['method']==m if m!='ideal' else r['method']=='A')]
                    vals.append(float(np.mean([r['ideal_focus' if m=='ideal' else 'focus'][name]for r in selected])))
                vals=np.array(vals);ax.plot(plan['depths_mm'],vals/max(vals.max(),1e-12),'-o',label={'ideal':'Numerical complex field','A':'Camera: identity','B':'Camera: LUT'}[m],markersize=3)
                peaks.append(dict(scene=s['scene'],region=name,method=m,peak_depth_mm=plan['depths_mm'][int(vals.argmax())],relative_span=float((vals.max()-vals.min())/max(vals.max(),1e-12)),scores=vals.tolist()))
            ax.set_title(f'{s["scene"][-6:]} | {name}');ax.grid(alpha=.3);ax.set_ylim(0,1.08)
            if i==2:ax.set_xlabel('Observed depth (mm)')
            if j==0:ax.set_ylabel('Focus score / own maximum')
    axes[0,0].legend(fontsize=7);fig.suptitle('Fixed hologram, moving camera | regional focus diagnostic (speckle-sensitive)')
    fig.tight_layout();fig.savefig(out/'focus_curves.png',dpi=150);plt.close(fig)
    summary['focus_peaks']=peaks;write(out/'summary.json',summary);write(out/'measurements.json',rows)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for s in plan['scenes']:
        scene=s['scene'];canvas=Image.new('RGB',(1680,1780),'white');d=ImageDraw.Draw(canvas)
        d.text((12,8),f'BeamSplat {scene} | world scale {s["summary"]["optics"]["world_scale"]} | fixed DPAC, camera depth scan',font=font,fill='black')
        for i,z in enumerate([1,3,5,7,9]):
            for j,m in enumerate(['ideal','A','B']):
                xx=10+j*555;yy=42+i*345
                d.text((xx,yy),f'{z} mm | '+{'ideal':'Numerical complex-field reference','A':'Camera: no LUT','B':'Camera: new LUT'}[m],font=font,fill='black')
                f=out/'aligned'/f'{scene}_z{z}_ideal.png' if m=='ideal' else out/'aligned'/f'{scene}_z{z}_r0_{m}.png'
                with Image.open(f)as im:canvas.paste(im.resize((545,306),Image.Resampling.LANCZOS),(xx,yy+24))
                if m!='ideal':
                    q=next(r['metrics']for r in rows if r['scene']==scene and r['depth_mm']==z and r['repeat']==0 and r['method']==m)
                    d.text((xx,yy+329),f'PSNR {q["srgb_psnr_db"]:.2f} dB | SSIM {q["ssim"]:.3f}',font=small,fill='black')
        canvas.save(out/f'volume_{scene}.png')
        frames=[]
        for z in plan['depths_mm']:
            frame=Image.new('RGB',(1380,310),'white');draw=ImageDraw.Draw(frame)
            draw.text((12,5),f'{scene} | camera plane {z} mm | same fixed hologram at every depth',font=font,fill='black')
            for j,m in enumerate(['ideal','A','B']):
                xx=10+j*460
                draw.text((xx,32),{'ideal':'Numerical reference','A':'Camera: identity','B':'Camera: LUT'}[m],font=font,fill='black')
                f=out/'aligned'/f'{scene}_z{z}_ideal.png' if m=='ideal' else out/'aligned'/f'{scene}_z{z}_r0_{m}.png'
                with Image.open(f)as im:frame.paste(im.resize((450,253),Image.Resampling.LANCZOS),(xx,56))
            frames.append(frame)
        frames[0].save(out/f'sweep_{scene}.gif',save_all=True,append_images=frames[1:]+frames[-2:0:-1],duration=650,loop=0)
    print(json.dumps({k:summary[k]for k in ['aggregate','amplitude_mse_reduction','max_white_drift']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
