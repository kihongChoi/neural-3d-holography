"""Offline clean/camera comparison using fixed independent geometric/radiometric calibration."""
import argparse, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from .targets import prepare_image
from .nh3d_lut_model import sha,write_json

SHAPE=(960,1728)
CROP=(16,16,1696,928)
HALF=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
V=np.arange(256,dtype=np.float32)/255
DECODE=65535*np.where(V<=.04045,V/12.92,((V+.055)/1.055)**2.4)

def srgb(v):
    v=np.clip(v,0,1)
    return np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)

def read_planes(run,record,H):
    p=run/record['path']
    if sha(p)!=record['png_sha256']:raise ValueError('Changed camera PNG')
    bgr=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if bgr.shape!=(3000,4096,3):raise ValueError('Expected native camera RGB PNG')
    planes=[]
    for ch in range(3):
        rgb=DECODE[bgr[...,2-ch]]
        samples=.5*(rgb[::2,1::2]+rgb[1::2,::2]) if ch==1 else rgb[ch//2::2,ch//2::2]
        planes.append(cv2.warpPerspective(samples,HALF@H,(1728,960),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP))
    return np.stack(planes,axis=-1)

def quality(intensity,target,valid):
    a=srgb(intensity);b=srgb(target)
    amp=np.sqrt(np.maximum(intensity,0));truth=np.sqrt(target)
    x,y,w,h=CROP
    crop=np.zeros(SHAPE,bool);crop[y:y+h,x:x+w]=True
    masks=valid&crop[...,None]
    if masks.mean()<.9:raise ValueError('Insufficient shared valid crop')
    err_amp=(amp-truth)**2;err_lin=(intensity-target)**2;err_rgb=(a-b)**2
    mean=lambda v:float(np.mean(v[masks],dtype=np.float64))
    blur=lambda v:cv2.GaussianBlur(v,(11,11),1.5,borderType=cv2.BORDER_REFLECT)
    ma,mb=blur(a),blur(b)
    va,vb,cov=blur(a*a)-ma*ma,blur(b*b)-mb*mb,blur(a*b)-ma*mb
    sm=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    ssim_mask=np.stack([cv2.erode(masks[...,c].astype('uint8'),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool) for c in range(3)],-1)
    q=dict(amplitude_mse=mean(err_amp),linear_intensity_psnr_db=-10*np.log10(max(mean(err_lin),1e-15)),srgb_psnr_db=-10*np.log10(max(mean(err_rgb),1e-15)),srgb_ssim=float(sm[ssim_mask].mean(dtype=np.float64)),srgb_mae=mean(np.abs(a-b)),mean_normalized_intensity=mean(intensity),valid_crop_fraction=float(masks[y:y+h,x:x+w].mean()),clipped_high_fraction=mean((intensity>1).astype('float32')))
    full=[]
    for c in range(3):
        m=valid[...,c]
        full.append(dict(amplitude_mse=float(err_amp[...,c][m].mean(dtype=np.float64)),srgb_psnr_db=float(-10*np.log10(max(err_rgb[...,c][m].mean(dtype=np.float64),1e-15))),mean_intensity=float(intensity[...,c][m].mean())))
    return q,full

def png(path,rgb):
    Image.fromarray(np.uint8(np.rint(255*np.clip(rgb,0,1)))).save(path)

def panel(out,pairs,z,scenes,repeat,filename):
    # Fixed 3-column layout. Scale each native crop identically.
    pw,ph=560,306; margin=20; gap=14; top=82; rowh=365
    im=Image.new('RGB',(margin*2+3*pw+2*gap,top+rowh*len(scenes)+15),'white');d=ImageDraw.Draw(im)
    fontpath='C:/Windows/Fonts/arial.ttf'
    f=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,16);title=ImageFont.truetype(fontpath,26)
    d.text((margin,12),f'Clean / Before / After | {z:g} mm | repeat {repeat+1}',font=title,fill='black')
    d.text((margin,47),'Fixed calibration, common crop, no per-image gain or color fitting',font=small,fill='#444444')
    for row,scene in enumerate(scenes):
        pair=next(p for p in pairs if p['scene']==scene and p['depth_mm']==z and p['repeat']==repeat)
        y=top+row*rowh
        for col,key in enumerate(['clean','A','B']):
            x=margin+col*(pw+gap)
            label=pair.get('labels',{'clean':'Clean','A':'Before (identity)','B':'After (model LUT)'})[key]
            d.text((x,y),scene+'  '+label,font=f,fill='black')
            with Image.open(out/pair['files'][key]) as photo:
                photo=photo.resize((pw,ph),Image.Resampling.LANCZOS);im.paste(photo,(x,y+27))
            if key!='clean':
                q=pair['metrics'][key]
                d.text((x,y+337),f'PSNR {q["srgb_psnr_db"]:.2f} dB  |  SSIM {q["srgb_ssim"]:.4f}',font=small,fill='#333333')
    im.save(out/filename)

def run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();(out/'aligned').mkdir(exist_ok=False)
    state=json.loads((source/'ab/status.json').read_text());plan=json.loads((source/'ab_plan.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    if state['state']!='sealed_test_ab_complete':raise ValueError('Completed held-out test required')
    records={r['name']:r for r in state['records']};pairs=[];audit=[];align=[];clean_cache={}
    for z in plan['depths_mm']:
        cal=next(c for c in state['calibrations'] if c['depth_mm']==z and c['mode']=='RGB')
        assert cal['registration']['accepted']
        H=np.asarray(cal['registration']['models']['pooled'])
        prefix=f'z{z:g}_RGB'
        dark=cv2.GaussianBlur(read_planes(source,records[prefix+'_dark'],H),(0,0),12)
        white=cv2.GaussianBlur(read_planes(source,records[prefix+'_white'],H),(0,0),12)-dark
        valid=np.stack([white[...,c]>max(32.,.1*float(np.median(white[...,c]))) for c in range(3)],-1)
        align.append(dict(depth_mm=z,homography_canvas_to_full_camera=H.tolist(),independent_registration_scores=cal['registration']['scores']['pooled'],crop_xywh=CROP,valid_fraction=valid.mean(axis=(0,1)).tolist(),reference_pngs={k:records[prefix+'_'+k]['png_sha256'] for k in ['dark','white']}))
        observations=[p for p in state['comparisons'] if p['mode']=='RGB' and p['depth_mm']==z]
        for p in observations:
            scene=p['scene'];repeat=p['repeat'];meta=json.loads((source/'patterns'/f'{p["id"]:04}.json').read_text())
            original=Path(manifest['dataset'])/manifest['sources'][scene]
            assert sha(original)==meta['target']['source_sha256']
            if scene not in clean_cache:
                amplitude,target_meta=prepare_image(original,SHAPE);clean_cache[scene]=(amplitude**2,target_meta)
            target,target_meta=clean_cache[scene]
            x,y,w,h=CROP
            clean_file=f'aligned/{scene}_clean.png'
            if not (out/clean_file).exists():png(out/clean_file,srgb(target[y:y+h,x:x+w]))
            item=dict(labels={'clean':'Clean','A':'Before (previous LUT)' if plan.get('baseline_candidate') else 'Before (identity)','B':'After (updated LUT)' if plan.get('baseline_candidate') else 'After (model LUT)'},scene=scene,depth_mm=z,repeat=repeat,order=p['order'],files={'clean':clean_file},metrics={},sources={},target=target_meta)
            for method in ['A','B']:
                rec=records[p['pair'][method]['name']]
                raw=read_planes(source,rec,H)
                intensity=np.maximum(raw-dark,0)/np.maximum(white,1)
                q,full=quality(intensity,target,valid)
                file=f'aligned/{scene}_z{z:g}_r{repeat}_{method}.png';png(out/file,srgb(intensity[y:y+h,x:x+w]))
                item['files'][method]=file;item['metrics'][method]=q;item['sources'][method]=dict(path=str(source/rec['path']),sha256=rec['png_sha256'])
                for c,tag in enumerate('RGB'):
                    native=rec['metrics'][tag]
                    audit.append(dict(scene=scene,depth_mm=z,repeat=repeat,method=method,channel=tag,amplitude_mse_delta=full[c]['amplitude_mse']-native['fixed_white_amplitude_mse'],srgb_psnr_delta_db=full[c]['srgb_psnr_db']-native['fixed_clipped_srgb_psnr_db'],mean_intensity_delta=full[c]['mean_intensity']-native['mean_normalized_intensity']))
            pairs.append(item)
        print(f'Aligned and evaluated {z:g} mm, {len(observations)} paired observations',flush=True)
        for repeat in [0,1]:
            panel(out,pairs,z,plan['scenes'],repeat,f'clean_before_after_{z:g}mm_r{repeat}.png')
    assert len(pairs)==64
    summary=[]
    for depth in [None,*plan['depths_mm']]:
        group=[p for p in pairs if depth is None or p['depth_mm']==depth]
        values={m:{k:float(np.mean([p['metrics'][m][k] for p in group])) for k in group[0]['metrics'][m]} for m in ['A','B']}
        summary.append(dict(depth_mm=depth,pairs=len(group),before=values['A'],after=values['B'],amplitude_mse_reduction=1-values['B']['amplitude_mse']/values['A']['amplitude_mse'],psnr_improved=int(sum(p['metrics']['B']['srgb_psnr_db']>p['metrics']['A']['srgb_psnr_db'] for p in group)),ssim_improved=int(sum(p['metrics']['B']['srgb_ssim']>p['metrics']['A']['srgb_ssim'] for p in group))))
    max_errors={k:max(abs(r[k]) for r in audit) for k in ['amplitude_mse_delta','srgb_psnr_delta_db','mean_intensity_delta']}
    write_json(out/'png_precision_audit.json',dict(max_absolute_errors=max_errors,source='native sensor-time fixed-white metrics over the same full active field, before the new 16-pixel border crop',rows=audit))
    write_json(out/'alignment.json',dict(method='Independent depth-specific marker homography shared by both methods; native CFA-site reconstruction',entries=align))
    write_json(out/'quality.json',dict(source_run=str(source),source_status_sha256=sha(source/'ab/status.json'),source_lut_sha256=sha(source/'rgb_lut_candidate.npz'),baseline_lut_sha256=plan.get('baseline_sha256'),baseline_label=plan.get('baseline_label','identity'),metric_notes='Joint RGB sRGB PSNR; mean-channel SSIM (Gaussian 11x11, sigma 1.5, C1=.01^2, C2=.03^2), valid common crop, no fitted gain; PNG reconstruction is approximate',summary=summary,pairs=pairs))
    with (out/'quality_per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['scene','depth_mm','repeat','method',*pairs[0]['metrics']['A']];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in pairs:
            for m in ['A','B']:writer.writerow(dict(scene=p['scene'],depth_mm=p['depth_mm'],repeat=p['repeat'],method=m,**p['metrics'][m]))
    panel(out,pairs,5.,plan['scenes'][:2],0,'preview_5mm.png')
    print(json.dumps(dict(summary=summary,max_png_audit_errors=max_errors)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.source,a.output)
