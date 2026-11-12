"""Fixed-calibration raw Bayer physical LUT comparison report."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from .dpac_lut_session import sha,now
from .rgb_study_block import rectify
from .targets import prepare_image
from .nh3d_aligned_quality import quality,srgb,CROP


def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def run(root):
    state=json.loads((root/'ab/status.json').read_text())
    assert state['state']=='physical_ab_complete' and not state.get('error')
    assert state['final_standby']['mode']=='Digital'
    assert state['final_standby']['enable_BGR']==['Off']*3
    assert state['final_standby']['percent_BGR']==[0]*3
    assert abs(state['stage_end']['position_mm']-state['stage_start']['position_mm'])<.002
    assert all(c['registration']['accepted'] for c in state['calibrations'])
    assert all(abs(d['fraction'])<=.03 for d in state['white_drift'])
    plan=json.loads((root/'ab_plan.json').read_text())
    assert len(state['comparisons'])==plan['expected_scene_captures']//3
    manifest=json.loads((root/'manifest.json').read_text())
    out=root/'report';out.mkdir(exist_ok=False);(out/'aligned').mkdir()
    records={r['name']:r for r in state['records']}
    assert len([r for r in records.values() if 'method' in r])==plan['expected_scene_captures']
    def planes(name,H):
        r=records[name];p=root/r['raw_path']
        assert sha(p)==r['raw_sha256'] and sha(root/r['path'])==r['png_sha256']
        with np.load(p,allow_pickle=False) as f:raw=f['raw']
        return np.stack([rectify(raw,c,H) for c in range(3)],-1)
    pairs=[];calibrations=[]
    x,y,w,h=CROP
    clean={}
    for s in plan['scenes']:
        source=Path(manifest['dataset'])/manifest['sources'][s]
        assert sha(source)==plan['source_sha256'][s]
        amp,_=prepare_image(source,(960,1728));clean[s]=amp**2
    for z in plan['depths_mm']:
        c=next(c for c in state['calibrations'] if c['mode']=='RGB' and c['depth_mm']==z)
        H=c['registration']['models']['pooled'];prefix=f'z{z:g}_RGB'
        dark=cv2.GaussianBlur(planes(prefix+'_dark',H),(0,0),12)
        white=cv2.GaussianBlur(planes(prefix+'_white',H),(0,0),12)-dark
        valid=np.stack([white[...,i]>max(32.,.1*float(np.median(white[...,i]))) for i in range(3)],-1)
        calibrations.append(dict(depth_mm=z,H=H,scores=c['registration']['scores']['pooled'],crop_xywh=CROP))
        for pair in [p for p in state['comparisons'] if p['mode']=='RGB' and p['depth_mm']==z]:
            assert len({r['base_rgb_sha256'] for r in pair['pair'].values()})==1
            s=pair['scene'];r=pair['repeat'];values={'clean':clean[s]};metrics={};files={}
            for method in 'ABC':
                v=np.maximum(planes(pair['pair'][method]['name'],H)-dark,0)/np.maximum(white,1)
                metrics[method],_=quality(v,clean[s],valid);values[method]=v
            for method,v in values.items():
                f=f'aligned/z{z:g}_{s}_r{r}_{method}.png'
                Image.fromarray(np.uint8(np.rint(srgb(v[y:y+h,x:x+w])*255))).save(out/f)
                files[method]=f
            pairs.append(dict(scene=s,depth_mm=z,repeat=r,metrics=metrics,files=files))
        print(f'Reported RGB depth {z:g}',flush=True)
    keys=['amplitude_mse','linear_intensity_psnr_db','srgb_psnr_db','srgb_ssim','mean_normalized_intensity']
    aggregate=lambda ps:{m:{k:float(np.mean([p['metrics'][m][k] for p in ps])) for k in keys} for m in 'ABC'}
    total=aggregate(pairs)
    depths={str(z):aggregate([p for p in pairs if p['depth_mm']==z]) for z in plan['depths_mm']}
    wins=sum(p['metrics']['B']['amplitude_mse']<p['metrics']['A']['amplitude_mse'] for p in pairs)
    repeat_diff={m:float(np.mean([abs(p['metrics'][m]['amplitude_mse']-next(q for q in pairs if q['scene']==p['scene'] and q['depth_mm']==p['depth_mm'] and q['repeat']==1)['metrics'][m]['amplitude_mse']) for p in pairs if p['repeat']==0])) for m in 'ABC'}
    summary=dict(completed_kst=now(),source_status_sha256=sha(root/'ab/status.json'),
        scene_photos=plan['expected_scene_captures'],RGB_pairs=len(pairs),aggregate=total,by_depth=depths,
        amplitude_mse_reduction_B_vs_A=1-total['B']['amplitude_mse']/total['A']['amplitude_mse'],
        amplitude_mse_reduction_B_vs_C=1-total['B']['amplitude_mse']/total['C']['amplitude_mse'],
        B_better_than_A_observations=wins,repeat_amplitude_mse_absolute_difference=repeat_diff,
        max_white_drift=max(abs(d['fraction']) for d in state['white_drift']),
        note='Physical camera measurements; fixed white normalization and independent geometry. Four external parents, not a large generalization study. Display clipped to sRGB, no per-image gain fitting.')
    write(out/'summary.json',summary);write(out/'pairs.json',pairs);write(out/'calibration.json',calibrations)
    render_montages(root)
    print(json.dumps(summary,indent=2))


def render_montages(root):
    """Re-export target/identity/new-LUT panels from existing aligned evidence."""
    out=root/'report'
    plan=json.loads((root/'ab_plan.json').read_text())
    pairs=json.loads((out/'pairs.json').read_text())
    methods=['clean','A','B']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    labels={'clean':'Clean target','A':'Before: identity','B':'After: new LUT'}
    for z in plan['depths_mm']:
        canvas=Image.new('RGB',(15+455*len(methods),1200),'white');draw=ImageDraw.Draw(canvas)
        draw.text((15,10),f'Physical RGB capture | {z:g} mm | repeat 1 | fixed calibration and common crop',fill='black',font=font)
        for i,s in enumerate(plan['scenes']):
            p=next(p for p in pairs if p['scene']==s and p['depth_mm']==z and p['repeat']==0)
            for j,m in enumerate(methods):
                xx=15+j*455;yy=45+i*287
                draw.text((xx,yy),f'{s}  {labels[m]}',fill='black',font=font)
                with Image.open(out/p['files'][m]) as im:canvas.paste(im.resize((440,241),Image.Resampling.LANCZOS),(xx,yy+25))
                if m!='clean':
                    q=p['metrics'][m];draw.text((xx,yy+267),f'PSNR {q["srgb_psnr_db"]:.2f} dB / SSIM {q["srgb_ssim"]:.3f}',fill='black',font=small)
        canvas.save(out/f'montage_z{z:g}.png')
    # One fixed center ROI per parent, native pixels with identical enlargement.
    for z in [7.,5.]:
        canvas=Image.new('RGB',(10+485*len(methods),1140),'white');draw=ImageDraw.Draw(canvas)
        draw.text((15,8),f'Detail | {z:g} mm | same central 320 x 160 crop, 1.5x nearest-neighbor',fill='black',font=font)
        for i,s in enumerate(plan['scenes']):
            p=next(p for p in pairs if p['scene']==s and p['depth_mm']==z and p['repeat']==0)
            for j,m in enumerate(methods):
                xx=10+j*485;yy=40+i*272
                draw.text((xx,yy),f'{s} {labels[m]}',fill='black',font=font)
                with Image.open(out/p['files'][m]) as im:canvas.paste(im.crop((688,384,1008,544)).resize((480,240),Image.Resampling.NEAREST),(xx,yy+25))
        canvas.save(out/f'detail_z{z:g}.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--montages-only',action='store_true',help='Replace montage/detail PNGs using existing aligned images and metrics.')
    args=p.parse_args()
    if args.montages_only:render_montages(args.root)
    else:run(args.root)
