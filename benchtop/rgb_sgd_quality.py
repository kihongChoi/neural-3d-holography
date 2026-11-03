"""Evaluate encoded SGD patterns at their design plane against the original target."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .hardware import encode_phase
from .optics import ASM, optimize_phase
from .paper_model import image_metrics
from .targets import prepare_image


def audit(patterns, pattern_id, out, limit, fixed_white=False):
    stem=f'{pattern_id:04}'
    settings=json.loads((patterns/'settings.json').read_text())
    meta=json.loads((patterns/(stem+'.json')).read_text())
    recipe=meta['recipe']
    if recipe['method']!='SGD' or recipe['iterations']>limit:
        raise ValueError('Choose an SGD sample with its original iteration count in range')
    out.mkdir(parents=True,exist_ok=False)
    source=patterns/(stem+'.png')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['png_sha256']
    original=cv2.imread(str(source))[...,::-1]
    shape=settings['shape']
    amp,info=prepare_image(meta['target']['source_path'],shape)
    assert info['source_sha256']==meta['target']['source_sha256']
    cfg=dict(settings['asm_config'],slm_shape=shape,distance_m=recipe['design_mm']/1000)
    checks=sorted({k for k in (0,recipe['iterations'],100,200,*range(500,limit+1,500),limit) if k<=limit})
    report=dict(recipe=recipe,source_hash=meta['png_sha256'],checks=checks,channels={},fixed_white_objective=fixed_white)
    def publish():
        tmp=out/'quality.tmp'
        tmp.write_text(json.dumps(report,indent=2))
        tmp.replace(out/'quality.json')
    for c,tag in enumerate('RGB'):
        model=ASM(dict(cfg,wavelength_m=settings['wavelengths_m'][c]),'cuda')
        t=torch.as_tensor(np.ascontiguousarray(amp[...,c]),device='cuda')
        gen=torch.Generator(device='cuda').manual_seed(recipe['seed']+c)
        initial=(torch.rand(t.shape,device='cuda',generator=gen)*2-1)*torch.pi
        def score(code):
            phi=(1-torch.tensor(code.copy(),device='cuda').float()/255)*2*torch.pi-torch.pi
            with torch.no_grad():
                pred=model(phi,quantize=False).abs()
                return image_metrics(pred,t) | image_metrics(pred,t,fit_brightness=False)
        records={}
        report['channels'][tag]=records
        best=float('inf')
        selected=None
        def trace(k, metrics, phase, *unused):
            nonlocal best,selected
            if metrics['amplitude_mse']<best:
                selected=phase.clone()
                best=metrics['amplitude_mse']
            if k in checks:
                code=encode_phase(selected.cpu().numpy())
                records[k]=score(code)
                if k==recipe['iterations']:
                    records[k]['original_png']=score(original[...,c])
                    records[k]['exact_code_fraction']=float((code==original[...,c]).mean())
                # Retain best encoded endpoint for later optical A/B measurement.
                if k==limit:
                    if not cv2.imwrite(str(out/(tag+'.png')),code):raise IOError(tag)
                publish()
        optimize_phase(model,t,cfg,iterations=limit,lr=settings['learning_rate'],initial=initial,trace=trace,
                       fixed_scale=1. if fixed_white else None)
    rgb=np.stack([cv2.imread(str(out/(tag+'.png')),cv2.IMREAD_GRAYSCALE) for tag in 'RGB'],axis=-1)
    if not cv2.imwrite(str(out/'drive.png'),rgb[...,::-1]):raise IOError('drive.png')
    report['summary']={}
    for k in checks:
        rows=[report['channels'][tag][k] for tag in 'RGB']
        if k==recipe['iterations'] and not fixed_white:rows=[r['original_png'] for r in rows]
        report['summary'][k]={key:float(np.mean([r[key] for r in rows])) for key in
                            ('fixed_white_mse','fitted_amplitude_mse','fitted_clipped_srgb_psnr','fitted_clipped_srgb_ssim',
                             'fixed_clipped_srgb_psnr','fixed_clipped_srgb_ssim')}
    publish()
    print(report['summary'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patterns',type=Path)
    parser.add_argument('pattern_id',type=int)
    parser.add_argument('out',type=Path)
    parser.add_argument('--limit',type=int,default=3000)
    parser.add_argument('--fixed-white',action='store_true')
    a=parser.parse_args()
    audit(a.patterns,a.pattern_id,a.out,a.limit,a.fixed_white)
