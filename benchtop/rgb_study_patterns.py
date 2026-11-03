"""Generate the matched RGB drives specified in a frozen study manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import step01_dot_grid as grid
from .hardware import encode_phase
from .optics import ASM, optimize_phase
from .paper_model import dpac
from .targets import prepare_image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest, out, start, stop):
    recipe=json.loads(manifest.read_text(encoding='utf8'))
    if stop is None: stop=len(recipe['patterns'])
    if not 0<=start<stop<=len(recipe['patterns']):
        raise ValueError('Invalid pattern range')
    out.mkdir(parents=True,exist_ok=True)
    settings=dict(plan=recipe['version'],manifest_sha256=digest(manifest),
                  generator_sha256=digest(Path(__file__)),shape=[grid.HEIGHT,grid.WIDTH],
                  wavelengths_m=list(grid.WAVELENGTHS),asm_config=grid.C,
                  sign='positive nominal design distance',learning_rate=.04,
                  encoding='inverted 8-bit phase, PNG decoded as RGB',
                  target='prepare_image: linearize, aspect-preserving fit, centered crop, sqrt')
    spec=out/'settings.json'
    if recipe.get('version')=='v3.0-dpac':
        if any(row['method']!='DPAC' or row['iterations']!=0 for row in recipe['patterns']):
            raise ValueError('This LUT dataset accepts only single-pass DPAC')
        settings.pop('learning_rate')
    if spec.exists():
        if json.loads(spec.read_text())!=settings:
            raise RuntimeError('Generation settings changed; use separate output')
    else:
        with spec.open('x') as f:
            json.dump(settings,f,indent=2)
    depth=None
    for row in recipe['patterns'][start:stop]:
        stem=f"{row['id']:04}"
        img=out/(stem+'.png')
        meta=out/(stem+'.json')
        if meta.exists():
            if digest(img)!=json.loads(meta.read_text())['png_sha256']:
                raise RuntimeError(f'Drive hash mismatch: {stem}')
            continue
        if row['design_mm']!=depth:
            depth=row['design_mm']
            cfg=dict(grid.C,slm_shape=[grid.HEIGHT,grid.WIDTH],distance_m=depth/1000)
            models=[ASM(dict(cfg,wavelength_m=w),'cuda') for w in grid.WAVELENGTHS]
        if 'sources' in recipe:
            source=Path(recipe['dataset'])/recipe['sources'][row['scene']]
        else:
            split,idx=row['scene'].split('/')
            source=Path(recipe['dataset'])/f'DIV2K_{split}_HR'/(idx+'.png')
        amp,info=prepare_image(source,(grid.HEIGHT,grid.WIDTH))
        codes=[]
        for ch,model in enumerate(models):
            target=torch.as_tensor(np.ascontiguousarray(amp[...,ch]),device='cuda')
            if row['method']=='DPAC':
                phase=dpac(model,target)
            else:
                gen=torch.Generator(device='cuda').manual_seed(row['seed']+ch)
                initial=(torch.rand(target.shape,device='cuda',generator=gen)*2-1)*torch.pi
                phase,_=optimize_phase(model,target,cfg,iterations=row['iterations'],lr=.04,initial=initial)
            codes.append(encode_phase(phase.detach().cpu().numpy()))
        rgb=np.stack(codes,axis=-1)
        tmp=out/(stem+'.partial.png')
        if not cv2.imwrite(str(tmp),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1]):
            raise IOError(tmp)
        if not np.array_equal(cv2.imread(str(tmp))[...,::-1],rgb):
            raise IOError(f'PNG readback failed: {tmp}')
        tmp.replace(img)
        rec=dict(recipe=row,target=info,png_sha256=digest(img),
                 rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        partial=out/(stem+'.partial.json')
        partial.write_text(json.dumps(rec,indent=2),encoding='utf8')
        partial.replace(meta)
        if row['id']%25==0 or row['id']+1==stop:
            print(f"Completed pattern {row['id']} ({row['method']})",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('manifest',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int)
    a=p.parse_args()
    build(a.manifest,a.out,a.start,a.stop)
