"""Start a dated DPAC-only LUT run, preserving independent scene splits."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import re


def new_run(results, label):
    if not re.fullmatch(r'[\w가-힣]+(?:-[\w가-힣]+)*',label):
        raise ValueError('Use a descriptive label such as rgb-dpac-lut-fit')
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S')
    path=results/f'{stamp}_{label}'
    path.mkdir(parents=True,exist_ok=False)
    return path


def plan(dataset):
    rng=random.Random(1105)
    # Prefer renderer exports: fit/scene-name/frame.png, etc. All views of a
    # Gaussian scene stay in one split; report that split at scene level.
    roles={'fit':24,'selection':8,'test':8}
    sources={}
    splits={}
    if (dataset/'DIV2K_train_HR').is_dir():
        pool=sorted((dataset/'DIV2K_train_HR').glob('*.png'))
        if len(pool)<sum(roles.values()):raise ValueError('Insufficient independent DIV2K images')
        rng.shuffle(pool)
        for role,count in roles.items():
            subset,pool=pool[:count],pool[count:]
            splits[role]=[p.stem for p in subset]
            sources.update({p.stem:p.relative_to(dataset).as_posix() for p in subset})
    else:
        seen=set()
        for role,limit in roles.items():
            files=sorted((dataset/role).glob('*/*.png'))
            scenes={p.parent.name for p in files}
            if not scenes or scenes & seen:
                raise ValueError(f'Expected independent scenes under {role}/scene-name/*.png')
            seen.update(scenes)
            rng.shuffle(files)
            subset=files[:limit]
            splits[role]=[p.relative_to(dataset).as_posix() for p in subset]
            sources.update({key:key for key in splits[role]})
    rows=[]
    for role,scenes in splits.items():
        for scene in scenes:
            for z in ((3.,5.,7.,9.) if role=='test' else (3.,7.,9.)):
                rows.append(dict(id=len(rows),scene=scene,split=role,method='DPAC',design_mm=z,iterations=0))
    return dict(version='v3.0-dpac',dataset=str(dataset.resolve()),sources=sources,splits=splits,
                patterns=rows,held_out_mm=5.,observation='at design plane')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    p.add_argument('--label',default='rgb-dpac-lut-dataset')
    a=p.parse_args()
    spec=plan(a.dataset)
    run=new_run(a.results,a.label)
    manifest=run/'manifest.json'
    with manifest.open('x',encoding='utf8') as f:json.dump(spec,f,indent=2,ensure_ascii=False)
    print(f'{run}: {len(spec["patterns"])} RGB DPAC patterns',flush=True)
    from .rgb_study_patterns import build
    build(manifest,run/'patterns',0,None)
