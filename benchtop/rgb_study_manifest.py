"""Freeze the study scene split and matched RGB pattern recipes before acquisition."""
import argparse
import json
from pathlib import Path

import numpy as np


VERSION = 'v2.2'
DEPTHS = (3., 5., 7., 9.)
HELD_OUT = 5.
TRAIN_DEPTHS = tuple(z for z in DEPTHS if z != HELD_OUT)
PATTERNS = 1100 * len(DEPTHS)


def create(dataset, dest):
    rng = np.random.default_rng(20261103)
    ids = []
    for split, size in [('train',800), ('valid',100)]:
        files = sorted((dataset/f'DIV2K_{split}_HR').glob('*.png'))
        if len(files) != size:
            raise ValueError(f'Unexpected DIV2K {split} size')
        ids.extend([f'{split}/{f.stem}' for f in files])
    # Old bench demonstrations used official validation images. Reserve the
    # new final set from training originals outside previous low-ID demos.
    final = sorted(rng.choice(ids[100:800],100,replace=False).tolist())
    pool = rng.permutation(sorted(set(ids)-set(final))).tolist()
    splits = dict(forward_train=pool[:600],forward_val=pool[600:660],
                  lut_train=pool[660:760],lut_val=pool[760:],final_2d=final)
    rows = []
    for z in DEPTHS:
        for method,total in [('DPAC',100),('SGD',1000)]:
            for split, n in [('forward_train',total*9//10),('forward_val',total//10)]:
                selections=[]
                while len(selections)<n:
                    selections.extend(rng.permutation(splits[split]).tolist())
                for scene in selections[:n]:
                    rows.append(dict(id=len(rows),scene=scene,split=split,method=method,
                                     design_mm=z,seed=int(rng.integers(2**31)),
                                     iterations=int(rng.integers(1,101)) if method=='SGD' else 0))
    assert len(rows)==PATTERNS and len(set(sum(splits.values(),[])))==900
    report=dict(version=VERSION,seed=20261103,dataset=str(dataset.resolve()),splits=splits,
                measured_depths_mm=list(DEPTHS),sealed_observation_mm=HELD_OUT,patterns=rows)
    with dest.open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('dataset',type=Path)
    p.add_argument('dest',type=Path)
    args=p.parse_args()
    create(args.dataset,args.dest)
