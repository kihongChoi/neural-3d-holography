from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
r=Path('results')/(datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S')+'_frozen-model-detail-montage')
r.mkdir(exist_ok=False)
remote='/shared/projects/khchoi/results/'+r.name
script=r'''import os,sys,json,hashlib,gc
from pathlib import Path
import numpy as np
import torch
base=Path('/shared/projects/khchoi/results')
blur=base/'20261108-101900_multimode-blur-control'
sys.path.insert(0,str(blur/'code'))
from benchtop.integrated_rgb_train import Frames,JointRGB
from benchtop.multimode_model import MultimodeRGB
from benchtop.multimode_blur_model import BlurMultimodeRGB
from benchtop.integrated_rgb_data import sha
from benchtop.multimode_train import validate_manifest,MANIFEST_SHA
out=Path(__file__).parent
assert os.environ.get('SLURM_JOB_ID')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cache=base/'20261106-212946_a100-integrated-rgb-training/cache'
assert sha(cache/'manifest.json')==MANIFEST_SHA
d=json.loads((cache/'manifest.json').read_text());validate_manifest(d)
rows=[min([r for r in d['records'] if r['split']=='selection' and r['scene']==scene and r['depth_mm']==7 and r['condition']=='current_lut'],key=lambda r:r['id']) for scene in ['0022','0108']]
ds=Frames(cache,rows)
rois=[dict(name='bridge',scene='0022',x=600,y=240,w=320,h=240),dict(name='boats',scene='0022',x=300,y=500,w=320,h=240),dict(name='coral',scene='0108',x=900,y=400,w=320,h=240)]
arrays={}
for i,r in enumerate(rows):
    assert sha(cache/r['intensity_path'])==r['intensity_sha256']
    assert sha(cache/r['codes_path'])==r['codes_sha256']
    _,codes,y=ds[i]
    for roi in rois:
        if roi['scene']==r['scene']:
            x,z,w,h=[roi[k] for k in ['x','y','w','h']]
            arrays[roi['name']+'_measured']=y[:,z:z+h,x:x+w].numpy()
    arrays['context_'+r['scene']]=y.numpy()
freeze=json.loads((base/'20261109-011639_multimode-sealed-test/freeze.json').read_text())
checks=[]
with torch.no_grad():
    for rank in [1,3,6,0,-1]:
        name='k1_blur_s117' if rank==-1 else f'k{rank}_s117'
        m=next(x for x in freeze['models'] if x['name']==name)
        assert sha(m['path'])==m['sha256']
        model=BlurMultimodeRGB(1,'cuda') if rank==-1 else (JointRGB(base/'20261106-185858_a100-dataset-transfer/initial_teacher') if rank==0 else MultimodeRGB(rank,'cuda'))
        ck=torch.load(m['path'],map_location='cpu',weights_only=False);model.load_weights(ck['weights']);del ck;model.eval()
        for i,r in enumerate(rows):
            _,codes,y=ds[i];pred=model(codes.cuda(),7).cpu().numpy();assert np.isfinite(pred).all()
            for roi in rois:
                if roi['scene']==r['scene']:
                    x,z,w,h=[roi[k] for k in ['x','y','w','h']]
                    arrays[roi['name']+'_'+name]=pred[:,z:z+h,x:x+w].copy()
        checks.append(m);del model;gc.collect();torch.cuda.empty_cache();print(name,flush=True)
np.savez_compressed(out/'details.npz',**arrays)
(out/'evidence.json').write_text(json.dumps(dict(state='complete',split='selection',depth_mm=7,seed=117,rois=rois,records=rows,models=checks,new_training=False,test_access=False,raw_resolution=[960,1728]),indent=2))
print('COMPLETE',flush=True)
'''
(r/'detail_inference.py').write_text(script,encoding='utf-8')
sb=f'''#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=frozen-details
#SBATCH --output={remote}/inference-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/shared/envs/py312-torch-cu121/bin/python /shared/projects/khchoi/results/20261108-101900_multimode-blur-control/source_gate.py
/shared/envs/py312-torch-cu121/bin/python {remote}/detail_inference.py
'''
(r/'detail.sbatch').write_text(sb,encoding='utf-8',newline='\n')
(r/'protocol.json').write_text(json.dumps(dict(purpose='User-requested native resolution detail montage',remote=remote,models='Frozen Nov9 seed117 five models',split='selection',scenes=['0022','0108'],depth=7,new_training=False,test_access=False,budget='1 GPU, 10 minutes'),indent=2))
with Path('docs/rgb-asm-nh3d-lut-plan.md').open('a',encoding='utf-8') as f:f.write(f'\n- 2026-11-11 세부 몽타주 요청: {r.name} 배타 생성. 기존 축소 패널 대신 동결11월9일 seed117 K1/K3/K6/CNN/blur로 선택0022/0108의7mm만 원해상도 추론하고 동일 ROI를 표시한다. 1GPU10분상한, 학습/시험/장비 접근 없음. 기존 코드게이트 및 동결 체크포인트 SHA 검증 후 실행. 새 rank 확장 실행과 독립이며 원본/가중치 수정 없음.\n')
print(str(r.resolve()))
