"""Single-pass DPAC, integrated RGB acquisition with lossless raw preservation.

No scene rectification, quality scoring or model fitting during collection.
Completed blocks are gated by fresh white stability and can be resumed.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import torch  # Load before vendor SDK DLLs.
import cv2
import numpy as np

from .dpac_lut_session import Session, now, sha
from . import step01_dot_grid as grid
from . import geometry_registration as geo
from .hardware import Bench, encode_phase
from .optics import ASM
from .paper_model import dpac
from .readybeam import ReadyBeam, exclusive
from .stage import Stage
from .targets import prepare_image


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    until = time.monotonic() + 5
    while True:
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= until:
                raise
            time.sleep(.1)


def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def active_paths(root):
    pointer=root/'active_plan.json'
    if pointer.exists():
        active=json.loads(pointer.read_text())
        plan_path=root/active['plan']
        assert sha(plan_path)==active['plan_sha256']
        return plan_path,root/active['verification']
    return root/'collection_plan.json',root/'plan_verification.json'


def augment_amplitude(base, aug):
    kind=aug['type']
    if kind=='canonical':return base
    if kind=='horizontal_flip':return base[:,::-1,:].copy()
    if kind=='vertical_flip':return base[::-1,:,:].copy()
    if kind=='linear_intensity_gain':return base*np.sqrt(aug['gain'])
    h,w=base.shape[:2]
    if kind=='zoom_crop':
        nw,nh=int(np.ceil(w*aug['zoom'])),int(np.ceil(h*aug['zoom']))
        intensity=cv2.resize(base**2,(nw,nh),interpolation=cv2.INTER_LINEAR)
        x=int(round((nw-w)*(aug['offset_x']+1)/2));y=int(round((nh-h)*(aug['offset_y']+1)/2))
        return np.sqrt(np.clip(intensity[y:y+h,x:x+w],0,1))
    if kind=='rotation':
        angle=aug['angle_degrees']
        if angle==180:
            result=base[::-1,::-1,:].copy()
        else:
            theta=np.deg2rad(angle);c,s=abs(np.cos(theta)),abs(np.sin(theta))
            # Isotropic cover scale: every output pixel samples inside the input.
            scale=max(c+s*h/w,c+s*w/h)*1.002
            matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,scale)
            intensity=cv2.warpAffine(base**2,matrix,(w,h),flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            result=np.sqrt(np.clip(intensity,0,1))
        if aug.get('horizontal_flip',False):result=result[:,::-1,:].copy()
        return result
    raise ValueError(aug)


def revise_augmentation(root):
    root=Path(root).resolve();old_path,_=active_paths(root)
    old=json.loads(old_path.read_text());new=json.loads(json.dumps(old))
    done=set()
    for p in (root/'completed_blocks').glob('*.json'):
        done.update(r['id'] for r in json.loads(p.read_text())['records'])
    assert done and all(json.loads(p.read_text())['state'] in ('interrupted','complete') for p in root.glob('session_*/status.json'))
    assert not (root/'collection_plan_v2.json').exists()
    rng=np.random.default_rng(202611061406)
    changed=[]
    for z in (3.,7.,9.):
        rows=[r for r in new['records'] if r['split']=='fit' and r['design_mm']==z and
              r['augmentation']['type']!='canonical' and r['id'] not in done]
        for i,r in enumerate(rows):
            choice=i%10
            if choice<5:
                aug=dict(type='rotation',angle_degrees=[90,180,270,15,-15][choice],
                         framing='isotropic cover + centered crop; no blank padding')
            elif choice==5:aug=dict(type='horizontal_flip')
            elif choice==6:aug=dict(type='vertical_flip')
            elif choice==7:
                aug=dict(type='rotation',angle_degrees=30 if (i//10)%2==0 else -30,
                         horizontal_flip=True,framing='isotropic cover + centered crop; no blank padding')
            elif choice==8:
                aug=dict(type='zoom_crop',zoom=float(rng.uniform(1.05,1.25)),
                         offset_x=float(rng.uniform(-1,1)),offset_y=float(rng.uniform(-1,1)))
            else:aug=dict(type='linear_intensity_gain',gain=float((.6,.75,.9)[(i//10)%3]))
            if aug==r['augmentation']:continue
            changed.append(dict(id=r['id'],before=r['augmentation'],after=aug))
            r['augmentation']=aug
            key=json.dumps(dict(scene=r['scene'],depth_mm=z,augmentation=aug),sort_keys=True)
            r['pattern_key']=hashlib.sha256(key.encode()).hexdigest()[:24]
    new.update(revision=2,revised_kst=now(),revision_reason='User requested rotation and flips; only uncommitted fit augmentation rows changed',
               previous_plans=[dict(path=str(old_path.relative_to(root)),sha256=sha(old_path))])
    validate_plan(new)
    assert all(new['records'][i]==old['records'][i] for i in done)
    assert all(a==b for a,b in zip(old['records'],new['records']) if a['split']!='fit')
    # Check rectangular rotation framing on a constant field before using hardware.
    for angle in (90,180,270,15,-15,30,-30):
        a=augment_amplitude(np.ones((960,1728,3),np.float32),dict(type='rotation',angle_degrees=angle))
        assert a.shape==(960,1728,3) and np.all(a==1),angle
    sample=np.arange(6*10*3,dtype=np.float32).reshape(6,10,3)/180
    assert np.array_equal(augment_amplitude(sample,dict(type='horizontal_flip')),sample[:,::-1])
    assert np.array_equal(augment_amplitude(sample,dict(type='vertical_flip')),sample[::-1])
    assert np.array_equal(augment_amplitude(sample,dict(type='rotation',angle_degrees=180)),sample[::-1,::-1])
    snapshot=root/'source_snapshot_v2';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    shutil.copy2('configs/benchtop.json',snapshot/'benchtop.json')
    shutil.copy2('docs/rgb-asm-nh3d-lut-plan.md',snapshot/'plan_at_prepare.md')
    write_json(root/'collection_plan_v2.json',new)
    summary={str(z):dict(Counter(r['augmentation']['type'] for r in new['records'] if r['split']=='fit' and r['design_mm']==z)) for z in (3.,7.,9.)}
    write_json(root/'plan_verification_v2.json',dict(accepted=True,completed_rows_preserved=len(done),
               changed_count=len(changed),changes=changed,augmentation_counts=summary,
               tests='counts/splits/duplicate drives/monotone LUT/committed and held-out unchanged/rotation no padding/flip direction passed',
               source_snapshot_sha256={p.name:sha(p) for p in snapshot.iterdir()}))
    write_json(root/'active_plan.json',dict(plan='collection_plan_v2.json',verification='plan_verification_v2.json',
                                         plan_sha256=sha(root/'collection_plan_v2.json')))
    print('REVISED',len(changed),'PRESERVED',len(done),json.dumps(summary),flush=True)


def make_probe(base, scene, z, channel, bound):
    seed = int(hashlib.sha256(f'{scene}|{z}|{channel}|integrated-v1'.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    knots = np.array([0,32,64,96,128,160,192,224,255])
    identity = np.arange(256)
    for attempt in range(100):
        delta = rng.uniform(-bound, bound, 7)
        pair = []
        for sign in (1,-1):
            y = base[channel,knots].astype(float)
            y[1:-1] += sign*delta
            y = np.clip(y, np.maximum(0, knots-12), np.minimum(255, knots+12))
            y = np.maximum.accumulate(y)
            y[0], y[-1] = 0, 255
            table = base.copy()
            table[channel] = np.rint(np.interp(identity, knots, y)).astype(np.uint8)
            pair.append(table)
        if all(np.any(t[channel] != base[channel]) for t in pair) and not np.array_equal(*pair):
            return pair
    raise RuntimeError('Cannot form distinct bounded probes')


def validate_plan(plan):
    splits = plan['splits']
    sets = [set(v) for v in splits.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 800
    assert {k:len(v) for k,v in splits.items()} == dict(fit=742,selection=25,test=25,legacy_test_reserved=8)
    rows = plan['records']
    assert len(rows) == 8200 and [r['id'] for r in rows] == list(range(8200))
    assert Counter(r['split'] for r in rows) == dict(fit=7500,selection=300,test=400)
    assert Counter(r['design_mm'] for r in rows if r['split']=='fit') == {3.:2500,7.:2500,9.:2500}
    seen = set()
    for r in rows:
        assert r['scene'] in splits[r['split']]
        assert r['design_mm'] in ((3.,5.,7.,9.) if r['split']=='test' else (3.,7.,9.))
        t = np.asarray(r['drive_lut'], dtype=int)
        assert t.shape == (3,256) and ((t >= 0) & (t <= 255)).all()
        assert (t[:,0] == 0).all() and (t[:,-1] == 255).all()
        assert (np.diff(t, axis=1) >= 0).all() and (np.abs(t-np.arange(256)) <= 12).all()
        key = (r['scene'],r['design_mm'],json.dumps(r['augmentation'],sort_keys=True),digest_array(t.astype('uint8')))
        assert key not in seen, ('Duplicate planned drive',r['id'])
        seen.add(key)
    for z in (3.,7.,9.):
        assert {r['scene'] for r in rows if r['split']=='fit' and r['design_mm']==z} == set(splits['fit'])


def prepare(root, previous):
    root = Path(root).resolve()
    if (root/'collection_plan.json').exists():
        raise FileExistsError(root/'collection_plan.json')
    old = json.loads(Path(previous).read_text(encoding='utf8'))
    assert sha(old['baseline_candidate']) == old['baseline_sha256']
    with np.load(old['baseline_candidate'],allow_pickle=False) as f:
        baseline = f['table'].copy()
    identity = np.tile(np.arange(256,dtype=np.uint8),(3,1))
    rng = np.random.default_rng(202611061350)
    fit = sorted(old['splits']['fit'])
    ordering = list(rng.permutation(fit))
    rows = []

    def add(scene, split, z, aug, condition, table, **extra):
        key = json.dumps(dict(scene=scene,depth_mm=z,augmentation=aug),sort_keys=True)
        rows.append(dict(id=len(rows),scene=scene,split=split,design_mm=z,
                         method='DPAC',iterations=0,augmentation=aug,
                         pattern_key=hashlib.sha256(key.encode()).hexdigest()[:24],
                         condition=condition,drive_lut=table.tolist(),**extra))

    for zi,z in enumerate((3.,7.,9.,5.)):
        roles = ('fit','selection','test') if z != 5. else ('test',)
        for split in roles:
            scenes = fit if split=='fit' else sorted(old['splits'][split])
            probe_scenes = set((ordering*2)[(zi*247)%742:(zi*247)%742+371]) if split=='fit' else set(scenes)
            aug_scenes = list(rng.permutation(fit))[:274] if split=='fit' else []
            aug_by_scene = {}
            for j,scene in enumerate(aug_scenes):
                if j < 110:
                    aug = dict(type='zoom_crop',zoom=float(rng.uniform(1.05,1.25)),
                               offset_x=float(rng.uniform(-1,1)),offset_y=float(rng.uniform(-1,1)))
                elif j < 192:
                    aug = dict(type='horizontal_flip')
                else:
                    aug = dict(type='linear_intensity_gain',gain=float((.6,.75,.9)[j%3]))
                aug_by_scene[scene] = aug
            for i,scene in enumerate(scenes):
                group = [('identity',identity),('current_lut',baseline)]
                channel = (i+zi)%3
                bound = (4,8,12)[(i+2*zi)%3]
                if scene in probe_scenes:
                    plus,minus = make_probe(baseline,scene,z,channel,bound)
                    group += [('probe_plus',plus),('probe_minus',minus)]
                if i%2:
                    group.reverse()
                for condition,table in group:
                    add(scene,split,z,dict(type='canonical'),condition,table,
                        probe_channel='RGB'[channel] if condition.startswith('probe') else None,
                        probe_bound=bound if condition.startswith('probe') else None)
                if scene in aug_by_scene:
                    condition,table = ('current_lut',baseline) if i%2 else ('identity',identity)
                    add(scene,split,z,aug_by_scene[scene],condition,table)
    plan = dict(version='integrated-rgb-raw-v1',created_kst=now(),source_plan=str(Path(previous).resolve()),
                source_plan_sha256=sha(previous),dataset=old['dataset'],sources=old['sources'],splits=old['splits'],
                baseline_sha256=old['baseline_sha256'],baseline_candidate=old['baseline_candidate'],
                expected_counts=dict(fit=7500,selection=300,test=400),expected_total=8200,
                depths_mm=[3.,7.,9.,5.],timing=dict(exposure_us=15000,trigger_delay_us=22),
                storage='full RGB8 PNG + full uint16 BayerRG16 uncompressed NPZ; no scene rectification',
                block_size=50,pilot_size=12,records=rows)
    validate_plan(plan)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,snapshot/p.name)
    shutil.copy2('configs/benchtop.json',snapshot/'benchtop.json')
    shutil.copy2('docs/rgb-asm-nh3d-lut-plan.md',snapshot/'plan_at_prepare.md')
    for name in ('source_similarity_audit.json',):
        shutil.copy2(Path('results/20261106-133612_dpac-800-full-acquisition')/name,root/name)
    shutil.copy2(old['baseline_candidate'],root/'baseline_lut.npz')
    source_checks = []
    for scene,rec in old['sources'].items():
        path = Path(old['dataset'])/rec['path']
        if sha(path) != rec['sha256']:
            raise ValueError(f'Source modified: {scene}')
        source_checks.append(scene)
    write_json(root/'collection_plan.json',plan)
    write_json(root/'plan_verification.json',dict(accepted=True,source_hashes_checked=len(source_checks),
               counts=plan['expected_counts'],unique_patterns=len({r['pattern_key'] for r in rows}),
               source_snapshot_sha256={p.name:sha(p) for p in snapshot.iterdir()}))
    print('PREPARED',root,plan['expected_counts'],flush=True)


class Generator:
    def __init__(self, root, plan):
        self.root,self.plan = root,plan
        self.folder = root/'patterns';self.folder.mkdir(exist_ok=True)
        self.depth = None
        self.last = None
        self.source_cache = None

    @torch.no_grad()
    def load(self,row):
        key = row['pattern_key']
        if self.last and self.last[0] == key:
            return self.last[1],self.last[2]
        path = self.folder/(key+'.png');meta_path=path.with_suffix('.json')
        expected = dict(scene=row['scene'],depth_mm=row['design_mm'],augmentation=row['augmentation'])
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert meta['input']==expected and sha(path)==meta['png_sha256']
            rgb=cv2.imread(str(path))[...,::-1].copy()
            assert digest_array(rgb)==meta['rgb_sha256']
        else:
            if path.exists():raise FileExistsError(path)
            z=row['design_mm']
            if self.depth != z:
                self.models=[ASM(dict(grid.C,slm_shape=[960,1728],distance_m=z/1000,wavelength_m=w),'cuda') for w in grid.WAVELENGTHS]
                self.depth=z
            scene=row['scene'];source=self.plan['sources'][scene]
            if not self.source_cache or self.source_cache[0]!=scene:
                amp,info=prepare_image(Path(self.plan['dataset'])/source['path'],(960,1728))
                assert info['source_sha256']==source['sha256']
                self.source_cache=(scene,amp,info)
            _,base,info=self.source_cache
            amp=augment_amplitude(base,row['augmentation'])
            codes=[]
            for ch,model in enumerate(self.models):
                phase=dpac(model,torch.as_tensor(np.ascontiguousarray(amp[...,ch]),device='cuda'))
                codes.append(encode_phase(phase.cpu().numpy()))
            rgb=np.stack(codes,-1)
            grid.png(path,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            assert np.array_equal(cv2.imread(str(path))[...,::-1],rgb)
            meta=dict(input=expected,target=info,target_amplitude_sha256=digest_array(amp),
                      png_sha256=sha(path),rgb_sha256=digest_array(rgb),method='single-pass DPAC',iterations=0)
            write_json(meta_path,meta)
        self.last=(key,rgb,meta)
        return rgb,meta


class Collector(Session):
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.plan_path,verification_path=active_paths(self.root)
        self.plan=json.loads(self.plan_path.read_text())
        validate_plan(self.plan)
        for name,digest in json.loads(verification_path.read_text())['source_snapshot_sha256'].items():
            if name.endswith('.py') and sha(Path(__file__).parent/name)!=digest:
                raise RuntimeError(f'Collector source changed after prepare: {name}')
        self.blocks=self.root/'completed_blocks';self.blocks.mkdir(exist_ok=True)
        self.done={}
        accepted_plans={sha(self.plan_path):self.plan}
        for previous in self.plan.get('previous_plans',[]):
            path=self.root/previous['path']
            assert sha(path)==previous['sha256']
            accepted_plans[previous['sha256']]=json.loads(path.read_text())
        for p in sorted(self.blocks.glob('*.json')):
            block=json.loads(p.read_text())
            assert block['plan_sha256'] in accepted_plans
            for r in block['records']:
                assert r['id'] not in self.done
                assert accepted_plans[block['plan_sha256']]['records'][r['id']]==self.plan['records'][r['id']]
                self.done[r['id']]=r
        index=1
        while (self.root/f'session_{index:03}').exists():index+=1
        self.total_captures=0
        super().__init__(self.root,f'session_{index:03}')
        self.raw_dir=self.out/'raw';self.raw_dir.mkdir()
        self.modes={'RGB':(15000,22)}
        self.report.update(version='integrated-rgb-raw-v1',requested_modes=self.modes,
                           plan_sha256=sha(self.plan_path),plan_path=str(self.plan_path.relative_to(self.root)),resumed_completed=len(self.done))
        self.generator=Generator(self.root,self.plan)
        self.journal=(self.out/'captures.jsonl').open('x',encoding='utf8')
        self.scene_journal=(self.out/'scenes.jsonl').open('x',encoding='utf8')
        self.block_index=len(list(self.blocks.glob('*.json')))
        self.capture_seconds=[]
        self.plan_hash=sha(self.plan_path)
        self.save()

    def save(self):
        self.report['updated_kst']=now()
        snapshot={k:v for k,v in self.report.items() if k!='records'}
        snapshot['last_capture']=self.report['records'][-1] if self.report['records'] else None
        snapshot['session_capture_count']=self.total_captures
        snapshot['committed_scene_count']=len(self.done)
        snapshot['expected_scene_count']=8200
        write_json(self.out/'status.json',snapshot)

    def capture(self,name):
        t=time.perf_counter()
        raw=super().capture(name)
        path=self.raw_dir/(name+'.npz')
        record=self.report['records'][-1]
        with path.open('xb') as f:
            np.savez(f,raw_bayer=raw,frame_id=np.int64(record['frame_id']),
                     timestamp_ns=np.int64(record['timestamp_ns']))
            f.flush();os.fsync(f.fileno())
        record.update(raw_path=str(path.relative_to(self.root)),raw_sha256=sha(path),
                      raw_shape=list(raw.shape),raw_dtype=str(raw.dtype),pixel_format='BayerRG16',
                      capture_store_seconds=time.perf_counter()-t)
        self.journal.write(json.dumps(record)+'\n');self.journal.flush();os.fsync(self.journal.fileno())
        self.total_captures+=1
        # Older captures are durable in the append-only journal; avoid quadratic status writes.
        self.report['records'][:]=[record]
        return raw

    def routing(self,cal):
        ref=cal['RGB'];mask=ref['mask']
        chosen=cv2.resize(mask.astype('f'),(128,94),interpolation=cv2.INTER_AREA)>.999
        checker=np.uint8(64+128*(np.indices((960,1728)).sum(0)%2))
        tiles={}
        self.laser.configure('Analog')
        try:
            self.b.set_timing(*self.modes['RGB'])
            for i,key in enumerate(['flat','R','G','B','flat','B','R','G','flat','G','B','R','flat']):
                frame=self.flat.copy()
                if key!='flat':frame[grid.ROI][...,'RGB'.index(key)]=checker
                self.b.show_codes(frame);raw=self.capture(f'route_{i:02}_{key}')
                values=[]
                for samples in ((raw[::2,::2],),(raw[::2,1::2],raw[1::2,::2]),(raw[1::2,1::2],)):
                    values.append(np.mean([cv2.resize(a.astype('f'),(128,94),interpolation=cv2.INTER_AREA)[chosen] for a in samples],axis=0)/65535)
                tiles.setdefault(key,[]).append(np.stack(values))
            flat=np.mean(tiles['flat'],axis=0)
            noise=np.max([np.mean(np.abs(a-b),axis=1) for i,a in enumerate(tiles['flat']) for b in tiles['flat'][:i]],axis=0)
            limit=np.maximum(noise,1/4095)
            response={k:np.mean(np.abs(np.mean(tiles[k],axis=0)-flat),axis=1) for k in 'RGB'}
            accepted=all(response[k][i]>5*limit[i] for i,k in enumerate('RGB'))
            self.report['routing']['RGB']=dict(accepted=accepted,noise_ceiling=limit.tolist(),
                    mae={k:v.tolist() for k,v in response.items()},
                    signed_sensor_change_by_drive_slot=np.stack([(flat-np.mean(tiles[k],axis=0)).mean(axis=1) for k in 'RGB'],axis=1).tolist())
            self.save()
            if not accepted:raise RuntimeError('Integrated RGB routing check failed')
        finally:self.laser.configure('Digital')

    def channel_stats(self,raw,mask):
        result={}
        for tag,sites in [('R',[(0,0)]),('G',[(0,1),(1,0)]),('B',[(1,1)])]:
            values=[raw[y::2,x::2][mask[y::2,x::2]] for y,x in sites]
            n=sum(a.size for a in values)
            result[tag]=dict(mean=float(sum(a.sum(dtype=np.float64) for a in values)/n),
                             near_clip=float(sum(np.count_nonzero(a>=.98*65535) for a in values)/n))
        return result

    def drift(self,z,cal,label):
        self.b.set_timing(*self.modes['RGB']);self.b.show_codes(self.flat)
        raw=self.capture(f'z{z:g}_{label}_white')
        ref=cal['RGB'];dark=self.channel_stats(ref['dark'],ref['mask'])
        base=self.channel_stats(ref['white'],ref['mask']);current=self.channel_stats(raw,ref['mask'])
        changes={c:(current[c]['mean']-dark[c]['mean'])/(base[c]['mean']-dark[c]['mean'])-1 for c in 'RGB'}
        passed=all(abs(v)<=.03 for v in changes.values()) and all(v['near_clip']<=.0001 for v in current.values())
        entry=dict(depth_mm=z,label=label,time_kst=now(),changes=changes,accepted=passed,
                   camera_path=self.report['records'][-1]['path'],raw_path=self.report['records'][-1]['raw_path'])
        self.report.setdefault('white_drift',[]).append(entry);self.save()
        if not passed:raise RuntimeError(f'White stability/clip failed: {entry}')
        return entry

    def scene(self,row,cal):
        start=time.perf_counter()
        rgb,meta=self.generator.load(row)
        tables=np.asarray(row['drive_lut'],dtype=np.uint8)
        driven=np.stack([tables[c,rgb[...,c]] for c in range(3)],-1)
        generated=time.perf_counter()
        self.b.set_timing(*self.modes['RGB'])
        self.b.show_codes(geo.pack(driven))
        raw=self.capture(f'scene_{row["id"]:05}_{row["split"]}_z{row["design_mm"]:g}')
        roi=self.channel_stats(raw,cal['RGB']['mask'])
        if any(x['near_clip']>.0001 for x in roi.values()):
            raise RuntimeError(f'Saturated scene {row["id"]}: {roi}')
        rec=dict(self.report['records'][-1],id=row['id'],scene=row['scene'],split=row['split'],
                 depth_mm=row['design_mm'],condition=row['condition'],augmentation=row['augmentation'],
                 pattern_path=str((self.generator.folder/(row['pattern_key']+'.png')).relative_to(self.root)),
                 pattern_sha256=meta['png_sha256'],base_rgb_sha256=meta['rgb_sha256'],
                 drive_lut=row['drive_lut'],displayed_rgb_sha256=digest_array(driven),
                 source_sha256=self.plan['sources'][row['scene']]['sha256'],roi=roi,
                 generation_load_seconds=generated-start,play_capture_store_seconds=time.perf_counter()-generated,
                 total_seconds=time.perf_counter()-start,calibration_session=str(self.out.relative_to(self.root)))
        self.scene_journal.write(json.dumps(rec)+'\n');self.scene_journal.flush();os.fsync(self.scene_journal.fileno())
        return rec

    def commit(self,records,drift):
        self.block_index+=1
        path=self.blocks/f'block_{self.block_index:04}.json'
        if path.exists():raise FileExistsError(path)
        write_json(path,dict(plan_sha256=self.plan_hash,session=str(self.out.relative_to(self.root)),
                            committed_kst=now(),white_check=drift,records=records))
        for r in records:
            if r['id'] in self.done:raise RuntimeError('Duplicate committed scene')
            self.done[r['id']]=r
        self.save()
        print('BLOCK',self.block_index,'COMMITTED',len(self.done),'/8200','WHITE',drift['changes'],flush=True)

    def pilot_verify(self,records,wall):
        for rec in records:
            png=self.root/rec['path'];raw_path=self.root/rec['raw_path']
            assert sha(png)==rec['png_sha256'] and sha(raw_path)==rec['raw_sha256']
            image=cv2.imread(str(png));assert image.shape==(3000,4096,3) and image.dtype==np.uint8
            with np.load(raw_path,allow_pickle=False) as f:
                raw=f['raw_bayer'];assert raw.shape==(3000,4096) and raw.dtype==np.uint16
                assert int(f['frame_id'])==rec['frame_id'] and int(f['timestamp_ns'])==rec['timestamp_ns']
                assert np.array_equal(image,self.enc[cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)])
        times=[r['play_capture_store_seconds'] for r in records]
        result=dict(accepted=True,count=len(records),time_kst=now(),
                    play_capture_store_mean_s=float(np.mean(times)),play_capture_store_median_s=float(np.median(times)),
                    generation_inclusive_mean_s=float(np.mean([r['total_seconds'] for r in records])),
                    wall_with_white_check_s=wall,estimated_8200_hours=float(np.mean(times)*8200/3600),
                    note='Estimated hours excludes later unique DPAC generation, calibration, movement and white checks.',
                    records=[r['id'] for r in records])
        write_json(self.out/'pilot_timing.json',result)
        print('PILOT_PASSED',json.dumps(result),flush=True)

    def collect_depth(self,z,cal):
        rows=[r for r in self.plan['records'] if r['design_mm']==z and r['id'] not in self.done]
        self.report['state']=f'collecting_z{z:g}'
        self.laser.configure('Analog')
        try:
            cursor=0
            if not self.done:
                n=self.plan['pilot_size'];start=time.perf_counter()
                records=[self.scene(r,cal) for r in rows[:n]]
                drift=self.drift(z,cal,'pilot_end')
                self.pilot_verify(records,time.perf_counter()-start)
                self.commit(records,drift);cursor=n
            while cursor<len(rows):
                if (self.root/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED')
                if shutil.disk_usage(self.root).free < 30*1024**3:raise RuntimeError('Disk free space below 30 GiB')
                state=self.stage.status();self.stage.require_stationary(state)
                if abs(state['position_mm']-(grid.ZERO-z))>.002:raise RuntimeError('Stage depth changed')
                batch=rows[cursor:cursor+self.plan['block_size']]
                start=time.perf_counter()
                records=[self.scene(r,cal) for r in batch]
                drift=self.drift(z,cal,f'block{self.block_index+1:04}_end')
                self.commit(records,drift);cursor+=len(batch)
                print('BLOCK_SECONDS',round(time.perf_counter()-start,3),'LAST_ID',batch[-1]['id'],flush=True)
        finally:self.laser.configure('Digital')

    def run_bank(self):
        try:
            pre=json.loads((self.root/'preflight.json').read_text())
            assert all(v['ok'] for v in pre['checks'].values())
            assert pre['checks']['stage']['result']['ready_for_motion']
            assert pre['checks']['display']['result']['expected_present']
            assert pre['checks']['compute']['result']['cuda']
            if shutil.disk_usage(self.root).free < 400*1024**3:raise RuntimeError('Require 400 GiB free for raw bank')
            remaining=[z for z in self.plan['depths_mm'] if any(r['design_mm']==z and r['id'] not in self.done for r in self.plan['records'])]
            if not remaining:raise RuntimeError('All scenes already committed; no acquisition needed')
            frames={z:geo.cached_patterns(z,self.root/'calibration_patterns') for z in [0.,*remaining]}
            # Prepare only the first scene before opening hardware; later DPAC is generated on demand.
            self.generator.load(next(r for r in self.plan['records'] if r['id'] not in self.done))
            with exclusive():
                self.laser=ReadyBeam()
                try:
                    self.report['standby']=self.laser.configure('Digital')
                    with Stage(interval=(90.780,101.000)) as self.stage:
                        start=self.stage.initial['position_mm'];self.report['stage_start']=self.stage.status()
                        try:
                            with Bench(dict(grid.C,settle_s=.25)) as self.b:
                                if (self.b.camera.Width.GetValue(),self.b.camera.Height.GetValue(),self.b.camera.OffsetX.GetValue(),self.b.camera.OffsetY.GetValue())!=(4096,3000,0,0):
                                    raise RuntimeError('Full sensor geometry mismatch')
                                self.report['camera_original']=self.b.original
                                self.report['state']='calibrating_zero'
                                self.routing(self.calibrate(0.,frames[0.],route=False))
                                for z in remaining:
                                    self.collect_depth(z,self.calibrate(z,frames[z],route=False))
                        finally:
                            self.report['standby_before_return']=self.laser.configure('Digital')
                            if self.stage.moves:self.stage.move(start)
                            self.report['stage_end']=self.stage.status();self.report['moves']=self.stage.moves
                finally:
                    self.report['final_standby']=self.laser.configure('Digital')
            counts=Counter(r['split'] for r in self.done.values())
            if dict(counts)!=self.plan['expected_counts']:raise RuntimeError(f'Counts incomplete: {counts}')
            self.report['state']='complete'
            write_json(self.root/'dataset_raw.json',dict(version='integrated-rgb-raw-v1',complete=True,
                       source_type='physical_camera',completed_kst=now(),counts=dict(counts),
                       plan_sha256=self.plan_hash,preprocessing='deferred',
                       records=[self.done[i] for i in sorted(self.done)]))
            print('COMPLETE',dict(counts),flush=True)
        except BaseException as exc:
            self.report.update(state='interrupted',error=repr(exc),traceback=traceback.format_exc())
            print('INTERRUPTED',repr(exc),flush=True)
            raise
        finally:
            self.save();self.journal.close();self.scene_journal.close()
            with Path('docs/rgb-asm-nh3d-lut-plan.md').open('a',encoding='utf8') as f:
                f.write(f'\n\n- {now()} 통합 raw 취득 `{self.root.name}` / `{self.out.name}`: {self.report["state"]}, 백색 통과 완료 장면 {len(self.done)}/8200. 근거 `{self.out.relative_to(Path.cwd())}/status.json`, `captures.jsonl`, `scenes.jsonl`, `completed_blocks/`. 오류: {self.report.get("error","없음")}. 최종 standby: {self.report.get("final_standby")}; stage: {self.report.get("stage_end")}. 장면 후처리·학습은 미실행.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('root',type=Path);a.add_argument('previous',type=Path)
    a=sub.add_parser('acquire');a.add_argument('root',type=Path)
    a=sub.add_parser('revise-augmentation');a.add_argument('root',type=Path)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.root,args.previous)
    elif args.command=='revise-augmentation':revise_augmentation(args.root)
    else:Collector(args.root).run_bank()


if __name__=='__main__':main()
