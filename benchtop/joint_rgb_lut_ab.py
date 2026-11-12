"""Fresh physical validation of the frozen joint RGB LUT, no legacy defaults."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import torch
import cv2
import numpy as np
from .dpac_lut_session import Session, sha, now
from .dpac_lut_fit import Fit, apply_lut
from .nh3d_lut_ab import fixed_quality, blue_timing_from_profile
from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .rgb_study_block import rectify
from .rgb_roi_calibration import stats
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .stage import Stage


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def validate_joint(package):
    checks = json.loads((package/'checksums.json').read_text())
    for name, expected in checks.items():
        if sha(package/name) != expected:
            raise ValueError(f'Joint package changed: {name}')
    spec = importlib.util.spec_from_file_location('frozen_joint_apply', package/'apply_rgb_lut.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    table, meta = mod.load_candidate(package/'rgb_lut_candidate.npz')
    freeze = json.loads((package/'freeze.json').read_text())
    assert sha(package/'freeze.json') == meta['freeze_sha256']
    assert freeze['teacher_sha256'] == meta['teacher_sha256']
    assert meta['channel_order'] == 'RGB' and meta['active_xywh'] == [96,60,1728,960]
    assert meta['wavelengths_nm'] == [638,520,450]
    assert np.array_equal(table, np.load(package/'identity_start/best_table.npy'))
    with np.load(package/'rgb_lut_candidate.npz', allow_pickle=False) as data:
        identity, previous = data['identity'].copy(), data['previous'].copy()
    assert np.array_equal(identity, np.tile(np.arange(256,dtype=np.uint8),(3,1)))
    for t in (table,previous):
        assert t.dtype == np.uint8 and t.shape == (3,256)
        assert np.all(t[:,0]==0) and np.all(t[:,-1]==255)
        assert np.all(np.diff(t.astype(int),axis=1)>=0)
        assert np.max(np.abs(t.astype(int)-np.arange(256)))<=12
    return dict(A=identity,B=table,C=previous), meta, checks


def prepare(root, package, acquisition, blue_profile=None, integrated_only=False):
    tables, meta, checks = validate_joint(package)
    source = json.loads(acquisition.read_text())
    scenes = ['0801','0826','0851','0876']
    dataset = Path(source['dataset'])
    sources = {s:f'DIV2K_valid_HR/{s}.png' for s in scenes}
    hashes = {s:sha(dataset/p) for s,p in sources.items()}
    used_hashes = {v['sha256'] for v in source['sources'].values()}
    assert not set(hashes.values()) & used_hashes
    assert not set(scenes) & set(source['sources'])
    depths = [3.,7.,9.,5.]
    rows = [dict(id=i,scene=s,split='external',method='DPAC',design_mm=z,iterations=0)
            for i,(z,s) in enumerate((z,s) for z in depths for s in scenes)]
    manifest = dict(version='v3.0-dpac',dataset=str(dataset),sources=sources,
                    splits={'external':scenes},patterns=rows)
    write(root/'manifest.json',manifest)
    blue_timing=list(blue_timing_from_profile(blue_profile)) if blue_profile else [175,12222]
    if blue_profile:shutil.copy2(blue_profile,root/'blue_timing_profile.json')
    write(root/'ab_plan.json',dict(created_kst=now(), scenes=scenes, source_sha256=hashes,
        blue_timing=blue_timing,blue_profile_sha256=sha(blue_profile) if blue_profile else None,
        source_split_audit='No IDs or file hashes overlap any acquisition source',
        acquisition_plan=str(acquisition.resolve()), acquisition_sha256=sha(acquisition),
        package=str(package.resolve()), package_checksums=checks, teacher_metadata=meta,
        depths_mm=depths, repeats=2, modes=['RGB'] if integrated_only else ['R','G','B','RGB'],
        methods={'A':'identity','B':'joint RGB candidate','C':'previous LUT'},
        expected_scene_captures=96 if integrated_only else 384, metric='fixed white, common geometry; no image-dependent gain',
        raw='one full-frame RGB8 PNG plus BayerRG16 NPZ per capture'))
    shutil.copy2(package/'rgb_lut_candidate.npz',root/'rgb_lut_candidate.npz')
    snap=root/'source_snapshot';snap.mkdir()
    for name in ['joint_rgb_lut_ab.py','dpac_lut_session.py','dpac_lut_fit.py','nh3d_lut_ab.py',
                 'hardware.py','readybeam.py','stage.py','geometry_registration.py','rgb_study_patterns.py']:
        shutil.copy2(Path(__file__).parent/name,snap/name)
    from .rgb_study_patterns import build
    build(root/'manifest.json',root/'patterns',0,None)
    write(root/'prepared.json',dict(state='prepared',kst=now(),manifest_sha256=sha(root/'manifest.json'),
                                  plan_sha256=sha(root/'ab_plan.json')))


class PhysicalAB(Fit):
    def __init__(self,root):
        Session.__init__(self,root,'ab')
        self.plan=json.loads((root/'ab_plan.json').read_text())
        self.manifest=json.loads((root/'manifest.json').read_text())
        prepared=json.loads((root/'prepared.json').read_text())
        assert sha(root/'ab_plan.json')==prepared['plan_sha256']
        assert sha(root/'manifest.json')==prepared['manifest_sha256']
        self.tables,meta,_=validate_joint(Path(self.plan['package']))
        assert meta==self.plan['teacher_metadata']
        blue=self.plan.get('blue_timing',[175,12222])
        if self.plan.get('blue_profile_sha256'):
            assert sha(root/'blue_timing_profile.json')==self.plan['blue_profile_sha256']
            assert list(blue_timing_from_profile(root/'blue_timing_profile.json'))==blue
        self.modes=dict(R=(175,22),G=(175,6222),B=tuple(blue),RGB=(15000,22))
        self.modes={k:v for k,v in self.modes.items() if k in self.plan['modes']}
        self.report.update(version='joint-rgb-physical-ab-v1',requested_modes=self.modes,
                           comparisons=[],teacher_metadata=meta)
        self.save()

    def capture(self,name):
        raw=super().capture(name)
        path=self.camera/(name+'.npz')
        np.savez(path,raw=raw)
        self.report['records'][-1].update(raw_path=str(path.relative_to(self.root)),raw_sha256=sha(path))
        self.save()
        return raw

    def depth(self,z,cal):
        refs={m:self.references(cal,m) for m in self.modes}
        rows=[r for r in self.manifest['patterns'] if r['design_mm']==z]
        self.report['state']=f'comparing_{z:g}mm';self.save()
        self.laser.configure('Analog')
        try:
            for i,row in enumerate(rows):
                codes,target,meta=self.load(row)
                for repeat in range(2):
                    base=['A','B','C'];order=base[i%3:]+base[:i%3]
                    if repeat:order=order[::-1]
                    for mode in self.modes:
                        pair={}
                        for method in order:
                            name=f'z{z:g}_{row["scene"]}_r{repeat}_{mode}_{method}'
                            self.b.set_timing(*self.modes[mode])
                            drive=apply_lut(codes,self.tables[method])
                            self.b.show_codes(geo.pack(drive));raw=self.capture(name)
                            roi=stats(raw,cal[mode]['mask']);metrics={}
                            for ch,(dark,white,valid) in refs[mode].items():
                                tag='RGB'[ch]
                                if roi[tag]['near_clip']>.0001:raise RuntimeError(f'{name}: saturated {tag}')
                                amp=np.sqrt(np.maximum(rectify(raw,ch,cal[mode]['H'])-dark,0)/np.maximum(white,1))
                                metrics[tag]=fixed_quality(amp,target[...,ch],valid)
                            rec=dict(name=name,scene=row['scene'],depth_mm=z,repeat=repeat,mode=mode,
                                     method=method,base_rgb_sha256=meta['rgb_sha256'],
                                     drive_sha256=hashlib.sha256(drive.tobytes()).hexdigest(),metrics=metrics)
                            self.report['records'][-1].update(rec);pair[method]=rec;self.save()
                        self.report['comparisons'].append(dict(scene=row['scene'],depth_mm=z,repeat=repeat,
                                                              mode=mode,order=order,pair=pair));self.save()
                print(f'Completed scene {row["scene"]} at {z:g} mm',flush=True)
            self.drift(z,cal,'end')
        finally:self.laser.configure('Digital')

    def execute(self):
        depths=self.plan['depths_mm']
        frames={z:geo.cached_patterns(z,self.root/'calibration_patterns') for z in [0.,*depths]}
        with exclusive():
            self.laser=ReadyBeam()
            try:
                self.report['standby']=self.laser.configure('Digital')
                with Stage(interval=(90.780,101.000)) as self.stage:
                    start=self.stage.initial['position_mm'];self.report['stage_start']=self.stage.status()
                    try:
                        with Bench(dict(grid.C,settle_s=.25)) as self.b:
                            assert (self.b.camera.Width.GetValue(),self.b.camera.Height.GetValue(),self.b.camera.OffsetX.GetValue(),self.b.camera.OffsetY.GetValue())==(4096,3000,0,0)
                            self.calibrate(0.,frames[0.],route=True)
                            for z in depths:self.depth(z,self.calibrate(z,frames[z]))
                    finally:
                        try:self.report['standby_before_return']=self.laser.configure('Digital')
                        finally:
                            if self.stage.moves:self.stage.move(start)
                            self.report['stage_end']=self.stage.status();self.report['moves']=self.stage.moves
                assert len(self.report['comparisons'])==self.plan['expected_scene_captures']//len(self.plan['methods'])
                self.report['state']='physical_ab_complete'
            except BaseException as exc:
                self.report.update(state='interrupted',error=repr(exc));raise
            finally:
                try:self.report['final_standby']=self.laser.configure('Digital')
                except BaseException as exc:self.report['final_standby_error']=repr(exc);raise
                finally:self.save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','acquire']);p.add_argument('root',type=Path)
    p.add_argument('--package',type=Path);p.add_argument('--acquisition',type=Path);p.add_argument('--blue-profile',type=Path);p.add_argument('--integrated-only',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.package,a.acquisition,a.blue_profile,a.integrated_only)
    else:PhysicalAB(a.root).execute()
