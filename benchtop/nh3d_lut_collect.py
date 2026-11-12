"""Prepare/acquire a new NH3D camera bank with explicit linear-array opt-in.

Preparation does not control hardware. Acquisition saves one full-frame PNG
per shot plus the explicitly requested numerical training arrays.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch  # Must precede vendor DLL imports.
import cv2
import numpy as np
from .dpac_lut_dataset import new_run
from .nh3d_lut_model import DEPTHS, optics, probe_tables, write_json, sha


def prepare(source_manifest, fit_count):
    source_manifest=Path(source_manifest).resolve()
    old=json.loads(source_manifest.read_text(encoding='utf-8'))
    if old['version']!='v3.0-dpac':raise ValueError('Explicit current DPAC manifest required')
    if '20261103' in str(source_manifest) or '2026-11-03' in str(source_manifest):
        raise ValueError('Discarded November 3 data are prohibited')
    if fit_count not in (24,48):raise ValueError('Supported proposed fit counts are 24 or 48')
    spec=json.loads(json.dumps(old))
    if len(spec['splits']['fit'])!=24 or len(spec['splits']['selection'])!=8 or len(spec['splits']['test'])!=8:
        raise ValueError('Expected frozen 24/8/8 source split')
    if fit_count==48:
        dataset=Path(spec['dataset'])
        if not (dataset/'DIV2K_train_HR').is_dir():
            raise ValueError('Renderer extension requires an explicit 48-scene frozen split')
        reserved=set(sum(spec['splits'].values(),[]))
        pool=sorted(p for p in (dataset/'DIV2K_train_HR').glob('*.png') if p.stem not in reserved)
        selected=np.random.default_rng(51105).permutation(len(pool))[:24]
        if len(selected)!=24:raise ValueError('Insufficient additional originals')
        for i in selected:
            p=pool[int(i)];spec['sources'][p.stem]=p.relative_to(dataset).as_posix();spec['splits']['fit'].append(p.stem)
    spec['patterns']=[]
    for role in ('fit','selection','test'):
        for scene in spec['splits'][role]:
            for z in ((3.,5.,7.,9.) if role=='test' else DEPTHS):
                spec['patterns'].append(dict(id=len(spec['patterns']),scene=scene,split=role,method='DPAC',design_mm=z,iterations=0))
    root=new_run(Path('results'),'nh3d-camera-model-lut')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in ('nh3d_lut_collect.py','nh3d_lut_model.py','nh3d_lut_data.py','nh3d_lut_train.py',
                 'dpac_lut_session.py','dpac_lut_fit.py','rgb_study_patterns.py','hardware.py','stage.py'):
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    shutil.copy2('docs/rgb-asm-nh3d-lut-plan.md',snapshot/'plan_at_prepare.md')
    write_json(root/'manifest.json',spec)
    from .rgb_study_patterns import build
    build(root/'manifest.json',root/'patterns',0,None)
    write_json(root/'collection_plan.json',dict(version='nh3d-camera-v4',source_split_path=str(source_manifest),
        source_split_sha256=sha(source_manifest),optics={tag:optics(tag) for tag in 'RGB'},
        fit_scenes=fit_count,selection_scenes=8,held_out_scenes=8,depths_mm=list(DEPTHS),
        variants=['identity','probe_plus','probe_minus'],expected_capture_count=(fit_count+8)*3*3*3,
        storage='awaiting explicit --save-linear-training-arrays',status='patterns_prepared_only'))
    print(root.resolve(),flush=True)
    return root


def acquire(root,save_arrays):
    if not save_arrays:
        raise ValueError('Explicit linear training-array storage choice required before acquisition')
    from .dpac_lut_session import Session, now
    from .dpac_lut_fit import Fit
    from . import geometry_registration as geo
    from . import step01_dot_grid as grid
    from .hardware import Bench
    from .readybeam import ReadyBeam, exclusive
    from .stage import Stage
    from .rgb_roi_calibration import stats
    from .rgb_study_block import rectify
    root=Path(root).resolve()
    plan=json.loads((root/'collection_plan.json').read_text())

    class Collector(Session):
        load=Fit.load
        references=Fit.references
        drift=Fit.drift

        def __init__(self):
            super().__init__(root,'model_acquisition')
            self.modes={k:tuple(v) for k,v in plan.get('requested_modes',dict(R=(175,22),G=(175,6222),B=(175,12222),RGB=(15000,22))).items()}
            self.manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            self.report.update(version='nh3d-camera-v4',requested_modes=self.modes,storage='full PNG plus requested linear arrays')
            self.arrays=self.out/'linear';self.arrays.mkdir()
            self.dataset=dict(version='nh3d-camera-v4',source_type='physical_camera',complete=False,created_kst=now(),
                optics=plan['optics'],expected_capture_count=plan['expected_capture_count'],records=[],
                split_manifest_sha256=sha(root/'manifest.json'),plan_sha256=sha(root/'collection_plan.json'),
                references=[],source_sha256={p.name:sha(p) for p in (root/'source_snapshot').glob('*')},
                normalization='sqrt(max(rectified Bayer-dark,0)/fixed smoothed white); no per-image brightness fit')
            self.dataset['baseline_variant']=plan.get('baseline_variant','identity')
            if 'update_protocol' in plan:
                self.dataset['update_protocol']=plan['update_protocol']
                self.dataset['drive_tables_sha256']=sha(root/'drive_tables.json')
            self.publish()

        def publish(self):
            self.save()
            self.dataset['session_status_sha256']=sha(self.out/'status.json')
            write_json(root/'model_dataset.json',self.dataset)

        def capture_depth(self,z,cal):
            self.report['state']=f'collecting_model_data_{z:g}mm';self.save()
            refs={tag:self.references(cal,tag)['RGB'.index(tag)] for tag in 'RGB'}
            for tag,(dark,white,valid) in refs.items():
                reference=self.arrays/f'reference_z{z:g}_{tag}.npz'
                np.savez_compressed(reference,dark=dark,white=white,valid=valid,H=cal[tag]['H'])
                self.dataset['references'].append(dict(path=str(reference.relative_to(root)),sha256=sha(reference),depth_mm=z,channel=tag))
            rows=[r for r in self.manifest['patterns'] if r['split'] in ('fit','selection') and r['design_mm']==z]
            self.laser.configure('Analog')
            try:
                for index,row in enumerate(rows):
                    codes,_,meta=self.load(row)
                    tables=probe_tables(row['scene'],z)
                    if 'update_protocol' in plan:
                        tables={k:np.asarray(v,dtype=np.uint8) for k,v in json.loads((root/'drive_tables.json').read_text())[str(row['id'])].items()}
                    # Alternate order to reduce a fixed time-order bias.
                    order=list(tables) if index%2==0 else list(reversed(tables))
                    for variant in order:
                        table=tables[variant]
                        driven=table[codes] if table.ndim==1 else np.stack([table[c,codes[...,c]] for c in range(3)],axis=-1)
                        self.b.show_codes(geo.pack(driven))
                        for ch,tag in enumerate('RGB'):
                            self.b.set_timing(*self.modes[tag])
                            name=f'model_z{z:g}_{row["id"]:04}_{variant}_{tag}'
                            raw=self.capture(name)
                            if stats(raw,cal[tag]['mask'])[tag]['near_clip']>.0001:
                                raise RuntimeError('Saturated training capture')
                            dark,white,valid=refs[tag]
                            amplitude=np.sqrt(np.maximum(rectify(raw,ch,cal[tag]['H'])-dark,0)/np.maximum(white,1)).astype(np.float32)
                            if not np.isfinite(amplitude).all():raise RuntimeError('Non-finite measurement')
                            path=self.arrays/(name+'.npz')
                            if path.exists():raise FileExistsError(path)
                            np.savez_compressed(path,amplitude=amplitude,valid=valid)
                            camera=self.report['records'][-1]
                            pattern=root/'patterns'/f'{row["id"]:04}.png'
                            source=Path(self.manifest['dataset'])/self.manifest['sources'][row['scene']]
                            self.dataset['records'].append(dict(name=name,scene=row['scene'],split=row['split'],
                                channel=tag,depth_mm=z,design_mm=row['design_mm'],variant=variant,drive_lut=(table if table.ndim==1 else table[ch]).tolist(),
                                displayed_channel_sha256=hashlib.sha256(driven[...,ch].tobytes()).hexdigest(),
                                array_path=str(path.relative_to(root)),array_sha256=sha(path),
                                camera_path=camera['path'],camera_sha256=camera['png_sha256'],
                                pattern_path=str(pattern.relative_to(root)),pattern_sha256=sha(pattern),
                                target_source=str(source.resolve()),target_sha256=meta['target']['source_sha256'],
                                timing=camera['timing'],frame_id=camera['frame_id'],timestamp_ns=camera['timestamp_ns']))
                            self.publish()
                    print(f'Acquired z{z:g} {row["split"]} {row["scene"]}: 9 linear pairs',flush=True)
                    if (index+1)%8==0 or index+1==len(rows):
                        self.drift(z,cal,f'block{index+1:03}')
            finally:self.laser.configure('Digital')

        def run_bank(self):
            frames={z:geo.cached_patterns(z,root/'calibration_patterns') for z in (0.,*DEPTHS)}
            with exclusive():
                self.laser=ReadyBeam()
                try:
                    self.report['standby']=self.laser.configure('Digital')
                    with Stage(interval=(90.780,101.000)) as self.stage:
                        start=self.stage.initial['position_mm']
                        self.report['stage_start']=self.stage.status()
                        try:
                            with Bench(dict(grid.C,settle_s=.25)) as self.b:
                                if (self.b.camera.Width.GetValue(),self.b.camera.Height.GetValue(),self.b.camera.OffsetX.GetValue(),self.b.camera.OffsetY.GetValue())!=(4096,3000,0,0):
                                    raise RuntimeError('Full sensor dimensions/offset gate')
                                self.report['camera_original']=self.b.original
                                self.calibrate(0.,frames[0.],route=True)
                                for z in DEPTHS:
                                    self.capture_depth(z,self.calibrate(z,frames[z]))
                                if len(self.dataset['records'])!=self.dataset['expected_capture_count']:
                                    raise RuntimeError('Incomplete collection')
                        finally:
                            self.report['standby_before_return']=self.laser.configure('Digital')
                            if self.stage.moves:self.stage.move(start)
                            self.report['stage_end']=self.stage.status();self.report['moves']=self.stage.moves
                    self.report['state']='camera_training_bank_complete'
                    self.dataset['complete']=True
                except BaseException as exc:
                    self.report.update(state='interrupted',error=repr(exc))
                    raise
                finally:
                    try:self.report['final_standby']=self.laser.configure('Digital')
                    finally:self.publish()
    Collector().run_bank()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('prepare');make.add_argument('source_manifest',type=Path);make.add_argument('--fit-count',type=int,choices=[24,48],default=24)
    take=sub.add_parser('acquire');take.add_argument('root',type=Path);take.add_argument('--save-linear-training-arrays',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source_manifest,a.fit_count)
    else:acquire(a.root,a.save_linear_training_arrays)

if __name__=='__main__':main()
