"""Fresh-camera A/B validation of a model-distilled RGB LUT on sealed test scenes."""
import argparse
import json
from pathlib import Path
import shutil
import torch
import numpy as np
from .dpac_lut_dataset import new_run
from .nh3d_lut_model import sha, write_json


def fixed_quality(amplitude,target,valid):
    from scipy.ndimage import gaussian_filter
    import cv2
    intensity=amplitude**2
    difference=intensity-target**2
    intensity_mse=float(np.mean(difference[valid]**2,dtype=np.float64))
    def display(a):
        v=np.clip(a*a,0,1)
        return np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)
    a,b=display(amplitude),display(target)
    display_mse=float(np.mean((a[valid]-b[valid])**2,dtype=np.float64))
    ma=gaussian_filter(a,1.5,truncate=3.5);mb=gaussian_filter(b,1.5,truncate=3.5)
    va=gaussian_filter(a*a,1.5,truncate=3.5)-ma*ma
    vb=gaussian_filter(b*b,1.5,truncate=3.5)-mb*mb
    cov=gaussian_filter(a*b,1.5,truncate=3.5)-ma*mb
    ssim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    inside=cv2.erode(valid.astype(np.uint8),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    if not inside.any():raise ValueError('Insufficient valid SSIM neighborhood')
    return dict(fixed_white_amplitude_mse=float(np.mean((amplitude[valid]-target[valid])**2,dtype=np.float64)),
        linear_intensity_psnr_db=float(-10*np.log10(max(intensity_mse,1e-15))),
        fixed_clipped_srgb_psnr_db=float(-10*np.log10(max(display_mse,1e-15))),
        fixed_clipped_srgb_ssim=float(np.mean(ssim[inside],dtype=np.float64)),
        mean_normalized_intensity=float(intensity[valid].mean()),valid_fraction=float(valid.mean()))


def validate_candidate(source,candidate):
    source=Path(source).resolve();candidate=Path(candidate).resolve()
    with np.load(candidate,allow_pickle=False) as f:
        table=f['table'].copy();metadata=json.loads(str(f['metadata']))
    if table.shape!=(3,256) or table.dtype!=np.uint8:
        raise ValueError('RGB uint8 LUT required')
    if not isinstance(metadata,list) or len(metadata)!=3:
        raise ValueError('A direct-search LUT is not a model-distilled artifact')
    for ch,tag in enumerate('RGB'):
        meta=metadata[ch]
        if meta['channel'] != tag:
            raise ValueError('LUT channel order mismatch')
        if meta.get('update_protocol') == 'model_update_v1':
            combined=Path(meta['combined_dataset_path'])
            if sha(combined)!=meta['combined_dataset_sha256']:raise ValueError('Replay provenance changed')
            replay=json.loads(combined.read_text())
            for entry in replay['banks']:
                if sha(Path(entry['path'])/'model_dataset.json')!=entry['sha256']:raise ValueError('Replay bank changed')
            checkpoint=Path(meta['checkpoint_path'])
            checks_path=Path(meta['forward_checks_path'])
        elif meta.get('pilot_protocol') == 'nov05-fit03-camera-png':
            provenance = Path(meta['provenance_path'])
            checkpoint = Path(meta['checkpoint_path'])
            checks_path = Path(meta['forward_checks_path'])
            lut_path = Path(meta['channel_lut_path'])
            if sha(source/'manifest.json') != meta['source_manifest_sha256']:
                raise ValueError('Pilot source patterns changed')
            if sha(source/'fit_03/status.json') != meta['source_status_sha256']:
                raise ValueError('Pilot camera source changed')
            if sha(provenance) != meta['dataset_sha256'] or sha(checks_path) != meta['forward_checks_sha256']:
                raise ValueError('Pilot training evidence changed')
            if sha(lut_path) != meta['channel_lut_sha256']:
                raise ValueError('Distilled channel file changed')
            with np.load(lut_path, allow_pickle=False) as original:
                if not np.array_equal(table[ch], original['table']):
                    raise ValueError('Packed LUT differs from its distilled channel')
        else:
            if meta['dataset_sha256'] != sha(source/'model_dataset.json'):
                raise ValueError('Camera dataset provenance mismatch')
            checkpoint = candidate.parent/tag/'best.pt'
            checks_path = candidate.parent/tag/'forward_checks.json'
        if sha(checkpoint) != meta['teacher_sha256']:
            raise ValueError('Teacher provenance mismatch')
        checks = json.loads(checks_path.read_text())
        if (not checks['accepted_for_distillation'] or checks['teacher_sha256'] != meta['teacher_sha256']
                or checks['dataset_sha256'] != meta['dataset_sha256']):
            raise ValueError('Teacher validation does not match this LUT')
        delta=table[ch].astype(int)-np.arange(256)
        if table[ch,0]!=0 or table[ch,-1]!=255 or np.any(np.diff(table[ch].astype(int))<0) or np.max(np.abs(delta))>12:
            raise ValueError('LUT violates the measured code-probe envelope')
    return table,metadata


def transfer_manifest(archive):
    archive = Path(archive).resolve()
    provenance = json.loads((archive/'source.json').read_text(encoding='utf-8'))
    image = (archive/provenance['stored_frame']).resolve()
    if not image.is_relative_to(archive) or sha(image) != provenance['frame_sha256']:
        raise ValueError('Transfer RGB source path or hash differs from its receipt')
    exports = provenance['exports']
    if not exports or any(export['sha256'] != provenance['frame_sha256'] for export in exports):
        raise ValueError('The archived exports must refer to the same RGB image')
    if any(export['camera'] != exports[0]['camera'] or export['ply'] != exports[0]['ply']
           for export in exports):
        raise ValueError('Expected one independent scene and camera pose')
    scene = 'mipnerf-pinhole'
    depths = (3., 7., 9.)
    return dict(version='v3.0-dpac', dataset=str(archive),
                sources={scene: image.relative_to(archive).as_posix()}, splits={'transfer': [scene]},
                patterns=[dict(id=index, scene=scene, split='transfer', method='DPAC', design_mm=z, iterations=0)
                          for index,z in enumerate(depths)])


def blue_timing_from_profile(path):
    profile = json.loads(Path(path).read_text(encoding='utf-8'))
    if profile.get('state') != 'timing_trials_complete' or profile.get('error'):
        raise ValueError('Completed blue timing diagnosis required')
    timing = profile.get('selected_timing')
    if timing not in ([175,12222],[85,12222],[85,12272]):
        raise ValueError('Unrecognized blue timing')
    repeats = [r for r in profile['trials'] if r['requested'] == timing]
    if len(repeats) != 2 or {r['repeat'] for r in repeats} != {0,1}:
        raise ValueError('Two distinct blue isolation trials required')
    if not all(r['accepted'] and r['routing']['accepted'] and abs(r['white_drift_fraction']) <= .03 for r in repeats):
        raise ValueError('Blue slot isolation or white drift gate failed')
    stable = profile['stability']
    if (not stable['accepted'] or stable['requested'] != timing or len(stable['means']) != 9
            or stable['elapsed_s'] < 119 or stable['max_fraction_from_first'] > .03):
        raise ValueError('Two-minute blue stability gate incomplete')
    return tuple(timing)


def run(source, candidate, repeats, *, pilot_selection=False, transfer_source=None, blue_profile=None, baseline_candidate=None, update_selection=False):
    source = Path(source).resolve()
    candidate = Path(candidate).resolve()
    table, metadata = validate_candidate(source, candidate)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    baseline_table=np.tile(np.arange(256,dtype=np.uint8),(3,1))
    baseline_label='identity'
    if baseline_candidate is not None:
        origin=json.loads((source/'collection_plan.json').read_text())
        baseline_table,_=validate_candidate(Path(origin['base_bank']),baseline_candidate)
        baseline_label='previous validated LUT'
    if update_selection and (baseline_candidate is None or repeats!=2):
        raise ValueError('Update selection requires old-LUT baseline and two repetitions')
    if transfer_source is not None:
        if pilot_selection or repeats < 2:
            raise ValueError('Transfer requires a separate, reversed-order A/B evaluation')
        manifest = transfer_manifest(transfer_source)
        split = 'transfer'
        depths = (3., 7., 9.)
        scenes = manifest['splits'][split]
        run_label = 'nh3d-distilled-lut-beamsplat-transfer'
    elif update_selection:
        split='selection';depths=(3.,7.,9.);scenes=manifest['splits']['selection']
        if len(scenes)!=8:raise ValueError('Eight independent selection scenes required')
        run_label='nh3d-model-update-selection-ab'
    elif pilot_selection:
        if repeats != 1:
            raise ValueError('The pilot plan specifies one paired observation per mode')
        split = 'selection'
        depths = (3., 7., 9.)
        scenes = manifest['splits'][split][:2]
        if scenes != ['0526', '0800']:
            raise ValueError('Pilot scenes differ from the recorded prospective plan')
        run_label = 'nh3d-distilled-lut-selection-ab'
    else:
        if repeats < 2:
            raise ValueError('Final A/B requires at least two reversed-order repetitions')
        split = 'test'
        depths = (3., 5., 7., 9.)
        scenes = manifest['splits'][split]
        if len(set(scenes)) != 8:
            raise ValueError('Expected eight sealed test scenes')
        run_label = 'nh3d-distilled-lut-test-ab'
    rows = [row for row in manifest['patterns'] if row['split'] == split and row['scene'] in scenes]
    expected = {(scene, depth) for scene in scenes for depth in depths}
    if len(rows) != len(expected) or {(row['scene'], row['design_mm']) for row in rows} != expected:
        raise ValueError('Scene/depth partition mismatch')
    blue_timing = blue_timing_from_profile(blue_profile) if blue_profile else (175,12222)
    root = new_run(Path('results'), run_label)
    if blue_profile:
        shutil.copyfile(blue_profile,root/'blue_timing_profile.json')
    (root/'patterns').mkdir();write_json(root/'manifest.json',manifest)
    if transfer_source is not None:
        from .rgb_study_patterns import build
        shutil.copyfile(Path(transfer_source)/'source.json',root/'transfer_source.json')
        build(root/'manifest.json',root/'patterns',0,None)
    else:
        for row in rows:
            for suffix in ('.png','.json'):
                path=source/'patterns'/f'{row["id"]:04}{suffix}'
                shutil.copy2(path,root/'patterns'/path.name)
    shutil.copy2(candidate,root/'rgb_lut_candidate.npz')
    if baseline_candidate:shutil.copy2(baseline_candidate,root/'baseline_lut.npz')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in ('nh3d_lut_ab.py','dpac_lut_session.py','dpac_lut_fit.py','hardware.py','stage.py','readybeam.py','nh3d_lut_freeze.py'):
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    write_json(root/'ab_plan.json', dict(source_run=str(source),
        source_dataset_sha256_by_channel=[m['dataset_sha256'] for m in metadata],
        candidate_source=str(candidate), candidate_sha256=sha(candidate), teacher_metadata=metadata,
        blue_timing=list(blue_timing),blue_profile_sha256=sha(blue_profile) if blue_profile else None,
        baseline_label=baseline_label,baseline_candidate=str(Path(baseline_candidate).resolve()) if baseline_candidate else None,baseline_sha256=sha(baseline_candidate) if baseline_candidate else None,
        repeats=repeats, expected_scene_photos=len(rows)*4*2*repeats, split=split, scenes=scenes, depths_mm=list(depths)))
    print(root.resolve(), flush=True)
    from .dpac_preflight import run as preflight
    readiness = preflight(root)['checks']
    if not all(check['ok'] for check in readiness.values()):
        raise RuntimeError('Fresh stationary preflight failed')
    stage = readiness['stage']['result']
    if (not stage['ready_for_motion'] or stage['upper_limit_mm'] > 124.001
            or not readiness['display']['result']['expected_present']):
        raise RuntimeError('Stage or HDMI readiness gate failed')
    from .dpac_lut_session import Session
    from .dpac_lut_fit import Fit
    from . import geometry_registration as geo
    from . import step01_dot_grid as grid
    from .hardware import Bench
    from .readybeam import ReadyBeam, exclusive
    from .stage import Stage
    from .rgb_roi_calibration import stats
    from .rgb_study_block import rectify
    from .dpac_lut_fit import apply_lut

    class AB(Fit):
        def __init__(self):
            Session.__init__(self,root,'ab')
            self.manifest=manifest
            self.current=table
            self.identity=baseline_table.copy()
            self.modes=dict(R=(175,22),G=(175,6222),B=blue_timing,RGB=(15000,22))
            self.report.update(version='nh3d-distilled-ab-v4',requested_modes=self.modes,comparisons=[],
                               candidate_sha256=sha(candidate),teacher_metadata=metadata,baseline_label=baseline_label,baseline_sha256=sha(baseline_candidate) if baseline_candidate else None)
            self.save()

        def measure(self,name,row,codes,target,tables,mode,cal,references,meta):
            self.b.set_timing(*self.modes[mode])
            self.b.show_codes(geo.pack(apply_lut(codes,tables)))
            raw=self.capture(name)
            roi=stats(raw,cal[mode]['mask'])
            measures={}
            for ch,(dark,white,valid) in references.items():
                tag='RGB'[ch]
                if roi[tag]['near_clip']>.0001:raise RuntimeError('Saturated A/B ROI')
                amplitude=np.sqrt(np.maximum(rectify(raw,ch,cal[mode]['H'])-dark,0)/np.maximum(white,1))
                measures[tag]=dict(fixed_quality(amplitude,target[...,ch],valid),near_clip=roi[tag]['near_clip'])
            record=dict(name=name,id=row['id'],scene=row['scene'],depth_mm=row['design_mm'],split=row['split'],
                        mode=mode,base_rgb_sha256=meta['rgb_sha256'],metrics=measures)
            self.report['records'][-1].update(record);self.save()
            return record

        def depth(self,z,cal):
            refs={mode:self.references(cal,mode) for mode in self.modes}
            depth_rows=[r for r in rows if r['design_mm']==z]
            self.report['state']=f'{split}_ab_{z:g}mm';self.save()
            self.laser.configure('Analog')
            try:
                for i,row in enumerate(depth_rows):
                    codes,target,meta=self.load(row)
                    for repeat in range(repeats):
                        order=['A','B'] if (i+repeat)%2==0 else ['B','A']
                        for mode in self.modes:
                            pair={}
                            for method in order:
                                name=f'{split}_z{z:g}_{row["id"]:04}_r{repeat}_{mode}_{method}'
                                pair[method]=self.measure(name,row,codes,target,
                                    self.identity if method=='A' else self.current,mode,cal,refs[mode],meta)
                            self.report['comparisons'].append(dict(scene=row['scene'],id=row['id'],depth_mm=z,
                                repeat=repeat,mode=mode,order=order,pair=pair))
                            self.save()
                    print(f'{split} A/B z{z:g}: {row["scene"]}, repeats={repeats}',flush=True)
                    if (i+1)%4==0 or i+1==len(depth_rows):self.drift(z,cal,f'{split}_block{i+1}')
            finally:self.laser.configure('Digital')

        def execute(self):
            frames={z:geo.cached_patterns(z,root/'calibration_patterns') for z in (0.,*depths)}
            with exclusive():
                self.laser=ReadyBeam()
                try:
                    self.report['standby']=self.laser.configure('Digital')
                    with Stage(interval=(90.780,101.000)) as self.stage:
                        start=self.stage.initial['position_mm'];self.report['stage_start']=self.stage.status()
                        try:
                            with Bench(dict(grid.C,settle_s=.25)) as self.b:
                                if (self.b.camera.Width.GetValue(),self.b.camera.Height.GetValue(),self.b.camera.OffsetX.GetValue(),self.b.camera.OffsetY.GetValue())!=(4096,3000,0,0):
                                    raise RuntimeError('Full sensor dimensions/offset gate')
                                self.calibrate(0.,frames[0.],route=True)
                                for z in depths:self.depth(z,self.calibrate(z,frames[z]))
                        finally:
                            self.report['standby_before_return']=self.laser.configure('Digital')
                            if self.stage.moves:self.stage.move(start)
                            self.report['stage_end']=self.stage.status();self.report['moves']=self.stage.moves
                    if len(self.report['comparisons'])!=len(rows)*4*repeats:raise RuntimeError('Incomplete A/B pairs')
                    self.report['state']={'selection':'pilot_selection_ab_complete','test':'sealed_test_ab_complete',
                                          'transfer':'beamsplat_transfer_ab_complete'}[split]
                except BaseException as exc:
                    self.report.update(state='interrupted',error=repr(exc));raise
                finally:
                    try:self.report['final_standby']=self.laser.configure('Digital')
                    finally:self.save()
    AB().execute()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_run',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--repeats',type=int)
    p.add_argument('--blue-profile',type=Path)
    p.add_argument('--baseline-candidate',type=Path)
    p.add_argument('--update-selection',action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--pilot-selection',action='store_true')
    mode.add_argument('--transfer-source',type=Path)
    a=p.parse_args()
    repeats=a.repeats if a.repeats is not None else (1 if a.pilot_selection else 2)
    run(a.source_run,a.candidate,repeats,pilot_selection=a.pilot_selection,transfer_source=a.transfer_source,blue_profile=a.blue_profile,baseline_candidate=a.baseline_candidate,update_selection=a.update_selection)
