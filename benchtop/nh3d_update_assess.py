"""Assess model-update selection gains against repeated camera observations."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
from .nh3d_lut_model import sha,write_json
from .nh3d_lut_ab_report import repeatability
from .nh3d_lut_freeze import depth_decision

def assess(root):
    root=Path(root).resolve()
    plan=json.loads((root/'ab_plan.json').read_text())
    result=json.loads((root/'measured_report/results.json').read_text())
    if plan['split']!='selection' or len(plan['scenes'])!=8 or plan['repeats']!=2 or not plan.get('baseline_candidate'):
        raise ValueError('Require eight-scene old/new LUT selection with two orders')
    if result['status_sha256']!=sha(root/'ab/status.json') or result['scene_photos']!=384:
        raise ValueError('Incomplete or changed measurement')
    source=Path(plan['source_run']);protocol=json.loads((source/'collection_plan.json').read_text())
    if plan['baseline_sha256']!=protocol['baseline_candidate_sha256']:
        raise ValueError('Selection baseline differs from update center')
    groups=repeatability(result['pairs'])['groups']
    rows=[depth_decision(result['pairs'],groups,c,m,z,plan['scenes']) for c in 'RGB' for m in (c,'RGB') for z in plan['depths_mm']]
    physical_gate=all(r['exceeds_repeat_variation'] for r in rows)
    now=datetime.now(timezone(timedelta(hours=9)))
    minutes=(datetime.fromisoformat(protocol['deadline_kst'])-now).total_seconds()/60
    report=dict(created_kst=now.isoformat(),selection_run=str(root),source_run=str(source),candidate_sha256=plan['candidate_sha256'],baseline_sha256=plan['baseline_sha256'],results_sha256=sha(root/'measured_report/results.json'),depth_channels=rows,physical_gate=physical_gate,minutes_remaining=minutes,round2_eligible=physical_gate and minutes>=120,integrated_rgb=result['summary']['RGB/equal_mean'],criterion='All 18 channel/depth/slot-or-integrated mean MSE gains exceed same-scene paired-repeat SD; >=120 minutes remain for round2')
    path=root/'measured_report/update_decision.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,report);return report

def seal(source,candidate,baseline):
    from .nh3d_lut_ab import validate_candidate
    from .nh3d_lut_freeze import verify_seal
    source=Path(source).resolve();candidate=Path(candidate).resolve();baseline=Path(baseline).resolve()
    protocol=json.loads((source/'collection_plan.json').read_text())
    validate_candidate(source,candidate);validate_candidate(Path(protocol['base_bank']),baseline)
    manifest=json.loads((source/'manifest.json').read_text())
    patterns=[]
    for row in manifest['patterns']:
        if row['split']!='test':continue
        path=source/'patterns'/f'{row["id"]:04}.png'
        meta=json.loads(path.with_suffix('.json').read_text())
        patterns.append(dict(row,rgb_sha256=meta['rgb_sha256'],pattern_sha256=sha(path)))
    registry=source/'final_test_protocol.json'
    if registry.exists():raise FileExistsError(registry)
    spec=dict(version='model-update-final-v1',bank=str(source),bank_sha256=sha(source/'model_dataset.json'),manifest_sha256=sha(source/'manifest.json'),candidate=str(candidate),candidate_sha256=sha(candidate),baseline_candidate=str(baseline),baseline_sha256=sha(baseline),test_scenes=manifest['splits']['test'],depths_mm=[3.,5.,7.,9.],patterns=patterns,scene_photos=512,repeats=2,modes=['R','G','B','RGB'],scheduled_start_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),purpose='Frozen updated LUT versus the previously validated LUT on eight fresh held-out scenes')
    write_json(registry,spec);verify_seal(registry);return registry

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);print(json.dumps(assess(p.parse_args().run)))
