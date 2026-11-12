"""Check the premeasurement seal and decide the tested RGB LUT by depth."""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from .nh3d_lut_ab import validate_candidate
from .nh3d_lut_ab_report import repeatability
from .nh3d_lut_model import sha, write_json


def read(path):
    return Path(path).read_text(encoding='utf-8')


def verify_seal(registry):
    spec = json.loads(read(registry))
    bank = Path(spec['bank']).resolve()
    candidate = Path(spec['candidate']).resolve()
    if sha(bank/'model_dataset.json') != spec['bank_sha256'] or sha(bank/'manifest.json') != spec['manifest_sha256']:
        raise ValueError('Training bank or split manifest changed after sealing')
    if sha(candidate) != spec['candidate_sha256']:
        raise ValueError('Frozen candidate changed')
    validate_candidate(bank, candidate)
    if spec.get('baseline_candidate') and sha(spec['baseline_candidate']) != spec['baseline_sha256']:
        raise ValueError('Frozen baseline changed')
    manifest = json.loads(read(bank/'manifest.json'))
    training = json.loads(read(bank/'model_dataset.json'))
    scenes = set(spec['test_scenes'])
    if len(scenes) != 8 or scenes != set(manifest['splits']['test']):
        raise ValueError('Sealed test scenes differ')
    if any(row['scene'] in scenes or row['depth_mm'] == 5 for row in training['records']):
        raise ValueError('Held-out scene or 5 mm in training/selection')
    expected = {(scene,z) for scene in scenes for z in (3.,5.,7.,9.)}
    rows = spec['patterns']
    if len(rows) != 32 or {(r['scene'],r['design_mm']) for r in rows} != expected:
        raise ValueError('Incomplete sealed pattern coverage')
    for row in rows:
        pattern = bank/'patterns'/f'{row["id"]:04}.png'
        metadata = json.loads(read(pattern.with_suffix('.json')))
        recipe = metadata['recipe']
        if (recipe['split'] != 'test' or recipe['method'] != 'DPAC' or recipe['iterations'] != 0
                or (recipe['scene'],recipe['design_mm']) != (row['scene'],row['design_mm'])
                or metadata['rgb_sha256'] != row['rgb_sha256'] or sha(pattern) != row['pattern_sha256']):
            raise ValueError('Sealed test pattern mismatch')
    if spec['scene_photos'] != 512 or spec['repeats'] != 2 or spec['modes'] != ['R','G','B','RGB']:
        raise ValueError('Sealed A/B protocol differs')
    return spec


def depth_decision(pairs, repeats, tag, mode, depth, scenes):
    cells = [r for r in repeats if r['component'] == tag and r['mode'] == mode and r['depth_mm'] == depth]
    observations = [r for r in pairs if r['mode'] == mode and r['depth_mm'] == depth]
    if len(cells) != 8 or {r['scene'] for r in cells} != set(scenes) or len(observations) != 16:
        raise ValueError('Incomplete per-depth scene or repeat count')
    values = {method:np.asarray([r['images'][method]['metrics'][tag]['fixed_white_amplitude_mse']
                                 for r in observations]) for method in ('A','B')}
    mean_delta = float(np.mean([r['mean_improvement'] for r in cells]))
    pooled_sd = float(np.sqrt(np.mean([r['sample_sd']**2 for r in cells])))
    assert np.isclose(mean_delta,(values['A']-values['B']).mean())
    return dict(channel=tag,mode=mode,depth_mm=depth,mse_A=float(values['A'].mean()),mse_B=float(values['B'].mean()),
                mean_improvement=mean_delta,pooled_within_scene_repeat_sd=pooled_sd,
                exceeds_repeat_variation=bool(mean_delta>pooled_sd),worsened_pairs=int(np.sum(values['B']>values['A'])),
                worsened_scenes=[r['scene'] for r in cells if r['mean_improvement'] < 0])


def assess(registry, run):
    spec = verify_seal(registry)
    run = Path(run).resolve()
    plan = json.loads(read(run/'ab_plan.json'))
    state = json.loads(read(run/'ab/status.json'))
    results = json.loads(read(run/'measured_report/results.json'))
    if (plan['split'] != 'test' or plan['scenes'] != spec['test_scenes'] or plan['depths_mm'] != spec['depths_mm']
            or plan['candidate_sha256'] != spec['candidate_sha256'] or sha(run/'rgb_lut_candidate.npz') != spec['candidate_sha256']
            or results['status_sha256'] != sha(run/'ab/status.json') or results['scene_photos'] != 512
            or state['state'] != 'sealed_test_ab_complete' or state.get('error')):
        raise ValueError('Report does not describe the registered completed test')
    if datetime.fromisoformat(state['started_kst']) < datetime.fromisoformat(spec['scheduled_start_kst']):
        raise ValueError('Test preceded its scheduled acquisition window')
    if spec.get('baseline_candidate') and (plan.get('baseline_sha256') != spec['baseline_sha256'] or sha(run/'baseline_lut.npz') != spec['baseline_sha256']):
        raise ValueError('Test used a different baseline LUT')
    final = state['final_standby']
    if (final['mode'] != 'Digital' or final['enable_BGR'] != ['Off']*3 or final['percent_BGR'] != [0.]*3
            or state['stage_end']['moving'] or abs(state['stage_end']['position_mm']-state['stage_start']['position_mm']) > .002):
        raise ValueError('Final equipment state requires attention')
    if len(state['calibrations']) != 20 or any(not c['registration']['accepted'] for c in state['calibrations']):
        raise ValueError('Expected 20 fresh accepted calibrations')
    expected_drifts = {(z,mode,tag,block) for z in spec['depths_mm'] for mode in spec['modes']
                       for tag in mode for block in ('test_block4','test_block8')}
    measured_drifts = {(r['depth_mm'],r['mode'],r['channel'],r['phase']) for r in state['white_drift']}
    if (expected_drifts != measured_drifts or len(state['white_drift']) != len(expected_drifts)
            or max(abs(r['fraction']) for r in state['white_drift']) > .03):
        raise ValueError('Missing or failed end-of-block white measurements')
    records = {r['name']:r for r in state['records']}
    compared_paths = set()
    expected_patterns = {r['id']:r for r in spec['patterns']}
    for observation in state['comparisons']:
        frozen = expected_patterns[observation['id']]
        if (observation['scene'],observation['depth_mm']) != (frozen['scene'],frozen['design_mm']):
            raise ValueError('Unregistered scene or depth in comparison')
        times = []
        for method in observation['order']:
            evidence = observation['pair'][method]
            rec = records[evidence['name']]
            path = run/rec['path']
            if (evidence['base_rgb_sha256'] != frozen['rgb_sha256'] or path in compared_paths
                    or sha(path) != rec['png_sha256']):
                raise ValueError('Code or unique camera exposure mismatch')
            with path.open('rb') as f:
                header = f.read(29)
            if (header[:8] != bytes.fromhex('89504e470d0a1a0a') or header[12:16] != b'IHDR'
                    or int.from_bytes(header[16:20],'big') != 4096 or int.from_bytes(header[20:24],'big') != 3000
                    or header[24:26] != bytes([8,2])):
                raise ValueError('Expected native full-frame 8-bit RGB PNG')
            compared_paths.add(path)
            times.append(datetime.fromisoformat(rec['time_kst']))
        if len(times) != 2 or times[1] <= times[0]:
            raise ValueError('Recorded A/B order and capture timestamps disagree')
    if len(compared_paths) != 512:
        raise ValueError('Incomplete camera image set')
    pairs = results['pairs']
    groups = repeatability(pairs)['groups']
    rows = [depth_decision(pairs,groups,c,mode,z,spec['test_scenes'])
            for c in 'RGB' for mode in (c,'RGB') for z in spec['depths_mm']]
    unchanged=set()
    if spec.get('baseline_candidate'):
        with np.load(spec['baseline_candidate'],allow_pickle=False) as a,np.load(spec['candidate'],allow_pickle=False) as b:
            unchanged={tag for i,tag in enumerate('RGB') if np.array_equal(a['table'][i],b['table'][i])}
    unchanged_retention={tag:all(r['mean_improvement'] >= -r['pooled_within_scene_repeat_sd'] for r in rows if r['channel']==tag) for tag in unchanged}
    channel_default = {tag:(('previous validated LUT unchanged' if unchanged_retention[tag] else 'unchanged drive but measured cross-channel regression; keep previous whole table') if tag in unchanged else
                           ('model LUT' if all(r['exceeds_repeat_variation'] for r in rows if r['channel']==tag)
                           else ('previous validated LUT recommended; mixed-table verification required' if spec.get('baseline_candidate') else 'identity recommended; mixed-table verification required')))
                       for tag in 'RGB'}
    outcome = dict(status='validated as tested' if all(v in ('model LUT','previous validated LUT unchanged') for v in channel_default.values())
                   else 'partial channel support',unchanged_channels=sorted(unchanged),unchanged_channel_retention=unchanged_retention,candidate_sha256=spec['candidate_sha256'],
                   baseline_label=plan.get('baseline_label','identity'),baseline_sha256=plan.get('baseline_sha256'),registered_protocol_sha256=sha(registry),run=str(run),scene_photos=512,
                   independent_test_scenes=8,depths_mm=spec['depths_mm'],repeats=2,
                   max_abs_white_drift=results['max_abs_white_drift'],results_sha256=sha(run/'measured_report/results.json'),
                   integrated_rgb=results['summary']['RGB/equal_mean'],depth_channels=rows,channel_default=channel_default,
                   created_kst=datetime.now(timezone(timedelta(hours=9))).isoformat())
    dest = run/'measured_report/freeze_decision.json'
    if dest.exists():
        raise FileExistsError(dest)
    write_json(dest,outcome)
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry',type=Path)
    parser.add_argument('--test-run',type=Path)
    args = parser.parse_args()
    if args.test_run:
        decision = assess(args.registry,args.test_run)
        print(decision['status'],decision['channel_default'])
    else:
        checked = verify_seal(args.registry)
        print('Verified',len(checked['patterns']),'DPAC patterns and LUT',checked['candidate_sha256'])
