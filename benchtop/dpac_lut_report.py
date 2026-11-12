"""Package measured DPAC LUT results without creating extra camera images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def package(folder):
    status_file = folder/'status.json'
    status = json.loads(status_file.read_text(encoding='utf8'))
    if status['state'] != 'candidate_and_selection_complete' or status.get('error'):
        raise ValueError('Incomplete acquisition; do not package it as a selected LUT')
    root = Path(status['root'])
    source = root/status['lut']['path']
    if digest(source) != status['lut']['sha256']:
        raise ValueError('Candidate checksum mismatch')
    out = folder/'handoff'
    out.mkdir(exist_ok=False)
    rows = []
    for comparison in status['selection']:
        for channel, a in comparison['pair']['A']['metrics'].items():
            b = comparison['pair']['B']['metrics'][channel]
            av, bv = a['fixed_white_amplitude_mse'], b['fixed_white_amplitude_mse']
            rows.append(dict(depth_mm=comparison['depth_mm'], scene=comparison['scene'],
                id=comparison['id'], mode=comparison['mode'], sensor_channel=channel,
                A_mse=av, B_mse=bv, delta=bv-av, improvement_percent=100*(av-bv)/av,
                A_mean_intensity=a['mean_normalized_intensity'], B_mean_intensity=b['mean_normalized_intensity'],
                A_png='../selection_camera/'+comparison['pair']['A']['name']+'.png',
                B_png='../selection_camera/'+comparison['pair']['B']['name']+'.png'))
    write_csv(out/'selection_ab.csv', rows)
    fit_rows = [{k:v for k,v in r.items() if k != 'order'} for r in status['comparisons']]
    write_csv(out/'fit_candidates_ab.csv', fit_rows)
    groups = []
    for mode in ('R', 'G', 'B', 'RGB'):
        for channel in ('RGB' if mode == 'RGB' else mode):
            for depth in (3., 7., 9., 'all'):
                subset = [r for r in rows if r['mode'] == mode and r['sensor_channel'] == channel
                          and (depth == 'all' or r['depth_mm'] == depth)]
                expected = 24 if depth == 'all' else 8
                if len(subset) != expected:
                    raise ValueError('Incomplete independent selection')
                a = float(np.mean([r['A_mse'] for r in subset]))
                b = float(np.mean([r['B_mse'] for r in subset]))
                groups.append(dict(mode=mode, sensor_channel=channel, depth_mm=depth,
                    scene_depth_pairs=len(subset), A_mse=a, B_mse=b,
                    improvement_percent=100*(a-b)/a,
                    improved_pairs=sum(r['delta'] < 0 for r in subset),
                    worsened_pairs=sum(r['delta'] > 0 for r in subset)))
    write_csv(out/'selection_summary.csv', groups)
    decisions = {ch:dict(value, validation='Independent selection measured; final repeated validation pending')
                 for ch,value in status['decisions'].items()}
    summary = dict(state=status['state'], candidate_decisions=decisions,
        fit_original_scenes=sorted({r['scene'] for r in status['comparisons']}),
        fit_paired_comparisons=len(status['comparisons']),
        selection_original_scenes=sorted({r['scene'] for r in status['selection']}),
        selection_ab_pairs=len(status['selection']), groups=groups,
        white_drift=status['white_drift'],
        limitations=['One candidate search and one independent selection pass; repeated final validation pending.',
                     'DIV2K only; Gaussian Splatting renderer transfer unverified.',
                     '5 mm and test scenes unmeasured.',
                     '100.780 mm optical zero remains a provisional reference; ND 0.4 is the specified installation condition.'],
        final_stage=status['stage_end'], final_laser=status['final_standby'],
        evidence_status_sha256=digest(status_file))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    with np.load(source, allow_pickle=False) as data:
        table = data['table'].copy()
        identity = data['identity'].copy()
        meta = json.loads(str(data['metadata']))
    meta.update(validation='Independent selection captured; final repeated validation pending',
        selection_summary='selection_summary.csv', slm_pitch_m=6.4e-6,
        relay_magnification=1., fourier_aperture=.5,
        camera_serial='25222112', slm_screen='LETO-3-0088', laser_serial='23', stage_serial='45897069',
        requested_capture_timing_us=status['requested_modes'],
        measured_calibration=[dict(depth_mm=r['depth_mm'], mode=r['mode'], timing=r['timing'],
                                   geometry=r['registration']['scores']['pooled']) for r in status['calibrations']],
        routing_source=status['routing_source'], routing_sha256=status['routing_sha256'],
        evidence_status_sha256=digest(status_file),
        table_bytes_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        channel_checksums={ch:hashlib.sha256(table[i].tobytes()).hexdigest() for i,ch in enumerate('RGB')})
    for tag in meta['candidate_tables']:
        meta['candidate_tables'][tag]['validation'] = 'Independent selection measured; consult selection_summary.csv'
    np.savez(out/'rgb_lut_candidate.npz', table=table, identity=identity,
             metadata=json.dumps(meta, ensure_ascii=False))
    meta['npz_sha256'] = digest(out/'rgb_lut_candidate.npz')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    (out/'apply_lut.py').write_text('''import numpy as np

def apply_rgb_lut(in_codes, npz_path, bypass=False):
    if in_codes.dtype != np.uint8 or in_codes.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 input codes")
    if bypass:
        return in_codes.copy()
    with np.load(npz_path, allow_pickle=False) as data:
        table = data["table"]
    if table.dtype != np.uint8 or table.shape != (3, 256):
        raise ValueError("Invalid RGB table")
    return np.stack([table[c][in_codes[..., c]] for c in range(3)], axis=-1)
''', encoding='utf8')
    checksums = {p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'sha256.json').write_text(json.dumps(checksums, indent=2), encoding='utf8')
    print(json.dumps([g for g in groups if g['depth_mm'] == 'all'], indent=2))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    package(p.parse_args().folder)
