"""One-pass, paired camera search of seven monotone code LUTs per channel."""
import argparse
import json
from pathlib import Path

import torch
import cv2
import numpy as np

from .dpac_lut_session import Session, sha, now
from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .rgb_roi_calibration import stats
from .rgb_study_block import rectify
from .stage import Stage
from .targets import prepare_image


def candidates():
    x = np.array([0, 64, 128, 192, 255])
    result = {'identity': np.arange(256, dtype=np.uint8)}
    for i in range(1, 4):
        for delta in (-12, 12):
            y = x.copy()
            y[i] += delta
            result[f'k{x[i]}_{delta:+d}'] = np.rint(np.interp(np.arange(256), x, y)).astype(np.uint8)
    return result


def apply_lut(codes, tables):
    if codes.dtype != np.uint8 or codes.shape != (grid.HEIGHT, grid.WIDTH, 3):
        raise ValueError('Expected original native RGB DPAC codes')
    if tables.dtype != np.uint8 or tables.shape != (3, 256):
        raise ValueError('Expected three uint8 LUTs')
    return np.stack([tables[ch][codes[..., ch]] for ch in range(3)], axis=-1)


class Fit(Session):
    def __init__(self, root, label, routing_source):
        super().__init__(root, label)
        self.routing_source = routing_source
        route = json.loads((routing_source / 'status.json').read_text(encoding='utf8'))
        if (route['state'] not in ('calibration_complete', 'routing_approved') or route.get('error') or
            not all(route['routing'].get(ch, {}).get('accepted') for ch in ('R', 'G', 'B', 'RGB')) or
            route.get('final_standby', {}).get('mode') != 'Digital'):
            raise ValueError('Require complete fresh calibration with all four routing gates')
        self.modes = {ch: tuple(v) for ch, v in route['requested_modes'].items()}
        self.report.update(requested_modes=self.modes, routing_source=str(routing_source),
                           routing_sha256=sha(routing_source / 'status.json'),
                           comparisons=[], selection=[], decisions={})
        self.manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
        if self.manifest['version'] != 'v3.0-dpac':
            raise ValueError('Only current single-pass DPAC manifest accepted')
        self.tables = candidates()
        self.identity = np.tile(self.tables['identity'], (3, 1))
        self.current = self.identity.copy()
        self.save()

    def load(self, row):
        path = self.root / 'patterns' / f'{row["id"]:04}.png'
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        if (meta['recipe'] != row or sha(path) != meta['png_sha256'] or
            row['method'] != 'DPAC' or row['iterations'] != 0):
            raise ValueError(f'DPAC drive identity mismatch: {path}')
        source = Path(self.manifest['dataset']) / self.manifest['sources'][row['scene']]
        if sha(source) != meta['target']['source_sha256']:
            raise ValueError('Input target changed')
        amp, _ = prepare_image(source, (grid.HEIGHT, grid.WIDTH))
        return cv2.imread(str(path))[..., ::-1].copy(), amp, meta

    def references(self, cal, mode):
        ref = cal[mode]
        channels = range(3) if mode == 'RGB' else ['RGB'.index(mode)]
        result = {}
        for ch in channels:
            dark = cv2.GaussianBlur(rectify(ref['dark'], ch, ref['H']), (0, 0), 12)
            white = cv2.GaussianBlur(rectify(ref['white'], ch, ref['H']), (0, 0), 12) - dark
            valid = white > max(32., .1*float(np.median(white)))
            if valid.mean() < .98:
                raise RuntimeError(f'{mode}: insufficient fixed-white valid field ({valid.mean()})')
            result[ch] = (dark, white, valid)
        return result

    def measure(self, name, row, codes, target, tables, mode, cal, references, meta):
        self.b.set_timing(*self.modes[mode])
        self.b.show_codes(geo.pack(apply_lut(codes, tables)))
        raw = self.capture(name)
        roi = stats(raw, cal[mode]['mask'])
        measures = {}
        for ch, (dark, white, valid) in references.items():
            tag = 'RGB'[ch]
            if roi[tag]['near_clip'] > .0001:
                raise RuntimeError(f'{name}: saturated {tag} ROI')
            intensity = np.maximum(rectify(raw, ch, cal[mode]['H']) - dark, 0) / np.maximum(white, 1)
            amp = np.sqrt(intensity)
            mse = float(np.mean((amp[valid] - target[..., ch][valid])**2, dtype=np.float64))
            intensity_mse = float(np.mean((intensity[valid]-target[..., ch][valid]**2)**2, dtype=np.float64))
            measures[tag] = dict(fixed_white_amplitude_mse=mse,
                                 linear_intensity_psnr_db=float(-10*np.log10(max(intensity_mse, 1e-15))),
                                 mean_normalized_intensity=float(intensity[valid].mean()),
                                 valid_fraction=float(valid.mean()), near_clip=roi[tag]['near_clip'])
        record = dict(name=name, id=row['id'], scene=row['scene'], depth_mm=row['design_mm'],
                      split=row['split'], mode=mode, base_rgb_sha256=meta['rgb_sha256'],
                      metrics=measures)
        self.report['records'][-1].update(record)
        self.save()
        return record

    def drift(self, z, cal, prefix):
        self.b.show_codes(self.flat)
        for mode in self.modes:
            self.b.set_timing(*self.modes[mode])
            raw = self.capture(f'{prefix}_z{z:g}_{mode}_white_end')
            for ch in range(3) if mode == 'RGB' else ['RGB'.index(mode)]:
                h = cal[mode]['H']
                d = rectify(cal[mode]['dark'], ch, h)
                denominator = float((rectify(cal[mode]['white'], ch, h)-d).mean())
                change = float((rectify(raw, ch, h)-d).mean()/denominator-1)
                self.report.setdefault('white_drift', []).append(dict(depth_mm=z, mode=mode,
                    channel='RGB'[ch], phase=prefix, fraction=change))
                self.save()
                if abs(change) > .03:
                    raise RuntimeError(f'{mode}: white drift exceeds 3%: {change}')

    def fit_depth(self, z, cal):
        rows = [r for r in self.manifest['patterns'] if r['split'] == 'fit' and r['design_mm'] == z][:2]
        if len(rows) != 2:
            raise ValueError('Require two independent fit scenes per training depth')
        self.report['state'] = f'fitting_{z:g}mm'
        self.save()
        self.laser.configure('Analog')
        try:
            for ch, tag in enumerate('RGB'):
                ref = self.references(cal, tag)
                for j, row in enumerate(rows):
                    codes, target, meta = self.load(row)
                    for k, (key, table) in enumerate(list(self.tables.items())[1:]):
                        candidate = self.identity.copy()
                        candidate[ch] = table
                        order = ['A', 'B'] if (j+k+ch) % 2 == 0 else ['B', 'A']
                        paired = {}
                        for method in order:
                            name = f'fit_z{z:g}_{tag}_{row["id"]:04}_{key}_{method}'
                            measured = self.measure(name, row, codes, target,
                                self.identity if method == 'A' else candidate, tag, cal, ref, meta)
                            paired[method] = measured['metrics'][tag]['fixed_white_amplitude_mse']
                        self.report['comparisons'].append(dict(depth_mm=z, channel=tag,
                            scene=row['scene'], id=row['id'], candidate=key, order=order,
                            A=paired['A'], B=paired['B'], delta=paired['B']-paired['A'],
                            base_rgb_sha256=meta['rgb_sha256']))
                        self.save()
                    print(f'Fit z{z:g} {tag} scene {row["scene"]}: six candidates measured', flush=True)
            self.drift(z, cal, 'fit')
        finally:
            self.laser.configure('Digital')

    def export(self):
        for ch, tag in enumerate('RGB'):
            scores = {'identity': 0.}
            for key in list(self.tables)[1:]:
                rows = [r for r in self.report['comparisons'] if r['channel'] == tag and r['candidate'] == key]
                if len(rows) != 6 or {r['depth_mm'] for r in rows} != {3., 7., 9.}:
                    raise ValueError('Incomplete measured candidate bank')
                scores[key] = float(np.mean([r['delta'] for r in rows]))
            chosen = min(scores, key=scores.get)
            self.current[ch] = self.tables[chosen]
            self.report['decisions'][tag] = dict(chosen=chosen, mean_paired_mse_delta=scores,
                                               validation='provisional; independent selection pending')
        metadata = dict(version='rgb-dpac-lut-v3-first-candidate', created_kst=now(),
            channel_order='RGB', dtype='uint8', shape=[3, 256], wavelengths_nm=[638, 520, 450],
            direction='input inverted DPAC uint8 code -> output HDMI uint8 code',
            application='out[..., c] = table[c][in_codes[..., c]]; exactly once',
            active_xywh=[96, 60, 1728, 960], full_display_wh=[1920, 1080],
            invert_phase=True, nd_od_planned=.4, nominal_zero_mm=grid.ZERO,
            design_depths_mm=[3, 7, 9], untouched_test_depths_mm=[3, 5, 7, 9],
            identity_bypass='Skip LUT lookup, or use identity[c]=arange(256, dtype=uint8)',
            metric='Fixed per-depth/channel dark-subtracted spatial white, Gaussian sigma 12 px; amplitude MSE',
            scope='DIV2K only; Gaussian Splatting transfer unverified; provisional until independent validation',
            manifest_sha256=sha(self.root/'manifest.json'),
            candidate_tables=self.report['decisions'])
        folder = self.out / 'lut'
        folder.mkdir(exist_ok=False)
        path = folder / 'rgb_lut_candidate.npz'
        np.savez(path, table=self.current, identity=self.identity,
                 metadata=json.dumps(metadata, ensure_ascii=False))
        metadata['npz_sha256'] = sha(path)
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
        self.report['lut'] = dict(path=str(path.relative_to(self.root)), sha256=sha(path))
        self.save()
        print('First RGB LUT candidate saved:', path, flush=True)

    def select_depth(self, z, cal):
        self.report['state'] = f'independent_selection_{z:g}mm'
        self.save()
        rows = [r for r in self.manifest['patterns'] if r['split'] == 'selection' and r['design_mm'] == z]
        refs = {mode: self.references(cal, mode) for mode in self.modes}
        self.laser.configure('Analog')
        try:
            for j, row in enumerate(rows):
                codes, target, meta = self.load(row)
                for mode in self.modes:
                    pair = {}
                    order = ['A', 'B'] if j % 2 == 0 else ['B', 'A']
                    for method in order:
                        name = f'selection_z{z:g}_{mode}_{row["id"]:04}_{method}'
                        pair[method] = self.measure(name, row, codes, target,
                            self.identity if method == 'A' else self.current, mode, cal, refs[mode], meta)
                    self.report['selection'].append(dict(id=row['id'], depth_mm=z,
                        scene=row['scene'], mode=mode, order=order, pair=pair))
                    self.save()
                print(f'Selection z{z:g} scene {row["scene"]}: slot and integrated A/B captured', flush=True)
            self.drift(z, cal, 'selection')
        finally:
            self.laser.configure('Digital')

    def run_fit(self):
        frames = {z: geo.cached_patterns(z, self.root/'calibration_patterns') for z in (3., 7., 9.)}
        with exclusive():
            self.laser = ReadyBeam()
            try:
                self.report['standby'] = self.laser.configure('Digital')
                with Stage(interval=(90.780, 101.000)) as self.stage:
                    start = self.stage.initial['position_mm']
                    self.report['stage_start'] = self.stage.status()
                    try:
                        with Bench(dict(grid.C, settle_s=.25)) as self.b:
                            if (self.b.camera.Width.GetValue(), self.b.camera.Height.GetValue(),
                                self.b.camera.OffsetX.GetValue(), self.b.camera.OffsetY.GetValue()) != (4096, 3000, 0, 0):
                                raise RuntimeError('Full frame geometry gate')
                            for z in (3., 7., 9.):
                                self.fit_depth(z, self.calibrate(z, frames[z]))
                            self.export()
                            # Separate subfolder prevents overwriting fit calibration photographs.
                            self.camera = self.out / 'selection_camera'
                            self.camera.mkdir()
                            for z in (3., 7., 9.):
                                self.select_depth(z, self.calibrate(z, frames[z]))
                            self.report['state'] = 'candidate_and_selection_complete'
                    finally:
                        self.report['standby_before_return'] = self.laser.configure('Digital')
                        if self.stage.moves:
                            self.stage.move(start)
                        self.report['stage_end'] = self.stage.status()
                        self.report['moves'] = self.stage.moves
            except BaseException as exc:
                self.report.update(state='interrupted', error=repr(exc))
                raise
            finally:
                try:
                    self.report['final_standby'] = self.laser.configure('Digital')
                finally:
                    self.save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--label', default='fit_01')
    p.add_argument('--routing', type=Path, required=True)
    a = p.parse_args()
    Fit(a.root, a.label, a.routing).run_fit()
