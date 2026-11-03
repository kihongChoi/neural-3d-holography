"""Recompute phase-null fits from saved, validated camera measurements."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fit(z, y):
    origin = float(np.mean(z))
    co = np.polyfit(z-origin, y, 2)
    prediction = np.polyval(co, z-origin)
    total = np.sum((y-np.mean(y))**2)
    minimum = float(origin-co[1]/(2*co[0]))
    return dict(extremum_mm=minimum, curvature=float(co[0]),
                r_squared=float(1-np.sum((y-prediction)**2)/total) if total else 0.,
                residual_rms=float(np.sqrt(np.mean((y-prediction)**2))),
                within_sampled_range=bool(min(z) <= minimum <= max(z)),
                polynomial_origin_mm=origin, polynomial=co.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.out)
    if (root/'INVALIDATED.json').exists():
        raise RuntimeError('These data were invalidated by sensor handling')
    rows = [json.loads(s) for s in (root/'measurements.jsonl').read_text().splitlines()]
    all_labels = list(dict.fromkeys(r['label'] for r in rows))
    labels = [label for label in all_labels if len({r['position_mm'] for r in rows if r['label'] == label}) >= 3]
    report = dict(measurement_positions=len(rows), labels={},
                  caveat='Fit residuals and scan repeatability are not absolute axial accuracy; different directions/field regions can have different optical nulls.')
    fig, axes = plt.subplots(len(labels), 2, figsize=(11, 3.3*len(labels)), squeeze=False)
    for j, label in enumerate(labels):
        part = [r for r in rows if r['label'] == label]
        z = np.array([r['position_mm'] for r in part])
        report['labels'][label] = {}
        for name in part[0]['probes']:
            values = np.array([r['probes'][name]['coefficients'] for r in part])
            tile_scores = np.mean(values**2, axis=2)
            curves = [np.mean(tile_scores, axis=1), tile_scores[:, 4]]
            entry = dict(global_fit=fit(z, curves[0]), center_fit=fit(z, curves[1]),
                         tiles=[fit(z, tile_scores[:, k]) for k in range(9)])
            report['labels'][label][name] = entry
            if name == 'x16':
                continue  # Huge non-null background; tabulated, not mixed into estimate.
            for k, y in enumerate(curves):
                ax = axes[j, k]
                f = entry['global_fit' if k == 0 else 'center_fit']
                points = ax.plot(z, y, 'o', label=f'{name}: {f["extremum_mm"]:.3f} mm')[0]
                zz = np.linspace(min(z), max(z), 150)
                ax.plot(zz, np.polyval(f['polynomial'], zz-f['polynomial_origin_mm']), color=points.get_color())
                ax.set(xlabel='Stage coordinate [mm]', ylabel='Differential carrier power',
                       title=f'{label} / '+('3x3 field mean' if k == 0 else 'central tile'))
                ax.axvline(101.04, color='gray', ls=':', label='User nominal 101.04' if name == list(part[0]['probes'])[-1] else None)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'phase_null_fits.png', dpi=160)
    (root/'fits.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({label:{name:{'center_mm':v['center_fit']['extremum_mm'],
                                      'r2':v['center_fit']['r_squared']}
                             for name,v in probes.items()} for label,probes in report['labels'].items()},indent=2))


if __name__ == '__main__':
    main()
