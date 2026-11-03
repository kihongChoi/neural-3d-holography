"""Axial phase-to-intensity null measurement, with restricted LTS motion.

Small +/- sinusoidal phase probes have opposite first-order intensity contrast
away from a conjugate plane. Measure several frequencies/orientations; their
null spread reveals frequency-dependent aberration rather than hiding it.
Existing +15 mm homography is only a provisional lateral mapping, not a depth
calibration. Raw sensor data are retained so analysis can be repeated.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import cv2

from .hardware import Bench
from .stage import Stage


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def probes(periods=(16, 24), amplitude=.65, kind='phase', axes=('x', 'y')):
    y, x = np.indices((1080, 1920), dtype=np.float32)
    result = []
    checker = (1-2*((x+y) % 2)).astype(np.float32)
    for axis, coord in [('x', x), ('y', y)]:
        if axis not in axes:
            continue
        for period in periods:
            wave = np.cos(2*np.pi*coord/period)
            pair = [sign*amplitude*wave for sign in [1, -1]]
            if kind == 'amplitude':
                pair = [checker*np.arccos(.5+sign*.45*wave) for sign in [1, -1]]
            result.append((f'{axis}{period}', pair, np.exp(-2j*np.pi*coord/period)))
    return result


def register(bench, output):
    """Register large dark DPAC disks directly near the conjugate plane.

    Independent validation points check the measured lateral mapping. This is
    not an axial calibration, nor an assumption that the current z is in focus.
    """
    y, x = np.indices((1080, 1920), dtype=np.float32)
    checker = (1-2*((x+y) % 2)).astype(np.float32)
    points = [(x, y) for y in [240, 540, 840] for x in [360, 960, 1560]]
    validation = [(660, 390), (1260, 390), (660, 690), (1260, 690)]
    found = []
    diagnostics = []
    for i, center in enumerate(points+validation):
        flat = bench.capture(np.zeros((1080, 1920), np.float32), frames=5)[..., 1]
        disk = ((x-center[0])**2+(y-center[1])**2) < 36**2
        phase = np.where(disk, checker*np.pi/2, 0).astype(np.float32)
        dark = bench.capture(phase, frames=5)[..., 1]
        delta = cv2.GaussianBlur(flat-dark, (0, 0), 3)
        _, peak, _, (ix, iy) = cv2.minMaxLoc(delta)
        if peak < .003:
            raise RuntimeError(f'No dark registration disk: {peak}')
        yy, xx = np.mgrid[max(0, iy-40):iy+41, max(0, ix-40):ix+41]
        weight = np.maximum(delta[yy, xx]-peak*.3, 0)
        xy = [float((xx*weight).sum()/weight.sum()), float((yy*weight).sum()/weight.sum())]
        found.append(xy)
        diagnostics.append(dict(slm_xy=center, sensor_half_xy=xy, peak=peak))
        np.savez_compressed(output/f'registration_{i:02d}.npz', flat=flat, dark=dark)
        print('Registration', i, center, xy, peak, flush=True)
    H, _ = cv2.findHomography(np.float64(points), np.float64(found[:9]), method=0)
    all_points = np.float64(points+validation)
    pred = cv2.perspectiveTransform(all_points[None], H)[0]
    error = np.linalg.norm(pred-found, axis=1)
    report = dict(homography_slm_to_half_sensor=H.tolist(), points=diagnostics,
                  fit_rms_px=float(np.sqrt(np.mean(error[:9]**2))),
                  validation_error_px=error[9:].tolist(),
                  frame_metadata=bench.frame_metadata,
                  note='Lateral registration only, measured after sensor handling; z not inferred.')
    write(output/'registration.json', report)
    return refine_registration(output)


def refine_registration(output):
    """Fit disk boundaries, avoiding illumination-biased intensity centroids."""
    from scipy.optimize import least_squares
    output = Path(output)
    report = json.loads((output/'registration.json').read_text())
    original = output/'registration_centroid.json'
    if not original.exists():
        write(original, report)
    found = []
    for i, point in enumerate(report['points']):
        data = np.load(output/f'registration_{i:02d}.npz')
        flat, dark = data['flat'], data['dark']
        contrast = cv2.GaussianBlur((flat-dark)/np.maximum(flat, .002), (0, 0), 1.2)
        ix, iy = np.round(point['sensor_half_xy']).astype(int)
        x0, y0 = ix-55, iy-55
        mask = np.uint8(contrast[y0:iy+56, x0:ix+56] > .25)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
        solution = least_squares(lambda q: np.linalg.norm(pts-q[:2], axis=1)-q[2],
                                 [55, 55, 33], loss='soft_l1', f_scale=.5).x
        if not 25 < solution[2] < 40:
            raise RuntimeError('Unexpected registration disk radius')
        xy = [float(solution[0]+x0), float(solution[1]+y0)]
        found.append(xy)
        point['sensor_half_xy'] = xy
    src = np.float64([p['slm_xy'] for p in report['points']])
    H, _ = cv2.findHomography(src[:9], np.float64(found[:9]), method=0)
    error = np.linalg.norm(cv2.perspectiveTransform(src[None], H)[0]-found, axis=1)
    report.update(homography_slm_to_half_sensor=H.tolist(),
                  fit_rms_px=float(np.sqrt(np.mean(error[:9]**2))),
                  validation_error_px=error[9:].tolist(),
                  detection='Normalized shadow contour at 0.25; robust circle boundary fit')
    write(output/'registration.json', report)
    if report['fit_rms_px'] > 2 or max(error[9:]) > 2:
        raise RuntimeError(f'Poor lateral registration: {error}')
    print('Lateral registration RMS', report['fit_rms_px'], 'validation', error[9:], flush=True)
    return H


def measure(bench, stage, output, label, positions, periods, amplitude, kind, axes):
    calibration = json.loads(Path('results/paper_cnnpropcnn_20261024/green/calibration/calibration.json').read_text())
    lateral = output/'registration.json'
    if lateral.exists():
        calibration = json.loads(lateral.read_text())
        if calibration['fit_rms_px'] > 2 or max(calibration['validation_error_px']) > 2:
            raise RuntimeError('Existing lateral registration failed validation')
    H = np.asarray(calibration['homography_slm_to_half_sensor'])
    rows = []
    p = probes(periods, amplitude, kind, axes)
    for i, pos in enumerate(positions):
        if (output/'STOP_REQUESTED').exists():
            raise RuntimeError('Stopped by request before stage movement')
        status = stage.status() if pos is None else stage.move(float(pos))
        z = status['position_mm']
        row = dict(position_mm=z, label=label, index=i, probes={}, timestamp=time.time(),
                   phase_amplitude_rad=amplitude, kind=kind)
        folder = output/f'{label}_{i:03d}_{z:.6f}'
        folder.mkdir()
        flat = bench.capture(np.zeros((1080, 1920), np.float32), frames=3)
        np.save(folder/'flat_green.npy', flat[..., 1])
        for name, pair, carrier in p:
            images = []
            for sign, phase in zip([1, -1], pair):
                rgb = bench.capture(phase, frames=3)
                raw = rgb[..., 1]
                np.save(folder/f'{name}_{sign:+d}.npy', raw.astype(np.float32))
                meta = bench.frame_metadata[-3:]
                if max(v['saturated_fraction'] for v in meta) > .0001:
                    raise RuntimeError('Saturated sensor samples; stop and reduce exposure')
                images.append(cv2.warpPerspective(raw, np.linalg.inv(H), (1920, 1080)))
            plus, minus = images
            # Central 1408x768 ROI avoids relay aperture edges. Pair normalization
            # cancels common illumination; low-frequency drift has little lock-in.
            roi = np.s_[156:924, 256:1664]
            mean = float(((plus+minus)/2)[roi].mean())
            if mean < .001:
                raise RuntimeError(f'Insufficient green signal: {mean}')
            diff = (plus-minus)/(2*mean)
            coefficients = []
            for ys in [slice(156, 412), slice(412, 668), slice(668, 924)]:
                for xs in [slice(256, 704), slice(704, 1152), slice(1152, 1600)]:
                    a = diff[ys, xs]
                    # Exact carrier fit with DC term; works with noninteger periods.
                    c = carrier[ys, xs]
                    A = np.stack([c.real.ravel(), c.imag.ravel(), np.ones(a.size)], axis=1)
                    v = np.linalg.lstsq(A, a.ravel(), rcond=None)[0][:2]
                    coefficients.append(v.tolist())
            row['probes'][name] = dict(mean_intensity=mean, coefficients=coefficients,
                                       score=float(np.mean(np.square(coefficients))))
        row['score'] = float(np.mean([v['score'] for v in row['probes'].values()]))
        row['stage_status'] = stage.status()
        row['frames'] = bench.frame_metadata[-(3+6*len(p)):]
        with (output/'measurements.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        rows.append(row)
        write(output/'progress.json', dict(label=label, completed=i+1, total=len(positions), last=row,
                                         all_moves=stage.moves))
        print(json.dumps(dict(label=label, index=i, position_mm=z, score=row['score'])), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--positions', type=float, nargs='*')
    ap.add_argument('--periods', type=int, nargs='+', default=[16, 24])
    ap.add_argument('--amplitude', type=float, default=.65)
    ap.add_argument('--kind', choices=['phase', 'amplitude'], default='phase')
    ap.add_argument('--axes', nargs='+', choices=['x', 'y'], default=['x', 'y'])
    ap.add_argument('--register', action='store_true')
    args = ap.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cal = json.loads(Path('results/paper_cnnpropcnn_20261024/green/calibration/calibration.json').read_text())
    config = cal['config']
    config.update(settle_s=.45, exposure_us=100., average_frames=3)
    # Keep existing gain, laser settings, CFS timing, and framebuffer verification.
    with Stage() as stage:
        write(output/f'{args.label}_stage_before.json', dict(status=stage.initial,
              limits=stage.original_limits, backlash_mm=stage.backlash_mm,
              config=config, positions=args.positions, periods=args.periods,
              phase_amplitude_rad=args.amplitude, method=args.kind, axes=args.axes))
        # Connect and verify the display/camera BEFORE arming or moving.
        with Bench(config, display_wait_s=60) as bench:
            write(output/f'{args.label}_camera_actual.json', dict(
                  exposure_us=bench.camera.ExposureTime.GetValue(),
                  gain_db=bench.camera.Gain.GetValue(),
                  trigger_delay_us=bench.camera.TriggerDelay.GetValue()))
            flat = bench.capture(np.zeros((1080, 1920), np.float32), frames=3)
            np.save(output/f'{args.label}_flat_rgb.npy', flat)
            from PIL import Image
            Image.fromarray(np.uint8(np.clip(flat/.2, 0, 1)*255)).save(output/f'{args.label}_flat.png')
            print('Flat mean RGB', flat.mean((0,1)).tolist(), 'max', flat.max(), flush=True)
            if args.register:
                register(bench, output)
            if args.positions:
                for target in args.positions:
                    from .stage import validate_target
                    validate_target(target, stage.interval, stage.backlash_mm)
                stage.arm()
                print('Runtime limit readback', stage.limits(), flush=True)
            try:
                measure(bench, stage, output, args.label, args.positions or [None], args.periods, args.amplitude, args.kind, args.axes)
            finally:
                write(output/f'{args.label}_stage_after.json', dict(status=stage.status(), moves=stage.moves,
                                                                   limits=stage.limits()))
                try:
                    bench.show(np.zeros((1080, 1920), np.float32))
                except Exception as exc:
                    warnings.warn(f'Could not verify final flat display: {exc}')


if __name__ == '__main__':
    main()
