"""Measured SLM-to-sensor registration with validation and provenance."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def channel_intensity(rgb, config):
    return rgb[..., ['red', 'green', 'blue'].index(config['channel'])]


def lens_pattern(config, center, radius=70):
    h, w = config['slm_shape']
    y, x = np.indices((h, w), dtype=np.float32)
    cx, cy = center
    r2 = (x-cx)**2+(y-cy)**2
    phase = np.zeros((h, w), np.float32)
    mask = r2 < radius**2
    pitch = config['slm_pitch_m']*abs(config['relay_magnification'])
    phase[mask] = -np.pi*pitch**2*r2[mask]/(config['wavelength_m']*config['distance_m'])
    return phase


def find_focus(image, reference):
    delta = gaussian_filter(image-reference, 2)
    noise = float(np.median(np.abs(delta-np.median(delta)))*1.4826+1e-6)
    iy, ix = np.unravel_index(np.argmax(delta), delta.shape)
    peak = float(delta[iy, ix])
    if peak < max(0.015, noise*15):
        raise RuntimeError(f'No reliable calibration focus: peak={peak:.4g}, noise={noise:.4g}. '
                           'Check display, LUT direction, distance and polarization.')
    r = 10
    y0, y1 = max(0, iy-r), min(image.shape[0], iy+r+1)
    x0, x1 = max(0, ix-r), min(image.shape[1], ix+r+1)
    weights = np.maximum(delta[y0:y1, x0:x1]-peak*0.25, 0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xy = [float((xx*weights).sum()/weights.sum()), float((yy*weights).sum()/weights.sum())]
    return xy, {'peak': peak, 'noise': noise, 'snr': peak/noise}


def calibrate(bench, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = channel_intensity(bench.capture(np.zeros(config['slm_shape'], np.float32)), config)
    np.save(output/'flat_intensity.npy', reference)
    h, w = config['slm_shape']
    rh, rw = config['roi_shape']
    if config.get('wide_registration'):
        points = [[w/2+dx*rw, h/2+dy*rh] for dy in [-.42, 0, .42] for dx in [-.42, 0, .42]]
        validation_points = [[w/2+dx*rw, h/2+dy*rh] for dy in [-.48, .48] for dx in [-.48, .48]]
    else:
        points = [[w/2+dx, h/2+dy] for dy in [-220, 0, 220] for dx in [-350, 0, 350]]
        validation_points = [[w/2-170, h/2-110], [w/2+170, h/2+110]]
    detected, diagnostics = [], []
    for i, center in enumerate(points):
        image = channel_intensity(bench.capture(lens_pattern(config, center)), config)
        np.save(output/f'focus_{i:02d}.npy', image)
        xy, diag = find_focus(image, reference)
        detected.append(xy)
        diagnostics.append(diag)
        print(f'Calibration {i+1}/{len(points)}: SLM {center} -> camera/2 {xy}, SNR={diag["snr"]:.1f}', flush=True)
    source = np.array(points, np.float64)
    dest = np.array(detected, np.float64)
    H, inliers = cv2.findHomography(source, dest, cv2.RANSAC, 2.0)
    if H is None or inliers.sum() < 8:
        raise RuntimeError('Calibration failed: fewer than 8 consistent correspondences.')
    predicted = cv2.perspectiveTransform(source[None].astype(np.float32), H)[0]
    rms = float(np.sqrt(np.mean(np.sum((predicted-dest)**2, axis=1))))
    if rms > 2.0:
        raise RuntimeError(f'Calibration residual {rms:.2f} camera/2 pixels exceeds 2.')
    validation = []
    for center in validation_points:
        image = channel_intensity(bench.capture(lens_pattern(config, center)), config)
        xy, diag = find_focus(image, reference)
        expected = cv2.perspectiveTransform(np.float32([[center]]), H)[0, 0]
        error = float(np.linalg.norm(np.array(xy)-expected))
        validation.append({'slm_xy': center, 'camera_xy': xy, 'error_px': error})
        if error > 3.0:
            raise RuntimeError(f'Held-out calibration error {error:.2f} pixels exceeds 3.')
    report = {'config': config, 'homography_slm_to_half_sensor': H.tolist(),
              'sensor_shape': list(reference.shape), 'rms_half_sensor_px': rms,
              'slm_points': points, 'camera_points': detected, 'diagnostics': diagnostics,
              'validation': validation, 'frame_metadata': bench.frame_metadata}
    if (output/'dark_intensity.npy').exists():
        report['dark_file'] = 'dark_intensity.npy'
    (output/'calibration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


class Rectifier:
    def __init__(self, path, config):
        path = Path(path)
        self.report = json.loads((path/'calibration.json').read_text(encoding='utf-8'))
        self.config = config
        for key in ['slm_shape', 'distance_m', 'wavelength_m', 'relay_magnification',
                    'slm_pitch_m', 'camera_serial', 'slm_screen', 'channel', 'invert_phase',
                    'exposure_us', 'trigger_delay_us', 'gain_db', 'fourier_aperture']:
            if self.report['config'][key] != config[key]:
                raise ValueError(f'Calibration mismatch for {key}; recalibrate.')
        self.H = np.asarray(self.report['homography_slm_to_half_sensor'])
        self.dark = (np.load(path/self.report['dark_file']) if 'dark_file' in self.report else
                     np.zeros(self.report['sensor_shape'], np.float32))
        self.flat = self.warp(np.maximum(np.load(path/'flat_intensity.npy')-self.dark, 0))
        h, w = config['slm_shape']
        rh, rw = config['roi_shape']
        self.roi = (slice((h-rh)//2, (h-rh)//2+rh), slice((w-rw)//2, (w-rw)//2+rw))
        ones = np.ones(self.report['sensor_shape'], np.float32)
        if np.min(self.warp(ones)[self.roi]) < 0.999:
            raise ValueError('Loss ROI is not fully inside the calibrated camera image.')
        self.intensity_scale = float(np.median(self.flat[self.roi]))
        if self.intensity_scale < 0.005:
            raise RuntimeError('Too little light in the calibrated ROI.')

    def warp(self, intensity):
        h, w = self.config['slm_shape']
        return cv2.warpPerspective(intensity, self.H, (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def amplitude(self, rgb):
        intensity = channel_intensity(rgb, self.config)
        raw_roi = self.warp(intensity)[self.roi]
        if float((raw_roi >= 0.98).mean()) > 0.001:
            raise RuntimeError('Camera saturation in training ROI; reduce exposure and recalibrate.')
        warped = self.warp(intensity-self.dark)[self.roi]
        return np.sqrt(np.maximum(warped, 0)/self.intensity_scale).astype(np.float32)

    def source_amplitude(self):
        # Smooth flat-field estimate, not a claim of measured SLM-plane amplitude.
        return np.sqrt(np.maximum(gaussian_filter(self.flat, 15), 0)/self.intensity_scale).astype(np.float32)
