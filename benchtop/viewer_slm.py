"""SLM image preparation and exact-once LUT application for the desktop viewer."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_XYWH = (96, 60, 1728, 960)
LUT_PATH = ROOT/'results/20261111-140112_joint-rgb-offline-lut-distillation/rgb_lut_candidate.npz'
LUT_SHA = '034d9e8454e86190c7e1e19be64c8f44a54bb699ea5c2e9eb6e2e0add217b569'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_lut(path=LUT_PATH):
    payload = Path(path).read_bytes()
    if sha(payload) != LUT_SHA:
        raise ValueError('실물 검증된 RGB LUT 파일의 SHA-256이 일치하지 않습니다.')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        table = data['table'].copy()
        metadata = json.loads(str(data['metadata']))
    if (table.dtype != np.uint8 or table.shape != (3, 256) or
            metadata['format'] != 'joint-rgb-offline-lut-v1' or
            metadata['channel_order'] != 'RGB' or metadata['active_xywh'] != list(ACTIVE_XYWH) or
            sha(table.tobytes()) != metadata['candidate_array_sha256'] or
            np.any(np.diff(table.astype(int), axis=1) < 0) or
            np.any(table[:, 0] != 0) or np.any(table[:, -1] != 255) or
            np.max(np.abs(table.astype(int)-np.arange(256))) > 12):
        raise ValueError('RGB LUT 형식/출처/범위 검사 실패')
    table.setflags(write=False)
    return table


def compose_frame(original, table=None):
    """Map the immutable original drive, then pack into the established 80% ROI."""
    if original.dtype != np.uint8 or original.shape != (960, 1728, 3):
        raise ValueError('1728×960 RGB8 원본 구동 이미지가 필요합니다.')
    active = original if table is None else np.stack([table[c, original[..., c]] for c in range(3)], -1)
    frame = np.full((1080, 1920, 3), 128, np.uint8)
    frame[60:1020, 96:1824] = active
    return frame


def read_image(path):
    path = Path(path)
    if any(part.startswith('20261103') for part in path.parts):
        raise ValueError('폐기된 2026-11-03 구동 파일은 사용할 수 없습니다.')
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.mode in ('I', 'F', 'I;16', 'I;16B', 'I;16L'):
            raise ValueError('8-bit 이미지 파일을 선택하세요.')
        image = ImageOps.exif_transpose(image)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            layer = image.convert('RGBA')
            canvas = Image.new('RGBA', layer.size, (0, 0, 0, 255))
            canvas.alpha_composite(layer)
            image = canvas
        rgb = np.asarray(image.convert('RGB')).copy()
    return rgb, dict(source=str(path.resolve()), source_sha256=sha(data), source_wh=[rgb.shape[1], rgb.shape[0]])


def direct_codes(rgb):
    """Keep phase pixels native; a full native frame contributes its active region."""
    h, w = rgb.shape[:2]
    if (h, w) == (1080, 1920):
        return rgb[60:1020, 96:1824].copy(), 'full-frame active ROI (no resampling)'
    if h > 960 or w > 1728:
        raise ValueError('구동 이미지는 1728×960 이하 또는 전체 1920×1080이어야 합니다. 위상 코드는 자동 축소하지 않습니다.')
    active = np.full((960, 1728, 3), 128, np.uint8)
    x, y = (1728-w)//2, (960-h)//2
    active[y:y+h, x:x+w] = rgb
    return active, 'native pixels, centered (no resampling)'


def target_amplitude(rgb):
    # Fit the complete photo without cropping or distorting its aspect ratio.
    import cv2
    scale = min(1728/rgb.shape[1], 960/rgb.shape[0])
    width, height = max(1, round(rgb.shape[1]*scale)), max(1, round(rgb.shape[0]*scale))
    canvas = np.zeros((960, 1728, 3), np.float32)
    x, y = (1728-width)//2, (960-height)//2
    srgb = rgb.astype(np.float32)/255
    linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
    linear = cv2.resize(linear, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas[y:y+height, x:x+width] = np.sqrt(np.clip(linear, 0, 1))
    return canvas


def generate_dpac(rgb, depth):
    if not math.isfinite(depth) or not 0 <= depth <= 20:
        raise ValueError('설계 깊이는 0–20 mm여야 합니다.')
    import torch
    from .optics import ASM
    from .paper_model import dpac
    from .hardware import encode_phase
    config = json.loads((ROOT/'configs/benchtop.json').read_text())
    config.update(slm_shape=[960, 1728], distance_m=depth/1000)
    target = target_amplitude(rgb)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codes = []
    with torch.inference_mode():
        for channel, wavelength in enumerate((638e-9, 520e-9, 450e-9)):
            model = ASM(dict(config, wavelength_m=wavelength), device)
            amplitude = torch.as_tensor(np.ascontiguousarray(target[..., channel]), device=device)
            codes.append(encode_phase(dpac(model, amplitude).cpu().numpy()))
    return np.stack(codes, -1)


def read_complex(path):
    """Load an SLM-plane complex field; never unpickle or reinterpret real data."""
    path = Path(path)
    if path.suffix.lower() != '.npy':
        raise ValueError('복소 CGH는 .npy 파일을 선택하세요.')
    if any(part.startswith('20261103') for part in path.parts):
        raise ValueError('폐기된 2026-11-03 파일은 사용할 수 없습니다.')
    payload = path.read_bytes()
    field = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(field, np.ndarray) or field.dtype.kind != 'c' or field.dtype.itemsize not in (8, 16):
        raise ValueError('complex64 또는 complex128 복소 배열이 필요합니다. 실수/위상 전용 배열은 지원하지 않습니다.')
    original_shape, original_dtype = list(field.shape), str(field.dtype)
    if field.ndim == 2:
        field = field[None]
        layout = 'HW'
    elif field.ndim == 3:
        first, last = field.shape[0] in (1, 3), field.shape[-1] in (1, 3)
        if first == last:
            raise ValueError('채널 축을 식별할 수 없습니다. [3,H,W], [H,W,3], [1,H,W] 또는 [H,W,1] 형식이 필요합니다.')
        layout = 'CHW' if first else 'HWC'
        if last:
            field = np.moveaxis(field, -1, 0)
    else:
        raise ValueError('복소 배열 형식은 [H,W], [C,H,W] 또는 [H,W,C]여야 합니다. C=1 또는 3(RGB).')
    if not field.size or not np.isfinite(field).all():
        raise ValueError('빈 배열 또는 NaN/Inf가 있는 복소장은 변환할 수 없습니다.')
    channels = field.shape[0]
    if channels == 1:
        field = np.repeat(field, 3, axis=0)
    return field, dict(source=str(path.resolve()), source_sha256=sha(payload),
                       source_shape=original_shape, source_dtype=original_dtype, source_layout=layout,
                       source_channels=channels, channel_order='RGB', single_channel_repeated=channels == 1)


def encode_complex(field, gain=1., amp_max=None):
    """User's checkerboard DPH equation, with one amplitude scale shared by RGB."""
    field = np.asarray(field)
    if field.ndim != 3 or not np.iscomplexobj(field) or not field.size or not np.isfinite(field).all():
        raise ValueError('유한한 비어 있지 않은 [C,H,W] 복소장이 필요합니다.')
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('DPAC gain은 유한한 양수여야 합니다.')
    amplitude = np.abs(field)
    amax = float(amplitude.max()) if amp_max is None else float(amp_max)
    if not math.isfinite(amax) or amax < 0 or (amp_max is not None and amax == 0):
        raise ValueError('amp_max는 유한한 양수여야 합니다.')
    # Divide first to avoid overflow for large finite inputs and positive gain.
    normalized = (amplitude / max(amax, 1e-30)) * gain
    clipped = (normalized > 1).mean(axis=(-2, -1)).tolist()
    offset = np.arccos(np.clip(normalized, 0, 1))
    y, x = np.ogrid[:field.shape[-2], :field.shape[-1]]
    phase = np.angle(field)
    theta = np.remainder(np.where((x+y) % 2 == 0, phase+offset, phase-offset), 2*np.pi)
    return theta.astype(np.float32), dict(amp_max=amax, gain=float(gain), clipped=clipped)


def complex_dpac(path, gain=1.):
    from .hardware import encode_phase
    field, meta = read_complex(path)
    amax = float(np.abs(field).max())
    h, w = field.shape[-2:]
    if (h, w) == (1080, 1920):
        field = field[:, 60:1020, 96:1824]
        framing = 'full-frame active ROI (no resampling)'
    elif h <= 960 and w <= 1728:
        active = np.zeros((3, 960, 1728), field.dtype)
        x, y = (1728-w)//2, (960-h)//2
        active[:, y:y+h, x:x+w] = field
        field = active
        framing = 'native pixels, complex-zero centered padding (no resampling)'
    else:
        raise ValueError('복소장은 1728×960 이하 또는 전체 1920×1080이어야 합니다. 복소 위상은 자동 축소하지 않습니다.')
    theta, info = encode_complex(field, gain, amp_max=amax if amax != 0 else None)
    codes = np.ascontiguousarray(np.moveaxis(encode_phase(theta), 0, -1))
    meta.update(mode='complex_dpac', framing=framing, normalization='shared RGB max over entire source',
                input_plane='SLM', propagation=False, lowpass=False, iterations=0,
                phase_encoding='even phi+acos(A), odd phi-acos(A); wrapped [0,2pi)',
                drive_encoding='hardware.encode_phase: +pi offset, inverted, round to uint8',
                clipped_domain='active 1728x960 including padding', **info)
    return codes, meta


def prepare(path, mode, depth, output, gain=1.):
    """Called in an executor; isolate Torch from camera/Kinesis vendor DLLs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if mode == 'dpac':
        result = subprocess.run([sys.executable, '-m', 'benchtop.viewer_slm', str(Path(path).resolve()),
                                 str(output.resolve()), '--depth', str(depth)], cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr[-2500:] or 'DPAC 생성 실패')
        with Image.open(output/'original_codes.png') as image:
            codes = np.asarray(image.convert('RGB')).copy()
        meta = json.loads((output/'pattern.json').read_text(encoding='utf-8'))
    elif mode == 'complex':
        codes, meta = complex_dpac(path, gain)
        meta.update(active_xywh=list(ACTIVE_XYWH), original_codes_sha256=sha(codes.tobytes()), lut_applied=False)
        Image.fromarray(codes).save(output/'original_codes.png')
        frame = compose_frame(codes)
        Image.fromarray(frame).save(output/'dpac_slm.png')
        meta.update(slm_png='dpac_slm.png', slm_frame_sha256=sha(frame.tobytes()))
    elif mode == 'codes':
        rgb, meta = read_image(path)
        codes, framing = direct_codes(rgb)
        meta.update(mode='original_drive', framing=framing, active_xywh=ACTIVE_XYWH,
                    original_codes_sha256=sha(codes.tobytes()), lut_applied=False)
        Image.fromarray(codes).save(output/'original_codes.png')
    else:
        raise ValueError(f'알 수 없는 입력 모드: {mode}')
    meta['output_directory'] = str(output.resolve())
    (output/'pattern.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    codes.setflags(write=False)
    return codes, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--depth', type=float, default=3.)
    args = parser.parse_args()
    rgb, meta = read_image(args.source)
    codes = generate_dpac(rgb, args.depth)
    meta.update(mode='single_pass_dpac', design_mm=args.depth, iterations=0, wavelengths_nm=[638, 520, 450],
                framing='aspect-preserving contain, no crop', active_xywh=ACTIVE_XYWH,
                original_codes_sha256=sha(codes.tobytes()), lut_applied=False)
    Image.fromarray(codes).save(args.output/'original_codes.png')
    (args.output/'pattern.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
