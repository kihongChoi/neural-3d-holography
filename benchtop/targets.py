"""Frozen DIV2K targets with explicit radiometry and source-image split isolation."""
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps


def srgb_to_linear(value):
    value = np.asarray(value, np.float32)
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def prepare_image(path, shape):
    """Decode sRGB intensity before antialiased resizing, then sqrt to amplitude."""
    path = Path(path)
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as original:
        image = ImageOps.exif_transpose(original)
        original_size = list(image.size)
        icc = image.info.get('icc_profile')
        color_policy = 'assumed_sRGB_no_embedded_ICC'
        if icc:
            image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                              ImageCms.createProfile('sRGB'), outputMode='RGB')
            color_policy = 'embedded_ICC_converted_to_sRGB'
        else:
            image = image.convert('RGB')
        linear = srgb_to_linear(np.asarray(image, np.float32)/255.)
    rh, rw = map(int, shape)
    h, w = linear.shape[:2]
    scale = max(rw/w, rh/h)
    nh, nw = max(rh, math.ceil(h*scale)), max(rw, math.ceil(w*scale))
    resized = cv2.resize(linear, (nw, nh), interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR)
    top, left = (nh-rh)//2, (nw-rw)//2
    intensity = resized[top:top+rh, left:left+rw]
    amplitude = np.sqrt(np.clip(intensity, 0, 1)).astype(np.float32)
    metadata = {'source_path': str(path.resolve()), 'source_sha256': hashlib.sha256(payload).hexdigest(),
                'source_size_wh': original_size, 'color_policy': color_policy,
                'resized_size_wh': [nw, nh], 'crop_xywh': [left, top, rw, rh],
                'radiometry': 'sRGB -> linear intensity -> spatial resize/crop -> sqrt amplitude',
                'normalization': 'No per-image or per-channel normalization; white amplitude = 1'}
    return amplitude, metadata


def create_div2k_plan(root, dataset_root, shape, pairs=24, seed=42):
    root, dataset_root = Path(root), Path(dataset_root).resolve()
    destination = root/'targets/manifest.json'
    if destination.exists():
        existing = json.loads(destination.read_text(encoding='utf-8'))
        requested = {'dataset_root': str(dataset_root), 'roi_shape': list(shape), 'pairs': pairs, 'seed': seed}
        if any(existing[key] != value for key, value in requested.items()):
            raise ValueError('Existing DIV2K target plan differs; use a new output directory.')
        return existing
    if pairs < 12:
        raise ValueError('At least 12 pairs required for train/validation/test separation.')
    pools = {name: sorted((dataset_root/f'DIV2K_{name}_HR').glob('*.png')) for name in ('train', 'valid')}
    if not all(pools.values()):
        raise FileNotFoundError('Expected DIV2K_train_HR and DIV2K_valid_HR with PNG images.')
    splits = ['test' if i % 6 == 5 else 'val' if i % 6 == 4 else 'train' for i in range(pairs)]
    n_train, n_valid = splits.count('train'), splits.count('val')+splits.count('test')+2
    if n_train > len(pools['train']) or n_valid > len(pools['valid']):
        raise ValueError(f'Request needs {n_train} train and {n_valid} held-out originals, more than available.')
    rng = np.random.default_rng(seed)
    training = iter(rng.permutation(pools['train'])[:n_train])
    heldout = iter(rng.permutation(pools['valid'])[:n_valid])
    selections = [(i, split, next(training) if split == 'train' else next(heldout)) for i, split in enumerate(splits)]
    selections += [(1000, 'citl_demo', next(heldout)), (2000, 'new_target', next(heldout))]
    target_dir = root/'targets'
    target_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for slot, split, file in selections:
        file = Path(file)
        amplitude, metadata = prepare_image(file, shape)
        name = f'target_{slot:04d}'
        np.save(target_dir/f'{name}_amplitude.npy', amplitude)
        intensity = amplitude**2
        preview = np.where(intensity <= .0031308, intensity*12.92, 1.055*intensity**(1/2.4)-.055)
        Image.fromarray(np.round(np.clip(preview, 0, 1)*255).astype(np.uint8)).save(target_dir/f'{name}_rgb.png')
        entries.append(dict(metadata, slot=slot, split=split, target_id=file.relative_to(dataset_root).as_posix(),
                            original_split='train' if split == 'train' else 'valid',
                            amplitude_file=f'{name}_amplitude.npy', preview_file=f'{name}_rgb.png'))
    plan = {'dataset': 'DIV2K', 'dataset_root': str(dataset_root), 'roi_shape': list(shape),
            'pairs': pairs, 'seed': seed, 'available_originals': {k: len(v) for k, v in pools.items()},
            'split_policy': 'Official train for fitting; official valid partitioned by original image into val/test/demo/new target',
            'targets': entries}
    destination.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def target_entry(root, slot):
    path = Path(root)/'targets/manifest.json'
    if not path.exists():
        return None
    plan = json.loads(path.read_text(encoding='utf-8'))
    matches = [row for row in plan['targets'] if row['slot'] == slot]
    if len(matches) != 1:
        raise ValueError(f'Target slot {slot} not found uniquely in frozen target plan')
    return matches[0]


def load_target(root, shape, slot):
    row = target_entry(root, slot)
    if row is None:
        from .rgb_run import target_rgb
        return target_rgb(shape, slot)
    amplitude = np.load(Path(root)/'targets'/row['amplitude_file'])
    if amplitude.shape != tuple(shape)+(3,) or not np.isfinite(amplitude).all():
        raise ValueError('Frozen DIV2K target has the wrong shape or invalid values')
    return amplitude
