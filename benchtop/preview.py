"""Read-only previews of recorded train/validation pairs.

The running trainer need not be restarted. Legacy workers expose their last
completed epoch via append-only training logs, not their in-flight batch ID.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image


def shared_bytes(path):
    """Read immutable samples with shared deletion; never use on hot checkpoints."""
    path = Path(path)
    if os.name != 'nt':
        return path.read_bytes()
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    kernel.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = ctypes.c_longlong()
        if not kernel.GetFileSizeEx(handle, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not 0 < size.value < 256*2**20:
            raise ValueError('Unexpected preview input size')
        buffer = ctypes.create_string_buffer(size.value)
        count = wintypes.DWORD()
        if not kernel.ReadFile(handle, buffer, size.value, ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if count.value != size.value:
            raise OSError('Incomplete checkpoint snapshot')
        return buffer.raw
    finally:
        kernel.CloseHandle(handle)


def read_json(path):
    return json.loads(shared_bytes(path).decode('utf-8'))


def select_sample(checkpoint, pool, split, serial='auto'):
    if split not in ('train', 'validation'):
        raise ValueError('Preview split must be train or validation')
    wanted = 'val' if split == 'validation' else 'train'
    rows = [r for r in pool if r['split'] == wanted]
    if not rows:
        raise ValueError('No recorded samples in this split')
    trained = checkpoint['branches']['red']['training_sample_serials']
    if serial != 'auto':
        selected = next((r for r in rows if r['serial'] == int(serial)), None)
        if selected is None or (wanted == 'train' and selected['serial'] not in trained):
            raise ValueError('Sample was not used in this training epoch')
        return selected, 'selected_recorded_sample'
    if wanted == 'val':
        return rows[0], 'fixed_validation_probe'
    offset = checkpoint.get('sample_offset', 0)
    global_epoch = checkpoint['global_epoch']
    if global_epoch < 1 and offset == 0:
        raise ValueError('This checkpoint has no completed training update')
    seed_epoch = global_epoch if offset else global_epoch-1
    order = np.random.default_rng(42+seed_epoch).permutation(trained).tolist()
    order = order[:offset] if offset else order
    lookup = {r['serial']:r for r in rows}
    for index, value in enumerate(reversed(order)):
        if value in lookup:
            return lookup[value], 'last_completed_train_sample' if index == 0 else 'last_retained_trained_sample'
    raise ValueError('Training pairs for this epoch have been pruned')


def events(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = shared_bytes(path).decode('utf-8').splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def training_context(root):
    root = Path(root)
    pool = read_json(root/'campaign/pool.json')
    rows = [r for r in events(root/'campaign/training.jsonl') if r.get('channel') == 'rgb']
    if not rows:
        raise ValueError('No completed joint RGB epoch is available')
    last = rows[-1]
    live = root/'live'
    pointer = read_json(live/'current.json')
    state = read_json(live/pointer['run_id']/'state.json')
    limit = state.get('retention',{}).get('rolling_train_pairs',192)
    ledger = events(root/'campaign/acquisition.jsonl')
    trained = [r['serial'] for r in ledger if r['split']=='train' and r['cycle']<=last['cycle']][-limit:]
    if not trained:
        trained = [r['serial'] for r in pool if r['split']=='train']
    sample_state = dict(global_epoch=last['global_epoch'],sample_offset=0,
                        branches={'red':{'training_sample_serials':trained}})
    cnns = [stream for stream in state['streams'].values() if stream['stage']=='cnn' and stream['channel']=='rgb']
    snapshot = cnns[-1]['snapshots'][-1] if cnns and cnns[-1]['snapshots'] else None
    return pool,last,sample_state,snapshot


def catalog(root):
    root = Path(root)
    pool = read_json(root/'campaign/pool.json')
    return dict(available=(root/'campaign/training.jsonl').exists(),
                samples=[dict(serial=r['serial'],split='validation' if r['split']=='val' else 'train',
                              target_id=r['source']['target_id']) for r in pool if r['split'] in ('train','val')],
                live_batch_available=False,
                note='Train auto previews the last completed epoch sample; the in-flight batch is not reported.')


class PreviewService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='readonly-sample-preview')
        self.lock = threading.Lock()
        self.cache = {}
        self.pending = set()

    def request(self, experiment, split, serial='auto'):
        root = (self.root/experiment).resolve()
        if root.parent != self.root:
            raise ValueError('Unknown experiment')
        pool,last,sample_state,snapshot = training_context(root)
        row,identity = select_sample(sample_state,pool,split,serial)
        if split=='validation' and row['serial']!=next(r['serial'] for r in pool if r['split']=='val'):
            snapshot = None
        version = (last['cycle'],last['global_epoch'],snapshot['step'] if snapshot else None)
        key = (experiment,version,split,row['serial'])
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            if not self.pending:
                self.pending.add(key)
                self.executor.submit(self._build,key,row,identity,last,snapshot)
        return dict(status='building',split=split,
                    message='저장된 RGB 샘플 프리뷰 준비 중 · 학습/카메라 제어 없음')

    def _build(self,key,row,identity,last,snapshot):
        experiment,version,split,serial = key
        root = self.root/experiment
        try:
            from .targets import prepare_image
            token = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
            folder = root/'preview_cache'/token/'frames'
            folder.mkdir(parents=True,exist_ok=True)
            plan = read_json(root/'campaign/plan.json')
            dataset = Path(plan['dataset_root']).resolve()
            target_path = (dataset/row['source']['target_id']).resolve()
            if not target_path.is_relative_to(dataset):
                raise ValueError('Target outside dataset')
            configs = read_json(root/'green/config.json')
            target,_ = prepare_image(target_path,configs['roi_shape'])
            fields = [('DIV2K target · '+split.upper(),target,'srgb_amplitude',0.,1.)]
            target_mse = None
            if split=='train' or snapshot is None:
                pair_path = (root/'campaign/pool'/row['file']).resolve()
                if pair_path.parent != (root/'campaign/pool').resolve():
                    raise ValueError('Sample path escapes pool')
                # These pair files are immutable; shared deletion also lets the
                # rolling pool prune them while a short byte snapshot is read.
                with np.load(io.BytesIO(shared_bytes(pair_path))) as values:
                    encoded = values['phase_rgb_u8'].copy()
                    measured = values['amplitude_rgb'].astype(np.float32)
                target_mse = float(np.mean((measured-target)**2))
                fields += [('Recorded camera · '+split.upper(),measured,'srgb_amplitude',0.,1.),
                           ('Displayed SLM codes · RGB',encoded.astype(np.float32)/255,'rgb',0.,1.),
                           ('Camera–target amplitude error',np.abs(measured-target),'rgb',0.,.5)]
            images = []
            for index,(label,values,mode,lo,hi) in enumerate(fields):
                if mode=='srgb_amplitude':
                    intensity = np.clip(values,0,1)**2
                    display = np.where(intensity<=.0031308,12.92*intensity,1.055*intensity**(1/2.4)-.055)
                else:
                    display = np.clip((values-lo)/(hi-lo),0,1)
                image = Image.fromarray((display*255).astype(np.uint8))
                image.thumbnail((640,384),Image.Resampling.BILINEAR)
                path = folder/f'{index}.png';image.save(path)
                images.append(dict(label=label,file=str(path.relative_to(root)).replace('\\','/'),
                    cmap=mode,vmin=lo,vmax=hi,clipped_fraction=float(np.mean((values<lo)|(values>hi)))))
            metrics = last
            if split=='validation' and snapshot:
                images += snapshot['images']
                metrics = snapshot['metrics']
            result = dict(status='ready',split=split,identity=identity,serial=serial,
                target_id=row['source']['target_id'],epoch=metrics['epoch'],
                global_epoch=metrics['global_epoch'],cycle=metrics['cycle'],
                sample_time=metrics['time'],rendered_at=time.time(),live_batch=False,
                metrics=metrics,images=images,target_reconstruction_mse=target_mse,
                note='최근 완료 epoch에서 실제 사용된 train 샘플 · 순간 배치 추적 아님' if split=='train' else '고정 validation 샘플 · 학습 프로세스가 저장한 프리뷰')
        except Exception as exc:
            result = dict(status='unavailable',split=split,message=str(exc))
        with self.lock:
            self.cache[key] = result
            self.pending.discard(key)
            while len(self.cache)>8:
                old_key = next(iter(self.cache));old = self.cache.pop(old_key)
                for image in old.get('images',[]):
                    path = self.root/old_key[0]/image['file']
                    if path.suffix=='.png' and path.parent.parent.parent==self.root/old_key[0]/'preview_cache':
                        try:path.unlink(missing_ok=True)
                        except OSError:pass


_managers = {}
_lock = threading.Lock()


def manager(root):
    key = str(Path(root).resolve())
    with _lock:
        if key not in _managers:
            _managers[key] = PreviewService(key)
        return _managers[key]
