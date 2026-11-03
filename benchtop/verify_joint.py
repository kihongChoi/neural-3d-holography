"""Short, isolated checks using recorded real pairs and one real RGB CITL target."""
import ctypes
import hashlib
from pathlib import Path
import shutil
import time

import torch

from .hardware import Bench
from .live import atomic_json
from .rgb_run import CHANNELS, read_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_verification(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Verification output must be new; original measurements are preserved')
    state = read_json(source/'campaign/status.json')
    if state['status'] in ('running', 'starting'):
        raise RuntimeError('Stop the source hardware campaign before verification')
    pool = read_json(source/'campaign/pool.json')
    selected = [r for r in pool if r['split'] == 'train'][-4:]+[r for r in pool if r['split'] == 'val'][:2]
    if len(selected) != 6:
        raise ValueError('Verification requires four real training and two validation pairs')
    output.mkdir(parents=True)
    directory = output/'campaign'
    (directory/'pool').mkdir(parents=True)
    (directory/'models').mkdir()
    paths = ['completed_prepare.json', 'campaign/plan.json', 'campaign/pool.json', 'campaign/status.json']
    for ch in CHANNELS:
        paths += [f'{ch}/config.json', f'{ch}/calibration/calibration.json',
                  f'{ch}/calibration/flat_intensity.npy', f'{ch}/calibration/dark_intensity.npy',
                  f'campaign/models/{ch}_best.pt']
    paths += ['campaign/pool/'+row['file'] for row in selected]
    hashes = {relative:digest(source/relative) for relative in paths}
    for relative in paths:
        if relative in ('campaign/pool.json', 'campaign/status.json'):
            continue
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    atomic_json(directory/'pool.json', selected)
    atomic_json(output/'verification_source.json', dict(source=str(source), hashes=hashes,
        note='Subset of recorded real RGB pairs. Original run is not resumed or modified.'))


def adam_steps(checkpoint):
    return [int(value['step']) for value in checkpoint['optimizer']['state'].values()]


def run_verification(worker):
    w = worker
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    try:
        if w.args.verify_stage != 'resume':
            w.fit()
        first = torch.load(w.directory/'models/rgb_latest.pt', map_location='cpu', weights_only=True)
        first_steps = adam_steps(first)
        del first
        if w.args.verify_stage == 'train':
            atomic_json(w.root/'verification_first_stage.json', dict(adam_steps=first_steps))
            w.state.update(status='stopped', stop_reason='verification_checkpoint', active_seconds=w.elapsed)
            w.progress('RGB 검증 1단계 종료 · 프로세스 재시작 후 Adam 복원 검사 대기')
            w.monitor.finish(reason='verification_checkpoint')
            return
        w.state['cycle'] += 1
        w.fit()  # Reload from disk, including all Adam moments and scheduler state.
        second = torch.load(w.directory/'models/rgb_latest.pt', map_location='cpu', weights_only=True)
        second_steps = adam_steps(second)
        assert first_steps and len(first_steps) == len(second_steps)
        assert all(b > a for a,b in zip(first_steps, second_steps)), 'Adam steps were reset'
        assert second['global_epoch'] == 2*w.args.epochs
        del second
        with Bench(w.base) as bench:
            w.drift_check(bench)
            result = w.evaluate(bench, w.plan['tracking'][0])
            w.drain_frames(bench, 'verification_end')
        provenance = read_json(w.root/'verification_source.json')
        unchanged = all(digest(Path(provenance['source'])/relative) == sha for relative,sha in provenance['hashes'].items())
        assert unchanged, 'Source run changed during isolated verification'
        w.state.update(status='completed', finished_at=time.time(), verification=True,
                       active_seconds=w.elapsed, stop_reason='short_verification')
        w.progress('짧은 RGB 공동 학습·실측 검증 완료 · 장시간 학습 재개 안 함')
        atomic_json(w.root/'verification.json', dict(status='completed', arguments=vars(w.args),
            first_adam_steps=first_steps, resumed_adam_steps=second_steps,
            original_run_unchanged=unchanged, evaluation=result, memory=w.state['memory'],
            storage=w.budget.check(True), elapsed_seconds=w.elapsed))
        w.monitor.finish(reason='short_verification')
    except BaseException as exc:
        w.state.update(status='stopped' if isinstance(exc, InterruptedError) else 'failed',
                       error=str(exc), error_type=type(exc).__name__, active_seconds=w.elapsed)
        atomic_json(w.status_file, w.state)
        w.monitor.finish(error=exc)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
