"""Read audited final results without opening checkpoints or controlling hardware."""
import base64
from functools import lru_cache
import hashlib
import json
from pathlib import Path

RUN = 'paper_cnnpropcnn_20261024'


def completed_data(results_root, name):
    if name != RUN:
        raise ValueError('No completed paper result for this experiment')
    root = Path(results_root).resolve().parent
    paths = [root / 'docs/learning-20261024/paper-results-data.json',
             root / 'results' / name / 'completed_all.json',
             root / 'results' / name / 'evaluation/summary.json',
             root / 'results' / name / 'training.jsonl']
    stamps = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    return _read(str(root), stamps)


@lru_cache(maxsize=2)
def _read(root, stamps):
    root = Path(root)
    data = json.loads(Path(stamps[0][0]).read_text(encoding='utf-8'))
    sources = {row['path']: row['sha256'] for row in data['sources']}
    for path, _, _ in stamps[1:]:
        path = Path(path)
        key = path.relative_to(root).as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources.get(key):
            raise ValueError('Stored result changed; refresh the audited result snapshot')
    return data


def summary(data):
    return {**data, 'targets': [{k: v for k, v in t.items() if k != 'images'}
                              for t in data['targets']]}


def image_bytes(data, target, image):
    if image not in ('target', 'dpac', 'dpac_fitted', 'sgd_asm', 'sgd_asm_fitted',
                     'sgd_cnnpropcnn', 'sgd_cnnpropcnn_fitted'):
        raise ValueError('Unknown result image')
    row = next((t for t in data['targets'] if t['id'] == target), None)
    if row is None:
        raise ValueError('Unknown evaluation target')
    prefix, payload = row['images'][image].split(',', 1)
    if prefix != 'data:image/webp;base64':
        raise ValueError('Unexpected preview format')
    return base64.b64decode(payload, validate=True)
