"""Apply a packaged RGB LUT exactly once to original uint8 DPAC codes."""
from pathlib import Path
import hashlib,json
import numpy as np

def load_candidate(path=None):
    path=Path(path) if path else Path(__file__).with_name('rgb_lut_candidate.npz')
    expected=json.loads(path.with_name('checksums.json').read_text())[path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Artifact hash mismatch')
    with np.load(path,allow_pickle=False) as f:
        table=f['table'].copy();meta=json.loads(str(f['metadata']))
    if table.shape!=(3,256) or table.dtype!=np.uint8:raise ValueError('Expected RGB uint8 table')
    if meta['format']!='joint-rgb-offline-lut-v1':raise ValueError('Unsupported provenance format')
    if hashlib.sha256(table.tobytes()).hexdigest()!=meta['candidate_array_sha256']:raise ValueError('Table hash mismatch')
    if not (np.all(table[:,0]==0) and np.all(table[:,-1]==255) and np.all(np.diff(table.astype(int),axis=1)>=0) and np.max(np.abs(table.astype(int)-np.arange(256)))<=12):raise ValueError('Invalid bounded monotone LUT')
    return table,meta

def apply_rgb_lut(original_codes,table):
    if original_codes.dtype!=np.uint8 or original_codes.ndim!=3 or original_codes.shape[-1]!=3:raise ValueError('Expected HWC RGB uint8 original DPAC codes')
    if table.shape!=(3,256) or table.dtype!=np.uint8:raise ValueError('Expected RGB uint8 table')
    return np.stack([table[c,original_codes[...,c]] for c in range(3)],axis=-1)
