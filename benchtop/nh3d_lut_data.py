"""Read committed, linear camera training pairs; never train on target images."""
import json
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .nh3d_lut_model import DEPTHS, optics, sha
from .targets import prepare_image


class CameraDataset:
    def __init__(self, root, verify=True):
        self.root = Path(root).resolve()
        self.manifest_path = self.root/'model_dataset.json'
        self.fingerprint = sha(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest['version'] != 'nh3d-camera-v4' or not self.manifest['complete']:
            raise ValueError('Require a complete v4 camera dataset')
        if self.manifest.get('source_type')!='physical_camera':
            raise ValueError('Synthetic arrays cannot be presented as physical camera data')
        if self.manifest.get('optics')!={tag:optics(tag) for tag in 'RGB'}:
            raise ValueError('Optical configuration mismatch')
        if datetime.fromisoformat(self.manifest['created_kst']).date()<date(2026,11,5):
            raise ValueError('Require fresh data from the November 5 restart or later')
        if verify:
            for ref in self.manifest['references']:
                if sha(self.path(ref['path']))!=ref['sha256']:raise ValueError('Reference hash mismatch')
            status_path=self.root/'model_acquisition/status.json'
            if sha(status_path)!=self.manifest['session_status_sha256']:
                raise ValueError('Acquisition status changed')
            status=json.loads(status_path.read_text(encoding='utf-8'))
            if status.get('state')!='camera_training_bank_complete' or status.get('error'):
                raise ValueError('Acquisition did not finish successfully')
            if not all(status['routing'].get(tag,{}).get('accepted') for tag in ('R','G','B','RGB')):
                raise ValueError('Missing routing validation')
            if len(status['calibrations'])!=16 or not all(c['registration']['accepted'] for c in status['calibrations']):
                raise ValueError('Missing or failed fresh geometry calibration')
            if any(abs(d['fraction'])>.03 for d in status['white_drift']):
                raise ValueError('Failed white stability gate')
        baseline=self.manifest.get('baseline_variant','identity')
        if baseline not in ('identity','current_lut'):raise ValueError('Unknown baseline variant')
        variants=sorted([baseline,'probe_plus','probe_minus'])
        tables=None
        if baseline=='current_lut':
            path=self.root/'drive_tables.json'
            if sha(path)!=self.manifest['drive_tables_sha256']:raise ValueError('Changed update probes')
            tables=json.loads(path.read_text())
        self.rows = self.manifest['records']
        if len(self.rows) != self.manifest['expected_capture_count'] or not self.rows:
            raise ValueError('Incomplete camera pair count')
        if len({r['name'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate camera pair names')
        seen = {}
        groups = {}
        for r in self.rows:
            if r['split'] not in ('fit','selection') or r['depth_mm'] not in DEPTHS:
                raise ValueError('Test scenes or held-out depths cannot enter training/selection')
            if r['channel'] not in 'RGB' or len(r['channel']) != 1:
                raise ValueError('Slot-resolved RGB channel required')
            if seen.setdefault(r['scene'],r['split']) != r['split']:
                raise ValueError('Scene leakage')
            if r['variant'] not in variants:
                raise ValueError('Unknown response probe')
            key=(r['scene'],r['channel'],r['depth_mm'])
            group=groups.setdefault(key,[])
            group.append(r['variant'])
            table = np.asarray(r['drive_lut'])
            if tables is not None:
                index=str(int(Path(r['pattern_path']).stem))
                expected=tables[index][r['variant']]['RGB'.index(r['channel'])]
                if r['drive_lut']!=expected:raise ValueError('Update drive table mismatch')
            if table.shape != (256,) or np.any(table != np.rint(table)) or table.min()<0 or table.max()>255:
                raise ValueError('Invalid drive table')
            if verify:
                for key in ('array','camera','pattern'):
                    if sha(self.path(r[key+'_path'])) != r[key+'_sha256']:
                        raise ValueError(f'{key} hash mismatch: {r["name"]}')
                source = Path(r['target_source'])
                if sha(source) != r['target_sha256']:
                    raise ValueError('Target source changed')
        if any(sorted(v)!=variants for v in groups.values()):
            raise ValueError('Every scene/channel/depth requires all three distinct probes')
        for scene in seen:
            if sum(k[0]==scene for k in groups)!=9:
                raise ValueError('Scene missing a depth or color')
        for split in ('fit','selection'):
            for tag in 'RGB':
                for z in DEPTHS:
                    if not any(r['split']==split and r['channel']==tag and r['depth_mm']==z for r in self.rows):
                        raise ValueError(f'Missing {split}/{tag}/{z} group')

    def path(self, name):
        p=(self.root/name).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Dataset path escapes its run')
        return p

    @lru_cache(maxsize=12)
    def pattern(self, relative):
        array=np.array(Image.open(self.path(relative)))
        if array.dtype != np.uint8 or array.shape != (960,1728,3):
            raise ValueError('Native RGB uint8 DPAC PNG required')
        return array

    def load(self, row, device):
        base=self.pattern(row['pattern_path'])[...,'RGB'.index(row['channel'])]
        codes=np.asarray(row['drive_lut'],dtype=np.uint8)[base]
        if __import__('hashlib').sha256(codes.tobytes()).hexdigest()!=row['displayed_channel_sha256']:
            raise ValueError('Measured code identity mismatch')
        with np.load(self.path(row['array_path']),allow_pickle=False) as f:
            amp=f['amplitude'].copy(); mask=f['valid'].copy()
        if amp.shape!=(960,1728) or mask.shape!=amp.shape or mask.dtype!=np.bool_:
            raise ValueError('Invalid linear camera array or mask')
        if mask.mean()<.98 or not np.isfinite(amp).all() or np.min(amp)<0:
            raise ValueError('Invalid measured amplitude')
        return (torch.as_tensor(codes,device=device),torch.as_tensor(amp,device=device,dtype=torch.float32),
                torch.as_tensor(mask,device=device))

    def original(self,row,device):
        return torch.as_tensor(self.pattern(row['pattern_path'])[...,'RGB'.index(row['channel'])].copy(),device=device)

    @lru_cache(maxsize=8)
    def target_array(self,path):
        return prepare_image(Path(path),(960,1728))[0]

    def target(self,row,device):
        # Only for student distillation/evaluation, not teacher training labels.
        a=self.target_array(row['target_source'])[...,'RGB'.index(row['channel'])]
        return torch.as_tensor(a.copy(),device=device)

    def check_unchanged(self):
        if sha(self.manifest_path)!=self.fingerprint:
            raise RuntimeError('Dataset manifest changed during training')
