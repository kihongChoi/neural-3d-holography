"""Depth-shared NH3D teacher and a bounded global code-LUT student.

No camera SDKs or historical run defaults. A LUT is an effective correction,
not an identified physical phase response.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .paper_model import PaperCNNpropCNN
from .optics import ASM, crop_center

KNOTS = np.array([0, 32, 64, 96, 128, 160, 192, 224, 255], dtype=np.float32)
DEPTHS = (3., 7., 9.)
WAVELENGTHS = dict(R=638e-9, G=520e-9, B=450e-9)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def optics(channel):
    return dict(slm_shape=[960,1728], slm_pitch_m=6.4e-6,
                relay_magnification=1., wavelength_m=WAVELENGTHS[channel],
                distance_m=.003, fourier_aperture=.5)


def decode_codes(codes):
    # Do not wrap +pi to -pi and then re-encode: the measured input is a code.
    return torch.pi - codes.float() * (2 * torch.pi / 255)


def probe_tables(scene, depth_mm):
    seed = int(hashlib.sha256(f'{scene}|{depth_mm:g}|nh3d-v4'.encode()).hexdigest()[:16], 16)
    delta = np.random.default_rng(seed).uniform(-12,12,7)
    result = {'identity': np.arange(256, dtype=np.uint8)}
    for name, sign in [('probe_plus',1), ('probe_minus',-1)]:
        y = KNOTS.copy()
        y[1:-1] += sign*delta
        result[name] = np.rint(np.interp(np.arange(256), KNOTS, y)).astype(np.uint8)
    return result


class DepthSharedNH3D(PaperCNNpropCNN):
    architecture = 'nh3d_shared_Cs8_32_512_ASM_depth_Ct5_8_128_v4'

    def __init__(self, config, device='cpu', depths=DEPTHS, recompute=True):
        super().__init__(config, device, recompute)
        self.propagators = nn.ModuleDict({str(float(z)).replace('.', '_'):
            ASM(dict(config, distance_m=float(z)/1000), device) for z in depths})

    def amplitude(self, codes, depth_mm):
        key = str(float(depth_mm)).replace('.', '_')
        if key not in self.propagators:
            raise ValueError(f'Depth {depth_mm} was not explicitly configured')
        asm = self.propagators[key]
        phase = decode_codes(codes)
        if tuple(phase.shape) != asm.shape:
            raise ValueError('Native physical SLM grid required')
        field = self.correct(torch.exp(1j*phase), self.slm_cnn, 8)
        field = crop_center(torch.fft.ifft2(torch.fft.fft2(F.pad(field, asm.padding))*asm.kernel), asm.shape)
        return self.correct(field, self.target_cnn, 5).abs()

    def ideal_amplitude(self, codes, depth_mm):
        key = str(float(depth_mm)).replace('.', '_')
        return self.propagators[key](decode_codes(codes), quantize=False).abs()

    def learnable_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('asm.', 'propagators.'))}

    def load_learnable(self, state):
        expected = {k for k in self.state_dict() if not k.startswith(('asm.', 'propagators.'))}
        if set(state) != expected:
            raise ValueError('Teacher architecture/weight keys mismatch')
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith(('asm.', 'propagators.')) for k in missing):
            raise ValueError((missing, unexpected))


class CodeLUT(nn.Module):
    def __init__(self, bound=12.):
        super().__init__()
        if not 0 < bound < 15.5:
            raise ValueError('Bound must preserve positive slopes')
        self.bound = float(bound)
        self.delta = nn.Parameter(torch.zeros(7))
        self.register_buffer('knots', torch.as_tensor(KNOTS))
        axis = torch.arange(256, dtype=torch.float32)
        lo = torch.searchsorted(self.knots, axis, right=True).sub(1).clamp(0,7)
        self.register_buffer('lo', lo)
        self.register_buffer('fraction', (axis-self.knots[lo])/(self.knots[lo+1]-self.knots[lo]))

    def table(self, quantize=True):
        y = self.knots + F.pad(self.bound*torch.tanh(self.delta), (1,1))
        table = torch.lerp(y[self.lo], y[self.lo+1], self.fraction)
        return table + (table.round()-table).detach() if quantize else table

    def forward(self, original_codes):
        if original_codes.dtype not in (torch.uint8, torch.int64, torch.int32):
            raise TypeError('Student input must be original discrete DPAC codes')
        return self.table()[original_codes.long()]

    def export(self):
        return self.table().detach().cpu().numpy().astype(np.uint8)


def masked_mse(predicted, measured, mask):
    if predicted.shape != measured.shape or measured.shape != mask.shape:
        raise ValueError('Prediction/measurement/mask shape mismatch')
    return (predicted[mask]-measured[mask]).square().mean()
