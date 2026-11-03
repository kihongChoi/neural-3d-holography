"""Paper-sized CNNpropCNN, using the authors' unchanged U-Net implementation.

Single calibrated plane is a restricted experiment, not the paper's 3D result.
The returned phase is from ASM BEFORE the target CNN, as in prop_model.py.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from unet import UnetGenerator, init_weights
from .optics import ASM, crop_center, quantized_phase


def pad_to(value, shape):
    h, w = value.shape[-2:]
    dh, dw = shape[0]-h, shape[1]-w
    return F.pad(value, (dw//2, dw-dw//2, dh//2, dh-dh//2))


class PaperCNNpropCNN(nn.Module):
    architecture = 'authors_unet_Cs8_32_512_Ct5_8_128_rect_single_plane'

    def __init__(self, config, device='cpu', recompute=True):
        super().__init__()
        self.config = config
        self.asm = ASM(config, device)
        self.recompute = recompute
        self.slm_cnn = UnetGenerator(2, 2, 8, 32, 512, nn.InstanceNorm2d, outer_skip=True)
        self.target_cnn = UnetGenerator(2, 2, 5, 8, 128, nn.InstanceNorm2d, outer_skip=True)
        init_weights(self.slm_cnn, 'normal')
        init_weights(self.target_cnn, 'normal')
        self.to(device)

    def correct(self, field, net, downs):
        shape = tuple(math.ceil(s/2**downs)*2**downs for s in self.asm.shape)
        field = pad_to(field, shape)
        channels = torch.stack((field.real, field.imag), 0)[None]
        if self.recompute and torch.is_grad_enabled():
            channels = checkpoint(net, channels, use_reentrant=False)
        else:
            channels = net(channels)
        return crop_center(torch.complex(channels[0, 0], channels[0, 1]), self.asm.shape)

    def forward(self, phase, quantize=True):
        if tuple(phase.shape) != self.asm.shape:
            raise ValueError('CNNpropCNN requires the full physical SLM grid')
        phase = quantized_phase(phase) if quantize else phase
        field = self.correct(torch.exp(1j*phase.float()), self.slm_cnn, 8)
        field = crop_center(torch.fft.ifft2(torch.fft.fft2(F.pad(field, self.asm.padding))*self.asm.kernel), self.asm.shape)
        corrected = self.correct(field, self.target_cnn, 5)
        return torch.polar(corrected.abs(), field.angle())

    def learnable_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('asm.')}

    def load_learnable(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if set(missing) != {'asm.kernel', 'asm.illumination'} or unexpected:
            raise ValueError((missing, unexpected))


def dpac(model, target):
    """Single-plane double phase encoding of a back-propagated target field.

    Checkerboard interleaves angle(u) +/- acos(|u|/max|u|). The aperture's
    conjugate is an adjoint, not an inverse of the blocked frequencies.
    """
    field = pad_to(target.to(torch.complex64), model.shape)
    field = crop_center(torch.fft.ifft2(torch.fft.fft2(F.pad(field, model.padding))*model.kernel.conj()), model.shape)
    field = field/model.illumination.clamp_min(0.05)
    offset = torch.acos((field.abs()/field.abs().max().clamp_min(1e-8)).clamp(0, 1))
    y = torch.arange(model.shape[0], device=field.device)[:, None]
    x = torch.arange(model.shape[1], device=field.device)[None, :]
    return field.angle()+torch.where((y+x)%2 == 0, offset, -offset)


def mse_metrics(prediction, target):
    """Fixed-white AND paper-style best scalar amplitude results; no hidden fit."""
    p, t = prediction.float(), target.float()
    scale = (p*t).mean()/p.square().mean().clamp_min(1e-12)
    fixed = float((p-t).square().mean())
    fitted = float((scale*p-t).square().mean())
    return dict(fixed_white_mse=fixed, fitted_amplitude_mse=fitted,
                fitted_amplitude_psnr=-10*math.log10(max(fitted, 1e-12)),
                fitted_amplitude_scale=float(scale))


def image_metrics(prediction, target):
    """Explicit display-space metrics in addition to linear-amplitude metrics."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    metrics = mse_metrics(prediction, target)
    scale = metrics['fitted_amplitude_scale']
    def display(a):
        intensity = np.clip(a**2, 0, 1)
        return np.where(intensity<=.0031308,12.92*intensity,1.055*intensity**(1/2.4)-.055)
    p = display(prediction.cpu().numpy()*scale)
    t = display(target.cpu().numpy())
    mse = float(np.mean((p-t)**2))
    mu_p,mu_t=gaussian_filter(p,1.5,truncate=3.5),gaussian_filter(t,1.5,truncate=3.5)
    vp=gaussian_filter(p*p,1.5,truncate=3.5)-mu_p**2
    vt=gaussian_filter(t*t,1.5,truncate=3.5)-mu_t**2
    cov=gaussian_filter(p*t,1.5,truncate=3.5)-mu_p*mu_t
    ssim=((2*mu_p*mu_t+.01**2)*(2*cov+.03**2))/((mu_p**2+mu_t**2+.01**2)*(vp+vt+.03**2))
    metrics.update(fitted_clipped_srgb_psnr=-10*math.log10(max(mse,1e-12)),
                   fitted_clipped_srgb_ssim=float(ssim[5:-5,5:-5].mean()))
    return metrics
