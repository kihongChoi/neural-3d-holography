"""Single-plane band-limited ASM, retaining the physical SLM pixel pitch."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def crop_center(value, shape):
    h, w = value.shape[-2:]
    rh, rw = shape
    if rh > h or rw > w:
        raise ValueError('ROI exceeds field size')
    return value[..., (h-rh)//2:(h-rh)//2+rh, (w-rw)//2:(w-rw)//2+rw]


def quantized_phase(phase):
    """8-bit LUT quantization with a straight-through gradient."""
    wrapped = (phase + torch.pi) % (2 * torch.pi)
    quantized = torch.round(wrapped * 255 / (2 * torch.pi)) * (2 * torch.pi / 255) - torch.pi
    return phase + (quantized - phase).detach()


class ASM(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.shape = tuple(config['slm_shape'])
        h, w = self.shape
        self.padding = (w//2, w-w//2, h//2, h-h//2)
        pitch = config['slm_pitch_m'] * abs(config['relay_magnification'])
        wavelength = config['wavelength_m']
        z = config['distance_m']
        # Calculate the phase in float64 and remove the irrelevant piston to
        # avoid loss of precision in k*z. FFTs and activations remain complex64.
        fy = np.fft.fftfreq(2*h, pitch)[:, None]
        fx = np.fft.fftfreq(2*w, pitch)[None, :]
        r2 = fy**2 + fx**2
        root = np.sqrt(np.maximum(1/wavelength**2-r2, 0))
        # Rationalized subtraction avoids cancellation near zero frequency.
        phase = -2*np.pi*z*r2/(root + 1/wavelength)
        fymax = 1/(wavelength*np.sqrt(1+(2*abs(z)/(2*h*pitch))**2))
        fxmax = 1/(wavelength*np.sqrt(1+(2*abs(z)/(2*w*pitch))**2))
        mask = ((np.abs(fx) < fxmax) & (np.abs(fy) < fymax) &
                (r2 <= config['fourier_aperture']**2 * 2*(0.5/pitch)**2))
        kernel = (np.exp(1j*phase)*mask).astype(np.complex64)
        self.register_buffer('kernel', torch.from_numpy(kernel).to(device))
        self.register_buffer('illumination', torch.ones(self.shape, device=device))

    def forward(self, phase, quantize=True):
        if tuple(phase.shape[-2:]) != self.shape:
            raise ValueError('Do not resize the physical SLM grid; reduce the loss ROI instead.')
        if quantize:
            phase = quantized_phase(phase)
        field = self.illumination * torch.exp(1j*phase.float())
        field = F.pad(field, self.padding)
        field = torch.fft.ifft2(torch.fft.fft2(field)*self.kernel)
        return crop_center(field, self.shape)


def make_target(shape, index=0, device='cpu'):
    """Smooth, extended targets avoid a high-intensity point focus."""
    h, w = shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                           torch.linspace(-1, 1, w, device=device), indexing='ij')
    if index % 3 == 0:
        a = torch.sigmoid((0.55-x.abs())*50)*torch.sigmoid((0.45-y.abs())*50)
        a = a*(0.55+0.45*torch.cos(x*7)**2)
    elif index % 3 == 1:
        r = torch.sqrt((1.4*x)**2+y**2+1e-8)
        a = torch.exp(-((r-0.50)/0.13)**2)
    else:
        a = torch.exp(-((x-0.30)**2+(y+0.15)**2)/0.065)
        a += 0.8*torch.exp(-((x+0.32)**2+(y-0.22)**2)/0.085)
    return (0.08+0.92*a).float()


def amplitude_loss(prediction, target):
    scale = ((prediction.detach()*target).mean() /
             prediction.detach().square().mean().clamp_min(1e-8))
    return (scale*prediction-target).square().mean(), scale


def optimize_phase(model, target, config, iterations=30, lr=0.04, capture=None,
                   initial=None, callback=None, trace=None, fixed_scale=None):
    """Peng/Choi CITL surrogate: camera values forward, ASM gradients backward."""
    phase = (torch.zeros(config['slm_shape'], device=target.device) if initial is None
             else initial.detach().clone()).requires_grad_(True)
    optimizer = torch.optim.Adam([phase], lr=lr)
    best_loss = float('inf')
    best_phase = phase.detach().clone()
    history = []
    for step in range(iterations+1):
        optimizer.zero_grad(set_to_none=True)
        prediction = crop_center(model(phase).abs(), target.shape)
        measured = None
        if capture is not None:
            measured = capture(phase.detach().cpu().numpy())
            measured = torch.as_tensor(measured, device=target.device, dtype=torch.float32)
            if measured.shape != target.shape or not torch.isfinite(measured).all():
                raise RuntimeError('Invalid rectified camera amplitude')
            prediction = prediction + (measured-prediction).detach()
        if fixed_scale is None:
            loss, scale = amplitude_loss(prediction, target)
        else:
            scale = float(fixed_scale)
            loss = (scale*prediction-target).square().mean()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError('Non-finite optimization loss')
        if value < best_loss:
            best_loss = value
            # Snapshot the phase that actually produced this measurement,
            # before the optimizer changes it (the upstream code aliases it).
            best_phase = phase.detach().clone()
        history.append({'step': step, 'amplitude_mse': value,
                        'psnr_db': -10*math.log10(max(value, 1e-12)), 'scale': float(scale)})
        if callback:
            callback(step, history[-1], prediction.detach().cpu().numpy()*float(scale))
        if step < iterations:
            loss.backward()
        if trace:
            trace(step, history[-1], phase.detach(), prediction.detach(),
                  None if phase.grad is None else phase.grad.detach())
        if step < iterations:
            optimizer.step()
    return best_phase.detach(), history


class CompactForwardModel(nn.Module):
    """Full-resolution physics plus a small camera-plane residual CNN.

    This is a single-plane smoke-test architecture, not the paper's large
    CNNpropCNN. Keep complex FFTs in float32; batch=1 and ROI-only CNN save VRAM.
    """
    def __init__(self, asm, roi_shape, channels=8):
        super().__init__()
        self.asm = asm
        self.roi_shape = roi_shape
        self.correction = nn.Sequential(nn.Conv2d(1, channels, 5, padding=2), nn.SiLU(),
                                        nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(channels, 1, 3, padding=1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.log_gain = nn.Parameter(torch.zeros(()))

    @property
    def kernel(self):
        """Expose the propagation device for the common optimization interface."""
        return self.asm.kernel

    def forward(self, phase):
        amp = crop_center(self.asm(phase).abs(), self.roi_shape)
        # A bounded correction preserves positive amplitude and a physics path.
        correction = self.correction(amp[None, None])[0, 0]
        return amp * torch.exp(self.log_gain + 0.5*torch.tanh(correction))


def load_forward_model(path, config, device='cpu'):
    """Reload a measured forward model for prediction or phase optimization."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'single_plane_asm_residual_cnn_8':
        raise ValueError('Unsupported forward-model architecture')
    if checkpoint.get('config') != config:
        raise ValueError('Checkpoint optics/acquisition config differs; use its original config or retrain.')
    model = CompactForwardModel(ASM(config, device), config['roi_shape']).to(device)
    model.correction.load_state_dict(checkpoint['correction'], strict=True)
    with torch.no_grad():
        model.log_gain.copy_(checkpoint['log_gain'])
        model.asm.illumination.copy_(checkpoint['illumination'])
    # Freeze network weights, while preserving gradients with respect to phase.
    model.eval().requires_grad_(False)
    return model
