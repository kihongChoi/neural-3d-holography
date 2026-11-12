"""Low-rank mutual-intensity calibration on the native SLM grid.

Original compact-grid implementation, not a reproduction of the SAWH MLP.
Modes are shared across scenes/depths and combined only after detection.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .optics import ASM, crop_center
from .nh3d_lut_model import optics


def interpolate(value, shape):
    return F.interpolate(value, size=shape, mode='bilinear', align_corners=False)


def lookup(table, codes):
    x = codes.float().clamp(0, 255) * ((table.numel()-1)/255)
    lo = x.floor().long().clamp(max=table.numel()-2)
    return torch.lerp(table[lo], table[lo+1], x-lo)


def propagate(field, kernel, padding, shape):
    return crop_center(torch.fft.ifft2(torch.fft.fft2(F.pad(field, padding))*kernel), shape)


class ModeBranch(nn.Module):
    def __init__(self, config, rank, device='cpu', mode_shape=(60,108), recompute=True):
        super().__init__()
        if rank not in (1,3,6):
            raise ValueError('Pre-registered ranks are 1,3,6')
        self.rank, self.shape, self.recompute = rank, tuple(config['slm_shape']), recompute
        self.propagators = nn.ModuleDict({str(z): ASM(dict(config, distance_m=z/1000), device)
                                         for z in (3,5,7,9)})
        gh, gw = min(mode_shape[0],self.shape[0]), min(mode_shape[1],self.shape[1])
        self.source_amp = nn.Parameter(torch.zeros(rank,1,gh,gw))
        self.source_phase = nn.Parameter(.002*torch.randn(rank,1,gh,gw))
        with torch.no_grad(): self.source_phase[0].zero_()
        angle = torch.arange(rank, dtype=torch.float32)*2*math.pi/max(rank-1,1)
        # The short 3-9 mm propagation needs resolvable angular diversity.
        # 24 cycles/aperture is about 0.5-1 pixel shear at 3 mm (green).
        tilts = torch.stack((angle.cos(),angle.sin()),1)*24.
        tilts[0].zero_()
        self.tilt_cycles = nn.Parameter(tilts)
        weights = torch.ones(rank) if rank==1 else torch.tensor([.95]+[.05/(rank-1)]*(rank-1))
        self.mode_logits = nn.Parameter(weights.log())
        self.phase_steps = nn.Parameter(torch.zeros(32))
        self.phase_span = nn.Parameter(torch.zeros(()))
        self.code_amp = nn.Parameter(torch.zeros(17))
        self.pupil_phase = nn.Parameter(torch.zeros(1,1,15,27))
        self.dc = nn.Parameter(torch.zeros(rank,2,15,27))
        yy, xx = torch.meshgrid((torch.arange(self.shape[0])-(self.shape[0]-1)/2)/self.shape[0],
                                (torch.arange(self.shape[1])-(self.shape[1]-1)/2)/self.shape[1], indexing='ij')
        self.register_buffer('xx',xx, persistent=False)
        self.register_buffer('yy',yy, persistent=False)
        self.to(device)

    def phase_table(self):
        increments=F.softplus(self.phase_steps)
        cumulative=F.pad(increments.cumsum(0),(1,0))/increments.sum()
        return math.pi-2*math.pi*(1+.5*self.phase_span.tanh())*cumulative

    def sources(self):
        amp=(.5*interpolate(self.source_amp,self.shape)[:,0].tanh()).exp()
        amp=amp/(amp.square().mean((-2,-1),keepdim=True)+1e-12).sqrt()
        phase=math.pi*interpolate(self.source_phase,self.shape)[:,0].tanh()
        # Bounded below the sampled SLM Nyquist frequency, with learnable tilts.
        tilt=128*(self.tilt_cycles/128).tanh()
        phase=phase+2*math.pi*(tilt[:,0,None,None]*self.xx+tilt[:,1,None,None]*self.yy)
        return torch.polar(amp,phase)

    def intensity(self, codes, depth):
        if tuple(codes.shape)!=self.shape or float(depth) not in (3.,5.,7.,9.):
            raise ValueError('Native grid and configured physical depth required')
        asm=self.propagators[str(int(depth))]
        field=torch.polar(lookup((.5*self.code_amp.tanh()).exp(),codes),lookup(self.phase_table(),codes))
        src=self.sources()
        dc=.2*interpolate(self.dc.tanh(),self.shape)
        dc=torch.complex(dc[:,0],dc[:,1])
        phase=2*math.pi*interpolate(self.pupil_phase.tanh(),asm.kernel.shape)[0,0]
        kernel=asm.kernel*torch.exp(1j*torch.fft.ifftshift(phase))
        weights=self.mode_logits.softmax(0)
        result=torch.zeros_like(codes,dtype=torch.float32)
        for k in range(self.rank):
            incident=src[k]*field+dc[k]
            def detect(u,h):
                return propagate(u,h,asm.padding,self.shape).abs().square()
            value=checkpoint(detect,incident,kernel,use_reentrant=False) if self.recompute and torch.is_grad_enabled() else detect(incident,kernel)
            result=result+weights[k]*value
        return result

    @torch.no_grad()
    def spectrum(self):
        source=(self.sources()*self.mode_logits.softmax(0).sqrt()[:,None,None]).flatten(1)
        gram=source@source.conj().T/source.shape[1]
        eig=torch.linalg.eigvalsh(gram).real.clamp_min(0).flip(0)
        eig=eig/eig.sum().clamp_min(1e-12)
        return dict(normalized_eigenvalues=eig.cpu().tolist(),effective_rank=float(1/eig.square().sum()))


class MultimodeRGB(nn.Module):
    def __init__(self,rank,device='cpu',configs=None,recompute=True):
        super().__init__()
        from .integrated_rgb_train import CameraHead
        self.rank=rank
        self.branches=nn.ModuleList([ModeBranch(cfg,rank,device,recompute=recompute)
                                    for cfg in (configs or [optics(c) for c in 'RGB'])])
        self.head=CameraHead().to(device)

    def forward(self,codes,z,ideal=False):
        if ideal:
            raise ValueError('Use the registered ASM baseline for ideal evaluation')
        return self.head(torch.stack([m.intensity(codes[i],z) for i,m in enumerate(self.branches)]))

    def weights(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if '.propagators.' not in k}

    def load_weights(self,state):
        expected={k for k in self.state_dict() if '.propagators.' not in k}
        if set(state)!=expected: raise ValueError('Mode model state mismatch')
        self.load_state_dict(state,strict=False)

    def diagnostics(self):
        return {c:b.spectrum() for c,b in zip('RGB',self.branches)}
