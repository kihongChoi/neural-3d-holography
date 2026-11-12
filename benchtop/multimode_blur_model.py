"""Pre-registered K1 detector-blur control; no changes to the main study."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .integrated_rgb_train import CameraHead
from .multimode_model import MultimodeRGB


class GaussianCameraHead(CameraHead):
    def __init__(self):
        super().__init__()
        self.raw_sigma=nn.Parameter(torch.full((3,),math.log(.3/(4.-.3))))

    def sigma(self):
        return 4*self.raw_sigma.sigmoid()

    def kernel(self):
        x=torch.arange(-8,9,device=self.raw_sigma.device,dtype=self.raw_sigma.dtype)
        kernel=torch.exp(-.5*(x[None,:]/self.sigma()[:,None].clamp_min(1e-6)).square())
        return kernel/kernel.sum(-1,keepdim=True)

    def blur(self,intensity):
        k=self.kernel()
        x=F.conv2d(F.pad(intensity[None],(8,8,0,0),mode='reflect'),k[:,None,None,:],groups=3)
        return F.conv2d(F.pad(x,(0,0,8,8),mode='reflect'),k[:,None,:,None],groups=3)[0]

    def forward(self,intensity):
        return super().forward(self.blur(intensity))


class BlurMultimodeRGB(MultimodeRGB):
    def __init__(self,rank=1,device='cpu',configs=None,recompute=True):
        if rank!=1:raise ValueError('Registered blur control is K1 only')
        super().__init__(rank,device,configs,recompute)
        self.head=GaussianCameraHead().to(device)

    def diagnostics(self):
        return dict(mode_spectrum=super().diagnostics(),sigma_pixels=self.head.sigma().detach().cpu().tolist())
