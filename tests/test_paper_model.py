import unittest
import numpy as np
import torch
from benchtop.paper_model import PaperCNNpropCNN, pad_to, dpac, mse_metrics
from benchtop.optics import ASM, crop_center
from benchtop.paper_campaign import canonical


def config(shape=(512,512)):
    return dict(slm_shape=list(shape),roi_shape=[480,480],slm_pitch_m=6.4e-6,
                wavelength_m=520e-9,distance_m=.015,relay_magnification=1.,fourier_aperture=.5,
                invert_phase=True)


class PaperTests(unittest.TestCase):
    def test_checkpoint_recompute_preserves_network_and_phase_gradient(self):
        torch.set_num_threads(4)
        torch.manual_seed(123)
        net=PaperCNNpropCNN(config(),recompute=False).eval().requires_grad_(False)
        self.assertEqual(sum(p.numel() for p in net.parameters()),66746156)
        p=torch.randn(512,512,requires_grad=True)
        out=net(p,quantize=False)
        loss=out.abs().square().mean()
        g1=torch.autograd.grad(loss,p)[0]
        net.recompute=True
        out2=net(p,quantize=False)
        g2=torch.autograd.grad(out2.abs().square().mean(),p)[0]
        torch.testing.assert_close(out,out2,rtol=0,atol=0)
        torch.testing.assert_close(g1,g2,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(g1).all())
        self.assertGreater(float(g1.norm()),0)

    def test_two_complex_channels_and_pre_target_phase(self):
        torch.manual_seed(42)
        net=PaperCNNpropCNN(config(),recompute=False).eval()
        p=torch.randn(512,512)
        with torch.no_grad():
            u=torch.exp(1j*p)
            inp=torch.cat((u.real[None,None],u.imag[None,None]),1)
            a=net.slm_cnn(inp)
            u=torch.complex(a[0,0],a[0,1])
            u=crop_center(torch.fft.ifft2(torch.fft.fft2(torch.nn.functional.pad(u,net.asm.padding))*net.asm.kernel),(512,512))
            b=net.target_cnn(torch.cat((u.real[None,None],u.imag[None,None]),1))
            expected=torch.polar(torch.complex(b[0,0],b[0,1]).abs(),u.angle())
            torch.testing.assert_close(net(p,quantize=False),expected)

    def test_phase_encoding_and_dpac_finite(self):
        c=config((40,64))
        p=np.linspace(-12*np.pi,12*np.pi,40*64,dtype=np.float32).reshape(40,64)
        b,r=canonical(p,c)
        b2,r2=canonical(r,c)
        np.testing.assert_array_equal(b,b2)
        np.testing.assert_array_equal(r,r2)
        a=ASM(c)
        phase=dpac(a,torch.ones(32,48)*.5)
        self.assertEqual(tuple(phase.shape),(40,64))
        self.assertTrue(torch.isfinite(phase).all())

    def test_fitted_scale_is_separate_from_fixed_white(self):
        target=torch.tensor([.1,.5,.8])
        m=mse_metrics(target/2,target)
        self.assertGreater(m['fixed_white_mse'],0)
        self.assertAlmostEqual(m['fitted_amplitude_scale'],2)
        self.assertLess(m['fitted_amplitude_mse'],1e-12)


if __name__=='__main__':
    unittest.main()
