import unittest
import torch
from benchtop.multimode_blur_model import GaussianCameraHead,BlurMultimodeRGB
from benchtop.multimode_model import MultimodeRGB
from benchtop.nh3d_lut_model import optics


class BlurTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_normalized_constant_and_nonnegative(self):
        head=GaussianCameraHead()
        with torch.no_grad():head.raw_sigma.copy_(torch.tensor([-2.,0.,2.]))
        self.assertTrue(torch.allclose(head.kernel().sum(-1),torch.ones(3)))
        x=torch.ones(3,24,40)*torch.tensor([.2,.5,.8])[:,None,None]
        self.assertTrue(torch.allclose(head.blur(x),x,atol=1e-6))
        self.assertTrue((head(torch.rand_like(x))>=0).all())

    def test_sigma_gradient_matches_finite_difference(self):
        torch.manual_seed(7);head=GaussianCameraHead().double()
        x=torch.rand(3,24,40,dtype=torch.float64);target=torch.rand_like(x)
        loss=lambda:(head(x)-target).square().mean()
        loss().backward();grad=head.raw_sigma.grad.clone()
        self.assertTrue(torch.isfinite(grad).all());self.assertTrue((grad.abs()>1e-7).all())
        for i in range(3):
            with torch.no_grad():head.raw_sigma[i]+=1e-5
            plus=float(loss().detach())
            with torch.no_grad():head.raw_sigma[i]-=2e-5
            minus=float(loss().detach())
            with torch.no_grad():head.raw_sigma[i]+=1e-5
            self.assertAlmostEqual(grad[i].item(),(plus-minus)/2e-5,places=7)

    def test_identity_limit_and_state_roundtrip(self):
        cfg=dict(optics('G'),slm_shape=[24,40]);torch.manual_seed(117)
        a=MultimodeRGB(1,configs=[cfg]*3,recompute=False)
        torch.manual_seed(117);b=BlurMultimodeRGB(configs=[cfg]*3,recompute=False)
        x=torch.randint(0,256,(3,24,40),dtype=torch.uint8)
        with torch.no_grad():b.head.raw_sigma.fill_(-20)
        self.assertTrue(torch.allclose(a(x,3),b(x,3,False),atol=1e-6))
        c=BlurMultimodeRGB(configs=[cfg]*3,recompute=False);c.load_weights(b.weights())
        self.assertTrue(torch.equal(b(x,7),c(x,7)))
        self.assertEqual(sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters()),3)


if __name__=='__main__':unittest.main()
