import unittest
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from benchtop.multimode_model import ModeBranch, MultimodeRGB, propagate
from benchtop.nh3d_lut_model import optics, decode_codes


class ModeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        self.cfg=dict(optics('G'),slm_shape=[24,40])
        self.codes=torch.randint(0,256,(24,40),dtype=torch.uint8)

    def test_rank1_reduces_to_asm(self):
        b=ModeBranch(self.cfg,1,recompute=False)
        self.assertTrue(torch.allclose(b.intensity(self.codes,3),b.propagators['3'](decode_codes(self.codes),quantize=False).abs().square(),atol=2e-6,rtol=2e-5))

    def test_unitary_basis_and_duplicate_modes(self):
        b=ModeBranch(self.cfg,3,recompute=False);a=b.propagators['3']
        modes=torch.randn(3,24,40,dtype=torch.complex64)
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.complex64))
        mixed=(q@modes.flatten(1)).reshape_as(modes)
        f=lambda u:propagate(u,a.kernel,a.padding,a.shape).abs().square().sum(0)
        self.assertTrue(torch.allclose(f(modes),f(mixed),atol=3e-6,rtol=2e-5))
        copies=modes[:1].repeat(3,1,1)/3**.5
        self.assertTrue(torch.allclose(f(copies),f(modes[:1]),atol=2e-6,rtol=2e-5))

    def test_gradients_code_and_parameters(self):
        b=ModeBranch(self.cfg,3,recompute=True)
        codes=self.codes.float().requires_grad_()
        y=b.intensity(codes,7)
        (y-torch.rand_like(y)).square().mean().backward()
        self.assertTrue(torch.isfinite(codes.grad).all())
        self.assertGreater(float(codes.grad.abs().sum()),0)
        for name,p in b.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        table=b.phase_table()
        self.assertTrue(torch.all(torch.diff(table)<0))

    def test_positive_rgb_and_depth_shared_state(self):
        m=MultimodeRGB(3,configs=[self.cfg]*3,recompute=False)
        x=self.codes.repeat(3,1,1)
        y=m(x,3)
        self.assertTrue(torch.equal(y,m(x,3,False)))
        self.assertEqual(tuple(y.shape),(3,24,40));self.assertTrue((y>=0).all())
        state=m.weights();m(x,5);self.assertEqual(set(state),set(m.weights()))
        m.load_weights(state)
        with self.assertRaises(ValueError):m(x[:,:,:20],3)
        for d in m.diagnostics().values():self.assertAlmostEqual(sum(d['normalized_eigenvalues']),1,places=5)

    def test_code_direction_matches_finite_difference(self):
        b=ModeBranch(self.cfg,3,recompute=False)
        codes=self.codes.float().clamp(2,253).requires_grad_()
        direction=torch.randn_like(codes);target=torch.rand_like(codes)
        loss=lambda x:(b.intensity(x,3)-target).square().mean()
        exact=(torch.autograd.grad(loss(codes),codes)[0]*direction).sum()
        eps=.05
        finite=(loss(codes.detach()+eps*direction)-loss(codes.detach()-eps*direction))/(2*eps)
        self.assertTrue(torch.allclose(exact,finite,atol=3e-6,rtol=.02))

    def test_manifest_rejects_sealed_depth_and_parent_leakage(self):
        from benchtop.multimode_train import validate_manifest
        p=Path('results/20261106-212946_a100-integrated-rgb-training/cache_manifest.json')
        if not p.exists():self.skipTest('Local metadata fixture unavailable')
        original=json.loads(p.read_text());validate_manifest(original)
        d=copy.deepcopy(original);d['records'][0]['depth_mm']=5.
        with self.assertRaises(ValueError):validate_manifest(d)
        d=copy.deepcopy(original);sel=next(r for r in d['records'] if r['split']=='selection');sel['scene']=d['records'][0]['scene']
        with self.assertRaises(ValueError):validate_manifest(d)

    def test_existing_evaluator_contract(self):
        from benchtop.integrated_rgb_train import evaluate
        m=MultimodeRGB(1,configs=[self.cfg]*3,recompute=False)
        codes=self.codes.repeat(3,1,1)
        target=m(codes,3).detach()
        rows=[dict(id=0,scene='synthetic',depth_mm=3.,condition='identity',reference_key='ref')]
        with patch.object(torch.Tensor,'cuda',lambda tensor,*a,**kw:tensor):
            ev=evaluate(m,[(0,codes,target)],rows,{'ref':torch.ones_like(target,dtype=torch.bool)})
        self.assertEqual(ev['parent_balanced_intensity_mse'],0.)
        self.assertTrue(all(abs(x-1)<1e-5 for x in ev['ssim_rgb']))


if __name__=='__main__':unittest.main()
