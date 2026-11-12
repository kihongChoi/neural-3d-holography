"""Hardware-free invariants for camera-model training and LUT distillation."""
import unittest
import json
import tempfile
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from benchtop.nh3d_lut_model import CodeLUT, DepthSharedNH3D, decode_codes, optics, probe_tables
from benchtop.nh3d_lut_data import CameraDataset
from benchtop.nh3d_lut_ab import fixed_quality, validate_candidate
from benchtop.paper_model import PaperCNNpropCNN
from benchtop.optics import ASM


class CodeTests(unittest.TestCase):
    def test_identity_and_boundaries_and_gradient(self):
        student=CodeLUT()
        codes=torch.arange(256,dtype=torch.uint8)
        self.assertTrue(torch.equal(student(codes),codes.float()))
        student(codes)[50:180].square().mean().backward()
        self.assertTrue(torch.isfinite(student.delta.grad).all())
        self.assertGreater(float(student.delta.grad.abs().sum()),0)
        for seed in range(10):
            with torch.no_grad():student.delta.copy_(torch.randn(7,generator=torch.Generator().manual_seed(seed))*10)
            table=student.export().astype(int)
            self.assertEqual((table[0],table[-1]),(0,255))
            self.assertTrue(np.all(np.diff(table)>=0))
            self.assertLessEqual(np.max(np.abs(table-np.arange(256))),12)

    def test_probe_reproducibility_and_inverted_code_direction(self):
        a=probe_tables('scene',7);b=probe_tables('scene',7)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertFalse(np.array_equal(a['probe_plus'],a['identity']))
        self.assertTrue(torch.all(torch.diff(decode_codes(torch.arange(256)))<0))
        self.assertAlmostEqual(float(decode_codes(torch.tensor(128))),-np.pi/255,places=6)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for full NH3D integration')
class ModelTests(unittest.TestCase):
    def test_shared_depth_matches_existing_teacher_and_lut_gradient(self):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.manual_seed(9)
        config=dict(optics('G'),slm_shape=[256,512])
        model=DepthSharedNH3D(config,'cuda').eval().requires_grad_(False)
        codes=torch.randint(0,256,(256,512),device='cuda',dtype=torch.uint8)
        with torch.no_grad():
            expected=PaperCNNpropCNN.forward(model,decode_codes(codes),quantize=False).abs()
            torch.testing.assert_close(model.amplitude(codes,3),expected,rtol=2e-5,atol=2e-6)
            p3=model.amplitude(codes,3)
            p7=model.amplitude(codes,7)
            self.assertGreater(float((p3-p7).abs().mean()),1e-7)
            model.asm=ASM(dict(config,distance_m=.007),'cuda')
            expected7=PaperCNNpropCNN.forward(model,decode_codes(codes),quantize=False).abs()
            torch.testing.assert_close(p7,expected7,rtol=2e-5,atol=2e-6)
        student=CodeLUT().cuda()
        model.amplitude(student(codes),9).square().mean().backward()
        self.assertTrue(torch.isfinite(student.delta.grad).all())
        self.assertGreater(float(student.delta.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):model.amplitude(codes,5)
        state=model.learnable_state()
        self.assertFalse(any(k.startswith(('asm.','propagators.')) for k in state))
        model.load_learnable(state)
        del model;torch.cuda.empty_cache()



class DataIsolationTests(unittest.TestCase):
    def spec(self):
        records=[]
        for split,scene in [('fit','train_scene'),('selection','selection_scene')]:
            for tag in 'RGB':
                for z in (3.,7.,9.):
                    for variant,table in probe_tables(scene,z).items():
                        records.append(dict(name=f'{scene}_{tag}_{z}_{variant}',scene=scene,split=split,
                            channel=tag,depth_mm=z,variant=variant,drive_lut=table.tolist()))
        return dict(version='nh3d-camera-v4',source_type='physical_camera',complete=True,
                    created_kst='2026-11-05T12:00:00+09:00',optics={t:optics(t) for t in 'RGB'},
                    expected_capture_count=len(records),records=records)

    def test_reject_leakage_heldout_and_missing_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model_dataset.json'
            valid=self.spec();path.write_text(json.dumps(valid))
            CameraDataset(tmp,verify=False)
            mutations=[]
            bad=deepcopy(valid);bad['records'][-1]['split']='fit';mutations.append(bad)
            bad=deepcopy(valid);bad['records'][0]['depth_mm']=5.;mutations.append(bad)
            bad=deepcopy(valid);bad['records'].pop();bad['expected_capture_count']-=1;mutations.append(bad)
            bad=deepcopy(valid);bad['source_type']='synthetic';mutations.append(bad)
            bad=deepcopy(valid);bad['created_kst']='2026-11-03T12:00:00+09:00';mutations.append(bad)
            for spec in mutations:
                path.write_text(json.dumps(spec))
                with self.assertRaises(ValueError):CameraDataset(tmp,verify=False)



class ABMetricTests(unittest.TestCase):
    def test_fixed_brightness_is_not_refit(self):
        target=np.random.default_rng(5).uniform(.1,.5,(64,64)).astype(np.float32)
        valid=np.ones(target.shape,dtype=bool)
        equal=fixed_quality(target,target,valid)
        brighter=fixed_quality(1.2*target,target,valid)
        self.assertEqual(equal['fixed_white_amplitude_mse'],0)
        self.assertAlmostEqual(equal['fixed_clipped_srgb_ssim'],1.,places=6)
        self.assertGreater(brighter['fixed_white_amplitude_mse'],0)
        self.assertLess(brighter['fixed_clipped_srgb_psnr_db'],equal['fixed_clipped_srgb_psnr_db'])

    def test_direct_search_lut_is_rejected_as_model_distilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'direct.npz'
            np.savez(p,table=np.tile(np.arange(256,dtype=np.uint8),(3,1)),metadata=json.dumps({'origin':'direct search'}))
            with self.assertRaises(ValueError):validate_candidate(Path(tmp),p)

if __name__=='__main__':unittest.main()
