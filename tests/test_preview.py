import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image

from benchtop.preview import PreviewService, catalog, select_sample, shared_bytes


class PreviewTests(unittest.TestCase):
    def test_epoch_identity_uses_actual_deterministic_shuffle_and_excludes_test(self):
        pool = [dict(serial=5,split='val'),dict(serial=7,split='train'),
                dict(serial=8,split='train'),dict(serial=9,split='test')]
        checkpoint = dict(global_epoch=3,sample_offset=0,branches={'red':{'training_sample_serials':[7,8]}})
        expected = np.random.default_rng(44).permutation([7,8])[-1]
        row, identity = select_sample(checkpoint,pool,'train')
        self.assertEqual(row['serial'],expected)
        self.assertEqual(identity,'last_completed_train_sample')
        self.assertEqual(select_sample(checkpoint,pool,'validation')[0]['serial'],5)
        with self.assertRaises(ValueError):
            select_sample(checkpoint,pool,'train','9')
        with self.assertRaises(ValueError):
            select_sample(checkpoint,pool,'test')
        checkpoint['sample_offset'] = 1
        expected = np.random.default_rng(45).permutation([7,8])[0]
        self.assertEqual(select_sample(checkpoint,pool,'train')[0]['serial'],expected)

    @unittest.skipUnless(os.name=='nt','Windows shared deletion')
    def test_readonly_pair_reader_allows_rolling_pool_deletion(self):
        import ctypes
        from ctypes import wintypes
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'immutable_pair'
            path.write_bytes(b'a'*65536)
            kernel = ctypes.WinDLL('kernel32',use_last_error=True)
            real_read = kernel.ReadFile
            real_read.argtypes = [wintypes.HANDLE,wintypes.LPVOID,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID]
            def delete_then_read(*args):
                path.unlink()  # Must succeed with the reader handle still open.
                return real_read(*args)
            proxy = SimpleNamespace(CreateFileW=kernel.CreateFileW,GetFileSizeEx=kernel.GetFileSizeEx,
                                    ReadFile=delete_then_read,CloseHandle=kernel.CloseHandle)
            with patch('ctypes.WinDLL',return_value=proxy):
                self.assertEqual(shared_bytes(path),b'a'*65536)
            self.assertFalse(path.exists())

    def test_cpu_rgb_preview_has_sample_identity_without_modifying_trainer_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent/'experiment'
            directory = root/'campaign'
            (directory/'models').mkdir(parents=True)
            (directory/'pool').mkdir()
            dataset = parent/'dataset'
            (dataset/'DIV2K_train_HR').mkdir(parents=True)
            for i in (1,2):
                Image.new('RGB',(24,16),(100+i,80,60)).save(dataset/f'DIV2K_train_HR/{i:04d}.png')
            config = dict(slm_shape=[16,24],roi_shape=[8,12],slm_pitch_m=6.4e-6,
                          relay_magnification=1.,distance_m=.015,wavelength_m=520e-9,
                          fourier_aperture=.5,invert_phase=True,seed=42)
            path = directory/'models/rgb_latest.pt'
            path.write_bytes(b'Hot checkpoint deliberately unreadable as a torch model')
            pool = []
            for serial,split in ((1,'train'),(2,'val')):
                name = f'{serial}.npz'
                np.savez(directory/'pool'/name,phase_rgb_u8=np.full((16,24,3),127,np.uint8),
                         amplitude_rgb=np.full((8,12,3),.7,np.float16))
                pool.append(dict(file=name,serial=serial,split=split,cycle=0,
                                 source={'target_id':f'DIV2K_train_HR/{serial:04d}.png'}))
            (directory/'pool.json').write_text(json.dumps(pool))
            (directory/'plan.json').write_text(json.dumps({'dataset_root':str(dataset)}))
            (root/'green').mkdir()
            (root/'green/config.json').write_text(json.dumps(config))
            (directory/'acquisition.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool))
            metrics = dict(channel='rgb',global_epoch=1,epoch=1,cycle=0,time=100.,val_mse=.1)
            (directory/'training.jsonl').write_text(json.dumps(metrics)+'\n')
            (root/'live/test').mkdir(parents=True)
            (root/'live/current.json').write_text(json.dumps({'run_id':'test'}))
            (root/'live/test/state.json').write_text(json.dumps({'streams':{}}))
            originals = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}
            self.assertTrue(catalog(root)['available'])
            service = PreviewService(parent)
            try:
                result = service.request('experiment','train')
                self.assertEqual(result['status'],'building')
                until = time.monotonic()+10
                while result['status']=='building' and time.monotonic()<until:
                    time.sleep(.02)
                    result = service.request('experiment','train')
                self.assertEqual(result['status'],'ready',result)
                self.assertEqual(result['target_id'],'DIV2K_train_HR/0001.png')
                self.assertEqual(result['identity'],'last_completed_train_sample')
                self.assertFalse(result['live_batch'])
                self.assertEqual(len(result['images']),4)
                self.assertTrue(all((root/i['file']).is_file() for i in result['images']))
                self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in originals.items()))
                with self.assertRaises(ValueError):
                    service.request('../experiment','train')
            finally:
                service.executor.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
