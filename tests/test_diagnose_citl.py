import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from benchtop.diagnose_citl import Diagnosis, cosine, fd_metrics, save_preview
from benchtop.live import LiveMonitor


class DiagnosticMathTests(unittest.TestCase):
    def test_central_difference_recovers_linear_response_and_noise_of_mean(self):
        rng=np.random.default_rng(4)
        matrix=rng.normal(size=(30,6)); phase=rng.normal(size=6); direction=rng.normal(size=6)
        epsilon=.05
        measured=((matrix@(phase+epsilon*direction))-(matrix@(phase-epsilon*direction)))/(2*epsilon)
        expected=matrix@direction
        np.testing.assert_allclose(measured,expected,atol=1e-12)
        response=measured.reshape(2,5,3)
        metrics=fd_metrics(np.stack([response+.02,response-.02]),response)
        self.assertAlmostEqual(metrics['cosine'],1.)
        self.assertAlmostEqual(metrics['estimated_noise_of_mean_rms'],.02)
        self.assertAlmostEqual(metrics['relative_rmse'],0.)
        self.assertAlmostEqual(cosine(response,-response),-1.)

    def test_zero_gradient_is_not_reported_as_agreement(self):
        self.assertIsNone(cosine(np.zeros(4),np.ones(4)))
        with self.assertRaises(ValueError):
            fd_metrics(np.ones((3,2,2,3)),np.ones((2,2,3)))

    def test_preview_uses_fixed_radiometry_without_per_image_rescale(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.png'
            data=np.full((4,8,3),.5,np.float32);data[0,0]=1.
            save_preview(path,data)
            with Image.open(path) as im:
                result=np.array(im)
            np.testing.assert_array_equal(result[0,0],[255,255,255])
            np.testing.assert_array_equal(result[1,1],[137,137,137])

    def test_fixed_pattern_measurements_publish_and_save_without_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            worker=Diagnosis.__new__(Diagnosis)
            worker.out=Path(folder);worker.args=SimpleNamespace(rounds=2)
            worker.started=time.monotonic()
            a=np.full((4,8,3),.5,np.float32)
            worker.cases=[dict(id=identity,target=a,phases={'initial':a,'old_best':a}) for identity in ('a','b')]
            worker.capture=lambda phase,label:a.copy()
            worker.monitor=LiveMonitor(folder,'synthetic fixed pattern measurement')
            try:
                worker.stability()
                stream=next(iter(worker.monitor.state['streams'].values()))
                self.assertEqual(len(stream['history']),8)
                self.assertEqual(stream['history'][-1]['step'],8)
                self.assertTrue((worker.out/'stability_summary.json').exists())
            finally:
                worker.monitor.finish()


if __name__=='__main__':
    unittest.main()
