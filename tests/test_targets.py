import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchtop.targets import prepare_image, srgb_to_linear, create_div2k_plan, load_target


class DIV2KTargetsTests(unittest.TestCase):
    def test_color_radiometry_and_aspect_preserving_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'target.png'
            Image.new('RGB', (60, 40), (128, 64, 255)).save(path)
            amplitude, meta = prepare_image(path, (4, 8))
            expected = np.sqrt(srgb_to_linear(np.array([128, 64, 255], np.float32)/255))
            self.assertEqual(amplitude.shape, (4, 8, 3))
            np.testing.assert_allclose(amplitude, np.broadcast_to(expected, amplitude.shape), atol=1e-6)
            self.assertEqual(meta['resized_size_wh'], [8, 6])
            self.assertEqual(meta['crop_xywh'], [0, 1, 8, 4])
            self.assertEqual(len(meta['source_sha256']), 64)

    def test_resampling_averages_linear_intensity_not_encoded_srgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'target.png'
            pixels = np.zeros((2, 2, 3), np.uint8)
            pixels[0] = 255
            Image.fromarray(pixels).save(path)
            amplitude, _ = prepare_image(path, (1, 1))
            np.testing.assert_allclose(amplitude, np.sqrt(.5), atol=1e-6)

    def test_frozen_plan_is_disjoint_and_never_mixes_official_training_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'dataset'
            for name, count in [('train', 12), ('valid', 8)]:
                folder = dataset/f'DIV2K_{name}_HR'
                folder.mkdir(parents=True)
                for index in range(count):
                    Image.new('RGB', (32, 24), (10*index, 60, 180)).save(folder/f'{index:04d}.png')
            plan = create_div2k_plan(root/'run', dataset, (8, 12), pairs=12)
            self.assertEqual(len(plan['targets']), 14)
            self.assertEqual(len({r['target_id'] for r in plan['targets']}), 14)
            for row in plan['targets']:
                self.assertEqual(row['original_split'], 'train' if row['split'] == 'train' else 'valid')
            before = load_target(root/'run', (8, 12), 2000)
            source = next(r for r in plan['targets'] if r['slot'] == 2000)
            Image.new('RGB', (32, 24), 'black').save(source['source_path'])
            np.testing.assert_array_equal(before, load_target(root/'run', (8, 12), 2000))
            self.assertEqual(create_div2k_plan(root/'run', dataset, (8, 12), pairs=12), plan)
            with self.assertRaises(ValueError):
                create_div2k_plan(root/'run', dataset, (10, 14), pairs=12)

    def test_offline_pattern_bank_matches_native_rgb_encoding_and_refuses_changed_settings(self):
        import json
        import torch
        from benchtop.rgb_run import generate_div2k_patterns
        from benchtop.optics import ASM
        from benchtop.hardware import encode_phase
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root/'targets'
            target_dir.mkdir()
            target = np.stack([np.full((8, 12), v, np.float32) for v in [.3, .5, .7]], axis=-1)
            np.save(target_dir/'target.npy', target)
            entry = {'slot': 0, 'split': 'train', 'target_id': 'DIV2K_train_HR/0001.png', 'amplitude_file': 'target.npy'}
            (target_dir/'manifest.json').write_text(json.dumps({'pairs': 1, 'targets': [entry]}))
            base = {'slm_shape': [16, 24], 'roi_shape': [8, 12], 'slm_pitch_m': 6.4e-6,
                    'relay_magnification': 1., 'distance_m': .015, 'wavelength_m': 520e-9,
                    'fourier_aperture': .5, 'invert_phase': True, 'seed': 42}
            for ch, wavelength in zip(['red', 'green', 'blue'], [638e-9, 520e-9, 450e-9]):
                (root/ch).mkdir()
                (root/ch/'config.json').write_text(json.dumps(dict(base, channel=ch, wavelength_m=wavelength)))
            with patch('benchtop.rgb_run.Rectifier'), patch('benchtop.rgb_run.channel_model', side_effect=lambda c, r, d: ASM(c, d)):
                manifest = generate_div2k_patterns(root, base, 'cpu', 1, iterations=2)
                saved = manifest['patterns'][0]
                phase = np.load(root/'patterns'/saved['phase_file'])
                self.assertEqual(phase.shape, (16, 24, 3))
                np.testing.assert_array_equal(np.asarray(Image.open(root/'patterns'/saved['file'])), encode_phase(phase))
                self.assertGreater(np.mean(np.abs(phase[..., 0]-phase[..., 1])), .01)
                self.assertEqual(generate_div2k_patterns(root, base, 'cpu', 1, iterations=2), manifest)
                with self.assertRaises(ValueError):
                    generate_div2k_patterns(root, base, 'cpu', 1, iterations=3)


if __name__ == '__main__':
    unittest.main()
