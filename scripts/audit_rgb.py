"""Read-only consistency audit for a completed RGB experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    result = {'channels': {}, 'fullcolor': {}, 'status': 'complete'}
    target_plan = read(root/'targets/manifest.json') if (root/'targets/manifest.json').exists() else None
    withheld = ({r['target_id'] for r in target_plan['targets'] if r['slot'] in [1000, 2000]}
                if target_plan else {1000, 2000})
    for channel in ['red', 'green', 'blue']:
        path = root/channel
        manifest = read(path/'pairs/manifest.json')
        splits = {s: {r['target_id'] for r in manifest['pairs'] if r['split'] == s}
                  for s in ['train', 'val', 'test']}
        assert all(splits.values())
        assert not (splits['train'] & splits['val'] or splits['train'] & splits['test'] or splits['val'] & splits['test'])
        assert not (withheld & set.union(*splits.values()))
        assert len({row['target_id'] for row in manifest['pairs']}) == len(manifest['pairs'])
        for row in manifest['pairs']:
            with np.load(path/'pairs'/row['file']) as pair:
                assert pair['phase'].shape == tuple(manifest['config']['slm_shape'])
                assert pair['amplitude'].shape == tuple(manifest['config']['roi_shape'])
                assert np.isfinite(pair['phase']).all() and np.isfinite(pair['amplitude']).all()
                if target_plan:
                    source = row['source_image']
                    assert source['original_split'] == ('train' if row['split'] == 'train' else 'valid')
                    frozen = np.load(root/'targets'/source['amplitude_file'])
                    ci = ['red', 'green', 'blue'].index(channel)
                    np.testing.assert_array_equal(pair['target'], frozen[..., ci])
                    rgb_phase = np.load(root/'patterns'/row['rgb_pattern_file'], mmap_mode='r')
                    np.testing.assert_array_equal(pair['phase'], rgb_phase[..., ci])
        checkpoint = torch.load(path/'forward_model.pt', map_location='cpu', weights_only=True)
        trace = torch.load(path/'model_trace'/f'epoch_{checkpoint["epoch"]:03d}.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(v, trace['correction'][k]) for k, v in checkpoint['correction'].items())
        assert torch.equal(checkpoint['log_gain'], trace['log_gain'])
        training = read(path/'forward_training.json')
        assert checkpoint['epoch'] == training['best_epoch']
        before_after = read(path/'citl_metrics.json')['repeat_validation']
        values = {label: float(np.mean([r['amplitude_mse'] for r in before_after if r['label'] == label]))
                  for label in ['before', 'after']}
        test = [r for r in training['evaluation'] if r['split'] == 'test']
        result['channels'][channel] = {'split_counts': {s: len(v) for s, v in splits.items()},
            'citl_repeat_mse': values, 'citl_mse_reduction_percent': 100*(1-values['after']/values['before']),
            'best_epoch': checkpoint['epoch'], 'checkpoint_matches_epoch_trace': True,
            'test_mse': {key: float(np.mean([r[key] for r in test])) for key in ['asm_mse', 'gain_only_mse', 'cnn_mse']}}
    fullcolor = read(root/'fullcolor/metrics.json')
    for variant in sorted({r['variant'] for r in fullcolor['repeat_validation']}):
        rows = [r for r in fullcolor['repeat_validation'] if r['variant'] == variant]
        assert len(rows) == 3
        phase = np.load(root/'fullcolor'/f'{variant}_phase_rgb.npy')
        assert phase.shape == (1080, 1920, 3) and np.isfinite(phase).all()
        # Independent inverted phase-to-HDMI round trip, including RGB channel order.
        expected = np.round((1-(phase+np.pi)%(2*np.pi)/(2*np.pi))*255).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(Image.open(root/'fullcolor'/f'{variant}_slm_rgb.png')), expected)
        result['fullcolor'][variant] = {'mean_mse': float(np.mean([r['amplitude_mse'] for r in rows])),
            'repeat_std': float(np.std([r['amplitude_mse'] for r in rows], ddof=1)),
            'per_channel_mean_mse': np.mean([r['amplitude_mse_rgb'] for r in rows], axis=0).tolist()}
    frames = read(root/'fullcolor/frame_metadata.json')
    timestamps = [f['timestamp_ns'] for f in frames]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    result['fullcolor_frames'] = len(frames)
    result['timestamps_strictly_increasing'] = True
    result['max_stage_gpu_allocated_gib'] = max(read(p).get('memory', {}).get('peak_allocated_gib', 0)
                                              for p in root.glob('completed_*.json'))
    if target_plan:
        bank = read(root/'patterns/manifest.json')
        assert len(bank['patterns']) == target_plan['pairs']
        assert len({r['target_id'] for r in target_plan['targets']}) == len(target_plan['targets'])
        for row in bank['patterns']:
            phase = np.load(root/'patterns'/row['phase_file'])
            expected = np.round((1-(phase+np.pi)%(2*np.pi)/(2*np.pi))*255).clip(0, 255).astype(np.uint8)
            np.testing.assert_array_equal(np.asarray(Image.open(root/'patterns'/row['file'])), expected)
        result['div2k'] = {'available_originals': target_plan['available_originals'],
                           'selected_distinct_originals': len(target_plan['targets']),
                           'rgb_patterns_verified': len(bank['patterns']),
                           'demo_and_new_target_excluded_from_fitting': True,
                           'captured_phases_match_rgb_pattern_bank': True,
                           'pair_targets_match_frozen_image_targets': True}
    (root/'completion_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
