"""Finish missing blue simulated phases while the physical desktop is locked."""
import json
from pathlib import Path
import numpy as np
import torch
from .calibration import Rectifier
from .optics import optimize_phase
from .paper_model import PaperCNNpropCNN, dpac
from .rgb_run import channel_model, read_json

root = Path('results/beamsplat_20261026_eval01')
trained = Path('results/paper_cnnpropcnn_20261024')
c = read_json(trained/'blue/config.json')
r = Rectifier(trained/'blue/calibration', c)
t = torch.from_numpy(np.load(root/'target_amplitude.npy')[..., 2]).cuda()
asm = channel_model(c, r, 'cuda')
with torch.no_grad():
    initial = dpac(asm, t)
for name in ('asm', 'cnn'):
    dest = root/f'best_blue_{name}.npy'
    if dest.exists():
        continue
    if name == 'asm':
        model = asm
    else:
        saved = torch.load(trained/'models/blue_best.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == c
        model = PaperCNNpropCNN(c, 'cuda')
        model.load_learnable(saved['weights'])
        model.eval().requires_grad_(False)
        del saved
    def callback(step, stats, prediction):
        if step % 100 == 0:
            print(json.dumps(dict(method='blue_'+name, **stats)), flush=True)
    best, history = optimize_phase(model, t, c, iterations=1000, lr=.01, initial=initial,
                                   fixed_scale=1., callback=callback)
    np.save(dest, best.cpu().numpy())
    (root/f'history_blue_{name}.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
print('Offline blue complete; camera not accessed.', flush=True)
