"""Full-grid actual-pair CNN/Adam restore verification, outside production weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import torch
from benchtop.paper_model import PaperCNNpropCNN
from benchtop.campaign import decode_phase, save_checkpoint
from benchtop.optics import crop_center
from benchtop.live import atomic_json

root=Path('results/paper_cnnpropcnn_20261024')
out=root/'verification'
out.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(9321)
torch.backends.cudnn.deterministic=True
config=json.loads((root/'green/config.json').read_text())
net=PaperCNNpropCNN(config,'cuda')
opt=torch.optim.Adam(net.parameters(),lr=.0005)
samples=[]
for i in range(2):
    with np.load(root/'pairs'/f'{i:04d}.npz') as f:
        samples.append((torch.from_numpy(decode_phase(f['phase_rgb_u8'][...,1])).cuda(),
                        torch.from_numpy(f['amplitude_rgb'][...,1].copy()).cuda()))
def step(index):
    t=time.perf_counter()
    p,a=samples[index]
    opt.zero_grad(set_to_none=True)
    pred=crop_center(net(p).abs(),a.shape)
    loss=(pred-a).abs().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    opt.step()
    torch.cuda.synchronize()
    return dict(loss=float(loss),seconds=time.perf_counter()-t)
first=step(0)
save_checkpoint(out/'resume_test.pt',dict(weights=net.learnable_state(),optimizer=opt.state_dict()))
continuous=step(1)
expected=net.learnable_state()
saved=torch.load(out/'resume_test.pt',map_location='cpu',weights_only=False)
net.load_learnable(saved['weights'])
opt.load_state_dict(saved['optimizer'])
resumed=step(1)
actual=net.learnable_state()
maximum=max(float((actual[k]-expected[k]).abs().max()) for k in expected)
steps=sorted(set(int(s['step'].item()) for s in opt.state.values()))
assert maximum<1e-6,(maximum,steps)
assert steps==[2],steps
report=dict(status='passed',physical_shape=config['slm_shape'],roi=config['roi_shape'],
    parameters=sum(p.numel() for p in net.parameters()),first_update=first,
    continuous_second_update=continuous,resumed_second_update=resumed,
    max_weight_difference_after_resume=maximum,optimizer_steps=steps,
    gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
    note='Two actual new camera pairs; functional training/resume verification only, not performance evidence')
atomic_json(out/'training_resume.json',report)
print(json.dumps(report,indent=2))
