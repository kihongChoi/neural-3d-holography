"""Joint RGB phase descent with streamed channel gradients and explicit stopping."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .optics import crop_center


@dataclass
class Convergence:
    minimum: int = 200
    window: int = 20
    patience: int = 3
    relative: float = .002
    absolute: float = 1e-5

    def __post_init__(self):
        if self.minimum < 0 or self.window < 2 or self.patience < 1:
            raise ValueError('Invalid convergence window/minimum/patience')
        if self.relative < 0 or self.absolute < 0:
            raise ValueError('Convergence tolerances must be nonnegative')
        self.values = []
        self.flat = 0
        self.checks = []

    def update(self, step, loss):
        self.values.append(float(loss))
        # Disjoint windows avoid counting almost identical rolling averages.
        if step < self.minimum or len(self.values) < 2*self.window or step % self.window:
            return False
        before = float(np.median(self.values[-2*self.window:-self.window]))
        recent = float(np.median(self.values[-self.window:]))
        tolerance = max(self.absolute, self.relative*abs(before))
        delta = before-recent
        self.flat = self.flat+1 if abs(delta) <= tolerance else 0
        self.checks.append(dict(step=step, previous_median=before, recent_median=recent,
                                improvement=delta, tolerance=tolerance, stable_windows=self.flat))
        return self.flat >= self.patience


def optimize_rgb_phase(models, target, configs, initial, *, max_steps=1000, lr=.02,
                       capture=None, convergence=None, trace=None, guard=None):
    """One RGB Adam update; evaluate/backpropagate one wavelength at a time.

    With capture, every measured RGB tuple uses the same displayed RGB phase.
    Models are wavelength-specific; the objective is mean RGB amplitude MSE.
    The return phase is the best *measured* phase, before its Adam update.
    """
    if not 0 <= max_steps <= 1000:
        raise ValueError('RGB phase iterations must be between 0 and 1000')
    device = models[0].kernel.device
    target = torch.as_tensor(target, device=device, dtype=torch.float32)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError('Expected H x W x RGB target')
    initial = torch.as_tensor(initial, device=device, dtype=torch.float32)
    if tuple(initial.shape) != tuple(configs[0]['slm_shape'])+(3,):
        raise ValueError('RGB phase must retain the native SLM grid')
    phases = [initial[..., c].detach().clone().requires_grad_(True) for c in range(3)]
    optimizer = torch.optim.Adam(phases, lr=lr)
    history, best, best_loss, best_step = [], initial.detach().clone(), math.inf, 0
    reason = 'iteration_limit'
    for step in range(max_steps+1):
        if guard:
            guard()
        optimizer.zero_grad(set_to_none=True)
        displayed = torch.stack([p.detach() for p in phases], -1)
        camera = None
        if capture:
            camera = torch.as_tensor(capture(displayed.cpu().numpy()), device=device, dtype=torch.float32)
            if camera.shape != target.shape or not torch.isfinite(camera).all():
                raise RuntimeError('Invalid joint RGB camera amplitude')
        values, previews = [], []
        for ci, model in enumerate(models):
            prediction = crop_center(model(phases[ci]).abs(), target.shape[:2])
            if camera is not None:
                prediction = prediction+(camera[..., ci]-prediction).detach()
            loss = (prediction-target[..., ci]).square().mean()/3
            values.append(float(loss.detach())*3)
            previews.append(prediction.detach())
            if step < max_steps:
                loss.backward()  # Release this channel's FFT graph before the next.
        value = float(np.mean(values))
        if not math.isfinite(value):
            raise RuntimeError('Non-finite joint RGB loss')
        if value < best_loss:
            best, best_loss, best_step = displayed.clone(), value, step
        converged = convergence is not None and convergence.update(step, value)
        stopping = converged or step == max_steps
        row = dict(step=step, amplitude_mse=value, amplitude_mse_rgb=values,
                   psnr_db=-10*math.log10(max(value, 1e-12)), scale=1.,
                   best_step=best_step, best_mse=best_loss,
                   learning_rate=optimizer.param_groups[0]['lr'],
                   phase_delta_rms_rgb=(((displayed-initial+torch.pi)%(2*torch.pi)-torch.pi)
                                        .square().mean((0,1)).sqrt().tolist()))
        if convergence and convergence.checks:
            row['convergence'] = convergence.checks[-1]
        if stopping:
            reason = 'converged' if converged else 'iteration_limit'
            row['stop_reason'] = reason
        history.append(row)
        if trace:
            gradient = None if phases[0].grad is None else torch.stack([p.grad.detach() for p in phases], -1)
            trace(step, row, displayed, torch.stack(previews, -1), gradient)
        if stopping:
            break
        optimizer.step()
    termination = dict(reason=reason, steps=history[-1]['step'], maximum=max_steps,
                       best_step=best_step, best_mse=best_loss,
                       policy=asdict(convergence) if convergence else None,
                       checks=convergence.checks if convergence else [])
    return best.detach(), history, termination
