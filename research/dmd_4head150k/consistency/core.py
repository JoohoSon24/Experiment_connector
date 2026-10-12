"""Discrete VP adaptation of Song et al. (2023), Algorithms 1 and 2.

Use y = x_VP / sqrt(alpha_bar), sigma = sqrt((1-alpha_bar)/alpha_bar).
The backbone still receives the original VP-normalized input and timestep.
Teacher DDIM transitions are Euler steps in these additive-noise coordinates.
This is not an exact reproduction of the paper's EDM experiments.
"""
from __future__ import annotations

import torch
from torch import nn

from teacher.cfg import apply_cfg


def noise_levels(alpha_bar: torch.Tensor) -> torch.Tensor:
    if alpha_bar.ndim != 1 or not torch.all((alpha_bar > 0) & (alpha_bar < 1)):
        raise ValueError("alpha_bar must be a one-dimensional VP schedule in (0, 1)")
    if not torch.all(alpha_bar[1:] < alpha_bar[:-1]):
        raise ValueError("alpha_bar must strictly decrease")
    return ((1 - alpha_bar) / alpha_bar).sqrt()


def expand(value: torch.Tensor) -> torch.Tensor:
    return value[:, None, None, None]


def denoise(network: nn.Module, y: torch.Tensor, times: torch.Tensor,
            labels: torch.Tensor, alpha_bar: torch.Tensor,
            sigma_data: float = 0.5) -> torch.Tensor:
    """Return a boundary-conditioned clean prediction; exactly one network call.

    At t=0, c_skip=1 and c_out=0, hence f(y,0)=y exactly. The backbone's
    output is a consistency residual, NOT the teacher's epsilon or student v.
    """
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")
    alpha = expand(alpha_bar[times].sqrt())
    sigma = expand(((1 - alpha_bar[times]) / alpha_bar[times]).sqrt())
    sigma_min = ((1 - alpha_bar[0]) / alpha_bar[0]).sqrt()
    delta = sigma - sigma_min
    c_skip = sigma_data ** 2 / (delta.square() + sigma_data ** 2)
    c_out = sigma_data * delta / (sigma.square() + sigma_data ** 2).sqrt()
    residual = network(alpha * y, times, labels).float()
    return c_skip * y + c_out * residual


@torch.no_grad()
def teacher_step(teacher: nn.Module, y: torch.Tensor, times: torch.Tensor,
                 next_times: torch.Tensor, labels: torch.Tensor,
                 alpha_bar: torch.Tensor, guidance_scale: float,
                 clip_denoised: bool) -> torch.Tensor:
    """Euler transition in y-space, equivalent to the teacher's VP DDIM step."""
    alpha = expand(alpha_bar[times].sqrt())
    sigma = expand(((1 - alpha_bar[times]) / alpha_bar[times]).sqrt())
    next_sigma = expand(((1 - alpha_bar[next_times]) / alpha_bar[next_times]).sqrt())
    x = alpha * y
    if guidance_scale == 1:
        eps = teacher(x, times, labels).float()
    elif guidance_scale == 0:
        eps = teacher(x, times, torch.full_like(labels, teacher.num_classes)).float()
    else:
        uncond = teacher(x, times, torch.full_like(labels, teacher.num_classes))
        cond = teacher(x, times, labels)
        eps = apply_cfg(uncond, cond, guidance_scale).float()
    if clip_denoised:
        clean = (y - sigma * eps).clamp(-1, 1)
        eps = (y - clean) / sigma
    return y + (next_sigma - sigma) * eps


def loss(student: nn.Module, target: nn.Module, teacher: nn.Module,
         images: torch.Tensor, labels: torch.Tensor, alpha_bar: torch.Tensor,
         grid: torch.Tensor, guidance_scale: float, sigma_data: float,
         clip_teacher: bool) -> torch.Tensor:
    """Uniform adjacent-grid CD, L2 metric, lambda(t)=1; no pretrained features."""
    index = torch.randint(1, len(grid), (len(labels),), device=labels.device)
    times, next_times = grid[index], grid[index - 1]
    sigma = expand(((1 - alpha_bar[times]) / alpha_bar[times]).sqrt())
    y = images.float() + sigma * torch.randn_like(images, dtype=torch.float32)
    with torch.no_grad():
        next_y = teacher_step(teacher, y, times, next_times, labels, alpha_bar,
                              guidance_scale, clip_teacher)
        desired = denoise(target, next_y, next_times, labels, alpha_bar, sigma_data)
    predicted = denoise(student, y, times, labels, alpha_bar, sigma_data)
    return (predicted - desired).square().mean()


def sampling_times(diffusion_steps: int, nfe: int, device=None) -> torch.Tensor:
    if nfe not in (1, 2, 3, 4) or diffusion_steps < 32:
        raise ValueError("Require 1–4 NFE and at least 32 diffusion steps")
    # Reuse the baseline's 32-node grid for 1/2/4 NFE. Never waste a call at
    # the identity boundary t=0. These are defaults, not FID-optimized nodes.
    grid = torch.linspace(diffusion_steps - 1, 0, 32, device=device).round().long()
    indices = torch.div(torch.arange(nfe, device=device) * 32, nfe, rounding_mode="floor")
    return grid[indices]


@torch.inference_mode()
def sample(network: nn.Module, alpha_bar: torch.Tensor, times: torch.Tensor,
           labels: torch.Tensor, seed: int, sigma_data: float = 0.5,
           image_size: int = 64) -> torch.Tensor:
    if times.ndim != 1 or len(times) not in (1, 2, 3, 4):
        raise ValueError("Expected 1–4 sampling timesteps")
    if (times[0] != len(alpha_bar) - 1 or torch.any(times <= 0)
            or torch.any(times[1:] >= times[:-1])):
        raise ValueError("Sampling must start at terminal noise and strictly descend above t=0")
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    shape = (len(labels), 3, image_size, image_size)
    sigmas = noise_levels(alpha_bar)
    def noise():
        return torch.randn(shape, device=labels.device, generator=generator)
    y = sigmas[times[0]] * noise()
    for index, time in enumerate(times):
        if index:
            y = clean + (sigmas[time].square() - sigmas[0].square()).clamp_min(0).sqrt() * noise()
        t = torch.full_like(labels, int(time))
        clean = denoise(network, y, t, labels, alpha_bar, sigma_data)
    return clean.clamp(-1, 1)
