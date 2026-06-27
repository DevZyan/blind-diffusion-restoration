"""
Conditional DDPM for blind image restoration.

We condition a denoising U-Net on the degraded image (concatenated channel-wise
to the noisy latent), following the SR3 / Palette conditioning scheme. The model
learns p(clean | degraded) by reversing a fixed Gaussian forward process applied
to the *clean* target.

Forward:   q(x_t | x_0) = N( sqrt(a_bar_t) x_0, (1 - a_bar_t) I )
Train:     predict epsilon from (x_t, t, condition); MSE loss.
Sample:    iterative ancestral sampling from x_T ~ N(0, I), conditioned on degraded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"  # "linear" | "cosine"


def _linear_beta(cfg: DiffusionConfig) -> torch.Tensor:
    return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)


def _cosine_beta(cfg: DiffusionConfig, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule."""
    steps = cfg.timesteps + 1
    t = torch.linspace(0, cfg.timesteps, steps) / cfg.timesteps
    alphas_bar = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class Diffusion:
    """Holds schedule buffers and the train/sample logic."""

    def __init__(self, cfg: DiffusionConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.T = cfg.timesteps

        betas = _cosine_beta(cfg) if cfg.schedule == "cosine" else _linear_beta(cfg)
        betas = betas.to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_bar = alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

        # posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    # --------------------------------------------------------------------- #
    def _gather(self, buf: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = buf.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    # --------------------------------------------------------------------- #
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Diffuse clean x0 to step t."""
        a = self._gather(self.sqrt_alphas_bar, t, x0.shape)
        b = self._gather(self.sqrt_one_minus_alphas_bar, t, x0.shape)
        return a * x0 + b * noise

    # --------------------------------------------------------------------- #
    def training_loss(self, model, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x0:   clean target  (B, 3, H, W)
        cond: degraded image (B, 3, H, W)
        """
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        model_in = torch.cat([x_t, cond], dim=1)          # 6-channel input
        out = model(model_in, t)
        pred = out.sample if hasattr(out, "sample") else out
        return F.mse_loss(pred, noise)

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, model, cond: torch.Tensor, progress: bool = False) -> torch.Tensor:
        """
        Ancestral sampling conditioned on `cond` (degraded image).
        Returns restored image in [-1, 1].
        """
        b, _, h, w = cond.shape
        x = torch.randn(b, 3, h, w, device=cond.device)

        iterator = reversed(range(self.T))
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=self.T, desc="sampling")

        for i in iterator:
            t = torch.full((b,), i, device=cond.device, dtype=torch.long)
            model_in = torch.cat([x, cond], dim=1)
            out = model(model_in, t)
            eps = out.sample if hasattr(out, "sample") else out

            beta_t = self._gather(self.betas, t, x.shape)
            sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_bar, t, x.shape)
            sqrt_recip = self._gather(self.sqrt_recip_alphas, t, x.shape)

            mean = sqrt_recip * (x - beta_t / sqrt_one_minus * eps)

            if i > 0:
                var = self._gather(self.posterior_var, t, x.shape)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean

        return x.clamp(-1, 1)

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def ddim_sample(self, model, cond: torch.Tensor, steps: int = 50, eta: float = 0.0):
        """Faster DDIM sampling (deterministic when eta=0). Returns [-1, 1]."""
        b, _, h, w = cond.shape
        x = torch.randn(b, 3, h, w, device=cond.device)

        step_seq = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=cond.device)
        for idx, ti in enumerate(step_seq):
            t = torch.full((b,), int(ti), device=cond.device, dtype=torch.long)
            model_in = torch.cat([x, cond], dim=1)
            out = model(model_in, t)
            eps = out.sample if hasattr(out, "sample") else out

            a_bar = self._gather(self.alphas_bar, t, x.shape)
            x0_pred = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)
            x0_pred = x0_pred.clamp(-1, 1)

            if idx < len(step_seq) - 1:
                t_next = torch.full((b,), int(step_seq[idx + 1]), device=cond.device, dtype=torch.long)
                a_bar_next = self._gather(self.alphas_bar, t_next, x.shape)
                sigma = eta * torch.sqrt((1 - a_bar_next) / (1 - a_bar) * (1 - a_bar / a_bar_next))
                dir_xt = torch.sqrt(1 - a_bar_next - sigma ** 2) * eps
                x = torch.sqrt(a_bar_next) * x0_pred + dir_xt + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x.clamp(-1, 1)
