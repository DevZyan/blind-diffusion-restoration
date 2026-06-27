"""
Image restoration metrics: PSNR, SSIM, LPIPS.

PSNR and SSIM are implemented in numpy (no torch dependency) so they can be used
anywhere. LPIPS requires torch + the `lpips` package and is wrapped lazily so the
module still imports on a torch-free machine.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


# --------------------------------------------------------------------------- #
# PSNR
# --------------------------------------------------------------------------- #
def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio. Inputs float in [0, data_range], HxWxC."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mse = np.mean((pred - target) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


# --------------------------------------------------------------------------- #
# SSIM  (Wang et al. 2004, Gaussian-free uniform-window variant)
# --------------------------------------------------------------------------- #
def _ssim_single(x: np.ndarray, y: np.ndarray, data_range: float, win: int = 7) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = uniform_filter(x, win)
    mu_y = uniform_filter(y, win)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = uniform_filter(x * x, win) - mu_x2
    sigma_y2 = uniform_filter(y * y, win) - mu_y2
    sigma_xy = uniform_filter(x * y, win) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    # crop borders affected by the window
    pad = win // 2
    return float(ssim_map[pad:-pad, pad:-pad].mean())


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM averaged over channels."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    if pred.ndim == 2:
        return _ssim_single(pred, target, data_range)
    return float(np.mean([
        _ssim_single(pred[..., c], target[..., c], data_range)
        for c in range(pred.shape[2])
    ]))


# --------------------------------------------------------------------------- #
# LPIPS  (lazy: needs torch + lpips)
# --------------------------------------------------------------------------- #
class LPIPSMetric:
    """Learned perceptual metric. Lower is better. Lazily loads the network."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import torch  # local import keeps the module torch-free until used
        import lpips

        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Inputs HxWxC float in [0, 1]; internally mapped to [-1, 1]."""
        t = self.torch

        def to_tensor(a):
            x = t.from_numpy(a.transpose(2, 0, 1)).float().unsqueeze(0)
            return (x * 2 - 1).to(self.device)

        with t.no_grad():
            d = self.model(to_tensor(pred), to_tensor(target))
        return float(d.item())


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
class MetricBank:
    """Accumulate PSNR/SSIM/LPIPS over a dataset and report means."""

    def __init__(self, use_lpips: bool = False, device: str = "cpu"):
        self._psnr: list[float] = []
        self._ssim: list[float] = []
        self._lpips: list[float] = []
        self.lpips_fn = LPIPSMetric(device=device) if use_lpips else None

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        self._psnr.append(psnr(pred, target))
        self._ssim.append(ssim(pred, target))
        if self.lpips_fn is not None:
            self._lpips.append(self.lpips_fn(pred, target))

    def result(self) -> dict:
        out = {
            "psnr": float(np.mean(self._psnr)) if self._psnr else float("nan"),
            "ssim": float(np.mean(self._ssim)) if self._ssim else float("nan"),
            "n": len(self._psnr),
        }
        if self._lpips:
            out["lpips"] = float(np.mean(self._lpips))
        return out
