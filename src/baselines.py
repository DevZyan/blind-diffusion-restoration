"""
Classical (non-learned) restoration baselines.

These give the diffusion model something honest to beat. All are pure
numpy/scipy/PIL so they run anywhere. They are intentionally simple, well-known
methods -- the point is a fair reference floor, not SOTA classical restoration.

  * non_local_means  -- patch-based denoiser (BM3D-spirit, simplified)
  * wiener_deblur    -- Wiener-filter deconvolution for blur
  * bilateral_denoise-- edge-preserving smoothing
  * identity         -- passthrough (lower bound = the degraded image itself)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve


def identity(img: np.ndarray) -> np.ndarray:
    return img.copy()


def gaussian_denoise(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Trivial Gaussian-smoothing denoiser (weak baseline)."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return np.clip(out, 0, 1)


def bilateral_denoise(img: np.ndarray, sigma_spatial: float = 2.0,
                      sigma_range: float = 0.1, win: int = 5) -> np.ndarray:
    """Edge-preserving bilateral filter (vectorized over shift offsets)."""
    pad = win // 2
    out = np.zeros_like(img)
    wsum = np.zeros(img.shape[:2] + (1,))
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = padded[pad + dy:pad + dy + img.shape[0],
                             pad + dx:pad + dx + img.shape[1]]
            spatial = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_spatial ** 2))
            rng = np.exp(-((shifted - img) ** 2).sum(axis=2, keepdims=True)
                         / (2 * sigma_range ** 2))
            w = spatial * rng
            out += w * shifted
            wsum += w
    return np.clip(out / np.maximum(wsum, 1e-8), 0, 1)


def non_local_means(img: np.ndarray, h: float = 0.1, patch: int = 3,
                    search: int = 7) -> np.ndarray:
    """
    Simplified non-local means. O(N * search^2) -- keep search window small.
    Good middle-ground denoising baseline (BM3D in spirit).
    """
    pad = search // 2 + patch // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    H, W, C = img.shape
    out = np.zeros_like(img)
    wsum = np.zeros((H, W, 1))
    sw = search // 2

    # precompute patch box filter helper
    def box(x):
        return uniform_filter(x, size=patch, mode="reflect")

    base = padded[pad:pad + H, pad:pad + W]
    for dy in range(-sw, sw + 1):
        for dx in range(-sw, sw + 1):
            shifted = padded[pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            diff2 = ((base - shifted) ** 2).mean(axis=2, keepdims=True)
            patch_dist = box(diff2[..., 0])[..., None]
            w = np.exp(-patch_dist / (h ** 2))
            out += w * shifted
            wsum += w
    return np.clip(out / np.maximum(wsum, 1e-8), 0, 1)


def wiener_deblur(img: np.ndarray, blur_sigma: float = 2.0, nsr: float = 0.01) -> np.ndarray:
    """
    Wiener deconvolution assuming a known Gaussian blur kernel.
    `nsr` is the noise-to-signal ratio regularizer.
    """
    # build Gaussian PSF
    k = int(2 * round(3 * blur_sigma) + 1)
    ax = np.arange(-(k // 2), k // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * blur_sigma ** 2))
    psf /= psf.sum()

    H, W, _ = img.shape
    psf_pad = np.zeros((H, W))
    psf_pad[:k, :k] = psf
    psf_pad = np.roll(psf_pad, -(k // 2), axis=(0, 1))
    Hf = np.fft.fft2(psf_pad)
    Wiener = np.conj(Hf) / (np.abs(Hf) ** 2 + nsr)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        G = np.fft.fft2(img[..., c])
        out[..., c] = np.real(np.fft.ifft2(G * Wiener))
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------- #
# Composite "best-effort" classical pipeline for the blind/mixed bucket
# --------------------------------------------------------------------------- #
def classical_blind(img: np.ndarray) -> np.ndarray:
    """
    A reasonable hand-built classical attempt at blind restoration:
    light deblur -> NLM denoise -> mild sharpening.
    """
    x = wiener_deblur(img, blur_sigma=1.2, nsr=0.02)
    x = non_local_means(x, h=0.08, patch=3, search=5)
    # unsharp mask
    blur = gaussian_filter(x, sigma=(1, 1, 0))
    x = np.clip(x + 0.4 * (x - blur), 0, 1)
    return x


BASELINES = {
    "identity": identity,
    "gaussian": gaussian_denoise,
    "bilateral": bilateral_denoise,
    "nlm": non_local_means,
    "wiener": wiener_deblur,
    "classical_blind": classical_blind,
}
