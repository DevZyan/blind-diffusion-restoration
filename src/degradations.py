"""
Blind multi-degradation pipeline for image restoration.

Applies a randomized composition of Gaussian blur, additive Gaussian noise, and
JPEG compression to clean images. The degradation parameters (and which
degradations are active) are sampled per image, so a model trained on these pairs
must restore *blindly* -- it never sees the degradation type or strength.

Design follows the BSRGAN / Real-ESRGAN philosophy of random degradation
synthesis, deliberately kept lightweight (PIL + scipy only, no OpenCV) so it runs
anywhere including Kaggle kernels.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DegradationConfig:
    """Ranges from which degradation parameters are sampled."""

    # Gaussian blur
    blur_prob: float = 0.7
    blur_sigma_range: tuple[float, float] = (0.5, 3.0)

    # Additive Gaussian noise (sigma on a 0-255 scale)
    noise_prob: float = 0.8
    noise_sigma_range: tuple[float, float] = (5.0, 50.0)

    # JPEG compression
    jpeg_prob: float = 0.8
    jpeg_quality_range: tuple[int, int] = (10, 60)

    # Always apply at least one degradation (avoid clean->clean pairs)
    force_at_least_one: bool = True

    # Randomize the order in which blur/noise/jpeg are applied
    random_order: bool = True


# --------------------------------------------------------------------------- #
# Individual degradations  (operate on float arrays in [0, 1], shape HxWxC)
# --------------------------------------------------------------------------- #
def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Per-channel isotropic Gaussian blur."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma, mode="reflect")
    return out


def apply_gaussian_noise(img: np.ndarray, sigma_255: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise. sigma given on a 0-255 scale."""
    sigma = sigma_255 / 255.0
    noise = rng.normal(0.0, sigma, size=img.shape)
    return img + noise


def apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through JPEG at a given quality to inject blocking artifacts."""
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return out


# --------------------------------------------------------------------------- #
# Blind composer
# --------------------------------------------------------------------------- #
class BlindDegrader:
    """Sample and apply a random degradation chain to a clean image."""

    def __init__(self, config: Optional[DegradationConfig] = None, seed: Optional[int] = None):
        self.cfg = config or DegradationConfig()
        self.rng = np.random.default_rng(seed)

    def _sample_ops(self) -> list[str]:
        ops = []
        if self.rng.random() < self.cfg.blur_prob:
            ops.append("blur")
        if self.rng.random() < self.cfg.noise_prob:
            ops.append("noise")
        if self.rng.random() < self.cfg.jpeg_prob:
            ops.append("jpeg")

        if not ops and self.cfg.force_at_least_one:
            ops = [self.rng.choice(["blur", "noise", "jpeg"])]

        if self.cfg.random_order:
            self.rng.shuffle(ops)
        return ops

    def degrade(self, clean: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Args:
            clean: float32 HxWx3 in [0, 1].
        Returns:
            (degraded_image, metadata_dict). metadata records the applied chain
            and parameters -- useful for per-degradation evaluation.
        """
        assert clean.dtype == np.float32, "expected float32 input"
        assert clean.ndim == 3 and clean.shape[2] == 3, "expected HxWx3"

        ops = self._sample_ops()
        meta: dict = {"ops": list(ops)}
        out = clean.copy()

        for op in ops:
            if op == "blur":
                sigma = float(self.rng.uniform(*self.cfg.blur_sigma_range))
                out = apply_gaussian_blur(out, sigma)
                meta["blur_sigma"] = sigma
            elif op == "noise":
                sigma = float(self.rng.uniform(*self.cfg.noise_sigma_range))
                out = apply_gaussian_noise(out, sigma, self.rng)
                meta["noise_sigma"] = sigma
            elif op == "jpeg":
                q = int(self.rng.integers(self.cfg.jpeg_quality_range[0],
                                          self.cfg.jpeg_quality_range[1] + 1))
                out = apply_jpeg(out, q)
                meta["jpeg_quality"] = q

        return np.clip(out, 0.0, 1.0).astype(np.float32), meta


# --------------------------------------------------------------------------- #
# Deterministic single-degradation factories (for per-type evaluation)
# --------------------------------------------------------------------------- #
def make_fixed_degrader(kind: str, **params):
    """
    Build a deterministic degrader for clean evaluation buckets.

    Examples:
        make_fixed_degrader("noise", sigma=25)
        make_fixed_degrader("blur", sigma=2.0)
        make_fixed_degrader("jpeg", quality=20)
        make_fixed_degrader("mixed", blur=2.0, sigma=25, quality=30)
    """
    def _fn(clean: np.ndarray, _rng=np.random.default_rng(0)) -> np.ndarray:
        out = clean.copy()
        if kind in ("blur", "mixed"):
            out = apply_gaussian_blur(out, params.get("blur", params.get("sigma_blur", 2.0)))
        if kind in ("noise", "mixed"):
            out = apply_gaussian_noise(out, params.get("sigma", 25.0), _rng)
        if kind in ("jpeg", "mixed"):
            out = apply_jpeg(out, params.get("quality", 20))
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    return _fn
