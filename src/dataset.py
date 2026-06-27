"""
DIV2K dataset with on-the-fly blind degradation.

Training pairs are generated dynamically: a random HR patch is cropped from a
clean DIV2K image, degraded with a freshly sampled blind degradation chain, and
returned as (degraded, clean). Because degradations are re-sampled every epoch the
effective dataset is enormous, which matters given DIV2K has only 800 train images.

Images are returned as torch tensors in [-1, 1] (the range diffusion expects).
"""

from __future__ import annotations

import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from degradations import BlindDegrader, DegradationConfig


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWx3 [0,1] float -> 3xHxW [-1,1] tensor."""
    t = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return t * 2.0 - 1.0


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """3xHxW [-1,1] tensor -> HxWx3 [0,1] numpy."""
    x = (t.detach().cpu().float() + 1.0) / 2.0
    x = x.clamp(0, 1).numpy().transpose(1, 2, 0)
    return x


class DIV2KRestoration(Dataset):
    def __init__(
        self,
        root: str,
        patch_size: int = 128,
        train: bool = True,
        deg_config: Optional[DegradationConfig] = None,
        repeat: int = 1,
        exts: tuple[str, ...] = ("png", "jpg", "jpeg"),
    ):
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")

        self.patch_size = patch_size
        self.train = train
        self.repeat = repeat
        self.deg_config = deg_config or DegradationConfig()
        # one degrader per worker is created lazily to avoid shared RNG state
        self._degrader: Optional[BlindDegrader] = None

    def __len__(self) -> int:
        return len(self.paths) * self.repeat

    def _degrader_for_worker(self) -> BlindDegrader:
        if self._degrader is None:
            info = torch.utils.data.get_worker_info()
            seed = None if info is None else (info.seed % (2 ** 32))
            self._degrader = BlindDegrader(self.deg_config, seed=seed)
        return self._degrader

    def _load(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0

    def _crop(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w, _ = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            # pad small images by reflection
            img = np.pad(
                img,
                ((0, max(0, ps - h)), (0, max(0, ps - w)), (0, 0)),
                mode="reflect",
            )
            h, w, _ = img.shape
        if self.train:
            top = int(rng.integers(0, h - ps + 1))
            left = int(rng.integers(0, w - ps + 1))
        else:
            top = (h - ps) // 2
            left = (w - ps) // 2
        return img[top:top + ps, left:left + ps]

    def __getitem__(self, idx: int):
        path = self.paths[idx % len(self.paths)]
        degrader = self._degrader_for_worker()
        clean = self._load(path)
        clean = self._crop(clean, degrader.rng)

        if self.train and degrader.rng.random() < 0.5:
            clean = clean[:, ::-1].copy()  # horizontal flip

        degraded, meta = degrader.degrade(clean)
        return {
            "degraded": _to_tensor(degraded),
            "clean": _to_tensor(clean),
        }
