"""
Qualitative visualization: degraded | diffusion | best-baseline | ground-truth.

Generates a comparison figure across degradation buckets for a handful of images.
Pure matplotlib so it renders inline in the Kaggle notebook.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from baselines import classical_blind
from degradations import make_fixed_degrader
from diffusion import Diffusion, DiffusionConfig
from metrics import psnr
from model import build_unet_from_pretrained, build_unet_from_scratch


def load_one(path, size=256):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_tensor(img, device):
    t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return (t * 2 - 1).to(device)


def from_tensor(t):
    x = (t.squeeze(0).cpu().float() + 1) / 2
    return x.clamp(0, 1).numpy().transpose(1, 2, 0)


def make_figure(data_dir, ckpt_path, out_path, n_images=3, size=256,
                ddim_steps=50, pretrained=""):
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    patch = ckpt.get("args", {}).get("patch", 128)
    model = (build_unet_from_pretrained(pretrained) if pretrained
             else build_unet_from_scratch(patch)).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = Diffusion(DiffusionConfig(
        timesteps=ckpt.get("args", {}).get("timesteps", 1000),
        schedule=ckpt.get("args", {}).get("schedule", "cosine")), device=device)

    deg = make_fixed_degrader("mixed", blur=1.5, sigma=20, quality=30)
    paths = sorted(sum([glob.glob(os.path.join(data_dir, f"**/*.{e}"), recursive=True)
                        for e in ("png", "jpg", "jpeg")], []))[:n_images]

    cols = ["Degraded", "Diffusion (ours)", "Classical", "Ground truth"]
    fig, axes = plt.subplots(n_images, 4, figsize=(14, 3.6 * n_images))
    if n_images == 1:
        axes = axes[None, :]

    for r, p in enumerate(paths):
        clean = load_one(p, size)
        degraded = deg(clean)
        with torch.no_grad():
            restored = from_tensor(diffusion.ddim_sample(
                model, to_tensor(degraded, device), steps=ddim_steps))
        classical = np.clip(classical_blind(degraded), 0, 1)

        panels = [degraded, restored, classical, clean]
        scores = [psnr(degraded, clean), psnr(restored, clean),
                  psnr(classical, clean), float("inf")]
        for c, (img, sc) in enumerate(zip(panels, scores)):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            title = cols[c] if r == 0 else ""
            psnr_txt = "" if sc == float("inf") else f"\n{sc:.2f} dB"
            axes[r, c].set_title(title + (psnr_txt if r == 0 else
                                          (f"{sc:.2f} dB" if sc != float('inf') else "")),
                                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pretrained", default="")
    args = ap.parse_args()
    make_figure(args.data, args.ckpt, args.out, n_images=args.n, pretrained=args.pretrained)
