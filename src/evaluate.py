"""
Evaluation: build per-degradation test buckets, run the diffusion model and all
classical baselines, and emit a comparison table (PSNR / SSIM / LPIPS).

Buckets mirror real evaluation practice -- each degradation type is held fixed so
you can see where the blind model wins and where it struggles:

    noise_25, noise_50, blur_2.0, jpeg_20, jpeg_40, mixed

Produces:
    results.csv   -- full table
    results.md    -- markdown table for the README
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import torch
from PIL import Image

from baselines import BASELINES
from degradations import make_fixed_degrader
from diffusion import Diffusion, DiffusionConfig
from metrics import MetricBank
from model import build_unet_from_pretrained, build_unet_from_scratch, EMA


BUCKETS = {
    "noise_25":  make_fixed_degrader("noise", sigma=25),
    "noise_50":  make_fixed_degrader("noise", sigma=50),
    "blur_2.0":  make_fixed_degrader("blur", blur=2.0),
    "jpeg_20":   make_fixed_degrader("jpeg", quality=20),
    "jpeg_40":   make_fixed_degrader("jpeg", quality=40),
    "mixed":     make_fixed_degrader("mixed", blur=1.5, sigma=20, quality=30),
}


def load_images(root, limit=None, size=256):
    paths = sorted(sum([glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
                        for e in ("png", "jpg", "jpeg")], []))
    if limit:
        paths = paths[:limit]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        # center-crop a square then resize for consistent eval
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((size, size), Image.BICUBIC)
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    return imgs


def to_tensor(img, device):
    t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return (t * 2 - 1).to(device)


def from_tensor(t):
    x = (t.squeeze(0).cpu().float() + 1) / 2
    return x.clamp(0, 1).numpy().transpose(1, 2, 0)


def evaluate_model(diffusion, model, imgs, device, ddim_steps, use_lpips):
    results = {}
    for name, deg_fn in BUCKETS.items():
        bank = MetricBank(use_lpips=use_lpips, device=device)
        for clean in imgs:
            degraded = deg_fn(clean)
            cond = to_tensor(degraded, device)
            with torch.no_grad():
                restored = diffusion.ddim_sample(model, cond, steps=ddim_steps)
            bank.update(from_tensor(restored), clean)
        results[name] = bank.result()
        print(f"  [model] {name:10s} {results[name]}")
    return results


def evaluate_baseline(baseline_fn, imgs, use_lpips, device):
    results = {}
    for name, deg_fn in BUCKETS.items():
        bank = MetricBank(use_lpips=use_lpips, device=device)
        for clean in imgs:
            degraded = deg_fn(clean)
            bank.update(np.clip(baseline_fn(degraded), 0, 1), clean)
        results[name] = bank.result()
    return results


def write_tables(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    buckets = list(BUCKETS.keys())

    # CSV
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric"] + buckets)
        for method, res in all_results.items():
            for metric in ("psnr", "ssim", "lpips"):
                if metric in res[buckets[0]]:
                    row = [method, metric] + [f"{res[b].get(metric, float('nan')):.4f}"
                                              for b in buckets]
                    w.writerow(row)

    # Markdown (PSNR / SSIM focus)
    lines = ["### PSNR (dB) -- higher is better\n",
             "| Method | " + " | ".join(buckets) + " |",
             "|" + "---|" * (len(buckets) + 1)]
    for method, res in all_results.items():
        lines.append("| " + method + " | " +
                     " | ".join(f"{res[b]['psnr']:.2f}" for b in buckets) + " |")
    lines.append("\n### SSIM -- higher is better\n")
    lines.append("| Method | " + " | ".join(buckets) + " |")
    lines.append("|" + "---|" * (len(buckets) + 1))
    for method, res in all_results.items():
        lines.append("| " + method + " | " +
                     " | ".join(f"{res[b]['ssim']:.3f}" for b in buckets) + " |")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="DIV2K HR val dir")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="./results")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--pretrained", default="")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--baselines", default="identity,nlm,wiener,classical_blind")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = load_images(args.data, limit=args.limit, size=args.size)
    print(f"loaded {len(imgs)} eval images")

    # load model (use EMA weights)
    ckpt = torch.load(args.ckpt, map_location=device)
    patch = ckpt.get("args", {}).get("patch", 128)
    model = (build_unet_from_pretrained(args.pretrained) if args.pretrained
             else build_unet_from_scratch(patch)).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = Diffusion(DiffusionConfig(
        timesteps=ckpt.get("args", {}).get("timesteps", 1000),
        schedule=ckpt.get("args", {}).get("schedule", "cosine")), device=device)

    all_results = {}
    print("\nEvaluating diffusion model...")
    all_results["diffusion (ours)"] = evaluate_model(
        diffusion, model, imgs, device, args.ddim_steps, args.lpips)

    for b in args.baselines.split(","):
        b = b.strip()
        if b in BASELINES:
            print(f"\nEvaluating baseline: {b}")
            all_results[b] = evaluate_baseline(BASELINES[b], imgs, args.lpips, device)

    write_tables(all_results, args.out)


if __name__ == "__main__":
    main()
