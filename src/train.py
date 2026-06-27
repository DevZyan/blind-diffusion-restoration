"""
Training entry point for the conditional diffusion restoration model.

Designed for a single Kaggle T4/P100 session:
  * mixed precision (AMP)
  * gradient accumulation
  * EMA weights
  * periodic checkpointing + sample previews
  * resumable from checkpoint

Usage (from the Kaggle notebook or CLI):
    python train.py --data /kaggle/input/div2k/DIV2K_train_HR \
                    --out /kaggle/working/ckpt --steps 20000
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import DIV2KRestoration, tensor_to_image
from degradations import DegradationConfig
from diffusion import Diffusion, DiffusionConfig
from model import EMA, build_unet_from_pretrained, build_unet_from_scratch, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="DIV2K HR training dir")
    p.add_argument("--out", default="./ckpt")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--pretrained", default="", help="diffusers model id; empty = from scratch")
    p.add_argument("--resume", default="")
    return p.parse_args()


def save_sample_grid(diffusion, model, batch, path, device):
    import numpy as np
    from PIL import Image

    model.eval()
    cond = batch["degraded"][:4].to(device)
    clean = batch["clean"][:4]
    with torch.no_grad():
        restored = diffusion.ddim_sample(model, cond, steps=50)

    rows = []
    for i in range(cond.shape[0]):
        d = tensor_to_image(cond[i])
        r = tensor_to_image(restored[i])
        c = tensor_to_image(clean[i])
        rows.append(np.concatenate([d, r, c], axis=1))
    grid = (np.concatenate(rows, axis=0) * 255).astype("uint8")
    Image.fromarray(grid).save(path)
    model.train()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # data
    ds = DIV2KRestoration(args.data, patch_size=args.patch, train=True,
                          deg_config=DegradationConfig(), repeat=100)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"dataset images: {len(ds.paths)}  effective length: {len(ds)}")

    # model + diffusion
    if args.pretrained:
        model = build_unet_from_pretrained(args.pretrained).to(device)
        print(f"loaded pretrained backbone: {args.pretrained}")
    else:
        model = build_unet_from_scratch(args.patch).to(device)
        print("built U-Net from scratch")
    print("params:", count_params(model))

    diffusion = Diffusion(DiffusionConfig(timesteps=args.timesteps,
                                          schedule=args.schedule), device=device)
    ema = EMA(model, decay=args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow = ckpt["ema"]
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}")

    model.train()
    data_iter = iter(dl)
    t0 = time.time()
    running = 0.0

    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            x0 = batch["clean"].to(device)
            cond = batch["degraded"].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = diffusion.training_loss(model, x0, cond) / args.accum
            scaler.scale(loss).backward()
            running += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        if (step + 1) % 50 == 0:
            speed = 50 / (time.time() - t0)
            print(f"step {step+1:6d}/{args.steps} | loss {running/50:.4f} "
                  f"| {speed:.2f} it/s")
            running = 0.0
            t0 = time.time()

        if (step + 1) % args.sample_every == 0:
            ema_model = build_unet_from_scratch(args.patch).to(device) \
                if not args.pretrained else build_unet_from_pretrained(args.pretrained).to(device)
            ema.copy_to(ema_model)
            save_sample_grid(diffusion, ema_model, batch,
                             os.path.join(args.out, f"sample_{step+1}.png"), device)
            del ema_model
            print(f"  saved preview sample_{step+1}.png")

        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict(),
                "step": step + 1,
                "args": vars(args),
            }, os.path.join(args.out, "latest.pt"))
            print(f"  checkpoint saved at step {step+1}")

    print("training complete")


if __name__ == "__main__":
    main()
