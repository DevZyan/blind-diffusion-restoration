"""
Conditional denoising U-Net.

Uses the diffusers `UNet2DModel` as the backbone. The network takes a 6-channel
input (noisy target xt | degraded condition) and predicts the 3-channel noise.

Two init paths:
  * from_scratch    -- compact U-Net sized for a Kaggle T4 (recommended, reliable)
  * from_pretrained -- load a pretrained DDPM U-Net and surgically expand the first
                       conv from 3 to 6 input channels (clean weights copied,
                       condition weights zero-init so training starts from the
                       pretrained prior). Requires internet on the Kaggle kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel


def build_unet_from_scratch(sample_size: int = 128) -> UNet2DModel:
    """Compact conditional U-Net (~30-40M params). Fits comfortably on a T4."""
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=6,            # xt (3) + condition (3)
        out_channels=3,           # predicted noise
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )


def _expand_first_conv(unet: UNet2DModel) -> UNet2DModel:
    """Turn a 3-channel-input U-Net into a 6-channel one, preserving prior."""
    old = unet.conv_in
    if old.in_channels == 6:
        return unet

    new = nn.Conv2d(6, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight          # noisy-image path: keep prior
        # condition channels (3:6) left at zero -> identity start
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.config.in_channels = 6
    return unet


def build_unet_from_pretrained(model_id: str = "google/ddpm-cifar10-32") -> UNet2DModel:
    """
    Load a pretrained DDPM U-Net and adapt it for 6-channel conditional input.

    NOTE: pretrained DDPM checkpoints are trained at fixed resolutions. For
    restoration at 128px we train on patches; the U-Net is fully convolutional
    so it accepts arbitrary sizes at inference. Choose a checkpoint whose
    architecture is close to your target compute budget.
    """
    unet = UNet2DModel.from_pretrained(model_id)
    return _expand_first_conv(unet)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class EMA:
    """Exponential moving average of model weights -- stabilizes diffusion samples."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
