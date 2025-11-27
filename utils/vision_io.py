"""Utilities for loading/saving grayscale images as flat tensors."""

import math
from typing import Optional

import torch
from PIL import Image


def _side_length(vision_dim: int) -> int:
    side = int(math.isqrt(vision_dim))
    if side * side != vision_dim:
        raise ValueError(f"vision_dim={vision_dim} is not a perfect square.")
    return side


def load_image_as_tensor(path: str, vision_dim: int, device: torch.device) -> torch.Tensor:
    """Load an image file and convert to flattened grayscale tensor in [0,1]."""

    side = _side_length(vision_dim)
    img = Image.open(path).convert("L").resize((side, side))
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32, device=device) / 255.0
    if arr.numel() != vision_dim:
        raise ValueError(f"Loaded image size mismatch: expected {vision_dim}, got {arr.numel()}")
    return arr


def save_tensor_as_image(tensor: torch.Tensor, path: str) -> None:
    """Save a flat grayscale tensor as PNG."""

    flat = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu()
    vision_dim = flat.numel()
    side = _side_length(vision_dim)
    clamped = torch.clamp(flat, 0.0, 1.0)
    img = Image.fromarray((clamped.view(side, side).numpy() * 255).astype("uint8"), mode="L")
    img.save(path)
