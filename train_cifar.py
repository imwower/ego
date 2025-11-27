"""Hebbian alignment training on CIFAR-100: vision-text dual injection."""

import argparse
import math
import os
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import default_hparams
from core import CheckpointManager, LanguageCortex, ProtoSelf, SNNEngine


def get_cifar100_loader(root: str, batch_size: int, img_size: int) -> Tuple[DataLoader, list]:
    """Download and prepare CIFAR-100 with grayscale resize and flatten."""

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return loader, trainset.classes


def train_cifar(
    epochs: int = 1,
    vision_dim: int = 256,
    batch_size: int = 1,
    hebb_lr: float = 0.05,
    save_every: int = 1000,
    log_every: int = 100,
    data_root: str = "./data",
    steps_per_sample: int = 1,
) -> None:
    img_size = int(math.sqrt(vision_dim))
    if img_size * img_size != vision_dim:
        raise ValueError("vision_dim must be a perfect square (e.g., 256 -> 16x16).")

    hp = default_hparams()
    hp.vision_dim = vision_dim

    proto = ProtoSelf(hparams=hp)
    snn = SNNEngine(hparams=hp)
    cortex = LanguageCortex(hparams=hp)
    ckpt_mgr = CheckpointManager(directory="checkpoints_cifar")

    loader, class_names = get_cifar100_loader(root=data_root, batch_size=batch_size, img_size=img_size)

    print(f"[init] Vision dim={hp.vision_dim}, device={hp.device}, classes={len(class_names)}")
    step_count = 0

    for epoch in range(epochs):
        for img_tensor, label_idx in loader:
            # Enforce batch_size==1 for simplicity; otherwise iterate items.
            if img_tensor.shape[0] != 1:
                img_tensor = img_tensor[0:1]
                label_idx = label_idx[0:1]

            label_name = class_names[int(label_idx[0])]
            vision_input = img_tensor.squeeze(0).to(hp.device)
            text_input = cortex.text_to_spikes(label_name)

            for _ in range(steps_per_sample):
                somatic = proto.step()
                sensory = {"vision": vision_input + somatic[: hp.vision_dim], "text": text_input}
                modulation = {"pain": proto.state()["pain"], "curiosity": proto.state()["curiosity"]}
                snn.step(sensory, modulation_signals=modulation)
                snn.update_weights_hebbian(learning_rate=hebb_lr, modulation_signals=modulation)

            step_count += 1

            if step_count % log_every == 0:
                print(f"[Epoch {epoch}] Step {step_count}: aligned '{label_name}'")

            if step_count % save_every == 0:
                path = ckpt_mgr.save(step=step_count, snn=snn, proto=proto, cortex=cortex)
                print(f"[save] checkpoint -> {path}")

    print("✅ Training complete: CIFAR-100 vision-text associations learned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hebbian alignment training on CIFAR-100")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--vision-dim", type=int, default=256, help="Must equal img_size*img_size")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hebb-lr", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--steps-per-sample", type=int, default=1)
    args = parser.parse_args()

    train_cifar(
        epochs=args.epochs,
        vision_dim=args.vision_dim,
        batch_size=args.batch_size,
        hebb_lr=args.hebb_lr,
        save_every=args.save_every,
        log_every=args.log_every,
        data_root=args.data_root,
        steps_per_sample=args.steps_per_sample,
    )
