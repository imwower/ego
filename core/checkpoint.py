"""Checkpoint manager for Ego-Sphere components."""

import os
from typing import Any, Dict

import torch


class CheckpointManager:
    def __init__(self, directory: str = "checkpoints") -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def save(self, step: int, snn: Any, proto: Any, cortex: Any, extras: Dict[str, Any] = None) -> str:
        """Save state_dicts and Proto-Self scalars to a .pt file."""

        state = {
            "step": step,
            "snn": snn.state_dict(),
            "proto": {
                "energy": proto.energy,
                "pain": proto.pain,
                "curiosity": proto.curiosity,
                "hparams": proto.hparams,
            },
            "cortex": cortex.state_dict() if hasattr(cortex, "state_dict") else {},
        }
        if extras:
            state["extras"] = extras
        path = os.path.join(self.directory, f"checkpoint_step_{step}.pt")
        torch.save(state, path)
        return path

    def load(self, path: str, snn: Any, proto: Any, cortex: Any) -> Dict[str, Any]:
        """Load a checkpoint into provided modules and return metadata."""

        ckpt = torch.load(path, map_location="mps", weights_only=False)
        snn.load_state_dict(ckpt["snn"])
        if "state_dict" in dir(cortex):
            cortex.load_state_dict(ckpt.get("cortex", {}))

        proto.energy = ckpt["proto"]["energy"].to(proto.device)
        proto.pain = ckpt["proto"]["pain"].to(proto.device)
        proto.curiosity = ckpt["proto"]["curiosity"].to(proto.device)

        return {"step": ckpt.get("step", 0), "extras": ckpt.get("extras")}


__all__ = ["CheckpointManager"]
