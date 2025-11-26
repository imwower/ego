"""Interactive CLI for stepping the Ego-Sphere SNN with vision/text commands."""

import argparse
import sys
from typing import Dict, Optional

import torch

from config import default_hparams
from core import ProtoSelf, SNNEngine, LanguageCortex
from bridge import TeacherBridge


def vision_pattern(pattern_id: str, dim: int, device: torch.device) -> torch.Tensor:
    """Map a pattern id to a vision vector."""
    v = torch.zeros(dim, device=device)
    pid = pattern_id.lower()
    if pid == "cat":
        v[0:3] = torch.tensor([1.0, 0.8, 0.6], device=device)
    elif pid == "edge":
        v[: dim // 4] = 0.5
    elif pid == "dot":
        v[0] = 1.0
    elif pid == "noise":
        v = torch.rand(dim, device=device) * 0.5
    else:
        raise ValueError(f"Unknown vision pattern: {pattern_id}")
    return v


def run_interactive(error_threshold: float = 0.3, use_teacher: bool = True) -> None:
    hp = default_hparams()
    proto = ProtoSelf(hparams=hp)
    snn = SNNEngine(hparams=hp)
    cortex = LanguageCortex(hparams=hp)
    teacher = TeacherBridge(use_mock=not use_teacher, provider="codex_cli") if use_teacher else None

    print("Interactive Ego-Sphere. Commands: vision <pattern>, text <word>, status, quit")
    print("Patterns: cat, edge, dot, noise")

    while True:
        try:
            cmd = input("ego> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not cmd:
            continue
        if cmd in {"quit", "exit"}:
            break
        if cmd == "status":
            state = proto.state()
            print(f"energy={state['energy']:.3f} pain={state['pain']:.3f} curiosity={state['curiosity']:.3f}")
            continue

        tokens = cmd.split(maxsplit=1)
        if len(tokens) != 2 or tokens[0] not in {"vision", "text"}:
            print("Invalid command. Use: vision <pattern>, text <word>, status, quit")
            continue

        sensory: Dict[str, Optional[torch.Tensor]] = {"vision": torch.zeros(hp.vision_dim, device=hp.device), "text": None}

        if tokens[0] == "vision":
            try:
                sensory["vision"] = vision_pattern(tokens[1], hp.vision_dim, hp.device)
            except ValueError as exc:
                print(exc)
                continue
        elif tokens[0] == "text":
            sensory["text"] = cortex.text_to_spikes(tokens[1])

        # ProtoSelf noise keeps network alive; add to vision.
        somatic = proto.step()
        sensory["vision"] = sensory["vision"] + somatic[: hp.vision_dim]

        firing, pred_err = snn.step(sensory)

        print(
            f"err={pred_err['norm']:.3f} spikes: v={len(firing['vision'])} t={len(firing['text'])} a={len(firing['assoc'])}"
        )

        if teacher and pred_err["norm"] > error_threshold:
            ctx = {
                "prediction_error_norm": float(pred_err["norm"].item()),
                "proto_state": proto.state(),
                "recent_spikes": {k: v.tolist() for k, v in firing.items()},
                "notes": f"interactive cmd={cmd}",
            }
            bridge_out = teacher.ask_gemini(ctx, trigger_type="SURPRISE")
            status = "ok" if bridge_out.get("provider_ok") else "fallback"
            print("Teacher prompt:\n", bridge_out["prompt"])
            print(f"Teacher reply [{bridge_out['provider']}/{status}]: {bridge_out['reply']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Ego-Sphere CLI")
    parser.add_argument("--error-threshold", type=float, default=0.3)
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher bridge")
    args = parser.parse_args()
    run_interactive(error_threshold=args.error_threshold, use_teacher=not args.no_teacher)
