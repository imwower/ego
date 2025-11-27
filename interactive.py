"""Interactive CLI loop for Ego-Sphere with sensory input, sleep, and checkpoints."""

import argparse
import os
import re
from typing import Dict, Optional

import torch

from config import default_hparams
from core import CheckpointManager, EpisodicMemory, LanguageCortex, ProtoSelf, SNNEngine
from bridge import TeacherBridge


def vision_pattern(pattern_id: str, dim: int, device: torch.device) -> torch.Tensor:
    """Map a pattern id to a vision vector (synthetic stimuli)."""
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


def extract_concept_word(reply: str) -> Optional[str]:
    """Prefer final-line English token; fallback to first English token."""

    if not reply:
        return None
    last_line = reply.strip().splitlines()[-1].strip()
    if re.fullmatch(r"[A-Za-z]+", last_line):
        return last_line
    match = re.search(r"[A-Za-z]+", reply)
    return match.group(0) if match else None


def dream_consolidation(
    proto: ProtoSelf,
    snn: SNNEngine,
    cortex: LanguageCortex,
    episodic: EpisodicMemory,
    samples: int = 3,
    learning_rate: float = 0.02,
    curiosity_drive: float = 0.02,
) -> None:
    """Replay sampled memories and consolidate via Hebbian updates."""

    hp = cortex.hparams
    memories = episodic.sample_memories(k=samples)
    if not memories:
        print("[dream] no stored memories; skipping.")
        return

    for mem in memories:
        text = mem.get("document") or ""
        text_spikes = cortex.text_to_spikes(text)
        somatic = proto.step(curiosity_drive=curiosity_drive)
        sensory = {"vision": somatic[: hp.vision_dim], "text": text_spikes}
        mod = {"pain": proto.state()["pain"], "curiosity": proto.state()["curiosity"]}
        snn.step(sensory, modulation_signals=mod)
        snn.update_weights_hebbian(learning_rate=learning_rate, modulation_signals=mod)
    print(f"[dream] replayed {len(memories)} memories")


def run_interactive(error_threshold: float = 0.3, use_teacher: bool = True, ckpt_dir: str = "checkpoints") -> None:
    hp = default_hparams()
    proto = ProtoSelf(hparams=hp)
    snn = SNNEngine(hparams=hp)
    cortex = LanguageCortex(hparams=hp)
    teacher = TeacherBridge(use_mock=not use_teacher, provider="codex_cli") if use_teacher else None
    episodic = EpisodicMemory(persist_directory="data/chroma_store")
    ckpt_mgr = CheckpointManager(directory=ckpt_dir)

    print("Interactive Ego-Sphere. Commands: see <pattern>, read <word>, sleep, status, save, load <step>, quit")
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
        if cmd == "sleep":
            try:
                dream_consolidation(
                    proto=proto,
                    snn=snn,
                    cortex=cortex,
                    episodic=episodic,
                    samples=3,
                    learning_rate=0.03,
                    curiosity_drive=0.02,
                )
            except Exception as exc:
                print(f"[warn] dream consolidation failed: {exc}")
            continue
        if cmd == "save":
            path = ckpt_mgr.save(step=0, snn=snn, proto=proto, cortex=cortex)
            print(f"Saved checkpoint to {path}")
            continue
        if cmd.startswith("load"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 2:
                path = parts[1]
            else:
                # load latest by sorting
                try:
                    files = sorted(os.listdir(ckpt_dir))
                    path = os.path.join(ckpt_dir, files[-1])
                except Exception:
                    print("No checkpoint found.")
                    continue
            meta = ckpt_mgr.load(path, snn=snn, proto=proto, cortex=cortex)
            print(f"Loaded checkpoint from {path}, step={meta.get('step')}")
            continue

        tokens = cmd.split(maxsplit=1)
        if len(tokens) != 2 or tokens[0] not in {"see", "read"}:
            print("Invalid command. Use: see <pattern>, read <word>, sleep, status, save, load <path>, quit")
            continue

        sensory: Dict[str, Optional[torch.Tensor]] = {"vision": torch.zeros(hp.vision_dim, device=hp.device), "text": None}

        if tokens[0] == "see":
            try:
                sensory["vision"] = vision_pattern(tokens[1], hp.vision_dim, hp.device)
            except ValueError as exc:
                print(exc)
                continue
        elif tokens[0] == "read":
            sensory["text"] = cortex.text_to_spikes(tokens[1])

        # ProtoSelf noise keeps network alive; add to vision.
        somatic = proto.step()
        sensory["vision"] = sensory["vision"] + somatic[: hp.vision_dim]

        state_mod = proto.state()
        modulation = {"pain": state_mod["pain"], "curiosity": state_mod["curiosity"]}
        firing, pred_err = snn.step(sensory, modulation_signals=modulation)

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
            # Store episodic memory grounded to assoc spikes.
            try:
                embedding = snn.spikes_assoc.detach().to("cpu").float().tolist()
                episodic.store_experience("SURPRISE", bridge_out["reply"], embedding)
                concept = extract_concept_word(bridge_out["reply"])
                if concept:
                    teacher_spikes = cortex.text_to_spikes(concept)
                    reinforce_input = {"vision": sensory["vision"], "text": teacher_spikes}
                    snn.step(reinforce_input, modulation_signals=modulation)
                    snn.update_weights_hebbian(learning_rate=0.05, modulation_signals=modulation)
            except Exception as exc:
                print(f"[warn] episodic store failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Ego-Sphere CLI")
    parser.add_argument("--error-threshold", type=float, default=0.3)
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher bridge")
    args = parser.parse_args()
    run_interactive(error_threshold=args.error_threshold, use_teacher=not args.no_teacher)
