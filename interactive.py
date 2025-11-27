"""Interactive CLI loop for Ego-Sphere with sensory input, sleep, and checkpoints."""

import argparse
import os
import random
import re
from typing import Dict, Optional

import torch
import torchvision
import torchvision.transforms as transforms

from config import default_hparams
from core import CheckpointManager, EpisodicMemory, LanguageCortex, ProtoSelf, SNNEngine
from bridge import TeacherBridge
from utils.vision_io import load_image_as_tensor, save_tensor_as_image


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

    # Preload CIFAR test split for describe/test_cifar.
    transform = transforms.Compose(
        [
            transforms.Resize((int(hp.vision_dim ** 0.5), int(hp.vision_dim ** 0.5))),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    cifar_test = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    class_to_indices: Dict[str, list] = {}
    for idx, (_, label_idx) in enumerate(cifar_test):
        label_name = cifar_test.classes[label_idx]
        class_to_indices.setdefault(label_name, []).append(idx)

    print("Interactive Ego-Sphere. Commands: test_cifar <class>, imagine <class>, read <word>, sleep, status, save, load <step>, quit")

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

        if len(tokens) == 2 and tokens[0] == "test_cifar":
            class_name = tokens[1]
            indices = class_to_indices.get(class_name)
            if not indices:
                print(f"[error] class '{class_name}' not found in CIFAR-100.")
                continue
            idx = random.choice(indices)
            vision_tensor, label_idx = cifar_test[idx]
            pred_text_sum = torch.zeros(hp.text_dim, device=hp.device)
            for _ in range(20):
                somatic = proto.step()
                sensory = {"vision": vision_tensor.to(hp.device) + somatic[: hp.vision_dim], "text": None}
                _, _, preds = snn.step(sensory)
                pred_text_sum += preds["text"]
            avg_pred_text = pred_text_sum / 20.0
            words = cortex.spikes_to_text(avg_pred_text, k=1)
            guess = words[0] if words else "unknown"
            print(f"I see a {guess} (True: {class_name})")
            continue

        if len(tokens) == 2 and tokens[0] == "imagine":
            class_name = tokens[1]
            text_spikes = cortex.text_to_spikes(class_name)
            pred_vision_sum = torch.zeros(hp.vision_dim, device=hp.device)
            for _ in range(20):
                somatic = proto.step()
                sensory = {"vision": somatic[: hp.vision_dim], "text": text_spikes}
                _, _, preds = snn.step(sensory)
                pred_vision_sum += preds["vision"]
            avg_pred_vision = pred_vision_sum / 20.0
            out_path = "imagination.png"
            save_tensor_as_image(avg_pred_vision, out_path)
            print(f"I imagined this image (saved to {out_path})")
            continue

        if len(tokens) == 2 and tokens[0] == "read":
            sensory: Dict[str, Optional[torch.Tensor]] = {
                "vision": torch.zeros(hp.vision_dim, device=hp.device),
                "text": cortex.text_to_spikes(tokens[1]),
            }
        else:
            print("Invalid command. Use: test_cifar <class>, imagine <class>, read <word>, sleep, status, save, load <path>, quit")
            continue

        # Closed-loop read path: run a short rollout with text input only.
        print(f"Thinking...", end="", flush=True)
        total_spikes = {"vision": 0, "text": 0, "assoc": 0}
        max_err = 0.0
        last_firing = None
        last_pred_err = None
        last_modulation = None
        last_vision = sensory["vision"]

        for _ in range(20):
            somatic = proto.step()
            current_vision = sensory["vision"] + somatic[: hp.vision_dim]
            last_vision = current_vision

            state_mod = proto.state()
            modulation = {"pain": state_mod["pain"], "curiosity": state_mod["curiosity"]}

            step_input = {"vision": current_vision, "text": sensory["text"]}
            firing, pred_err, _ = snn.step(step_input, modulation_signals=modulation)
            last_firing = firing
            last_pred_err = pred_err
            last_modulation = modulation

            total_spikes["vision"] += len(firing["vision"])
            total_spikes["text"] += len(firing["text"])
            total_spikes["assoc"] += len(firing["assoc"])
            max_err = max(max_err, pred_err["norm"].item())

        print(f"\rAction complete. Max Err={max_err:.3f}")
        print(f"Total Spikes: v={total_spikes['vision']} t={total_spikes['text']} a={total_spikes['assoc']}")

        if teacher and max_err > error_threshold:
            ctx = {
                "prediction_error_norm": float(max_err),
                "proto_state": proto.state(),
                "recent_spikes": {k: v.tolist() for k, v in (last_firing or {}).items()},
                "notes": f"interactive cmd={cmd}",
            }
            bridge_out = teacher.ask_gemini(ctx, trigger_type="SURPRISE")
            status = "ok" if bridge_out.get("provider_ok") else "fallback"
            print("Teacher prompt:\n", bridge_out["prompt"])
            print(f"Teacher reply [{bridge_out['provider']}/{status}]: {bridge_out['reply']}")
            try:
                embedding = snn.spikes_assoc.detach().to("cpu").float().tolist()
                episodic.store_experience("SURPRISE", bridge_out["reply"], embedding)
                concept = extract_concept_word(bridge_out["reply"])
                if concept:
                    teacher_spikes = cortex.text_to_spikes(concept)
                    reinforce_input = {"vision": sensory["vision"], "text": teacher_spikes}
                    snn.step(reinforce_input, modulation_signals=last_modulation)
                    snn.update_weights_hebbian(learning_rate=0.1, modulation_signals=last_modulation)
                    print(f"Learned connection: vision_state <-> {concept}")
            except Exception as exc:
                print(f"[warn] episodic store failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Ego-Sphere CLI")
    parser.add_argument("--error-threshold", type=float, default=0.3)
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher bridge")
    args = parser.parse_args()
    run_interactive(error_threshold=args.error_threshold, use_teacher=not args.no_teacher)
