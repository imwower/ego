"""Ego-Sphere simulation loops with dual coding, curiosity-driven prompts, and dream consolidation.

- Demo mode mirrors the original timeline (vision: Cat, text conflict: Dog).
- Active learning mode (Phase 3) supports text scripts, dual vision/text binding,
  curiosity-triggered teacher queries, and offline "dream" replay from memory.
"""

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from config import default_hparams
from core import LanguageCortex, MemoryBank, ProtoSelf, SNNEngine
from bridge import TeacherBridge


@dataclass
class StimulusWindow:
    start: int
    end: int
    text: Optional[str] = None
    vision_pattern: Optional[str] = None


def vision_pattern(pattern_id: str, dim: int, device: torch.device) -> torch.Tensor:
    """Map a pattern id to a synthetic vision vector."""

    v = torch.zeros(dim, device=device)
    pid = pattern_id.lower()
    if pid == "cat":
        v[0:3] = torch.tensor([1.0, 0.8, 0.6], device=device)
    elif pid == "dog":
        v[1:4] = torch.tensor([0.9, 0.7, 0.5], device=device)
    elif pid == "edge":
        v[: dim // 4] = 0.5
    elif pid == "dot":
        v[0] = 1.0
    elif pid == "noise":
        v = torch.rand(dim, device=device) * 0.5
    else:
        raise ValueError(f"Unknown vision pattern: {pattern_id}")
    return v


def default_child_stimuli() -> List[StimulusWindow]:
    """Built-in Phase 3 script with paired and conflicting cues."""

    return [
        StimulusWindow(start=15, end=45, text="cat animal", vision_pattern="cat"),
        StimulusWindow(start=55, end=80, text="striped edge", vision_pattern="edge"),
        StimulusWindow(start=95, end=120, text="dog bark", vision_pattern="dot"),  # text/vision conflict
        StimulusWindow(start=130, end=165, text="novel toy", vision_pattern="noise"),
    ]


def parse_stimulus_line(line: str) -> Optional[StimulusWindow]:
    """Parse lines like '10-30;cat animal;cat' into a stimulus window."""

    parts = [p.strip() for p in line.split(";") if p.strip()]
    if not parts:
        return None

    time_part = parts[0]
    if "-" in time_part:
        start_str, end_str = time_part.split("-", maxsplit=1)
        start, end = int(start_str), int(end_str)
    else:
        start = end = int(time_part)

    text = parts[1] if len(parts) >= 2 else None
    vision = parts[2] if len(parts) >= 3 else None

    return StimulusWindow(start=start, end=end, text=text or None, vision_pattern=vision or None)


def load_stimuli(script_path: Optional[str]) -> List[StimulusWindow]:
    """Load stimuli from a user script or fall back to defaults."""

    if script_path and os.path.exists(script_path):
        stimuli: List[StimulusWindow] = []
        with open(script_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                stim = parse_stimulus_line(stripped)
                if stim:
                    stimuli.append(stim)
        if stimuli:
            return stimuli
        print(f"[warn] No valid stimuli found in {script_path}; using defaults.")
    return default_child_stimuli()


def compose_sensory(t: int, cortex: LanguageCortex, stimuli: List[StimulusWindow]) -> Dict[str, Optional[torch.Tensor]]:
    """Compose sensory inputs at time t from the stimulus schedule."""

    hp = cortex.hparams
    vision = torch.zeros(hp.vision_dim, device=hp.device)
    text = None

    for stim in stimuli:
        if stim.start <= t < stim.end:
            if stim.vision_pattern:
                vision = vision + vision_pattern(stim.vision_pattern, hp.vision_dim, hp.device)
            if stim.text:
                spikes = cortex.text_to_spikes(stim.text)
                text = spikes if text is None else torch.maximum(text, spikes)

    return {"vision": vision, "text": text}


def extract_concept_word(reply: str) -> Optional[str]:
    """Use the final line (Stage 3 rule) or any English token as the concept."""

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
    memory: Optional[MemoryBank],
    epochs: int,
    samples_per_epoch: int,
    curiosity_drive: float,
    learning_rate: float,
) -> None:
    """Offline replay: sample stored teacher replies and rehearse them."""

    hp = cortex.hparams
    for epoch in range(epochs):
        memories = memory.sample_memories(k=samples_per_epoch) if memory else []
        if not memories:
            memories = [{"document": "anticipation", "embedding": None, "metadata": {}} for _ in range(samples_per_epoch)]

        for mem in memories:
            text = mem.get("document") or ""
            text_spikes = cortex.text_to_spikes(text)
            somatic = proto.step(curiosity_drive=curiosity_drive)
            sensory = {"vision": somatic[: hp.vision_dim], "text": text_spikes}
            mod = {"pain": proto.state()["pain"], "curiosity": proto.state()["curiosity"]}
            snn.step(sensory, modulation_signals=mod)
            snn.update_weights_hebbian(learning_rate=learning_rate, modulation_signals=mod)
        print(f"[dream] epoch {epoch + 1}/{epochs} replayed {len(memories)} memories")


def run_loop(stimuli: List[StimulusWindow], args: argparse.Namespace, label: str) -> None:
    hp = default_hparams()
    proto = ProtoSelf(hparams=hp)
    snn = SNNEngine(hparams=hp)
    cortex = LanguageCortex(hparams=hp)

    teacher = None
    if not args.no_teacher:
        teacher = TeacherBridge(use_mock=args.mock_teacher, provider=args.provider)

    memory: Optional[MemoryBank] = None
    try:
        memory = MemoryBank(persist_directory="data/chroma_store")
    except RuntimeError as exc:
        print(f"[warn] MemoryBank disabled: {exc}")

    last_teacher_step = -args.curiosity_cooldown

    print(
        f"[{label}] steps={args.steps} mode={label} provider={teacher.provider if teacher else 'none'} "
        f"text_script={'custom' if args.text_script else 'default'}"
    )

    for t in range(args.steps):
        sensory = compose_sensory(t, cortex, stimuli)
        somatic = proto.step(curiosity_drive=args.curiosity_drive)
        sensory["vision"] = sensory["vision"] + somatic[: hp.vision_dim]

        state_mod = proto.state()
        modulation = {"pain": state_mod["pain"], "curiosity": state_mod["curiosity"]}

        firing, pred_err = snn.step(sensory, modulation_signals=modulation)

        trigger_type = None
        trigger_reason = None
        if pred_err["norm"] > args.error_threshold:
            trigger_type = "SURPRISE"
            trigger_reason = f"error={pred_err['norm']:.3f}"
        elif (
            state_mod["curiosity"] > args.curiosity_threshold
            and (t - last_teacher_step) >= args.curiosity_cooldown
        ):
            trigger_type = "CONFUSION"
            trigger_reason = f"curiosity={state_mod['curiosity']:.3f}"

        bridge_out = None
        if trigger_type and teacher:
            ctx = {
                "prediction_error_norm": float(pred_err["norm"].item()),
                "proto_state": state_mod,
                "recent_spikes": {k: v.tolist() for k, v in firing.items()},
                "notes": f"{label} t={t} reason={trigger_reason}",
            }
            bridge_out = teacher.ask_gemini(ctx, trigger_type=trigger_type)
            last_teacher_step = t

            if memory is not None:
                embedding = snn.spikes_assoc.detach().to("cpu").float().tolist()
                metadata = {
                    "timestamp": time.time(),
                    "trigger_type": trigger_type,
                    "step": t,
                    "reason": trigger_reason,
                }
                memory.add_memory(bridge_out["reply"], embedding=embedding, metadata=metadata)

            concept = extract_concept_word(bridge_out["reply"])
            if concept:
                teacher_spikes = cortex.text_to_spikes(concept)
                reinforce_input = {"vision": sensory["vision"], "text": teacher_spikes}
                snn.step(reinforce_input, modulation_signals=modulation)
                snn.update_weights_hebbian(
                    learning_rate=args.teacher_learning_rate,
                    modulation_signals=modulation,
                )
        elif t % args.background_hebbian_interval == 0:
            snn.update_weights_hebbian(
                learning_rate=args.background_learning_rate,
                modulation_signals=modulation,
            )

        if t % args.log_every == 0 or bridge_out:
            print(
                f"t={t:03d} | energy={state_mod['energy']:.2f} pain={state_mod['pain']:.2f} "
                f"curiosity={state_mod['curiosity']:.2f} | err={pred_err['norm']:.3f} "
                f"spikes={{v:{len(firing['vision'])}, t:{len(firing['text'])}, a:{len(firing['assoc'])}}}"
            )
            if bridge_out:
                status = "ok" if bridge_out.get("provider_ok") else "fallback"
                print("  Teacher prompt:")
                print(f"    {bridge_out['prompt']}")
                print(f"  Teacher reply [{bridge_out['provider']}/{status}]: {bridge_out['reply']}")

    if args.dream_epochs > 0:
        dream_consolidation(
            proto=proto,
            snn=snn,
            cortex=cortex,
            memory=memory,
            epochs=args.dream_epochs,
            samples_per_epoch=args.dream_samples,
            curiosity_drive=args.curiosity_drive,
            learning_rate=args.dream_learning_rate,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ego-Sphere Phase 3 simulation")
    parser.add_argument("--mode", choices=["demo", "active_learning"], default="demo")
    parser.add_argument("--steps", type=int, default=170)
    parser.add_argument("--error-threshold", type=float, default=0.3)
    parser.add_argument("--curiosity-threshold", type=float, default=0.55)
    parser.add_argument("--curiosity-cooldown", type=int, default=25)
    parser.add_argument("--curiosity-drive", type=float, default=0.02)
    parser.add_argument("--text-script", type=str, default=None, help="Path to <start-end;text;vision> script")
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher bridge")
    parser.add_argument("--mock-teacher", action="store_true", help="Force mock teacher replies")
    parser.add_argument("--provider", type=str, default="codex_cli", help="Teacher provider: codex_cli or codex_api")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dream-epochs", type=int, default=2)
    parser.add_argument("--dream-samples", type=int, default=3)
    parser.add_argument("--dream-learning-rate", type=float, default=0.03)
    parser.add_argument("--teacher-learning-rate", type=float, default=0.05)
    parser.add_argument("--background-learning-rate", type=float, default=0.005)
    parser.add_argument("--background-hebbian-interval", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "demo":
        stimuli = [
            StimulusWindow(start=50, end=100, text="cat", vision_pattern="cat"),
            StimulusWindow(start=100, end=110, text="dog", vision_pattern=None),
        ]
        run_loop(stimuli, args, label="demo")
    else:
        stimuli = load_stimuli(args.text_script)
        run_loop(stimuli, args, label="active")


if __name__ == "__main__":
    main()
