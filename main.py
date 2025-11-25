"""Ego-Sphere simulation loop tying together ProtoSelf, SNN, Language Cortex, and TeacherBridge.

Scenario:
- t < 50: resting state
- 50 <= t < 100: visual input "Cat" (simulated vector)
- 100 <= t < 110: text input "Dog" (conflict vs prior visual)

When prediction error exceeds a threshold, the TeacherBridge is invoked to
provide guidance. Logs provide a coarse stream of consciousness (energy, pain,
curiosity, active modalities, prediction error).
"""

import math
from typing import Dict

import torch

from config import default_hparams
from core import ProtoSelf, SNNEngine, LanguageCortex
from bridge import TeacherBridge


def simulate_step(t: int, cortex: LanguageCortex) -> Dict[str, torch.Tensor]:
    """Return sensory inputs for timestep t."""
    hp = cortex.hparams
    vision = torch.zeros(hp.vision_dim, device=hp.device)
    text = None

    if 50 <= t < 100:
        # Simulated visual pattern for "Cat" (one-hot-ish bump).
        vision[0:3] = torch.tensor([1.0, 0.8, 0.6], device=hp.device)
    if 100 <= t < 110:
        # Conflicting text cue.
        text = cortex.text_to_spikes("Dog")

    return {"vision": vision, "text": text}


def main() -> None:
    hp = default_hparams()
    proto = ProtoSelf(hparams=hp)
    snn = SNNEngine(hparams=hp)
    cortex = LanguageCortex(hparams=hp)
    # Try codex CLI by default; falls back to mock if unavailable.
    teacher = TeacherBridge(use_mock=False, provider="codex_cli")

    steps = 150
    error_threshold = 0.3

    for t in range(steps):
        sensory = simulate_step(t, cortex)

        # ProtoSelf provides somatic background noise; add to vision to keep
        # activity alive even at rest.
        somatic = proto.step()
        sensory["vision"] = sensory["vision"] + somatic[: hp.vision_dim]

        firing, pred_err = snn.step(sensory)

        # Trigger teacher if surprised/confused.
        if pred_err["norm"] > error_threshold:
            ctx = {
                "prediction_error_norm": float(pred_err["norm"].item()),
                "proto_state": proto.state(),
                "recent_spikes": {k: v.tolist() for k, v in firing.items()},
                "notes": f"t={t}",
            }
            bridge_out = teacher.ask_gemini(ctx, trigger_type="SURPRISE")
        else:
            bridge_out = None

        if t % 10 == 0 or bridge_out:
            state = proto.state()
            active_modalities = []
            if sensory["vision"].abs().sum() > 0:
                active_modalities.append("vision")
            if sensory.get("text") is not None:
                active_modalities.append("text")

            print(
                f"t={t:03d} | energy={state['energy']:.2f} pain={state['pain']:.2f} "
                f"curiosity={state['curiosity']:.2f} | active={active_modalities} "
                f"err={pred_err['norm']:.3f} spikes={{v:{len(firing['vision'])}, t:{len(firing['text'])}, a:{len(firing['assoc'])}}}"
            )
            if bridge_out:
                print("  Teacher prompt:")
                print(f"    {bridge_out['prompt']}")
                status = "ok" if bridge_out.get("provider_ok") else "fallback"
                print(f"  Teacher reply [{bridge_out['provider']}/{status}]: {bridge_out['reply']}")


if __name__ == "__main__":
    main()
