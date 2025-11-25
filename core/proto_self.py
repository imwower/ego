"""Proto-Self: maintains internal homeostasis and somatic background noise.

The Proto-Self keeps track of coarse bodily variables (energy, pain, curiosity)
and produces low-amplitude "somatic" spiking noise that gives the SNN a living
baseline. Pain modulates noise gain to mimic arousal under stress.
"""

from typing import Optional, Dict

import torch

from config import HyperParams, default_hparams


class ProtoSelf:
    def __init__(self, hparams: Optional[HyperParams] = None) -> None:
        self.hparams = hparams or default_hparams()
        self.device = self.hparams.device
        if self.device.type != "mps":
            raise ValueError("ProtoSelf must run on an MPS device.")

        # Internal state lives on the MPS device to keep updates cheap.
        self.energy = torch.tensor(
            self.hparams.energy_init, device=self.device, dtype=torch.float32
        )
        self.pain = torch.tensor(
            self.hparams.pain_init, device=self.device, dtype=torch.float32
        )
        self.curiosity = torch.tensor(
            self.hparams.curiosity_init, device=self.device, dtype=torch.float32
        )

    def step(self, external_pain: float = 0.0, curiosity_drive: float = 0.0) -> torch.Tensor:
        """Advance one time-step of the Proto-Self dynamics.

        - Energy decays metabolically.
        - Pain integrates external pain plus energy deficit, then recovers.
        - Curiosity decays but is driven by boredom (low pain) and optional input.
        - Somatic noise is generated as MPS-resident Gaussian noise scaled by pain.

        Returns:
            Tensor: somatic noise vector on the MPS device.
        """

        hp = self.hparams
        dt = hp.dt

        pain_input = torch.as_tensor(external_pain, device=self.device, dtype=torch.float32)
        curiosity_input = torch.as_tensor(curiosity_drive, device=self.device, dtype=torch.float32)

        # Metabolic decay of energy.
        self.energy = torch.clamp(
            self.energy - dt * hp.energy_decay,
            min=hp.energy_floor,
            max=hp.energy_ceiling,
        )

        # Pain rises with external signals and with energy deficit, then recovers.
        energy_deficit = torch.relu(hp.energy_setpoint - self.energy)
        self.pain = self.pain + dt * (pain_input + hp.pain_from_energy_scale * energy_deficit)
        self.pain = self.pain - dt * hp.pain_recovery * self.pain
        self.pain = torch.clamp(self.pain, min=0.0, max=hp.pain_ceiling)

        # Curiosity decays but is replenished when boredom (low pain) is present.
        boredom_drive = hp.curiosity_boredom_drive * (1.0 - self.pain)
        self.curiosity = self.curiosity + dt * (curiosity_input + boredom_drive)
        self.curiosity = self.curiosity - dt * hp.curiosity_decay * self.curiosity
        self.curiosity = torch.clamp(self.curiosity, min=0.0, max=hp.curiosity_ceiling)

        # Somatic noise stays on MPS; pain increases its gain to mimic arousal.
        noise_scale = hp.baseline_noise * (1.0 + hp.pain_noise_gain * self.pain)
        somatic_noise = torch.randn(hp.somatic_noise_dim, device=self.device) * noise_scale

        return somatic_noise

    def state(self) -> Dict[str, float]:
        """Return scalar snapshots of the internal state for logging/telemetry."""

        return {
            "energy": float(self.energy.item()),
            "pain": float(self.pain.item()),
            "curiosity": float(self.curiosity.item()),
        }

    def __repr__(self) -> str:
        s = self.state()
        return (
            f"ProtoSelf(energy={s['energy']:.3f}, pain={s['pain']:.3f}, "
            f"curiosity={s['curiosity']:.3f}, device={self.device})"
        )
