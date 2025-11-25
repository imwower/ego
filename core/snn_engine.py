"""Recurrent Spiking Neural Network engine with predictive coding and reentry.

Regions:
- Vision: encodes visual tensor inputs.
- Text: encodes semantic/text embeddings.
- Association (Global Workspace): integrates multimodal spikes and reentrant
  self-history, then broadcasts predictions of the next sensory input.

Key mechanisms:
- Predictive coding: association spikes decode into a prediction of next-step
  sensory signals; error is prediction minus current sensory input.
- Reentry (self-memory): previous association spikes drive the next-step state
  via recurrent weights, providing temporal continuity.
"""

from typing import Dict, Optional, Tuple

import torch

from config import HyperParams, default_hparams


class SNNEngine:
    def __init__(self, hparams: Optional[HyperParams] = None) -> None:
        self.hparams = hparams or default_hparams()
        self.device = self.hparams.device
        if self.device.type != "mps":
            raise ValueError("SNNEngine must run on an MPS device.")

        self._init_weights()
        self.reset_state()

    def _init_weights(self) -> None:
        hp = self.hparams
        self.W_v_to_a = self._init_sparse_weight(hp.assoc_dim, hp.vision_dim)
        # Mask preserves the originally instantiated sparse topology for Hebbian updates.
        self.W_v_to_a_mask = (self.W_v_to_a != 0).float()
        self.W_t_to_a = self._init_sparse_weight(hp.assoc_dim, hp.text_dim)
        self.W_a_recurrent = self._init_sparse_weight(
            hp.assoc_dim, hp.assoc_dim, scale=hp.reentry_gain
        )

        # Decoders from association to sensory predictions.
        self.W_pred_vision = self._init_sparse_weight(
            hp.vision_dim, hp.assoc_dim, scale=hp.prediction_gain
        )
        self.W_pred_text = self._init_sparse_weight(
            hp.text_dim, hp.assoc_dim, scale=hp.prediction_gain
        )

    def _init_sparse_weight(self, out_dim: int, in_dim: int, scale: float = 1.0) -> torch.Tensor:
        """Create a dense weight matrix with sparse connectivity on MPS.

        True sparse operations are limited on MPS; we therefore materialize a
        dense tensor with a random sparsity mask to preserve compatibility while
        keeping connectivity light.
        """

        hp = self.hparams
        density = max(0.0, min(1.0, 1.0 - hp.sparsity))
        num_active = max(1, int(density * out_dim * in_dim))

        weight = torch.zeros((out_dim, in_dim), device=self.device)
        rows = torch.randint(0, out_dim, (num_active,), device=self.device)
        cols = torch.randint(0, in_dim, (num_active,), device=self.device)
        vals = torch.randn(num_active, device=self.device) * scale / max(1.0, in_dim**0.5)
        weight[rows, cols] = vals
        return weight

    def reset_state(self) -> None:
        hp = self.hparams
        zeros = lambda d: torch.zeros(d, device=self.device)
        self.v_vision = zeros(hp.vision_dim)
        self.v_text = zeros(hp.text_dim)
        self.v_assoc = zeros(hp.assoc_dim)

        self.spikes_vision = zeros(hp.vision_dim)
        self.spikes_text = zeros(hp.text_dim)
        self.spikes_assoc = zeros(hp.assoc_dim)

    def _prepare_input(self, x: Optional[torch.Tensor], dim: int, name: str) -> torch.Tensor:
        if x is None:
            return torch.zeros(dim, device=self.device)
        t = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if t.numel() == 1:
            t = t.expand(dim)
        if t.shape[0] != dim:
            raise ValueError(f"{name} input must have shape ({dim},), got {t.shape}")
        return t

    def _integrate_and_fire(self, v_mem: torch.Tensor, input_drive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hp = self.hparams
        v_mem = v_mem * (1.0 - hp.membrane_decay) + input_drive
        spikes = (v_mem >= hp.spike_threshold).float()
        v_mem = torch.where(spikes.bool(), torch.full_like(v_mem, hp.reset_potential), v_mem)
        return v_mem, spikes

    def step(self, sensory_input: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Advance one timestep of the SNN.

        Args:
            sensory_input: dict with optional keys "vision" and "text" mapping
                to tensors shaped (vision_dim,) and (text_dim,) on any device.

        Returns:
            firing_indices: dict of index tensors for spikes in each region.
            prediction_error: dict with per-modality prediction errors and a
                scalar L2 norm under key "norm".
        """

        hp = self.hparams
        vision = self._prepare_input(sensory_input.get("vision"), hp.vision_dim, "vision")
        text = self._prepare_input(sensory_input.get("text"), hp.text_dim, "text")

        # Sensory regions integrate inputs directly.
        self.v_vision, self.spikes_vision = self._integrate_and_fire(self.v_vision, vision)
        self.v_text, self.spikes_text = self._integrate_and_fire(self.v_text, text)

        # Association integrates multimodal spikes and its own history (reentry).
        prev_assoc_spikes = self.spikes_assoc
        assoc_drive = (
            torch.matmul(self.W_v_to_a, self.spikes_vision)
            + torch.matmul(self.W_t_to_a, self.spikes_text)
            + torch.matmul(self.W_a_recurrent, prev_assoc_spikes)
        )
        self.v_assoc, self.spikes_assoc = self._integrate_and_fire(self.v_assoc, assoc_drive)

        # Predict next-step sensory input from association spikes.
        pred_vision = torch.matmul(self.W_pred_vision, self.spikes_assoc)
        pred_text = torch.matmul(self.W_pred_text, self.spikes_assoc)

        vision_error = pred_vision - vision
        text_error = pred_text - text
        error_norm = torch.sqrt(torch.mean(torch.cat([vision_error, text_error]) ** 2))

        firing_indices = {
            "vision": torch.nonzero(self.spikes_vision, as_tuple=False).flatten(),
            "text": torch.nonzero(self.spikes_text, as_tuple=False).flatten(),
            "assoc": torch.nonzero(self.spikes_assoc, as_tuple=False).flatten(),
        }
        prediction_error = {
            "vision": vision_error,
            "text": text_error,
            "norm": error_norm,
        }

        return firing_indices, prediction_error

    def update_weights_hebbian(self, learning_rate: float = 0.01, max_weight: float = 1.0) -> None:
        """Hebbian plasticity: cells that fire together wire together.

        We only update existing sparse connections (mask) to avoid densifying
        the matrix. Simple STDP-like window collapsed into a single co-firing
        step: Δw_ij = lr * post_i * pre_j. Weights are clamped to prevent
        runaway excitation.
        """

        lr = torch.tensor(learning_rate, device=self.device, dtype=torch.float32)
        cofire = torch.outer(self.spikes_assoc, self.spikes_vision)
        delta = lr * cofire * self.W_v_to_a_mask
        self.W_v_to_a = torch.clamp(self.W_v_to_a + delta, max=max_weight)


__all__ = ["SNNEngine"]
