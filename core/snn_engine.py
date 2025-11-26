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

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "W_v_to_a": self.W_v_to_a,
            "W_v_to_a_mask": self.W_v_to_a_mask,
            "W_t_to_a": self.W_t_to_a,
            "W_t_to_a_mask": self.W_t_to_a_mask,
            "W_a_recurrent": self.W_a_recurrent,
            "W_pred_vision": self.W_pred_vision,
            "W_pred_text": self.W_pred_text,
            "v_vision": self.v_vision,
            "v_text": self.v_text,
            "v_assoc": self.v_assoc,
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.W_v_to_a = state["W_v_to_a"]
        self.W_v_to_a_mask = state.get("W_v_to_a_mask", (self.W_v_to_a != 0).float())
        self.W_t_to_a = state["W_t_to_a"]
        self.W_t_to_a_mask = state.get("W_t_to_a_mask", (self.W_t_to_a != 0).float())
        self.W_a_recurrent = state["W_a_recurrent"]
        self.W_pred_vision = state["W_pred_vision"]
        self.W_pred_text = state["W_pred_text"]
        self.v_vision = state["v_vision"]
        self.v_text = state["v_text"]
        self.v_assoc = state["v_assoc"]

    def _init_weights(self) -> None:
        hp = self.hparams
        self.W_v_to_a = self._init_sparse_weight(hp.assoc_dim, hp.vision_dim)
        # Masks preserve the originally instantiated sparse topology for Hebbian updates.
        self.W_v_to_a_mask = (self.W_v_to_a != 0).float()
        self.W_t_to_a = self._init_sparse_weight(hp.assoc_dim, hp.text_dim)
        self.W_t_to_a_mask = (self.W_t_to_a != 0).float()
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

    def _integrate_and_fire(self, v_mem: torch.Tensor, input_drive: torch.Tensor, threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hp = self.hparams
        v_mem = v_mem * (1.0 - hp.membrane_decay) + input_drive
        spikes = (v_mem >= threshold).float()
        v_mem = torch.where(spikes.bool(), torch.full_like(v_mem, hp.reset_potential), v_mem)
        return v_mem, spikes

    def step(self, sensory_input: Dict[str, torch.Tensor], modulation_signals: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Advance one timestep of the SNN.

        Args:
            sensory_input: dict with optional keys "vision" and "text" mapping
                to tensors shaped (vision_dim,) and (text_dim,) on any device.
            modulation_signals: dict with optional keys {\"pain\", \"curiosity\"}
                to modulate thresholds/learning rates (neuromodulation proxy).

        Returns:
            firing_indices: dict of index tensors for spikes in each region.
            prediction_error: dict with per-modality prediction errors and a
                scalar L2 norm under key "norm".
        """

        hp = self.hparams
        vision = self._prepare_input(sensory_input.get("vision"), hp.vision_dim, "vision")
        text = self._prepare_input(sensory_input.get("text"), hp.text_dim, "text")

        pain_signal = float(modulation_signals.get("pain", 0.0)) if modulation_signals else 0.0
        # Neuromodulation: pain lowers threshold (hyper-excitable).
        threshold_value = hp.spike_threshold * max(0.2, 1.0 - 0.5 * pain_signal)

        # Sensory regions integrate inputs directly.
        self.v_vision, self.spikes_vision = self._integrate_and_fire(
            self.v_vision, vision, torch.full_like(self.v_vision, threshold_value)
        )
        self.v_text, self.spikes_text = self._integrate_and_fire(
            self.v_text, text, torch.full_like(self.v_text, threshold_value)
        )

        # Association integrates multimodal spikes and its own history (reentry).
        prev_assoc_spikes = self.spikes_assoc
        assoc_drive = (
            torch.matmul(self.W_v_to_a, self.spikes_vision)
            + torch.matmul(self.W_t_to_a, self.spikes_text)
            + torch.matmul(self.W_a_recurrent, prev_assoc_spikes)
        )
        self.v_assoc, self.spikes_assoc = self._integrate_and_fire(
            self.v_assoc, assoc_drive, torch.full_like(self.v_assoc, threshold_value)
        )

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

    def update_weights_hebbian(self, learning_rate: float = 0.01, max_weight: float = 1.0, modulation_signals: Optional[Dict[str, float]] = None) -> None:
        """Hebbian plasticity: cells that fire together wire together.

        We only update existing sparse connections (mask) to avoid densifying
        the matrix. Simple STDP-like window collapsed into a single co-firing
        step: Δw_ij = lr * post_i * pre_j. Weights are clamped to prevent
        runaway excitation.
        """

        curiosity_signal = float(modulation_signals.get("curiosity", 0.0)) if modulation_signals else 0.0
        lr = torch.tensor(learning_rate * (1.0 + 0.5 * curiosity_signal), device=self.device, dtype=torch.float32)
        # Vision -> Association potentiation.
        cofire_v = torch.outer(self.spikes_assoc, self.spikes_vision)
        delta_v = lr * cofire_v * self.W_v_to_a_mask
        self.W_v_to_a = torch.clamp(self.W_v_to_a + delta_v, min=0.0, max=max_weight)

        # Text -> Association potentiation (language pathway).
        cofire_t = torch.outer(self.spikes_assoc, self.spikes_text)
        delta_t = lr * cofire_t * self.W_t_to_a_mask
        self.W_t_to_a = torch.clamp(self.W_t_to_a + delta_t, min=0.0, max=max_weight)


__all__ = ["SNNEngine"]
