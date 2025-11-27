"""Global hyperparameters for Ego-Sphere.

This module centralizes tunable constants (neuron counts, decay rates,
device selection) so the rest of the code can stay lightweight. Defaults
assume Apple Silicon with MPS enabled; the system should be run on that
hardware for correct performance.
"""

from dataclasses import dataclass
import torch


@dataclass
class HyperParams:
    # Simulation
    dt: float = 0.01

    # Device
    device: torch.device = torch.device("mps")

    # Initial internal state
    energy_init: float = 1.0
    pain_init: float = 0.1
    curiosity_init: float = 0.3

    # Homeostatic dynamics
    energy_decay: float = 0.01
    curiosity_decay: float = 0.005
    pain_recovery: float = 0.02

    # Couplings
    energy_setpoint: float = 0.8
    pain_from_energy_scale: float = 0.5
    curiosity_boredom_drive: float = 0.01

    # Noise generation
    baseline_noise: float = 0.05
    pain_noise_gain: float = 1.5
    somatic_noise_dim: int = 128

    # Clamps
    energy_floor: float = 0.0
    energy_ceiling: float = 1.0
    pain_ceiling: float = 1.0
    curiosity_ceiling: float = 1.0

    # SNN region sizes
    # Default to 32x32 flattened grayscale for CIFAR/vision alignment.
    vision_dim: int = 1024
    text_dim: int = 32
    assoc_dim: int = 128

    # Feedback coupling
    snn_feedback_gain: float = 1.0

    # Language cortex
    vocab_size: int = 4096

    # SNN dynamics
    membrane_decay: float = 0.1
    spike_threshold: float = 1.0
    reset_potential: float = 0.0
    sparsity: float = 0.8  # fraction of weights zeroed
    reentry_gain: float = 1.0
    prediction_gain: float = 1.0


def ensure_mps_device() -> torch.device:
    """Return an MPS device or raise if unavailable.

    Ego-Sphere is designed for Apple Silicon; failing early avoids silent
    CPU fallbacks that would change timing/latency characteristics.
    """

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device not available. Please run on Apple Silicon with PyTorch MPS support.")
    return torch.device("mps")


def default_hparams() -> HyperParams:
    """Construct hyperparameters with validated device settings."""

    device = ensure_mps_device()
    hp = HyperParams(device=device)
    # Ensure somatic noise covers the vision dimension to avoid shape mismatch.
    if hp.somatic_noise_dim < hp.vision_dim:
        hp.somatic_noise_dim = hp.vision_dim
    return hp
