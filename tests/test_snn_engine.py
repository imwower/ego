"""Unit tests for the SNNEngine predictive/reentrant dynamics."""

import pytest
import torch

from core.snn_engine import SNNEngine
from config import HyperParams


mps_unavailable = not torch.backends.mps.is_available()


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_step_outputs_indices_and_prediction_error_shapes():
    device = torch.device("mps")
    hp = HyperParams(
        device=device,
        vision_dim=8,
        text_dim=4,
        assoc_dim=16,
        spike_threshold=0.5,
        membrane_decay=0.05,
    )
    engine = SNNEngine(hparams=hp)

    vision = torch.rand(hp.vision_dim, device=device)
    text = torch.rand(hp.text_dim, device=device)

    firing, error = engine.step({"vision": vision, "text": text})

    assert set(firing.keys()) == {"vision", "text", "assoc"}
    assert error["vision"].shape[0] == hp.vision_dim
    assert error["text"].shape[0] == hp.text_dim
    assert error["norm"].ndim == 0

    # Ensure outputs reside on MPS.
    assert error["vision"].device.type == "mps"
    assert error["text"].device.type == "mps"


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_reentry_drives_activity_without_new_input():
    device = torch.device("mps")
    hp = HyperParams(
        device=device,
        vision_dim=4,
        text_dim=2,
        assoc_dim=4,
        membrane_decay=0.0,
        spike_threshold=0.2,
        sparsity=0.0,
    )
    engine = SNNEngine(hparams=hp)

    # Make connectivity deterministic and strong enough to spike.
    engine.W_v_to_a.fill_(0.5)
    engine.W_t_to_a.zero_()
    engine.W_a_recurrent.fill_(0.5)
    engine.W_pred_vision.zero_()
    engine.W_pred_text.zero_()

    vision = torch.ones(hp.vision_dim, device=device)
    text = torch.zeros(hp.text_dim, device=device)

    engine.reset_state()
    _, _ = engine.step({"vision": vision, "text": text})
    first_spikes = engine.spikes_assoc.clone()

    _, _ = engine.step({"vision": torch.zeros_like(vision), "text": torch.zeros_like(text)})

    assert first_spikes.sum() > 0  # initial activation fired
    assert engine.spikes_assoc.sum() > 0  # reentry keeps association active
