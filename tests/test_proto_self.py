"""Unit tests for the Proto-Self homeostatic module."""

import pytest
import torch

from core.proto_self import ProtoSelf
from config import HyperParams


mps_unavailable = not torch.backends.mps.is_available()


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_step_updates_state_and_returns_mps_noise():
    device = torch.device("mps")
    hparams = HyperParams(device=device)
    proto = ProtoSelf(hparams=hparams)

    noise = proto.step(external_pain=0.2, curiosity_drive=0.1)

    assert noise.device.type == "mps"
    assert noise.shape[0] == hparams.somatic_noise_dim

    state = proto.state()
    assert 0.0 <= state["energy"] <= hparams.energy_ceiling
    assert 0.0 <= state["pain"] <= hparams.pain_ceiling
    assert 0.0 <= state["curiosity"] <= hparams.curiosity_ceiling

    # Energy should decrease slightly after a step due to metabolic decay.
    assert state["energy"] < hparams.energy_init


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_pain_modulates_noise_gain():
    device = torch.device("mps")
    hparams = HyperParams(device=device)

    # Align random seeds so noise comparison is meaningful.
    torch.manual_seed(0)
    low_pain_proto = ProtoSelf(hparams=hparams)
    low_noise = low_pain_proto.step(external_pain=0.0)

    torch.manual_seed(0)
    high_pain_proto = ProtoSelf(hparams=hparams)
    high_pain_proto.pain = torch.tensor(0.8, device=device)
    high_noise = high_pain_proto.step(external_pain=0.0)

    expected_ratio = 1.0 + hparams.pain_noise_gain * 0.8
    actual_ratio = high_noise.std().item() / low_noise.std().item()

    # Allow tolerance because of finite sampling of noise.
    assert actual_ratio == pytest.approx(expected_ratio, rel=0.2)
