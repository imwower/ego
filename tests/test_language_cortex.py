"""Unit tests for LanguageCortex text-to-spike conversion."""

import pytest
import torch

from core.language_cortex import LanguageCortex
from config import HyperParams


mps_unavailable = not torch.backends.mps.is_available()


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_text_to_spikes_shape_device_and_range():
    torch.manual_seed(0)
    device = torch.device("mps")
    hp = HyperParams(device=device, text_dim=8, vocab_size=64)
    cortex = LanguageCortex(hparams=hp)

    spikes = cortex.text_to_spikes("hello world")

    assert spikes.shape == (hp.text_dim,)
    assert spikes.device.type == "mps"
    assert torch.all(spikes >= 0.0) and torch.all(spikes <= 1.0)


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_empty_text_returns_zero_vector():
    device = torch.device("mps")
    hp = HyperParams(device=device, text_dim=6, vocab_size=32)
    cortex = LanguageCortex(hparams=hp)

    spikes = cortex.text_to_spikes("")
    assert torch.count_nonzero(spikes) == 0
