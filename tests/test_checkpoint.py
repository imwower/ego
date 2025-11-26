"\"\"\"Tests for checkpoint save/load.\"\"\""

import os
import tempfile

import pytest
import torch

from core import CheckpointManager, ProtoSelf, SNNEngine, LanguageCortex
from config import HyperParams


mps_unavailable = not torch.backends.mps.is_available()


@pytest.mark.skipif(mps_unavailable, reason="MPS device not available")
def test_checkpoint_roundtrip():
    device = torch.device("mps")
    hp = HyperParams(device=device, vision_dim=4, text_dim=2, assoc_dim=4)
    proto = ProtoSelf(hparams=hp)
    snn = SNNEngine(hparams=hp)
    cortex = LanguageCortex(hparams=hp)

    proto.energy += 0.1
    cortex.embedding.weight.data.fill_(0.5)
    snn.W_v_to_a.fill_(0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(directory=tmpdir)
        path = mgr.save(step=42, snn=snn, proto=proto, cortex=cortex)
        assert os.path.exists(path)

        # Mutate states, then restore.
        proto.energy = torch.tensor(0.0, device=device)
        snn.W_v_to_a.zero_()
        cortex.embedding.weight.data.zero_()

        meta = mgr.load(path, snn=snn, proto=proto, cortex=cortex)
        assert meta["step"] == 42
        assert torch.allclose(proto.energy, torch.tensor(1.1, device=device))
        assert torch.allclose(snn.W_v_to_a, torch.full_like(snn.W_v_to_a, 0.3))
        assert torch.allclose(cortex.embedding.weight.data, torch.full_like(cortex.embedding.weight.data, 0.5))
