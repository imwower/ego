"\"\"\"Tests for EpisodicMemory basic store/recall (skipped if chroma/text encoder unavailable).\"\"\""

import pytest

from core.episodic_memory import EpisodicMemory


def test_store_and_recall_with_neural_embedding():
    try:
        mem = EpisodicMemory(collection="test_episodic", persist_directory="data/chroma_store")
    except Exception:
        pytest.skip("Chroma or encoder unavailable")

    emb = [0.1, 0.2, 0.3]
    doc_id = mem.store_experience("SURPRISE", "test content", emb)
    assert doc_id
    res = mem.recall(emb, k=1)
    assert res["ids"]
