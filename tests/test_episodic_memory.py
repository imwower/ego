"""Tests for EpisodicMemory storage using in-memory ChromaDB."""

import pytest

from core import EpisodicMemory


def test_memory_bank_adds_and_counts():
    try:
        bank = EpisodicMemory(collection="test_memories", persist_directory=None)
    except RuntimeError:
        pytest.skip("ChromaDB unavailable in this environment")
    before = bank.count()

    embedding = [0.1, 0.2, 0.3]
    meta = {"trigger_type": "SURPRISE"}
    doc_id = bank.store_experience("SURPRISE", "teacher reply", embedding, metadata=meta)

    assert doc_id
    assert bank.count() == before + 1


@pytest.mark.parametrize("embedding", ([[0.0], [1.0, 2.0, 3.0]]))
def test_memory_bank_accepts_embeddings(embedding):
    try:
        # Use distinct collection names per dimension to avoid dimension mismatch.
        bank = EpisodicMemory(collection=f"test_memories_param_{len(embedding)}", persist_directory=None)
    except RuntimeError:
        pytest.skip("ChromaDB unavailable in this environment")
    meta = {"trigger_type": "CONFUSION"}
    bank.store_experience("CONFUSION", "content", embedding, metadata=meta)
    assert bank.count() >= 1
