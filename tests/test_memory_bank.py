"""Tests for MemoryBank storage using in-memory ChromaDB."""

import pytest

from core.memory_bank import MemoryBank


def test_memory_bank_adds_and_counts():
    try:
        bank = MemoryBank(collection="test_memories", persist_directory=None)
    except RuntimeError:
        pytest.skip("ChromaDB unavailable in this environment")
    before = bank.count()

    embedding = [0.1, 0.2, 0.3]
    meta = {"trigger_type": "SURPRISE"}
    doc_id = bank.add_memory("teacher reply", embedding=embedding, metadata=meta)

    assert doc_id
    assert bank.count() == before + 1


@pytest.mark.parametrize("embedding", ([[0.0], [1.0, 2.0, 3.0]]))
def test_memory_bank_accepts_embeddings(embedding):
    try:
        bank = MemoryBank(collection="test_memories_param", persist_directory=None)
    except RuntimeError:
        pytest.skip("ChromaDB unavailable in this environment")
    meta = {"trigger_type": "CONFUSION"}
    bank.add_memory("content", embedding=embedding, metadata=meta)
    assert bank.count() >= 1
