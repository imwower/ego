"""Episodic memory backed by ChromaDB, grounding experiences to neural states.

Stores teacher feedback (content) keyed by an embedding. Preferred embedding is
the SNN association spike vector; falls back to a text embedding via
SentenceTransformers if provided neural state shape mismatches collection dim.
"""

from typing import Any, Dict, List, Optional
import time

import chromadb
from chromadb.config import Settings

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class EpisodicMemory:
    def __init__(self, collection: str = "ego_episodic", persist_directory: str = "data/chroma_store") -> None:
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection)
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None

    def _ensure_embedding(self, content: str, neural_state: Optional[List[float]]) -> List[float]:
        """Pick embedding: prefer neural_state if dimension matches collection, else text embedding."""
        if neural_state is not None:
            if self.collection.metadata is not None:
                dim = self.collection.metadata.get("dimension")
                if dim is None or len(neural_state) == dim:
                    return neural_state
            else:
                return neural_state
        if not self.text_encoder:
            raise RuntimeError("No neural embedding provided and text encoder unavailable.")
        return self.text_encoder.encode(content).tolist()

    def store_experience(self, trigger: str, content: str, neural_state: Optional[List[float]]) -> str:
        embedding = self._ensure_embedding(content, neural_state)
        doc_id = f"exp-{int(time.time() * 1e6)}"
        meta = {"timestamp": time.time(), "trigger": trigger}
        self.collection.add(ids=[doc_id], documents=[content], embeddings=[embedding], metadatas=[meta])
        return doc_id

    def recall(self, neural_state: List[float], k: int = 3) -> Dict[str, Any]:
        """Query similar past neural states; returns ids, docs, metadata, distances."""
        results = self.collection.query(query_embeddings=[neural_state], n_results=k)
        return results


__all__ = ["EpisodicMemory"]
