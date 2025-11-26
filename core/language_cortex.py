"""Language Cortex: maps words to spike-like rates for the SNN text region.

We approximate Wernicke's area with a simple embedding layer. Tokens are mapped
into vectors (embedding_dim == SNN text_dim), pooled, and passed through a
sigmoid to yield rate-coded spike intensities in [0, 1]. This output is ready to
feed the SNN text region without further shape conversion.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import HyperParams, default_hparams


class LanguageCortex(nn.Module):
    """Maps words to rate-coded spikes; dynamic vocab avoids hash collisions.

    Neuroscience rationale: distinct lexical items should map to separable neural
    patterns. A deterministic growing vocabulary ensures "different words →
    different axes" without collision-induced semantic loss.
    """

    def __init__(self, hparams: Optional[HyperParams] = None, vocab_size: Optional[int] = None) -> None:
        super().__init__()
        self.hparams = hparams or default_hparams()
        self.device = self.hparams.device
        if self.device.type != "mps":
            raise ValueError("LanguageCortex must run on an MPS device.")

        self.vocab_size = vocab_size or self.hparams.vocab_size
        self.vocab: Dict[str, int] = {}
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hparams.text_dim,
            device=self.device,
        )
        # Optional: initialize embeddings to be approximately orthogonal to reduce overlap.
        with torch.no_grad():
            rand = torch.randn(self.vocab_size, self.hparams.text_dim, device="cpu")
            q, _ = torch.linalg.qr(rand)
            weight = q[: self.vocab_size, :].to(self.device)
            self.embedding.weight.copy_(weight)

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def _get_or_add_token(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        idx = len(self.vocab)
        if idx >= self.vocab_size:
            # If vocab is full, wrap but keep determinism by reusing last index.
            idx = self.vocab_size - 1
        self.vocab[token] = idx
        return idx

    def text_to_spikes(self, text_str: str) -> torch.Tensor:
        """Convert text into a rate-coded spike vector aligned with SNN text_dim."""

        tokens = self._tokenize(text_str)
        if not tokens:
            return torch.zeros(self.hparams.text_dim, device=self.device)

        indices = torch.tensor([self._get_or_add_token(tok) for tok in tokens], device=self.device)
        embeddings = self.embedding(indices)

        pooled = embeddings.mean(dim=0)
        rates = torch.sigmoid(pooled)  # rate-coded spikes in [0, 1]
        return rates

    def __repr__(self) -> str:
        return (
            f"LanguageCortex(text_dim={self.hparams.text_dim}, vocab_size={self.vocab_size}, "
            f"device={self.device})"
        )


__all__ = ["LanguageCortex"]
