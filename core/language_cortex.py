"""Language Cortex: maps words to spike-like rates for the SNN text region.

We approximate Wernicke's area with a simple embedding layer. Tokens are mapped
into vectors (embedding_dim == SNN text_dim), pooled, and passed through a
sigmoid to yield rate-coded spike intensities in [0, 1]. This output is ready to
feed the SNN text region without further shape conversion.
"""

import hashlib
from typing import List, Optional

import torch
import torch.nn as nn

from config import HyperParams, default_hparams


class LanguageCortex(nn.Module):
    def __init__(self, hparams: Optional[HyperParams] = None, vocab_size: Optional[int] = None) -> None:
        super().__init__()
        self.hparams = hparams or default_hparams()
        self.device = self.hparams.device
        if self.device.type != "mps":
            raise ValueError("LanguageCortex must run on an MPS device.")

        self.vocab_size = vocab_size or self.hparams.vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hparams.text_dim,
            device=self.device,
        )

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.vocab_size

    def text_to_spikes(self, text_str: str) -> torch.Tensor:
        """Convert text into a rate-coded spike vector aligned with SNN text_dim.

        Returns a tensor of shape (text_dim,) on the MPS device. Values are in
        [0, 1], representing firing rates rather than binary spikes.
        """

        tokens = self._tokenize(text_str)
        if not tokens:
            return torch.zeros(self.hparams.text_dim, device=self.device)

        indices = torch.tensor([self._hash_token(tok) for tok in tokens], device=self.device)
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
