"""
Channel normalization for device-robust speaker verification.

Techniques implemented:
  - CMVN (Cepstral Mean and Variance Normalization): removes additive/convolutive
    channel effects from MFCC features by standardizing mean and variance per utterance.
  - Embedding-level mean subtraction: removes the channel-correlated bias from
    speaker embeddings using a cohort mean estimated from training data.
  - WCCN (Within-Class Covariance Normalization): whitens the embedding space
    to reduce within-speaker variance caused by channel differences.
"""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

SAMPLE_RATE = 16000
N_MFCC = 40


# ── MFCC-level CMVN ──────────────────────────────────────────────────────────

def extract_mfcc(waveform: torch.Tensor, n_mfcc: int = N_MFCC) -> torch.Tensor:
    """Extract MFCC features. Returns (n_mfcc, time) tensor."""
    transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 80},
    )
    return transform(waveform).squeeze(0)   # (n_mfcc, time)


def apply_cmvn(mfcc: torch.Tensor, normalize_variance: bool = True) -> torch.Tensor:
    """
    Apply utterance-level CMVN to MFCC features.
    Subtracts per-cepstral-coefficient mean and optionally divides by std.
    Input shape: (n_mfcc, time). Returns same shape.
    """
    mean = mfcc.mean(dim=1, keepdim=True)
    normalized = mfcc - mean
    if normalize_variance:
        std = mfcc.std(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = normalized / std
    return normalized


def waveform_to_cmvn_mfcc(waveform: torch.Tensor, n_mfcc: int = N_MFCC) -> torch.Tensor:
    """Full pipeline: waveform → MFCC → CMVN. Returns (n_mfcc, time)."""
    mfcc = extract_mfcc(waveform, n_mfcc)
    return apply_cmvn(mfcc)


# ── Embedding-level normalization ─────────────────────────────────────────────

def subtract_cohort_mean(
    embedding: torch.Tensor,
    cohort_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Subtract a pre-computed cohort mean from an embedding.
    Removes additive channel bias shared across the cohort.
    Both inputs should have the same dimensionality.
    Returns a normalized embedding.
    """
    import torch.nn.functional as F
    shifted = embedding - cohort_mean
    return F.normalize(shifted, dim=0)


def estimate_cohort_mean(embeddings: list[torch.Tensor]) -> torch.Tensor:
    """Compute mean embedding from a list of embeddings (e.g., from a dev set)."""
    return torch.stack(embeddings).mean(dim=0)


# ── WCCN (Within-Class Covariance Normalization) ──────────────────────────────

class WCCNTransform:
    """
    Whitens the embedding space using within-class covariance.

    Usage:
        # Fit on a set of (embedding, speaker_id) pairs from a dev set
        wccn = WCCNTransform()
        wccn.fit(embeddings, speaker_ids)

        # Apply to new embeddings
        normalized = wccn.transform(embedding)

        # Save/load
        wccn.save("configs/wccn.pt")
        wccn.load("configs/wccn.pt")
    """

    def __init__(self):
        self._W: torch.Tensor | None = None   # whitening matrix

    def fit(self, embeddings: list[torch.Tensor], speaker_ids: list[str]) -> None:
        """
        Estimate the within-class covariance matrix and compute its inverse Cholesky.

        Args:
            embeddings: list of normalized embedding tensors
            speaker_ids: parallel list of speaker labels
        """
        E = torch.stack(embeddings)  # (N, D)
        D = E.shape[1]

        # Group by speaker
        speakers: dict[str, list[int]] = {}
        for i, sid in enumerate(speaker_ids):
            speakers.setdefault(sid, []).append(i)

        # Within-class covariance
        W = torch.zeros(D, D)
        total = 0
        for idxs in speakers.values():
            if len(idxs) < 2:
                continue
            cluster = E[idxs]
            mu = cluster.mean(dim=0, keepdim=True)
            diff = cluster - mu
            W += diff.T @ diff
            total += len(idxs)

        if total == 0:
            raise ValueError("Need at least one speaker with ≥ 2 samples to fit WCCN.")

        W /= total
        W += 1e-6 * torch.eye(D)   # regularize

        # Whitening matrix = inverse Cholesky of W
        L = torch.linalg.cholesky(W)
        self._W = torch.linalg.inv(L).T   # (D, D)

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply WCCN whitening to a single embedding. Returns normalized result."""
        import torch.nn.functional as F
        if self._W is None:
            raise RuntimeError("WCCNTransform has not been fitted yet.")
        whitened = embedding @ self._W
        return F.normalize(whitened, dim=0)

    def save(self, path: str | Path) -> None:
        if self._W is None:
            raise RuntimeError("Nothing to save — fit first.")
        torch.save({"W": self._W}, str(path))

    def load(self, path: str | Path) -> None:
        data = torch.load(str(path), weights_only=True)
        self._W = data["W"]
