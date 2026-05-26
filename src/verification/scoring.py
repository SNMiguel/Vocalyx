"""
Adaptive scoring layer for speaker verification.

Responsibilities:
  1. Language-conditioned thresholds — cross-language verification is harder,
     so we relax thresholds slightly for non-native language pairs.
  2. Multi-backend score fusion — combine scores from several embedders via
     a simple weighted average.
  3. Accent disentanglement helper — projects out the language direction from
     a speaker embedding to reduce language bias.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from src.verification.multilingual import EmbedderBackend, get_embedding


# ── per-language threshold adjustments ───────────────────────────────────────
# Baseline threshold is 0.25. Cross-language pairs score lower on average,
# so we lower the threshold to maintain acceptable FRR.

_LANGUAGE_THRESHOLDS: dict[str, float] = {
    "english":   0.25,
    "french":    0.23,
    "spanish":   0.23,
    "arabic":    0.20,
    "mandarin":  0.20,
    "german":    0.22,
    "hindi":     0.20,
    "default":   0.22,   # fallback for unseen languages
}

# Weights for multi-backend fusion (must sum to 1)
_FUSION_WEIGHTS: dict[EmbedderBackend, float] = {
    EmbedderBackend.WAVLM: 0.50,
    EmbedderBackend.ECAPA: 0.30,
    EmbedderBackend.XLSR:  0.20,
}


@dataclass
class ScoringResult:
    score: float
    accepted: bool
    threshold: float
    language: str
    backend: str
    fusion_scores: dict[str, float] | None = None


def get_threshold(language: str = "default") -> float:
    return _LANGUAGE_THRESHOLDS.get(language.lower(), _LANGUAGE_THRESHOLDS["default"])


def score_pair(
    enrolled_embedding: torch.Tensor,
    probe_waveform: torch.Tensor,
    language: str = "default",
    backend: str | EmbedderBackend = EmbedderBackend.WAVLM,
) -> ScoringResult:
    """
    Score a verification trial using a single embedder backend.

    Args:
        enrolled_embedding: pre-computed embedding of the enrolled user
        probe_waveform: (1, samples) 16kHz waveform to verify
        language: spoken language — used to select adaptive threshold
        backend: which embedder to use for the probe

    Returns:
        ScoringResult with score, decision, and metadata
    """
    threshold = get_threshold(language)
    probe_embedding = get_embedding(probe_waveform, backend)
    score = F.cosine_similarity(
        enrolled_embedding.unsqueeze(0),
        probe_embedding.unsqueeze(0),
    ).item()

    return ScoringResult(
        score=round(score, 4),
        accepted=score >= threshold,
        threshold=threshold,
        language=language,
        backend=str(backend),
    )


def score_pair_fused(
    enrolled_embeddings: dict[str | EmbedderBackend, torch.Tensor],
    probe_waveform: torch.Tensor,
    language: str = "default",
    weights: dict[EmbedderBackend, float] | None = None,
) -> ScoringResult:
    """
    Fuse scores from multiple embedder backends via weighted average.

    Args:
        enrolled_embeddings: dict mapping backend → enrolled embedding tensor
        probe_waveform: (1, samples) 16kHz waveform to verify
        language: spoken language for adaptive threshold
        weights: per-backend weights (defaults to _FUSION_WEIGHTS)

    Returns:
        ScoringResult with fused score and per-backend breakdown
    """
    w = weights or _FUSION_WEIGHTS
    threshold = get_threshold(language)
    total_weight = 0.0
    fused_score = 0.0
    per_backend: dict[str, float] = {}

    for backend, enrolled_emb in enrolled_embeddings.items():
        backend_key = EmbedderBackend(backend)
        weight = w.get(backend_key, 0.0)
        if weight == 0.0:
            continue
        probe_emb = get_embedding(probe_waveform, backend_key)
        s = F.cosine_similarity(
            enrolled_emb.unsqueeze(0), probe_emb.unsqueeze(0)
        ).item()
        per_backend[str(backend_key)] = round(s, 4)
        fused_score += weight * s
        total_weight += weight

    if total_weight > 0:
        fused_score /= total_weight

    return ScoringResult(
        score=round(fused_score, 4),
        accepted=fused_score >= threshold,
        threshold=threshold,
        language=language,
        backend="fused",
        fusion_scores=per_backend,
    )


def disentangle_language(
    speaker_embedding: torch.Tensor,
    language_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Project out the language direction from a speaker embedding.

    Removes the component of `speaker_embedding` that lies along the
    `language_embedding` direction, reducing language-correlated variance
    while preserving speaker-discriminative information.

    Both inputs must be unit-normalized.
    Returns a normalized embedding.
    """
    # Gram-Schmidt: subtract projection onto language axis
    lang = F.normalize(language_embedding.float(), dim=0)
    projection = (speaker_embedding @ lang) * lang
    disentangled = speaker_embedding - projection
    return F.normalize(disentangled, dim=0)
