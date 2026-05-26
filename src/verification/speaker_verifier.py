import torch
from src.enrollment.embedder import get_embedding
from src.enrollment.enrollment_db import get_enrollment

DEFAULT_THRESHOLD = 0.25  # cosine distance; tune based on EER evaluation


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def verify(user_id: str, waveform: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Verify whether waveform belongs to user_id.

    Returns:
        {
          "user_id": str,
          "score": float,       # cosine similarity [−1, 1]; higher = more similar
          "accepted": bool,
          "threshold": float,
        }
    """
    enrolled_embedding = get_enrollment(user_id)
    probe_embedding = get_embedding(waveform)
    score = cosine_similarity(enrolled_embedding, probe_embedding)

    return {
        "user_id": user_id,
        "score": round(score, 4),
        "accepted": score >= threshold,
        "threshold": threshold,
    }
