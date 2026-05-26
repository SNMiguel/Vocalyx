"""
Evaluation metrics for speaker verification systems.

Terminology:
  - genuine trial: same speaker, should be accepted
  - impostor trial: different speaker, should be rejected
  - spoof trial: synthetic/replayed audio, should be rejected

  FAR  (False Accept Rate)  = impostor trials accepted / total impostor trials
  FRR  (False Reject Rate)  = genuine trials rejected / total genuine trials
  EER  (Equal Error Rate)   = threshold where FAR == FRR (lower is better)
  HTER (Half Total Error)   = (FAR + FRR) / 2 at a given threshold
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VerificationMetrics:
    threshold: float
    far: float
    frr: float
    hter: float
    eer: float
    eer_threshold: float
    n_genuine: int
    n_impostor: int


@dataclass
class SpoofMetrics:
    accuracy: float
    false_alarm_rate: float   # real speech flagged as spoof
    miss_rate: float          # spoof accepted as real
    n_real: int
    n_spoof: int


def compute_far_frr(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float,
) -> tuple[float, float]:
    """Compute FAR and FRR at a given cosine-similarity threshold."""
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    return far, frr


def compute_eer(
    genuine_scores: list[float],
    impostor_scores: list[float],
    n_thresholds: int = 1000,
) -> tuple[float, float]:
    """
    Compute EER by sweeping thresholds and finding where FAR ≈ FRR.
    Returns (eer, eer_threshold).
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    best_eer = 1.0
    best_thresh = thresholds[0]

    for t in thresholds:
        far, frr = compute_far_frr(genuine_scores, impostor_scores, t)
        # EER is where |FAR - FRR| is minimised
        diff = abs(far - frr)
        eer_candidate = (far + frr) / 2
        if diff < abs(best_eer * 2 - (far + frr)) or eer_candidate < best_eer:
            best_eer = eer_candidate
            best_thresh = t

    return best_eer, best_thresh


def compute_verification_metrics(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float = 0.25,
) -> VerificationMetrics:
    far, frr = compute_far_frr(genuine_scores, impostor_scores, threshold)
    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    return VerificationMetrics(
        threshold=threshold,
        far=round(far, 4),
        frr=round(frr, 4),
        hter=round((far + frr) / 2, 4),
        eer=round(eer, 4),
        eer_threshold=round(eer_threshold, 4),
        n_genuine=len(genuine_scores),
        n_impostor=len(impostor_scores),
    )


def compute_spoof_metrics(
    real_scores: list[float],
    spoof_scores: list[float],
    threshold: float = 0.5,
) -> SpoofMetrics:
    """
    real_scores: spoof-detection scores for genuine (real) speech — higher = more real
    spoof_scores: scores for spoofed/synthetic speech — lower = more spoof
    threshold: classify as real if score >= threshold
    """
    real = np.array(real_scores)
    spoof = np.array(spoof_scores)
    false_alarm = float(np.mean(real < threshold))   # real flagged as spoof
    miss = float(np.mean(spoof >= threshold))         # spoof accepted as real
    accuracy = float(np.mean(
        np.concatenate([real >= threshold, spoof < threshold])
    ))
    return SpoofMetrics(
        accuracy=round(accuracy, 4),
        false_alarm_rate=round(false_alarm, 4),
        miss_rate=round(miss, 4),
        n_real=len(real_scores),
        n_spoof=len(spoof_scores),
    )


def format_report(
    condition: str,
    metrics: VerificationMetrics,
    spoof: SpoofMetrics | None = None,
) -> str:
    lines = [
        f"=== {condition} ===",
        f"  Trials:    {metrics.n_genuine} genuine  |  {metrics.n_impostor} impostor",
        f"  EER:       {metrics.eer * 100:.2f}%  (threshold={metrics.eer_threshold:.3f})",
        f"  @ thresh {metrics.threshold:.3f}:  FAR={metrics.far * 100:.2f}%  FRR={metrics.frr * 100:.2f}%  HTER={metrics.hter * 100:.2f}%",
    ]
    if spoof:
        lines += [
            f"  Spoof det: accuracy={spoof.accuracy * 100:.1f}%  "
            f"FA={spoof.false_alarm_rate * 100:.1f}%  "
            f"Miss={spoof.miss_rate * 100:.1f}%",
        ]
    return "\n".join(lines)
