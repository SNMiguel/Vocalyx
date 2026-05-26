"""
Anti-spoofing fusion layer.

Combines signals from three detectors:
  1. Deepfake detector  (SpectralAntiSpoof or HFAntiSpoof)
  2. Replay detector    (ReplayDetector)
  3. Channel mismatch   (ChannelMismatchDetector — used to relax thresholds)

Decision logic:
  - Either detector flagging spoof/replay is sufficient to reject (OR gate)
    unless channel mismatch is detected, in which case thresholds are relaxed.
  - A confidence-band zone between accept and reject triggers a "retry" signal.
  - Final decision: ACCEPT | REJECT | RETRY

This is the "defense in depth" principle from the project brief:
never rely on a single signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from src.antispoofing.deepfake_detector import SpoofResult, detect_spoof
from src.antispoofing.replay_detector import ReplayResult, detect_replay
from src.antispoofing.channel_detector import (
    ChannelMismatchResult, detect_channel_mismatch, adjusted_spoof_threshold,
)


class SpoofDecision(str, Enum):
    ACCEPT = "accept"    # real speech, no attack detected
    REJECT = "reject"    # spoof or replay detected with high confidence
    RETRY  = "retry"     # low confidence — request another sample


@dataclass
class FusionResult:
    decision: SpoofDecision
    fused_spoof_score: float       # weighted combination of all detectors [0, 1]
    deepfake: SpoofResult
    replay: ReplayResult
    channel_mismatch: ChannelMismatchResult | None
    effective_threshold: float     # threshold after mismatch adjustment
    reject_threshold: float
    retry_threshold: float
    explanation: str               # human-readable reason for decision


# ── fusion weights ────────────────────────────────────────────────────────────

_DEEPFAKE_WEIGHT = 0.60
_REPLAY_WEIGHT   = 0.40

# Decision thresholds on the fused score
_DEFAULT_REJECT_THRESHOLD = 0.50   # above this → REJECT
_DEFAULT_RETRY_THRESHOLD  = 0.35   # between this and reject → RETRY
                                   # below retry → ACCEPT


def run_antispoof_fusion(
    probe_waveform: torch.Tensor,
    enroll_waveform: torch.Tensor | None = None,
    reject_threshold: float = _DEFAULT_REJECT_THRESHOLD,
    retry_threshold: float  = _DEFAULT_RETRY_THRESHOLD,
    use_hf_detector: bool   = False,
) -> FusionResult:
    """
    Run the full anti-spoofing pipeline on a probe waveform.

    Args:
        probe_waveform:  (1, samples) 16kHz audio to evaluate
        enroll_waveform: (1, samples) enrollment audio for channel mismatch check.
                         If None, channel mismatch detection is skipped.
        reject_threshold: fused score above this → REJECT
        retry_threshold:  fused score above this (but below reject) → RETRY
        use_hf_detector:  use HuggingFace model instead of spectral detector

    Returns:
        FusionResult with final decision and full score breakdown
    """
    # 1. Deepfake detection
    deepfake_result = detect_spoof(probe_waveform, use_hf=use_hf_detector)

    # 2. Replay detection
    replay_result = detect_replay(probe_waveform)

    # 3. Channel mismatch (adjusts effective thresholds)
    mismatch_result: ChannelMismatchResult | None = None
    effective_reject = reject_threshold
    effective_retry  = retry_threshold

    if enroll_waveform is not None:
        mismatch_result = detect_channel_mismatch(enroll_waveform, probe_waveform)
        relaxation = mismatch_result.spoof_suspicion_adjustment
        effective_reject += relaxation
        effective_retry  += relaxation

    # 4. Fuse scores
    fused = (
        _DEEPFAKE_WEIGHT * deepfake_result.spoof_score +
        _REPLAY_WEIGHT   * replay_result.replay_score
    )
    fused = max(0.0, min(1.0, fused))

    # 5. Decision
    if fused >= effective_reject:
        decision = SpoofDecision.REJECT
        explanation = _build_explanation(deepfake_result, replay_result, mismatch_result, "rejected")
    elif fused >= effective_retry:
        decision = SpoofDecision.RETRY
        explanation = _build_explanation(deepfake_result, replay_result, mismatch_result, "low-confidence — retry")
    else:
        decision = SpoofDecision.ACCEPT
        explanation = _build_explanation(deepfake_result, replay_result, mismatch_result, "accepted")

    return FusionResult(
        decision=decision,
        fused_spoof_score=round(fused, 4),
        deepfake=deepfake_result,
        replay=replay_result,
        channel_mismatch=mismatch_result,
        effective_threshold=round(effective_reject, 4),
        reject_threshold=reject_threshold,
        retry_threshold=retry_threshold,
        explanation=explanation,
    )


def _build_explanation(
    deepfake: SpoofResult,
    replay: ReplayResult,
    mismatch: ChannelMismatchResult | None,
    verdict: str,
) -> str:
    parts = [f"verdict={verdict}"]
    parts.append(f"deepfake_score={deepfake.spoof_score:.3f}({deepfake.detector})")
    parts.append(f"replay_score={replay.replay_score:.3f}")
    if mismatch and mismatch.is_mismatch:
        parts.append(
            f"channel_mismatch=yes(adjustment=+{mismatch.spoof_suspicion_adjustment:.3f})"
        )
    return " | ".join(parts)


def format_fusion_report(result: FusionResult) -> str:
    lines = [
        f"Anti-Spoof Decision: {result.decision.value.upper()}",
        f"  Fused score:   {result.fused_spoof_score:.3f}  "
        f"(threshold: reject>{result.effective_threshold:.3f}  retry>{result.retry_threshold:.3f})",
        f"  Deepfake:      score={result.deepfake.spoof_score:.3f}  "
        f"is_spoof={result.deepfake.is_spoof}  detector={result.deepfake.detector}",
        f"  Replay:        score={result.replay.replay_score:.3f}  "
        f"is_replay={result.replay.is_replay}",
    ]
    if result.channel_mismatch:
        cm = result.channel_mismatch
        lines.append(
            f"  Channel:       mismatch={cm.is_mismatch}  "
            f"score={cm.mismatch_score:.3f}  "
            f"threshold_relaxed_by={cm.spoof_suspicion_adjustment:.3f}"
        )
    lines.append(f"  Detail: {result.explanation}")
    return "\n".join(lines)
