"""
End-to-end decision fusion layer.

Combines:
  - Speaker verification score (cosine similarity, Phase 1/3)
  - Anti-spoof fusion result  (deepfake + replay + channel, Phase 5)

Into a single AuthDecision with four possible outcomes:
  ACCEPT   — verified speaker, no spoof detected
  REJECT   — wrong speaker OR confirmed spoof/replay attack
  RETRY    — low confidence on either signal; request another sample
  STEP_UP  — borderline speaker score + suspicious (not confirmed) spoof signal;
             escalate to stronger verification (PIN, face ID, etc.)

Adaptive thresholds:
  Thresholds shift based on the *confidence* of each sub-system.
  When the spoof detector is very confident the audio is real, we can
  afford to slightly relax the speaker similarity threshold (reducing FRR
  for genuine users under adverse conditions). Conversely, when spoof
  confidence is low, we tighten the speaker threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from src.antispoofing.fusion import FusionResult, SpoofDecision, run_antispoof_fusion
from src.verification.speaker_verifier import verify


class AuthDecision(str, Enum):
    ACCEPT  = "accept"    # verified + clean
    REJECT  = "reject"    # wrong speaker or confirmed attack
    RETRY   = "retry"     # try again — low confidence
    STEP_UP = "step_up"   # escalate to additional auth factor


@dataclass
class AuthResult:
    decision: AuthDecision
    speaker_score: float              # cosine similarity [−1, 1]
    spoof_score: float                # fused spoof score [0, 1]
    speaker_accepted: bool
    spoof_accepted: bool              # True = no spoof detected
    effective_sv_threshold: float     # speaker threshold after adaptation
    spoof_result: FusionResult
    explanation: str
    metadata: dict = field(default_factory=dict)


# ── threshold configuration ───────────────────────────────────────────────────

@dataclass
class DecisionConfig:
    # Speaker verification base thresholds
    sv_accept_threshold: float = 0.25    # above this → speaker verified
    sv_retry_threshold:  float = 0.15    # between this and accept → retry
    # (below sv_retry_threshold → hard reject on speaker)

    # Anti-spoof thresholds (forwarded to run_antispoof_fusion)
    spoof_reject_threshold: float = 0.50
    spoof_retry_threshold:  float = 0.35

    # Adaptive threshold range: how much spoof confidence shifts SV threshold
    sv_threshold_range: float = 0.05     # ±5% adjustment based on spoof confidence

    # Step-up: triggered when speaker is in the retry zone AND spoof in retry zone
    enable_step_up: bool = True


DEFAULT_CONFIG = DecisionConfig()


# ── adaptive threshold ────────────────────────────────────────────────────────

def _adapt_sv_threshold(
    base_threshold: float,
    spoof_result: FusionResult,
    config: DecisionConfig,
) -> float:
    """
    Adjust the speaker verification threshold based on anti-spoof confidence.

    When spoof detector is highly confident the audio is REAL (low spoof score,
    high confidence), we relax the SV threshold slightly — the clean channel
    gives us more certainty the speaker is who they claim to be.

    When spoof detector is uncertain (low confidence), we tighten the SV
    threshold — we need a stronger speaker match to compensate.
    """
    spoof_score = spoof_result.fused_spoof_score
    spoof_confidence = spoof_result.deepfake.confidence

    # How "real" is the audio? 0 = definitely spoof, 1 = definitely real
    realness = (1.0 - spoof_score) * spoof_confidence

    # Map realness [0, 1] → threshold adjustment [-range, +range]
    # realness=1.0 → relax by full range (lower threshold = easier to pass)
    # realness=0.0 → tighten by full range (higher threshold = harder to pass)
    adjustment = config.sv_threshold_range * (0.5 - realness)
    return round(base_threshold + adjustment, 4)


# ── main fusion function ──────────────────────────────────────────────────────

def make_auth_decision(
    user_id: str,
    probe_waveform: torch.Tensor,
    enroll_waveform: Optional[torch.Tensor] = None,
    config: DecisionConfig = DEFAULT_CONFIG,
    language: str = "default",
) -> AuthResult:
    """
    Full end-to-end authentication decision for a single attempt.

    Args:
        user_id:         the user claiming to authenticate
        probe_waveform:  (1, samples) 16kHz audio of the claim
        enroll_waveform: (1, samples) 16kHz enrollment audio for channel check.
                         Pass None to skip channel mismatch detection.
        config:          threshold and behaviour configuration
        language:        spoken language (for adaptive SV threshold, Phase 3)

    Returns:
        AuthResult with decision and full score breakdown
    """
    # 1. Anti-spoof check — runs first so result can adapt SV threshold
    spoof_result = run_antispoof_fusion(
        probe_waveform,
        enroll_waveform=enroll_waveform,
        reject_threshold=config.spoof_reject_threshold,
        retry_threshold=config.spoof_retry_threshold,
    )

    # Hard spoof reject — no need to run speaker verification
    if spoof_result.decision == SpoofDecision.REJECT:
        return AuthResult(
            decision=AuthDecision.REJECT,
            speaker_score=0.0,
            spoof_score=spoof_result.fused_spoof_score,
            speaker_accepted=False,
            spoof_accepted=False,
            effective_sv_threshold=config.sv_accept_threshold,
            spoof_result=spoof_result,
            explanation=f"rejected: confirmed spoof/replay (score={spoof_result.fused_spoof_score:.3f})",
        )

    # 2. Adaptive speaker verification threshold
    effective_sv_threshold = _adapt_sv_threshold(
        config.sv_accept_threshold, spoof_result, config
    )

    # 3. Speaker verification
    sv_result = verify(user_id, probe_waveform, threshold=effective_sv_threshold)
    speaker_score = sv_result["score"]
    speaker_accepted = sv_result["accepted"]

    spoof_accepted = spoof_result.decision != SpoofDecision.REJECT
    spoof_uncertain = spoof_result.decision == SpoofDecision.RETRY
    speaker_uncertain = (
        config.sv_retry_threshold <= speaker_score < effective_sv_threshold
    )

    # 4. Decision logic
    if speaker_accepted and spoof_accepted and not spoof_uncertain:
        decision = AuthDecision.ACCEPT
        explanation = (
            f"accepted: sv={speaker_score:.3f}≥{effective_sv_threshold:.3f}, "
            f"spoof={spoof_result.fused_spoof_score:.3f} (clean)"
        )

    elif not speaker_accepted and speaker_score < config.sv_retry_threshold:
        # Speaker score far below threshold → definitive wrong speaker
        decision = AuthDecision.REJECT
        explanation = (
            f"rejected: speaker score {speaker_score:.3f} below hard floor "
            f"{config.sv_retry_threshold:.3f}"
        )

    elif config.enable_step_up and speaker_uncertain and spoof_uncertain:
        # Both systems uncertain → escalate
        decision = AuthDecision.STEP_UP
        explanation = (
            f"step_up: sv borderline ({speaker_score:.3f}), "
            f"spoof uncertain ({spoof_result.fused_spoof_score:.3f})"
        )

    elif speaker_uncertain or spoof_uncertain:
        decision = AuthDecision.RETRY
        reason = []
        if speaker_uncertain:
            reason.append(f"sv borderline ({speaker_score:.3f})")
        if spoof_uncertain:
            reason.append(f"spoof uncertain ({spoof_result.fused_spoof_score:.3f})")
        explanation = "retry: " + ", ".join(reason)

    else:
        # Speaker score in retry zone but spoof clean, or vice versa
        decision = AuthDecision.RETRY
        explanation = (
            f"retry: sv={speaker_score:.3f} threshold={effective_sv_threshold:.3f} "
            f"spoof={spoof_result.fused_spoof_score:.3f}"
        )

    return AuthResult(
        decision=decision,
        speaker_score=speaker_score,
        spoof_score=spoof_result.fused_spoof_score,
        speaker_accepted=speaker_accepted,
        spoof_accepted=spoof_accepted,
        effective_sv_threshold=effective_sv_threshold,
        spoof_result=spoof_result,
        explanation=explanation,
    )


def format_auth_report(result: AuthResult) -> str:
    lines = [
        f"Auth Decision: {result.decision.value.upper()}",
        f"  Speaker:  score={result.speaker_score:.3f}  "
        f"threshold={result.effective_sv_threshold:.3f}  "
        f"accepted={result.speaker_accepted}",
        f"  Spoof:    score={result.spoof_score:.3f}  "
        f"accepted={result.spoof_accepted}  "
        f"({result.spoof_result.decision.value})",
        f"  Detail:   {result.explanation}",
    ]
    if result.spoof_result.channel_mismatch:
        cm = result.spoof_result.channel_mismatch
        lines.append(
            f"  Channel:  mismatch={cm.is_mismatch}  "
            f"adjustment={cm.spoof_suspicion_adjustment:+.3f}"
        )
    return "\n".join(lines)
