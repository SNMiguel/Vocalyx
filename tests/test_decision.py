"""
Tests for Phase 6: decision fusion layer and session manager.

Uses mocked sub-systems so tests run without model downloads or enrollment DB.
"""

import time
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.decision.fusion_layer import (
    AuthDecision, AuthResult, DecisionConfig,
    make_auth_decision, format_auth_report, _adapt_sv_threshold,
)
from src.decision.session import (
    SessionManager, SessionConfig, SessionStatus,
)
from src.antispoofing.fusion import FusionResult, SpoofDecision
from src.antispoofing.deepfake_detector import SpoofResult
from src.antispoofing.replay_detector import ReplayResult
from src.antispoofing.channel_detector import ChannelMismatchResult

SR = 16000


# ── test helpers ──────────────────────────────────────────────────────────────

def _waveform(duration=2.0) -> torch.Tensor:
    t = torch.linspace(0, duration, int(SR * duration))
    return (0.3 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)


def _make_spoof_result(spoof_score=0.1, decision=SpoofDecision.ACCEPT) -> FusionResult:
    """Build a minimal FusionResult for injection into tests."""
    deepfake = SpoofResult(
        is_spoof=spoof_score > 0.5,
        spoof_score=spoof_score,
        real_score=1.0 - spoof_score,
        confidence=abs(spoof_score - 0.5) * 2,
        detector="mock",
    )
    replay = ReplayResult(
        is_replay=False,
        replay_score=0.1,
        confidence=0.8,
        features={},
    )
    mismatch = ChannelMismatchResult(
        mismatch_score=0.0,
        is_mismatch=False,
        threshold=0.35,
        spectral_distance=0.0,
        bandwidth_ratio=1.0,
        snr_delta_db=0.0,
        spoof_suspicion_adjustment=0.0,
    )
    return FusionResult(
        decision=decision,
        fused_spoof_score=spoof_score,
        deepfake=deepfake,
        replay=replay,
        channel_mismatch=mismatch,
        effective_threshold=0.5,
        reject_threshold=0.5,
        retry_threshold=0.35,
        explanation="mock",
    )


def _make_sv_result(score=0.8, accepted=True) -> dict:
    return {"user_id": "test_user", "score": score, "accepted": accepted, "threshold": 0.25}


# ── adaptive threshold tests ──────────────────────────────────────────────────

class TestAdaptiveThreshold:
    def test_high_confidence_real_lowers_threshold(self):
        config = DecisionConfig(sv_accept_threshold=0.25, sv_threshold_range=0.05)
        spoof = _make_spoof_result(spoof_score=0.05)  # very real, high confidence
        thresh = _adapt_sv_threshold(0.25, spoof, config)
        assert thresh < 0.25, f"Expected threshold to drop below base, got {thresh}"

    def test_uncertain_spoof_raises_threshold(self):
        config = DecisionConfig(sv_accept_threshold=0.25, sv_threshold_range=0.05)
        spoof = _make_spoof_result(spoof_score=0.50)  # maximum uncertainty
        thresh = _adapt_sv_threshold(0.25, spoof, config)
        assert thresh >= 0.25, f"Expected threshold to stay or rise, got {thresh}"

    def test_adaptation_within_range(self):
        config = DecisionConfig(sv_accept_threshold=0.25, sv_threshold_range=0.05)
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            spoof = _make_spoof_result(spoof_score=score)
            thresh = _adapt_sv_threshold(0.25, spoof, config)
            assert 0.20 <= thresh <= 0.30, f"Threshold {thresh} out of ±range for score {score}"


# ── make_auth_decision tests ──────────────────────────────────────────────────

class TestMakeAuthDecision:

    def _patch_and_decide(self, sv_score, sv_accepted, spoof_score, spoof_decision,
                          config=None):
        spoof_result = _make_spoof_result(spoof_score, spoof_decision)
        sv_result = _make_sv_result(sv_score, sv_accepted)
        with patch("src.decision.fusion_layer.run_antispoof_fusion", return_value=spoof_result), \
             patch("src.decision.fusion_layer.verify", return_value=sv_result):
            return make_auth_decision(
                "test_user", _waveform(), config=config or DecisionConfig()
            )

    def test_accept_on_valid_speaker_clean_audio(self):
        result = self._patch_and_decide(
            sv_score=0.8, sv_accepted=True,
            spoof_score=0.1, spoof_decision=SpoofDecision.ACCEPT,
        )
        assert result.decision == AuthDecision.ACCEPT

    def test_reject_on_confirmed_spoof(self):
        result = self._patch_and_decide(
            sv_score=0.9, sv_accepted=True,
            spoof_score=0.9, spoof_decision=SpoofDecision.REJECT,
        )
        assert result.decision == AuthDecision.REJECT
        assert not result.spoof_accepted

    def test_reject_on_wrong_speaker(self):
        result = self._patch_and_decide(
            sv_score=0.05, sv_accepted=False,
            spoof_score=0.1, spoof_decision=SpoofDecision.ACCEPT,
        )
        assert result.decision == AuthDecision.REJECT

    def test_retry_on_borderline_speaker(self):
        config = DecisionConfig(sv_accept_threshold=0.25, sv_retry_threshold=0.15)
        result = self._patch_and_decide(
            sv_score=0.20, sv_accepted=False,
            spoof_score=0.1, spoof_decision=SpoofDecision.ACCEPT,
            config=config,
        )
        assert result.decision == AuthDecision.RETRY

    def test_retry_on_uncertain_spoof(self):
        result = self._patch_and_decide(
            sv_score=0.8, sv_accepted=True,
            spoof_score=0.40, spoof_decision=SpoofDecision.RETRY,
        )
        assert result.decision in (AuthDecision.RETRY, AuthDecision.STEP_UP)

    def test_step_up_on_dual_uncertainty(self):
        config = DecisionConfig(
            sv_accept_threshold=0.25, sv_retry_threshold=0.15, enable_step_up=True
        )
        result = self._patch_and_decide(
            sv_score=0.20, sv_accepted=False,   # borderline speaker
            spoof_score=0.42, spoof_decision=SpoofDecision.RETRY,  # uncertain spoof
            config=config,
        )
        assert result.decision == AuthDecision.STEP_UP

    def test_format_report_runs(self):
        result = self._patch_and_decide(
            sv_score=0.8, sv_accepted=True,
            spoof_score=0.1, spoof_decision=SpoofDecision.ACCEPT,
        )
        report = format_auth_report(result)
        assert "Auth Decision" in report
        assert "Speaker" in report


# ── session manager tests ─────────────────────────────────────────────────────

class TestSessionManager:

    def _make_mgr(self, max_attempts=4, step_up_after=2):
        auth_config = DecisionConfig()
        sess_config = SessionConfig(
            max_attempts=max_attempts,
            step_up_after=step_up_after,
            session_timeout_seconds=60.0,
            lockout_duration_seconds=60.0,
        )
        return SessionManager(auth_config=auth_config, session_config=sess_config)

    def _auth(self, mgr, session_id, sv_score=0.8, sv_accepted=True,
              spoof_score=0.1, spoof_decision=SpoofDecision.ACCEPT):
        spoof_result = _make_spoof_result(spoof_score, spoof_decision)
        sv_result = _make_sv_result(sv_score, sv_accepted)
        with patch("src.decision.fusion_layer.run_antispoof_fusion", return_value=spoof_result), \
             patch("src.decision.fusion_layer.verify", return_value=sv_result):
            return mgr.authenticate(session_id, _waveform())

    def test_accept_sets_session_accepted(self):
        mgr = self._make_mgr()
        sess = mgr.start_session("alice")
        self._auth(mgr, sess.session_id, sv_score=0.9, sv_accepted=True,
                   spoof_score=0.05, spoof_decision=SpoofDecision.ACCEPT)
        assert sess.status == SessionStatus.ACCEPTED

    def test_retry_increments_count(self):
        mgr = self._make_mgr(step_up_after=5)
        sess = mgr.start_session("bob")
        # Borderline speaker, uncertain spoof → RETRY
        self._auth(mgr, sess.session_id, sv_score=0.20, sv_accepted=False,
                   spoof_score=0.10, spoof_decision=SpoofDecision.ACCEPT)
        assert sess.retry_count == 1

    def test_step_up_forced_after_retries(self):
        mgr = self._make_mgr(step_up_after=1)
        sess = mgr.start_session("carol")
        # First attempt: RETRY
        r1 = self._auth(mgr, sess.session_id, sv_score=0.20, sv_accepted=False,
                        spoof_score=0.10, spoof_decision=SpoofDecision.ACCEPT)
        assert r1.decision == AuthDecision.RETRY
        # Second attempt: should be forced to STEP_UP
        r2 = self._auth(mgr, sess.session_id, sv_score=0.20, sv_accepted=False,
                        spoof_score=0.10, spoof_decision=SpoofDecision.ACCEPT)
        assert r2.decision == AuthDecision.STEP_UP

    def test_lockout_after_max_attempts(self):
        mgr = self._make_mgr(max_attempts=2)
        sess = mgr.start_session("dave")
        # Two hard rejects → lockout
        self._auth(mgr, sess.session_id, sv_score=0.01, sv_accepted=False,
                   spoof_score=0.05, spoof_decision=SpoofDecision.ACCEPT)
        self._auth(mgr, sess.session_id, sv_score=0.01, sv_accepted=False,
                   spoof_score=0.05, spoof_decision=SpoofDecision.ACCEPT)
        assert sess.is_locked

    def test_locked_session_raises(self):
        mgr = self._make_mgr(max_attempts=1)
        sess = mgr.start_session("eve")
        self._auth(mgr, sess.session_id, sv_score=0.01, sv_accepted=False,
                   spoof_score=0.05, spoof_decision=SpoofDecision.ACCEPT)
        with pytest.raises(PermissionError):
            self._auth(mgr, sess.session_id)

    def test_unknown_session_raises(self):
        mgr = self._make_mgr()
        with pytest.raises(ValueError):
            self._auth(mgr, "nonexistent-session-id")

    def test_session_summary_structure(self):
        mgr = self._make_mgr()
        sess = mgr.start_session("frank")
        self._auth(mgr, sess.session_id, sv_score=0.9, sv_accepted=True,
                   spoof_score=0.05, spoof_decision=SpoofDecision.ACCEPT)
        summary = mgr.session_summary(sess.session_id)
        assert summary["user_id"] == "frank"
        assert summary["total_attempts"] == 1
        assert len(summary["attempts"]) == 1

    def test_tightening_on_retry(self):
        """Verify threshold tightens after a retry attempt."""
        mgr = self._make_mgr(step_up_after=10)
        sess = mgr.start_session("grace")
        # First attempt: RETRY (borderline speaker)
        self._auth(mgr, sess.session_id, sv_score=0.20, sv_accepted=False,
                   spoof_score=0.10, spoof_decision=SpoofDecision.ACCEPT)
        cfg1 = mgr._escalated_config(sess)
        assert cfg1.sv_accept_threshold > DecisionConfig().sv_accept_threshold
