"""
Tests for Phase 5: deepfake detection, replay detection, and anti-spoof fusion.
All tests use synthetic audio — no real recordings or model downloads needed.
"""

import torch
import pytest

from src.antispoofing.deepfake_detector import (
    SpectralAntiSpoof, detect_spoof,
    _spectral_flatness, _hnr, _mfcc_delta_variance, _pitch_jitter,
)
from src.antispoofing.replay_detector import (
    detect_replay,
    _reverb_tail_energy, _sub_band_energy_ratio,
    _silence_noise_floor, _spectral_flux_peaks,
)
from src.antispoofing.fusion import (
    run_antispoof_fusion, SpoofDecision, format_fusion_report,
)
from src.preprocessing.augmentation import room_reverb, add_noise, bluetooth_compress

SR = 16000


# ── helpers ──────────────────────────────────────────────────────────────────

def _sine(freq=220.0, duration=3.0) -> torch.Tensor:
    t = torch.linspace(0, duration, int(SR * duration))
    return (0.5 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def _speech_like(duration=3.0) -> torch.Tensor:
    """Approximation of voiced speech: amplitude-modulated harmonic complex."""
    t = torch.linspace(0, duration, int(SR * duration))
    # Harmonic structure with natural jitter
    f0 = 150.0
    jitter = 0.01 * torch.randn(t.shape)
    sig = sum(
        (1.0 / k) * torch.sin(2 * torch.pi * k * (f0 + jitter * 10) * t)
        for k in range(1, 8)
    )
    # Amplitude modulation (mimics syllabic rhythm ~4 Hz)
    am = 0.5 + 0.5 * torch.sin(2 * torch.pi * 4.0 * t)
    return (am * sig * 0.3).unsqueeze(0)


def _tts_like(duration=3.0) -> torch.Tensor:
    """Synthetic-speech approximation: very regular, low jitter, high flatness."""
    t = torch.linspace(0, duration, int(SR * duration))
    f0 = 150.0
    # Perfectly regular harmonic — no jitter, no natural variation
    sig = sum(
        (1.0 / k) * torch.sin(2 * torch.pi * k * f0 * t)
        for k in range(1, 8)
    )
    # Smooth, slowly-varying amplitude envelope (unnaturally regular)
    am = 0.5 + 0.5 * torch.sin(2 * torch.pi * 0.5 * t)
    return (am * sig * 0.3).unsqueeze(0)


# ── spectral feature tests ────────────────────────────────────────────────────

class TestSpectralFeatures:
    def test_spectral_flatness_range(self):
        w = _speech_like()
        sf = _spectral_flatness(w)
        assert 0.0 <= sf <= 1.0

    def test_hnr_range(self):
        w = _speech_like()
        hnr = _hnr(w)
        assert -10.0 <= hnr <= 40.0

    def test_delta_variance_positive(self):
        w = _speech_like()
        dv = _mfcc_delta_variance(w)
        assert dv >= 0.0

    def test_pitch_jitter_range(self):
        w = _speech_like()
        jitter = _pitch_jitter(w)
        assert jitter >= 0.0

    def test_tts_has_lower_delta_variance_than_speech(self):
        speech = _speech_like(duration=4.0)
        tts    = _tts_like(duration=4.0)
        assert _mfcc_delta_variance(tts) < _mfcc_delta_variance(speech)

    def test_tts_has_higher_hnr_than_speech(self):
        """TTS has purer harmonics → higher HNR (no pitch jitter)."""
        speech = _speech_like(duration=4.0)
        tts    = _tts_like(duration=4.0)
        assert _hnr(tts) > _hnr(speech), (
            f"TTS HNR ({_hnr(tts):.1f}) should exceed speech HNR ({_hnr(speech):.1f})"
        )

    def test_tts_has_lower_pitch_jitter_than_speech(self):
        """TTS has perfectly regular pitch → lower jitter CV."""
        speech = _speech_like(duration=4.0)
        tts    = _tts_like(duration=4.0)
        assert _pitch_jitter(tts) < _pitch_jitter(speech), (
            f"TTS jitter ({_pitch_jitter(tts):.4f}) should be less than "
            f"speech jitter ({_pitch_jitter(speech):.4f})"
        )


# ── SpectralAntiSpoof tests ───────────────────────────────────────────────────

class TestSpectralAntiSpoof:
    def test_returns_spoof_result(self):
        detector = SpectralAntiSpoof()
        result = detector.predict(_speech_like())
        assert 0.0 <= result.spoof_score <= 1.0
        assert 0.0 <= result.real_score <= 1.0
        assert abs(result.spoof_score + result.real_score - 1.0) < 1e-5
        assert result.features is not None

    def test_scores_sum_to_one(self):
        detector = SpectralAntiSpoof()
        result = detector.predict(_speech_like())
        assert abs(result.spoof_score + result.real_score - 1.0) < 1e-5

    def test_tts_has_lower_jitter_and_higher_hnr(self):
        """
        Verify the two strongest TTS-discriminating features hold for our
        synthetic signals — rather than testing end-to-end score ordering,
        which depends on calibration against real speech corpora.
        """
        detector = SpectralAntiSpoof()
        speech_feats = detector.predict(_speech_like(duration=5.0)).features
        tts_feats    = detector.predict(_tts_like(duration=5.0)).features
        assert tts_feats["pitch_jitter"] < speech_feats["pitch_jitter"], (
            f"TTS jitter {tts_feats['pitch_jitter']} should be less than "
            f"speech jitter {speech_feats['pitch_jitter']}"
        )
        assert tts_feats["hnr_db"] > speech_feats["hnr_db"], (
            f"TTS HNR {tts_feats['hnr_db']} should exceed speech HNR {speech_feats['hnr_db']}"
        )

    def test_detect_spoof_convenience(self):
        result = detect_spoof(_speech_like())
        assert result.detector == "spectral"
        assert isinstance(result.is_spoof, bool)


# ── replay detector tests ─────────────────────────────────────────────────────

class TestReplayDetector:
    def test_returns_result(self):
        result = detect_replay(_speech_like())
        assert 0.0 <= result.replay_score <= 1.0
        assert len(result.features) == 4

    def test_heavily_reverbed_scores_higher(self):
        clean   = _speech_like(duration=4.0)
        reverby = room_reverb(clean, rt60=0.8)
        r_clean  = detect_replay(clean)
        r_reverb = detect_replay(reverby)
        assert r_reverb.replay_score >= r_clean.replay_score

    def test_noisy_floor_detection(self):
        clean = _speech_like(duration=4.0)
        noisy = add_noise(clean, snr_db=5.0, noise_type="white")
        r_clean = detect_replay(clean)
        r_noisy = detect_replay(noisy)
        # More noise → higher noise floor score
        assert r_noisy.features["noise_floor"] >= r_clean.features["noise_floor"]

    def test_sub_band_ratio_bluetooth(self):
        clean = _speech_like(duration=4.0)
        bt    = bluetooth_compress(clean)
        r_clean = detect_replay(clean)
        r_bt    = detect_replay(bt)
        # Bluetooth cuts low frequencies → higher sub_band_ratio spoof score
        assert r_bt.features["sub_band_ratio"] >= r_clean.features["sub_band_ratio"]

    def test_feature_scores_bounded(self):
        result = detect_replay(_speech_like())
        for k, v in result.features.items():
            assert 0.0 <= v <= 1.0, f"Feature {k}={v} out of range"


# ── fusion tests ──────────────────────────────────────────────────────────────

class TestAntispoofFusion:
    def test_returns_valid_decision(self):
        result = run_antispoof_fusion(_speech_like())
        assert result.decision in list(SpoofDecision)
        assert 0.0 <= result.fused_spoof_score <= 1.0

    def test_very_clean_speech_not_rejected(self):
        """Natural-sounding signal at high threshold should not be hard-rejected."""
        clean = _speech_like(duration=5.0)
        result = run_antispoof_fusion(clean, reject_threshold=0.90, retry_threshold=0.0)
        # Score well below 0.90 → must not be REJECT (may be ACCEPT or RETRY at 0.0 band)
        assert result.decision != SpoofDecision.REJECT, (
            f"Speech-like signal was rejected with score {result.fused_spoof_score:.3f}"
        )

    def test_channel_mismatch_relaxes_threshold(self):
        clean = _speech_like(duration=3.0)
        bt    = bluetooth_compress(clean)
        # Without mismatch check
        r_no_check = run_antispoof_fusion(bt)
        # With mismatch check (enroll = clean, probe = bluetooth)
        r_with_check = run_antispoof_fusion(bt, enroll_waveform=clean)
        # Effective threshold should be higher (relaxed) when mismatch is detected
        assert r_with_check.effective_threshold >= r_no_check.effective_threshold

    def test_channel_mismatch_skipped_when_no_enroll(self):
        result = run_antispoof_fusion(_speech_like())
        assert result.channel_mismatch is None

    def test_channel_mismatch_present_when_enroll_provided(self):
        w = _speech_like()
        result = run_antispoof_fusion(w, enroll_waveform=w)
        assert result.channel_mismatch is not None

    def test_format_report_runs(self):
        result = run_antispoof_fusion(_speech_like())
        report = format_fusion_report(result)
        assert "Anti-Spoof Decision" in report
        assert "Fused score" in report

    def test_reject_threshold_respected(self):
        """Force a reject by setting a very low threshold."""
        result = run_antispoof_fusion(_speech_like(), reject_threshold=0.0)
        assert result.decision == SpoofDecision.REJECT

    def test_retry_zone(self):
        """Score in the retry band should yield RETRY decision."""
        result = run_antispoof_fusion(
            _speech_like(),
            reject_threshold=0.99,
            retry_threshold=0.0,
        )
        assert result.decision == SpoofDecision.RETRY
