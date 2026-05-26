"""
Tests for Phase 4: channel normalization, augmentation, and mismatch detection.
"""

import torch
import pytest
import numpy as np

from src.preprocessing.channel_norm import (
    extract_mfcc, apply_cmvn, waveform_to_cmvn_mfcc,
    subtract_cohort_mean, estimate_cohort_mean, WCCNTransform,
)
from src.preprocessing.augmentation import (
    bluetooth_compress, room_reverb, add_noise, phone_mic,
    random_augment, AugmentConfig,
)
from src.antispoofing.channel_detector import (
    detect_channel_mismatch, adjusted_spoof_threshold,
    _spectral_envelope, _estimate_snr, _estimate_bandwidth,
)


# ── helpers ──────────────────────────────────────────────────────────────────

SR = 16000

def _sine(freq=220.0, duration=3.0) -> torch.Tensor:
    t = torch.linspace(0, duration, int(SR * duration))
    return (0.5 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)

def _rand_emb(dim=192) -> torch.Tensor:
    import torch.nn.functional as F
    return F.normalize(torch.randn(dim), dim=0)


# ── CMVN tests ────────────────────────────────────────────────────────────────

class TestCMVN:
    def test_mfcc_shape(self):
        mfcc = extract_mfcc(_sine())
        assert mfcc.dim() == 2
        assert mfcc.shape[0] == 40   # n_mfcc

    def test_cmvn_zero_mean(self):
        mfcc = extract_mfcc(_sine())
        normed = apply_cmvn(mfcc, normalize_variance=False)
        # Each cepstral coefficient should have ~zero mean
        means = normed.mean(dim=1)
        assert means.abs().max() < 1e-4, f"CMVN mean not zero: {means.abs().max()}"

    def test_cmvn_unit_variance(self):
        mfcc = extract_mfcc(_sine())
        normed = apply_cmvn(mfcc, normalize_variance=True)
        stds = normed.std(dim=1)
        # std should be ~1.0 for each coefficient
        assert (stds - 1.0).abs().max() < 0.1, f"CMVN variance not 1: {stds}"

    def test_full_pipeline_shape(self):
        result = waveform_to_cmvn_mfcc(_sine())
        assert result.shape[0] == 40


class TestCohortNorm:
    def test_cohort_mean_shape(self):
        embs = [_rand_emb() for _ in range(10)]
        mean = estimate_cohort_mean(embs)
        assert mean.shape == embs[0].shape

    def test_subtract_cohort_mean_normalized(self):
        import torch.nn.functional as F
        emb = _rand_emb()
        mean = _rand_emb() * 0.1
        result = subtract_cohort_mean(emb, mean)
        assert abs(result.norm().item() - 1.0) < 1e-5


class TestWCCN:
    def _make_data(self, n_speakers=5, samples_per_speaker=8, dim=64):
        embeddings, speaker_ids = [], []
        for i in range(n_speakers):
            center = torch.randn(dim)
            for _ in range(samples_per_speaker):
                emb = center + 0.1 * torch.randn(dim)
                import torch.nn.functional as F
                embeddings.append(F.normalize(emb, dim=0))
                speaker_ids.append(f"spk{i}")
        return embeddings, speaker_ids

    def test_fit_and_transform(self):
        embs, ids = self._make_data()
        wccn = WCCNTransform()
        wccn.fit(embs, ids)
        result = wccn.transform(embs[0])
        assert result.shape == embs[0].shape
        assert abs(result.norm().item() - 1.0) < 1e-5

    def test_save_load(self, tmp_path):
        embs, ids = self._make_data()
        wccn = WCCNTransform()
        wccn.fit(embs, ids)
        path = tmp_path / "wccn.pt"
        wccn.save(path)

        wccn2 = WCCNTransform()
        wccn2.load(path)
        r1 = wccn.transform(embs[0])
        r2 = wccn2.transform(embs[0])
        assert torch.allclose(r1, r2, atol=1e-5)

    def test_unfitted_raises(self):
        wccn = WCCNTransform()
        with pytest.raises(RuntimeError):
            wccn.transform(_rand_emb(64))


# ── augmentation tests ────────────────────────────────────────────────────────

class TestAugmentation:
    def test_bluetooth_preserves_shape(self):
        w = _sine()
        out = bluetooth_compress(w)
        assert out.shape == w.shape

    def test_bluetooth_clipped(self):
        w = _sine()
        out = bluetooth_compress(w)
        assert out.abs().max() <= 1.0

    def test_reverb_preserves_shape(self):
        w = _sine()
        out = room_reverb(w, rt60=0.3)
        assert out.shape == w.shape

    def test_reverb_adds_energy_tail(self):
        w = _sine(duration=1.0)
        out = room_reverb(w, rt60=0.5)
        # Reverb should change the signal
        assert not torch.allclose(w, out, atol=1e-3)

    def test_add_noise_white(self):
        w = _sine()
        out = add_noise(w, snr_db=20.0, noise_type="white")
        assert out.shape == w.shape
        assert not torch.allclose(w, out, atol=1e-4)

    def test_add_noise_pink(self):
        w = _sine()
        out = add_noise(w, snr_db=20.0, noise_type="pink")
        assert out.shape == w.shape

    def test_add_noise_babble(self):
        w = _sine()
        out = add_noise(w, snr_db=15.0, noise_type="babble")
        assert out.shape == w.shape

    def test_phone_mic(self):
        w = _sine()
        out = phone_mic(w)
        assert out.shape == w.shape

    def test_random_augment_deterministic(self):
        w = _sine()
        out1 = random_augment(w, seed=42)
        out2 = random_augment(w, seed=42)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_random_augment_different_seeds(self):
        w = _sine()
        out1 = random_augment(w, seed=1)
        out2 = random_augment(w, seed=99)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_snr_is_roughly_correct(self):
        w = _sine()
        noisy = add_noise(w, snr_db=20.0, noise_type="white")
        signal_power = w.pow(2).mean()
        noise_power = (noisy - w).pow(2).mean().clamp(min=1e-12)
        measured_snr = 10 * (signal_power / noise_power).log10().item()
        assert abs(measured_snr - 20.0) < 3.0, f"SNR off: {measured_snr:.1f} dB"


# ── channel mismatch detector tests ──────────────────────────────────────────

class TestChannelMismatchDetector:
    def test_same_signal_low_mismatch(self):
        w = _sine()
        result = detect_channel_mismatch(w, w)
        assert result.mismatch_score < 0.3
        assert not result.is_mismatch

    def test_bluetooth_vs_clean_triggers_mismatch(self):
        clean = _sine(duration=4.0)
        bt = bluetooth_compress(clean)
        result = detect_channel_mismatch(clean, bt)
        # Bluetooth filtering changes spectral envelope — should register some mismatch
        assert result.spectral_distance > 0.0

    def test_noisy_vs_clean_has_higher_score_than_same(self):
        clean = _sine(duration=4.0)
        noisy = add_noise(clean, snr_db=5.0, noise_type="white")
        same_result   = detect_channel_mismatch(clean, clean)
        noisy_result  = detect_channel_mismatch(clean, noisy)
        assert noisy_result.mismatch_score > same_result.mismatch_score

    def test_spoof_threshold_relaxed_on_mismatch(self):
        clean = _sine(duration=4.0)
        bt = bluetooth_compress(clean)
        result = detect_channel_mismatch(clean, bt, threshold=0.0)  # force mismatch
        result.is_mismatch = True
        result.spoof_suspicion_adjustment = 0.15
        adjusted = adjusted_spoof_threshold(0.5, result)
        assert adjusted > 0.5

    def test_no_relaxation_without_mismatch(self):
        w = _sine()
        result = detect_channel_mismatch(w, w)
        adjusted = adjusted_spoof_threshold(0.5, result)
        assert adjusted == 0.5   # no change when no mismatch

    def test_result_scores_bounded(self):
        w = _sine()
        result = detect_channel_mismatch(w, add_noise(w, snr_db=5.0))
        assert 0.0 <= result.mismatch_score <= 1.0

    def test_bandwidth_ratio_near_one_for_same_signal(self):
        w = _sine()
        result = detect_channel_mismatch(w, w)
        assert abs(result.bandwidth_ratio - 1.0) < 0.05
