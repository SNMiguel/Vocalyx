"""
Channel mismatch detector — the critical piece from the Phase 4 brief.

The problem it solves:
  Current spoof detectors fire when they see unusual spectral characteristics.
  A legitimate user who enrolled on a studio mic and now authenticates via
  Bluetooth earbuds will look "suspicious" to a naive spoof detector —
  different frequency response, codec artifacts, room acoustics.

  This module analyses the SPECTRAL FINGERPRINT of the audio (not the speaker
  identity) to decide: "is this difference because of a different device, or
  because this is synthetic/cloned audio?"

Key intuition:
  - Channel mismatch:  spectral envelope shifts smoothly and predictably
                       (lowpass roll-off, bandpass clipping, noise floor change)
  - Deepfake / TTS:    spectral artifacts are irregular, often with telltale
                       over-smoothing, unnatural periodicity, or harmonic distortion

Output:
  ChannelMismatchResult with:
    - mismatch_score   : 0=same device  →  1=different device/degraded channel
    - is_mismatch      : bool flag (mismatch_score > threshold)
    - spoof_suspicion_adjustment : float to add to the anti-spoof decision threshold
                                   (positive = relax threshold when mismatch detected)
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T
from dataclasses import dataclass

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP = 160


@dataclass
class ChannelMismatchResult:
    mismatch_score: float          # [0, 1] — higher = more channel difference
    is_mismatch: bool
    threshold: float
    spectral_distance: float       # L1 distance between spectral envelopes
    bandwidth_ratio: float         # probe_bandwidth / enroll_bandwidth
    snr_delta_db: float            # estimated SNR difference between the two
    spoof_suspicion_adjustment: float  # how much to relax spoof threshold


# ── spectral feature extraction ───────────────────────────────────────────────

def _spectral_envelope(waveform: torch.Tensor, n_bins: int = 64) -> torch.Tensor:
    """
    Compute a coarse spectral envelope (mean log-power per frequency bin).
    Returns a 1-D tensor of length n_bins, normalized to unit L2 norm.
    """
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        waveform.squeeze(),
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        return_complex=True,
    )
    power = spec.abs().pow(2)                        # (freq_bins, time)
    mean_power = power.mean(dim=1)                   # (freq_bins,)
    log_power = (mean_power + 1e-8).log()

    # Downsample to n_bins via averaging
    freq_bins = log_power.shape[0]
    bins_per_group = max(1, freq_bins // n_bins)
    trimmed = log_power[:bins_per_group * n_bins]
    envelope = trimmed.reshape(n_bins, bins_per_group).mean(dim=1)

    # Normalize
    envelope = envelope - envelope.mean()
    norm = envelope.norm().clamp(min=1e-8)
    return envelope / norm


def _estimate_snr(waveform: torch.Tensor, noise_percentile: float = 10.0) -> float:
    """
    Rough per-utterance SNR estimate.
    Uses the bottom-percentile frame energy as proxy for noise floor.
    """
    frame_size = HOP
    sig = waveform.squeeze()
    n_frames = len(sig) // frame_size
    if n_frames == 0:
        return 0.0
    frames = sig[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms = frames.pow(2).mean(dim=1).sqrt()
    noise_floor = torch.quantile(rms, noise_percentile / 100.0).clamp(min=1e-10)
    signal_level = rms.max().clamp(min=1e-10)
    return float(20 * (signal_level / noise_floor).log10())


def _estimate_bandwidth(waveform: torch.Tensor, db_threshold: float = -40.0) -> float:
    """
    Estimate the effective bandwidth of the signal in Hz.
    Finds the highest frequency with power above db_threshold relative to peak.
    """
    spec = torch.stft(
        waveform.squeeze(), n_fft=N_FFT, hop_length=HOP, return_complex=True
    )
    mean_power_db = 10 * (spec.abs().pow(2).mean(dim=1) + 1e-12).log10()
    peak_db = mean_power_db.max()
    above_threshold = (mean_power_db > (peak_db + db_threshold)).nonzero()
    if len(above_threshold) == 0:
        return 0.0
    highest_bin = above_threshold[-1].item()
    return float(highest_bin / (N_FFT / 2) * (SAMPLE_RATE / 2))


# ── main detector ─────────────────────────────────────────────────────────────

DEFAULT_MISMATCH_THRESHOLD = 0.35
# When channel mismatch is detected, relax the spoof threshold by this amount
# so the same user on a different device isn't flagged as a deepfake
SPOOF_RELAXATION = 0.15


def detect_channel_mismatch(
    enroll_waveform: torch.Tensor,
    probe_waveform: torch.Tensor,
    threshold: float = DEFAULT_MISMATCH_THRESHOLD,
) -> ChannelMismatchResult:
    """
    Compare spectral fingerprints between enrollment and probe audio.

    Args:
        enroll_waveform: (1, samples) 16kHz waveform used during enrollment
        probe_waveform:  (1, samples) 16kHz waveform being verified
        threshold:       mismatch_score above this = channel mismatch detected

    Returns:
        ChannelMismatchResult with mismatch flag and spoof threshold adjustment
    """
    enroll_env = _spectral_envelope(enroll_waveform)
    probe_env  = _spectral_envelope(probe_waveform)

    # L1 distance between envelopes (0=identical, larger=more different)
    spectral_distance = float((enroll_env - probe_env).abs().mean())

    # Bandwidth comparison
    enroll_bw = _estimate_bandwidth(enroll_waveform)
    probe_bw   = _estimate_bandwidth(probe_waveform)
    bandwidth_ratio = probe_bw / max(enroll_bw, 1.0)

    # SNR comparison
    enroll_snr = _estimate_snr(enroll_waveform)
    probe_snr  = _estimate_snr(probe_waveform)
    snr_delta  = probe_snr - enroll_snr

    # Mismatch score: weighted combination of spectral + bandwidth + SNR cues
    # Normalize spectral_distance to [0, 1] via sigmoid-like scaling
    spectral_component = min(spectral_distance / 0.5, 1.0)

    # Bandwidth ratio far from 1.0 signals device difference (e.g., phone lowpass)
    bw_component = min(abs(1.0 - bandwidth_ratio) * 2.0, 1.0)

    # Large SNR drop suggests noisier environment or worse device
    snr_component = min(abs(snr_delta) / 20.0, 1.0)

    mismatch_score = (
        0.55 * spectral_component +
        0.30 * bw_component +
        0.15 * snr_component
    )

    is_mismatch = mismatch_score > threshold

    # Graduated relaxation: the worse the mismatch, the more we relax the spoof threshold
    adjustment = SPOOF_RELAXATION * mismatch_score if is_mismatch else 0.0

    return ChannelMismatchResult(
        mismatch_score=round(mismatch_score, 4),
        is_mismatch=is_mismatch,
        threshold=threshold,
        spectral_distance=round(spectral_distance, 4),
        bandwidth_ratio=round(bandwidth_ratio, 4),
        snr_delta_db=round(snr_delta, 2),
        spoof_suspicion_adjustment=round(adjustment, 4),
    )


def adjusted_spoof_threshold(
    base_threshold: float,
    mismatch_result: ChannelMismatchResult,
) -> float:
    """
    Return a potentially relaxed spoof-detection threshold.
    When channel mismatch is detected, a higher threshold reduces false spoof alarms.
    """
    return base_threshold + mismatch_result.spoof_suspicion_adjustment
