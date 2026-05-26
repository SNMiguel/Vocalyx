"""
Replay attack detector.

A replay attack plays back a recorded sample of the legitimate user's voice
through a speaker. The recording-playback chain leaves distinctive artifacts:

  1. Double-room effect: the recording already has room acoustics; replaying
     it adds a second layer of reverberation, creating an unusually long and
     complex impulse response.

  2. Compressed bandwidth: most consumer replay devices (phones, laptops) have
     a highpass rolloff below ~200 Hz and lowpass rolloff above ~15kHz.
     Combined with the recording mic chain, real room ambiance in sub-100Hz
     and ultrasonic bands is attenuated.

  3. Spectral floor elevation: replayed audio has a raised noise floor in silent
     regions due to speaker hiss and room noise added on top of the original
     recording's noise floor.

  4. Narrowband periodicity: device and speaker resonances create narrow
     spectral peaks at frequencies related to the loudspeaker's own resonance
     (typically 100–3000 Hz), detectable via spectral flux.

  5. Unnatural inter-frame SNR: the gap between voiced and unvoiced segments
     shrinks in replayed audio — silence isn't really silent anymore.

Detection strategy: score each of these cues and combine into a replay score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160


@dataclass
class ReplayResult:
    is_replay: bool
    replay_score: float       # [0, 1] — 1 = almost certainly replay
    confidence: float
    features: dict


# ── individual feature detectors ─────────────────────────────────────────────

def _reverb_tail_energy(waveform: torch.Tensor) -> float:
    """
    Ratio of energy in the decaying tail of the signal vs. the onset.
    Double-room replay has longer reverb tails → higher ratio.
    """
    sig = waveform.squeeze().float()
    n = len(sig)
    onset = sig[:n // 4].pow(2).mean().clamp(min=1e-12)
    tail  = sig[3 * n // 4:].pow(2).mean().clamp(min=1e-12)
    ratio = float((tail / onset).clamp(max=10.0))
    # Normalize: ratio < 0.05 → live speech, > 0.3 → heavy reverb
    return min(ratio / 0.3, 1.0)


def _sub_band_energy_ratio(waveform: torch.Tensor) -> float:
    """
    Ratio of sub-100Hz energy to mid-band (500–4000Hz) energy.
    Replay devices roll off low frequencies → lower ratio signals replay.
    """
    spec = torch.stft(
        waveform.squeeze(), n_fft=N_FFT, hop_length=HOP,
        window=torch.hann_window(N_FFT), return_complex=True,
    )
    power = spec.abs().pow(2)   # (freq_bins, time)
    freq_res = SAMPLE_RATE / N_FFT

    sub100_end  = int(100 / freq_res)
    mid_start   = int(500 / freq_res)
    mid_end     = int(4000 / freq_res)

    sub_energy = power[:sub100_end].mean().clamp(min=1e-12)
    mid_energy = power[mid_start:mid_end].mean().clamp(min=1e-12)
    ratio = float((sub_energy / mid_energy).clamp(max=1.0))

    # Low ratio (< 0.02) → low-frequency attenuation → likely replay
    # Map: ratio 0.02 → score 1.0 (very suspicious), ratio 0.10 → score 0.0
    score = 1.0 - min(ratio / 0.10, 1.0)
    return score


def _silence_noise_floor(waveform: torch.Tensor) -> float:
    """
    Noise floor during nominally silent regions.
    Replayed audio has elevated noise floor: device hiss + double room noise.
    Returns normalized score: 0 = clean silence, 1 = elevated floor (replay-like).
    """
    sig = waveform.squeeze().float()
    frame_size = HOP
    n_frames = len(sig) // frame_size
    if n_frames < 4:
        return 0.0

    frames = sig[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms = frames.pow(2).mean(dim=1).sqrt()

    # Bottom 20% of frames (quietest = likely silence regions)
    quiet_rms = torch.quantile(rms, 0.20).item()
    # Top 80% (active speech)
    active_rms = torch.quantile(rms, 0.80).clamp(min=1e-10).item()

    floor_ratio = quiet_rms / active_rms
    # Real speech: floor_ratio ≈ 0.01–0.05 (60–40 dB SNR in silent regions)
    # Replay: floor_ratio ≈ 0.05–0.20 (26–14 dB) — noisier silence
    score = min(floor_ratio / 0.15, 1.0)
    return float(score)


def _spectral_flux_peaks(waveform: torch.Tensor) -> float:
    """
    Detects narrow spectral flux peaks characteristic of loudspeaker resonances.
    Returns normalized score: higher = more narrow-band peaks (replay-like).
    """
    spec = torch.stft(
        waveform.squeeze(), n_fft=N_FFT, hop_length=HOP,
        window=torch.hann_window(N_FFT), return_complex=True,
    )
    power = spec.abs().pow(2)   # (freq_bins, time)
    mean_spectrum = power.mean(dim=1)

    # Spectral smoothness: compare each bin to its neighbors
    kernel = torch.ones(1, 1, 5) / 5.0
    smoothed = F.conv1d(
        mean_spectrum.unsqueeze(0).unsqueeze(0),
        kernel, padding=2
    ).squeeze()

    # Relative deviation from smoothed envelope
    deviation = ((mean_spectrum - smoothed).abs() / (smoothed.clamp(min=1e-10))).mean()
    # High deviation → narrow peaks → possible loudspeaker resonance
    score = float(deviation.clamp(max=1.0))
    return score


# ── main detector ─────────────────────────────────────────────────────────────

DEFAULT_REPLAY_THRESHOLD = 0.40

_WEIGHTS = {
    "reverb_tail":     0.30,
    "sub_band_ratio":  0.25,
    "noise_floor":     0.30,
    "spectral_flux":   0.15,
}


def detect_replay(
    waveform: torch.Tensor,
    threshold: float = DEFAULT_REPLAY_THRESHOLD,
) -> ReplayResult:
    """
    Detect whether audio is a replay attack.

    Args:
        waveform: (1, samples) 16kHz tensor
        threshold: replay_score above this → flag as replay

    Returns:
        ReplayResult with is_replay, replay_score, and per-feature breakdown
    """
    features = {
        "reverb_tail":     round(_reverb_tail_energy(waveform), 4),
        "sub_band_ratio":  round(_sub_band_energy_ratio(waveform), 4),
        "noise_floor":     round(_silence_noise_floor(waveform), 4),
        "spectral_flux":   round(_spectral_flux_peaks(waveform), 4),
    }

    replay_score = sum(
        _WEIGHTS[k] * v for k, v in features.items()
    )
    replay_score = max(0.0, min(1.0, replay_score))
    confidence = abs(replay_score - 0.5) * 2.0

    return ReplayResult(
        is_replay=replay_score >= threshold,
        replay_score=round(replay_score, 4),
        confidence=round(confidence, 4),
        features=features,
    )
