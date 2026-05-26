"""
Audio augmentation pipeline for device and channel robustness.

Simulates real-world degradations a speaker's voice undergoes across devices:
  - bluetooth_compress  : Bluetooth codec artifacts (lowpass + quantization noise)
  - codec_artifact      : MP3/AAC-style lossy compression via torchaudio codec round-trip
  - room_reverb         : Synthetic room impulse response convolution
  - add_noise           : Additive white / babble / pink noise at a target SNR
  - random_augment      : Randomly apply one or more of the above

All functions accept and return (1, samples) float32 tensors at 16kHz.
"""

from __future__ import annotations

import random
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from dataclasses import dataclass, field

SAMPLE_RATE = 16000


# ── individual augmentations ──────────────────────────────────────────────────

def bluetooth_compress(
    waveform: torch.Tensor,
    cutoff_hz: float = 7000.0,
    noise_level: float = 0.002,
) -> torch.Tensor:
    """
    Simulate Bluetooth headset audio: lowpass filter + mild quantization noise.
    Bluetooth SBC/AAC codecs typically roll off above 7kHz.
    """
    # Lowpass filter
    filtered = F.lowpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=cutoff_hz)
    # Quantization noise
    noise = torch.randn_like(filtered) * noise_level
    return (filtered + noise).clamp(-1.0, 1.0)


def codec_artifact(
    waveform: torch.Tensor,
    bitrate: str = "32k",
) -> torch.Tensor:
    """
    Simulate lossy codec compression (MP3-style) using torchaudio's codec round-trip.
    Encodes to MP3 bytes in memory, then decodes back — captures real codec artifacts.
    """
    try:
        # torchaudio encode/decode round-trip through MP3
        encoded = torchaudio.functional.apply_codec(
            waveform, SAMPLE_RATE, format="mp3", compression=int(bitrate.rstrip("k"))
        )
        return encoded
    except Exception:
        # Fallback: simulate with lowpass + mild distortion if codec unavailable
        return bluetooth_compress(waveform, cutoff_hz=8000.0, noise_level=0.001)


def room_reverb(
    waveform: torch.Tensor,
    rt60: float = 0.3,
    room_scale: float = 0.5,
) -> torch.Tensor:
    """
    Simulate room reverberation with a synthetic exponentially-decaying RIR.

    Args:
        rt60: reverberation time in seconds (0.1=small room, 0.8=large hall)
        room_scale: controls RIR energy (0.0–1.0)
    """
    rir_length = int(SAMPLE_RATE * rt60)
    t = torch.arange(rir_length, dtype=torch.float32) / SAMPLE_RATE
    decay_rate = 6.91 / rt60   # -60 dB at rt60

    rir = torch.randn(rir_length) * torch.exp(-decay_rate * t)
    rir = rir * room_scale
    rir[0] = 1.0   # direct path

    # Normalize RIR energy
    rir = rir / rir.abs().max().clamp(min=1e-8)

    # Convolve waveform with RIR using FFT-based convolution
    sig = waveform.squeeze(0)
    reverbed = torch.nn.functional.conv1d(
        sig.unsqueeze(0).unsqueeze(0),
        rir.flip(0).unsqueeze(0).unsqueeze(0),
        padding=rir_length - 1,
    ).squeeze()[:len(sig)]

    # Match original peak to prevent clipping
    peak = reverbed.abs().max().clamp(min=1e-8)
    orig_peak = sig.abs().max().clamp(min=1e-8)
    reverbed = reverbed * (orig_peak / peak)

    return reverbed.unsqueeze(0).clamp(-1.0, 1.0)


def add_noise(
    waveform: torch.Tensor,
    snr_db: float = 15.0,
    noise_type: str = "white",
) -> torch.Tensor:
    """
    Add synthetic noise at a target SNR.

    Args:
        snr_db: signal-to-noise ratio in dB (lower = noisier)
        noise_type: "white" | "pink" | "babble"
    """
    sig_power = waveform.pow(2).mean().clamp(min=1e-12)

    if noise_type == "white":
        noise = torch.randn_like(waveform)
    elif noise_type == "pink":
        # Pink noise: color white noise with 1/f spectrum
        white = torch.fft.rfft(torch.randn_like(waveform))
        freqs = torch.fft.rfftfreq(waveform.shape[-1]).clamp(min=1e-6)
        pink_filter = (1.0 / freqs.sqrt()).unsqueeze(0)
        noise = torch.fft.irfft(white * pink_filter, n=waveform.shape[-1])
    elif noise_type == "babble":
        # Babble: sum of several random-phase sinusoids approximating speech noise
        t = torch.linspace(0, waveform.shape[-1] / SAMPLE_RATE, waveform.shape[-1])
        noise = sum(
            0.1 * torch.sin(2 * torch.pi * f * t + random.uniform(0, 6.28))
            for f in [200, 350, 500, 800, 1200, 2000, 3400]
        )
        noise = noise.unsqueeze(0)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    noise_power = noise.pow(2).mean().clamp(min=1e-12)
    snr_linear = 10 ** (snr_db / 10.0)
    scale = (sig_power / (snr_linear * noise_power)).sqrt()
    noisy = waveform + scale * noise
    return noisy.clamp(-1.0, 1.0)


def phone_mic(waveform: torch.Tensor) -> torch.Tensor:
    """Simulate phone microphone: slight bandpass (300–8000 Hz) + mild noise."""
    filtered = F.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=300.0)
    filtered = F.lowpass_biquad(filtered, SAMPLE_RATE, cutoff_freq=8000.0)
    return add_noise(filtered, snr_db=30.0, noise_type="white")


# ── augmentation config and random pipeline ───────────────────────────────────

@dataclass
class AugmentConfig:
    """Controls which augmentations are applied and their parameters."""
    use_bluetooth:  bool = True
    use_codec:      bool = False     # disabled by default (needs ffmpeg)
    use_reverb:     bool = True
    use_noise:      bool = True
    noise_snr_range: tuple[float, float] = (10.0, 30.0)
    reverb_rt60_range: tuple[float, float] = (0.1, 0.6)
    max_augments:   int = 2          # apply at most N augmentations per sample


def random_augment(
    waveform: torch.Tensor,
    config: AugmentConfig | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Apply a random combination of augmentations from the config.
    Deterministic when seed is provided.
    """
    cfg = config or AugmentConfig()
    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    pool: list[callable] = []
    if cfg.use_bluetooth:
        pool.append(lambda w: bluetooth_compress(w))
    if cfg.use_reverb:
        rt60 = rng.uniform(*cfg.reverb_rt60_range)
        pool.append(lambda w, r=rt60: room_reverb(w, rt60=r))
    if cfg.use_noise:
        snr = rng.uniform(*cfg.noise_snr_range)
        ntype = rng.choice(["white", "pink", "babble"])
        pool.append(lambda w, s=snr, t=ntype: add_noise(w, snr_db=s, noise_type=t))

    chosen = rng.sample(pool, k=min(cfg.max_augments, len(pool)))
    result = waveform
    for fn in chosen:
        result = fn(result)
    return result
