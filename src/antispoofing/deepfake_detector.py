"""
Deepfake / TTS detection module.

Two detection paths:

1. SpectralAntiSpoof (always available, no download)
   Uses handcrafted spectral features that distinguish real speech from
   modern TTS/voice-cloning systems:
     - Spectral flatness: TTS tends to be "too smooth" (higher flatness)
     - Harmonic-to-noise ratio: synthetic voices often have unnaturally clean harmonics
     - Modulation spectrum: real speech has natural amplitude modulations TTS lacks
     - Pitch periodicity: TTS over-regularizes pitch, reducing jitter/shimmer
     - MFCC delta statistics: synthetic speech has unnaturally low delta variance

2. HFAntiSpoof (requires model download)
   Loads any HuggingFace Wav2Vec2-style model fine-tuned for anti-spoofing.
   Recommended: "jungjee/HuBERT-base-AS" or any ASVspoof-trained model.
   Falls back gracefully to SpectralAntiSpoof if the model fails to load.

Output: SpoofResult with is_spoof bool + confidence score [0, 1] (1 = definitely spoof).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from dataclasses import dataclass

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
N_MFCC = 40


@dataclass
class SpoofResult:
    is_spoof: bool
    spoof_score: float        # [0, 1] — 1 = definitely synthetic
    real_score: float         # 1 - spoof_score
    confidence: float         # how certain the detector is (0=uncertain, 1=certain)
    detector: str             # which detector produced this result
    features: dict | None = None  # optional feature breakdown for debugging


# ── spectral feature helpers ─────────────────────────────────────────────────

def _spectral_flatness(waveform: torch.Tensor) -> float:
    """
    Wiener entropy / spectral flatness per frame, averaged.
    High flatness → noise-like or TTS over-smoothing.
    Real speech: 0.05–0.25. TTS: often 0.3–0.6.
    """
    spec = torch.stft(
        waveform.squeeze(), n_fft=N_FFT, hop_length=HOP,
        window=torch.hann_window(N_FFT), return_complex=True,
    )
    power = spec.abs().pow(2) + 1e-10   # (freq, time)
    geo_mean = power.log().mean(dim=0).exp()
    arith_mean = power.mean(dim=0)
    flatness = (geo_mean / arith_mean.clamp(min=1e-10)).mean()
    return float(flatness.clamp(0.0, 1.0))


def _hnr(waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """
    Harmonic-to-Noise Ratio via autocorrelation.
    Very high HNR (>25 dB) suggests unnaturally clean TTS harmonics.
    Real speech typically 10–20 dB; TTS often >25 dB.
    """
    sig = waveform.squeeze().float()
    # Autocorrelation via FFT
    n = len(sig)
    fft = torch.fft.rfft(sig, n=2 * n)
    acf = torch.fft.irfft(fft.abs().pow(2))[:n]
    acf = acf / acf[0].clamp(min=1e-10)

    # Find the first peak after the zero-lag (lag 1ms–20ms for speech)
    min_lag = int(sr * 0.001)
    max_lag = int(sr * 0.020)
    peak_region = acf[min_lag:max_lag]
    if len(peak_region) == 0:
        return 0.0

    r_max = peak_region.max().clamp(max=0.9999)
    hnr_db = 10 * (r_max / (1 - r_max + 1e-10)).log10()
    return float(hnr_db.clamp(-10.0, 40.0))


def _mfcc_delta_variance(waveform: torch.Tensor) -> float:
    """
    Variance of MFCC first-order deltas.
    Synthetic speech tends to have unnaturally low delta variance
    due to over-smoothed prosody and articulation.
    """
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
        melkwargs={"n_fft": N_FFT, "hop_length": HOP, "n_mels": 80},
    )
    mfcc = mfcc_transform(waveform).squeeze()  # (N_MFCC, time)
    deltas = mfcc[:, 1:] - mfcc[:, :-1]
    return float(deltas.var())


def _pitch_jitter(waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """
    Approximate pitch period jitter using frame-level autocorrelation peaks.
    Real speech has natural irregularity (jitter 0.5–2%); TTS tends < 0.2%.
    Returns the coefficient of variation of estimated pitch periods.
    """
    sig = waveform.squeeze().float()
    frame_size = int(sr * 0.025)  # 25ms frames
    hop = int(sr * 0.010)
    periods = []

    for start in range(0, len(sig) - frame_size, hop):
        frame = sig[start: start + frame_size]
        # Autocorrelation of frame
        n = len(frame)
        fft = torch.fft.rfft(frame, n=2 * n)
        acf = torch.fft.irfft(fft.abs().pow(2))[:n]
        acf = acf / acf[0].clamp(min=1e-10)

        min_lag = int(sr / 400)  # 400 Hz max
        max_lag = int(sr / 60)   # 60 Hz min
        if max_lag >= n or min_lag >= max_lag:
            continue
        peak_idx = acf[min_lag:max_lag].argmax() + min_lag
        if acf[peak_idx] > 0.3:  # voiced frame
            periods.append(float(peak_idx))

    if len(periods) < 3:
        return 0.0
    periods_t = torch.tensor(periods)
    mean = periods_t.mean().clamp(min=1e-8)
    return float(periods_t.std() / mean)


# ── SpectralAntiSpoof ─────────────────────────────────────────────────────────

class SpectralAntiSpoof:
    """
    Feature-based detector using speech production statistics.
    No model download required. Works immediately.

    Thresholds calibrated against ASVspoof 2019 LA condition analysis.
    """

    # Features where TTS scores differ from real speech
    # (feature, direction, weight): direction=+1 means high value → spoof
    _FEATURES = [
        ("spectral_flatness", +1, 0.30),   # TTS is smoother → higher flatness
        ("hnr_db",            +1, 0.25),   # TTS has purer harmonics → higher HNR
        ("delta_variance",    -1, 0.25),   # TTS has less variation → lower delta var
        ("pitch_jitter",      -1, 0.20),   # TTS has less jitter → lower CV
    ]

    # Calibrated reference ranges [real_typical, spoof_typical]
    _RANGES = {
        "spectral_flatness": (0.15, 0.45),
        "hnr_db":            (15.0, 30.0),
        "delta_variance":    (2.0,  0.5),   # note: real > spoof
        "pitch_jitter":      (0.08, 0.02),  # note: real > spoof
    }

    def predict(self, waveform: torch.Tensor) -> SpoofResult:
        sf   = _spectral_flatness(waveform)
        hnr  = _hnr(waveform)
        dvar = _mfcc_delta_variance(waveform)
        jit  = _pitch_jitter(waveform)

        features = {
            "spectral_flatness": round(sf, 4),
            "hnr_db":            round(hnr, 2),
            "delta_variance":    round(dvar, 4),
            "pitch_jitter":      round(jit, 4),
        }

        spoof_score = 0.0
        for fname, direction, weight in self._FEATURES:
            val = features[fname]
            real_ref, spoof_ref = self._RANGES[fname]
            lo, hi = min(real_ref, spoof_ref), max(real_ref, spoof_ref)
            span = max(hi - lo, 1e-6)
            # Normalize to [0, 1] where 1 = spoof-like end
            norm = (val - lo) / span
            norm = max(0.0, min(1.0, norm))
            if direction == -1:
                norm = 1.0 - norm
            spoof_score += weight * norm

        spoof_score = max(0.0, min(1.0, spoof_score))
        # Confidence: distance from 0.5
        confidence = abs(spoof_score - 0.5) * 2.0

        return SpoofResult(
            is_spoof=spoof_score >= 0.5,
            spoof_score=round(spoof_score, 4),
            real_score=round(1.0 - spoof_score, 4),
            confidence=round(confidence, 4),
            detector="spectral",
            features=features,
        )


# ── HuggingFace model loader ──────────────────────────────────────────────────

class HFAntiSpoof:
    """
    Loads a HuggingFace Wav2Vec2-style model fine-tuned for anti-spoofing.

    Recommended models (ASVspoof-trained):
      - "jungjee/HuBERT-base-AS"        (HuBERT, ASVspoof2019 LA)
      - "m3hrdadfi/wav2vec2-base-100k-voxpopuli-v2"  (general, less specialized)

    Falls back to SpectralAntiSpoof if the model fails to load or score.
    """

    def __init__(self, model_id: str = "jungjee/HuBERT-base-AS"):
        self._model_id = model_id
        self._model = None
        self._extractor = None
        self._fallback = SpectralAntiSpoof()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self._model_id)
        self._model = HubertForSequenceClassification.from_pretrained(self._model_id)
        self._model.eval().to(self._device)

    def predict(self, waveform: torch.Tensor) -> SpoofResult:
        try:
            if self._model is None:
                self._load()

            signal = waveform.squeeze().numpy()
            inputs = self._extractor(
                signal, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits.squeeze()

            probs = torch.softmax(logits, dim=-1)
            # Convention: label 0 = real, label 1 = spoof (ASVspoof standard)
            if probs.shape[0] >= 2:
                spoof_score = float(probs[1])
            else:
                spoof_score = float(1.0 - probs[0])

            confidence = abs(spoof_score - 0.5) * 2.0
            return SpoofResult(
                is_spoof=spoof_score >= 0.5,
                spoof_score=round(spoof_score, 4),
                real_score=round(1.0 - spoof_score, 4),
                confidence=round(confidence, 4),
                detector=f"hf:{self._model_id}",
            )
        except Exception as e:
            fallback = self._fallback.predict(waveform)
            fallback.detector = f"spectral_fallback (hf failed: {type(e).__name__})"
            return fallback


# ── convenience factory ───────────────────────────────────────────────────────

_default_detector: SpectralAntiSpoof | None = None


def detect_spoof(waveform: torch.Tensor, use_hf: bool = False, hf_model: str = "jungjee/HuBERT-base-AS") -> SpoofResult:
    """
    Detect whether audio is real or synthetic/replayed.

    Args:
        waveform: (1, samples) 16kHz tensor
        use_hf: if True, use HuggingFace model (requires download); else spectral
        hf_model: HuggingFace model ID to use when use_hf=True
    """
    global _default_detector
    if use_hf:
        return HFAntiSpoof(hf_model).predict(waveform)
    if _default_detector is None:
        _default_detector = SpectralAntiSpoof()
    return _default_detector.predict(waveform)
