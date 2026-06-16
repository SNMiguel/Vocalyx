"""
Isolated ML-component testing for the admin Model Testing dashboard.

Each runner takes an already-preprocessed waveform (1, samples) @ 16 kHz and
runs exactly ONE Vocalyx component — no Whisper challenge, no speaker
verification, no replay/channel/decision layers. This lets an admin benchmark a
single model (e.g. the deepfake detector) against arbitrary audio that the full
authentication flow would otherwise reject before the model ever sees it.

Adding a new model later = add a `_run_<model>` function and register it in
`RUNNERS`. The endpoint layer stays unchanged.
"""

from __future__ import annotations

import json
import time

import torch

from src.antispoofing.deepfake_detector import detect_spoof

TARGET_SR = 16000

# Stable identifiers persisted in the model_tests.model_type column.
MODEL_DEEPFAKE = "deepfake_wav2vec2"


def _duration_ms(waveform: torch.Tensor) -> int:
    return int(round(waveform.shape[-1] / TARGET_SR * 1000))


def run_deepfake_test(waveform: torch.Tensor) -> dict:
    """Run ONLY the Wav2Vec2 deepfake detector on a preprocessed waveform.

    Returns a dict whose keys line up with the model_tests columns the endpoint
    persists: predicted_label, confidence, deepfake_prob, genuine_prob,
    duration_ms, inference_ms, plus raw_output (JSON) for debugging.
    """
    start = time.perf_counter()
    result = detect_spoof(waveform)
    inference_ms = int(round((time.perf_counter() - start) * 1000))

    raw = {
        "is_spoof": bool(result.is_spoof),
        "spoof_score": result.spoof_score,
        "real_score": result.real_score,
        "confidence": result.confidence,
        "detector": result.detector,   # honest record of which detector actually ran
        "features": result.features,
    }

    return {
        "model_type": MODEL_DEEPFAKE,
        "predicted_label": "synthetic" if result.is_spoof else "genuine",
        "confidence": result.confidence,
        "deepfake_prob": result.spoof_score,
        "genuine_prob": result.real_score,
        "duration_ms": _duration_ms(waveform),
        "inference_ms": inference_ms,
        "raw_output": json.dumps(raw),
    }


# Registry of model_type → runner. Future: "replay_*", "speaker_*", "full_pipeline".
RUNNERS = {
    MODEL_DEEPFAKE: run_deepfake_test,
}
