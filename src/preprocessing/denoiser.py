"""
Spectral noise suppression using noisereduce.

Uses a non-stationary spectral gating approach: estimates the noise floor
from the full signal and attenuates frequency bins below the threshold.
Works at any sample rate, no model download required.

Gracefully degrades: if noisereduce is not installed, denoise() is a no-op.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger("voice_biometrics.denoiser")

_available = False


def load_model() -> bool:
    """Check that noisereduce is importable. Called at server startup."""
    global _available
    try:
        import noisereduce  # noqa: F401
        _available = True
        logger.info("noisereduce ready — noise suppression active.")
        return True
    except ImportError:
        logger.warning("noisereduce not installed; noise suppression disabled.")
        return False


def denoise(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Apply spectral noise suppression.

    Args:
        waveform: (1, samples) float32 mono tensor.
        sr:       Sample rate of the waveform.

    Returns:
        Denoised waveform with the same shape and sample rate.
        Returns the original waveform unchanged if denoising fails.
    """
    if not _available:
        return waveform

    try:
        import noisereduce as nr

        audio_np = waveform.squeeze(0).numpy().astype(np.float32)

        # stationary=False handles time-varying noise (traffic, voices, etc.)
        # prop_decrease=0.75 is a gentle reduction — avoids over-suppression
        reduced = nr.reduce_noise(
            y=audio_np,
            sr=sr,
            stationary=False,
            prop_decrease=0.75,
        )

        return torch.from_numpy(reduced).unsqueeze(0)

    except Exception as exc:
        logger.warning(f"Denoising failed ({exc}); using original waveform.")
        return waveform


def is_enabled() -> bool:
    return _available
