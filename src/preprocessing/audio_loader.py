import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

TARGET_SAMPLE_RATE = 16000


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio file and return (waveform, sample_rate). Waveform shape: (1, samples)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


def load_and_resample(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Load audio, resample to target_sr, return mono waveform (1, samples)."""
    waveform, sr = load_audio(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform


def load_from_numpy(array: np.ndarray, sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Convert numpy array to resampled torch waveform."""
    waveform = torch.from_numpy(array).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform
