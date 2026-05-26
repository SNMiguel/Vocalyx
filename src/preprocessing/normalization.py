import torch


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.9) -> torch.Tensor:
    """Scale waveform so its peak amplitude equals target_peak."""
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_peak / peak)


def rms_normalize(waveform: torch.Tensor, target_db: float = -23.0) -> torch.Tensor:
    """Normalize waveform to a target RMS level in dBFS."""
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-8:
        return waveform
    target_rms = 10 ** (target_db / 20.0)
    return waveform * (target_rms / rms)


def preemphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """Apply pre-emphasis filter to boost high frequencies."""
    emphasized = torch.cat([waveform[:, :1], waveform[:, 1:] - coeff * waveform[:, :-1]], dim=1)
    return emphasized
