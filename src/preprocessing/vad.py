import torch
import torchaudio

TARGET_SAMPLE_RATE = 16000
_silero_model = None


def _get_silero_model():
    """Lazy-load Silero VAD (avoids slow import at module level)."""
    global _silero_model
    if _silero_model is None:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _silero_model = model
    return _silero_model


def apply_vad_energy(waveform: torch.Tensor, sr: int = TARGET_SAMPLE_RATE,
                     frame_ms: int = 30, threshold_db: float = -40.0) -> torch.Tensor:
    """
    Energy-based VAD. Fast, no model required.
    Removes frames whose energy is below threshold_db.
    Returns concatenated speech-only waveform.
    """
    frame_size = int(sr * frame_ms / 1000)
    signal = waveform.squeeze(0)
    frames = signal.unfold(0, frame_size, frame_size)
    rms_db = 20 * torch.log10(frames.pow(2).mean(dim=1).sqrt().clamp(min=1e-10))
    speech_frames = frames[rms_db > threshold_db]

    if speech_frames.numel() == 0:
        return waveform  # nothing removed — return original to avoid empty tensor

    return speech_frames.reshape(1, -1)


def apply_vad_silero(waveform: torch.Tensor, sr: int = TARGET_SAMPLE_RATE,
                     threshold: float = 0.5) -> torch.Tensor:
    """
    Silero neural VAD — more accurate than energy VAD, requires internet on first run.
    Returns speech-only waveform.
    """
    model = _get_silero_model()
    model.reset_states()
    signal = waveform.squeeze(0)

    chunk_size = 512 if sr == 16000 else 256
    speech_chunks = []

    for i in range(0, len(signal) - chunk_size + 1, chunk_size):
        chunk = signal[i: i + chunk_size]
        prob = model(chunk.unsqueeze(0), sr).item()
        if prob >= threshold:
            speech_chunks.append(chunk)

    if not speech_chunks:
        return waveform

    return torch.cat(speech_chunks).unsqueeze(0)
