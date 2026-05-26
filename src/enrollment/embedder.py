import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

_classifier = None
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


def _get_model() -> EncoderClassifier:
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir=f"pretrained_models/{MODEL_ID.split('/')[-1]}",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    return _classifier


def get_embedding(waveform: torch.Tensor) -> torch.Tensor:
    """
    Compute a speaker embedding from a (1, samples) 16kHz waveform.
    Returns a normalized 192-d embedding tensor.
    """
    model = _get_model()
    with torch.no_grad():
        embedding = model.encode_batch(waveform)  # shape: (1, 1, 192)
    embedding = embedding.squeeze()  # shape: (192,)
    return torch.nn.functional.normalize(embedding, dim=0)
