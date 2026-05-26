"""
Multilingual speaker embedder factory.

Supported backends:
  - ecapa       : speechbrain/spkrec-ecapa-voxceleb  (English-focused baseline)
  - wavlm       : microsoft/wavlm-base-plus-sv       (multilingual, strong SV)
  - xlsr        : facebook/wav2vec2-xls-r-300m       (128-language SSL)
  - mms         : facebook/mms-1b                    (1000+ language coverage)

All backends expose the same interface:
  get_embedding(waveform: Tensor) -> Tensor  (normalized, 1-D)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from enum import Enum


class EmbedderBackend(str, Enum):
    ECAPA = "ecapa"
    WAVLM = "wavlm"
    XLSR  = "xlsr"
    MMS   = "mms"


# ── lazy singletons ──────────────────────────────────────────────────────────

_instances: dict[EmbedderBackend, "_BaseEmbedder"] = {}


def get_embedder(backend: str | EmbedderBackend = EmbedderBackend.WAVLM) -> "_BaseEmbedder":
    """Return (cached) embedder for the requested backend."""
    key = EmbedderBackend(backend)
    if key not in _instances:
        _instances[key] = _build(key)
    return _instances[key]


def get_embedding(
    waveform: torch.Tensor,
    backend: str | EmbedderBackend = EmbedderBackend.WAVLM,
) -> torch.Tensor:
    """Convenience wrapper: compute a normalized speaker embedding."""
    return get_embedder(backend).embed(waveform)


# ── base class ───────────────────────────────────────────────────────────────

class _BaseEmbedder:
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _normalize(t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t.squeeze().float(), dim=0)


# ── ECAPA-TDNN (existing baseline, re-exposed here) ──────────────────────────

class _EcapaEmbedder(_BaseEmbedder):
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self._model.encode_batch(waveform)
        return self._normalize(emb)


# ── WavLM (microsoft/wavlm-base-plus-sv) ─────────────────────────────────────

class _WavLMEmbedder(_BaseEmbedder):
    """
    Uses the WavLM model fine-tuned for speaker verification.
    Extracts the mean-pooled hidden states from the last transformer layer
    as the speaker embedding.
    """
    def __init__(self):
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        model_id = "microsoft/wavlm-base-plus-sv"
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self._model = WavLMModel.from_pretrained(model_id)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        signal = waveform.squeeze().numpy()
        inputs = self._extractor(
            signal, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Mean pool over time dimension
        hidden = outputs.last_hidden_state.mean(dim=1).squeeze()
        return self._normalize(hidden)


# ── XLS-R (facebook/wav2vec2-xls-r-300m) ─────────────────────────────────────

class _XLSREmbedder(_BaseEmbedder):
    """
    128-language SSL model. We mean-pool the last hidden layer to get a
    speaker-discriminative embedding. Not fine-tuned for SV but captures
    cross-lingual acoustic features well.
    """
    def __init__(self):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        model_id = "facebook/wav2vec2-xls-r-300m"
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self._model = Wav2Vec2Model.from_pretrained(model_id)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        signal = waveform.squeeze().numpy()
        inputs = self._extractor(
            signal, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state.mean(dim=1).squeeze()
        return self._normalize(hidden)


# ── MMS (facebook/mms-1b) ────────────────────────────────────────────────────

class _MMSEmbedder(_BaseEmbedder):
    """
    Massively Multilingual Speech (1000+ languages).
    Backbone: wav2vec2-style with adapter layers per language.
    We load the base model without language adapters for language-agnostic embeddings.
    """
    def __init__(self):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        model_id = "facebook/mms-1b"
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self._model = Wav2Vec2Model.from_pretrained(model_id)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        signal = waveform.squeeze().numpy()
        inputs = self._extractor(
            signal, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state.mean(dim=1).squeeze()
        return self._normalize(hidden)


# ── factory ───────────────────────────────────────────────────────────────────

def _build(backend: EmbedderBackend) -> _BaseEmbedder:
    return {
        EmbedderBackend.ECAPA: _EcapaEmbedder,
        EmbedderBackend.WAVLM: _WavLMEmbedder,
        EmbedderBackend.XLSR:  _XLSREmbedder,
        EmbedderBackend.MMS:   _MMSEmbedder,
    }[backend]()
