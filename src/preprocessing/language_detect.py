"""
Language detection from audio using speechbrain/lang-id-voxlingua107-ecapa.
Identifies 107 languages from a short audio clip.

Usage:
    result = detect_language(waveform)
    print(result.language, result.confidence)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from speechbrain.utils.fetching import LocalStrategy

_lang_model = None
MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"

# Map of SpeechBrain language codes → ISO 639-1 where known
LANG_CODE_MAP = {
    "en": "english", "fr": "french", "es": "spanish",
    "ar": "arabic",  "zh": "mandarin", "de": "german",
    "pt": "portuguese", "ru": "russian", "ja": "japanese",
    "ko": "korean",  "hi": "hindi",  "it": "italian",
}


@dataclass
class LanguageResult:
    language: str        # ISO-639-1 code (e.g. "en")
    language_name: str   # human-readable (e.g. "english")
    confidence: float    # probability [0, 1]
    top5: list[tuple[str, float]]  # [(lang_code, prob), ...]


def _get_model():
    global _lang_model
    if _lang_model is None:
        from speechbrain.inference.classifiers import EncoderClassifier
        _lang_model = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir=f"pretrained_models/{MODEL_ID.split('/')[-1]}",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    return _lang_model


def detect_language(waveform: torch.Tensor) -> LanguageResult:
    """
    Detect the spoken language from a (1, samples) 16kHz waveform.
    Returns a LanguageResult with the top prediction and confidence.
    """
    model = _get_model()
    with torch.no_grad():
        out_prob, score, index, label = model.classify_batch(waveform)

    probs = out_prob.squeeze().softmax(dim=0)
    top5_idx = probs.topk(5).indices.tolist()
    top5_probs = probs.topk(5).values.tolist()

    labels = model.hparams.label_encoder.decode_ndim(
        torch.tensor(top5_idx)
    )

    top5 = [(str(lbl), round(float(prob), 4)) for lbl, prob in zip(labels, top5_probs)]
    best_code = str(label[0]).strip()
    best_prob = float(score[0].exp())  # log-prob → prob

    return LanguageResult(
        language=best_code,
        language_name=LANG_CODE_MAP.get(best_code, best_code),
        confidence=round(best_prob, 4),
        top5=top5,
    )


def is_language(waveform: torch.Tensor, expected: str, min_confidence: float = 0.5) -> bool:
    """Quick check: is this waveform spoken in `expected` language?"""
    result = detect_language(waveform)
    return result.language == expected and result.confidence >= min_confidence
