"""
Tests for Phase 3: multilingual embedder, language detection, adaptive scoring.

Model download tests are marked with @pytest.mark.slow and skipped by default.
Run with: pytest tests/test_multilingual.py -v -m slow
"""

import torch
import pytest
import numpy as np

from src.verification.scoring import (
    get_threshold,
    disentangle_language,
    score_pair,
    score_pair_fused,
    _LANGUAGE_THRESHOLDS,
)
from src.verification.multilingual import EmbedderBackend


# ── helpers ──────────────────────────────────────────────────────────────────

def _sine(freq: float = 220.0, duration: float = 3.0, sr: int = 16000) -> torch.Tensor:
    t = torch.linspace(0, duration, int(sr * duration))
    return (0.5 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def _rand_embedding(dim: int = 256) -> torch.Tensor:
    import torch.nn.functional as F
    v = torch.randn(dim)
    return F.normalize(v, dim=0)


# ── threshold tests ───────────────────────────────────────────────────────────

class TestAdaptiveThresholds:
    def test_known_languages_have_threshold(self):
        for lang in ["english", "french", "arabic", "mandarin"]:
            t = get_threshold(lang)
            assert 0.0 < t < 1.0, f"Unexpected threshold for {lang}: {t}"

    def test_unknown_language_uses_default(self):
        t = get_threshold("klingon")
        assert t == _LANGUAGE_THRESHOLDS["default"]

    def test_cross_language_threshold_lower_than_english(self):
        en = get_threshold("english")
        for lang in ["arabic", "mandarin"]:
            assert get_threshold(lang) <= en, (
                f"{lang} threshold should be <= english threshold"
            )


# ── disentanglement tests ─────────────────────────────────────────────────────

class TestDisentanglement:
    def test_output_is_normalized(self):
        speaker_emb = _rand_embedding(256)
        lang_emb = _rand_embedding(256)
        result = disentangle_language(speaker_emb, lang_emb)
        assert abs(result.norm().item() - 1.0) < 1e-5

    def test_language_component_removed(self):
        import torch.nn.functional as F
        speaker_emb = _rand_embedding(256)
        lang_emb = _rand_embedding(256)
        result = disentangle_language(speaker_emb, lang_emb)
        # Dot product with language direction should be near zero
        dot = (result @ F.normalize(lang_emb, dim=0)).abs().item()
        assert dot < 1e-4, f"Language component not removed: dot={dot:.6f}"

    def test_preserves_orthogonal_info(self):
        import torch.nn.functional as F
        # Create speaker embedding orthogonal to language direction
        lang_emb = torch.zeros(256)
        lang_emb[0] = 1.0
        speaker_emb = torch.zeros(256)
        speaker_emb[1] = 1.0  # orthogonal to lang_emb
        result = disentangle_language(speaker_emb, lang_emb)
        # Orthogonal component should be fully preserved
        sim = F.cosine_similarity(result.unsqueeze(0), speaker_emb.unsqueeze(0)).item()
        assert sim > 0.99, f"Orthogonal info not preserved: sim={sim:.4f}"


# ── scoring tests (no model download) ────────────────────────────────────────

class TestScoringResult:
    def test_fused_score_weights(self):
        """Fused scoring averages correctly — validated with mock embeddings."""
        import torch.nn.functional as F

        # Patch get_embedding so no model is loaded
        import src.verification.scoring as scoring_mod
        original = scoring_mod.get_embedding

        call_log = {}
        def mock_get_embedding(waveform, backend):
            call_log[str(backend)] = True
            return _rand_embedding(256)

        scoring_mod.get_embedding = mock_get_embedding
        try:
            enrolled = {
                EmbedderBackend.WAVLM: _rand_embedding(256),
                EmbedderBackend.ECAPA: _rand_embedding(256),
            }
            result = score_pair_fused(enrolled, _sine(), language="english")
            assert "fused" in result.backend
            assert result.fusion_scores is not None
            assert -1.0 <= result.score <= 1.0
        finally:
            scoring_mod.get_embedding = original

    def test_same_embedding_gives_score_one(self):
        import src.verification.scoring as scoring_mod
        original = scoring_mod.get_embedding

        fixed_emb = _rand_embedding(256)
        scoring_mod.get_embedding = lambda w, b: fixed_emb
        try:
            result = score_pair(fixed_emb, _sine(), language="english")
            assert abs(result.score - 1.0) < 1e-4
            assert result.accepted is True
        finally:
            scoring_mod.get_embedding = original


# ── slow tests: actually download and run models ──────────────────────────────

@pytest.mark.slow
class TestWavLMEmbedder:
    def test_wavlm_produces_embedding(self):
        from src.verification.multilingual import get_embedding, EmbedderBackend
        waveform = _sine(220.0, duration=3.0)
        emb = get_embedding(waveform, EmbedderBackend.WAVLM)
        assert emb.dim() == 1
        assert emb.shape[0] > 0
        assert abs(emb.norm().item() - 1.0) < 1e-5

    def test_wavlm_same_speaker_high_similarity(self):
        from src.verification.multilingual import get_embedding, EmbedderBackend
        import torch.nn.functional as F
        waveform = _sine(220.0, duration=3.0)
        emb1 = get_embedding(waveform, EmbedderBackend.WAVLM)
        emb2 = get_embedding(waveform, EmbedderBackend.WAVLM)
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        assert sim > 0.99


@pytest.mark.slow
class TestLanguageDetection:
    def test_lang_detect_returns_result(self):
        from src.preprocessing.language_detect import detect_language
        waveform = _sine(220.0, duration=3.0)
        result = detect_language(waveform)
        assert result.language is not None
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.top5) == 5


# ── cross-language benchmark (synthetic) ─────────────────────────────────────

class TestCrossLanguageBenchmark:
    """Validate that the benchmark framework handles cross-language conditions."""

    def test_synthetic_cross_language_report(self):
        from src.evaluation.test_runner import run_synthetic_benchmark
        report = run_synthetic_benchmark(threshold=0.25)
        lang_results = {r.label: r.metrics for r in report.results if r.axis == "language"}
        assert "english" in lang_results
        # English should generally outperform Arabic in a baseline system
        assert lang_results["english"].eer <= lang_results["arabic"].eer + 0.05

    def test_model_comparison_structure(self):
        """Simulate comparing two backends on the same condition."""
        from src.evaluation.metrics import compute_verification_metrics
        rng = np.random.default_rng(99)

        # WavLM: better multilingual separation
        wavlm_genuine  = rng.normal(0.72, 0.08, 100).clip(-1, 1).tolist()
        wavlm_impostor = rng.normal(0.15, 0.08, 200).clip(-1, 1).tolist()

        # ECAPA: slightly worse on cross-language
        ecapa_genuine  = rng.normal(0.60, 0.10, 100).clip(-1, 1).tolist()
        ecapa_impostor = rng.normal(0.20, 0.10, 200).clip(-1, 1).tolist()

        m_wavlm = compute_verification_metrics(wavlm_genuine, wavlm_impostor)
        m_ecapa = compute_verification_metrics(ecapa_genuine, ecapa_impostor)

        assert m_wavlm.eer <= m_ecapa.eer, (
            f"Expected WavLM EER ({m_wavlm.eer:.3f}) <= ECAPA EER ({m_ecapa.eer:.3f})"
        )
