"""
Phase 8 — Validation runner tests.

Verifies that the end-to-end validation runner:
  - completes without raising
  - all 11 pipeline components pass
  - the synthetic benchmark runs and returns sensible values
  - the report serialises correctly to JSON
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.validate import (
    run_validation,
    ValidationReport,
    BenchmarkSummary,
    ComponentResult,
    check_preprocessing,
    check_channel_norm,
    check_augmentation,
    check_enrollment,
    check_speaker_verification,
    check_multilingual_scoring,
    check_channel_mismatch,
    check_antispoof_fusion,
    check_decision_fusion,
    check_session_manager,
    check_api_layer,
    check_synthetic_benchmark,
    KNOWN_LIMITATIONS,
    NEXT_STEPS,
)


# ── individual component smoke tests ─────────────────────────────────────────

class TestComponentChecks:
    def test_preprocessing(self):
        d = check_preprocessing()
        assert "shape" in d
        assert "peak" in d
        assert d["peak"] <= 1.0

    def test_channel_norm(self):
        d = check_channel_norm()
        assert d["mfcc_shape"][0] == 40

    def test_augmentation(self):
        d = check_augmentation()
        assert len(d["augmentations_verified"]) == 4

    def test_enrollment(self):
        d = check_enrollment()
        assert d["self_similarity"] > 0.99
        assert d["embedding_dim"] > 0

    def test_speaker_verification(self):
        d = check_speaker_verification()
        assert d["same_speaker_score"] > d["diff_speaker_score"]

    def test_multilingual_scoring(self):
        d = check_multilingual_scoring()
        assert d["languages_configured"] >= 3

    def test_channel_mismatch(self):
        d = check_channel_mismatch()
        assert d["same_channel_score"] < d["noisy_channel_score"]

    def test_antispoof_fusion(self):
        d = check_antispoof_fusion()
        assert d["spoof_decision"] in ("accept", "reject", "retry")
        assert 0.0 <= d["fused_score"] <= 1.0

    def test_decision_fusion(self):
        d = check_decision_fusion()
        assert d["decision"] == "accept"
        assert d["sv_score"] > 0.8

    def test_session_manager(self):
        d = check_session_manager()
        assert d["attempts"] == 1
        assert d["session_status"] in ("accepted", "active", "rejected")

    def test_api_layer(self):
        d = check_api_layer()
        assert d["health"] == "ok"
        assert d["version"] == "1.0.0"


# ── benchmark ─────────────────────────────────────────────────────────────────

class TestSyntheticBenchmark:
    def test_benchmark_runs(self):
        bm = check_synthetic_benchmark()
        assert isinstance(bm, BenchmarkSummary)

    def test_benchmark_has_12_conditions(self):
        bm = check_synthetic_benchmark()
        assert bm.n_conditions == 12

    def test_mean_eer_in_valid_range(self):
        bm = check_synthetic_benchmark()
        assert 0.0 <= bm.mean_eer <= 1.0

    def test_best_and_worst_populated(self):
        bm = check_synthetic_benchmark()
        assert "/" in bm.best_condition
        assert "/" in bm.worst_condition


# ── full runner ───────────────────────────────────────────────────────────────

class TestRunValidation:
    def test_run_validation_all_pass(self, tmp_path):
        report_path = tmp_path / "report.json"
        report = run_validation(report_path=str(report_path), verbose=False)
        assert isinstance(report, ValidationReport)
        assert report.passed, f"Failures: {[c for c in report.components if not c.passed]}"

    def test_all_components_pass(self, tmp_path):
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        failures = [c.name for c in report.components if not c.passed]
        assert failures == [], f"Failed components: {failures}"

    def test_component_count(self, tmp_path):
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        assert len(report.components) == 11

    def test_benchmark_present(self, tmp_path):
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        assert report.benchmark is not None
        assert report.benchmark.n_conditions == 12

    def test_report_serialises_to_json(self, tmp_path):
        report_path = tmp_path / "report.json"
        run_validation(report_path=str(report_path), verbose=False)
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "components" in data
        assert "benchmark" in data
        assert "known_limitations" in data
        assert "next_steps" in data
        assert len(data["components"]) == 11

    def test_known_limitations_populated(self, tmp_path):
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        assert len(report.known_limitations) >= 5

    def test_next_steps_populated(self, tmp_path):
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        assert len(report.next_steps) >= 5

    def test_duration_positive(self, tmp_path):
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        assert report.total_duration_seconds > 0

    def test_timestamp_format(self, tmp_path):
        import re
        report = run_validation(report_path=str(tmp_path / "r.json"), verbose=False)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", report.timestamp)


# ── metadata ──────────────────────────────────────────────────────────────────

class TestProjectMetadata:
    def test_known_limitations_not_empty(self):
        assert len(KNOWN_LIMITATIONS) >= 5

    def test_next_steps_not_empty(self):
        assert len(NEXT_STEPS) >= 5

    def test_limitations_are_strings(self):
        assert all(isinstance(s, str) for s in KNOWN_LIMITATIONS)

    def test_next_steps_are_strings(self):
        assert all(isinstance(s, str) for s in NEXT_STEPS)
