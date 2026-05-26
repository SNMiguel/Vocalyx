"""
Tests for evaluation metrics and the benchmark runner.
Uses synthetic score distributions — no audio files or model needed.
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_far_frr,
    compute_eer,
    compute_verification_metrics,
    compute_spoof_metrics,
)
from src.evaluation.test_runner import run_synthetic_benchmark


class TestFarFrr:
    def test_perfect_separation(self):
        genuine = [0.9, 0.85, 0.8]
        impostor = [0.1, 0.15, 0.2]
        far, frr = compute_far_frr(genuine, impostor, threshold=0.5)
        assert far == 0.0
        assert frr == 0.0

    def test_all_rejected(self):
        genuine = [0.1, 0.2]
        impostor = [0.1, 0.2]
        far, frr = compute_far_frr(genuine, impostor, threshold=0.9)
        assert far == 0.0
        assert frr == 1.0

    def test_all_accepted(self):
        genuine = [0.8, 0.9]
        impostor = [0.8, 0.9]
        far, frr = compute_far_frr(genuine, impostor, threshold=0.0)
        assert far == 1.0
        assert frr == 0.0


class TestEer:
    def test_eer_perfect_system(self):
        rng = np.random.default_rng(0)
        genuine = rng.normal(0.8, 0.05, 500).clip(-1, 1).tolist()
        impostor = rng.normal(0.1, 0.05, 500).clip(-1, 1).tolist()
        eer, thresh = compute_eer(genuine, impostor)
        assert eer < 0.02, f"EER too high for well-separated scores: {eer:.4f}"

    def test_eer_poor_system(self):
        rng = np.random.default_rng(1)
        # Heavily overlapping distributions
        genuine = rng.normal(0.5, 0.15, 500).clip(-1, 1).tolist()
        impostor = rng.normal(0.5, 0.15, 500).clip(-1, 1).tolist()
        eer, thresh = compute_eer(genuine, impostor)
        assert eer > 0.3, f"EER unexpectedly low for overlapping scores: {eer:.4f}"

    def test_eer_within_range(self):
        rng = np.random.default_rng(2)
        genuine = rng.normal(0.7, 0.1, 200).clip(-1, 1).tolist()
        impostor = rng.normal(0.2, 0.1, 200).clip(-1, 1).tolist()
        eer, thresh = compute_eer(genuine, impostor)
        assert 0.0 <= eer <= 1.0
        assert -1.0 <= thresh <= 1.0


class TestSpoofMetrics:
    def test_perfect_spoof_detector(self):
        real = [0.9, 0.85, 0.8, 0.95]
        spoof = [0.1, 0.05, 0.15, 0.2]
        m = compute_spoof_metrics(real, spoof, threshold=0.5)
        assert m.accuracy == 1.0
        assert m.false_alarm_rate == 0.0
        assert m.miss_rate == 0.0

    def test_terrible_spoof_detector(self):
        real = [0.1, 0.2]    # real speech flagged as spoof
        spoof = [0.9, 0.8]   # spoof accepted as real
        m = compute_spoof_metrics(real, spoof, threshold=0.5)
        assert m.false_alarm_rate == 1.0
        assert m.miss_rate == 1.0
        assert m.accuracy == 0.0


class TestSyntheticBenchmark:
    def test_benchmark_runs(self):
        report = run_synthetic_benchmark(threshold=0.25)
        assert len(report.results) > 0

    def test_clean_condition_better_than_noisy(self):
        report = run_synthetic_benchmark(threshold=0.25)
        results = {f"{r.axis}/{r.label}": r.metrics for r in report.results}
        clean_eer = results["condition/clean"].eer
        noisy_eer = results["condition/noisy"].eer
        assert clean_eer < noisy_eer, (
            f"Expected clean EER ({clean_eer:.3f}) < noisy EER ({noisy_eer:.3f})"
        )

    def test_normal_vocal_better_than_whispered(self):
        report = run_synthetic_benchmark(threshold=0.25)
        results = {f"{r.axis}/{r.label}": r.metrics for r in report.results}
        normal_eer = results["vocal_state/normal"].eer
        whispered_eer = results["vocal_state/whispered"].eer
        assert normal_eer < whispered_eer

    def test_report_serializable(self, tmp_path):
        report = run_synthetic_benchmark()
        save_path = tmp_path / "test_report.json"
        report.save(save_path)
        assert save_path.exists()
        import json
        data = json.loads(save_path.read_text())
        assert "results" in data
        assert len(data["results"]) > 0

    def test_all_metrics_in_range(self):
        report = run_synthetic_benchmark()
        for r in report.results:
            m = r.metrics
            assert 0.0 <= m.far <= 1.0, f"FAR out of range: {m.far}"
            assert 0.0 <= m.frr <= 1.0, f"FRR out of range: {m.frr}"
            assert 0.0 <= m.eer <= 1.0, f"EER out of range: {m.eer}"
