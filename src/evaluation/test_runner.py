"""
Automated benchmark runner for the voice biometrics pipeline.

Test matrix axes (from project brief):
  - language:   english, french, spanish, arabic, mandarin
  - device:     phone_mic, airpods, android_earbuds, wired_headset
  - condition:  clean, noisy, indoor, outdoor
  - vocal_state: normal, sick, emotional, whispered, tired

Each axis can be run independently or combined.
Audio files must follow the naming convention:
  data/test_matrix/<axis>/<label>/<speaker_id>_<trial_idx>.wav

For genuine trials:  same speaker_id, different trial_idx
For impostor trials: different speaker_id
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import torch

from src.evaluation.metrics import (
    VerificationMetrics,
    SpoofMetrics,
    compute_verification_metrics,
    format_report,
)

AXES = {
    "language":    ["english", "french", "spanish", "arabic", "mandarin"],
    "device":      ["phone_mic", "airpods", "android_earbuds", "wired_headset"],
    "condition":   ["clean", "noisy", "indoor", "outdoor"],
    "vocal_state": ["normal", "sick", "emotional", "whispered", "tired"],
}


@dataclass
class Trial:
    audio_path: str
    speaker_id: str
    label: str        # axis label (e.g. "english", "noisy")
    axis: str         # axis name (e.g. "language", "condition")
    is_genuine: bool  # True = same-speaker trial, False = impostor


@dataclass
class ConditionResult:
    axis: str
    label: str
    metrics: VerificationMetrics
    spoof_metrics: SpoofMetrics | None = None
    duration_seconds: float = 0.0


@dataclass
class BenchmarkReport:
    timestamp: str
    threshold: float
    results: list[ConditionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "threshold": self.threshold,
            "results": [
                {
                    "axis": r.axis,
                    "label": r.label,
                    "duration_seconds": r.duration_seconds,
                    "metrics": asdict(r.metrics),
                    "spoof_metrics": asdict(r.spoof_metrics) if r.spoof_metrics else None,
                }
                for r in self.results
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Report saved to {path}")

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"BENCHMARK REPORT  [{self.timestamp}]")
        print(f"{'='*60}")
        for r in self.results:
            print(format_report(
                condition=f"{r.axis}/{r.label}",
                metrics=r.metrics,
                spoof=r.spoof_metrics,
            ))
            print()


def _collect_trials_from_dir(axis_dir: Path, axis: str) -> list[Trial]:
    """
    Scan axis_dir/<label>/ subdirectories for .wav files.
    Files named <speaker_id>_<n>.wav are paired into genuine/impostor trials.
    """
    trials = []
    if not axis_dir.exists():
        return trials

    for label_dir in axis_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        wav_files = sorted(label_dir.glob("*.wav"))

        by_speaker: dict[str, list[Path]] = {}
        for wav in wav_files:
            speaker_id = wav.stem.rsplit("_", 1)[0]
            by_speaker.setdefault(speaker_id, []).append(wav)

        speakers = list(by_speaker.keys())
        for speaker_id, files in by_speaker.items():
            # Genuine: pairs within same speaker
            for i in range(len(files) - 1):
                trials.append(Trial(
                    audio_path=str(files[i]),
                    speaker_id=speaker_id,
                    label=label,
                    axis=axis,
                    is_genuine=True,
                ))
            # Impostor: cross-speaker pairs
            for other_id in speakers:
                if other_id != speaker_id and by_speaker[other_id]:
                    trials.append(Trial(
                        audio_path=str(files[0]),
                        speaker_id=other_id,   # enrolled as wrong speaker
                        label=label,
                        axis=axis,
                        is_genuine=False,
                    ))

    return trials


def run_benchmark(
    score_fn: Callable[[str, str], float],
    test_matrix_dir: str | Path = "data/test_matrix",
    threshold: float = 0.25,
    axes: list[str] | None = None,
    report_path: str | Path = "data/benchmark_report.json",
) -> BenchmarkReport:
    """
    Run the full benchmark across all axes and labels.

    Args:
        score_fn: function(enrolled_audio_path, probe_audio_path) -> cosine similarity score
        test_matrix_dir: root directory containing axis subdirectories
        threshold: acceptance threshold for FAR/FRR calculation
        axes: subset of axes to run (default: all)
        report_path: where to save the JSON report

    Returns:
        BenchmarkReport with per-condition results
    """
    test_matrix_dir = Path(test_matrix_dir)
    run_axes = axes or list(AXES.keys())
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        threshold=threshold,
    )

    for axis in run_axes:
        axis_dir = test_matrix_dir / axis
        trials = _collect_trials_from_dir(axis_dir, axis)

        if not trials:
            print(f"[{axis}] No audio files found in {axis_dir} — skipping.")
            continue

        # Group by label
        by_label: dict[str, list[Trial]] = {}
        for t in trials:
            by_label.setdefault(t.label, []).append(t)

        for label, label_trials in by_label.items():
            t0 = time.time()
            genuine_scores = []
            impostor_scores = []

            for trial in label_trials:
                try:
                    score = score_fn(trial.speaker_id, trial.audio_path)
                    if trial.is_genuine:
                        genuine_scores.append(score)
                    else:
                        impostor_scores.append(score)
                except Exception as e:
                    print(f"  [WARN] Trial failed ({trial.audio_path}): {e}")

            if not genuine_scores or not impostor_scores:
                print(f"[{axis}/{label}] Not enough trials — skipping.")
                continue

            metrics = compute_verification_metrics(genuine_scores, impostor_scores, threshold)
            result = ConditionResult(
                axis=axis,
                label=label,
                metrics=metrics,
                duration_seconds=round(time.time() - t0, 2),
            )
            report.results.append(result)
            print(format_report(f"{axis}/{label}", metrics))

    report.save(report_path)
    return report


def run_synthetic_benchmark(threshold: float = 0.25) -> BenchmarkReport:
    """
    Run a benchmark using synthetic score distributions (no audio files needed).
    Useful for validating the metrics pipeline before real data is available.
    """
    import numpy as np

    # Simulated conditions: (axis, label, genuine_mean, impostor_mean, std)
    # Higher std = more overlap = higher EER, simulating degraded conditions
    conditions = [
        ("language",    "english",         0.75, 0.15, 0.07),
        ("language",    "french",          0.68, 0.18, 0.09),
        ("language",    "arabic",          0.60, 0.22, 0.11),
        ("device",      "phone_mic",       0.78, 0.12, 0.07),
        ("device",      "airpods",         0.70, 0.18, 0.09),
        ("device",      "android_earbuds", 0.62, 0.25, 0.12),
        ("condition",   "clean",           0.80, 0.10, 0.07),
        ("condition",   "noisy",           0.55, 0.35, 0.14),
        ("condition",   "outdoor",         0.50, 0.38, 0.15),
        ("vocal_state", "normal",          0.78, 0.12, 0.07),
        ("vocal_state", "sick",            0.52, 0.32, 0.14),
        ("vocal_state", "whispered",       0.45, 0.38, 0.16),
    ]

    rng = np.random.default_rng(42)
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        threshold=threshold,
    )

    for axis, label, g_mean, i_mean, std in conditions:
        genuine_scores = rng.normal(g_mean, std, 100).clip(-1, 1).tolist()
        impostor_scores = rng.normal(i_mean, std, 200).clip(-1, 1).tolist()
        metrics = compute_verification_metrics(genuine_scores, impostor_scores, threshold)
        report.results.append(ConditionResult(axis=axis, label=label, metrics=metrics))

    return report
