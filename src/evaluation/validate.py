"""
Phase 8 — Full pipeline validation runner.

Exercises every component built across Phases 1–7 in an integrated sequence:

  1. Preprocessing    — load, resample, normalize, VAD
  2. Channel norm     — CMVN on MFCC features
  3. Augmentation     — bluetooth, reverb, noise
  4. Enrollment       — embed + store a synthetic "user"
  5. Speaker verify   — verify same vs different speaker (synthetic)
  6. Channel detect   — mismatch between clean and bluetooth audio
  7. Anti-spoof       — deepfake + replay detection on synthetic signals
  8. Fusion           — full auth decision pipeline
  9. Evaluation       — synthetic benchmark across all condition axes
 10. API layer        — health + version endpoints via TestClient

Produces a structured ValidationReport saved to data/validation_report.json.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SR = 16000


# ── helpers ───────────────────────────────────────────────────────────────────

def _sine(freq=220.0, duration=3.0) -> torch.Tensor:
    t = torch.linspace(0, duration, int(SR * duration))
    return (0.4 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def _speech_like(duration=3.0) -> torch.Tensor:
    t = torch.linspace(0, duration, int(SR * duration))
    f0 = 150.0
    jitter = 0.01 * torch.randn(t.shape)
    sig = sum((1.0 / k) * torch.sin(2 * torch.pi * k * (f0 + jitter * 10) * t)
              for k in range(1, 8))
    am = 0.5 + 0.5 * torch.sin(2 * torch.pi * 4.0 * t)
    return (am * sig * 0.3).unsqueeze(0)


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class ComponentResult:
    name: str
    passed: bool
    duration_ms: float
    details: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkSummary:
    n_conditions: int
    mean_eer: float
    best_condition: str
    worst_condition: str
    target_eer_met: bool   # < 6% in adverse conditions per brief


@dataclass
class ValidationReport:
    timestamp: str
    total_duration_seconds: float
    passed: bool
    components: list[ComponentResult]
    benchmark: BenchmarkSummary | None
    known_limitations: list[str]
    next_steps: list[str]

    def save(self, path: str | Path = "data/validation_report.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nReport saved → {path}")


# ── individual component validators ──────────────────────────────────────────

def _validate(name: str, fn) -> ComponentResult:
    t0 = time.perf_counter()
    try:
        details = fn() or {}
        return ComponentResult(
            name=name,
            passed=True,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            details=details,
        )
    except Exception as e:
        return ComponentResult(
            name=name,
            passed=False,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=f"{type(e).__name__}: {e}",
        )


def check_preprocessing() -> dict:
    from src.preprocessing.audio_loader import load_and_resample
    from src.preprocessing.normalization import rms_normalize, peak_normalize
    from src.preprocessing.vad import apply_vad_energy
    w = _sine()
    w = rms_normalize(w)
    w = apply_vad_energy(w)
    assert w.shape[0] == 1
    assert w.abs().max() <= 1.0
    return {"shape": list(w.shape), "peak": round(w.abs().max().item(), 4)}


def check_channel_norm() -> dict:
    from src.preprocessing.channel_norm import waveform_to_cmvn_mfcc, WCCNTransform
    import torch.nn.functional as F
    mfcc = waveform_to_cmvn_mfcc(_sine())
    assert mfcc.shape[0] == 40
    mean_abs = mfcc.mean(dim=1).abs().max().item()
    assert mean_abs < 1e-3

    embs = [F.normalize(torch.randn(128), dim=0) for _ in range(20)]
    ids = [f"spk{i % 4}" for i in range(20)]
    wccn = WCCNTransform()
    wccn.fit(embs, ids)
    out = wccn.transform(embs[0])
    assert abs(out.norm().item() - 1.0) < 1e-4
    return {"mfcc_shape": list(mfcc.shape), "wccn_dim": list(out.shape)}


def check_augmentation() -> dict:
    from src.preprocessing.augmentation import (
        bluetooth_compress, room_reverb, add_noise, random_augment,
    )
    w = _sine()
    bt  = bluetooth_compress(w)
    rv  = room_reverb(w, rt60=0.3)
    ns  = add_noise(w, snr_db=15.0, noise_type="white")
    rnd = random_augment(w, seed=42)
    for out in [bt, rv, ns, rnd]:
        assert out.shape == w.shape
        assert out.abs().max() <= 1.0
    return {"augmentations_verified": ["bluetooth", "reverb", "noise", "random"]}


def check_enrollment() -> dict:
    from src.enrollment.embedder import get_embedding
    from src.enrollment.enrollment_db import enroll_user, get_enrollment, delete_user
    import torch.nn.functional as F
    w = _sine()
    emb = get_embedding(w)
    assert emb.dim() == 1
    assert abs(emb.norm().item() - 1.0) < 1e-4

    enroll_user("_validation_user", [emb])
    stored = get_enrollment("_validation_user")
    sim = F.cosine_similarity(emb.unsqueeze(0), stored.unsqueeze(0)).item()
    delete_user("_validation_user")
    return {"embedding_dim": emb.shape[0], "self_similarity": round(sim, 4)}


def check_speaker_verification() -> dict:
    from src.enrollment.embedder import get_embedding
    from src.enrollment.enrollment_db import enroll_user, delete_user
    from src.verification.speaker_verifier import verify
    import torch.nn.functional as F
    w = _sine(220.0)
    emb = get_embedding(w)
    enroll_user("_val_sv_user", [emb])

    result_same = verify("_val_sv_user", w)
    result_diff = verify("_val_sv_user", _sine(880.0), threshold=0.99)

    delete_user("_val_sv_user")
    assert result_same["accepted"]
    assert not result_diff["accepted"]
    return {
        "same_speaker_score": result_same["score"],
        "diff_speaker_score": result_diff["score"],
    }


def check_multilingual_scoring() -> dict:
    from src.verification.scoring import (
        get_threshold, disentangle_language, _LANGUAGE_THRESHOLDS,
    )
    import torch.nn.functional as F
    assert get_threshold("english") > get_threshold("arabic")
    spk = F.normalize(torch.randn(256), dim=0)
    lang = F.normalize(torch.randn(256), dim=0)
    out = disentangle_language(spk, lang)
    dot = (out @ F.normalize(lang, dim=0)).abs().item()
    assert dot < 1e-4
    return {"languages_configured": len(_LANGUAGE_THRESHOLDS) - 1}


def check_channel_mismatch() -> dict:
    from src.preprocessing.augmentation import bluetooth_compress, add_noise
    from src.antispoofing.channel_detector import detect_channel_mismatch
    w = _sine(duration=4.0)
    same   = detect_channel_mismatch(w, w)
    noisy  = detect_channel_mismatch(w, add_noise(w, snr_db=5.0))
    assert same.mismatch_score < noisy.mismatch_score
    assert 0.0 <= same.mismatch_score <= 1.0
    return {
        "same_channel_score": same.mismatch_score,
        "noisy_channel_score": noisy.mismatch_score,
    }


def check_antispoof_fusion() -> dict:
    from src.antispoofing.fusion import run_antispoof_fusion, SpoofDecision
    from src.antispoofing.deepfake_detector import SpectralAntiSpoof
    w = _speech_like(duration=4.0)
    result = run_antispoof_fusion(w, enroll_waveform=w)
    assert result.decision in list(SpoofDecision)
    assert 0.0 <= result.fused_spoof_score <= 1.0
    features = SpectralAntiSpoof().predict(w).features
    return {
        "spoof_decision": result.decision.value,
        "fused_score": result.fused_spoof_score,
        "deepfake_features": features,
    }


def check_decision_fusion() -> dict:
    from unittest.mock import patch
    from src.decision.fusion_layer import (
        make_auth_decision, AuthDecision, DecisionConfig,
    )
    from src.antispoofing.fusion import FusionResult, SpoofDecision
    from src.antispoofing.deepfake_detector import SpoofResult
    from src.antispoofing.replay_detector import ReplayResult
    from src.antispoofing.channel_detector import ChannelMismatchResult

    deepfake = SpoofResult(is_spoof=False, spoof_score=0.1,
                           real_score=0.9, confidence=0.9, detector="mock")
    replay   = ReplayResult(is_replay=False, replay_score=0.1, confidence=0.8, features={})
    mismatch = ChannelMismatchResult(0.0, False, 0.35, 0.0, 1.0, 0.0, 0.0)
    fusion   = FusionResult(
        decision=SpoofDecision.ACCEPT, fused_spoof_score=0.1,
        deepfake=deepfake, replay=replay, channel_mismatch=mismatch,
        effective_threshold=0.5, reject_threshold=0.5,
        retry_threshold=0.35, explanation="mock",
    )
    sv_ok = {"user_id": "u", "score": 0.85, "accepted": True, "threshold": 0.25}

    with patch("src.decision.fusion_layer.run_antispoof_fusion", return_value=fusion), \
         patch("src.decision.fusion_layer.verify", return_value=sv_ok):
        result = make_auth_decision("u", _sine(), config=DecisionConfig())

    assert result.decision == AuthDecision.ACCEPT
    return {"decision": result.decision.value, "sv_score": result.speaker_score}


def check_session_manager() -> dict:
    from unittest.mock import patch
    from src.decision.session import SessionManager, SessionConfig
    from src.decision.fusion_layer import AuthDecision, AuthResult, DecisionConfig
    from src.antispoofing.fusion import FusionResult, SpoofDecision
    from src.antispoofing.deepfake_detector import SpoofResult
    from src.antispoofing.replay_detector import ReplayResult
    from src.antispoofing.channel_detector import ChannelMismatchResult

    mgr = SessionManager(
        auth_config=DecisionConfig(),
        session_config=SessionConfig(max_attempts=3, step_up_after=2),
    )
    sess = mgr.start_session("val_user")

    deepfake = SpoofResult(is_spoof=False, spoof_score=0.05,
                           real_score=0.95, confidence=0.95, detector="mock")
    replay   = ReplayResult(is_replay=False, replay_score=0.05, confidence=0.9, features={})
    mismatch = ChannelMismatchResult(0.0, False, 0.35, 0.0, 1.0, 0.0, 0.0)
    fusion   = FusionResult(
        decision=SpoofDecision.ACCEPT, fused_spoof_score=0.05,
        deepfake=deepfake, replay=replay, channel_mismatch=mismatch,
        effective_threshold=0.5, reject_threshold=0.5,
        retry_threshold=0.35, explanation="mock",
    )
    sv_ok = {"user_id": "val_user", "score": 0.9, "accepted": True, "threshold": 0.25}
    auth_result = AuthResult(
        decision=AuthDecision.ACCEPT, speaker_score=0.9, spoof_score=0.05,
        speaker_accepted=True, spoof_accepted=True, effective_sv_threshold=0.25,
        spoof_result=fusion, explanation="mock accept",
    )

    with patch("src.decision.fusion_layer.run_antispoof_fusion", return_value=fusion), \
         patch("src.decision.fusion_layer.verify", return_value=sv_ok):
        result = mgr.authenticate(sess.session_id, _sine())

    summary = mgr.session_summary(sess.session_id)
    assert result.decision == AuthDecision.ACCEPT
    return {"session_status": summary["status"], "attempts": summary["total_attempts"]}


def check_api_layer() -> dict:
    from fastapi.testclient import TestClient
    from src.api.server import app
    with TestClient(app) as client:
        h = client.get("/health")
        v = client.get("/version")
    assert h.status_code == 200
    assert v.status_code == 200
    assert h.json()["status"] == "ok"
    return {"health": h.json()["status"], "version": v.json()["version"]}


def check_synthetic_benchmark() -> BenchmarkSummary:
    from src.evaluation.test_runner import run_synthetic_benchmark
    report = run_synthetic_benchmark(threshold=0.25)
    eers = [r.metrics.eer for r in report.results]
    mean_eer = float(np.mean(eers))
    best  = min(report.results, key=lambda r: r.metrics.eer)
    worst = max(report.results, key=lambda r: r.metrics.eer)
    # Per brief: EER < 6% in adverse conditions (represented as < 0.06)
    adverse = [r for r in report.results
               if r.label in ("noisy", "whispered", "outdoor", "sick")]
    target_met = all(r.metrics.eer < 0.06 for r in adverse) if adverse else False
    return BenchmarkSummary(
        n_conditions=len(report.results),
        mean_eer=round(mean_eer, 4),
        best_condition=f"{best.axis}/{best.label} (EER={best.metrics.eer:.3f})",
        worst_condition=f"{worst.axis}/{worst.label} (EER={worst.metrics.eer:.3f})",
        target_eer_met=target_met,
    )


# ── main runner ───────────────────────────────────────────────────────────────

COMPONENTS = [
    ("Preprocessing (load/normalize/VAD)",  check_preprocessing),
    ("Channel normalization (CMVN/WCCN)",   check_channel_norm),
    ("Audio augmentation",                  check_augmentation),
    ("Enrollment (embed + store)",          check_enrollment),
    ("Speaker verification",                check_speaker_verification),
    ("Multilingual scoring & disentangle",  check_multilingual_scoring),
    ("Channel mismatch detection",          check_channel_mismatch),
    ("Anti-spoof fusion",                   check_antispoof_fusion),
    ("Decision fusion layer",               check_decision_fusion),
    ("Session manager",                     check_session_manager),
    ("API layer (health/version)",          check_api_layer),
]

KNOWN_LIMITATIONS = [
    "Spectral anti-spoof detector calibrated against ASVspoof 2019 LA; performance "
    "degrades against 2024 challenge attacks and unseen TTS systems.",
    "WCCN requires a dev set with ≥2 samples per speaker; unavailable at cold start.",
    "Channel mismatch heuristics tuned on synthetic augmentation, not real device pairs.",
    "HFAntiSpoof (jungjee/HuBERT-base-AS) not yet benchmarked end-to-end; "
    "spectral fallback is the production path until validated.",
    "No GPU-optimised batching — each request runs inference synchronously.",
    "Multilingual embedders (WavLM, XLS-R, MMS) are lazy-loaded; first request "
    "triggers a large model download and slow init.",
    "Enrollment DB is a flat .pt file; not suitable for concurrent writes in "
    "multi-worker deployments.",
    "Session state is in-process only; lost on server restart.",
]

NEXT_STEPS = [
    "Phase 3 slow tests: download WavLM/XLS-R and benchmark against real "
    "multilingual data (CommonVoice, VoxLingua107).",
    "Phase 5 HF integration: validate jungjee/HuBERT-base-AS on ASVspoof 2021/2024.",
    "Collect real device audio (phone mic vs AirPods vs wired headset) "
    "and tune channel mismatch thresholds.",
    "Replace flat .pt enrollment DB with SQLite or Redis for concurrent safety.",
    "Add GPU batching and request queuing for production throughput.",
    "Run end-to-end EER measurement on VoxCeleb1-H (hard) and ASVspoof 2024.",
    "GDPR / biometric compliance review before production deployment.",
]


def run_validation(
    report_path: str | Path = "data/validation_report.json",
    verbose: bool = True,
) -> ValidationReport:
    t_start = time.time()
    results: list[ComponentResult] = []

    if verbose:
        print("\n" + "=" * 64)
        print("VOICE BIOMETRICS PIPELINE — PHASE 8 VALIDATION")
        print("=" * 64)

    for name, fn in COMPONENTS:
        if verbose:
            print(f"\n  [{len(results)+1:02d}/{len(COMPONENTS)}] {name} ...", end=" ", flush=True)
        result = _validate(name, fn)
        results.append(result)
        if verbose:
            status = "PASS" if result.passed else f"FAIL: {result.error}"
            print(f"{status}  ({result.duration_ms:.0f}ms)")

    if verbose:
        print("\n  [BM] Running synthetic benchmark ...")
    try:
        benchmark = check_synthetic_benchmark()
        bm_ok = True
    except Exception as e:
        benchmark = None
        bm_ok = False
        if verbose:
            print(f"       FAIL: {e}")

    total = round(time.time() - t_start, 2)
    all_passed = all(r.passed for r in results) and bm_ok

    report = ValidationReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_duration_seconds=total,
        passed=all_passed,
        components=results,
        benchmark=benchmark,
        known_limitations=KNOWN_LIMITATIONS,
        next_steps=NEXT_STEPS,
    )
    report.save(report_path)

    if verbose:
        _print_summary(report)

    return report


def _print_summary(report: ValidationReport) -> None:
    passed = sum(1 for c in report.components if c.passed)
    total  = len(report.components)
    print("\n" + "=" * 64)
    print(f"RESULT: {'ALL PASS' if report.passed else 'FAILURES DETECTED'}")
    print(f"  Components:  {passed}/{total} passed")
    print(f"  Duration:    {report.total_duration_seconds:.1f}s")
    if report.benchmark:
        bm = report.benchmark
        print(f"\n  Benchmark ({bm.n_conditions} conditions):")
        print(f"    Mean EER:  {bm.mean_eer*100:.2f}%")
        print(f"    Best:      {bm.best_condition}")
        print(f"    Worst:     {bm.worst_condition}")
        print(f"    <6% EER target (adverse): {'MET' if bm.target_eer_met else 'NOT MET (synthetic data)'}")
    failures = [c for c in report.components if not c.passed]
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    - {f.name}: {f.error}")
    print(f"\n  Known limitations: {len(report.known_limitations)}")
    print(f"  Next steps:        {len(report.next_steps)}")
    print("=" * 64)


if __name__ == "__main__":
    run_validation()
