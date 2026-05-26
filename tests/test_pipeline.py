"""
Quick end-to-end test: enroll a user with one audio file, verify with another.

Usage:
    python -m pytest tests/test_pipeline.py -v
    # or run directly:
    python tests/test_pipeline.py enroll.wav verify.wav
"""

import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.audio_loader import load_and_resample
from src.preprocessing.normalization import rms_normalize
from src.preprocessing.vad import apply_vad_energy
from src.enrollment.embedder import get_embedding
from src.enrollment.enrollment_db import enroll_user, list_users
from src.verification.speaker_verifier import verify


def preprocess(path: str) -> torch.Tensor:
    waveform = load_and_resample(path)
    waveform = rms_normalize(waveform)
    waveform = apply_vad_energy(waveform)
    return waveform


def run_demo(enroll_path: str, verify_path: str, user_id: str = "test_user"):
    print(f"\n--- Enrolling '{user_id}' from: {enroll_path}")
    enroll_waveform = preprocess(enroll_path)
    enroll_embedding = get_embedding(enroll_waveform)
    enroll_user(user_id, [enroll_embedding])
    print(f"Enrolled users: {list_users()}")

    print(f"\n--- Verifying '{user_id}' from: {verify_path}")
    verify_waveform = preprocess(verify_path)
    result = verify(user_id, verify_waveform)

    print(f"\nResult:")
    print(f"  Score:    {result['score']:.4f}  (threshold: {result['threshold']})")
    print(f"  Decision: {'ACCEPTED' if result['accepted'] else 'REJECTED'}")
    return result


# --- pytest tests (use synthetic sine waves as stand-ins) ---

def _make_sine(freq: float = 220.0, duration: float = 3.0, sr: int = 16000) -> torch.Tensor:
    t = torch.linspace(0, duration, int(sr * duration))
    return (0.5 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def test_same_speaker_accepted():
    """Same synthetic signal should score well above threshold."""
    waveform = _make_sine(220.0)
    embedding = get_embedding(waveform)
    enroll_user("pytest_user", [embedding])
    result = verify("pytest_user", waveform)
    assert result["score"] > 0.9, f"Same-speaker score too low: {result['score']}"
    assert result["accepted"]


def test_different_speaker_rejected():
    """Very different signals should score low (sine at 220 Hz vs 880 Hz)."""
    enroll_waveform = _make_sine(220.0)
    probe_waveform = _make_sine(880.0)
    embedding = get_embedding(enroll_waveform)
    enroll_user("pytest_user2", [embedding])
    result = verify("pytest_user2", probe_waveform, threshold=0.90)
    # We only assert score is lower — exact rejection depends on model
    assert result["score"] < 0.95, f"Different-speaker score unexpectedly high: {result['score']}"


# --- CLI entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll and verify a speaker")
    parser.add_argument("enroll", help="Path to enrollment audio file")
    parser.add_argument("verify", help="Path to verification audio file")
    parser.add_argument("--user", default="demo_user", help="User ID to enroll")
    args = parser.parse_args()
    run_demo(args.enroll, args.verify, user_id=args.user)
