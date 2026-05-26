"""
API tests for Phase 7.

Uses FastAPI TestClient with mocked pipeline sub-systems so no model
downloads or audio files are needed.
"""

from __future__ import annotations

import io
import struct
import torch
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.auth_router import get_current_user
from src.decision.fusion_layer import AuthDecision, AuthResult, DecisionConfig
from src.antispoofing.fusion import FusionResult, SpoofDecision
from src.antispoofing.deepfake_detector import SpoofResult
from src.antispoofing.replay_detector import ReplayResult
from src.antispoofing.channel_detector import ChannelMismatchResult
from src.decision.session import Session, SessionStatus

SR = 16000


# ── helpers ──────────────────────────────────────────────────────────────────

def _wav_bytes(duration=2.0, freq=220.0) -> bytes:
    """Generate a minimal in-memory WAV file using soundfile (no FFmpeg needed)."""
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * freq * t))
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


def _make_auth_result(decision=AuthDecision.ACCEPT, sv_score=0.8, spoof_score=0.1) -> AuthResult:
    deepfake = SpoofResult(
        is_spoof=False, spoof_score=spoof_score,
        real_score=1-spoof_score, confidence=0.9, detector="mock",
    )
    replay = ReplayResult(is_replay=False, replay_score=0.1, confidence=0.8, features={})
    mismatch = ChannelMismatchResult(
        mismatch_score=0.0, is_mismatch=False, threshold=0.35,
        spectral_distance=0.0, bandwidth_ratio=1.0,
        snr_delta_db=0.0, spoof_suspicion_adjustment=0.0,
    )
    fusion = FusionResult(
        decision=SpoofDecision.ACCEPT,
        fused_spoof_score=spoof_score,
        deepfake=deepfake, replay=replay,
        channel_mismatch=mismatch,
        effective_threshold=0.5, reject_threshold=0.5,
        retry_threshold=0.35, explanation="mock",
    )
    return AuthResult(
        decision=decision,
        speaker_score=sv_score,
        spoof_score=spoof_score,
        speaker_accepted=sv_score >= 0.25,
        spoof_accepted=True,
        effective_sv_threshold=0.25,
        spoof_result=fusion,
        explanation="mock accept",
    )


@pytest.fixture
def client():
    """TestClient with mocked auth (admin) and mocked enrollment DB."""
    app.dependency_overrides[get_current_user] = lambda: {"username": "test_admin", "role": "admin"}
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── health / version ──────────────────────────────────────────────────────────

class TestServiceEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "enrolled_users" in body

    def test_version(self, client):
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == "1.0.0"
        assert "pipeline_phases" in body


# ── enrollment ────────────────────────────────────────────────────────────────

class TestEnrollEndpoint:
    def test_enroll_single_file(self, client):
        wav = _wav_bytes()
        with patch("src.api.server.get_embedding", return_value=torch.randn(192)), \
             patch("src.api.server.enroll_user"):
            r = client.post(
                "/enroll",
                data={"user_id": "alice"},
                files={"files": ("enroll.wav", wav, "audio/wav")},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["enrolled"] is True
        assert body["user_id"] == "alice"
        assert body["samples_used"] == 1

    def test_enroll_multiple_files(self, client):
        wav = _wav_bytes()
        with patch("src.api.server.get_embedding", return_value=torch.randn(192)), \
             patch("src.api.server.enroll_user"):
            r = client.post(
                "/enroll",
                data={"user_id": "bob"},
                files=[
                    ("files", ("a.wav", wav, "audio/wav")),
                    ("files", ("b.wav", wav, "audio/wav")),
                ],
            )
        assert r.status_code == 200
        assert r.json()["samples_used"] == 2

    def test_enroll_no_files_returns_400(self, client):
        r = client.post("/enroll", data={"user_id": "carol"})
        assert r.status_code == 422   # FastAPI validation: files required

    def test_enroll_invalid_audio_returns_422(self, client):
        with patch("src.api.server.get_embedding", return_value=torch.randn(192)):
            r = client.post(
                "/enroll",
                data={"user_id": "dave"},
                files={"files": ("bad.wav", b"not audio", "audio/wav")},
            )
        assert r.status_code == 422


# ── user management ───────────────────────────────────────────────────────────

class TestUserEndpoints:
    def test_get_enrolled_user(self, client):
        with patch("src.api.server.list_users", return_value=["alice"]):
            r = client.get("/users/alice")
        assert r.status_code == 200
        assert r.json()["enrolled"] is True

    def test_get_unenrolled_user(self, client):
        with patch("src.api.server.list_users", return_value=[]):
            r = client.get("/users/nobody")
        assert r.status_code == 200
        assert r.json()["enrolled"] is False

    def test_delete_enrolled_user(self, client):
        with patch("src.api.server.list_users", return_value=["alice"]), \
             patch("src.api.server.delete_user"):
            r = client.delete("/users/alice")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_missing_user_returns_404(self, client):
        with patch("src.api.server.list_users", return_value=[]):
            r = client.delete("/users/ghost")
        assert r.status_code == 404


# ── session flow ──────────────────────────────────────────────────────────────

class TestSessionEndpoints:
    def test_start_session_enrolled_user(self, client):
        with patch("src.api.server.get_enrollment", return_value=torch.randn(192)):
            r = client.post("/sessions", data={"user_id": "alice"})
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert body["user_id"] == "alice"

    def test_start_session_unenrolled_user_returns_404(self, client):
        with patch("src.api.server.get_enrollment", side_effect=KeyError("alice")):
            r = client.post("/sessions", data={"user_id": "alice"})
        assert r.status_code == 404

    def test_get_session_status(self, client):
        with patch("src.api.server.get_enrollment", return_value=torch.randn(192)):
            sess_r = client.post("/sessions", data={"user_id": "alice"})
        session_id = sess_r.json()["session_id"]

        r = client.get(f"/sessions/{session_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == session_id
        assert body["total_attempts"] == 0

    def test_get_missing_session_returns_404(self, client):
        r = client.get("/sessions/nonexistent-id")
        assert r.status_code == 404


# ── authenticate ──────────────────────────────────────────────────────────────

class TestAuthenticateEndpoint:
    def _start_session(self, client, user_id="alice") -> str:
        with patch("src.api.server.get_enrollment", return_value=torch.randn(192)):
            r = client.post("/sessions", data={"user_id": user_id})
        return r.json()["session_id"]

    def test_authenticate_accept(self, client):
        session_id = self._start_session(client)
        wav = _wav_bytes()
        auth_result = _make_auth_result(AuthDecision.ACCEPT)
        with patch("src.api.server._session_manager") as mock_mgr:
            mock_session = MagicMock()
            mock_session.is_locked = False
            mock_session.user_id = "alice"
            mock_session.total_attempts = 1
            mock_mgr.get_session.return_value = mock_session
            mock_mgr.authenticate.return_value = auth_result
            mock_mgr.session_summary.return_value = {}

            r = client.post(
                "/authenticate",
                data={"session_id": session_id},
                files={"file": ("probe.wav", wav, "audio/wav")},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "accept"
        assert body["speaker_score"] == pytest.approx(0.8, abs=1e-3)

    def test_authenticate_missing_session_returns_404(self, client):
        wav = _wav_bytes()
        r = client.post(
            "/authenticate",
            data={"session_id": "bad-id"},
            files={"file": ("probe.wav", wav, "audio/wav")},
        )
        assert r.status_code == 404

    def test_authenticate_locked_session_returns_403(self, client):
        session_id = self._start_session(client)
        wav = _wav_bytes()
        with patch("src.api.server._session_manager") as mock_mgr:
            mock_session = MagicMock()
            mock_session.is_locked = True
            mock_mgr.get_session.return_value = mock_session
            r = client.post(
                "/authenticate",
                data={"session_id": session_id},
                files={"file": ("probe.wav", wav, "audio/wav")},
            )
        assert r.status_code == 403

    def test_authenticate_response_contains_spoof_detail(self, client):
        session_id = self._start_session(client)
        wav = _wav_bytes()
        auth_result = _make_auth_result(AuthDecision.ACCEPT)
        with patch("src.api.server._session_manager") as mock_mgr:
            mock_session = MagicMock()
            mock_session.is_locked = False
            mock_session.user_id = "alice"
            mock_session.total_attempts = 1
            mock_mgr.get_session.return_value = mock_session
            mock_mgr.authenticate.return_value = auth_result
            r = client.post(
                "/authenticate",
                data={"session_id": session_id},
                files={"file": ("probe.wav", wav, "audio/wav")},
            )
        body = r.json()
        assert "spoof_detail" in body
        assert "fused_score" in body["spoof_detail"]
        assert "deepfake_score" in body["spoof_detail"]
        assert "replay_score" in body["spoof_detail"]

    def test_request_id_in_response_headers(self, client):
        r = client.get("/health")
        assert "x-request-id" in r.headers
        assert "x-response-time-ms" in r.headers
