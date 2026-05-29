"""
Voice Biometrics FastAPI inference service.

Endpoints:
  POST /auth/login               Get a JWT access token
  GET  /auth/me                  Current user info
  POST /enroll                   Upload audio to enroll a user
  POST /authenticate             Authenticate a user claim against a session
  POST /sessions                 Start a new authentication session
  GET  /sessions                 List all sessions (ops/admin)
  GET  /sessions/{session_id}    Get session status and attempt history
  GET  /users/{user_id}          Check if a user is enrolled
  DELETE /users/{user_id}        Remove a user's enrollment (admin only)
  GET  /health                   Service health check
  GET  /version                  Pipeline version info
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    AuthResponse, DeleteUserResponse, EnrollResponse, HealthResponse,
    SessionResponse, SpoofDetail, UserStatusResponse, VersionResponse,
    AttemptSummary, AuthDecisionEnum, SpoofDecisionEnum,
)
from src.api.auth_router import router as auth_router, get_current_user, require_role
import src.api.auth_router as _auth_mod
from src.api.app_db import (
    init_db,
    log_session as _log_session,
    list_session_logs as _list_session_logs,
    init_audit_log as _init_audit_log,
    log_audit as _log_audit,
    list_audit_logs as _list_audit_logs,
)
from src.decision.fusion_layer import DecisionConfig, make_auth_decision
from src.decision.session import SessionManager, SessionConfig
from src.enrollment.embedder import get_embedding
from src.enrollment.enrollment_db import (
    enroll_user, get_enrollment, list_users, delete_user,
)
from src.preprocessing.audio_loader import load_and_resample
from src.preprocessing.denoiser import denoise, is_enabled as denoiser_enabled, load_model as load_denoiser
from src.preprocessing.normalization import rms_normalize
from src.preprocessing.vad import apply_vad_energy

logger = logging.getLogger("voice_biometrics.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ── global state ─────────────────────────────────────────────────────────────

_session_manager: SessionManager | None = None
_decision_config: DecisionConfig | None = None
_whisper_model = None

TARGET_SR = 16000
MAX_AUDIO_BYTES = 50 * 1024 * 1024   # 50 MB
MIN_ENROLL_SPEECH_SECONDS = 10.0     # minimum total speech time after VAD

# ── rate limiting (in-memory, per IP) ────────────────────────────────────────

from collections import defaultdict as _defaultdict
_rate_buckets: dict[str, list[float]] = _defaultdict(list)
_RATE_WINDOW  = 60.0   # seconds
_RATE_MAX     = 10     # max session starts per IP per window

def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    bucket = [t for t in _rate_buckets[ip] if now - t < _RATE_WINDOW]
    _rate_buckets[ip] = bucket
    if len(bucket) >= _RATE_MAX:
        return True
    _rate_buckets[ip].append(now)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session_manager, _decision_config

    logger.info("Starting voice biometrics service...")

    # Load config
    cfg: dict = {}
    config_path = Path("configs/api.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    auth_cfg = cfg.get("auth", {})
    _auth_mod.configure(
        secret_key=auth_cfg.get("secret_key", "change-me"),
        expire_minutes=auth_cfg.get("token_expire_minutes", 60),
    )
    init_db(
        admin_username=auth_cfg.get("admin_username", "admin"),
        admin_password=auth_cfg.get("admin_password", "admin123"),
    )
    _init_audit_log()

    _decision_config = DecisionConfig()
    _session_manager = SessionManager(
        auth_config=_decision_config,
        session_config=SessionConfig(),
    )

    load_denoiser()

    # Load HuBERT anti-spoof model (replaces handcrafted spectral detector)
    from src.antispoofing.deepfake_detector import load_hf_model as _load_antispoof
    _load_antispoof("motheecreator/Deepfake-audio-detection")

    # Load Whisper for challenge-response verification
    global _whisper_model
    try:
        import whisper as _whisper
        _whisper_model = _whisper.load_model("base")
        logger.info("Whisper base loaded — challenge verification active.")
    except Exception as e:
        logger.warning(f"Whisper unavailable ({e}) — challenge verification disabled.")

    logger.info("Service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Voice Biometrics API",
    description="Speaker verification + anti-spoofing inference service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── logging middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"[{request_id}] {response.status_code} {elapsed:.1f}ms")
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ── audio loading helper ──────────────────────────────────────────────────────

# Use imageio-ffmpeg's bundled binary — avoids conda DLL conflicts entirely.
try:
    import imageio_ffmpeg as _iio_ffmpeg
    _FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()
    _AV_AVAILABLE = True
    logger.info(f"imageio-ffmpeg ready: {_FFMPEG_EXE}")
except Exception:
    _FFMPEG_EXE = None
    _AV_AVAILABLE = False
    logger.warning("imageio-ffmpeg not available — m4a/mp3 decoding unavailable.")


def _decode_with_ffmpeg(content: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Decode audio via imageio-ffmpeg's bundled binary — outputs raw PCM, no soundfile."""
    import subprocess, tempfile
    suffix = Path(filename or "audio.m4a").suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        # Output raw float32 mono PCM at TARGET_SR — numpy reads it directly, no soundfile needed.
        result = subprocess.run(
            [_FFMPEG_EXE, "-y", "-i", str(tmp_path),
             "-f", "f32le", "-ar", str(TARGET_SR), "-ac", "1", "pipe:1"],
            capture_output=True,
            check=True,
        )
        if not result.stdout:
            raise RuntimeError("ffmpeg produced no output")
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        # Return (samples, 1) to match soundfile's always_2d=True output format
        return audio[:, np.newaxis], TARGET_SR
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode(errors="replace"))
    finally:
        tmp_path.unlink(missing_ok=True)


async def _load_uploaded_audio(file: UploadFile) -> torch.Tensor:
    """Read an uploaded audio file and return a preprocessed (1, samples) tensor.

    Tries soundfile first (WAV/FLAC/OGG), then falls back to pydub/ffmpeg
    for everything else (m4a, mp3, aac, …).
    """
    content = await file.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds {MAX_AUDIO_BYTES // (1024*1024)} MB limit.",
        )

    data, sr = None, None

    # soundfile handles WAV/FLAC/OGG natively; skip it for formats it can't decode
    # (libsndfile segfaults on m4a/mp3 rather than raising a Python exception)
    _ext = Path(file.filename or "").suffix.lower()
    _soundfile_formats = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".snd"}
    if _ext in _soundfile_formats or _ext == "":
        try:
            data, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
        except Exception:
            pass

    # Fallback: ffmpeg subprocess (handles m4a, mp3, aac, …)
    if data is None:
        if not _AV_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot decode '{file.filename}'. Supported formats: WAV, FLAC, OGG, M4A, MP3.",
            )
        try:
            data, sr = _decode_with_ffmpeg(content, file.filename or "")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not decode audio file '{file.filename}': {e}")

    waveform = torch.from_numpy(data.T)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = denoise(waveform, sr)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    waveform = rms_normalize(waveform)
    waveform = apply_vad_energy(waveform)
    return waveform


def _transcribe_and_verify(waveform: torch.Tensor, challenge: str) -> tuple[bool, str]:
    """Transcribe probe audio with Whisper and verify all challenge words were spoken."""
    import re
    from difflib import SequenceMatcher

    if _whisper_model is None or not challenge:
        return True, ""
    try:
        audio_np = waveform.squeeze(0).numpy()
        # initial_prompt biases the decoder toward the expected words — large accuracy boost
        result = _whisper_model.transcribe(
            audio_np, language="en", fp16=False,
            initial_prompt=f"The speaker will say these words: {challenge}",
        )
        transcription = result["text"].lower()
        trans_words = re.sub(r"[^a-z\s]", "", transcription).split()
        challenge_words = challenge.lower().split()

        missing = []
        for cw in challenge_words:
            # fuzzy match: accept if any transcribed word is ≥80% similar
            found = any(SequenceMatcher(None, cw, tw).ratio() >= 0.80 for tw in trans_words)
            if not found:
                missing.append(cw)

        logger.info(f"Challenge: '{challenge}' | heard: '{transcription}' | missing={missing or 'none'}")
        if missing:
            return False, f"Did not hear: {', '.join(missing)}"
        return True, transcription
    except Exception as e:
        logger.warning(f"Whisper transcription failed ({e}) — skipping challenge check.")
        return True, ""


def _spoof_detail(fusion_result) -> SpoofDetail:
    cm = fusion_result.channel_mismatch
    return SpoofDetail(
        decision=SpoofDecisionEnum(fusion_result.decision.value),
        fused_score=fusion_result.fused_spoof_score,
        deepfake_score=fusion_result.deepfake.spoof_score,
        replay_score=fusion_result.replay.replay_score,
        channel_mismatch=cm.is_mismatch if cm else False,
        channel_adjustment=cm.spoof_suspicion_adjustment if cm else 0.0,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Service"])
async def health():
    """Service liveness — public, no auth required."""
    return HealthResponse(
        status="ok",
        enrolled_users=len(list_users()),
        active_sessions=len(_session_manager._sessions) if _session_manager else 0,
        denoising=denoiser_enabled(),
    )


@app.get("/version", response_model=VersionResponse, tags=["Service"])
async def version():
    """Pipeline version — public, no auth required."""
    return VersionResponse(
        version="1.0.0",
        pipeline_phases=[
            "preprocessing", "enrollment", "speaker_verification",
            "multilingual", "channel_normalization", "augmentation",
            "channel_mismatch", "deepfake_detection", "replay_detection",
            "antispoof_fusion", "decision_fusion", "session_management",
        ],
        default_sv_model="speechbrain/spkrec-ecapa-voxceleb",
        default_embedder_backend="ecapa",
    )


@app.post("/enroll", response_model=EnrollResponse, tags=["Enrollment"])
async def enroll(
    user_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    language: Annotated[Optional[str], Form()] = "default",
    current_user: dict = Depends(get_current_user),
):
    """Enroll a user with one or more audio samples. Requires authentication."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file required.")

    embeddings = []
    total_speech_samples = 0
    for f in files:
        waveform = await _load_uploaded_audio(f)
        total_speech_samples += waveform.shape[-1]
        emb = get_embedding(waveform)
        embeddings.append(emb)

    total_speech_seconds = total_speech_samples / TARGET_SR
    if total_speech_seconds < MIN_ENROLL_SPEECH_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Not enough voice data: {total_speech_seconds:.1f}s of speech detected "
                f"after noise removal (minimum {MIN_ENROLL_SPEECH_SECONDS:.0f}s required). "
                "Please record or upload more audio."
            ),
        )

    enroll_user(user_id, embeddings)
    logger.info(f"Enrolled user '{user_id}' with {len(embeddings)} sample(s) by '{current_user['username']}'.")
    _log_audit(current_user["username"], "enroll", user_id, f"samples={len(embeddings)}")

    return EnrollResponse(
        user_id=user_id,
        enrolled=True,
        samples_used=len(embeddings),
        message=f"User '{user_id}' enrolled successfully.",
    )


@app.post("/sessions", tags=["Authentication"])
async def start_session(
    request: Request,
    user_id: Annotated[str, Form()],
    current_user: dict = Depends(get_current_user),
):
    """Start a new authentication session for a voice-enrolled user."""
    client_ip = request.client.host if request.client else "unknown"
    if _is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many session requests. Please wait before trying again.",
        )
    try:
        get_enrollment(user_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' is not enrolled.",
        )

    session = _session_manager.start_session(user_id)
    logger.info(f"Session started: {session.session_id} for user '{user_id}' challenge='{session.challenge}'")
    return {
        "session_id": session.session_id,
        "user_id": user_id,
        "challenge": session.challenge,
        "expires_in": int(session.config.session_timeout_seconds),
    }


@app.post("/authenticate", response_model=AuthResponse, tags=["Authentication"])
async def authenticate(
    session_id: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    enroll_file: Annotated[Optional[UploadFile], File()] = None,
    language: Annotated[Optional[str], Form()] = "default",
    current_user: dict = Depends(get_current_user),
):
    """Authenticate one attempt within an existing session."""
    import traceback
    session = _session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.is_locked:
        raise HTTPException(status_code=403, detail="Session is locked. Start a new session.")

    try:
        logger.info("Loading probe audio...")
        probe_waveform = await _load_uploaded_audio(file)
        logger.info(f"Probe waveform shape: {probe_waveform.shape}, dtype: {probe_waveform.dtype}")
        probe_duration = probe_waveform.shape[-1] / TARGET_SR
        if probe_duration < 2.0:
            raise HTTPException(
                status_code=422,
                detail=f"Recording too short ({probe_duration:.1f}s of speech detected). Please speak clearly for at least 2 seconds and try again.",
            )
        enroll_waveform = None
        if enroll_file:
            enroll_waveform = await _load_uploaded_audio(enroll_file)
        # Verify challenge phrase before running speaker verification
        ok, detail = _transcribe_and_verify(probe_waveform, session.challenge)
        if not ok:
            raise HTTPException(status_code=401, detail=f"Challenge phrase not spoken correctly — {detail}.")

        logger.info("Running authentication pipeline...")
        result = _session_manager.authenticate(
            session_id=session_id,
            probe_waveform=probe_waveform,
            enroll_waveform=enroll_waveform,
            language=language,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Authentication pipeline crashed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    logger.info(
        f"Auth attempt {session.total_attempts} for '{session.user_id}': "
        f"{result.decision.value} (sv={result.speaker_score:.3f} spoof={result.spoof_score:.3f})"
    )

    # Persist terminal sessions to SQLite and remove from memory
    if session.status.value != "active":
        try:
            _log_session(_session_manager.session_summary(session_id))
            _session_manager.end_session(session_id)
        except Exception as _e:
            logger.warning(f"Failed to log session to DB: {_e}")

    return AuthResponse(
        decision=AuthDecisionEnum(result.decision.value),
        user_id=session.user_id,
        session_id=session_id,
        speaker_score=result.speaker_score,
        spoof_score=result.spoof_score,
        speaker_accepted=result.speaker_accepted,
        spoof_accepted=result.spoof_accepted,
        effective_threshold=result.effective_sv_threshold,
        spoof_detail=_spoof_detail(result.spoof_result),
        explanation=result.explanation,
        attempt_number=session.total_attempts,
    )


@app.get("/sessions", tags=["Authentication"])
async def list_sessions(
    current_user: dict = Depends(require_role("admin", "ops")),
):
    """List all sessions — merges SQLite history with active in-memory sessions."""
    # Logged (completed) sessions from SQLite
    logged = _list_session_logs()
    logged_ids = {s["session_id"] for s in logged}

    # Active sessions still in memory (not yet logged)
    active = []
    if _session_manager:
        for sid in list(_session_manager._sessions.keys()):
            if sid not in logged_ids:
                s = _session_manager.session_summary(sid)
                if s:
                    active.append(s)

    return active + logged


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["Authentication"])
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Retrieve attempt history and current status for a session."""
    summary = _session_manager.session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionResponse(
        session_id=summary["session_id"],
        user_id=summary["user_id"],
        status=summary["status"],
        total_attempts=summary["total_attempts"],
        retry_count=summary["retry_count"],
        is_locked=summary["is_locked"],
        attempts=[
            AttemptSummary(
                attempt=a["attempt"],
                decision=AuthDecisionEnum(a["decision"]),
                speaker_score=a["speaker_score"],
                spoof_score=a["spoof_score"],
                explanation=a["explanation"],
            )
            for a in summary["attempts"]
        ],
    )


@app.get("/users/{user_id}", response_model=UserStatusResponse, tags=["Enrollment"])
async def get_user(user_id: str):
    """Check whether a voice user is enrolled — public."""
    enrolled = user_id in list_users()
    return UserStatusResponse(user_id=user_id, enrolled=enrolled)


@app.get("/audit", tags=["Admin"])
async def get_audit_log(
    limit: int = 500,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin only — retrieve the audit log."""
    return _list_audit_logs(limit=limit)


@app.get("/users", tags=["Enrollment"])
async def get_all_users(current_user: dict = Depends(get_current_user)):
    """List all enrolled voice users."""
    return {"users": list_users()}


@app.delete("/users/{user_id}", response_model=DeleteUserResponse, tags=["Enrollment"])
async def remove_user(
    user_id: str,
    current_user: dict = Depends(require_role("admin")),
):
    """Delete a user's voice enrollment. Requires admin role."""
    if user_id not in list_users():
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    delete_user(user_id)
    logger.info(f"Deleted enrollment for user '{user_id}' by admin '{current_user['username']}'")
    _log_audit(current_user["username"], "delete_voice", user_id)
    return DeleteUserResponse(user_id=user_id, deleted=True)
