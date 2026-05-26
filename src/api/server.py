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
from src.api.app_db import init_db
from src.decision.fusion_layer import DecisionConfig, make_auth_decision
from src.decision.session import SessionManager, SessionConfig
from src.enrollment.embedder import get_embedding
from src.enrollment.enrollment_db import (
    enroll_user, get_enrollment, list_users, delete_user,
)
from src.preprocessing.audio_loader import load_and_resample
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

TARGET_SR = 16000
MAX_AUDIO_BYTES = 50 * 1024 * 1024   # 50 MB


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

    _decision_config = DecisionConfig()
    _session_manager = SessionManager(
        auth_config=_decision_config,
        session_config=SessionConfig(),
    )
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

async def _load_uploaded_audio(file: UploadFile) -> torch.Tensor:
    """Read an uploaded audio file and return a preprocessed (1, samples) tensor."""
    content = await file.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds {MAX_AUDIO_BYTES // (1024*1024)} MB limit.",
        )
    try:
        buffer = io.BytesIO(content)
        data, sr = sf.read(buffer, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio file: {e}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    waveform = rms_normalize(waveform)
    waveform = apply_vad_energy(waveform)
    return waveform


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
    for f in files:
        waveform = await _load_uploaded_audio(f)
        emb = get_embedding(waveform)
        embeddings.append(emb)

    enroll_user(user_id, embeddings)
    logger.info(f"Enrolled user '{user_id}' with {len(embeddings)} sample(s) by '{current_user['username']}'.")

    return EnrollResponse(
        user_id=user_id,
        enrolled=True,
        samples_used=len(embeddings),
        message=f"User '{user_id}' enrolled successfully.",
    )


@app.post("/sessions", tags=["Authentication"])
async def start_session(
    user_id: Annotated[str, Form()],
    current_user: dict = Depends(get_current_user),
):
    """Start a new authentication session for a voice-enrolled user."""
    try:
        get_enrollment(user_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' is not enrolled.",
        )

    session = _session_manager.start_session(user_id)
    logger.info(f"Session started: {session.session_id} for user '{user_id}'")
    return {"session_id": session.session_id, "user_id": user_id}


@app.post("/authenticate", response_model=AuthResponse, tags=["Authentication"])
async def authenticate(
    session_id: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    enroll_file: Annotated[Optional[UploadFile], File()] = None,
    language: Annotated[Optional[str], Form()] = "default",
    current_user: dict = Depends(get_current_user),
):
    """Authenticate one attempt within an existing session."""
    session = _session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.is_locked:
        raise HTTPException(status_code=403, detail="Session is locked. Start a new session.")

    probe_waveform = await _load_uploaded_audio(file)
    enroll_waveform = None
    if enroll_file:
        enroll_waveform = await _load_uploaded_audio(enroll_file)

    try:
        result = _session_manager.authenticate(
            session_id=session_id,
            probe_waveform=probe_waveform,
            enroll_waveform=enroll_waveform,
            language=language,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    logger.info(
        f"Auth attempt {session.total_attempts} for '{session.user_id}': "
        f"{result.decision.value} (sv={result.speaker_score:.3f} spoof={result.spoof_score:.3f})"
    )

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
    """List all active sessions. Requires admin or ops role."""
    if not _session_manager:
        return []
    summaries = []
    for session_id in list(_session_manager._sessions.keys()):
        s = _session_manager.session_summary(session_id)
        if s:
            summaries.append(s)
    return summaries


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
    return DeleteUserResponse(user_id=user_id, deleted=True)
