"""Pydantic request/response schemas for the voice biometrics API."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── shared enums ──────────────────────────────────────────────────────────────

class AuthDecisionEnum(str, Enum):
    ACCEPT  = "accept"
    REJECT  = "reject"
    RETRY   = "retry"
    STEP_UP = "step_up"


class SpoofDecisionEnum(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    RETRY  = "retry"


# ── enroll ────────────────────────────────────────────────────────────────────

class EnrollResponse(BaseModel):
    user_id: str
    enrolled: bool
    samples_used: int
    message: str


# ── authenticate ──────────────────────────────────────────────────────────────

class SpoofDetail(BaseModel):
    decision: SpoofDecisionEnum
    fused_score: float = Field(ge=0.0, le=1.0)
    deepfake_score: float = Field(ge=0.0, le=1.0)
    replay_score: float = Field(ge=0.0, le=1.0)
    channel_mismatch: bool
    channel_adjustment: float


class AuthResponse(BaseModel):
    decision: AuthDecisionEnum
    user_id: str
    session_id: str
    speaker_score: float = Field(ge=-1.0, le=1.0)
    spoof_score: float = Field(ge=0.0, le=1.0)
    speaker_accepted: bool
    spoof_accepted: bool
    effective_threshold: float
    spoof_detail: SpoofDetail
    explanation: str
    attempt_number: int


# ── session ───────────────────────────────────────────────────────────────────

class AttemptSummary(BaseModel):
    attempt: int
    decision: AuthDecisionEnum
    speaker_score: float
    spoof_score: float
    explanation: str


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: str
    total_attempts: int
    retry_count: int
    is_locked: bool
    created_at: Optional[float] = None
    challenge: Optional[str] = None
    attempts: list[AttemptSummary]


# ── user management ───────────────────────────────────────────────────────────

class UserStatusResponse(BaseModel):
    user_id: str
    enrolled: bool


class DeleteUserResponse(BaseModel):
    user_id: str
    deleted: bool


# ── health / version ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    enrolled_users: int
    active_sessions: int
    denoising: bool = False


class VersionResponse(BaseModel):
    version: str
    pipeline_phases: list[str]
    default_sv_model: str
    default_embedder_backend: str
