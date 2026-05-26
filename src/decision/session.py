"""
Session manager — tracks authentication attempts per session and enforces
retry/lockout/step-up escalation policy.

A "session" covers one login attempt sequence for a user. It begins when
the user claims an identity and ends with ACCEPT, hard REJECT, or LOCKOUT.

Escalation policy:
  Attempt 1:  normal thresholds
  Attempt 2:  if previous was RETRY → same thresholds, log warning
  Attempt 3:  if still RETRY → trigger STEP_UP regardless of spoof score
  Attempt N≥4: LOCKOUT (too many failed attempts)

The session is intentionally stateless across restarts (no DB) — for
production this state would live in Redis or a sessions table.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from src.decision.fusion_layer import (
    AuthDecision, AuthResult, DecisionConfig, DEFAULT_CONFIG, make_auth_decision,
)


class SessionStatus(str, Enum):
    ACTIVE   = "active"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    LOCKED   = "locked"


@dataclass
class AttemptRecord:
    attempt_number: int
    decision: AuthDecision
    speaker_score: float
    spoof_score: float
    timestamp: float
    explanation: str


@dataclass
class SessionConfig:
    max_attempts: int = 4              # lock out after this many non-accept attempts
    step_up_after: int = 2             # escalate to STEP_UP after N retries
    session_timeout_seconds: float = 300.0   # 5-minute session window
    lockout_duration_seconds: float = 600.0  # 10-minute lockout


@dataclass
class Session:
    session_id: str
    user_id: str
    status: SessionStatus
    attempts: list[AttemptRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    locked_at: Optional[float] = None
    config: SessionConfig = field(default_factory=SessionConfig)

    @property
    def retry_count(self) -> int:
        return sum(1 for a in self.attempts if a.decision == AuthDecision.RETRY)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.config.session_timeout_seconds

    @property
    def is_locked(self) -> bool:
        if self.status != SessionStatus.LOCKED:
            return False
        if self.locked_at and time.time() - self.locked_at > self.config.lockout_duration_seconds:
            return False  # lockout expired
        return True


class SessionManager:
    """
    Manages active sessions and orchestrates multi-attempt authentication.

    Usage:
        mgr = SessionManager()
        session = mgr.start_session(user_id="alice")
        result  = mgr.authenticate(session.session_id, probe_waveform)
        print(result.decision)
    """

    def __init__(
        self,
        auth_config: DecisionConfig = DEFAULT_CONFIG,
        session_config: SessionConfig = None,
    ):
        self._auth_config = auth_config
        self._session_config = session_config or SessionConfig()
        self._sessions: dict[str, Session] = {}

    # ── session lifecycle ─────────────────────────────────────────────────────

    def start_session(self, user_id: str) -> Session:
        """Create a new authentication session for user_id."""
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            user_id=user_id,
            status=SessionStatus.ACTIVE,
            config=self._session_config,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── authentication attempt ────────────────────────────────────────────────

    def authenticate(
        self,
        session_id: str,
        probe_waveform: torch.Tensor,
        enroll_waveform: Optional[torch.Tensor] = None,
        language: str = "default",
    ) -> AuthResult:
        """
        Process one authentication attempt within a session.

        Applies escalation policy on top of the base auth decision:
          - Forces STEP_UP if retry count ≥ step_up_after
          - Forces REJECT (lockout) if total_attempts ≥ max_attempts

        Returns the (possibly escalated) AuthResult.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        if session.is_locked:
            raise PermissionError(
                f"Session {session_id} is locked out. Try again later."
            )

        if session.is_expired:
            session.status = SessionStatus.LOCKED
            session.locked_at = time.time()
            raise PermissionError("Session expired. Please start a new session.")

        if session.status != SessionStatus.ACTIVE:
            raise PermissionError(
                f"Session is {session.status.value}, not active."
            )

        # Apply escalated config if retry threshold reached
        config = self._escalated_config(session)

        # Run auth decision
        result = make_auth_decision(
            user_id=session.user_id,
            probe_waveform=probe_waveform,
            enroll_waveform=enroll_waveform,
            config=config,
            language=language,
        )

        # Force STEP_UP if we've retried too many times
        if (
            result.decision == AuthDecision.RETRY
            and session.retry_count >= self._session_config.step_up_after
        ):
            result = AuthResult(
                **{**result.__dict__,
                   "decision": AuthDecision.STEP_UP,
                   "explanation": (
                       f"step_up forced after {session.retry_count} retries: "
                       + result.explanation
                   )}
            )

        # Record attempt
        record = AttemptRecord(
            attempt_number=session.total_attempts + 1,
            decision=result.decision,
            speaker_score=result.speaker_score,
            spoof_score=result.spoof_score,
            timestamp=time.time(),
            explanation=result.explanation,
        )
        session.attempts.append(record)

        # Update session status
        if result.decision == AuthDecision.ACCEPT:
            session.status = SessionStatus.ACCEPTED

        else:
            # Count all non-accept outcomes (REJECT, RETRY, STEP_UP)
            non_accept = sum(
                1 for a in session.attempts
                if a.decision != AuthDecision.ACCEPT
            )
            if non_accept >= self._session_config.max_attempts:
                session.status = SessionStatus.LOCKED
                session.locked_at = time.time()
                result = AuthResult(
                    **{**result.__dict__,
                       "decision": AuthDecision.REJECT,
                       "explanation": "LOCKED: max attempts exceeded. " + result.explanation}
                )
            # else: session stays ACTIVE — user can try again

        return result

    def _escalated_config(self, session: Session) -> DecisionConfig:
        """Tighten thresholds slightly on subsequent attempts."""
        if session.total_attempts == 0:
            return self._auth_config
        # Each retry tightens SV accept threshold by 0.02, capped at 3 retries
        tightening = min(session.retry_count, 3) * 0.02
        cfg = self._auth_config
        return DecisionConfig(
            sv_accept_threshold=cfg.sv_accept_threshold + tightening,
            sv_retry_threshold=cfg.sv_retry_threshold,
            spoof_reject_threshold=cfg.spoof_reject_threshold,
            spoof_retry_threshold=cfg.spoof_retry_threshold,
            sv_threshold_range=cfg.sv_threshold_range,
            enable_step_up=cfg.enable_step_up,
        )

    def session_summary(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            return {}
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "status": session.status.value,
            "total_attempts": session.total_attempts,
            "retry_count": session.retry_count,
            "is_locked": session.is_locked,
            "attempts": [
                {
                    "attempt": a.attempt_number,
                    "decision": a.decision.value,
                    "speaker_score": a.speaker_score,
                    "spoof_score": a.spoof_score,
                    "explanation": a.explanation,
                }
                for a in session.attempts
            ],
        }
