"""
SQLite store for app users (login credentials + roles).

Separate from the voice enrollment DB — this holds the web-app accounts,
not the speaker embeddings.

Roles:
  admin  — full access: enroll anyone, delete, view all sessions
  ops    — read-only access to all sessions
  user   — enroll/authenticate as themselves only
"""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/app.db")


def _hash(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return binascii.hexlify(salt).decode() + ":" + binascii.hexlify(key).decode()


def _verify(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":")
        salt = binascii.unhexlify(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return binascii.hexlify(key).decode() == key_hex
    except Exception:
        return False


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(admin_username: str, admin_password: str) -> None:
    """Create tables and upsert the admin account from config."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions_log (
                session_id     TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                status         TEXT NOT NULL,
                challenge      TEXT,
                created_at     REAL,
                completed_at   REAL,
                total_attempts INTEGER DEFAULT 0,
                retry_count    INTEGER DEFAULT 0,
                is_locked      INTEGER DEFAULT 0,
                attempts_json  TEXT DEFAULT '[]'
            )
        """)
        row = conn.execute(
            "SELECT id FROM app_users WHERE username = ?", (admin_username,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE app_users SET password_hash = ? WHERE username = ?",
                (_hash(admin_password), admin_username),
            )
        else:
            conn.execute(
                "INSERT INTO app_users (username, password_hash, role) VALUES (?, ?, ?)",
                (admin_username, _hash(admin_password), "admin"),
            )


def verify_user(username: str, password: str) -> dict | None:
    """Return {username, role} if credentials are valid, else None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT username, password_hash, role FROM app_users WHERE username = ?",
            (username,),
        ).fetchone()
    if row and _verify(password, row["password_hash"]):
        return {"username": row["username"], "role": row["role"]}
    return None


def create_user(username: str, password: str, role: str = "user") -> None:
    """Create a new app user. Raises ValueError if username already exists."""
    with _conn() as conn:
        try:
            conn.execute(
                "INSERT INTO app_users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, _hash(password), role),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists.")


def list_app_users() -> list[dict]:
    """Return all app users (without password hashes)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT username, role FROM app_users ORDER BY username"
        ).fetchall()
    return [{"username": r["username"], "role": r["role"]} for r in rows]


def update_role(username: str, role: str) -> None:
    """Change a user's role. Raises ValueError if user not found."""
    if role not in ("admin", "ops", "user"):
        raise ValueError(f"Invalid role: {role}")
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE app_users SET role = ? WHERE username = ?", (role, username)
        )
        if cur.rowcount == 0:
            raise ValueError(f"User '{username}' not found.")


def delete_app_user(username: str) -> None:
    """Delete an app user. Raises ValueError if user not found."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM app_users WHERE username = ?", (username,)
        )
        if cur.rowcount == 0:
            raise ValueError(f"User '{username}' not found.")


# ── session log ───────────────────────────────────────────────────────────────

# ── audit log ─────────────────────────────────────────────────────────────────

def init_audit_log() -> None:
    """Create audit_log table if not exists. Called at server startup."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                actor     TEXT NOT NULL,
                action    TEXT NOT NULL,
                target    TEXT,
                details   TEXT
            )
        """)


def log_audit(actor: str, action: str, target: str = None, details: str = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (timestamp, actor, action, target, details) VALUES (?, ?, ?, ?, ?)",
            (time.time(), actor, action, target, details),
        )


def list_audit_logs(limit: int = 500) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, actor, action, target, details FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": r["id"], "timestamp": r["timestamp"], "actor": r["actor"],
             "action": r["action"], "target": r["target"], "details": r["details"]}
            for r in rows]


def log_session(summary: dict) -> None:
    """Persist a completed session summary to SQLite. Upserts on session_id."""
    with _conn() as conn:
        conn.execute("""
            INSERT INTO sessions_log
                (session_id, user_id, status, challenge, created_at, completed_at,
                 total_attempts, retry_count, is_locked, attempts_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status         = excluded.status,
                completed_at   = excluded.completed_at,
                total_attempts = excluded.total_attempts,
                retry_count    = excluded.retry_count,
                is_locked      = excluded.is_locked,
                attempts_json  = excluded.attempts_json
        """, (
            summary["session_id"],
            summary["user_id"],
            summary["status"],
            summary.get("challenge"),
            summary.get("created_at"),
            time.time(),
            summary.get("total_attempts", 0),
            summary.get("retry_count", 0),
            1 if summary.get("is_locked") else 0,
            json.dumps(summary.get("attempts", [])),
        ))


def list_session_logs() -> list[dict]:
    """Return all logged sessions, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions_log ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        result.append({
            "session_id":     r["session_id"],
            "user_id":        r["user_id"],
            "status":         r["status"],
            "challenge":      r["challenge"],
            "created_at":     r["created_at"],
            "completed_at":   r["completed_at"],
            "total_attempts": r["total_attempts"],
            "retry_count":    r["retry_count"],
            "is_locked":      bool(r["is_locked"]),
            "attempts":       json.loads(r["attempts_json"] or "[]"),
        })
    return result


# ── model tests (admin Model Testing feature) ─────────────────────────────────

# Columns shared by INSERT and row-reads (mirror of the model_tests schema, minus id).
# voice_model = TTS engine/model (e.g. "speech-2.8-hd"); voice = speaker (e.g. "Calm Woman").
_MODEL_TEST_COLUMNS = (
    "test_id", "model_type", "filename", "voice_model", "voice", "language", "notes",
    "predicted_label", "confidence", "deepfake_prob", "genuine_prob",
    "duration_ms", "inference_ms", "tested_by", "tested_at", "raw_output",
)


def init_model_tests() -> None:
    """Create the model_tests table if it doesn't exist. Called at server startup."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_tests (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id         TEXT UNIQUE NOT NULL,
                model_type      TEXT NOT NULL,
                filename        TEXT NOT NULL,
                voice_model     TEXT,
                voice           TEXT,
                language        TEXT,
                notes           TEXT,
                predicted_label TEXT,
                confidence      REAL,
                deepfake_prob   REAL,
                genuine_prob    REAL,
                duration_ms     INTEGER,
                inference_ms    INTEGER,
                tested_by       TEXT NOT NULL,
                tested_at       TEXT NOT NULL,
                raw_output      TEXT
            )
        """)
        # Migrate DBs created before the `voice` column existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(model_tests)").fetchall()}
        if "voice" not in cols:
            conn.execute("ALTER TABLE model_tests ADD COLUMN voice TEXT")


def insert_model_test(record: dict) -> None:
    """Insert one model-test result. record keys must cover _MODEL_TEST_COLUMNS."""
    cols = ", ".join(_MODEL_TEST_COLUMNS)
    placeholders = ", ".join("?" for _ in _MODEL_TEST_COLUMNS)
    values = tuple(record.get(c) for c in _MODEL_TEST_COLUMNS)
    with _conn() as conn:
        conn.execute(
            f"INSERT INTO model_tests ({cols}) VALUES ({placeholders})", values
        )


def _row_to_model_test(r) -> dict:
    return {k: r[k] for k in ("id", *_MODEL_TEST_COLUMNS)}


def _model_test_where(model_type=None, date_from=None, date_to=None,
                      tested_by=None, voice_model=None, voice=None) -> tuple[str, list]:
    """Build a WHERE clause + params from optional filters. Dates are ISO 8601 strings."""
    clauses, params = [], []
    if model_type:
        clauses.append("model_type = ?"); params.append(model_type)
    if date_from:
        clauses.append("tested_at >= ?"); params.append(date_from)
    if date_to:
        clauses.append("tested_at <= ?"); params.append(date_to)
    if tested_by:
        clauses.append("tested_by = ?"); params.append(tested_by)
    if voice_model:
        clauses.append("voice_model = ?"); params.append(voice_model)
    if voice:
        clauses.append("voice = ?"); params.append(voice)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def list_model_tests(model_type=None, date_from=None, date_to=None, tested_by=None,
                     voice_model=None, voice=None, limit=50, offset=0) -> list[dict]:
    """Return model-test results matching the filters, newest first, paginated."""
    where, params = _model_test_where(model_type, date_from, date_to, tested_by, voice_model, voice)
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM model_tests{where} ORDER BY tested_at DESC, id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return [_row_to_model_test(r) for r in rows]


def count_model_tests(model_type=None, date_from=None, date_to=None,
                      tested_by=None, voice_model=None, voice=None) -> int:
    """Total number of model-test results matching the filters (for pagination)."""
    where, params = _model_test_where(model_type, date_from, date_to, tested_by, voice_model, voice)
    with _conn() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM model_tests{where}", params
        ).fetchone()
    return int(row["n"])


def delete_model_test(test_id: str) -> bool:
    """Delete one model-test result by test_id. Returns True if a row was removed."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM model_tests WHERE test_id = ?", (test_id,))
    return cur.rowcount > 0
