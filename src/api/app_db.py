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
import os
import sqlite3
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
