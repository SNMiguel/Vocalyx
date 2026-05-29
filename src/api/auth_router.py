"""
JWT authentication for the Vocalyx web dashboard.

Endpoints:
  POST /auth/login   — exchange credentials for a JWT
  GET  /auth/me      — return current user info from token

Dependency:
  get_current_user   — FastAPI dependency; raises 401 if token missing/invalid
  require_role([...])— factory for role-checking dependencies
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ── runtime config (set by server lifespan via configure()) ───────────────────

_SECRET_KEY: str = "change-me"
_ALGORITHM: str = "HS256"
_EXPIRE_MINUTES: int = 60


def configure(secret_key: str, expire_minutes: int = 60) -> None:
    global _SECRET_KEY, _EXPIRE_MINUTES
    _SECRET_KEY = secret_key
    _EXPIRE_MINUTES = expire_minutes


# ── router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["Auth"])
_bearer = HTTPBearer(auto_error=False)


# ── models ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class UpdateRoleRequest(BaseModel):
    role: str


# ── helpers ───────────────────────────────────────────────────────────────────

def _create_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": username, "role": role, "exp": expire},
        _SECRET_KEY,
        algorithm=_ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])


# ── dependency ────────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = _decode_token(credentials.credentials)
        return {"username": payload["sub"], "role": payload["role"]}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(*roles: str):
    """Dependency factory: raise 403 if the authenticated user's role is not in roles."""
    async def _check(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {' or '.join(roles)}",
            )
        return current_user
    return _check


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Exchange username + password for a JWT."""
    from src.api.app_db import verify_user
    user = verify_user(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = _create_token(user["username"], user["role"])
    return TokenResponse(
        access_token=token,
        username=user["username"],
        role=user["role"],
    )


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's info."""
    return current_user


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    """Public endpoint — create a new account with the default 'user' role."""
    from src.api.app_db import create_user
    try:
        create_user(body.username, body.password, role="user")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return {"username": body.username, "role": "user"}


@router.get("/users", dependencies=[Depends(require_role("admin"))])
async def list_users():
    """Admin only — list all app accounts."""
    from src.api.app_db import list_app_users
    return {"users": list_app_users()}


@router.patch("/users/{username}")
async def change_role(
    username: str,
    body: UpdateRoleRequest,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin only — change a user's role."""
    from src.api.app_db import update_role, log_audit
    try:
        update_role(username, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    log_audit(current_user["username"], "change_role", username, f"role={body.role}")
    return {"username": username, "role": body.role}


@router.delete("/users/{username}", status_code=204)
async def remove_user(
    username: str,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin only — delete an app account."""
    from src.api.app_db import delete_app_user, log_audit
    try:
        delete_app_user(username)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    log_audit(current_user["username"], "delete_account", username)
