"""Optional JWT bearer-token auth. Disabled unless AUTH_ENABLED=true + AUTH_JWT_SECRET are set."""

from __future__ import annotations

import os

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def auth_enabled() -> bool:
    return os.environ.get("AUTH_ENABLED", "false").lower() == "true"


async def verify_token(request: Request) -> dict | None:
    """FastAPI dependency: no-op unless AUTH_ENABLED=true, then requires a valid JWT."""
    if not auth_enabled():
        return None

    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    secret = os.environ.get("AUTH_JWT_SECRET")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_ENABLED=true but AUTH_JWT_SECRET is not configured",
        )

    try:
        return jwt.decode(credentials.credentials, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}"
        ) from exc
