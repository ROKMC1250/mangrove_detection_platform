"""
Session gate endpoints.

Clients call `/api/session/acquire` on page load. If the gate is free,
status=active is returned and the frontend boots; if another session holds
it, status=waiting is returned and the frontend shows a full-screen waiting
overlay while polling until granted. Active holders must `/heartbeat`
regularly to keep the gate; `/release` drops it on tab close.
"""

import uuid

from fastapi import APIRouter, Request, Response

from ..services.session_gate import SESSION_GATE


router = APIRouter(prefix="/api/session", tags=["session"])


SESSION_COOKIE = "mangrove_session_id"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _ensure_session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie(
            SESSION_COOKIE, sid,
            httponly=False,
            samesite="lax",
            max_age=_COOKIE_MAX_AGE,
            path="/",
        )
    return sid


@router.post("/acquire")
def acquire(request: Request, response: Response):
    sid = _ensure_session_id(request, response)
    return SESSION_GATE.try_acquire(sid)


@router.post("/heartbeat")
def heartbeat(request: Request, response: Response):
    sid = _ensure_session_id(request, response)
    return SESSION_GATE.heartbeat(sid)


@router.post("/release")
def release(request: Request, response: Response):
    sid = _ensure_session_id(request, response)
    return SESSION_GATE.release(sid)


@router.get("/status")
def status(request: Request, response: Response):
    sid = _ensure_session_id(request, response)
    return {
        "gate": SESSION_GATE.status(),
        "you": {
            "session_id": sid,
            "holds_gate": SESSION_GATE.holds(sid),
        },
    }
