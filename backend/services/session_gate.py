"""
Single-session admission control with FIFO queue.

Only one browser session is allowed to use the platform at a time. Other
sessions are queued in arrival order; when the active holder releases or
times out, the head of the queue is promoted to active.

The active holder must send a heartbeat every ~30s (timeout 90s ≈ 3 missed
beats). Waiters keep their place in the queue by polling `try_acquire`
every ~3s; if a waiter is not seen for `WAITER_TIMEOUT_S` seconds it is
dropped from the queue, so closed tabs do not block the line.

Replaces the previous FIFO GPU job queue: with only one active user, the
GPU naturally serialises without needing a per-job queue at all.
"""

import threading
import time
from typing import List, Optional


class SessionGate:
    HEARTBEAT_INTERVAL_S = 30
    DEFAULT_TIMEOUT_S = 90  # ~3 missed heartbeats from active holder
    WAITER_TIMEOUT_S = 15   # ~5 missed polls from a waiter

    def __init__(
        self,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        waiter_timeout_s: float = WAITER_TIMEOUT_S,
    ):
        self._lock = threading.Lock()
        self._active_session_id: Optional[str] = None
        self._last_heartbeat_at: float = 0.0
        self._timeout_s = timeout_s
        self._waiter_timeout_s = waiter_timeout_s
        # FIFO queue of waiting session ids; index 0 = next to be promoted.
        self._queue: List[str] = []
        # Last time each waiter was seen polling — used to evict stale waiters.
        self._waiter_last_seen: dict = {}

    # ----- internal helpers (must be called with _lock held) -----

    def _is_expired(self, now: float) -> bool:
        return (now - self._last_heartbeat_at) > self._timeout_s

    def _prune_waiters(self, now: float) -> None:
        alive = []
        for sid in self._queue:
            last = self._waiter_last_seen.get(sid, 0.0)
            if (now - last) <= self._waiter_timeout_s:
                alive.append(sid)
            else:
                self._waiter_last_seen.pop(sid, None)
        self._queue = alive

    def _promote_if_free(self, now: float) -> None:
        if self._active_session_id is not None and not self._is_expired(now):
            return
        # Active slot is empty or its holder timed out — promote next waiter.
        if self._queue:
            next_sid = self._queue.pop(0)
            self._waiter_last_seen.pop(next_sid, None)
            self._active_session_id = next_sid
            self._last_heartbeat_at = now
        else:
            self._active_session_id = None
            self._last_heartbeat_at = 0.0

    def _waiting_payload(self, session_id: str, now: float) -> dict:
        position = self._queue.index(session_id) + 1
        idle_for = now - self._last_heartbeat_at
        remaining = max(0.0, self._timeout_s - idle_for)
        return {
            "status": "waiting",
            "session_id": session_id,
            "queue_position": position,        # 1 = next in line
            "people_ahead": position - 1,
            "queue_length": len(self._queue),
            "active_idle_seconds": round(idle_for, 1),
            "timeout_seconds_remaining": round(remaining, 1),
        }

    # ----- public API -----

    def try_acquire(self, session_id: str) -> dict:
        now = time.time()
        with self._lock:
            # Refresh liveness if the caller is already a waiter.
            if session_id in self._waiter_last_seen:
                self._waiter_last_seen[session_id] = now

            self._prune_waiters(now)
            self._promote_if_free(now)

            # Already (or just) the active holder — renew heartbeat.
            if self._active_session_id == session_id:
                self._last_heartbeat_at = now
                return {
                    "status": "active",
                    "session_id": session_id,
                    "heartbeat_interval_s": self.HEARTBEAT_INTERVAL_S,
                }

            # No active holder and no queue ahead — take the slot.
            if self._active_session_id is None:
                self._active_session_id = session_id
                self._last_heartbeat_at = now
                return {
                    "status": "active",
                    "session_id": session_id,
                    "heartbeat_interval_s": self.HEARTBEAT_INTERVAL_S,
                }

            # Otherwise wait — append to queue if not already there.
            if session_id not in self._waiter_last_seen:
                self._queue.append(session_id)
                self._waiter_last_seen[session_id] = now

            return self._waiting_payload(session_id, now)

    def heartbeat(self, session_id: str) -> dict:
        now = time.time()
        with self._lock:
            if self._active_session_id == session_id and not self._is_expired(now):
                self._last_heartbeat_at = now
                return {"status": "active", "session_id": session_id}
            return {"status": "expired", "session_id": session_id}

    def release(self, session_id: str) -> dict:
        now = time.time()
        with self._lock:
            if self._active_session_id == session_id:
                self._active_session_id = None
                self._last_heartbeat_at = 0.0
                self._promote_if_free(now)
                return {"status": "released"}
            if session_id in self._waiter_last_seen:
                if session_id in self._queue:
                    self._queue.remove(session_id)
                self._waiter_last_seen.pop(session_id, None)
                return {"status": "left_queue"}
            return {"status": "not_holder"}

    def holds(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        now = time.time()
        with self._lock:
            return (
                self._active_session_id == session_id
                and not self._is_expired(now)
            )

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            return {
                "active_session": self._active_session_id,
                "last_heartbeat_at": self._last_heartbeat_at,
                "idle_seconds": (
                    round(now - self._last_heartbeat_at, 1)
                    if self._active_session_id else None
                ),
                "timeout_seconds": self._timeout_s,
                "queue_length": len(self._queue),
            }


SESSION_GATE = SessionGate()
