"""Live agent trace: every node emits events that are streamed (SSE) and persisted."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app import db

log = logging.getLogger("khetrakshak.events")

_events: dict[str, list[dict]] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}
_done: set[str] = set()
artifacts: dict[str, dict[str, Any]] = {}   # run_id -> {"audio": bytes, ...}


async def emit(run_id: str, node: str, status: str, message: str, data: Any = None) -> None:
    ev = {"seq": len(_events.setdefault(run_id, [])) + 1, "node": node, "status": status,
          "message": message, "data": data, "ts": datetime.now(timezone.utc).isoformat()}
    _events[run_id].append(ev)
    for q in _subscribers.get(run_id, []):
        q.put_nowait(ev)
    try:
        await db.add_event(run_id, ev)
    except Exception as exc:  # noqa: BLE001 - tracing must never break a run
        log.warning("event persist failed: %s", exc)


def close(run_id: str) -> None:
    _done.add(run_id)
    for q in _subscribers.get(run_id, []):
        q.put_nowait(None)


async def stream(run_id: str):
    """Yield past events, then live ones, until the run closes."""
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(run_id, []).append(q)
    try:
        for ev in list(_events.get(run_id, [])):
            yield ev
        if run_id in _done:
            return
        while True:
            ev = await q.get()
            if ev is None:
                return
            yield ev
    finally:
        _subscribers[run_id].remove(q)


def is_known(run_id: str) -> bool:
    return run_id in _events
