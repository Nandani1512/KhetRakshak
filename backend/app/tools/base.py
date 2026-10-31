"""Tool contract shared by every data tool.

Every tool returns a ToolResult and never raises: on failure it tries the next fallback,
then a *labelled* cache of a previous real response, then an explicit "data unavailable".
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel, Field

from app import config

# Sources the operator has switched off from the admin panel (demo of graceful degradation).
DISABLED_SOURCES: set[str] = set()


class ToolResult(BaseModel):
    data: Any | None = None
    source: str
    source_url: str = ""
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = 1.0
    error: str | None = None
    cached: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.data is not None and self.error is None


def redact(text: str) -> str:
    for secret in config.SECRETS:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


class SourceDisabled(Exception):
    pass


_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    """Shared async client. data.gov.in stalls httpx's default User-Agent, so we always send ours."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(25.0, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": config.USER_AGENT},
        )
    return _client


async def http(method: str, url: str, *, source: str, retries: int = 3, backoff: float = 1.5,
               **kwargs) -> httpx.Response:
    """HTTP with retries + exponential backoff on network errors, 429 and 5xx."""
    if source in DISABLED_SOURCES:
        raise SourceDisabled(f"{source} disabled by operator")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = await client().request(method, url, **kwargs)
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                return resp
            last = httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
        except httpx.TransportError as exc:
            last = exc
        if attempt < retries:
            await asyncio.sleep(backoff ** attempt)
    assert last is not None
    raise last


def describe_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return redact(f"HTTP {exc.response.status_code}: {exc.response.text[:160]}")
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    return redact(f"{type(exc).__name__}: {exc}")


Attempt = Callable[[], Awaitable[ToolResult]]


async def run_with_fallbacks(name: str, attempts: list[tuple[str, Attempt]],
                             cache_key: str | None = None) -> ToolResult:
    """Try each (source, coroutine) in order; then labelled cache; then 'data unavailable'.

    Successful results are written to the cache so a later outage can fall back to them.
    """
    from app import db  # local import: db is optional at import time

    errors: list[str] = []
    for source, attempt in attempts:
        started = time.perf_counter()
        try:
            result = await attempt()
            result.notes.extend(f"fallback after: {e}" for e in errors)
            result.notes.append(f"{source} answered in {int((time.perf_counter() - started) * 1000)} ms")
            if cache_key and result.ok:
                await db.cache_put(cache_key, result)
            return result
        except Exception as exc:  # noqa: BLE001 - tools must never raise
            errors.append(f"{source}: {describe_error(exc)}")

    if cache_key:
        cached = await db.cache_get(cache_key)
        if cached is not None:
            cached.cached = True
            cached.confidence = round(cached.confidence * 0.7, 2)
            cached.notes = [f"live sources failed ({'; '.join(errors)})",
                            f"using cached response from {cached.source} at {cached.fetched_at}"]
            return cached

    return ToolResult(source=name, error="data unavailable: " + "; ".join(errors), confidence=0.0)
