"""Persistence on Supabase Postgres.

Sync psycopg pool run in threads (psycopg's async mode does not work on Windows' default loop).
If the database is unreachable the app keeps working from memory and /health says so.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app import config

log = logging.getLogger("khetrakshak.db")

_pool = None
_memory: dict[str, dict] = {"fields": {}, "runs": {}, "events": {}, "alerts": {}, "cache": {}}
status = {"backend": "memory", "error": None}


def _json_default(o: Any) -> str:
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(type(o))


def _dumps(o: Any) -> str:
    return json.dumps(o, default=_json_default, ensure_ascii=False)


def _safe_dsn(url: str) -> str:
    """Percent-encode the password so characters like '@', '#', '/' in it don't break parsing."""
    scheme, sep, rest = url.partition("://")
    if not sep or "@" not in rest:
        return url
    userinfo, host = rest.rsplit("@", 1)
    user, _, password = userinfo.partition(":")
    if "%" in password:  # already encoded
        return url
    return f"{scheme}://{user}:{quote(password, safe='')}@{host}"


def init() -> None:
    global _pool
    if not config.DATABASE_URL:
        status["error"] = "DATABASE_URL not set; using in-memory store"
        return
    pool = None
    try:
        from psycopg_pool import ConnectionPool

        # check= validates a connection before handing it out: Supabase's pooler drops idle ones.
        pool = ConnectionPool(_safe_dsn(config.DATABASE_URL), min_size=1, max_size=5, timeout=15,
                              max_idle=120, check=ConnectionPool.check_connection,
                              kwargs={"autocommit": True, "prepare_threshold": None}, open=False)
        pool.open(wait=True, timeout=15)
        with pool.connection() as conn:
            conn.execute((Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))
        _pool = pool
        status.update(backend="supabase-postgres", error=None)
    except Exception as exc:  # noqa: BLE001
        if pool is not None:
            pool.close()
        status["error"] = f"database unavailable ({type(exc).__name__}); using in-memory store"
        log.warning(status["error"])


def _query(sql: str, params: tuple = (), fetch: str | None = None):
    import psycopg

    try:
        return _query_once(sql, params, fetch)
    except psycopg.OperationalError:  # connection died mid-query: retry once on a fresh one
        return _query_once(sql, params, fetch)


def _query_once(sql: str, params: tuple, fetch: str | None):
    with _pool.connection() as conn:
        cur = conn.execute(sql, params)
        if fetch == "one":
            row = cur.fetchone()
            return dict(zip([c.name for c in cur.description], row)) if row else None
        if fetch == "all":
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        return None


async def q(sql: str, params: tuple = (), fetch: str | None = None):
    return await asyncio.to_thread(_query, sql, params, fetch)


# ------------------------------------------------------------------ fields

FIELD_COLS = ("id", "name", "farmer_name", "polygon", "centroid_lat", "centroid_lon", "area_ha", "crop",
              "sowing_date", "language", "state", "district", "transport_rate_per_qtl_km")


async def create_field(f: dict) -> dict:
    if _pool is None:
        _memory["fields"][f["id"]] = {**f, "telegram_chat_id": None}
        return _memory["fields"][f["id"]]
    vals = tuple(_dumps(f[c]) if c == "polygon" else f.get(c) for c in FIELD_COLS)
    await q(f"insert into fields ({','.join(FIELD_COLS)}) values ({','.join(['%s'] * len(FIELD_COLS))})", vals)
    return await get_field(f["id"])


def _norm_field(row: dict | None) -> dict | None:
    if row and isinstance(row.get("sowing_date"), date):
        row["sowing_date"] = row["sowing_date"].isoformat()
    return row


async def get_field(field_id: str) -> dict | None:
    if _pool is None:
        return _memory["fields"].get(field_id)
    return _norm_field(await q("select * from fields where id=%s", (field_id,), "one"))


async def list_fields() -> list[dict]:
    if _pool is None:
        return list(_memory["fields"].values())
    return [_norm_field(r) for r in await q("select * from fields order by created_at desc", (), "all")]


async def delete_field(field_id: str) -> None:
    if _pool is None:
        _memory["fields"].pop(field_id, None)
        return
    await q("delete from fields where id=%s", (field_id,))


async def link_telegram(field_id: str, chat_id: int) -> dict | None:
    if _pool is None:
        if field_id in _memory["fields"]:
            _memory["fields"][field_id]["telegram_chat_id"] = chat_id
        return _memory["fields"].get(field_id)
    await q("update fields set telegram_chat_id=%s where id=%s", (chat_id, field_id))
    return await get_field(field_id)


async def field_for_chat(chat_id: int) -> dict | None:
    if _pool is None:
        return next((f for f in _memory["fields"].values() if f.get("telegram_chat_id") == chat_id), None)
    return _norm_field(await q("select * from fields where telegram_chat_id=%s order by created_at desc limit 1",
                               (chat_id,), "one"))


# ------------------------------------------------------------------ runs + events

async def create_run(run: dict) -> None:
    if _pool is None:
        _memory["runs"][run["id"]] = run
        _memory["events"][run["id"]] = []
        return
    await q("insert into runs (id, field_id, mode, as_of_date, question) values (%s,%s,%s,%s,%s)",
            (run["id"], run["field_id"], run["mode"], run.get("as_of_date"), run.get("question")))


async def finish_run(run_id: str, status_: str, result: dict | None, critic_loops: int) -> None:
    if _pool is None:
        _memory["runs"].get(run_id, {}).update(status=status_, result=result, critic_loops=critic_loops)
        return
    await q("update runs set status=%s, result=%s, critic_loops=%s, finished_at=now() where id=%s",
            (status_, _dumps(result), critic_loops, run_id))


async def add_event(run_id: str, ev: dict) -> None:
    if _pool is None:
        _memory["events"].setdefault(run_id, []).append(ev)
        return
    await q("insert into run_events (run_id, seq, node, status, message, data) values (%s,%s,%s,%s,%s,%s)",
            (run_id, ev["seq"], ev["node"], ev["status"], ev["message"], _dumps(ev.get("data"))))


async def get_run(run_id: str) -> dict | None:
    if _pool is None:
        run = _memory["runs"].get(run_id)
        return {**run, "events": _memory["events"].get(run_id, [])} if run else None
    run = await q("select * from runs where id=%s", (run_id,), "one")
    if run:
        run["events"] = await q("select seq, node, status, message, data, ts from run_events "
                                "where run_id=%s order by seq", (run_id,), "all")
    return run


async def list_runs(field_id: str, limit: int = 10) -> list[dict]:
    if _pool is None:
        return [r for r in _memory["runs"].values() if r["field_id"] == field_id][-limit:]
    return await q("select id, mode, as_of_date, question, status, critic_loops, started_at, finished_at "
                   "from runs where field_id=%s order by started_at desc limit %s", (field_id, limit), "all")


async def save_alert(a: dict) -> None:
    if _pool is None:
        _memory["alerts"].setdefault(a["field_id"], []).append(a)
        return
    await q("insert into alerts (run_id, field_id, language, text_en, text_local, citations, risk_types, delivered) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s)",
            (a["run_id"], a["field_id"], a["language"], a["text_en"], a.get("text_local"),
             _dumps(a.get("citations")), a.get("risk_types"), a.get("delivered", False)))


async def recent_alert_risks(field_id: str, days: int = 3) -> set[str]:
    """Risk types already delivered recently, so the scheduler does not repeat itself."""
    if _pool is None:
        return set()
    rows = await q("select risk_types from alerts where field_id=%s and delivered and "
                   "created_at > now() - make_interval(days => %s)", (field_id, days), "all")
    return {t for r in rows for t in (r["risk_types"] or [])}


# ------------------------------------------------------------------ labelled cache

async def cache_put(key: str, result) -> None:
    payload = result.model_dump()
    try:
        if _pool is None:
            _memory["cache"][key] = payload
            return
        await q("insert into api_cache (key, payload, fetched_at) values (%s,%s,now()) "
                "on conflict (key) do update set payload=excluded.payload, fetched_at=now()",
                (key, _dumps(payload)))
    except Exception as exc:  # noqa: BLE001 - caching must never break a tool
        log.warning("cache_put failed: %s", exc)


async def cache_get(key: str):
    from app.tools.base import ToolResult

    try:
        if _pool is None:
            payload = _memory["cache"].get(key)
        else:
            row = await q("select payload from api_cache where key=%s", (key,), "one")
            payload = row["payload"] if row else None
    except Exception as exc:  # noqa: BLE001
        log.warning("cache_get failed: %s", exc)
        return None
    return ToolResult(**payload) if payload else None
