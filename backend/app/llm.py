"""Gemini client over Google's OpenAI-compatible endpoint.

chat()  : retries, then falls back to a second model; raises LLMUnavailable only if both fail
          (callers then switch to a deterministic, data-only alert).
embed() : batched embeddings for RAG.
"""
from __future__ import annotations

import asyncio
import json
import re
import time

import httpx

from app import config
from app.tools.base import describe_error, http

stats = {"calls": 0, "fallbacks": 0, "last_latency_ms": None, "last_error": None}


class LLMUnavailable(RuntimeError):
    pass


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.GEMINI_API_KEY}"}


async def chat(messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False,
               temperature: float = 0.2, max_tokens: int = 4096) -> dict:
    """Return the assistant message dict ({content, tool_calls?}) and the model that answered."""
    errors = []
    for model in dict.fromkeys([config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL]):
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "reasoning_effort": "low"}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            try:
                resp = await http("POST", f"{config.GEMINI_BASE_URL}/chat/completions", source="Gemini",
                                  headers=_headers(), json=body, timeout=90.0, retries=2)
            except httpx.HTTPStatusError as exc:
                # Per-minute quota: wait for the window to roll over once (daily quota: give up).
                if exc.response.status_code != 429 or re.search(r"per ?day", exc.response.text, re.I):
                    raise
                await asyncio.sleep(12)
                resp = await http("POST", f"{config.GEMINI_BASE_URL}/chat/completions", source="Gemini",
                                  headers=_headers(), json=body, timeout=90.0, retries=1)
            msg = resp.json()["choices"][0]["message"]
            stats.update(calls=stats["calls"] + 1, last_latency_ms=int((time.perf_counter() - started) * 1000),
                         last_error=None)
            msg["model"] = model
            return msg
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model}: {describe_error(exc)}")
            stats["fallbacks"] += 1
    stats["last_error"] = "; ".join(errors)
    raise LLMUnavailable(stats["last_error"])


def parse_json(text: str | None) -> dict:
    """Parse model JSON, tolerating ```json fences or leading prose."""
    if not text:
        raise ValueError("empty model response")
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start:end + 1])


async def chat_json(system: str, user: str, *, temperature: float = 0.2) -> tuple[dict, str]:
    msg = await chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     json_mode=True, temperature=temperature)
    return parse_json(msg.get("content")), msg["model"]


async def embed(texts: list[str], batch: int = 50) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = await http("POST", f"{config.GEMINI_BASE_URL}/embeddings", source="Gemini",
                          headers=_headers(), timeout=90.0,
                          json={"model": config.GEMINI_EMBED_MODEL, "input": texts[i:i + batch],
                                "dimensions": 768})
        out.extend(d["embedding"] for d in resp.json()["data"])  # returned in input order
    return out
