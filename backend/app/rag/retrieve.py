"""Hybrid retrieval over the official-document index.

metadata filter (crop, state) -> BM25 keyword ranking + embedding cosine ranking
-> reciprocal-rank fusion -> top-k. Chunks below a cosine floor are dropped, so a question
with no supporting document gets *no* sources instead of loosely related ones.

Degrades to keyword-only (BM25) if the embedding API is unavailable; chunks not yet embedded
are still found by keywords.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app import config, llm

RRF_K = 60
MIN_COSINE = 0.55
last_mode = {"mode": None, "error": None}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict], np.ndarray]:
    chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(config.INDEX_DIR / "embeddings.npy").astype("float32")
    return chunks, vectors


def index_stats() -> dict:
    try:
        chunks, vectors = _index()
    except FileNotFoundError:
        return {"chunks": 0, "documents": 0, "error": "index not built"}
    embedded = int((np.abs(vectors).sum(axis=1) > 0).sum())
    return {"chunks": len(chunks), "documents": len({c["doc_id"] for c in chunks}), "dim": vectors.shape[1],
            "embedded": embedded, "last_mode": last_mode["mode"]}


async def retrieve(query: str, crop: str | None, state: str | None, k: int = 6) -> list[dict]:
    chunks, vectors = _index()
    allowed = [i for i, c in enumerate(chunks)
               if (crop is None or crop in c["crops"])
               and (state is None or "all" in c["states"] or state in c["states"])]
    if not allowed:
        return []

    bm25 = BM25Okapi([_tokens(chunks[i]["text"]) for i in allowed])
    kw = bm25.get_scores(_tokens(query))

    cos = None
    try:
        qvec = np.asarray((await llm.embed([query]))[0], dtype="float32")
        cos = vectors[allowed] @ (qvec / np.linalg.norm(qvec))
        last_mode.update(mode="hybrid", error=None)
    except Exception as exc:  # noqa: BLE001 - embeddings down: keyword-only retrieval
        last_mode.update(mode="keyword-only", error=type(exc).__name__)

    fused = np.zeros(len(allowed))
    rankings = [np.argsort(-kw)]
    if cos is not None:
        rankings.append(np.argsort(-cos))
    for ranking in rankings:
        for rank, j in enumerate(ranking[:50]):
            fused[j] += 1.0 / (RRF_K + rank + 1)

    out = []
    for j in np.argsort(-fused):
        if fused[j] == 0 or len(out) == k:
            break
        has_vec = cos is not None and vectors[allowed[j]].any()
        if has_vec and cos[j] < MIN_COSINE:
            continue
        if not has_vec and kw[j] <= 0:
            continue
        c = chunks[allowed[j]]
        out.append({**c, "score": round(float(fused[j]), 4),
                    "cosine": round(float(cos[j]), 3) if has_vec else None,
                    "bm25": round(float(kw[j]), 2), "retrieval": last_mode["mode"]})
    return out
