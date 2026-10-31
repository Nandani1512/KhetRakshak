"""Hybrid retrieval over the official-document index.

metadata filter (crop, state) -> BM25 keyword ranking + embedding cosine ranking
-> reciprocal-rank fusion -> top-k. Chunks below a cosine floor are dropped, so a question
with no supporting document gets *no* sources instead of loosely related ones.
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
    return {"chunks": len(chunks), "documents": len({c["doc_id"] for c in chunks}), "dim": vectors.shape[1]}


async def retrieve(query: str, crop: str | None, state: str | None, k: int = 6) -> list[dict]:
    chunks, vectors = _index()
    allowed = [i for i, c in enumerate(chunks)
               if (crop is None or crop in c["crops"])
               and (state is None or "all" in c["states"] or state in c["states"])]
    if not allowed:
        return []

    qvec = np.asarray((await llm.embed([query]))[0], dtype="float32")
    qvec /= np.linalg.norm(qvec)
    cos = vectors[allowed] @ qvec

    bm25 = BM25Okapi([_tokens(chunks[i]["text"]) for i in allowed])
    kw = bm25.get_scores(_tokens(query))

    fused = np.zeros(len(allowed))
    for ranking in (np.argsort(-cos), np.argsort(-kw)):
        for rank, j in enumerate(ranking[:50]):
            fused[j] += 1.0 / (RRF_K + rank + 1)

    out = []
    for j in np.argsort(-fused):
        if fused[j] == 0 or len(out) == k:
            break
        if cos[j] < MIN_COSINE:
            continue
        c = chunks[allowed[j]]
        out.append({**c, "score": round(float(fused[j]), 4), "cosine": round(float(cos[j]), 3)})
    return out
