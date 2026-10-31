"""Build the RAG index from the official documents listed in data/rag_sources/sources.csv.

    python -m app.rag.ingest            (from backend/, downloads missing files)

Output (committed, so the deployed app needs no re-embedding):
    data/index/chunks.json      text + metadata (doc, publisher, page, url, crops, states)
    data/index/embeddings.npy   float16 Gemini embeddings, row i = chunk i
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
from pathlib import Path

import httpx
import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app import config, llm

SOURCES = config.ROOT / "data" / "rag_sources" / "sources.csv"
RAW = config.ROOT / "data" / "rag_sources" / "raw"
CHUNK_CHARS = 1200
OVERLAP = 200
BIG_DOC_PAGES = 100  # in large multi-crop guides keep only pages that mention our crops

CROP_TERMS = {
    "paddy": r"\b(paddy|rice|basmati|samba|kuruvai|thaladi|transplant\w*|puddl\w*)\b",
    "wheat": r"\bwheat\b",
    "mustard": r"\b(mustard|raya|rapeseed|gobhi sarson|toria|canola)\b",
}


def _download(src: dict) -> Path:
    path = RAW / f"{src['id']}.{src['kind']}"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(src["url"], headers={"User-Agent": config.USER_AGENT}, timeout=120,
                         follow_redirects=True)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def _pages(src: dict, path: Path) -> list[tuple[int, str]]:
    if src["kind"] == "html":
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return [(1, soup.get_text(" "))]
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def _split(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):  # prefer to cut at a sentence end
            cut = text.rfind(". ", start + CHUNK_CHARS // 2, end)
            end = cut + 1 if cut != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return chunks


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    with SOURCES.open(encoding="utf-8") as fh:
        sources = list(csv.DictReader(fh))
    for src in sources:
        path = _download(src)
        pages = _pages(src, path)
        doc_crops = src["crops"].split(";")
        kept = 0
        for page_no, raw in pages:
            text = _clean(raw)
            if len(text) < 80:
                continue
            page_crops = [c for c, pat in CROP_TERMS.items() if re.search(pat, text, re.I)]
            if len(pages) > BIG_DOC_PAGES and not set(page_crops) & set(doc_crops):
                continue
            for part in _split(text):
                crops = [c for c, pat in CROP_TERMS.items() if re.search(pat, part, re.I)] or doc_crops
                chunks.append({
                    "id": f"{src['id']}:p{page_no}:{kept}", "doc_id": src["id"], "title": src["title"],
                    "publisher": src["publisher"], "year": src["year"], "page": page_no,
                    "url": src["url"] + (f"#page={page_no}" if src["kind"] == "pdf" else ""),
                    "crops": crops, "states": src["states"].split(";"), "text": part,
                })
                kept += 1
        print(f"{src['id']}: {len(pages)} pages -> {kept} chunks")
    return chunks


async def _embed_all(texts: list[str], batch: int = 20) -> list[list[float]]:
    """Embed with pacing for free-tier quotas; progress is checkpointed so a rerun resumes."""
    progress = RAW / "embeddings.partial.json"
    done: list[list[float]] = json.loads(progress.read_text()) if progress.exists() else []
    if len(done) > len(texts):
        done = []
    wait = 5.0
    while len(done) < len(texts):
        try:
            done.extend(await llm.embed(texts[len(done):len(done) + batch], batch=batch))
            progress.write_text(json.dumps(done))
            print(f"  embedded {len(done)}/{len(texts)}", flush=True)
            wait = 5.0
            await asyncio.sleep(1.0)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code != 429:
                raise
            print(f"  {type(exc).__name__}, waiting {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
            wait = min(wait * 2, 65.0)
    progress.unlink(missing_ok=True)
    return done


async def main() -> None:
    chunks = build_chunks()
    vectors = await _embed_all([f"{c['title']}. {c['text']}" for c in chunks])
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    matrix = np.asarray(vectors, dtype="float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(config.INDEX_DIR / "embeddings.npy", matrix.astype("float16"))
    print(f"indexed {len(chunks)} chunks, dim {matrix.shape[1]} -> {config.INDEX_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
