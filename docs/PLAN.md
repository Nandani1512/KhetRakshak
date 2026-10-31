# KhetRakshak — 48-Hour Build Plan

Roles: **A** = agent / LLM / RAG / guardrails · **B** = satellite / weather / risk engine · **C** = frontend + deployment · **D** = voice / Telegram / market + README / video / deck.
3-person team: C takes D's market + Telegram work, A takes README, everyone does video and deck in Phase 5.

`H` = hours since hackathon start. Each phase ends with an **exit gate**. If a gate is not green at its deadline, apply the **cut list** (bottom of this file) before continuing.

---

## 0. Key decisions (locked)

| # | Decision | Why |
|---|---|---|
| D1 | **Backend on Hugging Face Spaces (Docker)**, Render as backup | Free tier has 16 GB RAM and no 15-minute sleep. Render's free tier sleeps (30 s cold start = dead demo) and has 512 MB RAM (raster reads can run out of memory). |
| D2 | **pgvector on Supabase** instead of Chroma | One database for everything, persists across redeploys (HF Space disks are wiped on restart). |
| D3 | Embeddings: **`intfloat/multilingual-e5-small`** run locally | Free, works for Hindi/Tamil queries, ~120 MB, no extra API key. |
| D4 | **Crops: paddy, wheat, mustard** only. **States: Punjab + Tamil Nadu.** | Demo is in late Oct/Nov: Punjab = paddy harvest + wheat/mustard sowing; Tamil Nadu = samba paddy under the north-east monsoon (real heavy-rain risk right now). Other crops → explicit "unsupported crop, contact your KVK". |
| D5 | **Languages: Hindi, Tamil, Punjabi, English** | Matches the two states; Sarvam supports all four. |
| D6 | **Historical replay mode** ("as-of date") | Runs the same agent on a past date using *real archived* weather + satellite data (e.g., the March 2022 heatwave on Punjab wheat). Clearly labelled "Historical replay". Guarantees a high-risk demo, doubles as backtesting evidence. Not dummy data. |
| D7 | Risk thresholds live in `backend/app/risk/rules.yaml`, **each with a citation** (doc, page, URL) | Configuration taken from cited ICAR/PAU/TNAU documents, not invented numbers. Judges can trace every threshold. |
| D8 | Satellite reads = **windowed COG reads of the field bbox only** (rasterio), not full scenes | ~1 MB per scene instead of 200 MB; NDVI time series in seconds. |
| D9 | Streaming agent trace = **Server-Sent Events** (FastAPI → Next.js) | Simplest reliable live stream; works on Vercel + HF. |
| D10 | Scheduler = **APScheduler inside the backend** + "Run monitoring now" button; GitHub Actions cron pings `/health` every 10 min as a keep-warm | No extra infra. |

---

## 1. System design (reference for all phases)

### 1.1 Repo layout
```
backend/
  app/
    main.py                 FastAPI app, routers, scheduler start
    config.py               settings from env (pydantic-settings)
    api/                    fields.py, runs.py (SSE), telegram.py (webhook), health.py
    tools/
      base.py               ToolResult contract + retry/timeout/cache decorator
      weather.py            Open-Meteo forecast + archive
      satellite.py          Sentinel-2 STAC search, SCL cloud mask, NDVI over polygon
      market.py             Agmarknet (+ CEDA fallback), mandi geocode
      routing.py            OSRM (+ ORS fallback)
      geo.py                Nominatim: inside-India / sea checks, polygon validation
      voice.py              Sarvam translate / TTS / STT
    risk/
      crop_stage.py         growing-degree-day crop stage from sowing date
      engine.py             rule evaluation → scored, cited risks
      rules.yaml            thresholds + citations
    rag/
      ingest.py             PDF → chunks with metadata → embeddings → pgvector
      retrieve.py           hybrid (BM25 + vector) + metadata filter + rerank
    agents/
      state.py              AgentState TypedDict
      graph.py              LangGraph wiring
      nodes/                planner, executor, risk, rag, market, critic, composer, translator, delivery
      guardrails.py         off-topic / prompt-injection classifier
    db/
      schema.sql            tables (below)
      repo.py               typed DB access
  tests/
  Dockerfile
frontend/                   Next.js (App Router) + MapLibre + Recharts
scripts/                    smoke_test.py, ingest_docs.py, demo_preflight.py
data/rag_sources/           official PDFs + sources.csv (title, publisher, url, crop, state, date)
docs/                       PLAN.md, ARCHITECTURE.md, EDGE_CASES.md
```

### 1.2 Tool contract (every data tool)
```python
class ToolResult(BaseModel):
    data: Any | None
    source: str              # "Open-Meteo Forecast API"
    source_url: str          # exact request URL (keys redacted)
    fetched_at: datetime
    confidence: float        # 0..1
    error: str | None        # human-readable, None on success
    cached: bool = False     # True → UI shows "cached from <source> at <fetched_at>"
```
`@tool_call(timeout=, retries=3, backoff=1.5, fallbacks=[...], cache_ttl=)` decorator: timeout → retry → next fallback source → last *real* cached response (labelled) → `data=None, error="data unavailable"`. Never raises.

### 1.3 Agent graph (LangGraph)
```
guardrail → planner → executor ⇄ (tools) → risk_engine → rag_retrieve → market
         → composer → critic ──fail (max 2 loops)──▶ composer / executor
                            └─pass─▶ translator → tts → delivery → persist
```
`AgentState`: field, crop, sowing_date, as_of_date, language, tool_results{}, risks[], citations[], draft, critic_report, attempt, trace[], final_alert.

Critic = **deterministic checks first** (every number in the draft appears in a tool output within rounding; every `[n]` citation id exists in retrieved chunks; no advice sentence without a citation) **then an LLM judge** (claims supported by the cited chunk text?). Failures go back as a structured list; after 2 failed loops the unsupported sentence is replaced with "Contact your local KVK".

### 1.4 Database (Supabase Postgres + pgvector)
`farmers(id, name, telegram_chat_id, language, created_at)`
`fields(id, farmer_id, name, polygon geojson, centroid, area_ha, crop, sowing_date, state, district)`
`agent_runs(id, field_id, as_of_date, mode[live|replay], status, started_at, finished_at, critic_loops)`
`run_events(id, run_id, seq, node, type, payload jsonb, ts)` ← powers the live trace and replays
`alerts(id, run_id, language, text, audio_url, citations jsonb, delivered_at)`
`api_cache(key, source, payload jsonb, fetched_at)` ← labelled cache
`doc_chunks(id, doc_id, text, embedding vector(384), crop, state, stage, page, source_url)`

### 1.5 API
`POST /fields` · `GET /fields/{id}` · `GET /fields/{id}/ndvi` · `GET /fields/{id}/weather`
`POST /runs {field_id, as_of_date?}` → `run_id` · `GET /runs/{id}/stream` (SSE) · `GET /runs/{id}`
`POST /ask {field_id, text|audio}` · `POST /telegram/webhook` · `GET /health` (includes per-API status)

---

## Phase 0 — Lock-in (H0 → H3)

**Goal:** every API proven live, hello-world deployed on both hosts, scope frozen.

| Who | Task | Output |
|---|---|---|
| All | Get keys: LLM, Sarvam, data.gov.in, Supabase, Telegram (@BotFather), ORS, HF account, Vercel | `.env` filled |
| All | Run `scripts/smoke_test.py` until 0 blockers | screenshot of output in `docs/` |
| A | Create public GitHub repo, push skeleton, protect `main` (PRs or at least pull-before-push) | repo URL |
| A | Install Python 3.12, rebuild `.venv` | — |
| B | Pick **4 demo fields**: 2 Punjab (paddy harvest / wheat sowing), 2 Tamil Nadu delta (samba paddy). Check ≥10 clear scenes in the last 12 months each. Also pick 1 replay case (Punjab wheat, Mar 2022 heatwave). | `docs/demo_fields.md` with coordinates + scene counts |
| C | `npx create-next-app frontend` (TS, Tailwind, App Router) → Vercel. Backend Dockerfile → HF Space. Frontend calls `/health`. | two live URLs |
| D | Download RAG sources (see Phase 1 list); start `data/rag_sources/sources.csv` | ≥10 PDFs |

**Exit gate (H3):** smoke test green except optional checks · both URLs live · demo fields chosen · first commit pushed.

---

## Phase 1 — Data tools & foundations (H3 → H14)

**Goal:** every tool returns a real `ToolResult` for every demo field; RAG answers with citations; map registers a field.

### B — weather, satellite, risk
1. `tools/base.py`: ToolResult + decorator + `api_cache` read/write. (H3–4)
2. `tools/weather.py`: 16-day forecast (daily Tmax/Tmin, precip, precip probability, ET0, wind) + archive (sowing date → today for growing degree days; same window in the last 10 years as baseline). Fallback: labelled cache. (H4–6)
3. `tools/satellite.py`: STAC search over polygon bbox → per-scene windowed read of B04, B08, SCL → mask SCL classes 3, 8, 9, 10 (cloud shadow, cloud, cirrus) → mean NDVI inside polygon + `clear_fraction`. Confidence = clear_fraction × recency factor. Time series for the season + same window in the previous 3 years for the anomaly baseline. Cache per scene id (scenes never change). (H6–10)
4. `risk/crop_stage.py`: GDD from archive temperatures with crop base temperature (cited) → stage (sowing / tillering / flowering / grain fill / maturity). Missing sowing date → estimate from the NDVI green-up date, confidence lowered, says so. (H10–11)
5. `risk/rules.yaml` + `risk/engine.py`: dry spell, heat stress at flowering, heavy rain in harvest window, waterlogging, NDVI anomaly (z-score vs field's own history). Each risk = {type, score 0–1, window, evidence (tool values), rule citation}. (H11–14)

### A — RAG + LLM layer
1. RAG sources (with D): PAU Package of Practices (Rabi + Kharif), TNAU Agritech crop production guides, ICAR-IIWBR wheat advisories, ICAR-NRRI rice handbook, IMD agromet advisory pages for Punjab and TN districts, ICAR contingency plans for the chosen districts. Record URL + date in `sources.csv`. (H3–5)
2. `rag/ingest.py`: PyMuPDF text by page → section-aware chunks (~500 tokens, 80 overlap) → metadata {doc, publisher, crop, state, stage, page, url} → e5 embeddings → pgvector. (H5–9)
3. `rag/retrieve.py`: metadata filter (crop, state) → BM25 + vector hybrid → reciprocal-rank fusion → top-6 with similarity threshold. Below threshold → `no_supporting_source`. (H9–12)
4. LLM client wrapper: provider-agnostic, tool schemas generated from pydantic models, JSON-mode outputs, token/latency logging. (H12–14)

### C — frontend shell
1. Field registration page: MapLibre + satellite basemap (Esri World Imagery), draw polygon (terra-draw), crop / sowing date / language / Telegram link form, client-side validation (area 0.05–50 ha). (H3–9)
2. Dashboard shell: field header, NDVI chart (Recharts), 16-day weather strip, risk cards, empty "Agent trace" panel, explicit loading / error / "data unavailable" / "cached at …" states as components. (H9–14)

### D — market + voice
1. `tools/market.py`: Agmarknet by state + commodity → markets within N km. Mandi coordinates via Nominatim (cached, labelled). Fallback chain: Agmarknet → CEDA → labelled cache → unavailable. (H3–7)
2. `tools/routing.py`: OSRM distance/time, ORS fallback. Net price = modal price − transport cost, where the **per-km transport rate is entered by the farmer** (not invented). (H7–9)
3. `tools/voice.py`: translate, TTS (chunk long text; Sarvam has a per-request character limit), STT (convert Telegram OGG → WAV with ffmpeg first). (H9–12)
4. Telegram bot: `/start` links chat id to a farmer, receives voice + text, sends voice notes. Polling locally, webhook in production. (H12–14)

**Exit gate (H14):** `pytest backend/tests -m live` passes for every tool on all 4 demo fields · RAG returns cited chunks for 10 hand-written questions · a field can be drawn and saved · a Hindi voice note arrives on a teammate's phone.

---

## Phase 2 — Agent brain (H14 → H26)

**Goal:** one button runs the full graph and the UI streams every step.

| Who | Task | Hours |
|---|---|---|
| A | `agents/state.py`, `graph.py`, planner node (decides which tools are needed given field + question; skips market if not near harvest) | H14–17 |
| A | Composer (structured output: summary, risks, action calendar items {date, action, citation_ids}), critic (deterministic + LLM judge), correction loop, max-loop fallback | H17–21 |
| A | `guardrails.py`: off-topic + prompt-injection classifier before the planner; retrieved text treated as data, never instructions | H21–23 |
| B | Wrap tools as graph nodes with parallel execution (weather ‖ satellite ‖ market); partial-failure handling: node marks missing inputs, risk engine lowers confidence | H14–18 |
| B | Historical replay: `as_of_date` threads through every tool (forecast → archive data for "future" days, labelled; satellite → scenes ≤ as_of_date) | H18–22 |
| C | `run_events` → SSE endpoint; trace panel renders nodes live (tool name, source, latency, confidence, critic verdicts, loop count) | H14–20 |
| C | "Run monitoring now" + replay date picker; NDVI chart marks cloudy / skipped scenes | H20–24 |
| D | APScheduler daily job per field; alert de-duplication (don't re-send the same risk) | H14–17 |
| D | Ask flow: text or voice question → same graph in Q&A mode → cited answer | H17–22 |

**Sleep:** H18–22 A + C sleep, H22–26 B + D sleep. Whoever is awake owns the build.

**Exit gate (H26):** from the deployed frontend, "Run monitoring now" on a demo field streams the full trace and produces a cited alert; the critic has been seen catching at least one real error in logs; replay mode works for the 2022 case.

---

## Phase 3 — Integration (H26 → H36)

**Goal:** the full demo story works end to end, repeatedly, on production URLs.

1. End-to-end: risk detected → translated alert + action calendar → Hindi/Tamil voice note on Telegram → farmer voice question → spoken, cited answer. (all, H26–30)
2. Market agent in the graph: near harvest stage → "sell at mandi X (net ₹/qtl after transport)" or "prices are below the 3-year median for this month (CEDA)". Never a hard "hold" without that evidence. (D, H26–29)
3. Action calendar view in the dashboard + `.ics` download. (C, H28–31)
4. Source panel: every alert sentence shows its citation; clicking opens doc + page. (C, H30–33)
5. `/health` shows per-API live status; frontend status bar. (B, H29–31)
6. Performance: cache scene NDVI; run weather/satellite/market in parallel; target **< 25 s** for a full run. (B, H31–34)
7. **Reliability runs:** `scripts/demo_preflight.py` runs the full flow on all 4 fields + replay, 10×, logs pass rate and p95 latency. (A, H33–36)

**Exit gate (H36):** ≥ 9/10 preflight runs green per field · p95 < 30 s · voice round trip works on two phones.

---

## Phase 4 — Hardening (H36 → H42) · FEATURE FREEZE at H40

Work through `docs/EDGE_CASES.md`; every row needs a test + a visible UI state.

| Edge case | Handling | Owner |
|---|---|---|
| Cloudy imagery | last clear scene, confidence lowered, "Last clear view: 12 Oct (18 days ago)" | B |
| API down / slow | fallback chain, "cached from X at T" badge, agent continues on partial data | B, D |
| Point in sea / outside India | Nominatim + India bbox → reject with message | C |
| Field too small (< 0.05 ha ≈ fewer than 5 Sentinel pixels) / too large (> 50 ha) | reject with reason | C |
| Self-intersecting polygon | shapely `is_valid` → message | C |
| Unsupported crop | "We support paddy, wheat, mustard for now — contact your KVK" | A |
| Missing sowing date | estimate from NDVI green-up, say so | B |
| Contradictory inputs (sowing date in future, wheat sown in July) | validation + ask to confirm | A |
| Garbled / noisy voice | STT confidence or empty transcript → "Sorry, please repeat" voice reply | D |
| Hinglish / mixed language | Sarvam STT handles code-mix; normalise before guardrail | D |
| Off-topic / prompt injection | guardrail refusal, logged in trace | A |
| Unsourced claim / wrong number | critic loop; after 2 loops → "contact KVK" replacement | A |
| LLM API down | retry → secondary model → deterministic template alert built only from risk engine output | A |
| Telegram down | alert still shown in dashboard + downloadable audio | D |

Also: Sentry (or structured logs) on backend, global React error boundary, rate limiting on `/runs`.
Stretch (only if every gate above is green at H40): leaf-photo disease check via a vision LLM, cited to a RAG source.

**Exit gate (H42):** every edge case reproduced live on production without a crash · code frozen · tag `v1.0`.

---

## Phase 5 — Submission (H42 → H46, buffer to H48)

| Who | Deliverable | Done when |
|---|---|---|
| A | README: problem, solution, Mermaid architecture, data-source table, setup, env vars, edge-case table, team | a teammate sets up from scratch using only the README |
| B | `docs/ARCHITECTURE.md`: data flow, risk rules with citations, confidence maths | — |
| C | Final deploy check, custom domain optional, uptime ping | both URLs work in incognito on mobile |
| D | Demo video 2:45 (screen + phone via scrcpy, live run, no faked steps) | uploaded, link in README |
| All | Deck (6 slides): Title & Team → Problem → Solution & Features → Architecture & Innovation → Roadmap & Demo summary (+ Impact/USP slide) | PDF exported |
| All | Q&A drill: each person explains the full data flow alone in 3 minutes | everyone passes |

**Submit by H46.** H46–48 is buffer only.

### Demo video script (2:45)
1. 0:00–0:15 Problem: district bulletins vs a single field.
2. 0:15–0:40 TN delta field on the map: real NDVI curve, NE-monsoon forecast.
3. 0:40–1:30 "Run monitoring now": live trace weather ‖ satellite ‖ market → risk → RAG → critic verdict (shows the real check results, and a past logged correction if no error occurred live).
4. 1:30–1:55 Phone: Tamil/Hindi voice alert with sources + action calendar.
5. 1:55–2:20 Live voice question → spoken cited answer.
6. 2:20–2:35 Break it: disable an API from the admin toggle / cloudy date → graceful degradation.
7. 2:35–2:45 Replay of the 2022 heatwave + roadmap line.

---

## Rubric mapping

| Rubric item (pts) | Where we earn it |
|---|---|
| Technical depth (20) | pixel-level cloud masking, GDD crop stage, hybrid RAG with metadata, LangGraph state graph, parallel tools, tool contract |
| AI reasoning & agentic execution (25) | planner, multi-step tools, critic self-correction loop, guardrails, per-sentence citations, replay backtest |
| Impact & USP (25) | field-level vs district, proactive, voice-first in 4 languages, mandi net-price after transport, KVK fallback |
| Live demo & robustness (30) | fallback chains, labelled cache, `/health` status, 14 edge cases, preflight 10× runs, no cold starts |

## Hard-rule compliance checklist
- [ ] No hardcoded data anywhere: `grep` for literal prices / temperatures before submit.
- [ ] Every cached value shows source + timestamp in the UI.
- [ ] Every `rules.yaml` threshold has a citation.
- [ ] Tests call live APIs or compute on live results; no fixture JSON files.
- [ ] Commit at least hourly; history shows the work timeline.
- [ ] Each member can explain every module (Phase 5 drill).

## Checkpoints
Stand-up (10 min) at **H3, H8, H14, H20, H26, H32, H36, H40, H44**: each gate status, blockers, re-plan.

## Cut list (apply in this order when behind)
1. Leaf disease stretch goal
2. `.ics` export
3. Punjabi language (keep Hindi, Tamil, English)
4. CEDA trend comparison (keep Agmarknet net price only)
5. Scheduler (keep "Run monitoring now" + replay)
6. Mustard (keep paddy + wheat)
7. Telegram voice question (keep dashboard voice question)
**Never cut:** critic loop, citations, fallbacks, live trace, replay mode.
