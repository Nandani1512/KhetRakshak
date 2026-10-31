// backend/static/app.js — KhetRakshak dashboard: routing, i18n, map, charts, live agent trace.
(() => {
  const $ = (s) => document.querySelector(s);
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked: fine */ } },
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const S = { lang: store.get("kr-lang", "en"), lowdata: store.get("kr-lowdata", "0") === "1", fields: [], field: null,
              map: null, drawn: null, fieldLayer: null, charts: {}, es: null, health: null };

  const t = (k) => (I18N[S.lang] && I18N[S.lang][k]) ?? I18N.en[k] ?? k;
  const icons = () => window.lucide && lucide.createIcons();

  function toast(msg, kind = "") {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), 5000);
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}, ...opts });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const j = await res.json(); detail = j.detail || j.error || detail; } catch { /* not JSON */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res.status === 204 ? null : res.json();
  }

  // ---------------------------------------------------------------- i18n + low data
  function applyLang() {
    document.documentElement.lang = S.lang === "hi" ? "hi" : "en";
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    $("#lang-label").textContent = S.lang === "hi" ? "EN" : "हिंदी";
    const items = t("ticker");
    const run = items.map((x) => `<span>&nbsp;★&nbsp;${esc(x)}</span>`).join("");
    $("#ticker-track").innerHTML = run + run;  // duplicated for a seamless loop
    $("#ticker-static").innerHTML = items.map((x) => `<li>${esc(x)}</li>`).join("");
    renderStats();
  }
  function applyLowData() {
    document.body.classList.toggle("lowdata", S.lowdata);
    $("#btn-lowdata").setAttribute("aria-pressed", String(S.lowdata));
    $("#lowdata-badge").hidden = !S.lowdata;
    $("#webfonts").disabled = S.lowdata;
    if (S.map) setBasemap();
  }
  $("#btn-lang").onclick = () => { S.lang = S.lang === "hi" ? "en" : "hi"; store.set("kr-lang", S.lang); applyLang(); };
  $("#btn-lowdata").onclick = () => { S.lowdata = !S.lowdata; store.set("kr-lowdata", S.lowdata ? "1" : "0"); applyLowData(); };

  // ---------------------------------------------------------------- nav + routing
  const moreBox = $(".more");
  $("#more-btn").onclick = (e) => { e.stopPropagation(); const o = moreBox.classList.toggle("open"); $("#more-btn").setAttribute("aria-expanded", String(o)); };
  document.addEventListener("click", () => moreBox.classList.remove("open"));
  $("#btn-menu").onclick = () => { const d = $("#drawer"); d.hidden = !d.hidden; $("#btn-menu").setAttribute("aria-expanded", String(!d.hidden)); };

  function route() {
    const h = location.hash.replace("#/", "") || "home";
    const isApp = h === "app";
    $("#view-home").hidden = isApp;
    $("#view-app").hidden = !isApp;
    document.querySelectorAll(".nav-links > a").forEach((a) => a.classList.toggle("active", a.dataset.route === (isApp ? "app" : h === "how" ? "how" : "home")));
    $("#drawer").hidden = true;
    if (isApp) { initMap(); setTimeout(() => S.map.invalidateSize(), 50); window.scrollTo(0, 0); }
    else if (h === "how" || h === "faq") { document.getElementById(h)?.scrollIntoView(); }
    else window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  // ---------------------------------------------------------------- health + live stats
  async function loadHealth() {
    try { S.health = await api("/api/health"); } catch (e) { S.health = null; }
    const h = S.health;
    const pills = h ? [
      ["DB: " + (h.database.backend === "supabase-postgres" ? "Supabase" : "memory"), h.database.backend === "supabase-postgres" ? "ok" : "warn"],
      [`RAG: ${h.rag_index.chunks} passages`, h.rag_index.chunks ? "ok" : "bad"],
      [`LLM: ${h.llm.model}`, h.llm.last_error ? "warn" : "ok"],
      [`Telegram: ${h.telegram}`, String(h.telegram).match(/webhook|polling/) ? "ok" : "warn"],
      ...h.disabled_sources.map((s) => [`OFF: ${s}`, "bad"]),
    ] : [["API unreachable", "bad"]];
    $("#health").innerHTML = pills.map(([l, k]) => `<span class="pill ${k}">${esc(l)}</span>`).join("");
    $("#preview-list").innerHTML = pills.slice(0, 5).map(([l, k]) => `<li class="${k === "bad" ? "bad" : "ok"}"><span>${esc(l)}</span><span>${k === "bad" ? "✕" : "✓"}</span></li>`).join("");
    renderStats();
    renderToggles();
  }
  function renderStats() {
    const h = S.health;
    if (!h) return;
    const stats = [[h.rag_index.documents, t("stat_docs")], [h.rag_index.chunks, t("stat_chunks")],
                   [Object.keys(h.languages).length, t("stat_langs")], [h.crops.length, t("stat_crops")],
                   [S.fields.length, t("stat_fields")]];
    $("#stats").innerHTML = stats.map(([n, l]) => `<div class="stat"><span class="stat-n">${esc(n)}</span><span class="stat-l">${esc(l)}</span></div>`).join("");
  }
  const SOURCES = ["Open-Meteo Forecast", "Open-Meteo Archive", "Planetary Computer", "Agmarknet", "OSRM", "Gemini", "Sarvam"];
  function renderToggles() {
    const off = new Set(S.health?.disabled_sources || []);
    $("#source-toggles").innerHTML = SOURCES.map((s) => `<label class="toggle"><span>${esc(s)}</span><input type="checkbox" data-src="${esc(s)}" ${off.has(s) ? "" : "checked"} aria-label="${esc(s)} enabled"></label>`).join("");
    document.querySelectorAll("#source-toggles input").forEach((cb) => cb.onchange = async () => {
      try {
        await api("/api/admin/sources", { method: "POST", body: JSON.stringify({ source: cb.dataset.src, enabled: cb.checked }) });
        toast(`${cb.dataset.src} ${cb.checked ? "ON" : "OFF"}`, cb.checked ? "" : "warn");
        loadHealth();
      } catch (e) { toast(e.message, "error"); cb.checked = !cb.checked; }
    });
  }

  // ---------------------------------------------------------------- map
  let baseLayer, labelLayer;
  function setBasemap() {
    if (baseLayer) S.map.removeLayer(baseLayer);
    if (labelLayer) S.map.removeLayer(labelLayer);
    if (S.lowdata) {
      baseLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(S.map);
    } else {
      baseLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Imagery © Esri" }).addTo(S.map);
      labelLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 }).addTo(S.map);
    }
  }
  function initMap() {
    if (S.map) return;
    S.map = L.map("map").setView([22.5, 79], 5);
    setBasemap();
    S.drawn = new L.FeatureGroup().addTo(S.map);
    S.map.addControl(new L.Control.Draw({
      draw: { polygon: { allowIntersection: false, showArea: true, shapeOptions: { color: "#FFD43B", weight: 3 } },
              polyline: false, rectangle: { shapeOptions: { color: "#FFD43B", weight: 3 } }, circle: false, marker: false, circlemarker: false },
      edit: { featureGroup: S.drawn },
    }));
    S.map.on(L.Draw.Event.CREATED, (e) => {
      S.drawn.clearLayers();
      S.drawn.addLayer(e.layer);
      showForm(true);
      const ha = L.GeometryUtil.geodesicArea(e.layer.getLatLngs()[0]) / 10000;
      $("#draw-status").textContent = `✓ ${ha.toFixed(2)} ha`;
      setStep(3);
    });
  }
  function setStep(n) {
    document.querySelectorAll(".wstep").forEach((w) => {
      const s = Number(w.dataset.step);
      w.classList.toggle("active", s === n);
      w.classList.toggle("done", s < n);
      w.querySelector("span").textContent = s < n ? "✓" : s;
    });
  }
  function showForm(show) { $("#new-field").hidden = !show; if (show && !S.drawn.getLayers().length) setStep(1); }
  $("#btn-new").onclick = () => { showForm(true); $("#goto").focus(); };
  $("#btn-empty-new").onclick = () => { showForm(true); $("#goto").focus(); };
  $("#btn-goto").onclick = () => {
    const m = $("#goto").value.match(/(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)/);
    if (!m) { toast("Enter lat, lon like 30.85, 75.70", "warn"); $("#goto").classList.add("invalid"); return; }
    $("#goto").classList.remove("invalid");
    S.map.setView([Number(m[1]), Number(m[2])], 16);
    setStep(2);
  };

  $("#field-form").onsubmit = async (e) => {
    e.preventDefault();
    const err = $("#form-error");
    err.hidden = true;
    const layer = S.drawn.getLayers()[0];
    if (!layer) { err.querySelector("span").textContent = t("draw_hint"); err.hidden = false; return; }
    const fd = new FormData(e.target);
    if (!fd.get("name").trim()) { err.querySelector("span").textContent = t("f_name") + "?"; err.hidden = false; return; }
    const body = {
      name: fd.get("name"), farmer_name: fd.get("farmer_name") || null, crop: fd.get("crop"),
      sowing_date: fd.get("sowing_date") || null, language: fd.get("language"),
      transport_rate_per_qtl_km: fd.get("transport_rate_per_qtl_km") ? Number(fd.get("transport_rate_per_qtl_km")) : null,
      polygon: layer.toGeoJSON().geometry,
    };
    const btn = e.target.querySelector("button[type=submit]");
    btn.disabled = true;
    try {
      const f = await api("/api/fields", { method: "POST", body: JSON.stringify(body) });
      toast(`Saved · ${f.area_ha} ha · ${f.district || ""} ${f.state || ""}. Fetching satellite history…`);
      S.drawn.clearLayers(); e.target.reset(); showForm(false);
      await loadFields(); selectField(f.id);
    } catch (ex) { err.querySelector("span").textContent = ex.message; err.hidden = false; }
    finally { btn.disabled = false; }
  };

  // ---------------------------------------------------------------- fields
  async function loadFields() {
    try { S.fields = await api("/api/fields"); } catch (e) { toast(e.message, "error"); S.fields = []; }
    const ul = $("#field-list");
    ul.innerHTML = S.fields.length ? S.fields.map((f) => `<li class="item ${S.field?.id === f.id ? "active" : ""}" data-id="${esc(f.id)}" tabindex="0" role="button">
        <strong>${esc(f.name)}</strong><span class="sub">${esc(f.crop)} · ${esc(f.area_ha)} ha · ${esc(f.district || "")}${f.telegram_chat_id ? " · ✈" : ""}</span></li>`).join("")
      : `<li class="hint">${esc(t("empty_text"))}</li>`;
    ul.querySelectorAll(".item").forEach((li) => { li.onclick = () => selectField(li.dataset.id); li.onkeydown = (e) => e.key === "Enter" && selectField(li.dataset.id); });
    renderStats();
  }

  async function selectField(id) {
    const f = S.fields.find((x) => x.id === id) || await api(`/api/fields/${id}`);
    S.field = f;
    loadFields();
    $("#empty").hidden = true;
    $("#dashboard").hidden = false;
    $("#f-name").textContent = f.name;
    $("#f-crop").textContent = f.crop;
    $("#f-meta").textContent = `${f.area_ha} ha · ${f.district || "?"}, ${f.state || "?"} · ${f.sowing_date ? "sown " + f.sowing_date : "sowing date not given"} · ${f.language}`;
    $("#f-stage").textContent = "…";
    if (S.fieldLayer) S.map.removeLayer(S.fieldLayer);
    S.fieldLayer = L.geoJSON(f.polygon, { style: { color: "#FFD43B", weight: 4, fillOpacity: 0.08 } }).addTo(S.map);
    S.map.fitBounds(S.fieldLayer.getBounds(), { maxZoom: 17, padding: [30, 30] });
    $("#trace").innerHTML = `<li class="hint">${esc(t("trace_empty"))}</li>`;
    $("#output").innerHTML = `<p class="hint">${esc(t("no_alert"))}</p>`;
    loadOverview();
  }

  async function loadOverview(asOf) {
    const f = S.field;
    $("#risks").innerHTML = `<div class="risk skeleton"></div><div class="risk skeleton"></div>`;
    $("#w-src").textContent = "loading weather…"; $("#n-src").textContent = "loading satellite history (first time can take ~1 min)…";
    try {
      const o = await api(`/api/fields/${f.id}/overview${asOf ? "?as_of=" + asOf : ""}`);
      if (S.field?.id !== f.id) return;
      drawWeather(o.weather); drawNdvi(o.ndvi); drawRisks(o.assessment);
      const st = o.assessment.stage;
      $("#f-stage").textContent = `${st.stage}${st.days_after_sowing != null ? " · day " + st.days_after_sowing : ""}${o.replay ? " · REPLAY " + o.as_of : ""}`;
    } catch (e) { toast("Overview failed: " + e.message, "error"); }
  }

  function srcLine(r) {
    if (!r) return "";
    const warn = r.cached ? ` <span class="warn">CACHED ${esc(r.fetched_at.slice(0, 16))} UTC</span>` : "";
    const err = r.error ? ` <span class="warn">${esc(r.error)}</span>` : "";
    const notes = (r.notes || []).filter((n) => !/answered in/.test(n)).map(esc).join(" · ");
    return `SOURCE: ${esc(r.source)} · fetched ${esc((r.fetched_at || "").slice(0, 16))} UTC · confidence ${esc(r.confidence)}${warn}${err}${notes ? "<br>" + notes : ""}`;
  }

  Chart.defaults.font.family = "'Space Mono', ui-monospace, monospace";
  Chart.defaults.color = "#111";
  Chart.defaults.borderColor = "rgba(0,0,0,.12)";
  function chart(id, cfg) { S.charts[id]?.destroy(); S.charts[id] = new Chart(document.getElementById(id), cfg); }

  function drawWeather(w) {
    $("#w-src").innerHTML = srcLine(w);
    if (!w.data) { S.charts["c-weather"]?.destroy(); return; }
    const rows = [...w.data.past, ...w.data.forecast];
    const nPast = w.data.past.length;
    chart("c-weather", {
      data: { labels: rows.map((r) => r.date.slice(5)), datasets: [
        { type: "bar", label: "Rain mm", data: rows.map((r) => r.precipitation_sum), yAxisID: "y1",
          backgroundColor: rows.map((_, i) => i < nPast ? "#A0C4FF" : "#4A7BD0"), borderColor: "#111", borderWidth: 1 },
        { type: "line", label: "Tmax °C", data: rows.map((r) => r.temperature_2m_max), borderColor: "#FF6B6B", borderWidth: 3, pointRadius: 0, tension: 0 },
        { type: "line", label: "Tmin °C", data: rows.map((r) => r.temperature_2m_min), borderColor: "#111", borderWidth: 2, pointRadius: 0, borderDash: [4, 3], tension: 0 },
      ] },
      options: { maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { boxWidth: 12 } } },
        scales: { y: { title: { display: true, text: "°C" } }, y1: { position: "right", beginAtZero: true, grid: { drawOnChartArea: false }, title: { display: true, text: "mm" } },
                  x: { ticks: { maxTicksLimit: 12 } } } },
    });
  }

  function drawNdvi(n) {
    $("#n-src").innerHTML = srcLine(n) + (n.data ? `<br>${n.data.scenes_clear}/${n.data.scenes_total} scenes clear over this field` : "");
    if (!n.data) { S.charts["c-ndvi"]?.destroy(); return; }
    const pts = n.data.series;
    const clear = pts.filter((p) => p.ndvi != null).map((p) => ({ x: p.date, y: p.ndvi }));
    const skipped = pts.filter((p) => p.ndvi == null).map((p) => ({ x: p.date, y: 0, status: p.status }));
    const base = n.data.baseline_mean;
    const all = pts.map((p) => p.date);
    chart("c-ndvi", {
      type: "line",
      data: { labels: all, datasets: [
        { label: "NDVI (clear)", data: clear, borderColor: "#2E7D32", backgroundColor: "#A8E6CF", borderWidth: 3, pointRadius: 5, pointBorderColor: "#111", tension: 0 },
        { label: "cloudy / unusable", data: skipped, showLine: false, pointStyle: "crossRot", pointRadius: 7, borderColor: "#FF6B6B", borderWidth: 3 },
        ...(base != null ? [{ label: "previous years (same weeks)", data: all.length ? [{ x: all[0], y: base }, { x: all[all.length - 1], y: base }] : [], borderColor: "#111", borderDash: [6, 4], borderWidth: 2, pointRadius: 0 }] : []),
      ] },
      options: { maintainAspectRatio: false, animation: false,
        scales: { x: { ticks: { maxTicksLimit: 8, callback(v) { return String(this.getLabelForValue(v)).slice(5); } } }, y: { min: 0, max: 1 } },
        plugins: { tooltip: { callbacks: { label: (c) => c.raw.status ? `${c.raw.status}` : `NDVI ${c.raw.y}` } } } },
    });
  }

  function drawRisks(a) {
    const cards = a.risks.map((r) => `<article class="risk ${r.level}">
        <span class="tag">${esc(r.level)} risk · ${esc(r.window.join(" → "))}</span>
        <span class="t">${esc(r.title)}</span><span class="r">${esc(r.rule)}</span>
        <span class="c">${r.citation ? `SOURCE: <a href="${esc(r.citation.url)}" target="_blank" rel="noopener">${esc(r.citation.title)}, p.${esc(r.citation.page)}</a>` : "BASIS: " + esc(r.basis)}</span></article>`);
    if (!cards.length) cards.push(`<article class="risk none"><span class="tag">✓</span><span class="t">${esc(t("no_risk"))}</span><span class="r">${esc(t("no_risk_d"))}</span></article>`);
    a.gaps.forEach((g) => cards.push(`<article class="risk moderate"><span class="tag">DATA GAP</span><span class="r">${esc(g)}</span></article>`));
    $("#risks").innerHTML = cards.join("");
  }

  // ---------------------------------------------------------------- agent runs
  function traceItem(ev) {
    const li = document.createElement("li");
    li.className = ev.status;
    const hasData = ev.data && !(ev.status === "done" && Object.keys(ev.data).length === 1);
    li.innerHTML = `<span class="n">${esc(ev.node)}</span><span>${hasData
      ? `<details><summary>${esc(ev.message)}</summary><pre>${esc(JSON.stringify(ev.data, null, 1))}</pre></details>`
      : esc(ev.message)}</span>`;
    return li;
  }

  function watchRun(runId, kind) {
    S.es?.close();
    const ol = $("#trace");
    ol.innerHTML = "";
    $("#output").innerHTML = `<p class="hint">agent running…</p>`;
    $("#out-title").textContent = kind === "ask" ? t("answer") : t("alert");
    setBusy(true);
    const es = new EventSource(`/api/runs/${runId}/stream`);
    S.es = es;
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.status === "start" && ev.node !== "run") return;  // keep the trace readable
      ol.appendChild(traceItem(ev));
      ol.scrollTop = ol.scrollHeight;
      if (ev.node === "run" && ev.status === "complete") renderFinal(runId, ev.data.final);
      if (ev.node === "run" && ev.status === "error") { $("#output").innerHTML = `<p class="err">${esc(ev.message)}</p>`; }
    };
    es.addEventListener("end", () => { es.close(); setBusy(false); loadFields(); });
    es.onerror = () => { es.close(); setBusy(false); };
  }

  function renderFinal(runId, f) {
    if (!f) return;
    if (f.refused) { $("#output").innerHTML = `<div class="alert-box calm"><span class="tag">GUARDRAIL</span><h4>${esc(f.text_local || f.text_en)}</h4></div>`; return; }
    const risky = /risk|rain|heat|dry|below|warning/i.test(f.headline || "") && !/no (major|significant)/i.test(f.headline || "");
    const cites = (f.citations || []).map((c) => `<li>[${c.n}] ${c.type === "document"
        ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)}</a> — ${esc(c.publisher)}, p.${esc(c.page)}`
        : `${esc(c.title)} · ${esc((c.fetched_at || "").slice(0, 16))} UTC${c.cached ? " · CACHED" : ""}${c.url ? ` · <a href="${esc(c.url)}" target="_blank" rel="noopener">request</a>` : ""}`}</li>`).join("");
    const badges = [
      f.verified ? `<span class="pill pill-mint">✓ verified by critic</span>` : `<span class="pill pill-coral">partly verified · ${esc(f.removed_items)} item(s) removed</span>`,
      `<span class="pill pill-white">critic loops: ${esc(f.critic_loops)}</span>`,
      `<span class="pill pill-white">${esc(f.model)}</span>`,
      f.delivered ? `<span class="pill pill-yellow">✈ sent to Telegram</span>` : "",
    ].join("");
    $("#output").innerHTML = `
      <div class="alert-box ${risky ? "" : "calm"}"><span class="tag">${risky ? "⚠ ALERT" : "✓ STATUS"}</span><h4>${esc(f.headline)}</h4></div>
      <div class="badges">${badges}</div>
      ${f.text_local && f.text_local !== f.text_en ? `<div class="local">${esc(f.text_local)}</div>` : ""}
      <details ${f.text_local && f.text_local !== f.text_en ? "" : "open"}><summary class="card-label">English</summary><div class="local">${esc(f.text_en)}</div></details>
      ${f.audio_url ? `<audio controls src="${esc(f.audio_url)}"></audio>` : ""}
      ${cites ? `<span class="card-label">Sources</span><ol class="cites">${cites}</ol>` : ""}`;
  }

  function setBusy(b) { ["#btn-run", "#btn-replay", "#btn-ask", "#btn-mic"].forEach((s) => { $(s).disabled = b; }); }

  async function startRun(body, kind) {
    try { const r = await api("/api/runs", { method: "POST", body: JSON.stringify({ field_id: S.field.id, ...body }) }); watchRun(r.run_id, kind); }
    catch (e) { toast(e.message, "error"); }
  }
  $("#btn-run").onclick = () => { startRun({ mode: "monitor" }, "monitor"); loadOverview(); };
  $("#btn-replay").onclick = () => {
    const d = $("#replay-date").value;
    if (!d) { toast("Pick a past date first", "warn"); $("#replay-date").classList.add("invalid"); return; }
    $("#replay-date").classList.remove("invalid");
    startRun({ mode: "monitor", as_of: d }, "monitor"); loadOverview(d);
  };
  $("#btn-ask").onclick = () => {
    const q = $("#question").value.trim();
    if (!q) { toast(t("ask_title") + "?", "warn"); return; }
    startRun({ mode: "ask", question: q, language: S.lang === "hi" ? "hi-IN" : null }, "ask");
  };
  $("#question").onkeydown = (e) => { if (e.key === "Enter") $("#btn-ask").click(); };
  $("#btn-telegram").onclick = async () => {
    try { const r = await api(`/api/fields/${S.field.id}/telegram-link`); window.open(r.url, "_blank", "noopener"); toast("Open the bot and press START to link this field."); }
    catch (e) { toast(e.message, "error"); }
  };

  // voice question: click to start, click again to stop
  let rec = null, chunks = [];
  $("#btn-mic").onclick = async () => {
    if (rec && rec.state === "recording") { rec.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      rec = new MediaRecorder(stream);
      chunks = [];
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((tr) => tr.stop());
        $("#btn-mic").classList.remove("rec");
        $("#ask-status").textContent = "transcribing…";
        const fd = new FormData();
        fd.append("audio", new Blob(chunks, { type: rec.mimeType || "audio/webm" }), "question.webm");
        try {
          const r = await api(`/api/fields/${S.field.id}/ask-voice`, { method: "POST", body: fd });
          $("#ask-status").textContent = `heard: “${r.transcript}”`;
          $("#question").value = r.transcript;
          watchRun(r.run_id, "ask");
        } catch (e) { $("#ask-status").textContent = e.message; toast(e.message, "error"); }
      };
      rec.start();
      $("#btn-mic").classList.add("rec");
      $("#ask-status").textContent = "● recording… click 🎤 again to stop";
    } catch (e) { toast("Microphone not available: " + e.message, "error"); }
  };

  // ---------------------------------------------------------------- boot
  $("#replay-date").max = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  $("#replay-date").min = "2018-01-01";
  applyLang(); applyLowData(); icons(); route();
  loadHealth(); loadFields();
  setInterval(loadHealth, 60000);
})();
