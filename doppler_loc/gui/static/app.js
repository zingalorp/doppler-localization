"use strict";

// ---------------------------------------------------------------------------------------------
// Small helpers

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// --- showing that something is happening -----------------------------------------------------
// Some requests answer at once and some take many seconds (the first look at a whole country's
// flights), so everything the app waits for shows: a named spinner in the header, after a moment so
// that quick answers don't flicker, and at once on the button that was pressed.

const waiting = [];  // what is being waited for, latest last
let waitingTimer = null, lastPress = null;

function showWaiting() {
  const box = $("#activity");
  document.body.classList.toggle("waiting", waiting.length > 0);
  clearTimeout(waitingTimer);
  if (!waiting.length) { box.hidden = true; return; }
  if (!box.hidden) { $("#activity-text").textContent = waiting[waiting.length - 1]; return; }
  waitingTimer = setTimeout(() => {
    if (!waiting.length) return;
    $("#activity-text").textContent = waiting[waiting.length - 1];
    box.hidden = false;
  }, 120);
}

// What a request is for, in words.
function waitingFor(method, path) {
  const end = path.split("?")[0].replace(/^\/api\/projects(\/[^/]+)?/, "");
  if (method !== "GET") {
    if (end === "/spots") return "Checking that spot…";
    if (end === "/passes/measure") return "Measuring the trace…";
    if (/^\/jobs\/[^/]+\/stop$/.test(end)) return "Stopping…";
    if (end.startsWith("/jobs/")) return "Starting…";
    if (method === "POST" && path === "/api/projects") return "Creating the project…";
    if (method === "DELETE") return "Deleting…";
    return "Saving…";
  }
  if (end === "") return path.split("?")[0] === "/api/projects" ? "Loading projects…" : "Loading the project…";
  if (end === "/results") return "Loading the candidates…";
  if (end === "/aircraft") return "Loading the flights…";
  return "Loading…";
}

// The button just pressed waits with the request it caused, and with the ones that follow from it.
document.addEventListener("click", (e) => {
  const button = e.target.closest?.("button");
  if (button && !button.disabled) lastPress = { button, at: performance.now() };
}, true);

function pressedButton() {
  const held = lastPress && performance.now() - lastPress.at < 500 && lastPress.button.isConnected ? lastPress.button : null;
  held?.classList.add("waiting");
  return held;
}

async function api(method, path, body) {
  const quiet = method === "GET" && path.endsWith("/jobs");  // polled every second, with its own display
  const label = quiet ? null : waitingFor(method, path);
  const button = quiet ? null : pressedButton();
  if (label) { waiting.push(label); showWaiting(); }
  try {
    const res = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return res.headers.get("content-type")?.includes("json") ? await res.json() : res;
  } finally {
    if (label) { waiting.splice(waiting.lastIndexOf(label), 1); showWaiting(); }
    if (button) {
      button.classList.remove("waiting");
      if (lastPress?.button === button) lastPress.at = performance.now();  // a request it leads to is still this press
    }
  }
}

const projectPath = (rest = "") => `/api/projects/${encodeURIComponent(S.name)}${rest}`;

function clock(s) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  const mm = String(m).padStart(2, "0"), ss = String(sec).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

const km = (m) => m == null ? "–" : m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function parseLatLon(text) {
  const m = String(text || "").split(/[,\s]+/).filter(Boolean).map(Number);
  return m.length === 2 && m.every(isFinite) ? m : null;
}

const KIND_CLASS = { "likely aircraft": "likely", possible: "possible", "possibly turning": "turning", "not a pass": "nopass", harmonic: "nopass" };
const STATUS_COLOR = { accepted: "#4fd18b", undecided: "#f2b84b", rejected: "#777f90" };
// One colour per pass, the same on every page: its modelled curves and its flight's path on the map.
const PALETTE = ["#ff9f40", "#ff5dd2", "#5dffb0", "#ffe45d", "#5dc8ff", "#c08bff", "#ff6b6b", "#a6ff5c", "#40e0d0", "#ffb3a7", "#b0b8ff", "#e0e0e0"];
const EXPLAINED_AT = 0.6;  // a pass counts as explained when a flight follows this share of its line

// ---------------------------------------------------------------------------------------------
// Spectrogram viewer: an image from the server for the current window, with an overlay canvas.

class SpecView {
  constructor(container, opts = {}) {
    this.el = container;
    this.opts = opts;
    this.base = el("canvas");
    this.over = el("canvas");
    container.replaceChildren(this.base, this.over);
    this.view = { t0: 0, t1: 600, f0: 20, f1: 600 };
    this.limits = { t0: 0, t1: 600, f0: 20, f1: 1000 };
    this.img = null;
    this.imgView = null;
    this.pending = null;
    new ResizeObserver(() => this.resize()).observe(container);
    if (opts.interactive) this.bindEvents();
    else if (opts.onClick) container.addEventListener("click", (e) => opts.onClick(this.tAt(e.offsetX), this.fAt(e.offsetY), e));
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.el.clientWidth * dpr)), h = Math.max(1, Math.round(this.el.clientHeight * dpr));
    if (w === this.base.width && h === this.base.height) return;
    for (const c of [this.base, this.over]) { c.width = w; c.height = h; }
    this.draw();
    this.request();
  }

  get W() { return this.el.clientWidth; }
  get H() { return this.el.clientHeight; }
  x(t) { return (t - this.view.t0) / (this.view.t1 - this.view.t0) * this.W; }
  y(f) { return (this.view.f1 - f) / (this.view.f1 - this.view.f0) * this.H; }
  tAt(px) { return this.view.t0 + px / this.W * (this.view.t1 - this.view.t0); }
  fAt(py) { return this.view.f1 - py / this.H * (this.view.f1 - this.view.f0); }

  setView(v) {
    const L = this.limits;
    let { t0, t1, f0, f1 } = { ...this.view, ...v };
    const span = Math.min(Math.max(t1 - t0, 5), L.t1 - L.t0 + 120);
    t0 = Math.min(Math.max(t0, L.t0 - 60), L.t1 + 60 - span); t1 = t0 + span;
    const fspan = Math.min(Math.max(f1 - f0, 10), L.f1 - L.f0);
    f0 = Math.min(Math.max(f0, L.f0), L.f1 - fspan); f1 = f0 + fspan;
    this.view = { t0, t1, f0, f1 };
    this.draw();
    this.request();
    this.opts.onChange?.(this.view);
  }

  request() {
    if (!S.name || !this.W || !this.el.offsetParent) return;
    clearTimeout(this.pending);
    this.pending = setTimeout(() => {
      const dpr = window.devicePixelRatio || 1;
      const v = { ...this.view };
      const q = new URLSearchParams({
        t0: v.t0.toFixed(2), t1: v.t1.toFixed(2), f0: v.f0.toFixed(2), f1: v.f1.toFixed(2),
        w: Math.round(this.W * dpr), h: Math.round(this.H * dpr), vmax: S.vmax * (this.opts.vmaxScale || 1),
      });
      const img = new Image();
      const name = S.name;
      this.latest = img;
      clearTimeout(this.loadingTimer);
      this.loadingTimer = setTimeout(() => { if (this.latest === img) this.el.classList.add("loading"); }, 200);
      const settled = () => { if (this.latest === img) { clearTimeout(this.loadingTimer); this.el.classList.remove("loading"); } };
      img.onload = () => { settled(); if (name === S.name) { this.img = img; this.imgView = v; this.draw(); } };
      img.onerror = settled;
      img.src = `/api/projects/${encodeURIComponent(name)}/spectrogram.png?${q}`;
    }, 90);
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const ctx = this.base.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, this.base.width, this.base.height);
    if (this.img && this.imgView) {
      // Show the last image moved and stretched to the current view until the new one arrives.
      const iv = this.imgView;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.imageSmoothingEnabled = false;
      const x0 = this.x(iv.t0), x1 = this.x(iv.t1), y0 = this.y(iv.f1), y1 = this.y(iv.f0);
      ctx.drawImage(this.img, x0, y0, x1 - x0, y1 - y0);
    }
    const o = this.over.getContext("2d");
    o.setTransform(1, 0, 0, 1, 0, 0);
    o.clearRect(0, 0, this.over.width, this.over.height);
    o.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.opts.axes !== false) this.drawAxes(o);
    this.opts.drawOverlay?.(o, this);
  }

  drawAxes(o) {
    o.font = "11px system-ui, sans-serif";
    o.fillStyle = "rgba(255,255,255,.75)";
    o.strokeStyle = "rgba(255,255,255,.12)";
    const span = this.view.t1 - this.view.t0;
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
    const tStep = steps.find((s) => span / s <= Math.max(2, this.W / 110)) || 7200;
    for (let t = Math.ceil(this.view.t0 / tStep) * tStep; t <= this.view.t1; t += tStep) {
      const x = this.x(t);
      o.beginPath(); o.moveTo(x, 0); o.lineTo(x, this.H); o.stroke();
      o.fillText(clock(t), x + 3, this.H - 4);
    }
    const fspan = this.view.f1 - this.view.f0;
    const fStep = [5, 10, 20, 50, 100, 200].find((s) => fspan / s <= Math.max(2, this.H / 45)) || 200;
    for (let f = Math.ceil(this.view.f0 / fStep) * fStep; f <= this.view.f1; f += fStep) {
      const y = this.y(f);
      o.beginPath(); o.moveTo(0, y); o.lineTo(this.W, y); o.stroke();
      o.fillText(`${f} Hz`, 3, y - 3);
    }
  }

  bindEvents() {
    const c = this.el;
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const k = Math.exp(Math.sign(e.deltaY) * 0.18);
      const v = this.view;
      if (e.shiftKey) {
        const f = this.fAt(e.offsetY);
        this.setView({ f0: f - (f - v.f0) * k, f1: f + (v.f1 - f) * k });
      } else {
        const t = this.tAt(e.offsetX);
        this.setView({ t0: t - (t - v.t0) * k, t1: t + (v.t1 - t) * k });
      }
    }, { passive: false });
    let drag = null;
    c.addEventListener("mousedown", (e) => { drag = { x: e.offsetX, y: e.offsetY, v: { ...this.view }, moved: false }; });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const r = c.getBoundingClientRect();
      const dx = e.clientX - r.left - drag.x, dy = e.clientY - r.top - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (!drag.moved) return;
      const v = drag.v;
      const dt = dx / this.W * (v.t1 - v.t0), df = dy / this.H * (v.f1 - v.f0);
      this.setView({ t0: v.t0 - dt, t1: v.t1 - dt, f0: v.f0 + df, f1: v.f1 + df });
    });
    window.addEventListener("mouseup", (e) => {
      if (drag && !drag.moved && e.target === this.over) this.opts.onClick?.(this.tAt(drag.x), this.fAt(drag.y), e);
      drag = null;
    });
  }
}

function strokeCurve(o, view, ts, fs, style) {
  o.save();
  Object.assign(o, style);
  if (style.dash) o.setLineDash(style.dash);
  o.beginPath();
  let started = false;
  for (let i = 0; i < ts.length; i++) {
    if (fs[i] == null) { started = false; continue; }
    const x = view.x(ts[i]), y = view.y(fs[i]);
    if (!started) { o.moveTo(x, y); started = true; } else o.lineTo(x, y);
  }
  o.stroke();
  o.restore();
}

function dots(o, view, ts, fs, color, r = 1.3, every = 1) {
  o.fillStyle = color;
  for (let i = 0; i < ts.length; i += every) {
    o.beginPath(); o.arc(view.x(ts[i]), view.y(fs[i]), r, 0, 2 * Math.PI); o.fill();
  }
}

function label(o, v, text, t, f, color) {
  const x = Math.min(Math.max(v.x(t), 2), v.W - 60), y = Math.min(Math.max(v.y(f) - 6, 12), v.H - 4);
  o.font = "bold 12px system-ui, sans-serif";
  o.lineWidth = 3; o.strokeStyle = "rgba(0,0,0,.7)"; o.strokeText(text, x, y);
  o.fillStyle = color; o.fillText(text, x, y);
}

// A view of one pass: its traced line with some room around it.
function passWindow(p, pad = 30) {
  const [a, b] = p.t_range;
  const fmin = Math.min(...p.line_f), fmax = Math.max(...p.line_f);
  return { t0: a - pad, t1: b + pad, f0: Math.max(20, fmin * 0.75), f1: fmax * 1.35 };
}

// ---------------------------------------------------------------------------------------------
// App state

const S = {
  projects: [], name: null, p: null, results: null, page: "projects", vmax: 14,
  selectedPass: null, passSort: "confidence", tracing: false, guide: [], hideCurves: false, peek: false,
  selectedCand: null, locPass: null, specMode: "each", showPartial: false, poll: null, colorOf: {},
};

const passById = (id) => S.p?.passes.find((x) => x.id === id);
// Passes get their colour by time heard; flights heard on the spectrogram (S1, S2...) by number.
const colorOf = (id) => S.colorOf[id] || (/^S\d+$/.test(id) ? PALETTE[(Number(id.slice(1)) * 5 + 2) % PALETTE.length] : "#ffffff");
// When an entry (a pass, or a flight heard on the spectrogram) was heard in the recording.
const heardAt = (e) => e.heard_rec ?? passById(e.pass_id)?.t_heard ?? 0;
const cand = () => (S.results && S.selectedCand != null ? S.results.candidates[S.selectedCand] : null);

// Outages in the flight data (adsb.lol has no aircraft at all then: see adsb.coverage_gaps), as
// [start, end] unix s, and the one, if any, while pass p was heard (needs the recording's start).
const gaps = () => S.p?.adsb_gaps || [];
function passGap(p) {
  const t0 = S.p?.settings.start_unix;
  if (t0 == null || !p) return null;
  return gaps().find(([a, b]) => t0 + p.t_range[1] > a && t0 + p.t_range[0] < b) || null;
}
const gapText = ([a, b]) => `${new Date(a * 1000).toISOString().slice(11, 16)}–${new Date(b * 1000).toISOString().slice(11, 16)} UTC`;
// How well the flight data covers low flying over the area (adsb.low_coverage, squares about 10 km
// a side): receivers far away log airliners overhead but not the light aircraft heard close by.
const LOW_COVERED = 0.2;  // below this share of squares covered, warn outright
function thinSquares() {
  const c = S.p?.adsb_low;
  if (!c) return [];
  const out = [];
  c.counts.forEach((row, i) => row.forEach((k, j) => {
    if (!c.water[i][j] && k < c.min_aircraft) out.push({ i, j, k });
  }));
  return out;
}
function coverageNote(where) {
  const c = S.p?.adsb_low, thin = thinSquares();
  if (!c || !thin.length) return null;
  const land = c.water.flat().filter((w) => !w).length;
  const what = `fewer than ${c.min_aircraft} aircraft were logged below ${c.below_m} m above the ground all day`;
  const squares = `${thin.length} of the area's ${land} squares (about ${Math.round(c.cell_km)} km a side)`;
  return c.covered < LOW_COVERED
    ? el("p", { class: "warn-note" }, `Little low flying was logged here: in ${squares}, ${what}. `
      + "No receiver was near enough to hear planes that low, so light aircraft heard close by are probably "
      + "missing from adsb.lol's data: many passes will find no flight, and the search has little to go on.")
    : el("p", { class: "hint" }, `In ${squares}, ${what}`
      + (where === "map" ? " (shaded on the map)" : " (shaded on the map in step 3)")
      + ": light aircraft heard there may be missing from the flight data.");
}
const gapNote = (g) => `Heard during a gap in the flight data (${gapText(g)}): adsb.lol has no aircraft at all then, so no flight can explain it.`;

// What a candidate says about each pass: the flight that explains it best, with its modelled curve.
// Results from before every explained pass got a curve fall back to the flights it was found from.
// Then the flights heard there on the spectrogram (stacking, S1, S2...) that explain no pass.
function explainsOf(c) {
  if (!c) return [];
  const byPass = Object.fromEntries((c.flights || []).map((f) => [f.pass_id, f]));
  const list = !c.explains ? (c.flights || []).map((f) => ({ ...f, share: null })) : c.explains.map((e) => {
    if (e.curve_t || !e.flight) return e;
    const f = byPass[e.pass_id];
    return f && f.flight === e.flight ? { ...f, ...e } : e;
  });
  const heard = (c.heard || []).filter((h) => !list.some((e) => e.flight === h.flight && (e.share >= EXPLAINED_AT || e.verdict)))
    .map((h) => ({ ...h, share: null, heard: true }));
  return [...list, ...heard];
}

const entryName = (e) => `${e.registration || e.flight}${e.type ? " " + e.type : ""}`;

// A view of an entry's modelled curve: around its pass's line, or 100 s either side of when a flight
// heard on the spectrogram was heard.
function entryWindow(e) {
  const p = passById(e.pass_id);
  const ts = p ? [p.t_range[0] - 60, p.t_range[1] + 60]
    : e.heard_rec != null ? [e.heard_rec - 100, e.heard_rec + 100] : [e.curve_t[0], e.curve_t[e.curve_t.length - 1]];
  const fs = [...e.curve_f.filter((_, i) => e.curve_t[i] >= ts[0] && e.curve_t[i] <= ts[1]), ...(p?.line_f || [])].filter(isFinite);
  return { t0: ts[0], t1: ts[1], f0: Math.max(20, Math.min(...fs) * 0.85), f1: Math.max(...fs) * 1.15 };
}

// ---------------------------------------------------------------------------------------------
// Pages

const PAGES = ["projects", "recording", "passes", "flights", "search", "locations", "empty"];

function showPage(page) {
  if (page !== "projects" && page !== "empty" && !S.p) page = S.projects.length ? "projects" : "empty";
  if (page !== "passes" && S.listening) setListening(false);
  S.page = page;
  for (const p of PAGES) $(`#page-${p}`).hidden = p !== page;
  $$("#steps button").forEach((b) => b.classList.toggle("active", b.dataset.step === page));
  $("#project-btn").classList.toggle("on", page === "projects");
  if (page === "projects") renderProjects();
  if (page === "recording") renderAreaMap();
  if (page === "passes") { spectro.resize(); overview.resize(); spectro.request(); overview.request(); renderPasses(); }
  if (page === "flights") renderFlightsMap();
  else if (FL.timer) playFlights(false);
  if (page === "locations") renderLocations();
  try { if (S.name && page !== "projects") localStorage.setItem(`page:${S.name}`, page); } catch (_) { /* private mode */ }
}

// --- projects --------------------------------------------------------------------------------

async function loadProjects() {
  S.projects = await api("GET", "/api/projects");
  renderProjects();
}

function renderProjects() {
  const q = $("#project-filter").value.trim().toLowerCase();
  const rows = S.projects.filter((p) => !q || p.name.toLowerCase().includes(q) || p.date.includes(q));
  $("#project-table tbody").replaceChildren(...rows.map((p) => {
    const best = p.best;
    const rank = !p.truth ? el("span", { class: "muted" }, "–")
      : p.candidates == null ? el("span", { class: "muted" }, "not searched")
        : p.truth_rank ? el("span", { class: p.truth_rank === 1 ? "good" : "meh" }, `#${p.truth_rank}`)
          : el("span", { class: "bad" }, "not found");
    return el("tr", { class: p.name === S.name ? "selected" : "", onclick: () => openProject(p.name) },
      el("td", {}, el("b", {}, p.name)),
      el("td", {}, p.date),
      el("td", {}, p.duration ? clock(p.duration) : el("span", { class: "muted" }, "not decoded")),
      el("td", {}, p.passes ? `${p.accepted} of ${p.passes} accepted` : el("span", { class: "muted" }, "none yet")),
      el("td", {}, p.aircraft != null ? `${p.aircraft} aircraft` : el("span", { class: "muted" }, "not fetched")),
      el("td", {}, p.start_known ? "known" : el("span", { class: "muted" }, "unknown")),
      el("td", {}, p.candidates == null ? el("span", { class: "muted" }, "–")
        : !best ? el("span", { class: "muted" }, "no candidates")
          : [best.of ? `explains ${best.explained} of ${best.of}` : `${p.candidates} candidates`,
             best.truth_m != null ? el("span", { class: "muted" }, ` · ${km(best.truth_m)} from known`) : ""]),
      el("td", {}, rank),
      el("td", {}, el("button", { class: "small danger", title: "Delete this project", onclick: (e) => { e.stopPropagation(); deleteProject(p.name); } }, "Delete")),
    );
  }));
  if (!rows.length) $("#project-table tbody").replaceChildren(el("tr", {}, el("td", { colspan: 9, class: "hint" }, q ? "No project matches." : "No projects yet.")));
}

// Delete a project: its settings, passes (hand traces too), spectrogram and results. The recording
// and the flight data stay, since other projects may use them.
async function deleteProject(name) {
  if (!confirm(`Delete project “${name}”?\n\nIts passes (hand traces too), spectrogram and results are deleted. `
    + "The recording and the flight data are kept.")) return;
  try {
    await api("DELETE", `/api/projects/${encodeURIComponent(name)}`);
  } catch (e) { alert(e.message); return; }
  if (S.name === name) {
    S.name = null; S.p = null; S.results = null;
    try { localStorage.removeItem("project"); } catch (_) { /* private mode */ }
  }
  await loadProjects();
}

async function openProject(name, page) {
  if (S.listening) setListening(false);
  S.name = name;
  try { localStorage.setItem("project", name); } catch (_) { /* private mode */ }
  S.results = null; S.selectedCand = null; S.selectedPass = null; S.locPass = null; S.guide = []; S.spotBusy = null;
  AREA.area = undefined; AREA.name = null; AREA.drawing = null;
  for (const v of [spectro, overview, locSpectro, locOverview]) { v.img = v.imgView = null; }
  await refresh();
  const dur = S.p.duration || 600;
  for (const v of [spectro, overview, locSpectro, locOverview]) v.limits = { t0: 0, t1: dur, f0: 20, f1: 1000 };
  overview.view = locOverview.view = { t0: 0, t1: dur, f0: 20, f1: 600 };
  spectro.view = locSpectro.view = { t0: 0, t1: Math.min(dur, 600), f0: 20, f1: 400 };
  if (S.results?.candidates.length) S.selectedCand = 0;
  let saved = null;
  try { saved = localStorage.getItem(`page:${name}`); } catch (_) { /* private mode */ }
  const fallback = !S.p.duration ? "recording" : S.p.has_results ? "locations" : "passes";
  showPage(page || (saved && PAGES.includes(saved) && saved !== "projects" ? saved : fallback));
}

async function refresh() {
  S.p = await api("GET", projectPath());
  if (S.p.has_results && !S.results) S.results = await api("GET", projectPath("/results"));
  if (!S.p.has_results) S.results = null;
  render();
  watchJobs();
}

// --- jobs ------------------------------------------------------------------------------------

function runningJob() {
  return Object.values(S.p?.jobs || {}).find((j) => j && j.state === "running");
}

function watchJobs() {
  const running = runningJob();
  $("#job-banner").hidden = !running;
  $("#job-stop").hidden = !running;
  if (running) $("#job-text").textContent = `${running.step}: ${running.message}`;
  $$("[data-job]").forEach((b) => (b.disabled = !!running));
  if (running && !S.poll) {
    const name = S.name;
    S.poll = setInterval(async () => {
      if (name !== S.name) { clearInterval(S.poll); S.poll = null; return; }
      const jobs = await api("GET", projectPath("/jobs"));
      const was = runningJob();
      S.p.jobs = jobs;
      renderJobs();
      if (!runningJob()) {
        clearInterval(S.poll); S.poll = null;
        if (was?.step === "search" || was?.step === "explain") { S.results = null; S.selectedCand = null; }
        if (was?.step === "prepare") { await openProject(S.name, "passes"); return; }
        await refresh();
        if ((was?.step === "search" || was?.step === "explain") && S.results?.candidates.length) S.selectedCand = 0;
        if (was?.step === "search" && S.p.jobs.search?.state === "done") showPage("locations");
        else if (S.page === "locations") renderLocations();
        loadProjects();
      } else {
        const r = runningJob();
        $("#job-text").textContent = `${r.step}: ${r.message}`;
      }
    }, 1000);
  }
}

async function stopJob() {
  const r = runningJob();
  if (!r) return;
  const what = { search: "the search", detect: "finding passes", fetch: "getting the flight data", prepare: "making the spectrogram", explain: "the re-check" }[r.step] || r.step;
  if (!confirm(`Stop ${what}? What it has done so far is lost, and the results from before stay.`)) return;
  try {
    await api("POST", projectPath(`/jobs/${r.step}/stop`));
    await refresh();
  } catch (e) { alert(e.message); }
}

const STEP_WORDS = { prepare: "the spectrogram", detect: "finding passes", fetch: "the flight data", search: "the search", explain: "the re-check" };

async function startJob(step) {
  // Say so at once: the step starts in moments, but the project may take a while to reload after it.
  $("#job-banner").hidden = false;
  $("#job-stop").hidden = true;
  $("#job-text").textContent = `Starting ${STEP_WORDS[step] || step}…`;
  $$("[data-job]").forEach((b) => (b.disabled = true));
  try {
    const job = await api("POST", projectPath(`/jobs/${step}`));
    S.p.jobs = { ...S.p.jobs, [step]: job };
    renderJobs();
    watchJobs();
    await refresh();
  } catch (e) {
    alert(e.message);
    watchJobs();
    renderJobs();
  }
}

function renderJobs() {
  for (const box of $$("[data-job-status]")) {
    const j = S.p?.jobs?.[box.dataset.jobStatus];
    box.className = "job" + (j ? ` ${j.state}` : "");
    if (!j) { box.replaceChildren(); continue; }
    const extra = j.state === "done" && j.seconds != null ? ` (${clock(j.seconds)})` : "";
    const parts = [el("div", {}, j.state === "running" ? "Running: " : j.state === "error" ? "Failed: " : j.state === "stopped" ? "" : "Done: ", j.state === "stopped" ? "Stopped: run it again whenever you like." : j.message, extra)];
    if (j.state === "running") parts.push(el("div", { class: "bar" }, el("div", { style: `width:${Math.round(j.progress * 100)}%` })));
    box.replaceChildren(...parts);
  }
  $$("[data-job]").forEach((b) => (b.disabled = !!runningJob()));
}

// --- everything that depends on the project --------------------------------------------------

function render() {
  const p = S.p;
  [...p.passes].sort((a, b) => a.t_heard - b.t_heard).forEach((x, i) => (S.colorOf[x.id] = PALETTE[i % PALETTE.length]));
  $("#project-name").textContent = S.name;
  $("#project-meta").textContent = `${p.settings.date}${p.duration ? " · " + clock(p.duration) : ""}`;
  const ticks = {
    recording: !!p.duration && p.has_spec,  // a shared project carries its length but not its spectrogram
    passes: p.passes.some((x) => x.status === "accepted"),
    flights: !!p.adsb,
    search: p.has_results,
    locations: p.has_results && Object.keys(p.verdicts || {}).length > 0,
  };
  $$("#steps button").forEach((b) => { b.disabled = false; $(".tick", b).textContent = ticks[b.dataset.step] ? "✓" : ""; });
  renderSettings();
  renderPasses();
  renderFlights();
  renderSearch();
  renderJobs();
  if (S.page === "locations") renderLocations();
  if (S.page === "recording") renderAreaMap();
}

function field(name, text, value, attrs = {}, wide = false) {
  return el("label", { class: wide ? "wide" : "" }, text, el("input", { name, value: value ?? "", ...attrs }));
}

function renderSettings() {
  const s = S.p.settings;
  $("#settings-form").replaceChildren(
    el("label", { class: "wide" }, "Audio file", el("input", { value: s.audio, disabled: true, title: s.audio })),
    field("date", "Date recorded (UTC)", s.date, { type: "date" }),
    el("label", {}, "Length", el("input", { value: S.p.duration ? clock(S.p.duration) : "not decoded yet", disabled: true })),
    field("start", "Started (UTC), if known", s.start_unix != null ? new Date(s.start_unix * 1000).toISOString().slice(0, 19).replace("T", " ") : "",
      { placeholder: "2026-11-02 14:22:10" }),
    field("start_tol_s", "…to within (s)", s.start_tol_s, { type: "number", step: "any" }),
    el("h3", {}, "Area to search"),
    ...areaFields(s),
    field("mic_height", "Microphone height (m)", s.mic_height, { type: "number", step: "any" }),
    el("h3", {}, "For tests"),
    field("truth", "Known location (lat, lon)", s.truth_lat != null ? `${s.truth_lat}, ${s.truth_lon}` : "", { placeholder: "optional" }, true),
  );
}

// The shape being edited: `area` undefined until it is changed, null for back to a square.
const AREA = { area: undefined, name: null, drawing: null, places: null };
const areaNow = (s) => (AREA.area !== undefined ? AREA.area : s.area);

function areaFields(s) {
  const shape = areaNow(s), name = AREA.area !== undefined ? AREA.name : s.area_name;
  const corners = shape ? shape.reduce((n, r) => n + r.length, 0) : 0;
  const unsaved = AREA.area !== undefined;
  const pick = el("input", { id: "area-place", list: "area-places", placeholder: "Type a country or US state" });
  pick.addEventListener("focus", loadPlaces, { once: true });
  pick.addEventListener("change", () => choosePlace(pick));
  const draw = el("button", { type: "button", id: "area-draw" }, AREA.drawing ? "Finish shape" : "Draw on map");
  draw.addEventListener("click", () => (AREA.drawing ? finishDrawing() : startDrawing()));
  const clear = el("button", { type: "button", id: "area-clear", disabled: !shape && !AREA.drawing }, "Back to a square");
  clear.addEventListener("click", () => { AREA.drawing = null; setArea(null, null); });
  return [
    field("centre", "Centre (lat, lon)", `${s.lat}, ${s.lon}`, shape ? { disabled: true, title: "Set by the shape" } : {}, true),
    field("half_km", "This far each way (km)", s.half_km, { type: "number", step: "any", ...(shape ? { disabled: true, title: "Set by the shape" } : {}) }),
    el("label", { class: "wide" }, "Search a country or state", pick, el("datalist", { id: "area-places" })),
    el("div", { class: "wide area-tools" }, draw, clear,
      el("span", { class: "hint", id: "area-note" }, AREA.drawing ? "Click the corners, double-click or press Finish to close the shape. Esc cancels."
        : (shape ? `${name || "Drawn shape"}, ${corners} corners.` : "The whole square.") + (unsaved ? " Press Save changes to keep it." : ""))),
  ];
}

function setArea(rings, name) {
  AREA.area = rings;
  AREA.name = name;
  renderSettings();
  renderAreaMap();
  if (rings) fitOnce(baseMap("area-map"), `shape:${name}:${rings[0][0]}`, L.latLngBounds(rings.flat()));
}

function startDrawing() {
  AREA.drawing = [];
  baseMap("area-map").map.doubleClickZoom.disable();
  renderSettings();
  renderAreaMap();
}

function finishDrawing() {
  const pts = AREA.drawing;
  AREA.drawing = null;
  setTimeout(() => baseMap("area-map").map.doubleClickZoom.enable(), 300);  // after the double-click that ended it
  if (pts && pts.length >= 3) setArea([pts], null);
  else { renderSettings(); renderAreaMap(); }
}

// Countries (Natural Earth, public domain) and US states, fetched the first time the box is used.
async function loadPlaces() {
  if (AREA.places) return;
  try {
    const [w, u] = await Promise.all(["https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json", "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json"]
      .map((url) => fetch(url).then((r) => r.json())));
    const places = new Map();
    for (const f of topojson.feature(w, w.objects.countries).features) places.set(f.properties.name, f.geometry);
    for (const f of topojson.feature(u, u.objects.states).features) places.set(`${f.properties.name}, USA`, f.geometry);
    AREA.places = places;
    const list = $("#area-places");
    if (list) list.replaceChildren(...[...places.keys()].sort().map((n) => el("option", { value: n })));
  } catch (_) {
    const note = $("#area-note");
    if (note) note.textContent = "Couldn't load the list of countries (it comes from the internet); you can still draw a shape.";
  }
}

// Outer rings as [lat, lon], thinned to about `budget` corners over a whole country.
function geometryRings(g, budget = 3000) {
  const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
  const outer = polys.map((p) => p[0]);
  const total = outer.reduce((n, r) => n + r.length, 0);
  const keep = Math.max(1, Math.ceil(total / budget));
  return outer.filter((r) => r.length >= 4).map((r) => r.filter((_, i) => i % keep === 0).map(([lon, lat]) => [lat, lon])).filter((r) => r.length >= 3);
}

function choosePlace(input) {
  const g = AREA.places?.get(input.value);
  if (g) setArea(geometryRings(g), input.value);
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && AREA.drawing) { AREA.drawing = []; finishDrawing(); }
});

async function saveSettings(form, keys) {
  const data = Object.fromEntries(new FormData(form));
  const body = {};
  if (keys.includes("centre") && AREA.area !== undefined) {
    body.area = AREA.area;
    body.area_name = AREA.name;
  }
  for (const k of keys) {
    if (!(k in data)) continue;
    if (k === "centre") {
      const ll = parseLatLon(data.centre);
      if (!ll) return alert("Centre: two numbers, latitude then longitude");
      [body.lat, body.lon] = ll;
    } else if (k === "truth") {
      const ll = data.truth.trim() ? parseLatLon(data.truth) : [null, null];
      if (!ll) return alert("Known location: two numbers, or leave it empty");
      [body.truth_lat, body.truth_lon] = ll;
    } else if (k === "start") {
      const text = data.start.trim();
      if (!text) { body.start_unix = null; continue; }
      const t = Date.parse(text.replace(" ", "T") + (/[zZ]|[+-]\d\d:?\d\d$/.test(text) ? "" : "Z"));
      if (!isFinite(t)) return alert("Recording start: a UTC time like 2026-11-02 14:22:10, or leave it empty");
      body.start_unix = t / 1000;
      body.date = new Date(t).toISOString().slice(0, 10);
    } else if (k === "date" || k === "search_mode") body[k] = data[k];
    else body[k] = Number(data[k]);
  }
  try {
    S.p = await api("PATCH", projectPath("/settings"), body);
    if ("area" in body) { AREA.area = undefined; AREA.name = null; }
    render();
    loadProjects();
  } catch (e) { alert(e.message); }
}

// --- 2 passes --------------------------------------------------------------------------------

const overview = new SpecView($("#overview"), {
  axes: false, vmaxScale: 1.8,
  drawOverlay: (o, v) => {
    drawGaps(o, v, v.H);
    for (const p of S.p?.passes || []) {
      if (p.kind === "harmonic") continue;
      o.fillStyle = STATUS_COLOR[p.status];
      o.fillRect(v.x(p.t_heard) - 1, 0, 3, v.H);
    }
    const m = spectro.view;
    o.strokeStyle = "#fff"; o.lineWidth = 1.5;
    o.strokeRect(v.x(m.t0), v.y(m.f1), v.x(m.t1) - v.x(m.t0), v.y(m.f0) - v.y(m.f1));
  },
  onClick: (t) => {
    const v = spectro.view, half = (v.t1 - v.t0) / 2;
    spectro.setView({ t0: t - half, t1: t + half });
    if (S.listening) playFrom(t);
  },
});

const spectro = new SpecView($("#spectro"), {
  interactive: true,
  onChange: () => overview.draw(),
  drawOverlay: (o, v) => drawPasses(o, v),
  onClick: (t, f) => {
    if (S.listening) { playFrom(t); return; }
    if (S.tracing) { S.guide.push([t, f]); spectro.draw(); return; }
    const id = nearestPass(spectro, t, f);
    if (id) selectPass(id, false);
  },
});

// --- listening: with the speaker on, a click on the spectrogram plays the recording from there, and a
// line runs along with it. The line is an element of its own, so moving it redraws nothing.

const sound = new Audio();
sound.preload = "none";
const playheads = [spectro, overview].map((view) => {
  const line = el("div", { class: "playhead", hidden: true });
  view.el.append(line);
  return { view, line };
});
let headFrame = null, headWas = null;

function listenHint(text) { $("#listen-hint").textContent = text; }

function placeHead(t) {
  for (const { view, line } of playheads) {
    line.hidden = false;
    line.style.transform = `translateX(${view.x(t)}px)`;
  }
}

function hideHead() {
  cancelAnimationFrame(headFrame);
  headFrame = null;
  for (const { line } of playheads) line.hidden = true;
}

function followHead() {
  const t = sound.currentTime, v = spectro.view;
  // Off the right edge as it plays: turn the page, unless it was already out of view (panned away).
  if (headWas !== null && headWas <= v.t1 && t > v.t1) {
    const span = v.t1 - v.t0;
    spectro.setView({ t0: t - 0.05 * span, t1: t + 0.95 * span });
  }
  headWas = t;
  placeHead(t);
  if (!sound.seeking && !sound.paused && sound.readyState >= 3) listenHint(`Playing ${clock(t)}. Click elsewhere to jump there.`);
  headFrame = sound.paused ? null : requestAnimationFrame(followHead);
}

async function playFrom(t) {
  const src = projectPath("/audio");
  if (sound.dataset.src !== src) { sound.dataset.src = src; sound.src = src; }
  t = Math.max(0, t);
  headWas = t;
  placeHead(t);  // at once: the sound may take a moment to arrive
  listenHint("Loading the sound…");
  cancelAnimationFrame(headFrame);
  sound.currentTime = t;
  try {
    await sound.play();
    cancelAnimationFrame(headFrame);
    headFrame = requestAnimationFrame(followHead);
  } catch (e) {
    if (e.name === "AbortError") return;  // another click replaced this one
    hideHead();
    listenHint("This browser can't play the file, or it has moved.");
  }
}

function setListening(on) {
  S.listening = on;
  $("#listen-toggle").classList.toggle("on", on);
  $("#spectro").classList.toggle("listening", on);
  $("#listen-hint").hidden = !on;
  if (on) listenHint("Click the spectrogram to hear it from there.");
  else { sound.pause(); hideHead(); }
}

sound.addEventListener("waiting", () => { if (S.listening) listenHint("Loading the sound…"); });
sound.addEventListener("ended", () => { hideHead(); if (S.listening) listenHint("That's the end of the recording. Click to hear it from somewhere else."); });
sound.addEventListener("error", () => { hideHead(); if (S.listening) listenHint("This browser can't play the file, or it has moved."); });

// The pass whose curve is nearest a click, within 14 px.
function nearestPass(v, t, f, ids = null) {
  let best = null, bestD = 14;
  for (const p of S.p?.passes || []) {
    if (ids && !ids.has(p.id)) continue;
    if (p.kind === "harmonic" && p.id !== S.selectedPass) continue;  // not drawn: nothing to click on
    for (let i = 0; i < p.line_t.length; i += 2) {
      const d = Math.hypot(v.x(p.line_t[i]) - v.x(t), v.y(p.line_f[i]) - v.y(f));
      if (d < bestD) { bestD = d; best = p.id; }
    }
  }
  return best;
}

function showCandCurves() {
  return S.results && cand() && $("#cand-overlay").checked;
}

// Gaps in the flight data, as red bands `h` px tall from the top of a spectrogram view.
function drawGaps(o, v, h, text = false) {
  const t0 = S.p?.settings.start_unix;
  if (t0 == null) return;
  for (const [a, b] of gaps()) {
    if (b - t0 < v.view.t0 || a - t0 > v.view.t1) continue;
    const x0 = Math.max(v.x(a - t0), 0), x1 = Math.min(v.x(b - t0), v.W);
    o.fillStyle = "rgba(239,107,107,.45)";
    o.fillRect(x0, 0, x1 - x0, h);
    if (text && x1 - x0 > 120) {
      o.font = "bold 11px system-ui, sans-serif"; o.fillStyle = "#ffd6d6";
      o.fillText("no flight data", Math.max(x0, 48) + 4, h + 12);  // clear of the frequency labels
    }
  }
}

function drawPasses(o, v) {
  drawGaps(o, v, 5, true);
  if (S.hideCurves || S.peek) { drawGuideHarmonics(o, v); drawGuide(o, v); return; }  // just the spectrogram (and points being placed)
  const close = v.view.t1 - v.view.t0 < 900;
  // The selected candidate's modelled curves, solid in each pass's colour (see step 5).
  const modelled = showCandCurves() ? explainsOf(cand()).filter((e) => e.curve_t && e.share !== 0) : [];
  for (const e of modelled) {
    strokeCurve(o, v, e.curve_t, e.curve_f, { strokeStyle: colorOf(e.pass_id), lineWidth: e.pass_id === S.selectedPass ? 2.6 : 1.8, globalAlpha: 0.95 });
    // Flights heard on the spectrogram (S1, S2...) have no pass to carry their name.
    if (e.heard && e.heard_rec > v.view.t0 && e.heard_rec < v.view.t1) label(o, v, e.pass_id, e.heard_rec, e.f0 * 1.06, colorOf(e.pass_id));
  }
  for (const p of S.p?.passes || []) {
    const sel = p.id === S.selectedPass;
    if (p.kind === "harmonic" && !sel) continue;  // not listed, so not drawn unless opened from its pass
    const color = STATUS_COLOR[p.status];
    if (sel && S.showHarmonics) drawHarmonics(o, v, p);
    if (sel || close) dots(o, v, p.line_t, p.line_f, sel ? "#7fe7ff" : "rgba(127,231,255,.45)", sel ? 1.6 : 1.1, sel ? 1 : 2);
    strokeCurve(o, v, p.curve_t, p.curve_f, { strokeStyle: color, lineWidth: sel ? 2.2 : 1.2, dash: [6, 4], globalAlpha: modelled.length ? 0.6 : 1 });
    if (p.t_range[1] < v.view.t0 || p.t_range[0] > v.view.t1) continue;
    const i = Math.max(0, p.line_t.findIndex((t) => t >= v.view.t0));
    label(o, v, p.id, p.line_t[i], p.line_f[i], modelled.some((e) => e.pass_id === p.id) ? colorOf(p.id) : color);
  }
  drawGuideHarmonics(o, v);
  drawGuide(o, v);
}

// Where a pass's traced line would lie if a plane's harmonics are there: the line itself at each
// multiple of its pitch, nothing read from the spectrogram, so that a person can see whether anything
// bright runs along it.
const HARMONIC_MULTIPLES = [0.5, 2, 3, 4, 5, 6];
const multipleText = (r) => (r === 0.5 ? "x1/2" : `x${r}`);

function drawHarmonics(o, v, p) {
  for (const r of HARMONIC_MULTIPLES) {
    const fs = p.line_f.map((f) => f * r);
    dots(o, v, p.line_t, fs, "#7fe7ff", 1.1, 2);
    const i = Math.max(0, p.line_t.findIndex((t) => t >= v.view.t0));
    if (p.line_t[i] < v.view.t1 && fs[i] > v.view.f0 && fs[i] < v.view.f1) label(o, v, multipleText(r), p.line_t[i], fs[i], "#7fe7ff");
  }
}

// With Show harmonics on, where the points placed so far would lie on a plane's harmonics, dotted.
function drawGuideHarmonics(o, v) {
  if (!S.showHarmonics || S.guide.length < 2) return;
  const guide = [...S.guide].sort((a, b) => a[0] - b[0]);
  o.save();
  o.strokeStyle = "rgba(255,255,255,.55)"; o.lineWidth = 1; o.setLineDash([1, 4]);
  for (const r of HARMONIC_MULTIPLES) {
    o.beginPath();
    guide.forEach(([t, f], i) => (i ? o.lineTo(v.x(t), v.y(f * r)) : o.moveTo(v.x(t), v.y(f * r))));
    o.stroke();
    const [t, f] = guide[0];
    if (t > v.view.t0 && t < v.view.t1 && f * r > v.view.f0 && f * r < v.view.f1) label(o, v, multipleText(r), t, f * r, "rgba(255,255,255,.8)");
  }
  o.restore();
}

// The points placed while tracing by hand.
function drawGuide(o, v) {
  if (S.guide.length) {
    o.strokeStyle = "#fff"; o.lineWidth = 1; o.setLineDash([3, 3]);
    o.beginPath();
    S.guide.forEach(([t, f], i) => (i ? o.lineTo(v.x(t), v.y(f)) : o.moveTo(v.x(t), v.y(f))));
    o.stroke(); o.setLineDash([]);
    o.fillStyle = "#fff";
    for (const [t, f] of S.guide) {
      o.beginPath(); o.arc(v.x(t), v.y(f), 4, 0, 2 * Math.PI); o.fill();
    }
  }
}

// The last search's best flight match for a pass (see project._matches), or undefined.
const matchOf = (id) => S.results?.matches?.[id];

function setHideCurves(hide) {
  S.hideCurves = hide;
  $("#hide-curves").classList.toggle("on", hide);
  $("#hide-curves").textContent = hide ? "Show curves" : "Hide curves";
  spectro.draw();
}

function sortedPasses() {
  const by = {
    time: (a, b) => a.t_heard - b.t_heard,
    flights: (a, b) => ((matchOf(b.id)?.share ?? -1) - (matchOf(a.id)?.share ?? -1)) || (b.confidence - a.confidence),
    confidence: (a, b) => (b.confidence - a.confidence) || (a.t_heard - b.t_heard),
  };
  return S.p.passes.filter((p) => p.kind !== "harmonic").sort(by[S.passSort]);  // a harmonic goes with its main pass
}

// The other lines traced for the aircraft that made pass `p`, which aren't in the list: [{ r, q }], with
// r their pitch over p's (1 when the same line was traced twice).
function otherLinesOf(p) {
  return S.p.passes.filter((q) => q.harmonic_of === p.id).map((q) => {
    const at = new Map(p.line_t.map((t, i) => [Math.round(t * 2), p.line_f[i]]));
    const ratios = q.line_t.map((t, i) => q.line_f[i] / at.get(Math.round(t * 2))).filter(Number.isFinite).sort((a, b) => a - b);
    return { r: ratios.length ? ratios[ratios.length >> 1] : q.f0 / p.f0, q };
  }).sort((a, b) => a.r - b.r);
}

const ratioText = (r) => (Math.abs(r - 1) < 0.02 ? "the same line" : `x${+r.toFixed(2)}`);
try { S.showHarmonics = !!localStorage.getItem("doppler.showHarmonics"); } catch (_) { /* private mode */ }

function matchCell(p) {
  const m = matchOf(p.id), g = passGap(p);
  if (g && !m?.share) return el("td", { class: "bad", title: gapNote(g) }, "no data");
  if (!S.results?.matches) return el("td", { class: "muted", title: "Set the recording's start time and run the search to match passes with flights" }, "–");
  if (!m) return el("td", { class: "muted", title: p.kind === "harmonic" ? "a harmonic: its main pass is matched" : "not in the last search (traced since, or too flat)" }, "–");
  const who = m.flight ? `${m.registration || m.flight}${m.type ? " " + m.type : ""}` : "no flight follows half the line";
  return el("td", { class: m.share >= EXPLAINED_AT ? "good" : m.share > 0 ? "meh" : "muted", title: who },
    m.share ? `${Math.round(m.share * 100)}%` : "none");
}

async function setStatus(id, status) {
  await api("PATCH", projectPath(`/passes/${id}`), { status });
  await refresh();
  loadProjects();
}

function statusSeg(p) {
  return el("span", { class: "seg small" }, ...[["accepted", "Use"], ["undecided", "?"], ["rejected", "No"]].map(([st, text]) =>
    el("button", {
      class: p.status === st ? `sel-${st}` : "", title: { accepted: "Accept (A)", undecided: "Undecided (U)", rejected: "Reject (R)" }[st],
      onclick: (e) => { e.stopPropagation(); setStatus(p.id, st); },
    }, text)));
}

function renderPasses() {
  const passes = sortedPasses();
  const n = (st) => S.p.passes.filter((x) => x.status === st).length;
  $("#pass-count").textContent = passes.length
    ? `${passes.length} passes · ${n("accepted")} accepted, ${n("undecided")} undecided` : "Passes";
  $("#pass-count").title = "Lines that are harmonics of another pass aren't listed: they are linked from its details.";
  const tbody = $("#pass-table tbody");
  tbody.replaceChildren(...passes.map((p) => el("tr", {
    class: p.id === S.selectedPass ? "selected" : "", "data-id": p.id, title: p.notes.join("\n"), onclick: () => selectPass(p.id, true),
  },
    el("td", {}, el("b", {}, p.id), p.source === "manual" ? " ✎" : ""),
    el("td", {}, el("span", { class: `badge ${KIND_CLASS[p.kind]}` }, p.kind),
      el("span", { class: "conf", title: `confidence ${Math.round(p.confidence * 100)}%` }, el("div", { style: `width:${Math.round(p.confidence * 100)}%` }))),
    el("td", {}, clock(p.t_heard)),
    matchCell(p),
    el("td", {}, statusSeg(p)),
  )));
  if (!passes.length) tbody.replaceChildren(el("tr", {}, el("td", { colspan: 5, class: "hint" }, S.p.duration ? "No passes yet: press Find passes." : "Make the spectrogram first (step 1).")));
  const c = cand();
  $("#cand-overlay-label").hidden = !c;
  if (c) $("#cand-overlay-text").textContent = `Curves of candidate #${c.rank}`;
  renderPassDetail();
  overview.draw(); spectro.draw();
}

function renderPassDetail() {
  const p = passById(S.selectedPass);
  const box = $("#pass-detail");
  if (!p) { box.replaceChildren(heardDetail(S.selectedPass) || ""); return; }
  const c = cand();
  const e = c && explainsOf(c).find((x) => x.pass_id === p.id);
  const action = p.source === "manual"
    ? el("button", { title: "Delete this hand-traced pass", onclick: async () => {
        await api("DELETE", projectPath(`/passes/${p.id}`));
        S.selectedPass = null; await refresh();
      } }, "Delete")
    : el("button", { title: "Trace this pass again by hand", onclick: () => startTrace(p.id) }, "Retrace");
  let note = null;
  if (c) {
    note = el("div", { class: "cand-note" },
      el("span", { class: "swatch", style: `background:${colorOf(p.id)}` }), " Candidate ",
      el("a", { href: "#", onclick: (ev) => { ev.preventDefault(); S.locPass = p.id; showPage("locations"); } }, `#${c.rank}`), ": ",
      !e ? "this pass was not part of its check (not accepted, or not voting)."
        : e.flight ? `${e.registration || e.flight}${e.type ? " " + e.type : ""} follows ${Math.round((e.share ?? 0) * 100)}% of the line` +
          (e.cpa_m != null ? `, closest ${km(e.cpa_m)} at ${e.speed} m/s, ${e.alt_m} m up` : "") + "."
          : "no flight explains this pass there.");
  }
  box.replaceChildren(...[
    el("div", { class: "row" }, el("b", {}, p.id), el("span", { class: `badge ${KIND_CLASS[p.kind]}` }, p.kind),
      el("span", { class: "muted" }, `confidence ${Math.round(p.confidence * 100)}%`), el("span", { class: "spacer" }), statusSeg(p), action),
    el("div", { class: "row" }, el("button", {
      class: S.showHarmonics ? "on" : "", title: "Draw this traced line where its harmonics would lie, at multiples of its pitch, to see whether anything bright runs along them",
      onclick: () => {
        S.showHarmonics = !S.showHarmonics;
        try { localStorage.setItem("doppler.showHarmonics", S.showHarmonics ? "1" : ""); } catch (_) { /* private mode */ }
        renderPassDetail();
        spectro.draw();
      },
    }, S.showHarmonics ? "Hide harmonics" : "Show harmonics")),
    el("dl", {},
      el("dt", {}, "Heard at"), el("dd", {}, clock(p.t_heard)), el("dt", {}, "Pitch"), el("dd", {}, `${p.f0.toFixed(1)} Hz`),
      ...(p.kind === "possibly turning"
        ? [el("dt", {}, "Speed"), el("dd", { class: "muted", title: `straight-pass fit: ${Math.round(p.speed)} m/s` }, "uncertain"),
           el("dt", {}, "Closest"), el("dd", { class: "muted", title: `straight-pass fit: ${km(p.dist)}` }, "uncertain")]
        : [el("dt", {}, "Speed"), el("dd", {}, `${Math.round(p.speed)} m/s`), el("dt", {}, "Closest"), el("dd", {}, km(p.dist))]),
    ),
    p.notes.length ? el("ul", {}, ...p.notes.map((n) => el("li", {}, n))) : null,
    p.kind === "harmonic" ? el("div", { class: "cand-note" }, "One of the lines of ",
      el("a", { href: "#", onclick: (ev) => { ev.preventDefault(); selectPass(p.harmonic_of, true); } }, p.harmonic_of), ": not listed, and not searched with on its own.") : null,
    otherLinesOf(p).length ? el("div", { class: "cand-note" }, "Other lines of this plane: ", ...otherLinesOf(p).flatMap(({ r, q }, k) => [
      k ? ", " : "", el("a", { href: "#", title: `The line traced as ${q.id}`, onclick: (ev) => { ev.preventDefault(); selectPass(q.id, true); } }, `${ratioText(r)} (${q.id})`)])) : null,
    passGap(p) ? el("div", { class: "cand-note bad" }, gapNote(passGap(p))) : null,
    passGap(p) && !matchOf(p.id)?.share ? null : matchNote(p),  // "no flight follows it" says nothing then
    note,
  ].filter(Boolean));  // the DOM writes a null child as the text "null"
}

// The detail box for a flight heard on the spectrogram (opened from step 5): not a pass, but its
// modelled curve shows where to look for one.
function heardDetail(id) {
  const c = cand(), e = c && explainsOf(c).find((x) => x.pass_id === id && x.heard);
  if (!e) return null;
  return el("div", {},
    el("div", { class: "row" }, el("b", { style: `color:${colorOf(id)}` }, id), el("span", { class: "muted" }, "heard on the spectrogram"),
      el("span", { class: "spacer" }),
      el("button", { title: "Trace its line by hand, to add it as a pass", onclick: () => startTrace() }, "Trace")),
    el("div", { class: "cand-note" }, el("span", { class: "swatch", style: `background:${colorOf(id)}` }),
      ` Candidate #${c.rank}: ${entryName(e)}, heard ${clock(e.heard_rec)}` +
      (e.cpa_m != null ? `, closest ${km(e.cpa_m)} at ${e.speed} m/s, ${e.alt_m} m up` : "") +
      `. Not a pass: its modelled curve (solid) gathers the spectrogram's peaks. Trace the line it follows to add it, then press Re-check passes in step 5.`));
}

function matchNote(p) {
  const m = matchOf(p.id);
  if (!m) return null;
  return el("div", { class: "cand-note" }, "Best flight anywhere in the area: ", m.flight
    ? `${m.registration || m.flight}${m.type ? " " + m.type : ""} follows ${Math.round(m.share * 100)}% of the line, `
      + `heard from ${m.lat.toFixed(4)}, ${m.lon.toFixed(4)} with the recording started ${m.start_utc.slice(11, 19)}.`
    : "no flight follows even half the line at the known start: a bird, a vehicle or a plane without ADS-B.");
}

function selectPass(id, zoom) {
  S.selectedPass = id;
  const p = passById(id);
  if (p && zoom) spectro.setView(passWindow(p));
  renderPasses();
  $(`#pass-table tr[data-id="${id}"]`)?.scrollIntoView({ block: "nearest" });
}

let retraceId = null;
function startTrace(replace = null) {
  if (S.listening) setListening(false);
  S.tracing = true; S.guide = []; retraceId = replace;
  $("#trace-toggle").classList.add("on");
  $("#trace-tools").hidden = false;
  $("#spectro").classList.add("tracing");
  $("#trace-hint").textContent = replace ? `Click along ${replace}'s curve, from start to end.` : "Click along the curve, from start to end.";
  if (replace) selectPass(replace, true);
  spectro.draw();
}

function stopTrace() {
  S.tracing = false; S.guide = []; retraceId = null;
  $("#trace-toggle").classList.remove("on");
  $("#trace-tools").hidden = true;
  $("#spectro").classList.remove("tracing");
  spectro.draw();
}

// --- 3 flights and 4 search ------------------------------------------------------------------

function renderFlights() {
  const a = S.p.adsb;
  $("#flights-summary").replaceChildren(a
    ? el("p", {}, `${a.aircraft} aircraft came within ${Math.round(a.radius_km)} km of the area's centre on ${S.p.settings.date}.`)
    : el("p", { class: "hint" }, "No flight data yet for this day and area."), coverageNote("map") || "");
  if (S.page === "flights") renderFlightsMap();
}

function renderSearch() {
  const acc = S.p.passes.filter((p) => p.status === "accepted");
  const pairs = acc.length * (acc.length - 1) / 2;
  const known = S.p.settings.start_unix != null;
  const voters = S.p.passes.filter((p) => p.status !== "rejected" && p.kind !== "harmonic"
    && Math.log(Math.max(...p.line_f) / Math.min(...p.line_f)) >= 0.05);
  const how = known
    ? ["the spectrogram is stacked along every flight's curve: this needs no passes",
       `${voters.length} passes not rejected vote for places at the known start (±${S.p.settings.start_tol_s} s)`,
       `each accepted pass is searched on its own` + (pairs ? `, and ${pairs} pair${pairs === 1 ? "" : "s"} heard then` : "")]
    : [`${pairs} pair${pairs === 1 ? "" : "s"} of accepted passes`, "set the recording's start time in step 1 to let every pass vote"];
  $("#search-passes").replaceChildren(
    el("h3", {}, "Searching with"),
    acc.length >= 2 || known
      ? el("div", {}, el("ul", { class: "hint" }, ...how.map((h) => el("li", {}, h))),
          ...acc.map((p) => el("span", { class: "chip", title: `${p.f0.toFixed(0)} Hz` },
            el("span", { class: "swatch", style: `background:${colorOf(p.id)}` }), ` ${p.id} at ${clock(p.t_heard)}`)))
      : el("p", { class: "hint" }, known ? "No passes accepted yet: accept at least one in step 2."
        : "Accept at least two passes in step 2, or one if you set the recording's start time in step 1."),
    ...(S.p.adsb ? [coverageNote()].filter(Boolean) : [el("p", { class: "error" }, "Get the flight data first (step 3).")]),
  );
  const s = S.p.settings;
  $("#search-form").replaceChildren(
    el("h3", {}, "Settings"),
    field("spacing_m", "Grid spacing (m)", s.spacing_m, { type: "number", step: "any", title: SPACING_NOTE }),
    field("audible_km", "Farthest a plane is heard (km)", s.audible_km, { type: "number", step: "any" }),
    field("max_alt_m", "Highest a plane is heard (m)", s.max_alt_m, { type: "number", step: "any" }),
    el("label", {}, "Search",
      el("select", { name: "search_mode", title: MODE_NOTE },
        el("option", { value: "thorough", selected: s.search_mode !== "fast" }, "Thorough"),
        el("option", { value: "fast", selected: s.search_mode === "fast" }, "Fast"))),
    el("p", { class: "hint wide" }, SPACING_NOTE),
    el("p", { class: "hint wide" }, MODE_NOTE),
  );
  renderLastSearch();
}

const MODE_NOTE = "Search: Fast leaves out the pairs of a straight pass and a turning one, so it can miss the place when few passes are straight. "
  + "It is much quicker across a country, and makes no difference if every accepted pass is straight.";

const SPACING_NOTE = "Grid spacing: a coarser grid is quicker, and each promising flight pair is then moved to where it fits best. "
  + "But too coarse can lose the right answer altogether, and with a start time it can land a few hundred metres further off.";

function renderLastSearch() {
  const R = S.results, box = $("#search-last");
  if (!R) { box.replaceChildren(el("p", { class: "hint" }, "No search yet.")); return; }
  const rows = [
    ...R.event_pairs.map((e) => [`${e.a} → ${e.b}`, e.evidence == null ? "–" : Math.round(e.evidence), e.flight_pairs, e.funnel["+ silence"]]),
    ...(R.singles || []).map((e) => [`${e.key} alone, at the known start`, "–", `${e.flights} flights`, e.funnel["+ speed & distance"]]),
    ...(R.votes ? [[`${R.votes.passes.length} passes voting, at the known start`, "–", "–", `${R.votes.places} places`]] : []),
    ...(R.stack ? [["The spectrogram stacked along every flight, at the known start", "–", `${R.stack.flights} flights`,
      `${R.stack.places} places`]] : []),
  ];
  box.replaceChildren(
    el("p", {}, `${R.candidates.length} candidates` + (R.seconds != null ? `, found in ${clock(R.seconds)}` : "") +
      (R.explained_over ? `, ranked by how many of ${R.explained_over.length} passes they explain` : "") +
      (R.stack ? `${R.explained_over ? "" : ", ranked"} plus how many flights are heard there on the spectrogram.` : ".")),
    el("table", { class: "data compact" },
      el("thead", {}, el("tr", {}, ...["Searched", "Evidence", "Flights", "Places left"].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...rows.map((r) => el("tr", {}, ...r.map((x) => el("td", {}, x)))))),
    el("h3", { style: "margin-top:12px" }, "Best candidates"),
    el("table", { class: "data compact" },
      el("tbody", {}, ...R.candidates.slice(0, 5).map((c, k) => el("tr", { onclick: () => { S.selectedCand = k; showPage("locations"); } },
        el("td", {}, `#${c.rank}`), el("td", { class: confClass(c), title: CONF_TITLE }, confText(c)), el("td", {}, c.explains ? `explains ${c.explained} of ${c.explains.length}` : ""),
        el("td", {}, `${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}`),
        el("td", {}, c.truth_m != null ? `${km(c.truth_m)} from known` : ""))))),
  );
}

// --- maps ------------------------------------------------------------------------------------

const maps = {};
function baseMap(id) {
  if (maps[id]) { setTimeout(() => maps[id].map.invalidateSize(), 0); return maps[id]; }
  const map = L.map(id, { preferCanvas: true });
  const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "© OpenStreetMap contributors",
  });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19, attribution: "Imagery © Esri",
  });
  osm.addTo(map);
  L.control.layers({ Map: osm, Satellite: sat }).addTo(map);
  fullscreenControl().addTo(map);
  const layer = L.layerGroup().addTo(map);
  maps[id] = { map, layer, fitted: null };
  new ResizeObserver(() => map.invalidateSize()).observe($(`#${id}`));
  return maps[id];
}

// A button under the zoom buttons that shows the map's panel on the whole screen, and back (or Esc).
function fullscreenControl() {
  const Full = L.Control.extend({
    options: { position: "topleft" },
    onAdd(map) {
      const bar = L.DomUtil.create("div", "leaflet-bar leaflet-control");
      const a = L.DomUtil.create("a", "map-full", bar);
      a.href = "#"; a.role = "button"; a.title = "Full screen";
      a.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="#333" stroke-width="2">'
        + '<path d="M1 6V1h5M10 1h5v5M15 10v5h-5M6 15H1v-5"/></svg>';
      L.DomEvent.disableClickPropagation(bar);
      L.DomEvent.on(a, "click", (e) => {
        L.DomEvent.preventDefault(e);
        const host = map.getContainer().parentElement;
        if (document.fullscreenElement) document.exitFullscreen();
        else host.requestFullscreen?.();
      });
      document.addEventListener("fullscreenchange", () => {
        const on = document.fullscreenElement === map.getContainer().parentElement;
        a.title = on ? "Leave full screen (Esc)" : "Full screen";
        setTimeout(() => map.invalidateSize(), 50);
      });
      return bar;
    },
  });
  return new Full();
}

function areaBox(s) {
  const dLat = s.half_km / 111.2, dLon = s.half_km / (111.2 * Math.cos(s.lat * Math.PI / 180));
  return [[s.lat - dLat, s.lon - dLon], [s.lat + dLat, s.lon + dLon]];
}

// The area searched, on a map: the shape when there is one, else the square.
function areaOutline(s, opts = {}) {
  const style = { color: "#6aa9ff", weight: 2, fill: false, ...opts };
  return s.area ? L.polygon(s.area, style) : L.rectangle(areaBox(s), style);
}

function truthMarker(s) {
  return s.truth_lat != null
    ? L.marker([s.truth_lat, s.truth_lon], { title: "known location", icon: L.divIcon({ className: "", html: '<div style="color:#ff5d5d;font-size:22px;line-height:22px;text-shadow:0 0 3px #000">★</div>', iconSize: [22, 22], iconAnchor: [11, 11] }) })
    : null;
}

// Fit a map to `bounds` only when they change (a new project or area), not on every redraw.
function fitOnce(m, key, bounds, opts = {}) {
  if (m.fitted === key) return;
  m.fitted = key;
  setTimeout(() => { m.map.invalidateSize(); m.map.fitBounds(bounds, { padding: [30, 30], ...opts }); }, 0);
}

function renderAreaMap() {
  if (!S.p) return;
  const m = baseMap("area-map");
  m.layer.clearLayers();
  const s = S.p.settings;
  const form = $("#settings-form");
  const ll = parseLatLon(form.elements.centre?.value) || [s.lat, s.lon];
  const half = Number(form.elements.half_km?.value) || s.half_km;
  const shape = areaNow(s);
  if (shape) L.polygon(shape, { color: "#6aa9ff", weight: 2, fillOpacity: 0.08 }).bindTooltip("area to search").addTo(m.layer);
  else {
    L.rectangle(areaBox({ lat: ll[0], lon: ll[1], half_km: half }), { color: "#6aa9ff", weight: 2, fill: false }).bindTooltip("area to search").addTo(m.layer);
    L.circleMarker(ll, { radius: 4, color: "#6aa9ff", fillOpacity: 1 }).addTo(m.layer);
  }
  if (AREA.drawing?.length) {
    L.polyline(AREA.drawing, { color: "#ffd166", weight: 2, dashArray: "5 4" }).addTo(m.layer);
    for (const p of AREA.drawing) L.circleMarker(p, { radius: 4, color: "#ffd166", fillOpacity: 1, interactive: false }).addTo(m.layer);
  }
  truthMarker(s)?.addTo(m.layer);
  fitOnce(m, `${S.name}:${s.lat},${s.lon},${s.half_km}`, s.area ? L.latLngBounds(s.area.flat()) : areaBox(s));
  if (!m.bound) {
    m.bound = true;
    m.map.on("click", (e) => {
      if (AREA.drawing) {
        AREA.drawing.push([e.latlng.lat, e.latlng.lng]);
        renderAreaMap();
        return;
      }
      const input = $("#settings-form").elements.centre;
      if (!input || input.disabled) return;
      input.value = `${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`;
      renderAreaMap();
    });
    m.map.on("dblclick", () => {
      if (!AREA.drawing) return;
      AREA.drawing.pop();  // the second click of the double-click landed on the same corner
      finishDrawing();
    });
  }
}

function renderFlightsMap() {
  if (!S.p) return;
  const m = baseMap("flights-map");
  m.layer.clearLayers();
  const s = S.p.settings;
  const box = areaBox(s);
  areaOutline(s).bindTooltip("area to search").addTo(m.layer);
  const c = S.p.adsb_low;
  if (c) {
    const kx = 111.2 * Math.cos(s.lat * Math.PI / 180), size = c.cell_km;
    for (const { i, j, k } of thinSquares()) {
      const y0 = i * size - s.half_km, x0 = j * size - s.half_km;
      L.rectangle([[s.lat + y0 / 111.2, s.lon + x0 / kx], [s.lat + (y0 + size) / 111.2, s.lon + (x0 + size) / kx]],
        { color: "#9aa1b1", weight: 0, fillColor: "#000", fillOpacity: 0.35 })
        .bindTooltip(`${k || "no"} aircraft logged below ${c.below_m} m above the ground here all day`).addTo(m.layer);
    }
  }
  if (S.p.adsb) L.circle([s.lat, s.lon], { radius: S.p.adsb.radius_km * 1000, color: "#9aa1b1", weight: 1, fill: false, dashArray: "4 4" }).bindTooltip("flight data kept").addTo(m.layer);
  truthMarker(s)?.addTo(m.layer);
  fitOnce(m, `${S.name}:${s.lat},${s.lon},${s.half_km},${!!S.p.adsb}`, S.p.adsb
    ? L.latLng(s.lat, s.lon).toBounds(S.p.adsb.radius_km * 2000) : box);
  setupFlights(m);
}

// --- 3 flights: replay the aircraft over the area ----------------------------------------------

// The aircraft loaded (`data`, for unix times `loaded`), the time shown (`t`) within the slider's
// `range` (the recording +-10 min when its start is known, else the whole UTC day), the map objects
// of each aircraft shown (`objs`, by ICAO address) and the one selected (`sel`).
const FL = { key: null, range: null, loaded: null, data: [], t: null, timer: null, objs: new Map(), sel: null, req: 0, layer: null };
const GAP_S = 90;  // reports further apart than this: the aircraft wasn't seen in between

function flightsRange() {
  const s = S.p.settings;
  if (s.start_unix != null) return [Math.floor(s.start_unix - 600), Math.ceil(s.start_unix + (S.p.duration || 3600) + 600)];
  const d = Date.parse(`${s.date}T00:00:00Z`) / 1000;
  return [d, d + 86400];
}

function setupFlights(m) {
  if (!FL.layer) {
    FL.layer = L.layerGroup().addTo(m.map);
    m.map.on("click", () => { if (FL.sel) { FL.sel = null; clearPlanes(); showFlightsAt(FL.t); } });
  }
  const range = flightsRange();
  const key = `${S.name}|${range}|${S.p.adsb?.region}`;
  if (key !== FL.key) {
    FL.key = key; FL.range = range; FL.loaded = null; FL.data = []; FL.sel = null;
    clearPlanes();
    FL.t = S.p.settings.start_unix ?? range[0] + 12 * 3600;
    const slider = $("#fl-time");
    slider.min = range[0]; slider.max = range[1];
  }
  $(".time-bar").hidden = !S.p.adsb;
  renderTicks();
  if (S.p.adsb) showFlightsAt(FL.t);
}

function clearPlanes() {
  FL.layer?.clearLayers();
  FL.objs.clear();
}

// Load the aircraft around time t (all of the slider's range when it is 6 h or less).
async function loadFlights(t) {
  const [r0, r1] = FL.range, pad = 900, span = 6 * 3600 - 60;
  const need = [Math.max(r0, t - pad), Math.min(r1, t + pad)];
  if (FL.loaded && FL.loaded[0] <= need[0] && FL.loaded[1] >= need[1]) return;
  let w0 = r0, w1 = r1;
  if (r1 - r0 > span) { w0 = Math.max(r0, t - span / 2); w1 = Math.min(r1, w0 + span); w0 = Math.max(r0, w1 - span); }
  const req = ++FL.req, key = FL.key;
  const d = await api("GET", projectPath(`/aircraft?t0=${w0}&t1=${w1}`));
  if (req !== FL.req || key !== FL.key) return;
  FL.data = d.aircraft.map((a) => ({ ...a, t: a.pts.map((x) => x[0]) }));
  FL.loaded = [w0, w1];
  clearPlanes();
  showFlightsAt(FL.t);
}

// Where aircraft `a` was at time T, between its reports: {lat, lon, alt, gs, deg}, or null.
function positionAt(a, T) {
  const t = a.t, n = t.length;
  if (n < 2 || T < t[0] || T > t[n - 1]) return null;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] <= T) lo = mid; else hi = mid; }
  if (t[hi] - t[lo] > GAP_S) return null;
  const f = (T - t[lo]) / (t[hi] - t[lo] || 1), A = a.pts[lo], B = a.pts[hi];
  const deg = Math.atan2((B[2] - A[2]) * Math.cos(A[1] * Math.PI / 180), B[1] - A[1]) * 180 / Math.PI;
  return { lat: A[1] + f * (B[1] - A[1]), lon: A[2] + f * (B[2] - A[2]), alt: A[3] + f * (B[3] - A[3]), gs: A[4] + f * (B[4] - A[4]), deg };
}

// The reports between times `from` and `to`, joined to the current position, split where unseen.
function trail(a, from, to, pos, past) {
  const segs = [[]];
  let prev = null;
  for (let i = 0; i < a.t.length; i++) {
    if (a.t[i] < from || a.t[i] > to) continue;
    if (prev != null && a.t[i] - prev > GAP_S) segs.push([]);
    segs[segs.length - 1].push([a.pts[i][1], a.pts[i][2]]);
    prev = a.t[i];
  }
  if (past) segs[segs.length - 1].push([pos.lat, pos.lon]);
  else segs[0].unshift([pos.lat, pos.lon]);
  return segs.filter((x) => x.length > 1);
}

function planeIcon(color, deg, big) {
  const size = big ? 26 : 20;
  return L.divIcon({ className: "", iconSize: [size, size], iconAnchor: [size / 2, size / 2],
    html: `<div class="plane-icon" style="width:${size}px;height:${size}px;transform:rotate(${deg}deg)">`
      + `<svg viewBox="0 0 24 24" width="${size}" height="${size}"><path fill="${color}" stroke="#000" stroke-width=".8" `
      + 'd="M12 1.5c.8 0 1.3 1 1.3 2.4V9l8.2 4.6v2.1l-8.2-2.4v5.1l2.4 1.8v1.6L12 21l-3.7.8v-1.6l2.4-1.8v-5.1l-8.2 2.4v-2.1L10.7 9V3.9c0-1.4.5-2.4 1.3-2.4z"/></svg></div>' });
}

const planeName = (a) => a.reg || a.icao;

function showFlightsAt(T) {
  if (!S.p?.adsb || !FL.range) return;
  T = Math.min(Math.max(T, FL.range[0]), FL.range[1]);
  FL.t = T;
  $("#fl-time").value = T;
  const s = S.p.settings;
  const utcT = new Date(T * 1000).toISOString().slice(11, 19);
  $("#fl-clock").textContent = `${utcT} UTC` + (s.start_unix != null
    ? (T >= s.start_unix && T <= s.start_unix + S.p.duration ? ` · recording ${clock(T - s.start_unix)}` : " · not recording") : "");
  loadFlights(T).catch((e) => console.warn(e));
  const jets = $("#fl-jets").checked, span = Number($("#fl-trail").value);
  const ref = s.truth_lat != null ? L.latLng(s.truth_lat, s.truth_lon) : L.latLng(s.lat, s.lon);
  const seen = new Set(), rows = [];
  for (const a of FL.data) {
    if (a.jet && !jets) continue;
    const pos = positionAt(a, T);
    if (!pos) continue;
    seen.add(a.icao);
    const sel = a.icao === FL.sel, col = sel ? "#7fe7ff" : a.jet ? "#9aa1b1" : "#ffb347";
    let o = FL.objs.get(a.icao);
    if (!o) {
      o = {
        past: L.polyline([], { color: col, weight: sel ? 3.5 : 2.2, opacity: 0.9, interactive: false }).addTo(FL.layer),
        ahead: L.polyline([], { color: col, weight: sel ? 2.5 : 1.6, opacity: 0.75, dashArray: "4 6", interactive: false }).addTo(FL.layer),
        marker: L.marker([pos.lat, pos.lon], { icon: planeIcon(col, pos.deg, sel), zIndexOffset: sel ? 1000 : 0 })
          .bindTooltip("", { permanent: sel, direction: "right", offset: [10, 0], className: "plane-label" })
          .on("click", () => selectPlane(a.icao, false)).addTo(FL.layer),
      };
      FL.objs.set(a.icao, o);
    }
    o.past.setLatLngs(trail(a, T - span, T, pos, true));
    o.ahead.setLatLngs(trail(a, T, T + span, pos, false));
    o.marker.setLatLng([pos.lat, pos.lon]);
    const svg = o.marker.getElement()?.firstChild;
    if (svg) svg.style.transform = `rotate(${pos.deg}deg)`;
    o.marker.setTooltipContent(`${planeName(a)}${a.type ? " " + a.type : ""} · ${Math.round(pos.alt)} m`);
    rows.push({ a, pos, away: ref.distanceTo([pos.lat, pos.lon]) });
  }
  for (const [icao, o] of FL.objs) {
    if (!seen.has(icao)) { FL.layer.removeLayer(o.past); FL.layer.removeLayer(o.ahead); FL.layer.removeLayer(o.marker); FL.objs.delete(icao); }
  }
  rows.sort((x, y) => x.away - y.away);
  $("#fl-count").textContent = `${rows.length} aircraft in the area at ${utcT}`;
  $("#fl-table tbody").replaceChildren(...rows.map(({ a, pos, away }) => el("tr", {
    class: a.icao === FL.sel ? "selected" : "", title: a.desc || "", onclick: () => selectPlane(a.icao, true),
  },
    el("td", {}, el("span", { class: "swatch", style: `background:${a.jet ? "#9aa1b1" : "#ffb347"}` })),
    el("td", {}, el("b", {}, planeName(a))),
    el("td", {}, a.type || ""),
    el("td", {}, pos.alt ? `${Math.round(pos.alt)} m` : "ground"),
    el("td", {}, `${Math.round(pos.gs)} m/s`),
    el("td", {}, km(away)),
  )));
}

function selectPlane(icao, pan) {
  FL.sel = FL.sel === icao ? null : icao;
  clearPlanes();
  showFlightsAt(FL.t);
  const o = FL.objs.get(icao);
  if (pan && o && FL.sel) maps["flights-map"].map.panTo(o.marker.getLatLng());
}

// Marks on the slider at each pass, when the recording's start is known: click one to go there.
function renderTicks() {
  const box = $("#fl-ticks"), s = S.p.settings;
  if (s.start_unix == null || !FL.range) { box.replaceChildren(); return; }
  const [r0, r1] = FL.range;
  const pct = (u) => (Math.min(Math.max(u, r0), r1) - r0) / (r1 - r0) * 100;
  box.replaceChildren(
    ...gaps().filter(([a, b]) => b > r0 && a < r1).map((g) => el("div", {
      class: "tick-gap", title: `No flight data ${gapText(g)}: adsb.lol has no aircraft at all then`,
      style: `left:${pct(g[0])}%;width:${pct(g[1]) - pct(g[0])}%`,
    })),
    ...S.p.passes.filter((p) => p.kind !== "harmonic").map((p) => el("div", {
      class: "tick-pass", title: `${p.id} heard at ${clock(p.t_heard)} (${p.status})`,
      style: `left:${(s.start_unix + p.t_heard - r0) / (r1 - r0) * 100}%;background:${STATUS_COLOR[p.status]}`,
      onclick: () => showFlightsAt(s.start_unix + p.t_heard),
    })));
}

function playFlights(on = !FL.timer) {
  clearInterval(FL.timer);
  FL.timer = null;
  $("#fl-play").textContent = on ? "❚❚" : "▶";
  if (!on) return;
  FL.timer = setInterval(() => {
    const T = FL.t + Number($("#fl-speed").value) * 0.2;
    if (T >= FL.range[1]) playFlights(false);
    showFlightsAt(T);
  }, 200);
}

function viridis(x) {
  const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  x = Math.min(Math.max(x, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c})`;
}

// --- 5 locations -----------------------------------------------------------------------------

const locOverview = new SpecView($("#loc-overview"), {
  axes: false, vmaxScale: 1.8,
  drawOverlay: (o, v) => {
    for (const e of shownExplains()) {
      const p = passById(e.pass_id);
      if (!p) continue;
      o.fillStyle = colorOf(e.pass_id);
      o.fillRect(v.x(p.t_heard) - 1, 0, 3, v.H);
    }
    const m = locSpectro.view;
    o.strokeStyle = "#fff"; o.lineWidth = 1.5;
    o.strokeRect(v.x(m.t0), v.y(m.f1), v.x(m.t1) - v.x(m.t0), v.y(m.f0) - v.y(m.f1));
  },
  onClick: (t) => {
    const v = locSpectro.view, half = (v.t1 - v.t0) / 2;
    locSpectro.setView({ t0: t - half, t1: t + half });
  },
});

const locSpectro = new SpecView($("#loc-spectro"), {
  interactive: true,
  onChange: () => locOverview.draw(),
  drawOverlay: (o, v) => drawExplains(o, v, shownExplains()),
  onClick: (t, f) => {
    const id = nearestPass(locSpectro, t, f, new Set(shownExplains().map((e) => e.pass_id)));
    if (id) selectLocPass(id, false);
  },
});

// The passes the selected candidate explains (or partly, when asked), with their modelled curves.
function shownExplains() {
  return explainsOf(cand()).filter((e) => e.curve_t && (e.verdict || (S.showPartial ? e.share !== 0 : e.share == null || e.share >= EXPLAINED_AT)))
    .sort((a, b) => heardAt(a) - heardAt(b));
}

function drawExplains(o, v, list, only = null) {
  for (const e of list) {
    if (only && e.pass_id !== only) continue;
    const p = passById(e.pass_id), sel = e.pass_id === S.locPass;
    if (p) dots(o, v, p.line_t, p.line_f, sel || only ? "#7fe7ff" : "rgba(127,231,255,.6)", sel || only ? 1.5 : 1.1, 1);
    strokeCurve(o, v, e.curve_t, e.curve_f, { strokeStyle: colorOf(e.pass_id), lineWidth: sel ? 2.4 : 1.7 });
    if (!only && p && p.t_range[1] > v.view.t0 && p.t_range[0] < v.view.t1) {
      label(o, v, e.pass_id, Math.max(p.t_heard, v.view.t0), Math.max(...p.line_f) * 1.04, colorOf(e.pass_id));
    } else if (!only && !p && e.heard_rec != null && e.heard_rec > v.view.t0 && e.heard_rec < v.view.t1) {
      label(o, v, e.pass_id, e.heard_rec, e.f0 * 1.06, colorOf(e.pass_id));
    }
  }
}

function renderLocations() {
  if (!S.p) return;
  const m = baseMap("locations-map");
  if (!m.bound) {
    m.bound = true;
    mapOptionsControl().addTo(m.map);
    m.map.on("click", (ev) => trySpot(ev.latlng));
  }
  m.layer.clearLayers();
  const s = S.p.settings;
  areaOutline(s, { weight: 1.5, interactive: false }).addTo(m.layer);
  truthMarker(s)?.addTo(m.layer);
  $("#truth-col").hidden = s.truth_lat == null;
  const low = S.p.adsb_low?.covered < LOW_COVERED ? coverageNote() : null;
  const R = S.results;
  $("#loc-coverage").replaceChildren(...[low, R && passesChanged(R) ? el("p", { class: "warn-note" },
    "Passes changed since these candidates were checked. Press Re-check passes to update them: no new search needed.") : null].filter(Boolean));
  if (!R) {
    $("#cand-table tbody").replaceChildren(el("tr", {}, el("td", { colspan: 8, class: "hint" }, "No results yet: run the search (step 4).")));
    $("#explains-table tbody").replaceChildren();
    $("#explains-title").textContent = "Passes";
    $("#explains-note").textContent = "";
    $("#spec-each").replaceChildren();
    fitOnce(m, `${S.name}:area`, areaBox(s));
    return;
  }
  const c = cand();
  const heat = R.heat && mapOpt.heat ? heatOverlay(R, s) : null;
  if (heat) heat.addTo(m.layer);
  // Without a heatmap (no start known, or results from before one), the selected search's points.
  const method = (x) => x?.group ? x.group.members[0].pair : x?.pair;
  const pts = heat || !mapOpt.heat ? [] : R.survivors[method(c)] || R.survivors[method(R.candidates[0])] || [];
  if (pts.length) {
    const lo = pts.reduce((a, p) => Math.min(a, p[2]), Infinity), hi = pts.reduce((a, p) => Math.max(a, p[2]), -Infinity);
    for (const [la, lo_, v] of pts) {
      L.circleMarker([la, lo_], { radius: 2, stroke: false, fillOpacity: c ? 0.35 : 0.75, fillColor: viridis((v - lo) / (hi - lo || 1)), interactive: false }).addTo(m.layer);
    }
  }
  const shown = shownExplains();
  for (const e of shown) if (e.path && c) drawFlight(m.layer, e, c);
  if (S.spotBusy) L.circleMarker([S.spotBusy.lat, S.spotBusy.lon], { radius: 8, color: "#fff", weight: 2, fillOpacity: 0, interactive: false }).addTo(m.layer);
  if (c?.group) {  // where the searches that agree on it each put it
    for (const x of c.group.members) {
      L.circleMarker([x.lat, x.lon], { radius: 4, color: "#fff", weight: 1.5, fill: false, opacity: 0.8,
        interactive: false }).addTo(m.layer);
    }
  }
  const sure = confidentCands(R);
  R.candidates.forEach((x, k) => {
    const selected = k === S.selectedCand;
    if (S.locPass && !selected) return;  // looking at one flight: only its candidate
    if (!selected && !mapOpt.all && !sure.has(k) && !x.kept) return;
    const size = selected ? 26 : 22;
    L.marker([x.lat, x.lon], {
      zIndexOffset: selected ? 1000 : 0, title: `candidate ${x.rank}`,
      icon: L.divIcon({ className: `cand-pin${selected ? " sel" : ""}${x.rank > 9 ? " wide" : ""}`, html: String(x.rank),
        iconSize: [size, size], iconAnchor: [size / 2, size / 2] }),
    }).on("click", () => selectCand(k)).addTo(m.layer);
  });
  fitOnce(m, `${S.name}:area`, areaBox(s));

  renderCandidates();
  renderExplains(c, shown);
  renderSpecs(shown);
}

// Whether the passes are no longer the ones the candidates were last checked against.
function passesChanged(R) {
  if (!R.explained_over) return false;
  const counted = S.p.passes.filter((p) => (R.votes ? p.status !== "rejected" && p.kind !== "harmonic"
    && Math.log(Math.max(...p.line_f) / Math.min(...p.line_f)) >= 0.05 : p.status === "accepted"));
  const was = new Set(R.explained_over);
  return counted.length !== was.size || counted.some((p) => !was.has(p.id) || (R.checked_at && p.traced_at > R.checked_at));
}

// --- the map's heatmap and which candidates it marks ---------------------------------------------

// Shown on the locations map: the heatmap, and every candidate rather than the confident ones.
const mapOpt = (() => {
  try { return Object.assign({ heat: true, all: false }, JSON.parse(localStorage.getItem("doppler.mapOpt") || "{}")); }
  catch { return { heat: true, all: false }; }
})();

function mapOptionsControl() {
  const ctl = L.control({ position: "topright" });
  ctl.onAdd = () => {
    const box = el("div", { class: "leaflet-bar map-opts" });
    for (const [key, text, title] of [
      ["heat", "Heatmap", "How strongly the spectrogram's peaks pile up along flights' curves at each place (stacking), at its best start"],
      ["all", "All candidates", "Mark every candidate, not only the confident ones (the best, and any within 70% of its score)"]]) {
      const input = el("input", { type: "checkbox" });
      input.checked = mapOpt[key];
      input.addEventListener("change", () => {
        mapOpt[key] = input.checked;
        try { localStorage.setItem("doppler.mapOpt", JSON.stringify(mapOpt)); } catch {}
        renderLocations();
      });
      box.append(el("label", { title }, input, ` ${text}`));
    }
    L.DomEvent.disableClickPropagation(box);
    return box;
  };
  return ctl;
}

// Candidates worth a marker: the best, and up to four more scoring at least 70% of it.
function confidentCands(R) {
  const val = (x) => x.combined ?? x.score ?? 0, top = R.candidates.length ? val(R.candidates[0]) : 0;
  const out = new Set();
  R.candidates.forEach((x, k) => { if (k === 0 || (k < 5 && top > 0 && val(x) >= 0.7 * top)) out.add(k); });
  return out;
}

// The stacking score over the area as one image (rows south to north), drawn once per results.
function heatOverlay(R, s) {
  const h = R.heat;
  if (!h._url) {
    const cv = document.createElement("canvas");
    cv.width = h.cols; cv.height = h.rows;
    const g = cv.getContext("2d"), img = g.createImageData(h.cols, h.rows);
    const sorted = h.values.filter((v) => v > 0).sort((a, b) => a - b);
    const vmax = sorted.length ? sorted[Math.floor(0.995 * (sorted.length - 1))] : 1;
    const floor = Math.min(h.floor ?? 0.35 * vmax, 0.8 * vmax);
    h.values.forEach((v, i) => {
      const r = Math.floor(i / h.cols), col = i % h.cols, o = ((h.rows - 1 - r) * h.cols + col) * 4;
      // Only where flights agree: one flight alone gathers its share (`floor`) all along its path,
      // and tinting all of that hid the map without saying anything.
      const x = (Math.min(v, vmax) - floor) / (vmax - floor || 1);
      if (x <= 0) return;
      const [cr, cg, cb] = viridis(x).match(/\d+/g).map(Number);
      img.data.set([cr, cg, cb, Math.round(255 * (0.2 + 0.6 * x))], o);
    });
    g.putImageData(img, 0, 0);
    h._url = cv.toDataURL();
  }
  // Cell edges: half a grid step beyond the outermost points, each image pixel `cell_m` wide.
  const kmLat = 111.2, kmLon = 111.2 * Math.cos(s.lat * Math.PI / 180);
  const west = -(h.half_m + h.spacing_m / 2) / 1000, south = west;
  const bounds = [[s.lat + south / kmLat, s.lon + west / kmLon],
    [s.lat + (south + h.rows * h.cell_m / 1000) / kmLat, s.lon + (west + h.cols * h.cell_m / 1000) / kmLon]];
  return L.imageOverlay(h._url, bounds, { opacity: 1, interactive: false, className: "heat" });
}

// --- trying any spot (click the map): the server checks it as it does the search's candidates -----

async function trySpot(ll) {
  if (S.spotBusy || !S.results) return;
  S.spotBusy = { lat: ll.lat, lon: ll.lng };
  renderLocations();
  try {
    const R = await api("POST", projectPath("/spots"), { lat: Number(ll.lat.toFixed(6)), lon: Number(ll.lng.toFixed(6)) });
    S.results = R;
    S.locPass = null;
    S.selectedCand = R.candidates.findIndex((c) => c.key === R.tried);
  } catch (e) { alert(e.message); }
  S.spotBusy = null;
  renderLocations();
  renderPasses();
}

async function keepSpot(c, keep) {
  try {
    const showing = cand()?.key;
    S.results = await api("POST", projectPath("/spots/keep"), { key: c.key, keep });
    const k = S.results.candidates.findIndex((x) => x.key === showing);
    S.selectedCand = k >= 0 ? k : S.results.candidates.length ? 0 : null;
    if (!keep) S.locPass = null;
  } catch (e) { alert(e.message); }
  renderLocations();
  renderPasses();
}

// One flight on the map: its path over a dark outline (readable on streets and satellite alike),
// arrows every minute in the direction it flew, and where it came closest to the candidate,
// tied to it by a dashed line and labelled with the pass it explains.
function drawFlight(layer, e, c) {
  const col = colorOf(e.pass_id), sel = e.pass_id === S.locPass, faded = S.locPass && !sel;
  const partial = e.share != null && e.share < EXPLAINED_AT;
  const w = sel ? 5 : 3, op = faded ? 0.3 : 1;
  const name = `${e.pass_id}: ${e.registration || e.flight}${e.type ? " " + e.type : ""}`;
  L.polyline(e.path, { color: "#000", weight: w + 3, opacity: 0.6 * op, interactive: false }).addTo(layer);
  // A click on the selected pass's own flight tries the pass from there, as anywhere else on the map.
  const click = () => selectLocPass(e.pass_id, true);
  L.polyline(e.path, { color: col, weight: w, opacity: op, dashArray: partial ? "8 6" : null, bubblingMouseEvents: false })
    .bindTooltip(name, { sticky: true }).on("click", click).addTo(layer);
  for (let i = 6; i < e.path.length; i += 6) {  // path points are 10 s apart
    const [a, b] = [e.path[i - 1], e.path[i]];
    const deg = Math.atan2((b[1] - a[1]) * Math.cos(a[0] * Math.PI / 180), b[0] - a[0]) * 180 / Math.PI;
    L.marker(b, { interactive: false, opacity: op, icon: L.divIcon({ className: "", iconSize: [14, 14], iconAnchor: [7, 7],
      html: `<div class="path-arrow" style="color:${col};transform:rotate(${deg}deg)">▲</div>` }) }).addTo(layer);
  }
  const near = nearestOnPath(e.path, [c.lat, c.lon]);
  L.polyline([[c.lat, c.lon], near], { color: col, weight: 1.5, opacity: 0.9 * op, dashArray: "3 5", interactive: false }).addTo(layer);
  L.circleMarker(near, { radius: sel ? 7 : 5, color: "#000", weight: 1.5, fillColor: col, fillOpacity: op, opacity: op, bubblingMouseEvents: false })
    .bindTooltip(e.pass_id, { permanent: !faded, direction: "right", offset: [6, 0], className: "path-label" })
    .on("click", click).addTo(layer);
}

// The point of a path (list of [lat, lon]) nearest `p`, on its segments, in a local flat frame.
function nearestOnPath(path, p) {
  const k = Math.cos(p[0] * Math.PI / 180);
  const xy = ([la, lo]) => [(lo - p[1]) * k, la - p[0]];
  let best = path[0], bestD = Infinity;
  for (let i = 1; i < path.length; i++) {
    const [ax, ay] = xy(path[i - 1]), [bx, by] = xy(path[i]);
    const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
    const t = len ? Math.min(Math.max(-(ax * dx + ay * dy) / len, 0), 1) : 0;
    const x = ax + t * dx, y = ay + t * dy, d = x * x + y * y;
    if (d < bestD) { bestD = d; best = [p[0] + y, p[1] + x / k]; }
  }
  return best;
}

// A candidate's confidence as shown: never quite 0 or 100, the test recordings being few.
const confText = (c) => (c.confidence == null ? "–" : `${Math.max(1, Math.min(99, Math.round(c.confidence * 100)))}%`);
const confClass = (c) => (c.confidence == null ? "muted" : c.confidence >= 0.9 ? "good" : c.confidence >= 0.5 ? "meh" : "bad");
const CONF_TITLE = "How likely this spot is right, to within about a kilometre, from its lead over the best spot elsewhere "
  + "and the share of accepted passes it explains. Fits? answers change it at once. A guide, tuned on 19 test searches.";

function renderCandidates() {
  const R = S.results, s = S.p.settings, verdicts = S.p.verdicts || {};
  $("#cand-table tbody").replaceChildren(...R.candidates.map((c, k) => {
    const key = `${c.pair}@${c.lat},${c.lon}`;
    const seg = el("span", { class: "seg small" }, ...[["right", "✓", "Right"], ["wrong", "✗", "Wrong"]].map(([v, text, title]) =>
      el("button", { class: verdicts[key] === v ? `sel-${v}` : "", title, onclick: async (e) => {
        e.stopPropagation();
        S.p.verdicts = await api("POST", projectPath("/verdicts"), { key, verdict: verdicts[key] === v ? null : v });
        renderCandidates();
      } }, text)));
    const from = c.pair === "map" ? (c.kept ? "your spot" : "your spot, not kept") : c.group ? `${c.group.members.length} searches agree` : c.voted ? `votes: ${c.voted.join(" ")}`
      : c.pair === "stack" ? "spectrogram" : c.pair.replace("-", " → ");
    const fromTitle = c.group ? groupText(c) : from;
    return el("tr", { class: k === S.selectedCand ? "selected" : "", "data-k": k, onclick: () => selectCand(k),
      title: `score ${c.score.toFixed(1)}\n${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}` },
      el("td", {}, el("b", {}, c.rank)),
      el("td", { class: confClass(c), title: CONF_TITLE }, confText(c)),
      el("td", {}, c.explains ? `${c.explained} of ${c.explains.length}` : "–"),
      el("td", { title: (c.heard || []).map((h) => `${h.pass_id}: ${entryName(h)} · ${heardGrade(h).text}`).join("\n") },
        !c.heard ? "–" : c.heard.some((h) => h.status) ? `${c.heard.filter((h) => h.status === "accepted").length} of ${c.heard.length}` : c.heard.length),
      el("td", { title: fromTitle, style: "max-width:150px;overflow:hidden;text-overflow:ellipsis" }, from),
      el("td", {}, c.start_utc.slice(11, 19)),
      el("td", { hidden: s.truth_lat == null }, c.truth_m != null ? km(c.truth_m) : ""),
      el("td", {}, seg),
    );
  }));
  if (!R.candidates.length) $("#cand-table tbody").replaceChildren(el("tr", {}, el("td", { colspan: 8, class: "hint" }, "The search found no candidates.")));
}

// A merged candidate: which searches found it, where (the white rings on the map), and how it was placed.
function groupText(c) {
  const g = c.group, s = S.p.settings;
  const lines = [
    `Found by ${g.members.length} searches on the same evidence (${g.methods.join(", ")}), up to ${km(g.spread_m)} apart.`,
    c.placed_by === "all evidence" ? "Placed by all their flights' curves together."
      : "Placed where the best of them was: all the curves together explained less.",
    ...g.members.map((x) => `· ${x.pair === "stack" ? "spectrogram" : x.pair}`
      + (s.truth_lat != null && x.truth_m != null ? ` (${km(x.truth_m)} from known)` : "")),
  ];
  return lines.join("\n");
}

function renderExplains(c, shown) {
  if (!c) {
    $("#explains-title").textContent = "Passes";
    $("#explains-note").textContent = "Pick a candidate.";
    $("#explains-table tbody").replaceChildren();
    return;
  }
  $("#explains-title").textContent = `#${c.rank} · ${c.lat.toFixed(5)}, ${c.lon.toFixed(5)} · started ${c.start_utc.slice(11, 19)}`;
  $("#explains-title").title = c.group ? groupText(c) : "";
  const all = explainsOf(c);
  const missing = all.filter((e) => e.flight && !e.curve_t).length;
  const proven = c.explained ?? all.filter((e) => e.share != null && e.share >= EXPLAINED_AT).length;
  const spotActions = c.pair !== "map" ? []
    : c.kept ? [" ", el("button", { title: "Remove this spot from the candidates", onclick: () => keepSpot(c, false) }, "Remove")]
    : [" ", el("button", { class: "primary", title: "Keep it among the candidates, through later tries and searches", onclick: () => keepSpot(c, true) }, "Keep"), " ",
       el("button", { title: "Forget this spot", onclick: () => keepSpot(c, false) }, "Discard")];
  const nPasses = all.filter((e) => !e.heard).length, nHeard = (c.heard || []).length;
  const nSeen = (c.heard || []).filter((h) => h.status === "accepted").length;
  $("#explains-note").replaceChildren(
    c.confidence != null ? el("span", { title: CONF_TITLE }, "Confidence ", el("b", { class: confClass(c) }, confText(c)),
      S.p.settings.half_km >= 500 ? " (less reliable over an area this wide). " : ". ") : "",
    c.explains ? `Explains ${proven} of ${nPasses} ${S.results.votes ? "voting" : "accepted"} passes` : "",
    S.results.stack ? `${c.explains ? "; " : ""}${nHeard} flight${nHeard === 1 ? "" : "s"} heard on the spectrogram`
      + ((c.heard || []).some((h) => h.status) ? `, ${nSeen} showing up there` : "") : "",
    c.explains || S.results.stack ? "." : "",
    missing ? el("span", { class: "meh" }, ` ${missing} explained without a curve saved: press Re-check passes.`) : "",
    S.spotBusy ? el("span", { class: "muted" }, " Checking that spot…") : el("span", { class: "muted" }, " Click the map to check any spot."),
    ...spotActions);
  const shownIds = new Set(shown.map((e) => e.pass_id));
  const rows = [...all].sort((a, b) => (!!a.heard - !!b.heard)
    || ((b.share ?? 1) >= EXPLAINED_AT) - ((a.share ?? 1) >= EXPLAINED_AT) || heardAt(a) - heardAt(b));
  $("#explains-table tbody").replaceChildren(...rows.map((e) => {
    const p = passById(e.pass_id), g = e.heard ? heardGrade(e) : null;
    const pct = e.heard ? g.label : e.share == null ? "found" : `${Math.round(e.share * 100)}%`;
    return el("tr", { class: (e.pass_id === S.locPass ? "selected " : "") + (shownIds.has(e.pass_id) ? "" : "dim"), "data-id": e.pass_id,
      onclick: () => selectLocPass(e.pass_id, true),
      title: p ? p.notes.join("\n") : e.heard ? `Not a pass: its curve gathers the spectrogram's peaks (evidence ${e.z} of 4). ${g.text}` : "" },
      el("td", {}, el("span", { class: "swatch", style: `background:${colorOf(e.pass_id)}` })),
      el("td", {}, el("b", {}, e.pass_id), p ? ` ${clock(p.t_heard)}` : e.heard_rec != null ? ` ${clock(e.heard_rec)}` : "",
        p && p.status !== "accepted" ? el("span", { class: "muted", title: `${p.status === "rejected" ? "Rejected" : "Undecided"} in step 2, for the whole project` },
          p.status === "rejected" ? " ✗" : " ?") : ""),
      el("td", { class: g ? g.cls : e.share == null || e.share >= EXPLAINED_AT ? "good" : e.share > 0 ? "meh" : "muted", title: g ? g.text : "" }, pct),
      el("td", { style: "max-width:120px;overflow:hidden;text-overflow:ellipsis", title: e.desc || "" },
        e.flight ? `${e.registration || e.flight}${e.type ? " " + e.type : ""}`
          : passGap(p) ? el("span", { class: "bad", title: gapNote(passGap(p)) }, "no flight data")
            : el("span", { class: "muted" }, "no flight fits")),
      el("td", { title: e.cpa_m != null ? `closest ${km(e.cpa_m)} at ${e.speed} m/s, ${e.alt_m} m up, heard ${e.heard_utc.slice(11, 19)} UTC` : "" },
        e.cpa_m != null ? km(e.cpa_m) : ""),
      el("td", {}, e.status && e.flight ? fitSeg(e) : ""),
    );
  }));
}

// A heard flight's automatic grade (visible.curve_visibility): does its curve follow a line one can see?
function heardGrade(h) {
  const nums = h.best60 != null ? ` Best minute: ${Math.round(h.best60 * 100)}% of frames on a line along the curve, `
    + `against ${Math.round(h.chance * 100)}% for copies slid up and down in pitch (at most ${Math.round(h.chance_max * 100)}%).` : "";
  const by = h.verdict ? " Judged by hand." : "";
  if (h.grade === "visible") return { label: "seen", cls: "good", text: `Its curve shows on the spectrogram.${nums}${by}` };
  if (h.grade === "absent") return { label: "not seen", cls: "muted", text: `Its curve doesn't show on the spectrogram.${nums}${by}` };
  if (h.grade === "unsure") return { label: "unsure", cls: "meh", text: `Unclear whether its curve shows.${h.why ? " (" + h.why + ")" : ""}${nums}${by}` };
  return { label: "heard", cls: "good", text: "Not graded yet." };
}

// Yes / ? / No at this candidate only: does this flight's curve fit a traced pass, or show up on the
// spectrogram? Faded while it is the automatic grade; a click makes it a verdict that ranks the
// candidates again at once, and a click on the verdict goes back to the grade.
function fitSeg(e) {
  const [yes, no] = e.heard ? ["Its curve shows up", "Its curve doesn't show up"] : ["The curve fits the traced line", "The curve doesn't fit the traced line"];
  return el("span", { class: "seg small" + (e.verdict ? "" : " auto"),
    title: e.verdict ? "Your verdict, for this candidate only: click it again to go back to the automatic grade" : "Automatic grade: click to judge it yourself, for this candidate only" },
  ...[["accepted", "Yes", `${yes}: counts in full`], ["undecided", "?", "Unsure: counts half"], ["rejected", "No", `${no}: counts nothing`]].map(([st, text, title]) =>
    el("button", { class: e.status === st ? `sel-${st}` : "", title, onclick: (ev) => {
      ev.stopPropagation();
      const seg = ev.currentTarget.parentElement, status = e.verdict === st ? null : st;
      // Ranking again takes a moment on large results: show the click now.
      seg.classList.toggle("auto", !status);
      for (const b of seg.children) b.className = "";
      if (status) ev.currentTarget.className = `sel-${status}`;
      setFit(e, status);
    } }, text)));
}

// One at a time: each request reads and saves the whole project, so two at once could lose one.
let fitQueue = Promise.resolve();
function setFit(e, status) {
  const key = cand().key;
  fitQueue = fitQueue.then(async () => {
    try {
      const results = await api("POST", projectPath(e.heard ? "/heard" : "/fit"), { key, pass_id: e.pass_id, flight: e.flight, status });
      const showing = cand()?.key;  // the candidates may have moved, and another may be selected by now
      S.results = results;
      if (showing != null) S.selectedCand = Math.max(0, results.candidates.findIndex((c) => c.key === showing));
    } catch (err) { alert(err.message); }
    renderLocations();
    renderPasses();
    loadProjects();
  });
}

// One small spectrogram per pass ("Each pass"), or one for the whole recording.
let cells = [];
function renderSpecs(shown) {
  $("#open-in-passes").disabled = !S.locPass;
  $("#spec-each").hidden = S.specMode !== "each";
  $("#spec-one").hidden = S.specMode !== "one";
  if (S.specMode === "one") {
    locSpectro.resize(); locOverview.resize(); locSpectro.request(); locOverview.request();
    locSpectro.draw(); locOverview.draw();
    return;
  }
  const grid = $("#spec-each");
  const key = `${S.name}|${S.selectedCand}|${shown.map((e) => e.pass_id + e.flight).join(",")}|${S.vmax}`;
  if (grid.dataset.key === key) { cells.forEach((x) => { x.cell.classList.toggle("selected", x.id === S.locPass); x.view.draw(); }); return; }
  grid.dataset.key = key;
  if (!shown.length) {
    cells = [];
    grid.style.gridTemplateColumns = "";
    grid.replaceChildren(el("div", { class: "empty" }, cand()
      ? "This candidate explains none of the passes well. Tick “partial matches too” to see its nearest flights."
      : "Pick a candidate."));
    return;
  }
  cells = shown.map((e) => {
    const p = passById(e.pass_id);
    const box = el("div", { class: "spec" });
    const cell = el("div", { class: "cell" + (e.pass_id === S.locPass ? " selected" : ""),
      title: "Click to select (again to unselect), double-click to open in step 2; scroll to zoom out along the modelled curve, drag to pan",
      ondblclick: () => openInPasses(e.pass_id) },
      box, el("div", { class: "cap" }, el("b", { style: `color:${colorOf(e.pass_id)}` }, e.pass_id),
        ` ${entryName(e)} · ${e.heard ? `heard on the spectrogram, ${heardGrade(e).label} · ` : e.share == null ? "" : Math.round(e.share * 100) + "% · "}${km(e.cpa_m)}`));
    const view = new SpecView(box, { interactive: true, drawOverlay: (o, v) => drawExplains(o, v, [e], e.pass_id),
      onClick: () => selectLocPass(e.pass_id, true) });  // a click, not the end of a drag
    view.limits = { t0: 0, t1: S.p.duration, f0: 20, f1: 1000 };
    view.view = entryWindow(e);
    return { id: e.pass_id, cell, view };
  });
  grid.replaceChildren(...cells.map((x) => x.cell));
  layoutGrid();
}

// Columns and row height so every panel fits the space without scrolling, as wide as can be.
function layoutGrid() {
  const grid = $("#spec-each"), n = cells.length;
  if (!n || grid.hidden) return;
  const W = grid.clientWidth, H = grid.clientHeight, gap = 6;
  let best = { cols: 1, h: 0, score: -1 };
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const w = (W - gap * (cols - 1)) / cols, h = (H - gap * (rows - 1)) / rows;
    const score = Math.min(w / 1.7, h);  // panels about 1.7 times as wide as tall look best
    if (score > best.score) best = { cols, h, score };
  }
  const h = Math.max(best.h, 110);
  grid.style.gridTemplateColumns = `repeat(${best.cols}, minmax(0, 1fr))`;
  cells.forEach((x) => (x.cell.style.height = `${Math.floor(h)}px`));
}
new ResizeObserver(() => layoutGrid()).observe($("#spec-each"));

function selectCand(k) {
  if (k === S.selectedCand) return unselectLoc(true);
  S.selectedCand = k;
  if (S.locPass && !explainsOf(cand()).some((e) => e.pass_id === S.locPass)) S.locPass = null;
  renderLocations();
  renderPasses();
  $(`#cand-table tr[data-k="${k}"]`)?.scrollIntoView({ block: "nearest" });
}

function selectLocPass(id, zoom) {
  if (id === S.locPass) return unselectLoc(false);
  S.locPass = id;
  const p = passById(id);
  if (zoom && p && S.specMode === "one") locSpectro.setView(passWindow(p));
  renderLocations();
  $(`#explains-table tr[data-id="${id}"]`)?.scrollIntoView({ block: "nearest" });
}

// Unselect the flight (or, with `candidate` or when no flight is selected, the candidate too).
function unselectLoc(candidate) {
  if (!S.locPass || candidate) S.selectedCand = null;
  S.locPass = null;
  renderLocations();
  renderPasses();
}

function openInPasses(id) {
  S.selectedPass = id;
  showPage("passes");
  if (passById(id)) { selectPass(id, true); return; }
  // A flight heard on the spectrogram (S1, S2...) is no pass yet: show its modelled curve there, so
  // its line can be found and traced.
  const e = explainsOf(cand()).find((x) => x.pass_id === id && x.curve_t);
  if (!e) return;
  $("#cand-overlay").checked = true;
  spectro.setView(entryWindow(e));
  renderPasses();
}

// ---------------------------------------------------------------------------------------------
// Wiring

$$("#steps button").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.step)));
$("#project-btn").addEventListener("click", () => {
  if (S.page === "projects" && S.p) showPage(localStorage.getItem(`page:${S.name}`) || "passes");
  else { loadProjects(); showPage("projects"); }
});
$("#project-filter").addEventListener("input", renderProjects);
$$("[data-job]").forEach((b) => b.addEventListener("click", () => startJob(b.dataset.job)));
$("#job-stop").addEventListener("click", stopJob);
$("#listen-toggle").addEventListener("click", () => setListening(!S.listening));
$("#save-settings").addEventListener("click", () => saveSettings($("#settings-form"), ["date", "centre", "half_km", "mic_height", "truth", "start", "start_tol_s"]));
$("#settings-form").addEventListener("input", (e) => { if (["centre", "half_km"].includes(e.target.name)) renderAreaMap(); });
$("#save-search").addEventListener("click", () => saveSettings($("#search-form"), ["spacing_m", "audible_km", "max_alt_m", "search_mode"]));
$$(".vmax").forEach((r) => r.addEventListener("input", (e) => {
  S.vmax = Number(e.target.value);
  $$(".vmax").forEach((x) => (x.value = S.vmax));
  for (const v of [spectro, overview, locSpectro, locOverview, ...cells.map((x) => x.view)]) v.request();
}));
$("#fl-time").addEventListener("input", (e) => showFlightsAt(Number(e.target.value)));
$("#fl-play").addEventListener("click", () => playFlights());
for (const id of ["#fl-trail", "#fl-jets"]) $(id).addEventListener("change", () => showFlightsAt(FL.t));
$("#zoom-all").addEventListener("click", () => spectro.setView({ t0: 0, t1: S.p?.duration || 600, f0: 20, f1: 600 }));
$("#cand-overlay").addEventListener("change", () => { spectro.draw(); renderPassDetail(); });
$("#hide-curves").addEventListener("click", () => setHideCurves(!S.hideCurves));
$$("#pass-sort button").forEach((b) => b.addEventListener("click", () => {
  S.passSort = b.dataset.sort;
  $$("#pass-sort button").forEach((x) => x.classList.toggle("sel", x === b));
  renderPasses();
}));
$("#trace-toggle").addEventListener("click", () => (S.tracing ? stopTrace() : startTrace()));
$("#trace-clear").addEventListener("click", () => { S.guide = []; spectro.draw(); });
$("#trace-measure").addEventListener("click", async () => {
  try {
    const rec = await api("POST", projectPath("/passes/measure"), { points: S.guide, replace: retraceId });
    stopTrace();
    await refresh();
    selectPass(rec.id, false);
  } catch (e) { alert(e.message); }
});
$$("#spec-mode button").forEach((b) => b.addEventListener("click", () => {
  S.specMode = b.dataset.mode;
  $$("#spec-mode button").forEach((x) => x.classList.toggle("sel", x === b));
  if (S.specMode === "one") {
    const ps = shownExplains().map((e) => passById(e.pass_id)).filter(Boolean);
    const p = passById(S.locPass);
    if (p) locSpectro.setView(passWindow(p));
    else if (ps.length) locSpectro.setView({ t0: Math.min(...ps.map((x) => x.t_range[0])) - 60, t1: Math.max(...ps.map((x) => x.t_range[1])) + 60,
      f0: 20, f1: Math.max(...ps.map((x) => Math.max(...x.line_f))) * 1.3 });
  }
  renderSpecs(shownExplains());
}));
$("#show-partial").addEventListener("change", (e) => { S.showPartial = e.target.checked; renderLocations(); });
$("#open-in-passes").addEventListener("click", () => S.locPass && openInPasses(S.locPass));

// Esc on the Locations page unselects, the flight first and then the candidate.
document.addEventListener("keydown", (e) => {
  if (S.page === "locations" && e.key === "Escape" && (S.locPass || S.selectedCand != null)) unselectLoc(!S.locPass);
});

// Keys on the Passes page: up/down to move through the list, A/U/R to set the selected pass's status.
document.addEventListener("keydown", (e) => {
  if (S.page !== "passes" || e.target.closest("input, textarea, select, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
  const list = sortedPasses();
  const i = list.findIndex((p) => p.id === S.selectedPass);
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const j = i < 0 ? 0 : Math.min(Math.max(i + (e.key === "ArrowDown" ? 1 : -1), 0), list.length - 1);
    if (list[j]) selectPass(list[j].id, true);
  } else if (passById(S.selectedPass) && { a: 1, u: 1, r: 1 }[e.key.toLowerCase()]) {
    setStatus(S.selectedPass, { a: "accepted", u: "undecided", r: "rejected" }[e.key.toLowerCase()]);
  } else if (e.key === "h" || e.key === "H") {
    setHideCurves(!S.hideCurves);
  } else if (e.key === " " && !e.repeat) {
    e.preventDefault();
    S.peek = true; spectro.draw();
  } else if (e.key === "Escape" && S.tracing) stopTrace();
});
// Keys on the Flights page: Space plays or pauses, left/right step 10 s (60 with Shift).
document.addEventListener("keydown", (e) => {
  if (S.page !== "flights" || !FL.range || e.target.closest("input:not([type=range]), select, dialog")) return;
  if (e.key === " ") { e.preventDefault(); playFlights(); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    showFlightsAt(FL.t + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 60 : 10));
  }
});
document.addEventListener("keydown", (e) => {
  if (S.page !== "locations" || e.key !== "Escape" || document.fullscreenElement || e.target.closest("input, dialog")) return;
  if (S.locPass || S.selectedCand != null) unselectLoc(false);
});
document.addEventListener("keyup", (e) => {
  if (e.key === " " && S.peek) { e.preventDefault(); S.peek = false; spectro.draw(); }  // not a click on a focused button
});
window.addEventListener("blur", () => { if (S.peek) { S.peek = false; spectro.draw(); } });

for (const id of ["new-project", "empty-new"]) $(`#${id}`).addEventListener("click", () => { $("#new-error").textContent = ""; $("#new-dialog").showModal(); });

// Picking a recording: the server lists this computer's folders (a web page can't see file paths).
const browse = { path: "" };
async function browseTo(path) {
  try {
    const d = await api("GET", `/api/browse?path=${encodeURIComponent(path)}`);
    browse.path = d.path;
    browse.parent = d.parent;
    $("#browse-error").textContent = "";
    $("#browse-path").value = d.path;
    const roots = $("#browse-roots");
    roots.hidden = d.roots.length < 2;
    roots.replaceChildren(...d.roots.map((r) => el("option", { value: r }, r)));
    roots.value = d.roots.find((r) => d.path.startsWith(r)) || d.roots[0];
    $("#browse-up").disabled = !d.parent;
    const row = (cls, label, onclick, extra) => {
      const b = el("button", { type: "button", class: cls }, label, extra ? el("span", { class: "hint" }, extra) : "");
      b.addEventListener("click", onclick);
      return el("li", {}, b);
    };
    const sep = d.path.endsWith(d.sep) ? "" : d.sep;
    $("#browse-list").replaceChildren(
      ...d.dirs.map((n) => row("folder", `📁 ${n}`, () => browseTo(d.path + sep + n))),
      ...d.files.map((f) => row("file", f.name, () => pickRecording(f), `${(f.size / 1e6).toFixed(1)} MB`)),
    );
    if (!d.dirs.length && !d.files.length) $("#browse-list").replaceChildren(el("li", { class: "hint" }, "No folders or audio files here."));
    try { localStorage.setItem("browse", d.path); } catch (_) { /* private mode */ }
  } catch (e) { $("#browse-error").textContent = e.message; }
}

// A recording's name often says when it was made: fill in what the form still lacks.
function pickRecording(f) {
  const form = $("#new-form");
  form.elements.audio.value = f.path;
  const stem = f.name.replace(/\.[^.]+$/, "");
  const date = stem.match(/(\d{4})-?(\d{2})-?(\d{2})/);
  if (date && !form.elements.date.value) {
    const iso = `${date[1]}-${date[2]}-${date[3]}`;
    if (!isNaN(Date.parse(iso))) form.elements.date.value = iso;
  }
  if (!form.elements.name.value) form.elements.name.value = stem.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^[-_]+/, "").slice(0, 64);
  $("#browse-dialog").close();
}

$("#browse-open").addEventListener("click", () => {
  let last = "";
  try { last = localStorage.getItem("browse") || ""; } catch (_) { /* private mode */ }
  $("#browse-dialog").showModal();
  browseTo($("#new-form").elements.audio.value.replace(/[^\\/]*$/, "") || last);
});
$("#browse-go").addEventListener("click", () => browseTo($("#browse-path").value));
$("#browse-path").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); browseTo(e.target.value); } });
$("#browse-up").addEventListener("click", () => browse.parent && browseTo(browse.parent));
$("#browse-roots").addEventListener("change", (e) => browseTo(e.target.value));
$("#browse-cancel").addEventListener("click", () => $("#browse-dialog").close());

$("#new-form").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "ok") return;
  e.preventDefault();
  const d = Object.fromEntries(new FormData(e.target));
  const centre = parseLatLon(d.centre);
  const truth = d.truth.trim() ? parseLatLon(d.truth) : [null, null];
  if (!centre) { $("#new-error").textContent = "Centre: two numbers, latitude then longitude."; return; }
  if (!truth) { $("#new-error").textContent = "Known location: two numbers, or leave it empty."; return; }
  try {
    await api("POST", "/api/projects", {
      name: d.name, audio: d.audio, date: d.date, lat: centre[0], lon: centre[1], half_km: Number(d.half_km),
      mic_height: Number(d.mic_height), truth_lat: truth[0], truth_lon: truth[1],
    });
    $("#new-dialog").close();
    await loadProjects();
    await openProject(d.name, "recording");
  } catch (err) { $("#new-error").textContent = err.message; }
});

(async () => {
  await loadProjects();
  let last = null;
  try { last = localStorage.getItem("project"); } catch (_) { /* private mode */ }
  if (last && S.projects.some((p) => p.name === last)) await openProject(last);
  else showPage(S.projects.length ? "projects" : "empty");
})().catch((e) => alert(e.message));
