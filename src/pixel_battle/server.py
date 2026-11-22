"""Lightweight FastAPI live web dashboard for watching and steering the 512x512 pixel battle."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

if TYPE_CHECKING:
    from pixel_battle.engine import SimulationEngine


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Pixel Battle — Local AI r/place</title>
<style>
  :root {
    --bg: #0c0f16;
    --panel: #141925;
    --border: #252e42;
    --text: #e6edf7;
    --muted: #8a97b0;
    --accent: #38bdf8;
    --green: #34d399;
    --red: #fb7185;
    --amber: #fbbf24;
    --purple: #c084fc;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    height: 100vh;
    display: grid;
    grid-template-rows: 54px 1fr;
    overflow: hidden;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }
  .pill {
    font-size: 12px;
    padding: 3px 9px;
    border-radius: 999px;
    background: #1d2638;
    color: var(--accent);
    border: 1px solid #2b3a55;
    font-variant-numeric: tabular-nums;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  button, select {
    background: #1d2638;
    color: var(--text);
    border: 1px solid #303f5e;
    border-radius: 6px;
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.12s;
  }
  button:hover { background: #27344d; }
  button.primary { background: #0284c7; border-color: #38bdf8; }
  button.primary:hover { background: #0369a1; }
  button.danger { background: #7f1d1d; border-color: #f43f5e; }
  button.danger:hover { background: #991b1b; }
  main {
    display: grid;
    grid-template-columns: 1fr 390px;
    min-height: 0;
  }
  .stage {
    position: relative;
    background: #07090e;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
  }
  #viewport {
    position: relative;
    transform-origin: center center;
    cursor: grab;
  }
  #viewport:active { cursor: grabbing; }
  #canvasImg, #heatImg, #overlayCanvas {
    display: block;
    width: 512px;
    height: 512px;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
  }
  #heatImg, #overlayCanvas {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  }
  .hud-bar {
    position: absolute;
    left: 14px;
    bottom: 14px;
    display: flex;
    gap: 8px;
    background: rgba(15, 20, 30, 0.88);
    border: 1px solid var(--border);
    padding: 7px 12px;
    border-radius: 8px;
    font-size: 12px;
    backdrop-filter: blur(6px);
    align-items: center;
  }
  .hud-top {
    position: absolute;
    left: 14px;
    top: 14px;
    background: rgba(15, 20, 30, 0.88);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    color: var(--accent);
    max-width: 72%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  label.toggle {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
    color: var(--text);
  }
  aside {
    background: var(--panel);
    border-left: 1px solid var(--border);
    display: grid;
    grid-template-rows: 46% 54%;
    min-height: 0;
  }
  .pane {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-bottom: 1px solid var(--border);
  }
  .pane-title {
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: var(--muted);
    background: #101520;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
  }
  .scroll {
    overflow-y: auto;
    padding: 8px 12px;
    flex: 1;
  }
  .fac-card {
    background: #192030;
    border: 1px solid #263147;
    border-radius: 7px;
    padding: 8px 10px;
    margin-bottom: 7px;
    font-size: 12px;
  }
  .fac-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-bottom: 4px;
  }
  .fac-name {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
  }
  .swatch {
    width: 11px;
    height: 11px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.35);
    display: inline-block;
  }
  .bar-bg {
    height: 5px;
    background: #0d121d;
    border-radius: 4px;
    overflow: hidden;
    margin: 5px 0;
  }
  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #34d399);
  }
  .meta { color: var(--muted); font-size: 11px; }
  .chron-item {
    font-size: 12px;
    line-height: 1.4;
    padding: 6px 0;
    border-bottom: 1px solid #1e2638;
  }
  .badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    margin-right: 6px;
    text-transform: uppercase;
  }
  .b-founded { background: #1e3a5f; color: #7dd3fc; }
  .b-alliance { background: #064e3b; color: #6ee7b7; }
  .b-diplomacy { background: #1e293b; color: #93c5fd; }
  .b-war, .b-betrayal { background: #7f1d1d; color: #fda4af; }
  .b-void { background: #3b0764; color: #e879f9; }
  .b-morph, .b-expand { background: #4c1d95; color: #d8b4fe; }
  .b-fortify, .b-exodus { background: #78350f; color: #fde68a; }
  .b-system, .b-collapse { background: #1f2937; color: #9ca3af; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>PIXEL BATTLE</span>
    <span class="pill" id="statTick">Tick 0</span>
    <span class="pill" id="statPix">0 px</span>
    <span class="pill" id="statRate">0 px/s</span>
  </div>
  <div class="controls">
    <button id="btnPause" class="primary" onclick="togglePause()">Pause</button>
    <select id="speedSel" onchange="setSpeed(this.value)">
      <option value="6">Slow (6 tps)</option>
      <option value="18" selected>Live (18 tps)</option>
      <option value="45">Fast (45 tps)</option>
      <option value="0">Uncapped</option>
    </select>
    <button onclick="postAction('/api/discover')">Discover Faction</button>
    <button onclick="postAction('/api/epoch')">Run Epoch</button>
    <button class="danger" onclick="postAction('/api/void')">Summon Void</button>
    <button onclick="downloadPng()">Snapshot PNG</button>
  </div>
</header>

<main>
  <section class="stage" id="stage">
    <div class="hud-top" id="brainStatus">Booting simulation...</div>
    <div id="viewport">
      <img id="canvasImg" src="/api/canvas.png" alt="Canvas" draggable="false" />
      <img id="heatImg" src="/api/heatmap.png" alt="Heatmap" style="display:none" draggable="false" />
      <canvas id="overlayCanvas" width="512" height="512"></canvas>
    </div>
    <div class="hud-bar">
      <label class="toggle"><input type="checkbox" id="chkBoxes" onchange="drawOverlay()"> Factions</label>
      <label class="toggle"><input type="checkbox" id="chkLinks" onchange="drawOverlay()"> Alliances/Wars</label>
      <label class="toggle"><input type="checkbox" id="chkHeat" onchange="toggleHeat()"> Heatmap</label>
      <span class="meta" id="coordReadout">Scroll to zoom · Drag to pan · Dbl-click reset</span>
    </div>
  </section>

  <aside>
    <div class="pane">
      <div class="pane-title">
        <span>Active Factions</span>
        <span id="facSummary">0 factions</span>
      </div>
      <div class="scroll" id="factionList"></div>
    </div>
    <div class="pane">
      <div class="pane-title">
        <span>Chronicle &amp; Diplomacy</span>
        <span id="epochReadout">Epoch 0</span>
      </div>
      <div class="scroll" id="chronicleList"></div>
    </div>
  </aside>
</main>

<script>
let state = null;
let scale = 1.25;
let panX = 0, panY = 0;
let dragging = false, dragSX = 0, dragSY = 0;

const viewport = document.getElementById("viewport");
const stage = document.getElementById("stage");
const canvasImg = document.getElementById("canvasImg");
const heatImg = document.getElementById("heatImg");
const overlay = document.getElementById("overlayCanvas");
const ctx = overlay.getContext("2d");

function applyTransform() {
  viewport.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}
applyTransform();

stage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 0.87;
  scale = Math.max(0.5, Math.min(12, scale * factor));
  applyTransform();
}, { passive: false });

stage.addEventListener("mousedown", (e) => {
  dragging = true;
  dragSX = e.clientX - panX;
  dragSY = e.clientY - panY;
});
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (dragging) {
    panX = e.clientX - dragSX;
    panY = e.clientY - dragSY;
    applyTransform();
  }
  const rect = canvasImg.getBoundingClientRect();
  if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
    const w = state ? state.width : 512;
    const h = state ? state.height : 512;
    const px = Math.floor(((e.clientX - rect.left) / rect.width) * w);
    const py = Math.floor(((e.clientY - rect.top) / rect.height) * h);
    document.getElementById("coordReadout").textContent = `(${px}, ${py}) · Zoom ${scale.toFixed(1)}x`;
  }
});
stage.addEventListener("dblclick", () => {
  scale = 1.25; panX = 0; panY = 0; applyTransform();
});

async function postAction(url) {
  await fetch(url, { method: "POST" });
  pollState();
}
async function togglePause() {
  await postAction("/api/toggle_pause");
}
async function setSpeed(val) {
  await fetch(`/api/speed?tps=${encodeURIComponent(val)}`, { method: "POST" });
}
function toggleHeat() {
  heatImg.style.display = document.getElementById("chkHeat").checked ? "block" : "none";
}
function downloadPng() {
  const a = document.createElement("a");
  a.href = `/api/canvas.png?t=${Date.now()}`;
  a.download = `pixel-battle-tick-${state ? state.tick : 0}.png`;
  a.click();
}

function drawOverlay() {
  if (!state) return;
  if (overlay.width !== state.width || overlay.height !== state.height) {
    overlay.width = state.width;
    overlay.height = state.height;
    canvasImg.style.width = state.width + "px";
    canvasImg.style.height = state.height + "px";
    heatImg.style.width = state.width + "px";
    heatImg.style.height = state.height + "px";
    overlay.style.width = state.width + "px";
    overlay.style.height = state.height + "px";
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const showBoxes = document.getElementById("chkBoxes").checked;
  const showLinks = document.getElementById("chkLinks").checked;
  const byId = {};
  for (const f of state.factions) byId[f.id] = f;

  if (showLinks) {
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "rgba(52, 211, 153, 0.9)";
    for (const [a, b] of state.alliances) {
      const fa = byId[a], fb = byId[b];
      if (!fa || !fb) continue;
      ctx.beginPath(); ctx.moveTo(fa.cx, fa.cy); ctx.lineTo(fb.cx, fb.cy); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(251, 113, 133, 0.9)";
    for (const [a, b] of state.wars) {
      const fa = byId[a], fb = byId[b];
      if (!fa || !fb) continue;
      ctx.beginPath(); ctx.moveTo(fa.cx, fa.cy); ctx.lineTo(fb.cx, fb.cy); ctx.stroke();
    }
  }

  if (showBoxes) {
    ctx.font = "bold 9px monospace";
    for (const f of state.factions) {
      const [x0, y0, x1, y1] = f.bbox;
      ctx.strokeStyle = f.is_void ? "#f43f5e" : "rgba(255,255,255,0.75)";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
      const label = f.name;
      const tw = ctx.measureText(label).width + 6;
      const ly = Math.max(10, y0 - 2);
      ctx.fillStyle = f.is_void ? "rgba(120,0,40,0.88)" : "rgba(12,16,26,0.82)";
      ctx.fillRect(x0, ly - 9, tw, 11);
      ctx.fillStyle = f.is_void ? "#fda4af" : "#f8fafc";
      ctx.fillText(label, x0 + 3, ly - 1);
    }
  }
}

function renderPanels() {
  if (!state) return;
  document.getElementById("statTick").textContent = `Tick ${state.tick.toLocaleString()} / ${state.max_ticks.toLocaleString()}`;
  document.getElementById("statPix").textContent = `${state.total_placements.toLocaleString()} px`;
  document.getElementById("statRate").textContent = `${state.pixels_per_sec.toLocaleString()} px/s (${state.actual_tps} tps)`;
  document.getElementById("brainStatus").textContent = `${state.device.toUpperCase()} · ${state.policy_name || "policy"} · ${state.brain_status}`;
  document.getElementById("epochReadout").textContent = `Epoch ${state.epoch}`;
  document.getElementById("btnPause").textContent = state.paused ? "Resume" : "Pause";
  document.getElementById("facSummary").textContent = `${state.factions.length} factions · ${state.free_agents.toLocaleString()} free`;

  const fList = document.getElementById("factionList");
  fList.innerHTML = state.factions.map(f => `
    <div class="fac-card" title="${escapeHtml(f.art_prompt)}">
      <div class="fac-head">
        <div class="fac-name">
          <span class="swatch" style="background:${f.primary_hex}"></span>
          <span class="swatch" style="background:${f.secondary_hex}"></span>
          <span>${escapeHtml(f.name)}</span>
        </div>
        <span class="meta">S${f.stage} · ${f.last_action}</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width:${Math.round(f.integrity * 100)}%"></div></div>
      <div class="meta">
        ${f.members.toLocaleString()} users · ${Math.round(f.integrity * 100)}% art · ${f.art_mode}
        ${f.allies.length ? ` · ${f.allies.length} ally` : ""}
        ${f.enemies.length ? ` · ${f.enemies.length} war` : ""}
      </div>
      <div class="meta" style="margin-top:2px;color:#cbd5e1">${escapeHtml(f.art_prompt)}</div>
    </div>
  `).join("") || `<div class="meta">Primordial chaos — waiting for clusters to form...</div>`;

  const cList = document.getElementById("chronicleList");
  cList.innerHTML = state.chronicle.map(c => `
    <div class="chron-item">
      <span class="badge b-${c.kind}">${c.kind}</span>
      <span class="meta">t=${c.tick}</span>
      <div>${escapeHtml(c.summary)}</div>
    </div>
  `).join("");
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

let imgBusy = false;
function refreshImages() {
  if (imgBusy) return;
  imgBusy = true;
  const next = new Image();
  next.onload = () => {
    canvasImg.src = next.src;
    imgBusy = false;
  };
  next.onerror = () => { imgBusy = false; };
  next.src = `/api/canvas.png?t=${Date.now()}`;
  if (document.getElementById("chkHeat").checked) {
    heatImg.src = `/api/heatmap.png?t=${Date.now()}`;
  }
}

async function pollState() {
  try {
    const r = await fetch("/api/state");
    state = await r.json();
    renderPanels();
    drawOverlay();
  } catch (e) {}
}

setInterval(refreshImages, 250);
setInterval(pollState, 600);
pollState();
</script>
</body>
</html>
"""


def create_app(engine: SimulationEngine) -> FastAPI:
    app = FastAPI(title="Pixel Battle Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        return JSONResponse(engine.get_ui_state())

    @app.get("/api/canvas.png")
    def get_canvas_png() -> Response:
        with engine._state_lock:
            rgb = engine.canvas.to_rgb_numpy()
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=1)
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/heatmap.png")
    def get_heatmap_png() -> Response:
        with engine._state_lock:
            rgba = engine.canvas.render_heatmap_rgba()
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=1)
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/toggle_pause")
    def toggle_pause() -> JSONResponse:
        engine.paused = not engine.paused
        return JSONResponse({"paused": engine.paused})

    @app.post("/api/speed")
    def set_speed(tps: float = Query(18.0, ge=0.0, le=500.0)) -> JSONResponse:
        engine.target_tps = tps
        return JSONResponse({"target_tps": engine.target_tps})

    @app.post("/api/discover")
    def trigger_discover() -> JSONResponse:
        engine.queue_job("DISCOVER")
        return JSONResponse({"queued": "DISCOVER"})

    @app.post("/api/epoch")
    def trigger_epoch() -> JSONResponse:
        engine.queue_job("EPOCH")
        return JSONResponse({"queued": "EPOCH"})

    @app.post("/api/void")
    def trigger_void() -> JSONResponse:
        engine.queue_job("VOID")
        return JSONResponse({"queued": "VOID"})

    return app
