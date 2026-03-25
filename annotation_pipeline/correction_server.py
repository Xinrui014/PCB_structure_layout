#!/usr/bin/env python3
"""
PCB Annotation Correction Server — Human-in-the-loop review UI.

Combines features from: annotator.py, orientation_labeler.py, review_server.py,
color_analysis.py, resolution_analysis.py.

Usage:
    python correction_server.py --anno_dir pipeline_output/ --image_dir /path/to/images --port 8899

Features:
    - Browse boards (search, filter by color/resolution/reviewed/class)
    - View detections overlaid on image
    - Add/delete/resize bounding boxes
    - Change component class (dropdown or keyboard 1-9/0)
    - Change orientation (8-direction wheel)
    - Override board color and resolution
    - Exclude boards
    - Track reviewed/unreviewed status
    - Save corrections (preserves originals in backup/)
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from collections import Counter
from flask import Flask, jsonify, request, send_file, Response

app = Flask(__name__)

# ── Globals (set by main) ──
ANNO_DIR = None
IMAGE_DIR = None
BACKUP_DIR = None

CATEGORIES = [
    {"id": 1, "name": "Resistor"},
    {"id": 2, "name": "Capacitor"},
    {"id": 3, "name": "Inductor"},
    {"id": 4, "name": "Connector"},
    {"id": 5, "name": "Diode"},
    {"id": 7, "name": "Switch"},
    {"id": 8, "name": "Transistor"},
    {"id": 9, "name": "Integrated Circuit"},
    {"id": 10, "name": "Oscillator"},
]

CAT_COLORS = {
    1: "#FF6B6B",   # Resistor - red
    2: "#4ECDC4",   # Capacitor - teal
    3: "#FFE66D",   # Inductor - yellow
    4: "#A8E6CF",   # Connector - green
    5: "#FF8B94",   # Diode - pink
    7: "#B8A9C9",   # Switch - purple
    8: "#F7DC6F",   # Transistor - gold
    9: "#85C1E9",   # IC - blue
    10: "#F0B27A",  # Oscillator - orange
}

# Runtime state
reviewed_boards = set()
excluded_boards = set()


def load_state():
    global reviewed_boards, excluded_boards
    state_file = ANNO_DIR / ".review_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        reviewed_boards = set(state.get("reviewed", []))
        excluded_boards = set(state.get("excluded", []))


def save_state():
    state_file = ANNO_DIR / ".review_state.json"
    state = {
        "reviewed": sorted(reviewed_boards),
        "excluded": sorted(excluded_boards),
    }
    state_file.write_text(json.dumps(state, indent=2))


def find_image(board):
    for ext in [".png", ".jpg", ".JPG", ".jpeg", ".JPEG", ".PNG"]:
        p = IMAGE_DIR / f"{board}{ext}"
        if p.exists():
            return p
    return None


# ── API Routes ──

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/categories")
def api_categories():
    return jsonify(CATEGORIES)


@app.route("/api/boards")
def api_boards():
    jsons = sorted(ANNO_DIR.glob("*.json"))
    boards = []
    for jp in jsons:
        if jp.name.startswith("."):
            continue
        board = jp.stem
        data = json.loads(jp.read_text())
        anns = data.get("annotations", [])
        cat_counts = Counter(a.get("category_id") for a in anns)

        boards.append({
            "name": board,
            "num_annotations": len(anns),
            "color": data.get("board_color", "unknown"),
            "resolution": data.get("resolution_class"),
            "reviewed": board in reviewed_boards,
            "excluded": board in excluded_boards,
            "categories": dict(cat_counts),
        })
    return jsonify(boards)


@app.route("/api/board/<board_name>")
def api_board(board_name):
    json_path = ANNO_DIR / f"{board_name}.json"
    if not json_path.exists():
        return jsonify({"error": "Not found"}), 404
    data = json.loads(json_path.read_text())
    data["_reviewed"] = board_name in reviewed_boards
    data["_excluded"] = board_name in excluded_boards
    return jsonify(data)


@app.route("/api/image/<board_name>")
def api_image(board_name):
    img_path = find_image(board_name)
    if img_path is None:
        return "Image not found", 404
    return send_file(str(img_path))


@app.route("/api/save/<board_name>", methods=["POST"])
def api_save(board_name):
    json_path = ANNO_DIR / f"{board_name}.json"
    if not json_path.exists():
        return jsonify({"error": "Not found"}), 404

    # Backup original
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"{board_name}.json"
    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)

    # Load current data
    data = json.loads(json_path.read_text())

    # Update from request
    body = request.json
    if "annotations" in body:
        data["annotations"] = body["annotations"]
    if "board_color" in body:
        data["board_color"] = body["board_color"]
    if "resolution_class" in body:
        data["resolution_class"] = body["resolution_class"]

    # Re-compute IDs
    for i, ann in enumerate(data.get("annotations", [])):
        ann["id"] = i + 1
        if "bbox" in ann:
            b = ann["bbox"]
            ann["area"] = round(b[2] * b[3], 2)

    json_path.write_text(json.dumps(data, indent=2))

    # Mark as reviewed
    reviewed_boards.add(board_name)
    save_state()

    return jsonify({"ok": True, "count": len(data.get("annotations", []))})


@app.route("/api/mark_reviewed/<board_name>", methods=["POST"])
def api_mark_reviewed(board_name):
    reviewed_boards.add(board_name)
    save_state()
    return jsonify({"ok": True})


@app.route("/api/toggle_exclude/<board_name>", methods=["POST"])
def api_toggle_exclude(board_name):
    if board_name in excluded_boards:
        excluded_boards.discard(board_name)
        action = "included"
    else:
        excluded_boards.add(board_name)
        action = "excluded"
    save_state()
    return jsonify({"ok": True, "action": action})


@app.route("/api/stats")
def api_stats():
    jsons = sorted(ANNO_DIR.glob("*.json"))
    total = len([j for j in jsons if not j.name.startswith(".")])
    return jsonify({
        "total": total,
        "reviewed": len(reviewed_boards),
        "excluded": len(excluded_boards),
        "remaining": total - len(reviewed_boards) - len(excluded_boards),
    })


# ── HTML ──

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB Annotation Correction</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; height: 100vh; overflow: hidden; background: #1a1a2e; color: #eee; }

/* Sidebar */
#sidebar { width: 300px; background: #16213e; display: flex; flex-direction: column; border-right: 1px solid #0f3460; }
#sidebar-header { padding: 10px; border-bottom: 1px solid #0f3460; }
#sidebar-header h2 { font-size: 14px; margin-bottom: 8px; color: #e94560; }
#search { width: 100%; padding: 6px 10px; background: #0f3460; border: 1px solid #1a1a4e; color: #eee; border-radius: 4px; font-size: 13px; }
#filters { display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; }
#filters select { padding: 3px 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #eee; border-radius: 3px; font-size: 11px; }
#board-list { flex: 1; overflow-y: auto; }
.board-item { padding: 8px 10px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 12px; display: flex; justify-content: space-between; align-items: center; }
.board-item:hover { background: #0f3460; }
.board-item.active { background: #e94560; color: white; }
.board-item.reviewed { border-left: 3px solid #2ecc71; }
.board-item.excluded { opacity: 0.4; text-decoration: line-through; }
.board-item .count { color: #888; font-size: 11px; }
#stats { padding: 8px 10px; border-top: 1px solid #0f3460; font-size: 11px; color: #888; }

/* Main */
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* Toolbar */
#toolbar { padding: 8px 12px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
#toolbar .board-name { font-weight: bold; font-size: 14px; min-width: 200px; }
#toolbar .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
.badge-green { background: #27ae60; }
.badge-red { background: #e74c3c; }
.badge-blue { background: #2980b9; }
.badge-black { background: #2c3e50; }
.badge-white { background: #bdc3c7; color: #333; }
.badge-res { background: #8e44ad; }
#toolbar select { padding: 3px 8px; background: #0f3460; border: 1px solid #1a1a4e; color: #eee; border-radius: 3px; font-size: 12px; }
#toolbar button { padding: 4px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }
.btn-save { background: #27ae60; color: white; }
.btn-save:hover { background: #2ecc71; }
.btn-exclude { background: #e74c3c; color: white; }
.btn-exclude:hover { background: #c0392b; }
.btn-nav { background: #0f3460; color: #eee; }
.btn-nav:hover { background: #1a1a4e; }
.btn-add { background: #2980b9; color: white; }
.btn-add:hover { background: #3498db; }
.btn-reviewed { background: #f39c12; color: white; }

/* Canvas area */
#canvas-wrap { flex: 1 1 0; position: relative; overflow: hidden; background: #111; min-height: 0; }
#canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }

/* Component panel */
#comp-panel { height: 60px; min-height: 40px; max-height: 200px; background: #16213e; border-top: 1px solid #0f3460; display: flex; flex-direction: column; }
#comp-panel-header { padding: 4px 10px; font-size: 11px; color: #888; border-bottom: 1px solid #0f3460; display: flex; justify-content: space-between; }
#comp-list { flex: 1; overflow-x: auto; display: flex; gap: 4px; padding: 6px; align-items: flex-start; }
.comp-card { min-width: 60px; background: #0f3460; border-radius: 4px; padding: 4px; text-align: center; cursor: pointer; font-size: 10px; border: 2px solid transparent; }
.comp-card:hover { border-color: #e94560; }
.comp-card.selected { border-color: #e94560; background: #1a1a4e; }
.comp-card img { width: 40px; height: 40px; object-fit: contain; background: #111; border-radius: 2px; }
.comp-card .label { margin-top: 2px; }
.comp-card .ori { color: #888; font-size: 9px; }

/* Orientation wheel */
#ori-wheel { display: none; position: absolute; width: 120px; height: 120px; background: rgba(0,0,0,0.85); border-radius: 50%; border: 2px solid #e94560; z-index: 100; }
.ori-btn { position: absolute; width: 28px; height: 28px; border-radius: 50%; background: #0f3460; border: 1px solid #e94560; color: white; font-size: 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.ori-btn:hover { background: #e94560; }

/* Toast */
#toast { position: fixed; bottom: 20px; right: 20px; padding: 10px 20px; background: #27ae60; color: white; border-radius: 6px; font-size: 13px; display: none; z-index: 200; }

/* Class palette */
#class-palette { display: none; position: absolute; background: #16213e; border: 1px solid #0f3460; border-radius: 6px; padding: 4px; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
.class-btn { display: block; width: 100%; padding: 4px 10px; text-align: left; background: none; border: none; color: #eee; cursor: pointer; font-size: 12px; border-radius: 3px; }
.class-btn:hover { background: #0f3460; }
</style>
</head>
<body>

<!-- Sidebar -->
<div id="sidebar">
  <div id="sidebar-header">
    <h2>🔧 PCB Annotation Correction</h2>
    <input type="text" id="search" placeholder="Search boards...">
    <div id="filters">
      <select id="filter-color"><option value="">All Colors</option></select>
      <select id="filter-res"><option value="">All Res</option></select>
      <select id="filter-status">
        <option value="">All</option>
        <option value="unreviewed">Unreviewed</option>
        <option value="reviewed">✅ Reviewed</option>
        <option value="excluded">🚫 Excluded</option>
      </select>
    </div>
  </div>
  <div id="board-list"></div>
  <div id="stats"></div>
</div>

<!-- Main -->
<div id="main">
  <div id="toolbar">
    <span class="board-name" id="board-name">Select a board</span>
    <span class="badge" id="color-badge"></span>
    <select id="color-override"></select>
    <span class="badge badge-res" id="res-badge"></span>
    <select id="res-override"></select>
    <span id="ann-count" style="color:#888;font-size:12px;"></span>
    <div style="flex:1;"></div>
    <button class="btn-add" id="btn-add-mode" title="Draw new bbox (D)">+ Add</button>
    <button class="btn-exclude" id="btn-exclude">🚫 Exclude</button>
    <button class="btn-reviewed" id="btn-mark-reviewed">✅ Reviewed</button>
    <button class="btn-save" id="btn-save" title="Ctrl+S">💾 Save</button>
    <button class="btn-nav" id="btn-prev" title="←">◀ Prev</button>
    <button class="btn-nav" id="btn-next" title="→">Next ▶</button>
  </div>

  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
  </div>

  <div id="comp-panel">
    <div id="comp-panel-header" style="cursor:pointer;" onclick="toggleCompPanel()">
      <span id="comp-panel-title">Components</span>
      <span>Click: select | 1-9/0: change class | O: orientation | F/Del: delete</span>
    </div>
    <div id="comp-list"></div>
  </div>
</div>

<!-- Overlays -->
<div id="ori-wheel"></div>
<div id="class-palette"></div>
<div id="toast"></div>

<script>
const CATEGORIES = [
  {id:1,name:"Resistor",key:"1"},{id:2,name:"Capacitor",key:"2"},{id:3,name:"Inductor",key:"3"},
  {id:4,name:"Connector",key:"4"},{id:5,name:"Diode",key:"5"},{id:7,name:"Switch",key:"7"},
  {id:8,name:"Transistor",key:"8"},{id:9,name:"IC",key:"9"},{id:10,name:"Oscillator",key:"0"}
];
const CAT_COLORS = {1:"#FF6B6B",2:"#4ECDC4",3:"#FFE66D",4:"#A8E6CF",5:"#FF8B94",7:"#B8A9C9",8:"#F7DC6F",9:"#85C1E9",10:"#F0B27A"};
const CAT_MAP = {}; CATEGORIES.forEach(c => CAT_MAP[c.id] = c);
const ANGLES = [0,45,90,135,180,225,270,315];

let boards = [];
let currentBoard = null;
let boardData = null;
let annotations = [];
let selectedIdx = -1;
let boardImg = null;
let addMode = false;
let drawStart = null;
let dirty = false;

// Canvas state
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let scale = 1;
let offsetX = 0, offsetY = 0;

// ── Init ──
async function init() {
  const res = await fetch("/api/boards");
  boards = await res.json();
  populateFilters();
  renderBoardList();
  updateStats();

  // Color/res override dropdowns
  const colorSel = document.getElementById("color-override");
  ["green","red","blue","black","white"].forEach(c => {
    const o = document.createElement("option"); o.value = c; o.textContent = c; colorSel.appendChild(o);
  });
  const resSel = document.getElementById("res-override");
  ["R1","R2","R3","R4","R5","R6","R7"].forEach(r => {
    const o = document.createElement("option"); o.value = r; o.textContent = r; resSel.appendChild(o);
  });

  // Auto-load first unreviewed
  const first = boards.find(b => !b.reviewed && !b.excluded) || boards[0];
  if (first) loadBoard(first.name);
}

function populateFilters() {
  const colors = new Set(boards.map(b => b.color));
  const sel = document.getElementById("filter-color");
  colors.forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = c; sel.appendChild(o); });
  const ress = new Set(boards.map(b => b.resolution).filter(Boolean));
  const rsel = document.getElementById("filter-res");
  [...ress].sort().forEach(r => { const o = document.createElement("option"); o.value = r; o.textContent = r; rsel.appendChild(o); });
}

function renderBoardList() {
  const search = document.getElementById("search").value.toLowerCase();
  const colorF = document.getElementById("filter-color").value;
  const resF = document.getElementById("filter-res").value;
  const statusF = document.getElementById("filter-status").value;

  const list = document.getElementById("board-list");
  list.innerHTML = "";
  boards.filter(b => {
    if (search && !b.name.toLowerCase().includes(search)) return false;
    if (colorF && b.color !== colorF) return false;
    if (resF && b.resolution !== resF) return false;
    if (statusF === "reviewed" && !b.reviewed) return false;
    if (statusF === "unreviewed" && (b.reviewed || b.excluded)) return false;
    if (statusF === "excluded" && !b.excluded) return false;
    return true;
  }).forEach(b => {
    const div = document.createElement("div");
    div.className = "board-item" + (b.reviewed ? " reviewed" : "") + (b.excluded ? " excluded" : "") + (currentBoard === b.name ? " active" : "");
    div.innerHTML = `<span>${b.name}</span><span class="count">${b.num_annotations}</span>`;
    div.onclick = () => loadBoard(b.name);
    list.appendChild(div);
  });
}

async function updateStats() {
  const res = await fetch("/api/stats");
  const s = await res.json();
  document.getElementById("stats").textContent = `Total: ${s.total} | ✅ ${s.reviewed} | 🚫 ${s.excluded} | Remaining: ${s.remaining}`;
}

// ── Load Board ──
async function loadBoard(name) {
  if (dirty && !confirm("Unsaved changes. Discard?")) return;
  currentBoard = name;
  const res = await fetch(`/api/board/${name}`);
  boardData = await res.json();
  annotations = boardData.annotations || [];
  selectedIdx = -1;
  dirty = false;

  // Update toolbar
  document.getElementById("board-name").textContent = name;
  const color = boardData.board_color || "unknown";
  const badge = document.getElementById("color-badge");
  badge.textContent = color;
  badge.className = "badge badge-" + color;
  document.getElementById("color-override").value = color;

  const resClass = boardData.resolution_class || "";
  document.getElementById("res-badge").textContent = resClass || "N/A";
  document.getElementById("res-override").value = resClass || "";

  document.getElementById("ann-count").textContent = `${annotations.length} annotations`;

  // Load image
  boardImg = new Image();
  boardImg.onload = () => {
    // Immediate fit
    fitCanvas(); render();
    // Re-fit after layout settles (component panel may shift things)
    setTimeout(() => { fitCanvas(); render(); }, 100);
    setTimeout(() => { fitCanvas(); render(); }, 300);
  };
  boardImg.src = `/api/image/${name}`;

  renderBoardList();
  renderCompPanel();
}

function fitCanvas() {
  const wrap = document.getElementById("canvas-wrap");
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  const imgW = boardImg.naturalWidth || boardImg.width;
  const imgH = boardImg.naturalHeight || boardImg.height;
  if (!ww || !wh || !imgW || !imgH) {
    requestAnimationFrame(() => { fitCanvas(); render(); });
    return;
  }
  // Fit image to canvas while maintaining aspect ratio, centered
  const scaleX = ww / imgW;
  const scaleY = wh / imgH;
  scale = Math.min(scaleX, scaleY);
  const renderW = Math.round(imgW * scale);
  const renderH = Math.round(imgH * scale);
  canvas.width = ww;
  canvas.height = wh;
  offsetX = Math.round((ww - renderW) / 2);
  offsetY = Math.round((wh - renderH) / 2);
}

// ── Render ──
function render() {
  if (!boardImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw image
  const imgW = boardImg.naturalWidth || boardImg.width;
  const imgH = boardImg.naturalHeight || boardImg.height;
  ctx.drawImage(boardImg, offsetX, offsetY, imgW * scale, imgH * scale);

  // Draw annotations
  annotations.forEach((ann, i) => {
    const [x, y, w, h] = ann.bbox;
    const sx = x * scale + offsetX;
    const sy = y * scale + offsetY;
    const sw = w * scale;
    const sh = h * scale;
    const color = CAT_COLORS[ann.category_id] || "#fff";
    const isSelected = i === selectedIdx;

    ctx.strokeStyle = isSelected ? "#fff" : color;
    ctx.lineWidth = isSelected ? 2.5 : 1.5;
    ctx.strokeRect(sx, sy, sw, sh);

    // Label
    const cat = CAT_MAP[ann.category_id];
    const label = cat ? cat.name.slice(0, 3) : "?";
    const ori = ann.orientation != null ? ` ${ann.orientation}°` : "";
    ctx.fillStyle = isSelected ? "#fff" : color;
    ctx.font = "bold 10px sans-serif";
    ctx.fillText(label + ori, sx + 2, sy - 3);

    // Fill with transparency
    ctx.fillStyle = color + (isSelected ? "40" : "15");
    ctx.fillRect(sx, sy, sw, sh);
  });

  // Draw in-progress box
  if (drawStart) {
    ctx.strokeStyle = "#e94560";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    const dx = drawStart.cx - drawStart.sx;
    const dy = drawStart.cy - drawStart.sy;
    ctx.strokeRect(drawStart.sx, drawStart.sy, dx, dy);
    ctx.setLineDash([]);
  }
}

function renderCompPanel() {
  const list = document.getElementById("comp-list");
  list.innerHTML = "";
  document.getElementById("comp-panel-title").textContent = `Components (${annotations.length})`;

  annotations.forEach((ann, i) => {
    const card = document.createElement("div");
    card.className = "comp-card" + (i === selectedIdx ? " selected" : "");
    const cat = CAT_MAP[ann.category_id];
    const color = CAT_COLORS[ann.category_id] || "#666";
    const ori = ann.orientation != null ? `${ann.orientation}°` : "";
    card.innerHTML = `<div class="label" style="color:${color}">${cat ? cat.name : "?"}</div><div class="ori">${ori} | ${Math.round(ann.bbox[2])}×${Math.round(ann.bbox[3])}</div>`;
    card.style.borderColor = i === selectedIdx ? "#e94560" : "transparent";
    card.onclick = () => { selectedIdx = i; render(); renderCompPanel(); };
    list.appendChild(card);
  });
}

// ── Canvas interaction ──
function canvasToImg(cx, cy) {
  return [(cx - offsetX) / scale, (cy - offsetY) / scale];
}

canvas.addEventListener("mousedown", e => {
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const [ix, iy] = canvasToImg(cx, cy);

  if (addMode) {
    drawStart = {sx: cx, sy: cy, cx: cx, cy: cy, ix: ix, iy: iy};
    return;
  }

  // Find clicked annotation (top-most)
  let found = -1;
  for (let i = annotations.length - 1; i >= 0; i--) {
    const [x, y, w, h] = annotations[i].bbox;
    if (ix >= x && ix <= x + w && iy >= y && iy <= y + h) { found = i; break; }
  }
  selectedIdx = found;
  render();
  renderCompPanel();
});

canvas.addEventListener("mousemove", e => {
  if (!drawStart) return;
  const rect = canvas.getBoundingClientRect();
  drawStart.cx = e.clientX - rect.left;
  drawStart.cy = e.clientY - rect.top;
  render();
});

canvas.addEventListener("mouseup", e => {
  if (!drawStart) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const [ix2, iy2] = canvasToImg(cx, cy);

  const x = Math.min(drawStart.ix, ix2);
  const y = Math.min(drawStart.iy, iy2);
  const w = Math.abs(ix2 - drawStart.ix);
  const h = Math.abs(iy2 - drawStart.iy);

  drawStart = null;

  if (w < 5 || h < 5) { render(); return; }

  // Add new annotation (default: Resistor)
  annotations.push({
    id: annotations.length + 1,
    image_id: 0,
    category_id: 1,
    bbox: [round2(x), round2(y), round2(w), round2(h)],
    area: round2(w * h),
    orientation: w >= h ? 0 : 90,
    score: 1.0,
    iscrowd: 0,
  });
  selectedIdx = annotations.length - 1;
  dirty = true;
  addMode = false;
  document.getElementById("btn-add-mode").style.background = "";
  render();
  renderCompPanel();
  showToast("Added new bbox — press 1-9/0 to set class");
});

function round2(v) { return Math.round(v * 100) / 100; }

// ── Keyboard shortcuts ──
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // 1-9, 0: change class
  const keyMap = {"1":1,"2":2,"3":3,"4":4,"5":5,"7":7,"8":8,"9":9,"0":10};
  if (keyMap[e.key] !== undefined && selectedIdx >= 0) {
    annotations[selectedIdx].category_id = keyMap[e.key];
    dirty = true;
    render();
    renderCompPanel();
    return;
  }

  // Delete
  if ((e.key === "f" || e.key === "F" || e.key === "Delete" || e.key === "Backspace" || e.key === "x" || e.key === "X") && selectedIdx >= 0) {
    annotations.splice(selectedIdx, 1);
    selectedIdx = -1;
    dirty = true;
    render();
    renderCompPanel();
    return;
  }

  // D: toggle add mode
  if (e.key === "d" || e.key === "D") {
    addMode = !addMode;
    document.getElementById("btn-add-mode").style.background = addMode ? "#e94560" : "";
    canvas.style.cursor = addMode ? "crosshair" : "default";
    return;
  }

  // O: orientation wheel
  if ((e.key === "o" || e.key === "O") && selectedIdx >= 0) {
    showOriWheel();
    return;
  }

  // Arrow keys: navigate boards
  if (e.key === "ArrowRight" || e.key === "ArrowDown") { navigateBoard(1); e.preventDefault(); return; }
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") { navigateBoard(-1); e.preventDefault(); return; }

  // Ctrl+S: save
  if (e.key === "s" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveBoard(); return; }
});

// ── Orientation wheel ──
function showOriWheel() {
  if (selectedIdx < 0) return;
  const wheel = document.getElementById("ori-wheel");
  wheel.style.display = "block";
  // Position near center of canvas
  const wrap = document.getElementById("canvas-wrap");
  const rect = wrap.getBoundingClientRect();
  wheel.style.left = (rect.width / 2 - 60) + "px";
  wheel.style.top = (rect.height / 2 - 60) + "px";

  wheel.innerHTML = "";
  const positions = [
    {a:0,x:46,y:2},{a:45,x:82,y:12},{a:90,x:92,y:46},{a:135,x:82,y:82},
    {a:180,x:46,y:92},{a:225,x:12,y:82},{a:270,x:2,y:46},{a:315,x:12,y:12}
  ];
  positions.forEach(p => {
    const btn = document.createElement("button");
    btn.className = "ori-btn";
    btn.textContent = p.a + "°";
    btn.style.left = p.x + "px";
    btn.style.top = p.y + "px";
    const currentOri = annotations[selectedIdx].orientation;
    if (p.a === currentOri) btn.style.background = "#e94560";
    btn.onclick = () => {
      annotations[selectedIdx].orientation = p.a;
      dirty = true;
      wheel.style.display = "none";
      render();
      renderCompPanel();
    };
    wheel.appendChild(btn);
  });

  // Close on click outside
  setTimeout(() => {
    document.addEventListener("click", function handler(ev) {
      if (!wheel.contains(ev.target)) {
        wheel.style.display = "none";
        document.removeEventListener("click", handler);
      }
    });
  }, 100);
}

// ── Navigation ──
function navigateBoard(dir) {
  if (!currentBoard) return;
  const filtered = getFilteredBoards();
  const idx = filtered.findIndex(b => b.name === currentBoard);
  const next = idx + dir;
  if (next >= 0 && next < filtered.length) loadBoard(filtered[next].name);
}

function getFilteredBoards() {
  const search = document.getElementById("search").value.toLowerCase();
  const colorF = document.getElementById("filter-color").value;
  const resF = document.getElementById("filter-res").value;
  const statusF = document.getElementById("filter-status").value;
  return boards.filter(b => {
    if (search && !b.name.toLowerCase().includes(search)) return false;
    if (colorF && b.color !== colorF) return false;
    if (resF && b.resolution !== resF) return false;
    if (statusF === "reviewed" && !b.reviewed) return false;
    if (statusF === "unreviewed" && (b.reviewed || b.excluded)) return false;
    if (statusF === "excluded" && !b.excluded) return false;
    return true;
  });
}

// ── Save ──
async function saveBoard() {
  if (!currentBoard) return;
  const body = {
    annotations: annotations,
    board_color: document.getElementById("color-override").value,
    resolution_class: document.getElementById("res-override").value,
  };
  const res = await fetch(`/api/save/${currentBoard}`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
  });
  const data = await res.json();
  if (data.ok) {
    dirty = false;
    // Update local board info
    const b = boards.find(b => b.name === currentBoard);
    if (b) { b.reviewed = true; b.num_annotations = data.count; b.color = body.board_color; b.resolution = body.resolution_class; }
    renderBoardList();
    updateStats();
    showToast(`Saved ${data.count} annotations`);
  }
}

// ── Buttons ──
document.getElementById("btn-save").onclick = saveBoard;
document.getElementById("btn-add-mode").onclick = () => {
  addMode = !addMode;
  document.getElementById("btn-add-mode").style.background = addMode ? "#e94560" : "";
  canvas.style.cursor = addMode ? "crosshair" : "default";
};
document.getElementById("btn-exclude").onclick = async () => {
  if (!currentBoard) return;
  const res = await fetch(`/api/toggle_exclude/${currentBoard}`, {method:"POST"});
  const data = await res.json();
  const b = boards.find(b => b.name === currentBoard);
  if (b) b.excluded = data.action === "excluded";
  renderBoardList();
  updateStats();
  showToast(data.action);
};
document.getElementById("btn-mark-reviewed").onclick = async () => {
  if (!currentBoard) return;
  await fetch(`/api/mark_reviewed/${currentBoard}`, {method:"POST"});
  const b = boards.find(b => b.name === currentBoard);
  if (b) b.reviewed = true;
  renderBoardList();
  updateStats();
  showToast("Marked as reviewed");
};
document.getElementById("btn-prev").onclick = () => navigateBoard(-1);
document.getElementById("btn-next").onclick = () => navigateBoard(1);

// Filters
document.getElementById("search").oninput = renderBoardList;
document.getElementById("filter-color").onchange = renderBoardList;
document.getElementById("filter-res").onchange = renderBoardList;
document.getElementById("filter-status").onchange = renderBoardList;

// Resize
window.addEventListener("resize", () => { if (boardImg) { fitCanvas(); render(); } });

let compPanelCollapsed = false;
function toggleCompPanel() {
  const panel = document.getElementById("comp-panel");
  const list = document.getElementById("comp-list");
  compPanelCollapsed = !compPanelCollapsed;
  if (compPanelCollapsed) {
    list.style.display = "none";
    panel.style.height = "24px";
    panel.style.minHeight = "24px";
  } else {
    list.style.display = "flex";
    panel.style.height = "60px";
    panel.style.minHeight = "40px";
  }
  // Recalc canvas after layout change
  setTimeout(() => { if (boardImg) { fitCanvas(); render(); } }, 50);
}

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.display = "block";
  setTimeout(() => t.style.display = "none", 2000);
}

init();
</script>
</body>
</html>
"""


def main():
    global ANNO_DIR, IMAGE_DIR, BACKUP_DIR

    parser = argparse.ArgumentParser(description="PCB Annotation Correction Server")
    parser.add_argument("--anno_dir", type=str, required=True, help="Directory with annotation JSONs (pipeline output)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with board images")
    parser.add_argument("--port", type=int, default=8899, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    ANNO_DIR = Path(args.anno_dir)
    IMAGE_DIR = Path(args.image_dir)
    BACKUP_DIR = ANNO_DIR / "backup"

    if not ANNO_DIR.exists():
        print(f"Error: {ANNO_DIR} does not exist")
        return
    if not IMAGE_DIR.exists():
        print(f"Error: {IMAGE_DIR} does not exist")
        return

    load_state()

    print(f"Annotation dir: {ANNO_DIR}")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Backup dir: {BACKUP_DIR}")
    print(f"Server: http://0.0.0.0:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
