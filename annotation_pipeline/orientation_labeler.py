#!/usr/bin/env python3
"""
Orientation Labeling Tool — Flask server
Browse component crops by category in batches and label orientation (8 directions, 45° each).

Usage:
  python orientation_labeler.py [--port 8890]

Labels saved to: component_pool/orientation_labels.json
  { "train_BUF802RGTEVM-top_00_0001": 90, ... }  (angle in degrees: 0,45,90,135,180,225,270,315)
"""

import argparse
import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from PIL import Image

# ---------------------------------------------------------------------------
# Parse args early
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8890)
parser.add_argument("--data_root", type=str, default="/home/xinrui/projects/data/ti_pcb")
parser.add_argument("--project_dir", type=str, default="/home/xinrui/projects/PCB_structure_layout/component_pool")
_args = parser.parse_args()
DATA_ROOT = Path(_args.data_root)
PROJECT_DIR = Path(_args.project_dir)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOARD_SIZE  = 512
EDGE_MARGIN = 10
METADATA_CSV  = DATA_ROOT / "components" / "metadata_train.csv"
BOARD_IMG_DIR = DATA_ROOT / "COCO_label" / "cropped_512" / "images" / "train"
EXCLUDE_FILE  = PROJECT_DIR / "excluded_components.json"
LABEL_FILE    = PROJECT_DIR / "orientation_labels.json"
ORI_RECLASS_FILE = PROJECT_DIR / "orientation_reclassified.json"

ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
ORI_LABELS   = {
    0:   "→ E",  45:  "↗ NE", 90:  "↑ N",  135: "↖ NW",
    180: "← W", 225: "↙ SW", 270: "↓ S",  315: "↘ SE",
}
ORI_COLORS = {
    0: "#FF6B6B", 45: "#FFA07A", 90: "#4ECDC4", 135: "#45B7D1",
    180: "#96CEB4", 225: "#DDA0DD", 270: "#F0E68C", 315: "#FFB347",
}
CATS = ["RESISTOR","CAPACITOR","INDUCTOR","CONNECTOR","DIODE","LED","SWITCH","TRANSISTOR","IC"]
CAT_NAME_TO_TOKEN = {
    "Resistor": "RESISTOR", "Capacitor": "CAPACITOR", "Inductor": "INDUCTOR",
    "Connector": "CONNECTOR", "Diode": "DIODE", "LED": "LED",
    "Switch": "SWITCH", "Transistor": "TRANSISTOR",
    "Integrated_Circuit": "IC", "Integrated Circuit": "IC",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_metadata():
    by_cat = defaultdict(list)
    excluded_ids = set()
    if EXCLUDE_FILE.exists():
        excluded_ids = set(json.loads(EXCLUDE_FILE.read_text()))

    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            cat = CAT_NAME_TO_TOKEN.get(row["category_name"])
            if cat is None:
                continue
            x, y = float(row["bbox_x"]), float(row["bbox_y"])
            w, h = float(row["bbox_w"]), float(row["bbox_h"])
            if w <= 0 or h <= 0:
                continue
            by_cat[cat].append({
                "id": row["id"],
                "source_image": row["source_image"],
                "bbox_x": x, "bbox_y": y,
                "bbox_w": w, "bbox_h": h,
                "category": cat,
                "is_excluded": row["id"] in excluded_ids,
                "is_edge": (x < EDGE_MARGIN or y < EDGE_MARGIN or
                            x + w > BOARD_SIZE - EDGE_MARGIN or
                            y + h > BOARD_SIZE - EDGE_MARGIN),
            })

    total = sum(len(v) for v in by_cat.values())
    print(f"Loaded {total} crops (all, including excluded/edge)")
    for cat in sorted(by_cat):
        print(f"  {cat}: {len(by_cat[cat])}")
    return by_cat, excluded_ids

def load_labels():
    if LABEL_FILE.exists():
        return json.loads(LABEL_FILE.read_text())
    return {}

def save_labels(labels):
    LABEL_FILE.write_text(json.dumps(labels, indent=2))

def load_ori_reclassified():
    if ORI_RECLASS_FILE.exists():
        return json.loads(ORI_RECLASS_FILE.read_text())
    return {}

def save_ori_reclassified(data):
    ORI_RECLASS_FILE.write_text(json.dumps(data, indent=2))

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app     = Flask(__name__)
by_cat, excluded_ids = load_metadata()
all_crops = {}
for cat, entries in by_cat.items():
    for e in entries:
        all_crops[e["id"]] = e

labels = load_labels()
ori_reclassified = load_ori_reclassified()
print(f"Existing labels: {len(labels)}")
print(f"Existing ori-reclassifications: {len(ori_reclassified)}")

img_cache = {}

def get_board_image(source_image):
    if source_image not in img_cache:
        path = BOARD_IMG_DIR / f"{source_image}.png"
        if not path.exists():
            return None
        img_cache[source_image] = Image.open(path).convert("RGB")
        if len(img_cache) > 300:
            del img_cache[next(iter(img_cache))]
    return img_cache.get(source_image)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return HTML_PAGE

@app.route("/api/crops")
def api_crops():
    cat      = request.args.get("category", "RESISTOR")
    page     = int(request.args.get("page", 0))
    per_page = int(request.args.get("per_page", 150))
    filter_  = request.args.get("filter", "all")   # all | labeled | unlabeled
    sort_by  = request.args.get("sort_by", "default")  # default | orientation

    entries = list(by_cat.get(cat, []))
    # Default "normal" = exclude excluded+edge crops (original pool only)
    if filter_ == "normal":
        entries = [e for e in entries if not e["is_excluded"] and not e["is_edge"]]
    elif filter_ == "labeled":
        entries = [e for e in entries if e["id"] in labels]
    elif filter_ == "unlabeled":
        entries = [e for e in entries if e["id"] not in labels]
    elif filter_ == "needs_review":
        entries = [e for e in entries
                   if e["bbox_h"] > 0 and 0.6 <= (e["bbox_w"] / e["bbox_h"]) <= 1.8]
    elif filter_ == "excluded":
        entries = [e for e in entries if e["is_excluded"]]

    # Sort by orientation group: 0, 45, 90, 135, 180, 225, 270, 315, then unlabeled
    if sort_by == "orientation":
        def ori_key(e):
            lbl = labels.get(e["id"])
            if lbl is None:
                return (999, 0)   # unlabeled goes last
            return (ORIENTATIONS.index(lbl), 0)
        entries = sorted(entries, key=ori_key)

    start = page * per_page
    page_entries = entries[start:start + per_page]

    # Build group headers info for the page
    groups = []
    if sort_by == "orientation" and page_entries:
        prev_ori = "INIT"
        for e in page_entries:
            lbl = labels.get(e["id"])
            if lbl != prev_ori:
                groups.append({"orientation": lbl, "start_id": e["id"]})
                prev_ori = lbl

    return jsonify({
        "category": cat,
        "page": page,
        "per_page": per_page,
        "total": len(entries),
        "total_pages": max(1, (len(entries) + per_page - 1) // per_page),
        "sort_by": sort_by,
        "groups": groups,
        "crops": [
            {**e,
             "orientation": labels.get(e["id"]),
             "reclassified_to": ori_reclassified.get(e["id"]),
             "needs_review": 0.6 <= (e["bbox_w"] / e["bbox_h"]) <= 1.8
                             if e["bbox_h"] > 0 else False,
             "is_excluded": e.get("is_excluded", False)}
            for e in page_entries
        ],
    })

@app.route("/api/crop_image/<crop_id>")
def api_crop_image(crop_id):
    entry = all_crops.get(crop_id)
    if entry is None:
        return "Not found", 404
    board = get_board_image(entry["source_image"])
    if board is None:
        return "Board image not found", 404
    x1, y1 = entry["bbox_x"], entry["bbox_y"]
    x2, y2 = x1 + entry["bbox_w"], y1 + entry["bbox_h"]
    crop = board.crop((x1, y1, x2, y2))
    crop.thumbnail((128, 128), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/api/label", methods=["POST"])
def api_label():
    """Label one or multiple crops."""
    data  = request.json
    ids   = data.get("ids", [])
    angle = data.get("angle")
    if angle is None or angle not in ORIENTATIONS:
        return jsonify({"error": "Invalid angle"}), 400
    for crop_id in ids:
        if crop_id in all_crops:
            labels[crop_id] = angle
    save_labels(labels)
    return jsonify({"ok": True, "total_labeled": len(labels)})

@app.route("/api/unlabel", methods=["POST"])
def api_unlabel():
    """Remove labels from crops."""
    ids = request.json.get("ids", [])
    for crop_id in ids:
        labels.pop(crop_id, None)
    save_labels(labels)
    return jsonify({"ok": True, "total_labeled": len(labels)})

@app.route("/api/label_all_page", methods=["POST"])
def api_label_all_page():
    """Label all crops on the current page with the same angle."""
    data     = request.json
    ids      = data.get("ids", [])
    angle    = data.get("angle")
    if angle is None or angle not in ORIENTATIONS:
        return jsonify({"error": "Invalid angle"}), 400
    for crop_id in ids:
        if crop_id in all_crops:
            labels[crop_id] = angle
    save_labels(labels)
    return jsonify({"ok": True, "labeled": len(ids), "total_labeled": len(labels)})

@app.route("/api/reclassify", methods=["POST"])
def api_reclassify():
    data    = request.json
    ids     = data.get("ids", [])
    new_cat = data.get("category", "")
    if new_cat not in CATS:
        return jsonify({"error": "Invalid category"}), 400
    for crop_id in ids:
        entry = all_crops.get(crop_id)
        if entry is None:
            continue
        if new_cat == entry["category"]:
            ori_reclassified.pop(crop_id, None)  # undo if same as original
        else:
            ori_reclassified[crop_id] = new_cat
    save_ori_reclassified(ori_reclassified)
    return jsonify({"ok": True, "total_reclassified": len(ori_reclassified)})

@app.route("/api/undo_reclassify", methods=["POST"])
def api_undo_reclassify():
    ids = request.json.get("ids", [])
    for crop_id in ids:
        ori_reclassified.pop(crop_id, None)
    save_ori_reclassified(ori_reclassified)
    return jsonify({"ok": True, "total_reclassified": len(ori_reclassified)})

@app.route("/api/stats")
def api_stats():
    stats = {}
    for cat in CATS:
        entries = by_cat.get(cat, [])
        labeled = sum(1 for e in entries if e["id"] in labels)
        stats[cat] = {
            "total": len(entries),
            "labeled": labeled,
            "unlabeled": len(entries) - labeled,
        }
    # Orientation distribution across all
    ori_dist = defaultdict(int)
    for angle in labels.values():
        ori_dist[angle] += 1
    return jsonify({"categories": stats, "orientation_distribution": dict(ori_dist)})

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Orientation Labeler</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #0f0f1a; color: #ddd; padding: 16px; }
h1 { text-align: center; color: #4ECDC4; margin-bottom: 12px; font-size: 22px; }
.controls { display: flex; gap: 10px; align-items: center; justify-content: center; flex-wrap: wrap; margin-bottom: 10px; }
.controls select, .controls button, .controls input {
  padding: 6px 12px; border-radius: 6px; border: 1px solid #444;
  background: #1e1e2e; color: #ddd; font-size: 13px; cursor: pointer;
}
.controls button:hover { background: #2e2e4e; }
.controls button.danger { border-color: #e74c3c; color: #e74c3c; }
.ori-bar { display: flex; gap: 6px; align-items: center; justify-content: center; flex-wrap: wrap; margin-bottom: 10px; }
.ori-btn {
  padding: 6px 12px; border-radius: 6px; border: 2px solid #444;
  background: #1e1e2e; color: #ddd; font-size: 13px; cursor: pointer;
  transition: all 0.15s; min-width: 70px; text-align: center;
}
.ori-btn:hover { transform: scale(1.08); }
.ori-btn.active-filter { border-width: 3px; font-weight: bold; }
.stats { text-align: center; font-size: 13px; color: #888; margin-bottom: 8px; }
.cat-stats { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin: 6px 0; }
.cat-badge { font-size: 11px; padding: 3px 8px; border-radius: 4px; background: #1e1e2e; border: 1px solid #333; }
.pager { display: flex; gap: 8px; align-items: center; justify-content: center; margin: 10px 0; }
.pager button { padding: 4px 12px; border-radius: 4px; border: 1px solid #444; background: #1e1e2e; color: #ddd; cursor: pointer; }
.pager button:disabled { opacity: 0.3; cursor: default; }
.pager span { font-size: 13px; color: #888; }
.pager .page-input { width: 50px; text-align: center; background: #1e1e2e; color: #ddd; border: 1px solid #444; border-radius: 4px; padding: 2px 4px; font-size: 13px; }
.grid { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
.card {
  position: relative; width: 90px; border: 2px solid #333;
  border-radius: 6px; overflow: hidden; cursor: pointer;
  transition: all 0.15s; background: #1a1a2e;
}
.card:hover { border-color: #4ECDC4; transform: scale(1.05); }
.card.selected { outline: 3px solid #f1c40f; }
.card.needs-review { border-color: #f39c12; }
.card.needs-review .review-dot { display: block; }
.review-dot { display: none; position: absolute; top: 2px; left: 2px; width: 8px; height: 8px; background: #f39c12; border-radius: 50%; }
.card.reclassified { border-color: #f39c12; }
.card.is-excluded { border-style: solid; opacity: 1; }
.reclass-badge { position: absolute; bottom: 18px; right: 2px; background: #f39c12; color: #000; font-size: 9px; padding: 1px 4px; border-radius: 3px; font-weight: bold; }
.card img { display: block; width: 100%; height: 66px; object-fit: contain; background: #fff; }
.card .info { font-size: 10px; padding: 3px 4px; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card .ori-badge {
  position: absolute; top: 2px; right: 2px;
  font-size: 10px; padding: 1px 5px; border-radius: 3px;
  font-weight: bold; color: #000;
}
.toast { position: fixed; bottom: 20px; right: 20px; background: #4ECDC4; color: #000; padding: 10px 20px; border-radius: 8px; font-size: 14px; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 999; }
.toast.show { opacity: 1; }
.section-title { color: #888; font-size: 12px; text-align: center; margin: 4px 0; }
.kbkey { width:36px;height:36px;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;transition:transform 0.1s,filter 0.1s;border:1px solid rgba(255,255,255,0.15); color:#000; }
.kbkey:hover { transform:scale(1.12); filter:brightness(1.2); }
.kbkey:active { transform:scale(0.95); }
.kbkey span { font-size:13px; font-weight:bold; line-height:1; }
.kbkey small { font-size:9px; line-height:1; }
kbd { background:#2e2e4e; border:1px solid #555; border-radius:3px; padding:1px 5px; font-size:11px; color:#ddd; }
</style>
</head><body>
<h1>🧭 Orientation Labeler</h1>

<!-- Category + filter controls -->
<div class="controls">
  <select id="catSelect"></select>
  <select id="filterSelect">
    <option value="normal">Normal (pool only)</option>
    <option value="all">All</option>
    <option value="excluded">🚫 Excluded only</option>
    <option value="unlabeled">Unlabeled only</option>
    <option value="labeled">Labeled only</option>
    <option value="needs_review">⚠️ Needs Review (AR 0.6–1.8)</option>
  </select>
  <select id="sortSelect">
    <option value="default">Sort: Default</option>
    <option value="orientation">Sort: By Orientation</option>
  </select>
  <button onclick="loadPage(0)">Reload</button>
  <button onclick="selectAll()">Select All</button>
  <button onclick="clearSelection()">Clear</button>
  <button class="danger" onclick="unlabelSelected()">Remove Labels</button>
  <span style="color:#555">|</span>
  <span style="color:#f39c12">Reclassify to:</span>
  <select id="reclassCatSelect"></select>
  <button style="border-color:#f39c12;color:#f39c12" onclick="reclassifySelected()">Apply</button>
  <button onclick="undoReclassifySelected()">Undo Reclass</button>
</div>

<!-- Orientation buttons — apply to selected -->
<div class="section-title">Label selected crops:</div>
<div style="display:flex;gap:24px;align-items:center;justify-content:center;margin:6px 0 4px 0;flex-wrap:wrap">
  <!-- WASD key grid -->
  <div>
    <div style="font-size:11px;color:#666;text-align:center;margin-bottom:4px">Keyboard (select cards first)</div>
    <div style="display:grid;grid-template-columns:repeat(3,36px);grid-template-rows:repeat(3,36px);gap:3px">
      <div class="kbkey" style="background:#45B7D1" onclick="kbLabel(135)"><span>Q</span><small>↖NW</small></div>
      <div class="kbkey" style="background:#4ECDC4" onclick="kbLabel(90)"><span>W</span><small>↑N</small></div>
      <div class="kbkey" style="background:#FFA07A" onclick="kbLabel(45)"><span>E</span><small>↗NE</small></div>
      <div class="kbkey" style="background:#96CEB4" onclick="kbLabel(180)"><span>A</span><small>←W</small></div>
      <div class="kbkey" style="background:#333;color:#555;cursor:default"><span>·</span><small></small></div>
      <div class="kbkey" style="background:#FF6B6B" onclick="kbLabel(0)"><span>D</span><small>→E</small></div>
      <div class="kbkey" style="background:#DDA0DD" onclick="kbLabel(225)"><span>Z</span><small>↙SW</small></div>
      <div class="kbkey" style="background:#F0E68C;color:#333" onclick="kbLabel(270)"><span>S</span><small>↓S</small></div>
      <div class="kbkey" style="background:#FFB347;color:#333" onclick="kbLabel(315)"><span>C</span><small>↘SE</small></div>
    </div>
  </div>
  <!-- number key row -->
  <div>
    <div style="font-size:11px;color:#666;text-align:center;margin-bottom:4px">or number keys 1–8</div>
    <div style="display:flex;gap:3px">
      <div class="kbkey" style="background:#45B7D1" onclick="kbLabel(135)"><span>1</span><small>↖</small></div>
      <div class="kbkey" style="background:#4ECDC4" onclick="kbLabel(90)"><span>2</span><small>↑</small></div>
      <div class="kbkey" style="background:#FFA07A" onclick="kbLabel(45)"><span>3</span><small>↗</small></div>
      <div class="kbkey" style="background:#96CEB4" onclick="kbLabel(180)"><span>4</span><small>←</small></div>
      <div class="kbkey" style="background:#FF6B6B" onclick="kbLabel(0)"><span>5</span><small>→</small></div>
      <div class="kbkey" style="background:#DDA0DD" onclick="kbLabel(225)"><span>6</span><small>↙</small></div>
      <div class="kbkey" style="background:#F0E68C;color:#333" onclick="kbLabel(270)"><span>7</span><small>↓</small></div>
      <div class="kbkey" style="background:#FFB347;color:#333" onclick="kbLabel(315)"><span>8</span><small>↘</small></div>
    </div>
  </div>
  <!-- other keys -->
  <div style="font-size:11px;color:#666;line-height:1.8">
    <div><kbd>Esc</kbd> clear selection</div>
    <div><kbd>←</kbd><kbd>→</kbd> prev/next page</div>
    <div><kbd>A</kbd> select all (when nothing selected)</div>
  </div>
</div>
<div class="ori-bar" id="oriBar"></div>

<!-- Batch: label ALL on page -->


<div class="stats" id="stats"></div>
<div class="cat-stats" id="catStats"></div>
<div id="boardIndicator" style="text-align:center;font-size:13px;min-height:22px;margin:4px 0;color:#4ECDC4;font-family:monospace;letter-spacing:0.5px;"></div>
<div class="pager" id="pagerTop"></div>
<div class="grid" id="grid"></div>
<div class="pager" id="pagerBottom"></div>
<div class="toast" id="toast"></div>

<script>
const CATS = ["RESISTOR","CAPACITOR","INDUCTOR","CONNECTOR","DIODE","LED","SWITCH","TRANSISTOR","IC"];
const ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315];
const ORI_LABELS = {0:"→ E", 45:"↗ NE", 90:"↑ N", 135:"↖ NW", 180:"← W", 225:"↙ SW", 270:"↓ S", 315:"↘ SE"};
const ORI_COLORS = {0:"#FF6B6B", 45:"#FFA07A", 90:"#4ECDC4", 135:"#45B7D1", 180:"#96CEB4", 225:"#DDA0DD", 270:"#F0E68C", 315:"#FFB347"};

let currentCat = "RESISTOR", currentPage = 0, totalPages = 0, totalCrops = 0;
let selected = new Set();
let lastClicked = -1;
let cropsData = [];

// Build category select
const catSel = document.getElementById("catSelect");
CATS.forEach(c => { const o = document.createElement("option"); o.value=c; o.text=c; catSel.appendChild(o); });
catSel.onchange = () => { currentCat = catSel.value; loadPage(0); };

// Build reclassify select
const reclassSel = document.getElementById("reclassCatSelect");
CATS.forEach(c => { const o = document.createElement("option"); o.value=c; o.text=c; reclassSel.appendChild(o); });

document.getElementById("filterSelect").onchange = () => loadPage(0);
document.getElementById("sortSelect").onchange = () => loadPage(0);

// Build orientation bars
function buildOriBar(containerId, clickFn) {
  const bar = document.getElementById(containerId);
  ORIENTATIONS.forEach(a => {
    const btn = document.createElement("button");
    btn.className = "ori-btn";
    btn.style.borderColor = ORI_COLORS[a];
    btn.style.color = ORI_COLORS[a];
    btn.innerHTML = ORI_LABELS[a] + "<br><small>" + a + "°</small>";
    btn.onclick = () => clickFn(a);
    bar.appendChild(btn);
  });
}
buildOriBar("oriBar", (a) => labelSelected(a));


function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2000);
}

async function loadStats() {
  const r = await fetch("/api/stats");
  const data = await r.json();
  let totalL = 0, totalU = 0;
  let badges = "";
  for (const cat of CATS) {
    const s = data.categories[cat] || {total:0, labeled:0, unlabeled:0};
    totalL += s.labeled; totalU += s.unlabeled;
    badges += '<span class="cat-badge">' + cat + ': <span style="color:#4ECDC4">' + s.labeled + '</span>/' + s.total + '</span>';
  }
  document.getElementById("stats").innerHTML =
    'Labeled: <b style="color:#4ECDC4">' + totalL + '</b> | Unlabeled: <b style="color:#888">' + totalU + '</b> | Total: ' + (totalL+totalU);
  document.getElementById("catStats").innerHTML = badges;
}

async function loadPage(page) {
  currentPage = page;
  selected.clear(); lastClicked = -1;
  const filter  = document.getElementById("filterSelect").value;
  const sortBy  = document.getElementById("sortSelect").value;
  const r = await fetch("/api/crops?category=" + currentCat + "&page=" + page + "&per_page=150&filter=" + filter + "&sort_by=" + sortBy);
  const data = await r.json();
  totalPages = data.total_pages; totalCrops = data.total;
  cropsData = data.crops;
  currentGroups = data.groups || [];
  renderGrid();
  renderPager();
  loadStats();
}

let currentGroups = [];

function renderGrid() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  // Build a set of IDs that start a new group (for group headers)
  const groupStartIds = new Set((currentGroups || []).map(g => g.start_id));

  cropsData.forEach((crop, idx) => {
    // Insert group header divider
    if (groupStartIds.has(crop.id)) {
      const grp = (currentGroups || []).find(g => g.start_id === crop.id);
      const ori = grp ? grp.orientation : null;
      const divider = document.createElement("div");
      divider.style.cssText = "width:100%;text-align:center;padding:6px 0;font-size:13px;font-weight:bold;color:" + (ori !== null ? ORI_COLORS[ori] : "#888");
      divider.textContent = ori !== null ? ("— " + ORI_LABELS[ori] + " (" + ori + "°) —") : "— Unlabeled —";
      grid.appendChild(divider);
    }

    const card = document.createElement("div");
    card.className = "card" + (selected.has(idx) ? " selected" : "");
    const ori = crop.orientation;
    const badge = ori !== null && ori !== undefined
      ? '<span class="ori-badge" style="background:' + ORI_COLORS[ori] + '">' + ORI_LABELS[ori] + '</span>'
      : '';
    if (crop.needs_review) card.classList.add("needs-review");
    if (crop.is_excluded) card.classList.add("is-excluded");
    if (crop.reclassified_to) card.classList.add("reclassified");
    const reclBadge = crop.reclassified_to
      ? '<span style="position:absolute;bottom:2px;right:2px;background:#f39c12;color:#000;font-size:9px;padding:1px 4px;border-radius:3px;font-weight:bold">' + crop.reclassified_to + '</span>'
      : '';
    card.innerHTML = badge + reclBadge
      + '<span class="review-dot" title="Near-square (AR 0.6-1.8): check orientation"></span>'
      + '<img src="/api/crop_image/' + crop.id + '" loading="lazy">'
      + '<div class="info">' + Math.round(crop.bbox_w) + 'x' + Math.round(crop.bbox_h) + ' | ' + crop.source_image.slice(0,18) + '</div>';
    card.onclick = (e) => handleClick(idx, e);
    grid.appendChild(card);
  });
}

function handleClick(idx, e) {
  if (e.shiftKey && lastClicked >= 0) {
    const lo = Math.min(lastClicked, idx), hi = Math.max(lastClicked, idx);
    for (let i = lo; i <= hi; i++) selected.add(i);
  } else {
    if (selected.has(idx)) selected.delete(idx); else selected.add(idx);
  }
  lastClicked = idx;
  // Show board name indicator
  const crop = cropsData[idx];
  const ind = document.getElementById("boardIndicator");
  if (selected.size === 1 && selected.has(idx)) {
    ind.textContent = "📋 " + crop.source_image + "  |  " + crop.category + "  |  " + Math.round(crop.bbox_w) + "×" + Math.round(crop.bbox_h) + "px  |  AR " + (crop.bbox_w/crop.bbox_h).toFixed(2);
  } else if (selected.size === 0) {
    ind.textContent = "";
  } else {
    ind.textContent = "📋 " + selected.size + " selected  |  last: " + crop.source_image;
  }
  renderGrid();
}

function selectAll() {
  cropsData.forEach((_, i) => selected.add(i));
  renderGrid();
}

function clearSelection() {
  selected.clear();
  document.getElementById("boardIndicator").textContent = "";
  renderGrid();
}

async function labelSelected(angle) {
  const ids = [...selected].map(i => cropsData[i].id);
  if (!ids.length) { toast("Nothing selected — click cards first"); return; }
  await fetch("/api/label", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ids, angle})
  });
  ids.forEach(id => { const c = cropsData.find(x=>x.id===id); if(c) c.orientation=angle; });
  selected.clear();
  renderGrid();
  loadStats();
  toast("Labeled " + ids.length + " crops as " + ORI_LABELS[angle]);
}

async function labelAllPage(angle) {
  const ids = cropsData.map(c => c.id);
  if (!ids.length) { toast("No crops on this page"); return; }
  await fetch("/api/label_all_page", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ids, angle})
  });
  cropsData.forEach(c => c.orientation = angle);
  selected.clear();
  renderGrid();
  loadStats();
  toast("Batch labeled " + ids.length + " crops as " + ORI_LABELS[angle]);
}

async function reclassifySelected() {
  const ids = [...selected].map(i => cropsData[i].id);
  if (!ids.length) { toast("Nothing selected"); return; }
  const newCat = document.getElementById("reclassCatSelect").value;
  await fetch("/api/reclassify", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ids, category: newCat})
  });
  ids.forEach(id => {
    const c = cropsData.find(x=>x.id===id);
    if (c) c.reclassified_to = (newCat === c.category ? null : newCat);
  });
  selected.clear();
  renderGrid();
  toast("Reclassified " + ids.length + " crops to " + newCat);
}

async function undoReclassifySelected() {
  const ids = [...selected].map(i => cropsData[i].id).filter(i => {
    const c = cropsData.find(x=>x.id===i); return c && c.reclassified_to;
  });
  if (!ids.length) { toast("No reclassified crops selected"); return; }
  await fetch("/api/undo_reclassify", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ids})
  });
  ids.forEach(id => { const c = cropsData.find(x=>x.id===id); if(c) c.reclassified_to = null; });
  selected.clear();
  renderGrid();
  toast("Undid reclassification for " + ids.length + " crops");
}

async function unlabelSelected() {
  const ids = [...selected].map(i => cropsData[i].id);
  if (!ids.length) { toast("Nothing selected"); return; }
  await fetch("/api/unlabel", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ids})
  });
  ids.forEach(id => { const c = cropsData.find(x=>x.id===id); if(c) c.orientation=null; });
  selected.clear();
  renderGrid();
  loadStats();
  toast("Removed labels from " + ids.length + " crops");
}

function renderPager() {
  const prevDis = currentPage === 0 ? " disabled" : "";
  const nextDis = currentPage >= totalPages - 1 ? " disabled" : "";
  const html = '<button onclick="loadPage(' + (currentPage-1) + ')"' + prevDis + '>&laquo; Prev</button>'
    + '<span> Page </span>'
    + '<input type="number" class="page-input" min="1" max="' + totalPages + '" value="' + (currentPage+1) + '">'
    + '<span> / ' + totalPages + ' &nbsp;(' + totalCrops + ' crops)</span>'
    + '<button onclick="loadPage(' + (currentPage+1) + ')"' + nextDis + '>Next &raquo;</button>';
  ["pagerTop","pagerBottom"].forEach(id => {
    document.getElementById(id).innerHTML = html;
    document.getElementById(id).querySelectorAll(".page-input").forEach(inp => {
      inp.addEventListener("change", () => {
        const p = parseInt(inp.value) - 1;
        if (p >= 0 && p < totalPages) loadPage(p);
      });
      inp.addEventListener("keydown", e => {
        if (e.key === "Enter") { const p = parseInt(inp.value)-1; if(p>=0&&p<totalPages) loadPage(p); }
      });
    });
  });
}

// Keyboard: ← → for page nav
function kbLabel(angle) {
  if (selected.size > 0) labelSelected(angle);
  else toast("Select cards first, then press a key or click a key button");
}

// Hotkey map: key → angle
function kbLabel(angle) { if (selected.size > 0) labelSelected(angle); else toast("Select cards first"); }

const HOTKEYS = {
  'd': 0,    // → E
  'e': 45,   // ↗ NE
  'w': 90,   // ↑ N
  'q': 135,  // ↖ NW
  'a': 180,  // ← W  (also select-all when nothing selected)
  'z': 225,  // ↙ SW
  's': 270,  // ↓ S
  'c': 315,  // ↘ SE
  // number keys
  '1': 0, '2': 45, '3': 90, '4': 135,
  '5': 180, '6': 225, '7': 270, '8': 315,
};

document.addEventListener("keydown", e => {
  if (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "SELECT") return;
  if (e.key === "ArrowLeft"  && currentPage > 0) loadPage(currentPage - 1);
  if (e.key === "ArrowRight" && currentPage < totalPages - 1) loadPage(currentPage + 1);
  if (e.key === "Escape") { clearSelection(); return; }
  // 'a' = select-all when nothing selected, else label W(180°)
  if (e.key === "a") {
    if (selected.size === 0) { selectAll(); return; }
  }
  const angle = HOTKEYS[e.key];
  if (angle !== undefined) kbLabel(angle);
});

loadPage(0);
</script>
</body></html>"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting orientation labeler on port {_args.port}")
    print(f"Open: http://10.97.27.230:{_args.port}")
    app.run(host="0.0.0.0", port=_args.port, debug=False)
