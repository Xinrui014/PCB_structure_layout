#!/usr/bin/env python3
"""
Resolution analysis: compute per-board median Resistor body size and serve
a histogram + visual gallery with delete functionality.

Run on L40: python resolution_analysis.py --port 8895
"""
import json, os, glob, argparse, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import statistics

ANNO_DIR = "/home/xinrui/projects/PCB_structure_layout/re-annotation/final_annotations"
IMG_DIR = "/home/xinrui/projects/data/ti_pcb/images_top"
DELETED_FILE = os.path.join(os.path.dirname(ANNO_DIR), "resolution_deleted_boards.json")

ANCHOR_CATS = ["Resistor", "Capacitor"]  # priority order: R first, C fallback


def load_deleted():
    if os.path.exists(DELETED_FILE):
        with open(DELETED_FILE) as f:
            return set(json.load(f))
    return set()


def save_deleted(deleted):
    with open(DELETED_FILE, "w") as f:
        json.dump(sorted(deleted), f, indent=2)


def analyze():
    """Analyze all boards. Returns (with_resistors, no_resistors) lists."""
    with_r = []
    no_r = []
    for path in sorted(glob.glob(os.path.join(ANNO_DIR, "*.json"))):
        board = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        # Skip non-COCO files (e.g. board_colors_v2.json)
        if "categories" not in data or "annotations" not in data:
            continue
        cats = {c["id"]: c["name"] for c in data["categories"]}
        # Count by category
        cat_counts = {}
        sizes_by_cat = {c: [] for c in ANCHOR_CATS}
        all_anns = data.get("annotations", [])
        for ann in all_anns:
            cat = cats.get(ann["category_id"], "Unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if cat in sizes_by_cat:
                x, y, w, h = ann["bbox"]
                body = min(w, h)
                if body > 0:
                    sizes_by_cat[cat].append(body)

        # Pick anchor: Resistor first, then Capacitor fallback
        anchor_sizes = None
        anchor_used = None
        for ac in ANCHOR_CATS:
            if sizes_by_cat[ac]:
                anchor_sizes = sizes_by_cat[ac]
                anchor_used = ac
                break

        info = {
            "board": board,
            "n_total": len(all_anns),
            "cat_counts": cat_counts,
        }
        if anchor_sizes:
            info["median_r"] = round(statistics.median(anchor_sizes), 2)
            info["mean_r"] = round(statistics.mean(anchor_sizes), 2)
            info["min_r"] = round(min(anchor_sizes), 2)
            info["max_r"] = round(max(anchor_sizes), 2)
            info["n_r"] = len(anchor_sizes)
            info["anchor"] = anchor_used
            with_r.append(info)
        else:
            info["median_r"] = None
            info["n_r"] = 0
            info["anchor"] = None
            no_r.append(info)

    with_r.sort(key=lambda r: r["median_r"])
    no_r.sort(key=lambda r: r["board"])
    return with_r, no_r


def find_image(board):
    for ext in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(IMG_DIR, board + ext)
        if os.path.exists(p):
            return p
    return None


def build_html(with_r, no_r, deleted):
    board_data = []
    for r in with_r:
        board_data.append({
            "board": r["board"],
            "median": r["median_r"],
            "mean": r.get("mean_r"),
            "min": r.get("min_r"),
            "max": r.get("max_r"),
            "n_r": r["n_r"],
            "n_total": r["n_total"],
            "cats": r["cat_counts"],
            "has_r": True,
            "anchor": r.get("anchor", "Resistor"),
            "deleted": r["board"] in deleted,
        })
    for r in no_r:
        board_data.append({
            "board": r["board"],
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "n_r": 0,
            "n_total": r["n_total"],
            "cats": r["cat_counts"],
            "has_r": False,
            "anchor": None,
            "deleted": r["board"] in deleted,
        })

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB Resolution Analysis — Resistor Body Size</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }
h1 { text-align: center; padding: 20px; color: #00d4ff; font-size: 1.5em; }
.container { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
.stats { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-card { background: #16213e; border-radius: 8px; padding: 15px; flex: 1; min-width: 120px; text-align: center; }
.stat-card .val { font-size: 1.6em; color: #00d4ff; font-weight: bold; }
.stat-card .lbl { font-size: 0.8em; color: #888; margin-top: 5px; }
.stat-card.warn .val { color: #e94560; }

.hist-section { background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.hist-section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.1em; }
.hist-container { position: relative; height: 250px; display: flex; align-items: flex-end; gap: 2px; padding-bottom: 30px; }
.hist-bar { flex: 1; border-radius: 4px 4px 0 0; cursor: pointer; position: relative; min-width: 8px; transition: background 0.2s; }
.hist-bar:hover { opacity: 0.8; }
.hist-bar.selected { outline: 2px solid #fff; }
.hist-bar .count { position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 11px; color: #aaa; white-space: nowrap; }
.hist-bar .label { position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #888; white-space: nowrap; }

.controls { background: #16213e; border-radius: 12px; padding: 15px 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
.controls label { color: #aaa; font-size: 13px; }
.controls select, .controls input { background: #0a0a23; border: 1px solid #333; color: #fff; padding: 5px 8px; border-radius: 4px; }
.controls button { background: #0f3460; color: #fff; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
.controls button:hover { background: #00d4ff; color: #000; }
.class-summary { color: #e94560; font-size: 13px; }

.gallery-section { background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.gallery-section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.1em; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
.card { background: #0a0a23; border-radius: 8px; overflow: hidden; position: relative; transition: transform 0.2s; }
.card:hover { transform: scale(1.02); }
.card.is-deleted { opacity: 0.4; }
.card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; }
.card .info { padding: 10px; font-size: 12px; }
.card .info .board-name { color: #00d4ff; font-weight: bold; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card .info .metrics { color: #aaa; line-height: 1.6; }
.card .info .metrics b { color: #e94560; }
.card .info .cat-list { color: #888; font-size: 11px; margin-top: 4px; }
.card .res-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
.card .del-btn { position: absolute; top: 8px; right: 8px; background: #e94560; color: #fff; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; z-index: 10; }
.card .del-btn:hover { background: #ff1744; }
.card .del-btn.undo { background: #4caf50; }
.card .del-btn.undo:hover { background: #66bb6a; }

.pagination { display: flex; justify-content: center; gap: 8px; margin-top: 15px; }
.pagination button { background: #0f3460; color: #fff; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
.pagination button:hover { background: #00d4ff; color: #000; }
.pagination button:disabled { opacity: 0.3; cursor: default; }
.pagination .page-info { color: #aaa; line-height: 32px; font-size: 13px; }
.no-r-badge { background: #e9456033; color: #e94560; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
</style>
</head>
<body>
<h1>🔍 PCB Resolution Analysis — Resistor Body Size</h1>
<div class="container">
    <div class="stats" id="stats"></div>
    <div class="hist-section">
        <h2>📊 Histogram — Median Resistor Body Size (min(w,h) px) per Board</h2>
        <div class="hist-container" id="histogram"></div>
    </div>
    <div class="controls">
        <label>Filter:</label>
        <select id="filter-select" onchange="page=0;updateGallery()">
            <option value="all">All boards</option>
            <option value="has_r">With resistors</option>
            <option value="no_r">⚠️ No resistors (""" + str(len(no_r)) + """)</option>
            <option value="remove">Remove zone (0-6px)</option>
            <option value="R1">R1 (0-6px)</option>
            <option value="R2">R2 (6-8px)</option>
            <option value="R3">R3 (8-12px)</option>
            <option value="R4">R4 (12-16px)</option>
            <option value="R5">R5 (16-22px)</option>
            <option value="R6">R6 (22-28px)</option>
            <option value="R7">R7 (28+px)</option>
            <option value="cap_anchor">⚡ Cap anchor</option>
            <option value="deleted">🗑️ Deleted</option>
            <option value="not_deleted">Not deleted</option>
        </select>
        <select id="sort-select" onchange="updateGallery()">
            <option value="asc">↑ Smallest first</option>
            <option value="desc">↓ Largest first</option>
            <option value="name">A-Z name</option>
        </select>
        <input type="text" id="search" placeholder="Search board..." oninput="page=0;updateGallery()">
        <div class="class-summary" id="class-summary"></div>
    </div>
    <div class="gallery-section">
        <h2 id="gallery-title">🖼️ Board Gallery</h2>
        <div class="gallery" id="gallery"></div>
        <div class="pagination" id="pagination"></div>
    </div>
</div>

<script>
const DATA = """ + json.dumps(board_data) + """;
let page = 0;
const PAGE_SIZE = 24;
let selectedBin = null;

const CLASS_BINS = [
    {name: 'Remove', lo: 0, hi: 6, color: '#666'},
    {name: 'R1', lo: 6, hi: 8, color: '#e94560'},
    {name: 'R2', lo: 8, hi: 12, color: '#f5a623'},
    {name: 'R3', lo: 12, hi: 16, color: '#00d4ff'},
    {name: 'R4', lo: 16, hi: 22, color: '#4caf50'},
    {name: 'R5', lo: 22, hi: 28, color: '#9c27b0'},
    {name: 'R6', lo: 28, hi: 9999, color: '#ff9800'},
];

function getClassInfo(median) {
    if (median === null) return {name: 'No R', color: '#555'};
    for (const c of CLASS_BINS) {
        if (median >= c.lo && median < c.hi) return c;
    }
    return CLASS_BINS[CLASS_BINS.length - 1];
}

function renderStats() {
    const withR = DATA.filter(d => d.has_r && !d.deleted);
    const noR = DATA.filter(d => !d.has_r && !d.deleted);
    const deleted = DATA.filter(d => d.deleted);
    const medians = withR.map(d => d.median);
    const s = document.getElementById('stats');
    s.innerHTML = `
        <div class="stat-card"><div class="val">${DATA.length}</div><div class="lbl">Total Boards</div></div>
        <div class="stat-card"><div class="val">${withR.length}</div><div class="lbl">With Resistors</div></div>
        <div class="stat-card warn"><div class="val">${noR.length}</div><div class="lbl">No Resistors</div></div>
        <div class="stat-card"><div class="val">${deleted.length}</div><div class="lbl">🗑️ Deleted</div></div>
        <div class="stat-card"><div class="val">${medians.length ? Math.min(...medians).toFixed(1) : '-'}</div><div class="lbl">Min Median</div></div>
        <div class="stat-card"><div class="val">${medians.length ? (medians.reduce((a,b)=>a+b)/medians.length).toFixed(1) : '-'}</div><div class="lbl">Mean Median</div></div>
        <div class="stat-card"><div class="val">${medians.length ? Math.max(...medians).toFixed(1) : '-'}</div><div class="lbl">Max Median</div></div>
    `;
    // Class summary
    const counts = {};
    CLASS_BINS.forEach(c => counts[c.name] = 0);
    withR.forEach(d => { const ci = getClassInfo(d.median); counts[ci.name] = (counts[ci.name]||0) + 1; });
    document.getElementById('class-summary').innerHTML =
        CLASS_BINS.map(c => `<span style="color:${c.color}">${c.name}:${counts[c.name]||0}</span>`).join(' | ');
}

function renderHistogram() {
    const medians = DATA.filter(d => d.has_r).map(d => d.median);
    if (!medians.length) return;
    const binW = 2;
    const maxVal = Math.ceil(Math.max(...medians));
    const bins = [];
    for (let i = 0; i <= maxVal; i += binW) bins.push(i);
    const counts = new Array(bins.length - 1).fill(0);
    for (const m of medians) {
        for (let i = 0; i < bins.length - 1; i++) {
            if (m >= bins[i] && m < bins[i+1]) { counts[i]++; break; }
            if (i === bins.length - 2) counts[i]++;
        }
    }
    const maxCount = Math.max(...counts);
    const h = document.getElementById('histogram');
    h.innerHTML = '';
    for (let i = 0; i < counts.length; i++) {
        const bar = document.createElement('div');
        const midVal = (bins[i] + bins[i+1]) / 2;
        const ci = getClassInfo(midVal);
        bar.className = 'hist-bar' + (selectedBin === i ? ' selected' : '');
        const pct = maxCount > 0 ? (counts[i] / maxCount * 100) : 0;
        bar.style.height = Math.max(pct, 1) + '%';
        bar.style.background = ci.color + (selectedBin === i ? '' : '88');
        bar.innerHTML = `<span class="count">${counts[i] || ''}</span><span class="label">${bins[i]}</span>`;
        bar.onclick = () => {
            selectedBin = selectedBin === i ? null : i;
            window._binRange = selectedBin !== null ? [bins[i], bins[i+1]] : null;
            renderHistogram();
            page = 0;
            updateGallery();
        };
        h.appendChild(bar);
    }
}

function getFiltered() {
    let items = [...DATA];
    const f = document.getElementById('filter-select').value;
    if (f === 'has_r') items = items.filter(d => d.has_r);
    else if (f === 'no_r') items = items.filter(d => !d.has_r);
    else if (f === 'cap_anchor') items = items.filter(d => d.anchor === 'Capacitor');
    else if (f === 'deleted') items = items.filter(d => d.deleted);
    else if (f === 'not_deleted') items = items.filter(d => !d.deleted);
    else if (f === 'remove') items = items.filter(d => d.has_r && d.median < 6);
    else if (f.startsWith('R')) {
        const cls = CLASS_BINS.find(c => c.name === f);
        if (cls) items = items.filter(d => d.has_r && d.median >= cls.lo && d.median < cls.hi);
    }
    const search = document.getElementById('search').value.toLowerCase();
    if (search) items = items.filter(d => d.board.toLowerCase().includes(search));
    if (window._binRange) {
        const [lo, hi] = window._binRange;
        items = items.filter(d => d.has_r && d.median >= lo && d.median < hi);
    }
    const sort = document.getElementById('sort-select').value;
    if (sort === 'asc') items.sort((a,b) => (a.median||0) - (b.median||0));
    else if (sort === 'desc') items.sort((a,b) => (b.median||999) - (a.median||999));
    else items.sort((a,b) => a.board.localeCompare(b.board));
    return items;
}

function toggleDelete(board) {
    const d = DATA.find(x => x.board === board);
    const action = d.deleted ? 'undelete' : 'delete';
    fetch('/api/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({board, action})
    }).then(r => r.json()).then(res => {
        if (res.ok) {
            d.deleted = (action === 'delete');
            renderStats();
            updateGallery();
        } else {
            alert('Error: ' + res.error);
        }
    });
}

function updateGallery() {
    const items = getFiltered();
    const totalPages = Math.ceil(items.length / PAGE_SIZE);
    if (page >= totalPages) page = Math.max(0, totalPages - 1);
    const start = page * PAGE_SIZE;
    const slice = items.slice(start, start + PAGE_SIZE);
    const g = document.getElementById('gallery');
    g.innerHTML = '';
    for (const d of slice) {
        const ci = getClassInfo(d.median);
        const card = document.createElement('div');
        card.className = 'card' + (d.deleted ? ' is-deleted' : '');
        const catStr = Object.entries(d.cats).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${k}:${v}`).join(', ');
        const anchorTag = d.anchor === 'Capacitor' ? ' <span style="color:#f5a623;font-size:10px">⚡Cap anchor</span>' : '';
        const medStr = d.median !== null
            ? `Median: <b style="color:${ci.color}">${d.median}px</b> | Range: ${d.min}-${d.max}px | ${d.anchor?d.anchor[0]:'?'}: ${d.n_r}${anchorTag}`
            : '<span class="no-r-badge">No R/C Anchor</span>';
        const delLabel = d.deleted ? '↩ Undo' : '🗑️ Delete';
        const delClass = d.deleted ? 'del-btn undo' : 'del-btn';
        card.innerHTML = `
            <button class="${delClass}" onclick="event.stopPropagation();toggleDelete('${d.board}')">${delLabel}</button>
            <img src="/img/${d.board}" loading="lazy" onerror="this.style.background='#222';this.alt='No image'">
            <div class="info">
                <div class="board-name">${d.board}</div>
                <div class="metrics">${medStr} | Total: ${d.n_total}
                    ${d.median !== null ? `<span class="res-tag" style="background:${ci.color}22;color:${ci.color}">${ci.name}</span>` : ''}
                </div>
                <div class="cat-list">${catStr}</div>
            </div>
        `;
        g.appendChild(card);
    }
    const p = document.getElementById('pagination');
    p.innerHTML = `
        <button onclick="page=0;updateGallery()" ${page===0?'disabled':''}>⏮</button>
        <button onclick="page--;updateGallery()" ${page===0?'disabled':''}>◀</button>
        <span class="page-info">Page ${page+1}/${totalPages || 1} (${items.length} boards)</span>
        <button onclick="page++;updateGallery()" ${page>=totalPages-1?'disabled':''}>▶</button>
        <button onclick="page=${totalPages-1};updateGallery()" ${page>=totalPages-1?'disabled':''}>⏭</button>
    `;
    document.getElementById('gallery-title').textContent =
        `🖼️ Board Gallery (${items.length} boards)`;
}

renderStats();
renderHistogram();
updateGallery();
</script>
</body>
</html>"""
    return html


def run_server(port):
    with_r, no_r = analyze()
    deleted = load_deleted()
    print(f"Analyzed {len(with_r)} boards with resistors, {len(no_r)} without")
    print(f"Previously deleted: {len(deleted)}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(ANNO_DIR), "resolution_analysis.csv")
    with open(csv_path, "w") as f:
        f.write("board,median_rc,mean_rc,min_rc,max_rc,n_rc,n_total\n")
        for r in with_r:
            f.write(f"{r['board']},{r['median_r']},{r.get('mean_r','')},{r.get('min_r','')},{r.get('max_r','')},{r['n_r']},{r['n_total']}\n")
    print(f"Saved CSV: {csv_path}")

    html_content = build_html(with_r, no_r, deleted)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html_content.encode())
            elif self.path.startswith("/img/"):
                board = urllib.parse.unquote(self.path[5:])
                img_path = find_image(board)
                if img_path:
                    self.send_response(200)
                    ext = os.path.splitext(img_path)[1].lower().strip(".")
                    ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
                    self.send_header("Content-Type", ct)
                    self.end_headers()
                    with open(img_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/delete":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                board = body.get("board", "")
                action = body.get("action", "delete")
                nonlocal deleted, html_content
                if action == "delete":
                    deleted.add(board)
                else:
                    deleted.discard(board)
                save_deleted(deleted)
                # Rebuild HTML with updated deleted set
                html_content = build_html(with_r, no_r, deleted)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "deleted_count": len(deleted)}).encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server running on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument("--anno_dir", type=str, default="/home/xinrui/projects/PCB_structure_layout/re-annotation/final_annotations")
    parser.add_argument("--image_dir", type=str, default="/home/xinrui/projects/data/ti_pcb/images_top")
    args = parser.parse_args()
    import sys
    m = sys.modules[__name__]
    m.ANNO_DIR = args.anno_dir
    m.IMG_DIR = args.image_dir
    m.DELETED_FILE = os.path.join(os.path.dirname(args.anno_dir), "metadata", "resolution_deleted_boards.json")
    run_server(args.port)
