#!/usr/bin/env python3
"""
Color analysis web UI with two pages:
  / — dashboard (stats + histogram)
  /gallery — dedicated 3x5 labeling view

Run on L40: python color_analysis.py --port 8896
"""
import json, os, argparse, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

DATA_FILE = "/home/xinrui/projects/PCB_structure_layout/re-annotation/board_colors_v2.json"
IMG_DIR = "/home/xinrui/projects/data/ti_pcb/images_top"


def classify_rgb(r, g, b):
    if r < 30 and g < 30 and b < 30: return "black"
    if r > 180 and g > 180 and b > 180: return "white"
    if r >= g and r >= b: return "red"
    if g >= r and g >= b: return "green"
    return "blue"


def find_image(board):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(IMG_DIR, board + ext)
        if os.path.exists(p):
            return p
    return None


def build_dashboard(data):
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB Color Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }
h1 { text-align: center; padding: 20px; color: #00d4ff; font-size: 1.5em; }
.container { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
.stats { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-card { background: #16213e; border-radius: 8px; padding: 15px; flex: 1; min-width: 100px; text-align: center; cursor: pointer; }
.stat-card:hover { outline: 2px solid #00d4ff; }
.stat-card .val { font-size: 1.6em; font-weight: bold; }
.stat-card .lbl { font-size: 0.8em; color: #888; margin-top: 5px; }
.section { background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.1em; }
.hist-container { position: relative; height: 250px; display: flex; align-items: flex-end; gap: 1px; padding-bottom: 30px; }
.hist-bar { flex: 1; border-radius: 3px 3px 0 0; position: relative; min-width: 4px; }
.hist-bar .count { position: absolute; top: -18px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #aaa; white-space: nowrap; }
.hist-bar .label { position: absolute; bottom: -22px; left: 50%; transform: translateX(-50%); font-size: 9px; color: #888; white-space: nowrap; }
.tabs { display: flex; gap: 5px; margin-bottom: 15px; }
.tabs button { background: #0f3460; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
.tabs button.active { background: #00d4ff; color: #000; }
.tabs button:hover { background: #00d4ff88; }
.gallery-link { display: inline-block; background: #e94560; color: #fff; padding: 12px 24px; border-radius: 8px; font-size: 1.1em; text-decoration: none; margin: 20px auto; }
.gallery-link:hover { background: #ff1744; }
.center { text-align: center; }
</style>
</head>
<body>
<h1>🎨 PCB Board Color Dashboard</h1>
<div class="container">
    <div class="stats" id="stats"></div>
    <div class="center"><a class="gallery-link" href="/gallery">🖼️ Open Gallery Labeler (3×5)</a></div>
    <div class="section">
        <h2 id="hist-title">📊 RGB Distribution</h2>
        <div class="tabs" id="tabs"></div>
        <div class="hist-container" id="histogram"></div>
    </div>
</div>
<script>
const DATA = """ + json.dumps(data) + """;
const COLOR_MAP = {green:'#4caf50',blue:'#2196f3',red:'#f44336',orange:'#ff9800',yellow:'#ffeb3b',white:'#e0e0e0',black:'#333',gray:'#888',purple:'#9c27b0',other:'#666'};
let mode = 'r';
let activeColor = null;

function renderStats() {
    const counts = {};
    DATA.forEach(d => { counts[d.color] = (counts[d.color]||0) + 1; });
    const sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]);
    const el = document.getElementById('stats');
    el.innerHTML = `<div class="stat-card" onclick="activeColor=null;renderStats();renderHist()"><div class="val" style="color:#00d4ff">${DATA.length}</div><div class="lbl">All</div></div>`;
    sorted.forEach(([c, n]) => {
        el.innerHTML += `<div class="stat-card" onclick="activeColor='${c}';renderStats();renderHist()"><div class="val" style="color:${COLOR_MAP[c]||'#fff'}">${n}</div><div class="lbl">${c}</div></div>`;
    });
}
function renderTabs() {
    const t = document.getElementById('tabs');
    t.innerHTML = '';
    ['r','g','b'].forEach(m => {
        const labels = {r:'Red', g:'Green', b:'Blue'};
        const btn = document.createElement('button');
        btn.textContent = labels[m]; btn.className = mode===m?'active':'';
        btn.onclick = () => { mode=m; renderTabs(); renderHist(); };
        t.appendChild(btn);
    });
}
function renderHist() {
    let items = activeColor ? DATA.filter(d=>d.color===activeColor) : DATA;
    const label = activeColor || 'All';
    document.getElementById('hist-title').textContent = '📊 ' + label.charAt(0).toUpperCase()+label.slice(1) + ' — ' + {r:'Red',g:'Green',b:'Blue'}[mode] + ' Channel';
    const vals = items.map(d=>d[mode]);
    const binW=5, lo=0, hi=260;
    const bins=[]; for(let i=lo;i<hi;i+=binW) bins.push(i);
    const counts=new Array(bins.length).fill(0);
    vals.forEach(v=>{for(let i=0;i<bins.length;i++){if(v>=bins[i]&&v<bins[i]+binW){counts[i]++;break;}if(i===bins.length-1)counts[i]++;}});
    const maxC=Math.max(...counts,1);
    const h=document.getElementById('histogram'); h.innerHTML='';
    bins.forEach((b,i)=>{
        const bar=document.createElement('div'); bar.className='hist-bar';
        bar.style.height=Math.max(counts[i]/maxC*100,1)+'%';
        const c = mode==='r'?`rgb(${b},60,60)`:mode==='g'?`rgb(60,${b},60)`:`rgb(60,60,${b})`;
        bar.style.background=c;
        bar.innerHTML=`<span class="count">${counts[i]||''}</span><span class="label">${b}</span>`;
        h.appendChild(bar);
    });
}
renderStats(); renderTabs(); renderHist();
</script>
</body></html>"""


def build_gallery(data):
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB Color Gallery Labeler</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }

.topbar { background: #16213e; padding: 10px 20px; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; position: sticky; top: 0; z-index: 100; }
.topbar h1 { color: #00d4ff; font-size: 1.2em; margin-right: 10px; }
.topbar select, .topbar input { background: #0a0a23; border: 1px solid #333; color: #fff; padding: 5px 8px; border-radius: 4px; font-size: 13px; }
.topbar button { background: #0f3460; color: #fff; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.topbar button:hover { background: #00d4ff; color: #000; }
.topbar .info { color: #aaa; font-size: 13px; }
.topbar .hotkeys { color: #666; font-size: 11px; }
.topbar .hotkeys b { color: #aaa; }
.topbar a { color: #00d4ff; text-decoration: none; }

.grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; padding: 15px; max-width: 1600px; margin: 0 auto; }
.card { background: #0a0a23; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.15s; }
.card:hover { transform: scale(1.01); }
.card.selected { outline: 3px solid #e94560; }
.card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; }
.card .info { padding: 6px 10px; font-size: 12px; display: flex; justify-content: space-between; align-items: center; }
.card .board-name { color: #00d4ff; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%; }
.card .color-label { padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 11px; }
.card .rgb { color: #666; font-size: 10px; }
.cluster-dot:hover { transform: scale(1.3); border-color: #e94560 !important; }

.pagination { display: flex; justify-content: center; align-items: center; gap: 10px; padding: 15px; position: sticky; bottom: 0; background: #1a1a2e; border-top: 1px solid #333; }
.pagination button { background: #0f3460; color: #fff; border: none; padding: 8px 18px; border-radius: 4px; cursor: pointer; font-size: 14px; }
.pagination button:hover { background: #00d4ff; color: #000; }
.pagination button:disabled { opacity: 0.3; cursor: default; }
.pagination .page-info { color: #aaa; font-size: 14px; }
.pagination input { width: 60px; text-align: center; }
</style>
</head>
<body>
<div class="topbar">
    <h1>🎨 Color Labeler</h1>
    <a href="/">← Dashboard</a>
    <select id="filter" onchange="page=0;selectedBoards.clear();updateGallery()">
        <option value="all">All</option>
    </select>
    <select id="sort" onchange="page=0;updateGallery()">
        <option value="name">A-Z</option>
        <option value="r">Sort R</option>
        <option value="g">Sort G</option>
        <option value="b">Sort B</option>
    </select>
    <input type="text" id="search" placeholder="Search..." oninput="page=0;updateGallery()">
    <span class="info" id="sel-count"></span>
    <div class="hotkeys">
        <b>G</b>=green <b>B</b>=blue <b>R</b>=red <b>K</b>=black <b>W</b>=white <b>O</b>=orange | Click select, Shift+click range
    </div>
</div>
<div class="grid" id="grid"></div>
<div class="pagination" id="pagination"></div>

<script>
const DATA = """ + json.dumps(data) + """;
const COLOR_MAP = {green:'#4caf50',blue:'#2196f3',red:'#f44336',orange:'#ff9800',white:'#e0e0e0',black:'#555',gray:'#888'};
const PAGE_SIZE = 15;
let page = 0;
let selectedBoards = new Set();
let lastClickIdx = -1;
let currentSlice = [];

// Build filter options
(function(){
    const counts = {};
    DATA.forEach(d => counts[d.color] = (counts[d.color]||0)+1);
    const sel = document.getElementById('filter');
    Object.entries(counts).sort((a,b)=>b[1]-a[1]).forEach(([c,n]) => {
        sel.innerHTML += `<option value="${c}">${c} (${n})</option>`;
    });
})();

function getFiltered() {
    let items = [...DATA];
    const f = document.getElementById('filter').value;
    if (f !== 'all') items = items.filter(d => d.color === f);
    const s = document.getElementById('search').value.toLowerCase();
    if (s) items = items.filter(d => d.board.toLowerCase().includes(s));
    const sort = document.getElementById('sort').value;
    if (sort === 'name') items.sort((a,b) => a.board.localeCompare(b.board));
    else items.sort((a,b) => a[sort] - b[sort]);
    return items;
}

function toggleSelect(board, idx, shiftKey) {
    if (shiftKey && lastClickIdx >= 0) {
        const lo = Math.min(lastClickIdx, idx), hi = Math.max(lastClickIdx, idx);
        for (let i = lo; i <= hi; i++) if (currentSlice[i]) selectedBoards.add(currentSlice[i].board);
    } else {
        if (selectedBoards.has(board)) selectedBoards.delete(board);
        else selectedBoards.add(board);
    }
    lastClickIdx = idx;
    refreshCards();
}

function refreshCards() {
    document.querySelectorAll('.card').forEach(c => {
        c.classList.toggle('selected', selectedBoards.has(c.dataset.board));
    });
    document.getElementById('sel-count').textContent = selectedBoards.size ? selectedBoards.size + ' selected' : '';
}

function relabel(newColor) {
    if (!selectedBoards.size) return;
    const boards = [...selectedBoards];
    fetch('/api/relabel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({boards, color: newColor})
    }).then(r => r.json()).then(res => {
        if (res.ok) {
            boards.forEach(b => { const d = DATA.find(x => x.board === b); if (d) d.color = newColor; });
            selectedBoards.clear();
            // Rebuild filter counts
            const counts = {};
            DATA.forEach(d => counts[d.color] = (counts[d.color]||0)+1);
            const sel = document.getElementById('filter');
            const prev = sel.value;
            sel.innerHTML = '<option value="all">All</option>';
            Object.entries(counts).sort((a,b)=>b[1]-a[1]).forEach(([c,n]) => {
                sel.innerHTML += `<option value="${c}">${c} (${n})</option>`;
            });
            sel.value = prev;
            updateGallery();
        }
    });
}

function pickCluster(board, clusterIdx) {
    console.log('pickCluster', board, clusterIdx);
    fetch('/api/pick_cluster', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({board, cluster_idx: clusterIdx})
    }).then(r => r.json()).then(res => {
        if (res.ok) {
            const d = DATA.find(x => x.board === board);
            if (d) {
                d.r = res.r; d.g = res.g; d.b = res.b;
                d.color = res.color; d.dom_idx = clusterIdx;
            }
            // Rebuild filter counts
            const counts = {};
            DATA.forEach(d => counts[d.color] = (counts[d.color]||0)+1);
            const sel = document.getElementById('filter');
            const prev = sel.value;
            sel.innerHTML = '<option value="all">All</option>';
            Object.entries(counts).sort((a,b)=>b[1]-a[1]).forEach(([c,n]) => {
                sel.innerHTML += `<option value="${c}">${c} (${n})</option>`;
            });
            sel.value = prev;
            updateGallery();
        }
    });
}

document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    const map = {g:'green',b:'blue',r:'red',k:'black',w:'white',o:'orange'};
    if (map[e.key.toLowerCase()]) { e.preventDefault(); relabel(map[e.key.toLowerCase()]); }
    if (e.key === 'ArrowRight' || e.key === 'd') { e.preventDefault(); nextPage(); }
    if (e.key === 'ArrowLeft' || e.key === 'a') { e.preventDefault(); prevPage(); }
});

function nextPage() { const tp = Math.ceil(getFiltered().length/PAGE_SIZE); if (page < tp-1) { page++; selectedBoards.clear(); updateGallery(); } }
function prevPage() { if (page > 0) { page--; selectedBoards.clear(); updateGallery(); } }

function updateGallery() {
    const items = getFiltered();
    const totalPages = Math.ceil(items.length / PAGE_SIZE) || 1;
    if (page >= totalPages) page = totalPages - 1;
    const start = page * PAGE_SIZE;
    const slice = items.slice(start, start + PAGE_SIZE);
    currentSlice = slice;
    const g = document.getElementById('grid');
    g.innerHTML = '';
    slice.forEach((d, idx) => {
        const card = document.createElement('div');
        card.className = 'card' + (selectedBoards.has(d.board) ? ' selected' : '');
        card.dataset.board = d.board;
        const bg = COLOR_MAP[d.color] || '#666';
        const fg = (d.color==='white'||d.color==='yellow') ? '#000' : '#fff';
        const domDot = `rgb(${d.r.toFixed(0)},${d.g.toFixed(0)},${d.b.toFixed(0)})`;
        let clusterDots = '';
        if (d.clusters) {
            const di = d.dom_idx || 0;
            clusterDots = d.clusters.map((c,ci) => 
                `<span class="cluster-dot" data-board="${d.board}" data-ci="${ci}" style="display:inline-block;width:20px;height:20px;border-radius:50%;background:rgb(${c.r.toFixed(0)},${c.g.toFixed(0)},${c.b.toFixed(0)});border:2px solid ${ci===di?'#fff':'#555'};margin-right:3px;cursor:pointer;z-index:50;position:relative" title="Click to set dominant — RGB(${c.r.toFixed(0)},${c.g.toFixed(0)},${c.b.toFixed(0)}) n=${c.n}"></span>`
            ).join('');
        }
        card.innerHTML = `
            <img src="/img/${d.board}" loading="lazy" onerror="this.style.background='#222'">
            <div class="info">
                <span class="board-name">${d.board}</span>
                <span class="color-label" style="background:${bg};color:${fg}">${d.color}</span>
            </div>
            <div style="padding:2px 10px 6px;display:flex;align-items:center;gap:6px">
                <span style="display:inline-block;width:18px;height:18px;border-radius:3px;background:${domDot};border:2px solid #fff" title="Dominant: RGB(${d.r.toFixed(0)},${d.g.toFixed(0)},${d.b.toFixed(0)})"></span>
                <span class="rgb">R:${d.r.toFixed(0)} G:${d.g.toFixed(0)} B:${d.b.toFixed(0)}</span>
                ${clusterDots ? '<span style="margin-left:4px">'+clusterDots+'</span>' : ''}
            </div>
        `;
        card.onclick = (e) => {
            if (e.target.classList.contains('cluster-dot')) {
                pickCluster(e.target.dataset.board, parseInt(e.target.dataset.ci));
                return;
            }
            toggleSelect(d.board, idx, e.shiftKey);
        };
        g.appendChild(card);
    });
    document.getElementById('sel-count').textContent = selectedBoards.size ? selectedBoards.size + ' selected' : '';
    const p = document.getElementById('pagination');
    p.innerHTML = `
        <button onclick="page=0;selectedBoards.clear();updateGallery()">⏮</button>
        <button onclick="prevPage()">◀ Prev</button>
        <span class="page-info">Page <input type="number" id="page-jump" value="${page+1}" min="1" max="${totalPages}"
            onchange="page=Math.max(0,Math.min(${totalPages-1},parseInt(this.value)-1));selectedBoards.clear();updateGallery()"
            style="background:#0a0a23;border:1px solid #333;color:#fff;border-radius:4px;padding:3px;width:50px;text-align:center">
            / ${totalPages} (${items.length} boards)</span>
        <button onclick="nextPage()">Next ▶</button>
        <button onclick="page=${totalPages-1};selectedBoards.clear();updateGallery()">⏭</button>
    `;
}

updateGallery();
</script>
</body></html>"""


def run_server(port):
    with open(DATA_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} boards from {DATA_FILE}")

    dashboard_html = build_dashboard(data)
    gallery_html = build_gallery(data)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal dashboard_html, gallery_html
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(dashboard_html.encode())
            elif self.path == "/gallery":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(gallery_html.encode())
            elif self.path.startswith("/img/"):
                board = urllib.parse.unquote(self.path[5:])
                img_path = find_image(board)
                if img_path:
                    self.send_response(200)
                    ext = os.path.splitext(img_path)[1].lower().strip(".")
                    ct = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}.get(ext,"image/jpeg")
                    self.send_header("Content-Type", ct)
                    self.end_headers()
                    with open(img_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            nonlocal data, dashboard_html, gallery_html
            if self.path == "/api/pick_cluster":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                board = body.get("board", "")
                ci = body.get("cluster_idx", 0)
                d = next((x for x in data if x["board"] == board), None)
                if d and d.get("clusters") and ci < len(d["clusters"]):
                    cl = d["clusters"][ci]
                    d["r"] = cl["r"]
                    d["g"] = cl["g"]
                    d["b"] = cl["b"]
                    d["dom_idx"] = ci
                    d["color"] = classify_rgb(cl["r"], cl["g"], cl["b"])
                    with open(DATA_FILE, "w") as f:
                        json.dump(data, f)
                    dashboard_html = build_dashboard(data)
                    gallery_html = build_gallery(data)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True, "r": d["r"], "g": d["g"], "b": d["b"], "color": d["color"]}).encode())
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "error": "not found"}).encode())
                return
            elif self.path == "/api/relabel":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                boards = set(body.get("boards", []))
                new_color = body.get("color", "")
                changed = 0
                for d in data:
                    if d["board"] in boards:
                        d["color"] = new_color
                        changed += 1
                with open(DATA_FILE, "w") as f:
                    json.dump(data, f)
                dashboard_html = build_dashboard(data)
                gallery_html = build_gallery(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "changed": changed}).encode())
            else:
                self.send_error(404)

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8896)
    args = parser.parse_args()
    run_server(args.port)
