#!/usr/bin/env python3
"""
Orientation Review Server — browse inferred orientation results on final annotations.
Supports click-to-select + hotkey relabeling (WASD/QEC/Z).
Run: python orientation_review_server.py [--port 8892] [--anno_dir ...] [--image_dir ...]
"""

import argparse, json, io, math
from collections import defaultdict, Counter
from pathlib import Path
from flask import Flask, jsonify, request, send_file, Response
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8892)
parser.add_argument("--anno_dir", type=str, default="/home/xinrui/projects/PCB_structure_layout/re-annotation/final_annotations")
parser.add_argument("--image_dir", type=str, default="/home/xinrui/projects/data/ti_pcb/COCO_label/images/train")
args = parser.parse_args()

FINAL_DIR = Path(args.anno_dir)
IMG_DIR = Path(args.image_dir)

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
ANGLE_ARROWS = {0: "→", 45: "↗", 90: "↑", 135: "↖", 180: "←", 225: "↙", 270: "↓", 315: "↘"}
CAT_NAMES = {1:"Resistor", 2:"Capacitor", 3:"Inductor", 4:"Connector",
             5:"Diode", 7:"Switch", 8:"Transistor", 9:"IC", 10:"Oscillator"}

# Load all annotations
print("Loading annotations...")
boards = {}
all_anns = []
angle_dist = Counter()
cat_dist = Counter()

for f in sorted(FINAL_DIR.glob("*.json")):
    data = json.loads(f.read_text())
    if isinstance(data, list): continue
    board = f.stem
    boards[board] = data
    for ann in data.get("annotations", []):
        ori = ann.get("orientation")
        cat = ann.get("category_id")
        if ori is not None:
            all_anns.append({"board": board, "ann": ann})
            angle_dist[ori] += 1
            cat_dist[cat] += 1

print(f"Loaded {len(boards)} boards, {len(all_anns)} annotations with orientation")
print(f"Angle distribution: {dict(sorted(angle_dist.items()))}")


def rebuild_indices():
    """Rebuild angle/category indices after relabeling."""
    global by_angle, by_angle_cat, angle_dist, cat_dist
    by_angle = defaultdict(list)
    by_angle_cat = defaultdict(list)
    angle_dist = Counter()
    cat_dist = Counter()
    for item in all_anns:
        a = item["ann"].get("orientation", 0)
        c = item["ann"].get("category_id", 0)
        by_angle[a].append(item)
        by_angle_cat[(a, c)].append(item)
        angle_dist[a] += 1
        cat_dist[c] += 1


rebuild_indices()

app = Flask(__name__)
img_cache = {}

def get_image(board):
    if board not in img_cache:
        for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
            p = IMG_DIR / f"{board}{ext}"
            if p.exists():
                img_cache[board] = Image.open(p).convert("RGB")
                break
        if len(img_cache) > 200:
            oldest = next(iter(img_cache))
            del img_cache[oldest]
    return img_cache.get(board)

@app.route("/api/crop/<board>/<int:ann_id>")
def api_crop(board, ann_id):
    data = boards.get(board)
    if not data:
        return "Board not found", 404
    ann = None
    for a in data.get("annotations", []):
        if a["id"] == ann_id:
            ann = a
            break
    if not ann:
        return "Annotation not found", 404
    img = get_image(board)
    if not img:
        return "Image not found", 404
    bx, by, bw, bh = ann["bbox"]
    x1, y1 = max(0, int(bx)), max(0, int(by))
    x2, y2 = min(img.width, int(bx+bw)), min(img.height, int(by+bh))
    if x2 <= x1 or y2 <= y1:
        return "Invalid bbox", 400
    crop = img.crop((x1, y1, x2, y2))
    # Arrow removed — orientation shown in card label
    
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total": len(all_anns),
        "boards": len(boards),
        "angles": {str(a): angle_dist.get(a, 0) for a in ANGLES},
        "categories": {CAT_NAMES.get(c, f"Cat{c}"): cnt for c, cnt in sorted(cat_dist.items())},
    })

@app.route("/api/browse")
def api_browse():
    angle = int(request.args.get("angle", 0))
    cat = request.args.get("category", "")
    page = int(request.args.get("page", 0))
    per_page = int(request.args.get("per_page", 100))
    
    if cat:
        cat_id = None
        for k, v in CAT_NAMES.items():
            if v.lower() == cat.lower():
                cat_id = k
                break
        items = by_angle_cat.get((angle, cat_id), [])
    else:
        items = by_angle.get(angle, [])
    
    start = page * per_page
    end = start + per_page
    page_items = items[start:end]
    
    return jsonify({
        "angle": angle,
        "category": cat,
        "page": page,
        "per_page": per_page,
        "total": len(items),
        "total_pages": max(1, (len(items) + per_page - 1) // per_page),
        "items": [{
            "board": it["board"],
            "ann_id": it["ann"]["id"],
            "category": CAT_NAMES.get(it["ann"]["category_id"], "?"),
            "bbox": it["ann"]["bbox"],
            "orientation": it["ann"].get("orientation"),
        } for it in page_items],
    })

@app.route("/api/relabel", methods=["POST"])
def api_relabel():
    """Relabel orientation for selected annotations and save to disk."""
    body = request.get_json()
    items = body.get("items", [])  # [{board, ann_id}, ...]
    new_angle = int(body.get("angle", 0))
    
    if new_angle not in ANGLES:
        return jsonify({"ok": False, "error": f"Invalid angle {new_angle}"}), 400
    
    changed = 0
    changed_boards = set()
    for item in items:
        board = item["board"]
        ann_id = item["ann_id"]
        data = boards.get(board)
        if not data:
            continue
        for ann in data.get("annotations", []):
            if ann["id"] == ann_id:
                old_ori = ann.get("orientation")
                if old_ori != new_angle:
                    ann["orientation"] = new_angle
                    changed += 1
                    changed_boards.add(board)
                break
    
    # Save changed boards to disk
    for board in changed_boards:
        out_path = FINAL_DIR / f"{board}.json"
        out_path.write_text(json.dumps(boards[board], indent=2))
    
    # Rebuild indices
    rebuild_indices()
    
    print(f"Relabeled {changed} annotations to {new_angle}° across {len(changed_boards)} boards")
    return jsonify({"ok": True, "changed": changed, "boards": len(changed_boards)})

@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Delete selected annotations from boards and save to disk."""
    body = request.get_json()
    items = body.get("items", [])  # [{board, ann_id}, ...]
    
    deleted = 0
    changed_boards = set()
    for item in items:
        board = item["board"]
        ann_id = item["ann_id"]
        data = boards.get(board)
        if not data:
            continue
        anns = data.get("annotations", [])
        before = len(anns)
        data["annotations"] = [a for a in anns if a["id"] != ann_id]
        if len(data["annotations"]) < before:
            deleted += 1
            changed_boards.add(board)
    
    # Remove from all_anns list
    del_set = {(it["board"], it["ann_id"]) for it in items}
    all_anns[:] = [a for a in all_anns if (a["board"], a["ann"]["id"]) not in del_set]
    
    # Save changed boards to disk
    for board in changed_boards:
        out_path = FINAL_DIR / f"{board}.json"
        out_path.write_text(json.dumps(boards[board], indent=2))
    
    # Rebuild indices
    rebuild_indices()
    
    print(f"Deleted {deleted} annotations across {len(changed_boards)} boards")
    return jsonify({"ok": True, "deleted": deleted, "boards": len(changed_boards)})

@app.route("/")
def index():
    return HTML_PAGE

HTML_PAGE = """<!DOCTYPE html>
<html><head><title>Orientation Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui;background:#1a1a2e;color:#eee;display:flex;flex-direction:column;height:100vh}
.toolbar{display:flex;gap:8px;padding:8px 12px;background:#16213e;align-items:center;flex-wrap:wrap}
.toolbar select,.toolbar button{padding:4px 10px;border-radius:4px;border:1px solid #555;background:#0f3460;color:#eee;cursor:pointer;font-size:13px}
.toolbar button:hover{background:#e94560}
.toolbar .info{color:#aaa;font-size:12px}
.toolbar .sel-count{color:#e94560;font-weight:bold;font-size:13px}
.grid{display:flex;flex-wrap:wrap;gap:4px;padding:8px;overflow-y:auto;flex:1}
.card{width:90px;height:90px;border:2px solid #333;border-radius:4px;overflow:hidden;cursor:pointer;position:relative;transition:border-color 0.1s}
.card img{width:100%;height:100%;object-fit:contain;background:#222}
.card .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.7);color:#fff;font-size:9px;text-align:center;padding:1px}
.card:hover{border-color:#e94560}
.card.selected{border-color:#e94560;box-shadow:0 0 6px #e94560}
.pager{display:flex;gap:8px;padding:8px 12px;background:#16213e;align-items:center;justify-content:center}
.pager button{padding:4px 12px;border-radius:4px;border:1px solid #555;background:#0f3460;color:#eee;cursor:pointer}
.stats{padding:6px 12px;background:#0f3460;font-size:12px;display:flex;gap:16px;flex-wrap:wrap}
.stats span{color:#aaa}
.hotkeys{padding:4px 12px;background:#0a0a23;font-size:11px;color:#666;display:flex;gap:12px;flex-wrap:wrap}
.hotkeys b{color:#aaa}
</style></head><body>
<div class="stats" id="stats"></div>
<div class="toolbar">
  <label>Angle: <select id="angle-sel" onchange="curPage=0;browse()"></select></label>
  <label>Category: <select id="cat-sel" onchange="curPage=0;browse()">
    <option value="">All</option>
    <option>Resistor</option><option>Capacitor</option><option>Inductor</option>
    <option>Connector</option><option>Diode</option><option>Switch</option>
    <option>Transistor</option><option>IC</option><option>Oscillator</option>
  </select></label>
  <button onclick="browse()">Refresh</button>
  <button onclick="deleteSelected()" style="background:#e94560">🗑️ Delete</button>
  <span class="sel-count" id="sel-count"></span>
  <div class="info" id="page-info"></div>
</div>
<div class="hotkeys">
  <span>Select: <b>Click</b> card, <b>Shift+Click</b> range, <b>A</b> select all, <b>Esc</b> clear</span>
  <span>|</span>
  <span>Relabel: <b>Q</b>=↖135° <b>W</b>=↑90° <b>E</b>=↗45° <b>A</b>=←180° <b>D</b>=→0° <b>Z</b>=↙225° <b>S</b>=↓270° <b>C</b>=↘315°</span>
  <span>|</span>
  <span>Delete: <b>X</b> or <b>Del</b></span>
  <span>|</span>
  <span>Nav: <b>←→</b> pages</span>
</div>
<div class="grid" id="grid"></div>
<div class="pager">
  <button onclick="prevPage()">← Prev</button>
  <span id="pager-text">Page 1</span>
  <button onclick="nextPage()">Next →</button>
</div>
<script>
let curPage=0, totalPages=1;
let currentItems=[];
let selected=new Set(); // indices into currentItems
let lastClickIdx=-1;
const arrows={0:"→",45:"↗",90:"↑",135:"↖",180:"←",225:"↙",270:"↓",315:"↘"};
const HOTKEYS={d:0,e:45,w:90,q:135,a:180,z:225,s:270,c:315};
// Also numeric: 1-8
const NUM_KEYS={'1':135,'2':90,'3':45,'4':180,'5':null,'6':0,'7':225,'8':270,'9':315};

async function loadStats(){
  const r=await(await fetch("/api/stats")).json();
  const el=document.getElementById("stats");
  let h=`<span>Total: ${r.total} annotations, ${r.boards} boards</span>`;
  h+=`<span>|</span>`;
  for(const[a,c]of Object.entries(r.angles)){
    h+=`<span>${arrows[a]||a}° ${a}: ${c}</span>`;
  }
  el.innerHTML=h;
  const sel=document.getElementById("angle-sel");
  sel.innerHTML="";
  for(const a of [0,45,90,135,180,225,270,315]){
    const opt=document.createElement("option");
    opt.value=a;
    opt.textContent=`${arrows[a]} ${a}° (${r.angles[a]||0})`;
    sel.appendChild(opt);
  }
  const prevAngle = window._curAngle || "0";
  sel.value = prevAngle;
}

async function browse(page){
  if(page!==undefined)curPage=page;
  const angle=document.getElementById("angle-sel").value;
  window._curAngle = angle;
  const cat=document.getElementById("cat-sel").value;
  const r=await(await fetch(`/api/browse?angle=${angle}&category=${cat}&page=${curPage}&per_page=100`)).json();
  totalPages=r.total_pages;
  currentItems=r.items;
  if(curPage>=totalPages) curPage=Math.max(0,totalPages-1);
  selected.clear();
  lastClickIdx=-1;
  updateSelCount();
  document.getElementById("page-info").textContent=`${r.total} items`;
  document.getElementById("pager-text").textContent=`Page ${curPage+1}/${totalPages}`;
  renderGrid();
}

function renderGrid(){
  const grid=document.getElementById("grid");
  grid.innerHTML="";
  currentItems.forEach((it,idx)=>{
    const d=document.createElement("div");
    d.className="card"+(selected.has(idx)?" selected":"");
    d.dataset.idx=idx;
    d.innerHTML=`<img src="/api/crop/${it.board}/${it.ann_id}" loading="lazy">
      <div class="label">${it.category} ${arrows[it.orientation]||""}</div>`;
    d.title=`${it.board} #${it.ann_id} ${it.category} ${it.orientation}°\nbbox: ${it.bbox.map(v=>Math.round(v)).join(",")}`;
    d.onclick=(e)=>toggleSelect(idx, e.shiftKey);
    grid.appendChild(d);
  });
}

function toggleSelect(idx, shift){
  if(shift && lastClickIdx>=0){
    const lo=Math.min(lastClickIdx,idx), hi=Math.max(lastClickIdx,idx);
    for(let i=lo;i<=hi;i++) selected.add(i);
  } else {
    if(selected.has(idx)) selected.delete(idx);
    else selected.add(idx);
  }
  lastClickIdx=idx;
  refreshSelection();
}

function selectAll(){
  for(let i=0;i<currentItems.length;i++) selected.add(i);
  refreshSelection();
}

function clearSelection(){
  selected.clear();
  lastClickIdx=-1;
  refreshSelection();
}

function refreshSelection(){
  document.querySelectorAll(".card").forEach(c=>{
    const idx=parseInt(c.dataset.idx);
    c.classList.toggle("selected", selected.has(idx));
  });
  updateSelCount();
}

function updateSelCount(){
  document.getElementById("sel-count").textContent=selected.size?selected.size+" selected":"";
}

async function relabel(newAngle){
  if(selected.size===0) return;
  const items=[];
  selected.forEach(idx=>{
    const it=currentItems[idx];
    items.push({board:it.board, ann_id:it.ann_id});
  });
  const r=await fetch("/api/relabel",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({items, angle:newAngle})
  });
  const res=await r.json();
  if(res.ok){
    // Update local data
    selected.forEach(idx=>{
      currentItems[idx].orientation=newAngle;
    });
    selected.clear();
    // Refresh stats + grid (images will re-render with new arrow)
    await loadStats();
    // Re-browse to get updated indices
    await browse(curPage);
  }
}

async function deleteSelected(){
  if(selected.size===0) return;
  if(!confirm(`Delete ${selected.size} annotation(s)?`)) return;
  const items=[];
  selected.forEach(idx=>{
    const it=currentItems[idx];
    items.push({board:it.board, ann_id:it.ann_id});
  });
  const r=await fetch("/api/delete",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({items})
  });
  const res=await r.json();
  if(res.ok){
    selected.clear();
    await loadStats();
    await browse(curPage);
  }
}

function prevPage(){if(curPage>0){curPage--;browse();}}
function nextPage(){if(curPage<totalPages-1){curPage++;browse();}}

document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
  
  // Delete
  if((e.key==="x"||e.key==="X"||e.key==="Delete")&&selected.size>0){e.preventDefault();deleteSelected();return;}
  
  // Navigation
  if(e.key==="ArrowLeft"){prevPage();return;}
  if(e.key==="ArrowRight"){nextPage();return;}
  
  // Select all (only when nothing selected, otherwise 'a' = relabel 180°)
  if(e.key==="a"&&selected.size===0){e.preventDefault();selectAll();return;}
  if(e.key==="Escape"){clearSelection();return;}
  
  // WASD/QEC/Z relabeling
  const angle=HOTKEYS[e.key.toLowerCase()];
  if(angle!==undefined&&selected.size>0){
    e.preventDefault();
    relabel(angle);
    return;
  }
  
  // Numeric 1-9
  const numAngle=NUM_KEYS[e.key];
  if(numAngle!==undefined&&numAngle!==null&&selected.size>0){
    e.preventDefault();
    relabel(numAngle);
  }
});

loadStats().then(()=>browse(0));
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=True)
