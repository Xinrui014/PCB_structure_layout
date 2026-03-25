#!/usr/bin/env python3
"""Build test.jsonl for v2 layout dataset from annotation pipeline test output."""
import json
import shutil
from pathlib import Path
from collections import Counter

# === Paths ===
ANNO_DIR = Path("/home/xinrui/projects/PCB_structure_layout/annotation_pipeline/test_output/annotation")
META_DIR = Path("/home/xinrui/projects/PCB_structure_layout/annotation_pipeline/test_output/metadata")
SRC_IMG_DIR = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/images/test")
OUT_DIR = Path("/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh")
OUT_IMG_DIR = OUT_DIR / "image" / "test"
OUT_IMG_EXCL_DIR = OUT_DIR / "image" / "test_exclude"
OUT_JSONL = OUT_DIR / "test.jsonl"

# === Category mapping (same as train) ===
CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC", 10: "OSCILLATOR"
}
CAT_NAME_LOWER = {
    1: "resistors", 2: "capacitors", 3: "inductors", 4: "connectors",
    5: "diodes", 7: "switches", 8: "transistors", 9: "ics", 10: "oscillators"
}

# === Load exclusion lists ===
excluded_boards = set()
excl_file = META_DIR / "excluded_boards.json"
if excl_file.exists():
    excluded_boards.update(json.loads(excl_file.read_text()))
    print(f"Excluded boards (annotator): {len(excluded_boards)}")

res_del_file = META_DIR / "resolution_deleted_boards.json"
if res_del_file.exists():
    res_del = json.loads(res_del_file.read_text())
    if isinstance(res_del, list):
        excluded_boards.update(res_del)
    elif isinstance(res_del, dict):
        excluded_boards.update(k for k, v in res_del.items() if v)
    print(f"Resolution-deleted boards: {len(res_del) if isinstance(res_del, list) else sum(1 for v in res_del.values() if v)}")

print(f"Total excluded boards: {len(excluded_boards)}")

# === Load color labels ===
color_file = META_DIR / "board_colors_v2.json"
color_data = json.loads(color_file.read_text())
board_colors = {d["board"]: d["color"].upper() for d in color_data}
print(f"Color labels: {len(board_colors)} boards")

# === Process boards ===
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG_EXCL_DIR.mkdir(parents=True, exist_ok=True)

anno_files = sorted(ANNO_DIR.glob("*.json"))
print(f"\nAnnotation files: {len(anno_files)}")

samples = []
excluded_samples = []
stats = {
    "total_files": len(anno_files),
    "excluded": 0,
    "no_color": 0,
    "no_resolution": 0,
    "no_annotations": 0,
    "included": 0,
}

SYSTEM_MSG = "You are a PCB layout generator. Given a description of components, output their placement as a sequence of tokens representing component type, x-position, y-position, width, and height on a 1280\u00d7720 board."

for af in anno_files:
    board = af.stem
    data = json.loads(af.read_text())
    if isinstance(data, list):
        continue

    anns = data.get("annotations", [])
    
    # Find source image
    src_img = None
    for ext in [".png", ".jpg", ".jpeg"]:
        p = SRC_IMG_DIR / f"{board}{ext}"
        if p.exists():
            src_img = p
            break

    # Check exclusion
    is_excluded = board in excluded_boards
    
    # Check color
    color = board_colors.get(board)
    if not color:
        stats["no_color"] += 1
        is_excluded = True
    
    # Check resolution
    res_class = data.get("resolution_class", "")
    if not res_class:
        stats["no_resolution"] += 1
        is_excluded = True
    
    # Check annotations
    if not anns:
        stats["no_annotations"] += 1
        is_excluded = True

    if is_excluded:
        stats["excluded"] += 1
        if src_img:
            shutil.copy2(str(src_img), str(OUT_IMG_EXCL_DIR / src_img.name))
        continue

    # === Build token sequence ===
    # Sort by Y then X
    sorted_anns = sorted(anns, key=lambda a: (a["bbox"][1], a["bbox"][0]))
    
    tokens = ["[BOS]", f"[COLOR_{color}]", f"[{res_class}]"]
    
    cat_counts = Counter()
    for ann in sorted_anns:
        cat_id = ann["category_id"]
        cat_name = CAT_ID_TO_NAME.get(cat_id)
        if not cat_name:
            continue
        cat_counts[cat_id] += 1
        x, y, w, h = ann["bbox"]
        tokens.extend([
            f"[{cat_name}]",
            f"[{int(round(x)):04d}]",
            f"[{int(round(y)):04d}]",
            f"[{int(round(w)):04d}]",
            f"[{int(round(h)):04d}]",
        ])
    
    tokens.append("[EOS]")
    
    # Build prompt
    prompt_parts = []
    for cat_id in sorted(cat_counts.keys()):
        count = cat_counts[cat_id]
        name = CAT_NAME_LOWER[cat_id]
        prompt_parts.append(f"{count} {name}")
    prompt = "Generate a PCB layout with " + ", ".join(prompt_parts)
    
    # Build ChatML sample
    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": " ".join(tokens)},
        ]
    }
    
    samples.append(sample)
    stats["included"] += 1
    
    # Copy image
    if src_img:
        shutil.copy2(str(src_img), str(OUT_IMG_DIR / src_img.name))

# === Write test.jsonl ===
with open(str(OUT_JSONL), "w") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# === Report ===
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Total annotation files:  {stats['total_files']}")
print(f"Excluded:                {stats['excluded']}")
print(f"  - In excluded_boards:  {len(excluded_boards)}")
print(f"  - No color label:      {stats['no_color']}")
print(f"  - No resolution:       {stats['no_resolution']}")
print(f"  - No annotations:      {stats['no_annotations']}")
print(f"Included (test.jsonl):   {stats['included']}")
print(f"")
print(f"Output:")
print(f"  test.jsonl:            {OUT_JSONL} ({stats['included']} lines)")
print(f"  test images:           {OUT_IMG_DIR} ({len(list(OUT_IMG_DIR.glob('*')))} files)")
print(f"  excluded images:       {OUT_IMG_EXCL_DIR} ({len(list(OUT_IMG_EXCL_DIR.glob('*')))} files)")

# Component stats
all_cat_counts = Counter()
total_tokens = 0
for s in samples:
    content = s["messages"][2]["content"]
    toks = content.split()
    total_tokens += len(toks)
    for t in toks:
        if t.startswith("[") and t.endswith("]"):
            name = t[1:-1]
            if name in CAT_ID_TO_NAME.values():
                all_cat_counts[name] += 1

print(f"\nComponent distribution ({sum(all_cat_counts.values())} total):")
for name in ["RESISTOR", "CAPACITOR", "INDUCTOR", "CONNECTOR", "DIODE", "SWITCH", "TRANSISTOR", "IC", "OSCILLATOR"]:
    print(f"  {name:12s}: {all_cat_counts.get(name, 0)}")

print(f"\nToken stats:")
print(f"  Total tokens:  {total_tokens}")
print(f"  Min:           {min(len(s['messages'][2]['content'].split()) for s in samples)}")
print(f"  Max:           {max(len(s['messages'][2]['content'].split()) for s in samples)}")
print(f"  Mean:          {total_tokens / len(samples):.0f}")
