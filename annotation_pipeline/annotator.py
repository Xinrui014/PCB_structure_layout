#!/usr/bin/env python3
"""PCB COCO Annotation Tool — web-based bbox editor for re-annotating PCB boards."""

import argparse
import csv
import cv2
import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_file

# ── Paths ──────────────────────────────────────────────────────────────
# ── Paths (set by _init_paths() called from __main__) ──
# Defaults used if run without args (backward compat)
ANNO_DIR       = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/train")
IMG_DIR        = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/images/train")
PATCH_ANNO_DIR = ANNO_DIR.parent / "cropped_512" / "annotations" / "train"
META_CSV       = ANNO_DIR.parent.parent / "components" / "metadata_train.csv"
EXCLUDE_FILE   = Path("/home/xinrui/projects/PCB_structure_layout/component_pool/excluded_components.json")
RECLASS_FILE1  = Path("/home/xinrui/projects/PCB_structure_layout/component_pool/reclassified_components.json")
RECLASS_FILE2  = Path("/home/xinrui/projects/PCB_structure_layout/component_pool/orientation_reclassified.json")
ORI_LABELS_FILE= Path("/home/xinrui/projects/PCB_structure_layout/component_pool/orientation_labels.json")
BACKUP_DIR     = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/train_backup")

def _init_paths(anno_dir, image_dir):
    global ANNO_DIR, IMG_DIR, PATCH_ANNO_DIR, META_CSV, EXCLUDE_FILE
    global RECLASS_FILE1, RECLASS_FILE2, ORI_LABELS_FILE, BACKUP_DIR
    global PREDICT_DIRS, PREDICT_DIR, CUSTOM_CAT_FILE, EXCLUDED_BOARDS_FILE
    global FLAG_FILE, CORRECTIONS_FILE, _ORI_FINAL_DIR
    ANNO_DIR = Path(anno_dir)
    IMG_DIR = Path(image_dir)
    PATCH_ANNO_DIR = ANNO_DIR.parent / "cropped_512" / "annotations" / "train"
    META_CSV = ANNO_DIR.parent.parent / "components" / "metadata_train.csv"
    EXCLUDE_FILE = ANNO_DIR / "excluded_components.json"
    RECLASS_FILE1 = ANNO_DIR / "reclassified_components.json"
    RECLASS_FILE2 = ANNO_DIR / "orientation_reclassified.json"
    ORI_LABELS_FILE = ANNO_DIR / "orientation_labels.json"
    BACKUP_DIR = ANNO_DIR / "backup"
    ORIGINAL_ANNO_DIR = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/test")
    PREDICT_DIRS = {"pipeline": ANNO_DIR, "v6": ANNO_DIR / "backup", "original": ORIGINAL_ANNO_DIR}
    PREDICT_DIR = ANNO_DIR
    CUSTOM_CAT_FILE = ANNO_DIR / "custom_categories.json"
    EXCLUDED_BOARDS_FILE = ANNO_DIR.parent / "metadata" / "excluded_boards.json"
    FLAG_FILE = ANNO_DIR / "flagged_annotations.json"
    CORRECTIONS_FILE = ANNO_DIR / "corrections.jsonl"
    _ORI_FINAL_DIR = ANNO_DIR
PREDICT_DIRS   = {
    "pipeline": ANNO_DIR,
    "v6": ANNO_DIR / "backup",
    "original": Path("/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/test"),
}
PREDICT_DIR    = PREDICT_DIRS["pipeline"]
CUSTOM_CAT_FILE= ANNO_DIR / "custom_categories.json"
EXCLUDED_BOARDS_FILE = ANNO_DIR.parent / "metadata" / "excluded_boards.json"
FLAG_FILE      = ANNO_DIR / "flagged_annotations.json"
CORRECTIONS_FILE = ANNO_DIR / "corrections.jsonl"

CROP_SIZE  = 1024
PATCH_SIZE = 512

# ── Categories ─────────────────────────────────────────────────────────
CATEGORIES = [
    {"id": 1, "name": "Resistor"},
    {"id": 2, "name": "Capacitor"},
    {"id": 3, "name": "Inductor"},
    {"id": 4, "name": "Connector"},
    {"id": 5, "name": "Diode"},
    # LED (6) merged into Diode (5)
    {"id": 7, "name": "Switch"},
    {"id": 8, "name": "Transistor"},
    {"id": 9, "name": "Integrated Circuit"},
]

app = Flask(__name__)

# ── Global state ───────────────────────────────────────────────────────
board_list = []
excluded_ids = set()
# Per-board maps: board_name → { orig_ann_id: {info} }
board_excl_annids = {}   # board_name → set of original ann IDs that are excluded
board_reclass = {}       # board_name → {orig_ann_id: new_category_name}
board_ori_labels = {}    # board_name → {orig_ann_id: angle}
board_ann_to_crops = {}  # board_name → {orig_ann_id: [crop_id, ...]}  (reverse map for saving)
flagged_anns = {}        # board_name → [ann_id, ...]
# Crop-level data (for preserving on save)
crop_ori_labels = {}     # crop_id → angle
crop_reclass = {}        # crop_id → new_category_name


# ── Helper: IoU ────────────────────────────────────────────────────────
def bbox_iou(b1, b2):
    """COCO format [x,y,w,h] IoU."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1, a2 = b1[2]*b1[3], b2[2]*b2[3]
    return inter / (a1 + a2 - inter + 1e-6)


def parse_crop_id(crop_id):
    """Parse crop ID like 'train_BOARD_PATCH_SEQ' → (board, patch_suffix, seq_int)."""
    rest = crop_id[6:] if crop_id.startswith("train_") else crop_id
    parts = rest.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    return None, None, None


def build_patch_to_orig_map(board, orig_anns, img_w, img_h):
    """Build mapping: (patch_suffix, seq_id) → orig_ann_id for a board."""
    min_side = min(img_w, img_h)
    sc = CROP_SIZE / float(min_side)
    new_w = int(round(img_w * sc))
    new_h = int(round(img_h * sc))
    if min(new_w, new_h) < CROP_SIZE:
        new_w = int(math.ceil(img_w * sc))
        new_h = int(math.ceil(img_h * sc))
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2

    mapping = {}  # (patch_suffix, seq_id) → orig_ann_id

    for row_i in (0, 1):
        for col_i in (0, 1):
            patch_suffix = f"{row_i}{col_i}"
            patch_anno_file = PATCH_ANNO_DIR / f"{board}_{patch_suffix}.json"
            if not patch_anno_file.exists():
                continue
            patch_data = json.loads(patch_anno_file.read_text())
            patch_anns = patch_data.get("annotations", [])

            px = col_i * PATCH_SIZE
            py = row_i * PATCH_SIZE

            for pann in patch_anns:
                pbbox = pann["bbox"]
                crop_x = pbbox[0] + px
                crop_y = pbbox[1] + py
                orig_x = (crop_x + left) / sc
                orig_y = (crop_y + top) / sc

                best_dist = float("inf")
                best_id = None
                for oann in orig_anns:
                    obbox = oann.get("bbox", [])
                    if len(obbox) != 4:
                        continue
                    dist = abs(obbox[0] - orig_x) + abs(obbox[1] - orig_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = oann["id"]
                if best_id is not None and best_dist < 5.0:
                    mapping[(patch_suffix, pann["id"])] = best_id

    return mapping


def build_board_list():
    """Build board list + map excluded/reclassified/orientation to original ann IDs."""
    global board_list, excluded_ids, board_excl_annids, board_reclass, board_ori_labels
    global crop_ori_labels, crop_reclass

    # Load all crop-level data
    excluded_ids = set(json.load(open(EXCLUDE_FILE))) if EXCLUDE_FILE.exists() else set()
    reclass1 = json.load(open(RECLASS_FILE1)) if RECLASS_FILE1.exists() else {}
    reclass2 = json.load(open(RECLASS_FILE2)) if RECLASS_FILE2.exists() else {}
    crop_ori_labels = json.load(open(ORI_LABELS_FILE)) if ORI_LABELS_FILE.exists() else {}
    # Merge reclassified
    crop_reclass = {**reclass1, **reclass2}

    print(f"Excluded crop IDs: {len(excluded_ids)}")
    print(f"Reclassified crops: {len(crop_reclass)}")
    print(f"Orientation labels: {len(crop_ori_labels)}")

    # Group crop-level data by board
    excl_by_patch = defaultdict(set)
    reclass_by_board = defaultdict(dict)   # board → {(patch, seq): new_cat}
    ori_by_board = defaultdict(dict)       # board → {(patch, seq): angle}

    for eid in excluded_ids:
        board, patch, seq = parse_crop_id(eid)
        if board:
            excl_by_patch[f"{board}_{patch}"].add(seq)

    for cid, cat in crop_reclass.items():
        board, patch, seq = parse_crop_id(cid)
        if board:
            reclass_by_board[board][(patch, seq)] = cat

    for cid, angle in crop_ori_labels.items():
        board, patch, seq = parse_crop_id(cid)
        if board:
            ori_by_board[board][(patch, seq)] = angle

    anno_boards = set(f.stem for f in ANNO_DIR.glob("*.json"))

    meta_boards = set()
    boards_with_excluded = set()
    if META_CSV.exists():
        with open(META_CSV) as f:
            for row in csv.DictReader(f):
                if row["split"] != "train":
                    continue
                board = re.sub(r"_\d{2}$", "", row["source_image"])
                meta_boards.add(board)
                if row["id"] in excluded_ids:
                    boards_with_excluded.add(board)
    else:
        print(f"Warning: {META_CSV} not found, skipping metadata")

    # Include ALL 1280x720 boards with annotations (full 4k set minus empties and non-standard)
    all_candidates = anno_boards
    empty_boards = set()
    non_standard = set()
    for board in all_candidates:
        anno_file = ANNO_DIR / f"{board}.json"
        if anno_file.exists():
            data = json.loads(anno_file.read_text())
            anns = data.get("annotations", [])
            imgs = data.get("images", [])
            if not anns:
                empty_boards.add(board)
            elif imgs and (imgs[0].get("width") != 1280 or imgs[0].get("height") != 720):
                non_standard.add(board)
    all_candidates -= empty_boards | non_standard
    board_list = sorted(all_candidates)
    print(f"  Empty boards removed: {len(empty_boards)}")
    print(f"  Non-1280x720 boards removed: {len(non_standard)}")
    print(f"Boards to annotate: {len(board_list)}")

    # Build reverse mappings for all boards that have any crop-level data
    print("Building reverse mappings (excluded + reclassified + orientation)...")
    board_excl_annids.clear()
    board_reclass.clear()
    board_ori_labels.clear()

    boards_needing_map = set()
    for board in board_list:
        if board in boards_with_excluded or board in reclass_by_board or board in ori_by_board:
            boards_needing_map.add(board)

    for i, board in enumerate(sorted(boards_needing_map)):
        orig_anno_file = ANNO_DIR / f"{board}.json"
        if not orig_anno_file.exists():
            continue
        orig_data = json.loads(orig_anno_file.read_text())
        orig_images = orig_data.get("images", [])
        orig_anns = orig_data.get("annotations", [])
        if not orig_images or not orig_anns:
            continue

        img_w = orig_images[0]["width"]
        img_h = orig_images[0]["height"]

        mapping = build_patch_to_orig_map(board, orig_anns, img_w, img_h)

        # Map excluded
        excl_ids = set()
        for row_i in (0, 1):
            for col_i in (0, 1):
                patch_suffix = f"{row_i}{col_i}"
                for seq in excl_by_patch.get(f"{board}_{patch_suffix}", set()):
                    orig_id = mapping.get((patch_suffix, seq))
                    if orig_id is not None:
                        excl_ids.add(orig_id)
        if excl_ids:
            board_excl_annids[board] = excl_ids

        # Map reclassified
        rc = {}
        for (patch, seq), cat in reclass_by_board.get(board, {}).items():
            orig_id = mapping.get((patch, seq))
            if orig_id is not None:
                rc[orig_id] = cat
        if rc:
            board_reclass[board] = rc

        # Map orientation labels + build reverse crop ID map
        ori = {}
        ann_to_crops = defaultdict(list)
        for (patch, seq), angle in ori_by_board.get(board, {}).items():
            orig_id = mapping.get((patch, seq))
            if orig_id is not None:
                ori[orig_id] = angle
                ann_to_crops[orig_id].append(f"train_{board}_{patch}_{seq:04d}")
        # Also map all annotations (not just orientation-labeled ones)
        for (patch_suffix, seq_id), orig_id in mapping.items():
            crop_id = f"train_{board}_{patch_suffix}_{seq_id:04d}"
            if orig_id not in ann_to_crops:
                ann_to_crops[orig_id].append(crop_id)
        if ori:
            board_ori_labels[board] = ori
        if ann_to_crops:
            board_ann_to_crops[board] = dict(ann_to_crops)

    print(f"  Excluded: {sum(len(v) for v in board_excl_annids.values())} annotations across {len(board_excl_annids)} boards")
    print(f"  Reclassified: {sum(len(v) for v in board_reclass.values())} annotations across {len(board_reclass)} boards")
    print(f"  Orientation: {sum(len(v) for v in board_ori_labels.values())} annotations across {len(board_ori_labels)} boards")


def find_image(board_name):
    for ext in [".png", ".jpg", ".JPG", ".jpeg", ".JPEG", ".PNG"]:
        p = IMG_DIR / (board_name + ext)
        if p.exists():
            return p
    return None


# ── Correction Tracking Helpers ────────────────────────────────────────

def compute_corrections(board_name, v6_anns, saved_anns):
    """Compute diffs between v6 predictions and saved annotations. Return list of correction dicts."""
    corrections = []
    ts = datetime.utcnow().isoformat() + "Z"

    # Build best-match mapping: v6 → saved (greedy by IoU)
    used_saved = set()
    v6_to_saved = {}  # v6 index → saved index

    for vi, va in enumerate(v6_anns):
        best_iou = 0
        best_si = None
        for si, sa in enumerate(saved_anns):
            if si in used_saved:
                continue
            iou = bbox_iou(va["bbox"], sa["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_si = si
        if best_si is not None and best_iou > 0.3:
            v6_to_saved[vi] = (best_si, best_iou)
            used_saved.add(best_si)

    # Analyze each v6 annotation
    for vi, va in enumerate(v6_anns):
        if vi not in v6_to_saved:
            # Deleted — no match in saved
            corrections.append({
                "action": "deleted",
                "board": board_name,
                "ann_id": va.get("id"),
                "bbox": va["bbox"],
                "v6_class": va["category_id"],
                "v6_conf": va.get("score", va.get("confidence")),
                "timestamp": ts,
            })
        else:
            si, iou = v6_to_saved[vi]
            sa = saved_anns[si]
            # Check reclassify
            if va["category_id"] != sa["category_id"] and iou > 0.7:
                corrections.append({
                    "action": "reclassify",
                    "board": board_name,
                    "ann_id": va.get("id"),
                    "bbox": va["bbox"],
                    "v6_class": va["category_id"],
                    "v6_conf": va.get("score", va.get("confidence")),
                    "corrected_class": sa["category_id"],
                    "timestamp": ts,
                })
            # Check bbox modified (matched but IoU < 0.9)
            if iou < 0.9:
                corrections.append({
                    "action": "bbox_modified",
                    "board": board_name,
                    "ann_id": va.get("id"),
                    "bbox_v6": va["bbox"],
                    "bbox_corrected": sa["bbox"],
                    "iou": round(iou, 3),
                    "v6_class": va["category_id"],
                    "v6_conf": va.get("score", va.get("confidence")),
                    "timestamp": ts,
                })

    # Check for added annotations (saved but no match in v6)
    for si, sa in enumerate(saved_anns):
        if si not in used_saved:
            corrections.append({
                "action": "added",
                "board": board_name,
                "bbox": sa["bbox"],
                "corrected_class": sa["category_id"],
                "timestamp": ts,
            })

    return corrections


def log_corrections(corrections):
    """Append corrections to CORRECTIONS_FILE as JSONL."""
    if not corrections:
        return
    CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_FILE, "a") as f:
        for c in corrections:
            f.write(json.dumps(c) + "\n")


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/categories")
def api_categories():
    return jsonify(CATEGORIES)


@app.route("/api/boards")
def api_boards():
    reviewed = set(f.stem for f in BACKUP_DIR.glob("*.json")) if BACKUP_DIR.exists() else set()
    boards_info = [{"name": b, "reviewed": b in reviewed, "excluded": b in _excluded_boards} for b in board_list]
    return jsonify({"boards": board_list, "boards_info": boards_info, "total": len(board_list)})


@app.route("/api/board/<board_name>")
def api_board(board_name):
    anno_file = ANNO_DIR / (board_name + ".json")
    if not anno_file.exists():
        return jsonify({"error": "Annotation not found"}), 404
    img_file = find_image(board_name)
    if not img_file:
        return jsonify({"error": "Image not found"}), 404

    # Check if this board has been manually reviewed
    backup_file = BACKUP_DIR / (board_name + ".json")
    is_reviewed = backup_file.exists()

    # Determine annotation source based on version
    predict_file = PREDICT_DIR / (board_name + ".json")
    # Check which version is active
    cur_ver = "v6"
    for k, v in PREDICT_DIRS.items():
        if v == PREDICT_DIR:
            cur_ver = k
            break

    if cur_ver == "orig":
        # Always show original annotations
        data = json.loads(anno_file.read_text())
        annotations = data.get("annotations", [])
        use_predictions = False
    elif not is_reviewed and predict_file.exists():
        # Use YOLO predictions for unreviewed boards
        pred_data = json.loads(predict_file.read_text())
        annotations = pred_data.get("annotations", [])
        data = json.loads(anno_file.read_text())
        use_predictions = True
    else:
        data = json.loads(anno_file.read_text())
        annotations = data.get("annotations", [])
        use_predictions = False

    images = data.get("images", [])
    img_w = images[0]["width"] if images else 0
    img_h = images[0]["height"] if images else 0

    excl_set = board_excl_annids.get(board_name, set())
    rc_map = board_reclass.get(board_name, {})
    ori_map = board_ori_labels.get(board_name, {})
    flag_set = set(flagged_anns.get(board_name, []))

    # Merge LED (6) → Diode (5)
    MERGE_MAP = {6: 5}

    anno_out = []
    for ann in annotations:
        aid = ann["id"]
        cat_id = MERGE_MAP.get(ann["category_id"], ann["category_id"])
        anno_out.append({
            "id": aid,
            "category_id": cat_id,
            "bbox": ann["bbox"],
            "is_excluded": aid in excl_set if not use_predictions else False,
            "is_flagged": aid in flag_set if not use_predictions else False,
            "is_predicted": use_predictions,
            "reclassified_to": rc_map.get(aid) if not use_predictions else None,
            "orientation": ori_map.get(aid),
            "score": ann.get("score"),
        })

    return jsonify({
        "board": board_name,
        "image_width": img_w,
        "image_height": img_h,
        "annotations": anno_out,
        "source": "predicted" if use_predictions else "manual" if is_reviewed else "original",
        "index": board_list.index(board_name) if board_name in board_list else -1,
        "total": len(board_list),
        "excluded": board_name in _excluded_boards,
    })


@app.route("/api/image/<board_name>")
def api_image(board_name):
    img_file = find_image(board_name)
    if not img_file:
        return "Not found", 404
    return send_file(img_file)


def load_custom_categories():
    """Load custom categories from persistent file and merge into CATEGORIES."""
    if CUSTOM_CAT_FILE.exists():
        custom = json.loads(CUSTOM_CAT_FILE.read_text())
        existing_ids = {c["id"] for c in CATEGORIES}
        for c in custom:
            if c["id"] not in existing_ids:
                CATEGORIES.append(c)
                existing_ids.add(c["id"])
        print(f"Loaded {len(custom)} custom categories from {CUSTOM_CAT_FILE}")


def save_custom_categories():
    """Save non-default categories to persistent file."""
    default_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    custom = [c for c in CATEGORIES if c["id"] not in default_ids]
    CUSTOM_CAT_FILE.write_text(json.dumps(custom, indent=2))


@app.route("/api/add_category", methods=["POST"])
def api_add_category():
    data = request.json
    new_id = data.get("id")
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "Empty name"}), 400
    # Check duplicates
    for c in CATEGORIES:
        if c["name"].lower() == new_name.lower():
            return jsonify({"error": f"Already exists: {c['name']}"}), 400
    CATEGORIES.append({"id": new_id, "name": new_name})
    save_custom_categories()
    print(f"Added category: id={new_id} name={new_name} (saved to {CUSTOM_CAT_FILE})")
    return jsonify({"ok": True, "id": new_id, "name": new_name})


# ── Excluded Boards (manual + auto) ──
_excluded_boards = set()
def load_excluded_boards():
    global _excluded_boards
    if EXCLUDED_BOARDS_FILE.exists():
        _excluded_boards = set(json.loads(EXCLUDED_BOARDS_FILE.read_text()))
    print(f"Excluded boards: {len(_excluded_boards)}")
def save_excluded_boards():
    EXCLUDED_BOARDS_FILE.write_text(json.dumps(sorted(_excluded_boards), indent=2))

@app.route("/api/exclude_board", methods=["POST"])
def api_exclude_board():
    board = request.json.get("board", "")
    action = request.json.get("action", "exclude")  # exclude or include
    if action == "exclude":
        _excluded_boards.add(board)
    else:
        _excluded_boards.discard(board)
    save_excluded_boards()
    return jsonify({"ok": True, "excluded": board in _excluded_boards, "total_excluded": len(_excluded_boards)})

@app.route("/api/excluded_boards")
def api_excluded_boards():
    return jsonify({"boards": sorted(_excluded_boards), "total": len(_excluded_boards)})

@app.route("/api/predict_version", methods=["GET", "POST"])
def api_predict_version():
    global PREDICT_DIR
    if request.method == "POST":
        ver = request.json.get("version", "v2")
        if ver in PREDICT_DIRS:
            PREDICT_DIR = PREDICT_DIRS[ver]
            return jsonify({"ok": True, "version": ver})
        return jsonify({"error": f"Unknown version: {ver}"}), 400
    # GET — set version via query param or return current
    ver = request.args.get("set")
    if ver:
        if ver in PREDICT_DIRS:
            PREDICT_DIR = PREDICT_DIRS[ver]
            return jsonify({"ok": True, "version": ver, "versions": list(PREDICT_DIRS.keys())})
        return jsonify({"error": f"Unknown version: {ver}"}), 400
    for k, v in PREDICT_DIRS.items():
        if v == PREDICT_DIR:
            return jsonify({"version": k, "versions": list(PREDICT_DIRS.keys())})
    return jsonify({"version": "unknown"})


@app.route("/api/save/<board_name>", methods=["POST"])
def api_save(board_name):
    anno_file = ANNO_DIR / (board_name + ".json")
    if not anno_file.exists():
        return jsonify({"error": "Original not found"}), 404

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_file = BACKUP_DIR / (board_name + ".json")
    first_save = not backup_file.exists()
    if first_save:
        shutil.copy2(anno_file, backup_file)

    original = json.loads(anno_file.read_text())
    new_annos = request.json.get("annotations", [])

    # ── Feature 1: Correction Tracking ──
    num_corrections = 0
    if first_save:
        # Always diff against v6 predictions (baseline) regardless of current version
        predict_file = PREDICT_DIRS["v6"] / (board_name + ".json")
        if predict_file.exists():
            pred_data = json.loads(predict_file.read_text())
            v6_anns = pred_data.get("annotations", [])
            corrections = compute_corrections(board_name, v6_anns, new_annos)
            num_corrections = len(corrections)
            log_corrections(corrections)

    coco_annos = []
    for ann in new_annos:
        bbox = ann["bbox"]
        coco_annos.append({
            "id": ann.get("id", 0),
            "image_id": original["images"][0]["id"] if original.get("images") else 0,
            "category_id": ann["category_id"],
            "bbox": [round(b, 2) for b in bbox],
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
        })

    original["annotations"] = coco_annos
    original["categories"] = [{"id": c["id"], "name": c["name"]} for c in CATEGORIES]
    anno_file.write_text(json.dumps(original, indent=2))
    return jsonify({"ok": True, "count": len(coco_annos), "corrections": num_corrections})


# ── Feature 2: Smart Review Ordering ──────────────────────────────────

@app.route("/api/board_priorities")
def api_board_priorities():
    """Return ALL boards sorted by uncertainty. Reviewed boards get score 0 but stay in list."""
    reviewed = set(f.stem for f in BACKUP_DIR.glob("*.json")) if BACKUP_DIR.exists() else set()
    confused_cats = {3, 5, 7}  # Inductor, Diode, Switch
    results = []

    for board_name in board_list:
        is_reviewed = board_name in reviewed
        predict_file = PREDICT_DIR / (board_name + ".json")
        if not predict_file.exists() or is_reviewed:
            # Reviewed or no predictions — score 0, still in list
            results.append({
                "board": board_name,
                "n_low_conf": 0,
                "n_confused": 0,
                "score": 0,
                "reviewed": is_reviewed,
            })
            continue
        try:
            pred_data = json.loads(predict_file.read_text())
        except Exception:
            results.append({"board": board_name, "n_low_conf": 0, "n_confused": 0, "score": 0, "reviewed": False})
            continue
        anns = pred_data.get("annotations", [])

        n_low_conf = 0
        n_confused = 0
        for a in anns:
            conf = a.get("score", a.get("confidence", 1.0))
            if conf is None:
                conf = 1.0
            if conf < 0.5:
                n_low_conf += 1
            if a.get("category_id") in confused_cats and conf < 0.7:
                n_confused += 1

        score = n_low_conf * 2 + n_confused
        results.append({
            "board": board_name,
            "n_low_conf": n_low_conf,
            "n_confused": n_confused,
            "score": score,
            "reviewed": False,
        })

    results.sort(key=lambda x: -x["score"])
    return jsonify(results)


# ── Feature 3: Class-Focused Review ───────────────────────────────────

@app.route("/class_review")
def class_review_page():
    return CLASS_REVIEW_HTML


@app.route("/api/class_crops/<int:cat_id>")
def api_class_crops(cat_id):
    """Return all annotations matching cat_id from unreviewed boards, sorted by confidence ascending."""
    reviewed = set(f.stem for f in BACKUP_DIR.glob("*.json")) if BACKUP_DIR.exists() else set()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 200, type=int)

    items = []
    for board_name in board_list:
        if board_name in reviewed:
            continue
        predict_file = PREDICT_DIR / (board_name + ".json")
        if not predict_file.exists():
            continue
        try:
            pred_data = json.loads(predict_file.read_text())
        except Exception:
            continue
        for a in pred_data.get("annotations", []):
            if a.get("category_id") == cat_id:
                items.append({
                    "board": board_name,
                    "ann_id": a["id"],
                    "bbox": a["bbox"],
                    "conf": a.get("score", a.get("confidence", 1.0)),
                    "category_id": cat_id,
                })

    # Sort by confidence ascending (lowest first)
    items.sort(key=lambda x: x["conf"] if x["conf"] is not None else 1.0)

    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]
    total_pages = max(1, (total + per_page - 1) // per_page)

    return jsonify({
        "items": page_items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
    })


@app.route("/api/class_crop_image/<board_name>/<int:ann_id>")
def api_class_crop_image(board_name, ann_id):
    """Crop a component from the board image and return as JPEG."""
    img_file = find_image(board_name)
    if not img_file:
        return "Image not found", 404

    # Find annotation — check prediction file first, then annotation file
    ann = None
    for src_file in [PREDICT_DIR / (board_name + ".json"), ANNO_DIR / (board_name + ".json")]:
        if src_file.exists():
            try:
                data = json.loads(src_file.read_text())
                for a in data.get("annotations", []):
                    if a["id"] == ann_id:
                        ann = a
                        break
            except Exception:
                continue
        if ann:
            break

    if not ann:
        return "Annotation not found", 404

    # Load image with cv2
    img = cv2.imread(str(img_file))
    if img is None:
        return "Failed to load image", 500

    h_img, w_img = img.shape[:2]
    x, y, w, h = ann["bbox"]

    # 1.5x padding
    cx, cy = x + w / 2, y + h / 2
    pw, ph = w * 1.5, h * 1.5
    x1 = max(0, int(cx - pw / 2))
    y1 = max(0, int(cy - ph / 2))
    x2 = min(w_img, int(cx + pw / 2))
    y2 = min(h_img, int(cy + ph / 2))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "Empty crop", 500

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/class_reclassify", methods=["POST"])
def api_class_reclassify():
    """Batch reclassify annotations in prediction files."""
    items = request.json.get("items", [])
    ts = datetime.utcnow().isoformat() + "Z"
    corrections = []
    changed = 0

    # Group by board for efficiency
    by_board = defaultdict(list)
    for item in items:
        by_board[item["board"]].append(item)

    for board_name, board_items in by_board.items():
        predict_file = PREDICT_DIR / (board_name + ".json")
        if not predict_file.exists():
            continue
        try:
            pred_data = json.loads(predict_file.read_text())
        except Exception:
            continue

        ann_map = {a["id"]: a for a in pred_data.get("annotations", [])}
        modified = False

        for item in board_items:
            ann_id = item["ann_id"]
            new_cat = item["new_category_id"]
            if ann_id in ann_map:
                old_ann = ann_map[ann_id]
                old_cat = old_ann["category_id"]
                if old_cat != new_cat:
                    corrections.append({
                        "action": "reclassify",
                        "board": board_name,
                        "ann_id": ann_id,
                        "bbox": old_ann["bbox"],
                        "v6_class": old_cat,
                        "v6_conf": old_ann.get("score", old_ann.get("confidence")),
                        "corrected_class": new_cat,
                        "source": "class_review",
                        "timestamp": ts,
                    })
                    old_ann["category_id"] = new_cat
                    modified = True
                    changed += 1

        if modified:
            predict_file.write_text(json.dumps(pred_data, indent=2))

    log_corrections(corrections)
    return jsonify({"ok": True, "changed": changed, "corrections": len(corrections)})


# ── HTML ───────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB COCO Annotator</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; display: flex; height: 100vh; overflow: hidden; }

/* Sidebar */
#sidebar { width: 280px; min-width: 0; flex-shrink: 0; background: #16213e; display: flex; flex-direction: column; border-right: 1px solid #0f3460; transition: width 0.2s ease, min-width 0.2s ease; overflow: hidden; }
#sidebar.collapsed { width: 0; border-right: none; }
#sidebar h2 { padding: 12px 16px; font-size: 16px; background: #0f3460; white-space: nowrap; }
#search { margin: 8px; padding: 6px 10px; background: #1a1a2e; border: 1px solid #444; color: #eee; border-radius: 4px; min-width: 240px; }
#board-list { flex: 1; overflow-y: auto; }
.bitem { padding: 6px 16px; cursor: pointer; font-size: 12px; border-bottom: 1px solid #1a1a2e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bitem:hover { background: #1a1a4e; }
.bitem.active { background: #0f3460; font-weight: bold; }
/* Toggle buttons */
#toggle-sidebar { position: fixed; top: 50%; left: 0; transform: translateY(-50%); z-index: 100; width: 18px; height: 48px; background: #0f3460; border: 1px solid #4488cc; border-left: none; border-radius: 0 6px 6px 0; cursor: pointer; color: #adf; font-size: 11px; display: flex; align-items: center; justify-content: center; transition: left 0.2s ease; user-select: none; }
#toggle-sidebar:hover { background: #1a5090; }
#toggle-ann { position: fixed; top: 50%; right: 0; transform: translateY(-50%); z-index: 100; width: 18px; height: 48px; background: #0f3460; border: 1px solid #4488cc; border-right: none; border-radius: 6px 0 0 6px; cursor: pointer; color: #adf; font-size: 11px; display: flex; align-items: center; justify-content: center; transition: right 0.2s ease; user-select: none; }
#toggle-ann:hover { background: #1a5090; }

/* Main */
#main { flex: 1; display: flex; flex-direction: column; min-width: 0; overflow: hidden; }

/* Toolbar */
#toolbar { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #16213e; border-bottom: 1px solid #0f3460; flex-wrap: wrap; }
.info { font-size: 13px; color: #aaa; }
.board-name { font-weight: bold; color: #e94560; font-size: 14px; }
.btn { padding: 5px 12px; border: 1px solid #444; background: #1a1a2e; color: #eee; border-radius: 4px; cursor: pointer; font-size: 12px; }
.btn:hover { background: #0f3460; }
.btn.save { background: #27ae60; border-color: #27ae60; }
.btn.save:hover { background: #2ecc71; }
.btn.del { background: #c0392b; border-color: #c0392b; }
.btn.del:hover { background: #e74c3c; }
.btn.nav { background: #2980b9; border-color: #2980b9; }
.btn.active-mode { background: #f39c12; border-color: #f39c12; }

/* Category bar */
#cat-bar { display: flex; gap: 4px; padding: 6px 12px; background: #111; flex-wrap: wrap; align-items: center; }
.cat-btn { padding: 4px 10px; border: 2px solid #444; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; background: transparent; color: #ccc; }
.cat-btn:hover { opacity: 0.8; }
.cat-btn.active { box-shadow: 0 0 8px rgba(255,255,255,0.5); color: #fff; }
.cat-btn kbd { background: #333; padding: 1px 5px; border-radius: 2px; font-size: 11px; margin-right: 4px; }

/* Canvas */
#canvas-wrap { flex: 1; position: relative; overflow: scroll; background: #111; min-width: 0; min-height: 0; }
#canvas-inner { width: max-content; padding: 10px; }
#canvas { cursor: crosshair; display: block; }

/* Annotation list */
#ann-panel { width: 220px; min-width: 0; flex-shrink: 0; background: #16213e; border-left: 1px solid #0f3460; overflow-y: auto; transition: width 0.2s ease; overflow: hidden; }
#ann-panel.collapsed { width: 0; border-left: none; }
#ann-panel h3 { padding: 8px 12px; font-size: 13px; background: #0f3460; white-space: nowrap; }
.ann-item { padding: 4px 12px; font-size: 11px; cursor: pointer; border-bottom: 1px solid #1a1a2e; display: flex; justify-content: space-between; align-items: center; }
.ann-item:hover { background: #1a1a4e; }
.ann-item.sel { background: #0f3460; }
.ann-item .tag { padding: 1px 6px; border-radius: 3px; font-size: 10px; }
.ann-item .excl { color: #e74c3c; }

/* Status */
#status { padding: 6px 12px; background: #16213e; font-size: 12px; color: #aaa; border-top: 1px solid #0f3460; display: flex; justify-content: space-between; }

/* Legend */
.legend { display: flex; gap: 12px; align-items: center; font-size: 12px; }
.ldot { width: 12px; height: 12px; border-radius: 2px; display: inline-block; vertical-align: middle; margin-right: 3px; }
</style>
<script>
document.addEventListener("DOMContentLoaded",function(){
  document.getElementById("canvas-wrap").addEventListener("contextmenu",function(e){e.preventDefault();e.stopPropagation();return false;},true);
  document.getElementById("canvas").addEventListener("contextmenu",function(e){e.preventDefault();e.stopPropagation();return false;},true);
});
</script>
</head>
<body>

<div id="toggle-sidebar" onclick="toggleSidebar()" title="Toggle sidebar [\\]">◀</div>
<div id="toggle-ann" onclick="toggleAnn()" title="Toggle annotations [/]">▶</div>

<div id="sidebar">
  <h2>🔧 PCB Annotator <span style="font-size:11px;color:#888" id="bcount"></span></h2>
  <input type="text" id="search" placeholder="Search boards...">
  <div style="display:flex;gap:4px;padding:4px 8px;background:#111;border-bottom:1px solid #0f3460">
    <button id="filter-all" class="btn" onclick="setFilter('all')" style="flex:1;font-size:11px;padding:3px 6px;background:#0f3460;border-color:#4488cc">All</button>
    <button id="filter-todo" class="btn" onclick="setFilter('todo')" style="flex:1;font-size:11px;padding:3px 6px">⏳ Todo</button>
    <button id="filter-done" class="btn" onclick="setFilter('done')" style="flex:1;font-size:11px;padding:3px 6px">✅ Done</button>
    <button id="filter-smart" class="btn" onclick="smartSort()" style="flex:1;font-size:11px;padding:3px 6px;background:#e94560;border-color:#e94560" title="Sort by uncertainty">🎯 Smart</button>
  </div>
  <div id="board-list"></div>
</div>

<div id="main">
  <div id="toolbar">
    <button class="btn nav" onclick="navBoard(-1)" title="A / ←">◀ Prev</button>
    <button class="btn nav" onclick="navBoard(1)" title="D / →">Next ▶</button>
    <span class="board-name" id="bname">—</span>
    <span class="info" id="binfo"></span>
    <button class="btn" id="ver-btn" onclick="toggleVersion()" style="background:#8e44ad;border-color:#8e44ad" title="Switch prediction model">🔀 Pipeline</button>
    <a href="/class_review" class="btn" style="background:#e94560;border-color:#e94560;text-decoration:none" title="Class-focused review">🔬 Class Review</a>
    <div style="flex:1"></div>
    <div class="legend">
      <span><span class="ldot" style="background:rgba(0,120,255,1)"></span>Normal</span>
      <span><span class="ldot" style="background:rgba(255,30,30,1)"></span>Excluded</span>
      <span><span class="ldot" style="background:rgba(0,255,100,1)"></span>Selected</span>
      <span><span class="ldot" style="background:rgba(0,200,255,0.9);border:1px dashed #0cf"></span>YOLO</span>
      <span><span class="ldot" style="background:rgba(255,0,255,1)"></span>Flagged</span>
      <span><span class="ldot" style="background:rgba(255,165,0,1);border:1px dashed #f90"></span>Reclassified</span>
      <span><span class="ldot" style="background:rgba(255,255,0,0.8)"></span>Orientation</span>
    </div>
    <!-- Mode is auto: click bbox=select, click empty=draw -->
    <button class="btn" onclick="resetZoom()" title="Reset zoom">🔍 Reset</button>
    <button class="btn del" onclick="delSelected()" title="Delete / X">🗑 Delete</button>
    <button id="excl-board-btn" class="btn" onclick="toggleExcludeBoard()" style="background:#7f8c8d;border-color:#7f8c8d;font-size:11px" title="Exclude/include board from training">🚫 Exclude Board</button>
    <button class="btn save" onclick="saveBoard()" title="Ctrl+S">💾 Save</button>
  </div>

  <div id="cat-bar"></div>

  <div style="flex:1;display:flex;overflow:hidden;min-height:0">
    <div id="canvas-wrap" oncontextmenu="return false;">
      <div id="canvas-inner"><canvas id="canvas" oncontextmenu="return false;"></canvas></div>
    </div>
    <div id="ann-panel">
      <h3>Annotations <span id="ann-count"></span></h3>

      <div id="ann-list"></div>
    </div>
  </div>

  <div id="status">
    <span id="st-left">Ready</span>
    <span id="st-right">1-9,0,Q-Y=category | X/Del/F=delete | Ctrl+S=save | A/D/←→=nav | \ /=panels | Scroll=zoom | Space+drag=pan | R=reset</span>
  </div>
</div>

<script>
let CAT_NAMES = {1:"Resistor",2:"Capacitor",3:"Inductor",4:"Connector",5:"Diode",7:"Switch",8:"Transistor",9:"Integrated Circuit"};
let CAT_SHORT = {1:"Res",2:"Cap",3:"Ind",4:"Con",5:"Dio",7:"Swi",8:"Tra",9:"IC"};
const CAT_COLORS = {
  1:"#ff4444",   // Resistor — RED
  2:"#9b59b6",   // Capacitor — PURPLE
  3:"#00dddd",   // Inductor — CYAN
  4:"#00dd00",   // Connector — GREEN
  5:"#ffffff",   // Diode — WHITE
  7:"#ff8800",   // Switch — ORANGE
  8:"#ffdd00",   // Transistor — YELLOW
  9:"#4488ff",   // IC — BLUE
  10:"#88ff88",  // Oscillator — LIGHT GREEN
  11:"#333333",  // Fuse — BLACK
};

let boards=[], boardsInfo=[], filtered=[], curBoard=null, curIdx=-1;
let boardFilter='all';
let anns=[], imgW=0, imgH=0, curSource="";
let origAnns={};  // snapshot of original annotations on load: {id: {category_id, bbox}}
let selId=null, activeCat=1;
let multiSel=new Set();
let boardImg=null, scale=1;
let dirty=false, mode="select", dispScale=1;
let zoomLevel=1, panX=0, panY=0, panning=false, panStart=null;
let drawing=false, drawStart=null, drawCur=null;
let dragging=false, dragType=null, dragStart=null, dragOrig=null;
const HS=6;
const cv=document.getElementById("canvas"), cx=cv.getContext("2d");
let smartPriorities=null;  // cached smart sort data

// ── Init ──
async function init(){
  // Load categories from server (includes custom ones)
  const cats=await(await fetch("/api/categories")).json();
  cats.forEach(c=>{
    CAT_NAMES[c.id]=c.name;
    CAT_SHORT[c.id]=c.name.slice(0,3);
    if(!CAT_COLORS[c.id]) CAT_COLORS[c.id]=`hsl(${(c.id*47)%360},70%,60%)`;
  });

  const r=await(await fetch("/api/boards")).json();
  boards=r.boards; boardsInfo=r.boards_info||boards.map(b=>({name:b,reviewed:false}));
  filtered=[...boards];
  document.getElementById("bcount").textContent=`(${boards.length})`;
  renderList(); buildCatBar();
  if(boards.length) loadBoard(boards[0]);
}

function setFilter(f){
  boardFilter=f; smartPriorities=null;
  ['all','todo','done'].forEach(k=>{
    const btn=document.getElementById('filter-'+k);
    btn.style.background=k===f?'#0f3460':'';
    btn.style.borderColor=k===f?'#4488cc':'#444';
  });
  document.getElementById('filter-smart').style.background=f==='smart'?'#0f3460':'#e94560';
  applyFilter();
}

async function smartSort(){
  boardFilter='smart';
  ['all','todo','done'].forEach(k=>{
    const btn=document.getElementById('filter-'+k);
    btn.style.background=''; btn.style.borderColor='#444';
  });
  document.getElementById('filter-smart').style.background='#0f3460';
  document.getElementById('filter-smart').style.borderColor='#4488cc';

  document.getElementById("st-left").textContent="⏳ Loading priorities...";
  try {
    const r=await(await fetch("/api/board_priorities")).json();
    smartPriorities={};
    r.forEach((item,i)=>{smartPriorities[item.board]={...item,rank:i};});
    applyFilter();
    document.getElementById("st-left").textContent=`🎯 Smart sort: ${r.length} boards ranked by uncertainty`;
  } catch(e) {
    document.getElementById("st-left").textContent="❌ Failed to load priorities";
  }
}

function applyFilter(){
  const q=document.getElementById("search").value.toLowerCase();
  const info={};boardsInfo.forEach(b=>info[b.name]=b.reviewed);

  if(boardFilter==='smart' && smartPriorities){
    filtered=boards.filter(b=>{
      if(q&&!b.toLowerCase().includes(q))return false;
      return true;
    });
    filtered.sort((a,b)=>(smartPriorities[a]?.rank||9999)-(smartPriorities[b]?.rank||9999));
  } else {
    filtered=boards.filter(b=>{
      if(q&&!b.toLowerCase().includes(q))return false;
      if(boardFilter==='todo'&&info[b])return false;
      if(boardFilter==='done'&&!info[b])return false;
      if(boardFilter==='excluded'&&!exclSet.has(b))return false;
      return true;
    });
  }

  const todoCount=boards.filter(b=>!info[b]).length;
  const doneCount=boards.length-todoCount;
  document.getElementById("bcount").textContent=`(${filtered.length}/${boards.length})`;
  document.getElementById("filter-todo").textContent=`⏳ ${todoCount}`;
  document.getElementById("filter-done").textContent=`✅ ${doneCount}`;
  renderList();
}

function renderList(){
  const info={};boardsInfo.forEach(b=>info[b.name]=b.reviewed);
  document.getElementById("board-list").innerHTML=filtered.map(b=>{
    let badge='';
    if(boardFilter==='smart' && smartPriorities && smartPriorities[b]){
      const p=smartPriorities[b];
      badge=`<span style="float:right;font-size:10px;color:#e94560" title="Low conf: ${p.n_low_conf}, Confused: ${p.n_confused}">⚠${p.score}</span>`;
    }
    return `<div class="bitem ${exclSet.has(b)?"excluded ":""}${b===curBoard?'active':''}" onclick="loadBoard('${b}')" title="${b}">
      ${exclSet.has(b)?'<span style="color:#e74c3c;font-size:10px">✗ </span>':info[b]?'<span style="color:#2ecc71;font-size:10px">✓ </span>':''}${b}${badge}
    </div>`;
  }).join("");
}

function addNewCategory(){
  const name=prompt("Enter new category name:");
  if(!name||!name.trim())return;
  const trimmed=name.trim();
  // Check if already exists
  for(const[k,v] of Object.entries(CAT_NAMES)){if(v.toLowerCase()===trimmed.toLowerCase()){alert("Category already exists: "+v);return;}}
  // Find next ID
  const ids=Object.keys(CAT_NAMES).map(Number);
  const newId=Math.max(...ids)+1;
  // Add to server
  fetch("/api/add_category",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:newId,name:trimmed})})
  .then(r=>r.json()).then(d=>{
    if(d.ok){
      CAT_NAMES[newId]=trimmed;
      CAT_SHORT[newId]=trimmed.slice(0,3);
      if(!CAT_COLORS[newId])CAT_COLORS[newId]=`hsl(${(newId*47)%360},70%,60%)`;
      buildCatBar();
    } else alert("Error: "+d.error);
  });
}
// Explicit hotkey↔category mapping (LED merged into Diode, keys 6-0 unchanged)
const KEY_TO_CAT={"1":1,"2":2,"3":3,"4":4,"5":5,"7":7,"8":8,"9":9,"0":10};
const CAT_TO_KEY={};Object.entries(KEY_TO_CAT).forEach(([k,v])=>CAT_TO_KEY[v]=k);
// Extra keys for future categories
const EXTRA_KEYS="QWERTY";
function catHotkey(id){
  if(CAT_TO_KEY[id])return CAT_TO_KEY[id];
  // Assign extra keys for categories not in explicit map
  const mapped=new Set(Object.values(KEY_TO_CAT));
  const unmapped=Object.keys(CAT_NAMES).map(Number).filter(x=>!mapped.has(x)).sort((a,b)=>a-b);
  const idx=unmapped.indexOf(id);
  if(idx>=0&&idx<EXTRA_KEYS.length)return EXTRA_KEYS[idx];
  return"";
}
function buildCatBar(){
  const ids=Object.keys(CAT_NAMES).map(Number).sort((a,b)=>a-b);
  document.getElementById("cat-bar").innerHTML=ids.map(id=>
    `<div class="cat-btn ${id===activeCat?'active':''}" style="color:${CAT_COLORS[id]||'#ccc'};border-color:${id===activeCat?(CAT_COLORS[id]||'#ccc'):'#444'}" onclick="setCat(${id})"><kbd>${catHotkey(id)}</kbd>${CAT_NAMES[id]}</div>`
  ).join("")+`<div class="cat-btn" style="color:#8f8;border-color:#444" onclick="addNewCategory()"><kbd>+</kbd>New</div>`;
}

function setCat(c){
  activeCat=c; buildCatBar();
  // Apply to multi-selection
  if(multiSel.size>0){
    let changed=0;
    for(const a of anns){ if(multiSel.has(a.id)&&a.category_id!==c){ a.category_id=c; changed++; } }
    if(changed>0){ dirty=true; document.getElementById("st-left").textContent=`✅ Changed ${changed} annotations to ${CAT_NAMES[c]||c}`; }
    render();renderAnnList();updateStatus();return;
  }
  // Apply to single selection
  if(selId!==null){
    const a=anns.find(x=>x.id===selId);
    if(a&&a.category_id!==c){ a.category_id=c; dirty=true; }
    render();renderAnnList();updateStatus();
  }
}

// ── Load board ──
async function loadBoard(name){
  if(dirty&&curBoard){ if(!confirm("Unsaved changes! Discard?")) return; }
  const r=await(await fetch(`/api/board/${name}`)).json();
  if(r.error){alert(r.error);return;}
  curBoard=name; curIdx=r.index; curSource=r.source||"original";
  anns=r.annotations; imgW=r.image_width; imgH=r.image_height;
  origAnns={};
  anns.forEach(a=>origAnns[a.id]={category_id:a.category_id,bbox:[...a.bbox]});
  selId=null; dirty=false; zoomLevel=1;
  document.getElementById("bname").textContent=name;
  const srcLabel=r.source==="predicted"?"🤖 YOLO":r.source==="manual"?"✅ Reviewed":"📋 Original";
  document.getElementById("binfo").textContent=`${imgW}×${imgH} | ${anns.length} ann | ${srcLabel} | #${curIdx+1}/${r.total}`;
  boardImg=new Image();
  boardImg.onload=()=>{fitCanvas();render();};
  boardImg.src=`/api/image/${name}`;
  renderList(); renderAnnList(); updateStatus(); updateExclBtn();
}

function fitCanvas(){
  const w=document.getElementById("canvas-wrap");
  const mW=w.clientWidth-20, mH=w.clientHeight-20;
  dispScale=Math.min(mW/imgW, mH/imgH, 3);
  applyZoom();
}
function applyZoom(){
  const dpr=window.devicePixelRatio||1;
  const ds=dispScale*zoomLevel;
  const renderScale=Math.max(ds*dpr,1);
  cv.width=imgW*renderScale; cv.height=imgH*renderScale;
  cv.style.width=(imgW*ds)+"px"; cv.style.height=(imgH*ds)+"px";
  scale=renderScale;
}

// ── Render ──
function render(){
  if(!boardImg)return;
  cx.clearRect(0,0,cv.width,cv.height);
  cx.drawImage(boardImg,0,0,cv.width,cv.height);

  for(const a of anns){
    const [bx,by,bw,bh]=a.bbox;
    const sx=bx*scale, sy=by*scale, sw=bw*scale, sh=bh*scale;
    const isSel=a.id===selId;
    let col;
    if(isSel) col="rgba(0,255,100,1)";
    else col=CAT_COLORS[a.category_id]||"rgba(0,120,255,1)";

    const lowConf = a.score!=null && a.score<0.6;
    cx.strokeStyle=col; cx.lineWidth=isSel?4:3;
    if(lowConf) cx.setLineDash([6,6]);
    cx.strokeRect(sx,sy,sw,sh);
    if(lowConf) cx.setLineDash([]);
    // Confidence label for low conf
    if(lowConf){
      const confTxt=a.score.toFixed(2);
      cx.font="bold 10px monospace";
      const tw=cx.measureText(confTxt).width;
      cx.fillStyle="rgba(0,0,0,0.7)"; cx.fillRect(sx,sy-12,tw+4,13);
      cx.fillStyle="#ff6600"; cx.fillText(confTxt,sx+2,sy-2);
    }

    // Reclassified: orange inner border
    if(a.reclassified_to){
      cx.strokeStyle="rgba(255,165,0,1)"; cx.lineWidth=2;
      cx.setLineDash([4,3]); cx.strokeRect(sx+3,sy+3,sw-6,sh-6); cx.setLineDash([]);
    }

    // Orientation arrow
    if(a.orientation!==null&&a.orientation!==undefined){
      const deg=a.orientation;
      const rad=-(deg-90)*Math.PI/180; // convert: 0°=E→right, 90°=N→up
      const acx=sx+sw/2, acy=sy+sh/2;
      const alen=Math.min(sw,sh)*0.3;
      const ax2=acx+Math.cos(rad)*alen, ay2=acy-Math.sin(rad)*alen;
      cx.strokeStyle="rgba(255,255,0,0.9)"; cx.lineWidth=2;
      cx.beginPath(); cx.moveTo(acx,acy); cx.lineTo(ax2,ay2); cx.stroke();
      // Arrowhead
      const ha=0.4, hl=alen*0.35;
      cx.beginPath();
      cx.moveTo(ax2,ay2);
      cx.lineTo(ax2-hl*Math.cos(rad-ha),ay2+hl*Math.sin(rad-ha));
      cx.moveTo(ax2,ay2);
      cx.lineTo(ax2-hl*Math.cos(rad+ha),ay2+hl*Math.sin(rad+ha));
      cx.stroke();
    }

    if(isSel){
      cx.fillStyle="#2ecc71";
      for(const h of Object.values(handles(sx,sy,sw,sh)))
        cx.fillRect(h.x-HS/2,h.y-HS/2,HS,HS);
    }
  }

  // Draw preview
  if(drawing&&drawStart&&drawCur){
    const x1=Math.min(drawStart.x,drawCur.x), y1=Math.min(drawStart.y,drawCur.y);
    const w1=Math.abs(drawCur.x-drawStart.x), h1=Math.abs(drawCur.y-drawStart.y);
    cx.strokeStyle="rgba(241,196,15,0.9)"; cx.lineWidth=2;
    cx.setLineDash([5,5]); cx.strokeRect(x1,y1,w1,h1); cx.setLineDash([]);
  }

  // Draw multi-selection highlights
  if(multiSel.size>0){
    cx.save();
    for(const a of anns){
      if(!multiSel.has(a.id)) continue;
      const [bx,by,bw,bh]=a.bbox;
      cx.strokeStyle="#00ff88"; cx.lineWidth=3; cx.setLineDash([6,3]);
      cx.strokeRect(bx*scale,by*scale,bw*scale,bh*scale);
    }
    cx.setLineDash([]); cx.restore();
  }
  // Draw box selection rectangle
  if(boxSelecting&&boxSelStart&&boxSelCur){
    cx.save(); cx.strokeStyle="#00ffff"; cx.lineWidth=2; cx.setLineDash([8,4]);
    cx.fillStyle="rgba(0,255,255,0.08)";
    const bx=Math.min(boxSelStart.x,boxSelCur.x), by=Math.min(boxSelStart.y,boxSelCur.y);
    const bw=Math.abs(boxSelCur.x-boxSelStart.x), bh=Math.abs(boxSelCur.y-boxSelStart.y);
    cx.fillRect(bx,by,bw,bh); cx.strokeRect(bx,by,bw,bh);
    cx.setLineDash([]); cx.restore();
  }
}


function handles(sx,sy,sw,sh){
  return {nw:{x:sx,y:sy},ne:{x:sx+sw,y:sy},sw:{x:sx,y:sy+sh},se:{x:sx+sw,y:sy+sh},
          n:{x:sx+sw/2,y:sy},s:{x:sx+sw/2,y:sy+sh},w:{x:sx,y:sy+sh/2},e:{x:sx+sw,y:sy+sh/2}};
}

function renderAnnList(){
  document.getElementById("ann-count").textContent=`(${anns.length})`;
  document.getElementById("ann-list").innerHTML=anns.map(a=>{
    const cat=CAT_SHORT[a.category_id]||"?";
    const mod=isModified(a);
    const col=a.is_flagged?"#f0f":a.is_predicted?"#0cf":a.is_excluded?"#e74c3c":"#3498db";
    const modTag="";
    const rc=a.reclassified_to?` → <span style="color:#f39c12">${a.reclassified_to}</span>`:"";
    const ori=a.orientation!==null&&a.orientation!==undefined?` ↻${a.orientation}°`:"";
    return `<div class="ann-item ${a.id===selId?'sel':''}" onclick="selectAnn(${a.id})">
      <span><span class="tag" style="background:${col}">${cat}</span> #${a.id}${modTag}${rc}<span style="color:#ff0;font-size:10px">${ori}</span></span>
      <span style="font-size:10px;color:#666">${Math.round(a.bbox[2])}×${Math.round(a.bbox[3])}</span>
    </div>`;
  }).join("");
}

function isModified(a){
  const o=origAnns[a.id];
  if(!o) return "added";
  if(o.category_id!==a.category_id) return "recat";
  return null;
}

function selectAnn(id){
  selId=id;
  // Highlight category in sidebar
  const a=anns.find(x=>x.id===id);
  if(a){activeCat=a.category_id;buildCatBar();}
  render();renderAnnList();updateStatus();
}

// ── Mouse ──
function cp(e){
  const r=cv.getBoundingClientRect();
  // Convert screen coords to canvas pixel coords (account for CSS scaling)
  const cssX=e.clientX-r.left, cssY=e.clientY-r.top;
  const canvasX=cssX*(cv.width/r.width), canvasY=cssY*(cv.height/r.height);
  return{x:canvasX,y:canvasY};
}
function ip(p){return{x:p.x/scale,y:p.y/scale};}

// Hit test: find annotation under point (image coords)
function hitTestAnn(ix,iy){
  for(let j=anns.length-1;j>=0;j--){
    const [bx,by,bw,bh]=anns[j].bbox;
    if(ix>=bx&&ix<=bx+bw&&iy>=by&&iy<=by+bh) return anns[j];
  }
  return null;
}

// Auto cursor on mouse move
cv.addEventListener("mousemove",e=>{
  if(dragging||drawing||spaceHeld)return;
  const p=cp(e), i=ip(p);
  // Check resize handles first
  if(selId!==null){
    const a=anns.find(x=>x.id===selId);
    if(a){
      const hs=handles(a.bbox[0]*scale,a.bbox[1]*scale,a.bbox[2]*scale,a.bbox[3]*scale);
      for(const k in hs){
        if(Math.abs(p.x-hs[k].x)<HS+3&&Math.abs(p.y-hs[k].y)<HS+3){
          cv.style.cursor={n:"ns-resize",s:"ns-resize",e:"ew-resize",w:"ew-resize",nw:"nwse-resize",se:"nwse-resize",ne:"nesw-resize",sw:"nesw-resize"}[k]||"move";
          return;
        }
      }
    }
  }
  const hit=hitTestAnn(i.x,i.y);
  cv.style.cursor=hit?"move":"crosshair";
});

let boxSelecting=false, boxSelStart=null, boxSelCur=null;
cv.addEventListener("mousedown",e=>{
  if(e.button!==0||spaceHeld)return;
  const p=cp(e), i=ip(p);

  // Shift+click → box selection mode
  if(e.shiftKey){
    boxSelecting=true; boxSelStart=p; boxSelCur=p;
    return;
  }

  // Check resize handles first (single select only)
  if(selId!==null&&multiSel.size===0){
    const a=anns.find(x=>x.id===selId);
    if(a){
      const hs=handles(a.bbox[0]*scale,a.bbox[1]*scale,a.bbox[2]*scale,a.bbox[3]*scale);
      for(const k in hs){
        if(Math.abs(p.x-hs[k].x)<HS+3&&Math.abs(p.y-hs[k].y)<HS+3){
          dragging=true;dragType=k;dragStart=p;dragOrig=[...a.bbox];return;
        }
      }
    }
  }

  // Auto-detect: click on annotation → select/move, click on empty → draw
  const clicked=hitTestAnn(i.x,i.y);
  if(clicked){
    selId=clicked.id; multiSel.clear();
    activeCat=clicked.category_id; buildCatBar();
    dragging=true;dragType="move";dragStart=p;dragOrig=[...clicked.bbox];
  } else {
    // Draw mode — start drawing new bbox
    selId=null; multiSel.clear();
    drawing=true;drawStart=p;drawCur=p;
  }
  render();renderAnnList();updateStatus();
});

cv.addEventListener("mousemove",e=>{
  const p=cp(e);
  if(boxSelecting){boxSelCur=p;render();return;}
  if(drawing){drawCur=p;render();return;}
  if(dragging&&selId!==null){
    const a=anns.find(x=>x.id===selId);if(!a)return;
    const dx=(p.x-dragStart.x)/scale, dy=(p.y-dragStart.y)/scale;
    const [ox,oy,ow,oh]=dragOrig;
    if(dragType==="move") a.bbox=[ox+dx,oy+dy,ow,oh];
    else if(dragType==="se") a.bbox=[ox,oy,Math.max(5,ow+dx),Math.max(5,oh+dy)];
    else if(dragType==="nw") a.bbox=[ox+dx,oy+dy,Math.max(5,ow-dx),Math.max(5,oh-dy)];
    else if(dragType==="ne") a.bbox=[ox,oy+dy,Math.max(5,ow+dx),Math.max(5,oh-dy)];
    else if(dragType==="sw") a.bbox=[ox+dx,oy,Math.max(5,ow-dx),Math.max(5,oh+dy)];
    else if(dragType==="n") a.bbox=[ox,oy+dy,ow,Math.max(5,oh-dy)];
    else if(dragType==="s") a.bbox=[ox,oy,ow,Math.max(5,oh+dy)];
    else if(dragType==="w") a.bbox=[ox+dx,oy,Math.max(5,ow-dx),oh];
    else if(dragType==="e") a.bbox=[ox,oy,ow+dx>5?ow+dx:5,oh];
    dirty=true;render();
  }
});

cv.addEventListener("mouseup",e=>{
  // Box selection finalize
  if(boxSelecting){
    boxSelecting=false;
    const p=cp(e);
    const sx1=Math.min(boxSelStart.x,p.x)/scale, sy1=Math.min(boxSelStart.y,p.y)/scale;
    const sx2=Math.max(boxSelStart.x,p.x)/scale, sy2=Math.max(boxSelStart.y,p.y)/scale;
    if(Math.abs(p.x-boxSelStart.x)>5&&Math.abs(p.y-boxSelStart.y)>5){
      multiSel.clear(); selId=null;
      for(const a of anns){
        const [ax,ay,aw,ah]=a.bbox;
        const bx1=ax, by1=ay, bx2=ax+aw, by2=ay+ah;
        if(bx1>=sx1&&by1>=sy1&&bx2<=sx2&&by2<=sy2) multiSel.add(a.id);
      }
      if(multiSel.size>0) document.getElementById("st-left").textContent=`🔲 ${multiSel.size} selected — press 1-9 to change class, X to delete`;
    }
    boxSelStart=boxSelCur=null;
    render();renderAnnList();updateStatus();return;
  }
  if(drawing){
    drawing=false;
    const p=cp(e);
    const x1=Math.min(drawStart.x,p.x)/scale, y1=Math.min(drawStart.y,p.y)/scale;
    const w1=Math.abs(p.x-drawStart.x)/scale, h1=Math.abs(p.y-drawStart.y)/scale;
    if(w1>3&&h1>3){
      const nid=anns.length?Math.max(...anns.map(a=>a.id))+1:1;
      anns.push({id:nid,category_id:activeCat,bbox:[x1,y1,w1,h1],is_excluded:false});
      selId=nid; multiSel.clear(); dirty=true;
      render();renderAnnList();updateStatus();
    }
    drawStart=drawCur=null;return;
  }
  if(dragging){dragging=false;dragType=null;}
});

// ── Keyboard ──
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT")return;
  // Hotkeys: explicit map + extra keys
  const hk=e.key;
  if(KEY_TO_CAT[hk]&&CAT_NAMES[KEY_TO_CAT[hk]]){setCat(KEY_TO_CAT[hk]);return;}
  const eki=EXTRA_KEYS.indexOf(hk.toUpperCase());
  if(eki>=0){
    const mapped=new Set(Object.values(KEY_TO_CAT));
    const unmapped=Object.keys(CAT_NAMES).map(Number).filter(x=>!mapped.has(x)).sort((a,b)=>a-b);
    if(eki<unmapped.length){setCat(unmapped[eki]);return;}
  }
  if(e.key==="Backspace"||e.key==="Delete"||e.key==="x"||e.key==="X"||e.key==="f"||e.key==="F"){e.preventDefault();delSelected();return;}
  // N key removed — mode is auto
  if(e.key==="s"&&(e.ctrlKey||e.metaKey)){e.preventDefault();saveBoard();return;}
  if(e.key==="r"||e.key==="R"){resetZoom();return;}
  if(e.key==="ArrowLeft"||e.key==="a"||e.key==="A"){navBoard(-1);return;}
  if(e.key==="ArrowRight"||e.key==="d"||e.key==="D"){navBoard(1);return;}
  if(e.key==="Escape"){selId=null;multiSel.clear();
    render();renderAnnList();updateStatus();return;}
});

// Mode is auto — no toggle needed

function delSelected(){
  if(multiSel.size>0){
    const count=multiSel.size;
    anns=anns.filter(a=>!multiSel.has(a.id));
    multiSel.clear(); selId=null; dirty=true;
    document.getElementById("st-left").textContent=`🗑️ Deleted ${count} annotations`;
    render();renderAnnList();updateStatus();return;
  }
  if(selId===null)return;
  anns=anns.filter(a=>a.id!==selId);
  selId=null;dirty=true;render();renderAnnList();updateStatus();
}

async function saveBoard(){
  if(!curBoard)return;

  const r=await(await fetch(`/api/save/${curBoard}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({annotations:anns})})).json();
  if(r.ok){
    dirty=false;
    let msg=`✅ Saved ${r.count} annotations`;
    if(r.corrections>0) msg+=` | 📝 ${r.corrections} corrections logged`;
    document.getElementById("st-left").textContent=msg;
    // Mark board as reviewed in local state
    const bi=boardsInfo.find(b=>b.name===curBoard);
    if(bi) bi.reviewed=true;
    applyFilter();
  }
  else alert("Save failed: "+(r.error||"?"));
}

function navBoard(dir){
  if(dirty&&curBoard){if(!confirm("Unsaved changes! Discard?"))return;}
  const list=filtered.length?filtered:boards;
  const idx=list.indexOf(curBoard);
  const next=idx+dir;
  if(next>=0&&next<list.length) loadBoard(list[next]);
}

let exclSet=new Set();
async function loadExcluded(){
  try{const r=await(await fetch("/api/excluded_boards")).json();exclSet=new Set(r.boards||[]);updateExclBtn();}catch(e){}
}
function updateExclBtn(){
  const btn=document.getElementById("excl-board-btn");
  if(!btn||!curBoard)return;
  const ex=exclSet.has(curBoard);
  btn.textContent=ex?"\u2705 Include Board":"\u{1F6AB} Exclude Board";
  btn.style.background=ex?"#27ae60":"#7f8c8d";
  btn.style.borderColor=ex?"#27ae60":"#7f8c8d";
}
async function toggleExcludeBoard(){
  if(!curBoard)return;
  const ex=exclSet.has(curBoard);
  const r=await(await fetch("/api/exclude_board",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:curBoard,action:ex?"include":"exclude"})})).json();
  if(r.ok){
    if(r.excluded)exclSet.add(curBoard);else exclSet.delete(curBoard);
    updateExclBtn(); renderList();
    document.getElementById("st-left").textContent=r.excluded?"\u{1F6AB} Board excluded ("+r.total_excluded+" total)":"\u2705 Board included";
  }
}
function updateStatus(){
  const m=selId!==null?"🖱 SELECT":"🖊 AUTO";
  const s=selId!==null?` | Sel #${selId}`:"";
  const d=dirty?" | ⚠ Unsaved":"";
  const z=zoomLevel!==1?` | 🔍 ${Math.round(zoomLevel*100)}%`:"";
  document.getElementById("st-left").textContent=`${m}${s}${d}${z} | ${anns.length} bboxes`;
}

document.getElementById("search").addEventListener("input",e=>{
  applyFilter();
});

// Scroll zoom (centered on cursor)
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  e.stopPropagation();
  const wrap=document.getElementById("canvas-wrap");
  const rect=cv.getBoundingClientRect();
  // Cursor position relative to image
  const mx=(e.clientX-rect.left)/rect.width;
  const my=(e.clientY-rect.top)/rect.height;
  const oldW=parseFloat(cv.style.width), oldH=parseFloat(cv.style.height);

  const oldZoom=zoomLevel;
  if(e.deltaY<0) zoomLevel=Math.min(zoomLevel*1.2, 10);
  else zoomLevel=Math.max(zoomLevel/1.2, 0.5);
  applyZoom(); render();

  const newW=parseFloat(cv.style.width), newH=parseFloat(cv.style.height);
  // Adjust scroll to keep cursor at same image point
  wrap.scrollLeft+=mx*(newW-oldW);
  wrap.scrollTop+=my*(newH-oldH);
  updateStatus();
},{passive:false});

// Middle-click OR Space+left-click pan
cv.addEventListener("mousedown",e=>{
  if(e.button===1||(e.button===0&&spaceHeld)){
    e.preventDefault();panning=true;panStart={x:e.clientX,y:e.clientY};
    const wrap=document.getElementById("canvas-wrap");
    panStart.scrollX=wrap.scrollLeft; panStart.scrollY=wrap.scrollTop;
  }
});
document.addEventListener("mousemove",e=>{
  if(panning&&panStart){
    e.preventDefault();
    const wrap=document.getElementById("canvas-wrap");
    wrap.scrollLeft=panStart.scrollX-(e.clientX-panStart.x);
    wrap.scrollTop=panStart.scrollY-(e.clientY-panStart.y);
  }
});
document.addEventListener("mouseup",e=>{if(panning){panning=false;e.preventDefault();}if(boxSelecting){boxSelecting=false;boxSelStart=boxSelCur=null;}});
// Prevent default middle-click auto-scroll
cv.addEventListener("auxclick",e=>{if(e.button===1)e.preventDefault();});

// Right-click on bbox → delete it (always prevent browser context menu on canvas)
document.getElementById("canvas-wrap").addEventListener("contextmenu",function(e){
  e.preventDefault();
  e.stopPropagation();
  e.stopImmediatePropagation();
  // If click was on the canvas, try to delete bbox
  if(e.target===cv){
    const p=cp(e), i=ip(p);
    const clicked=hitTestAnn(i.x,i.y);
    if(clicked){
      anns=anns.filter(a=>a.id!==clicked.id);
      if(selId===clicked.id) selId=null;
      multiSel.delete(clicked.id);
      dirty=true;
      render(); renderAnnList(); updateStatus();
    }
  }
  return false;
}, true);

let spaceHeld=false;
document.addEventListener("keydown",e=>{if(e.code==="Space"&&e.target.tagName!=="INPUT"){e.preventDefault();spaceHeld=true;cv.style.cursor="grab";}});
document.addEventListener("keyup",e=>{if(e.code==="Space"){spaceHeld=false;cv.style.cursor="crosshair";}});

// Reset zoom
function resetZoom(){zoomLevel=1;applyZoom();render();updateStatus();}

let curVersion="pipeline";
const VER_ORDER=["pipeline","v6","original"];
const VER_LABELS={"pipeline":"Pipeline","v6":"v6-backup","original":"Original"};
const VER_COLORS={"pipeline":"#27ae60","v6":"#e74c3c","original":"#f39c12"};
async function toggleVersion(){
  const idx=VER_ORDER.indexOf(curVersion);
  const newVer=VER_ORDER[(idx+1)%VER_ORDER.length];
  const r=await(await fetch("/api/predict_version",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({version:newVer})})).json();
  if(r.ok){
    curVersion=newVer;
    updateVerBtn();
    if(curBoard) loadBoard(curBoard);
  }
}
function updateVerBtn(){
  const btn=document.getElementById("ver-btn");
  btn.textContent="🔀 "+(VER_LABELS[curVersion]||curVersion);
  btn.style.background=VER_COLORS[curVersion]||"#444";
  btn.style.borderColor=VER_COLORS[curVersion]||"#444";
}
async function initVersion(){
  const r=await(await fetch("/api/predict_version")).json();
  curVersion=r.version||"pipeline";
  updateVerBtn();
}

// ── Collapsible panels ──
function toggleSidebar(){
  const sb=document.getElementById("sidebar");
  const btn=document.getElementById("toggle-sidebar");
  const collapsed=sb.classList.toggle("collapsed");
  btn.textContent=collapsed?"▶":"◀";
  btn.style.left=collapsed?"0":"280px";
  setTimeout(()=>{if(boardImg){fitCanvas();render();}},220);
}
function toggleAnn(){
  const ap=document.getElementById("ann-panel");
  const btn=document.getElementById("toggle-ann");
  const collapsed=ap.classList.toggle("collapsed");
  btn.textContent=collapsed?"◀":"▶";
  btn.style.right=collapsed?"0":"220px";
  setTimeout(()=>{if(boardImg){fitCanvas();render();}},220);
}
// Keyboard shortcut: \ = sidebar, / = ann panel
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT")return;
  if(e.key==="\\"){ toggleSidebar(); return; }
  if(e.key==="/"){ e.preventDefault(); toggleAnn(); return; }
});
// Set initial button positions
document.getElementById("toggle-sidebar").style.left="280px";
document.getElementById("toggle-ann").style.right="220px";

window.addEventListener("resize",()=>{if(boardImg){fitCanvas();render();}});
async function startApp(){await initVersion();
loadExcluded();await init();}
startApp();
</script>
</body>
</html>
"""


# ── Class Review HTML ──────────────────────────────────────────────────

CLASS_REVIEW_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB Class Review</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

.topbar { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #16213e; border-bottom: 1px solid #0f3460; flex-wrap: wrap; }
.topbar h2 { font-size: 16px; color: #e94560; }
.topbar a { color: #4488ff; text-decoration: none; font-size: 13px; }
.topbar a:hover { text-decoration: underline; }
.topbar select, .topbar button { padding: 5px 10px; background: #1a1a2e; border: 1px solid #444; color: #eee; border-radius: 4px; font-size: 13px; cursor: pointer; }
.topbar select:hover, .topbar button:hover { border-color: #888; }
.topbar .nav-btn { background: #2980b9; border-color: #2980b9; }
.topbar .nav-btn:disabled { opacity: 0.4; cursor: default; }

.statsbar { padding: 6px 16px; background: #111; font-size: 13px; color: #aaa; border-bottom: 1px solid #333; display: flex; gap: 20px; }

.grid-wrap { flex: 1; overflow-y: auto; padding: 12px; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; }

.card { width: 120px; border: 2px solid #444; border-radius: 6px; background: #16213e; cursor: pointer; overflow: hidden; position: relative; transition: border-color 0.15s; }
.card:hover { border-color: #888; }
.card.selected { border-color: #4488ff !important; box-shadow: 0 0 8px rgba(68,136,255,0.5); }
.card.low-conf { border-color: #ff8800; }
.card.reclassified::after { content: "✓"; position: absolute; top: 4px; right: 4px; background: #27ae60; color: #fff; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; }
.card img { width: 120px; height: 100px; object-fit: cover; display: block; background: #222; }
.card .info { padding: 3px 6px; font-size: 10px; }
.card .conf { font-weight: bold; }
.card .bname { color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
.card .new-class { color: #2ecc71; font-weight: bold; font-size: 11px; }

.bottom-bar { padding: 8px 16px; background: #16213e; border-top: 1px solid #0f3460; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.cat-hotkey { padding: 4px 10px; border: 2px solid #444; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; background: transparent; }
.cat-hotkey:hover { opacity: 0.8; }
.cat-hotkey kbd { background: #333; padding: 1px 5px; border-radius: 2px; font-size: 11px; margin-right: 3px; }
.apply-btn { padding: 6px 16px; background: #27ae60; border: none; color: #fff; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: bold; }
.apply-btn:hover { background: #2ecc71; }
.apply-btn:disabled { opacity: 0.4; cursor: default; }

.status-msg { font-size: 12px; color: #aaa; margin-left: auto; }
</style>
</head>
<body>

<div class="topbar">
  <h2>🔬 Class Review</h2>
  <a href="/">← Back to Annotator</a>
  <span style="color:#666">|</span>
  <label style="font-size:13px">Class:</label>
  <select id="class-sel"></select>
  <label style="font-size:13px">Filter:</label>
  <select id="filter-sel">
    <option value="all">All</option>
    <option value="low">Low Confidence (&lt;0.5)</option>
    <option value="confused">Confused (&lt;0.7)</option>
  </select>
  <span style="color:#666">|</span>
  <button class="nav-btn" id="prev-page" onclick="changePage(-1)">◀ Prev</button>
  <span id="page-info" style="font-size:13px">Page 1</span>
  <button class="nav-btn" id="next-page" onclick="changePage(1)">Next ▶</button>
</div>

<div class="statsbar">
  <span id="stats-text">Loading...</span>
  <span id="sel-text"></span>
</div>

<div class="grid-wrap" id="grid-wrap">
  <div class="grid" id="grid"></div>
</div>

<div class="bottom-bar">
  <span style="font-size:12px;color:#aaa">Reclassify selected:</span>
  <div id="hotkey-bar"></div>
  <button class="apply-btn" id="apply-btn" onclick="applyChanges()" disabled>Apply Changes</button>
  <span class="status-msg" id="status-msg">Select cards, then press 1-9/0 or click a class to reclassify</span>
</div>

<script>
const CAT_NAMES = {1:"Resistor",2:"Capacitor",3:"Inductor",4:"Connector",5:"Diode",7:"Switch",8:"Transistor",9:"Integrated Circuit",10:"Oscillator"};
const CAT_SHORT = {1:"Res",2:"Cap",3:"Ind",4:"Con",5:"Dio",7:"Swi",8:"Tra",9:"IC",10:"Osc"};
const CAT_COLORS = {1:"#ff4444",2:"#9b59b6",3:"#00dddd",4:"#00dd00",5:"#ffffff",7:"#ff8800",8:"#ffdd00",9:"#4488ff",10:"#88ff88"};
const KEY_TO_CAT = {"1":1,"2":2,"3":3,"4":4,"5":5,"7":7,"8":8,"9":9,"0":10};

let allItems=[], displayItems=[], selectedSet=new Set(), reclassMap={};
let currentCat=1, currentPage=1, totalPages=1, totalItems=0;
let lastClickIdx=-1;
const PER_PAGE=200;

function initUI(){
  // Build class dropdown
  const sel=document.getElementById("class-sel");
  const catIds=Object.keys(CAT_NAMES).map(Number).sort((a,b)=>a-b);
  sel.innerHTML=catIds.map(id=>`<option value="${id}">${CAT_NAMES[id]} (${id})</option>`).join("");
  sel.value=currentCat;
  sel.onchange=()=>{currentCat=parseInt(sel.value);currentPage=1;loadCrops();};

  document.getElementById("filter-sel").onchange=()=>{currentPage=1;loadCrops();};

  // Build hotkey bar
  const bar=document.getElementById("hotkey-bar");
  const hotkeys=[["1",1],["2",2],["3",3],["4",4],["5",5],["7",7],["8",8],["9",9],["0",10]];
  bar.innerHTML=hotkeys.map(([k,id])=>
    `<div class="cat-hotkey" style="color:${CAT_COLORS[id]||'#ccc'}" onclick="reclassifySelected(${id})"><kbd>${k}</kbd>${CAT_SHORT[id]||'?'}</div>`
  ).join("");

  loadCrops();
}

async function loadCrops(){
  const filter=document.getElementById("filter-sel").value;
  document.getElementById("stats-text").textContent="Loading...";
  document.getElementById("grid").innerHTML='<div style="padding:40px;color:#888">Loading crops...</div>';

  try{
    const r=await(await fetch(`/api/class_crops/${currentCat}?page=${currentPage}&per_page=${PER_PAGE}`)).json();
    allItems=r.items;
    totalItems=r.total;
    totalPages=r.total_pages;
    currentPage=r.page;

    // Apply client-side filter
    if(filter==='low') displayItems=allItems.filter(i=>(i.conf||1)<0.5);
    else if(filter==='confused') displayItems=allItems.filter(i=>(i.conf||1)<0.7);
    else displayItems=[...allItems];

    selectedSet.clear();
    reclassMap={};
    lastClickIdx=-1;
    updateApplyBtn();
    renderGrid();
    updateStats();
    updatePageNav();
  }catch(e){
    document.getElementById("grid").innerHTML=`<div style="padding:40px;color:#e94560">Error: ${e.message}</div>`;
  }
}

function renderGrid(){
  const grid=document.getElementById("grid");
  grid.innerHTML=displayItems.map((item,idx)=>{
    const key=`${item.board}__${item.ann_id}`;
    const isLow=(item.conf||1)<0.5;
    const isSel=selectedSet.has(key);
    const reclass=reclassMap[key];
    const cls=['card'];
    if(isSel) cls.push('selected');
    if(isLow) cls.push('low-conf');
    if(reclass!==undefined) cls.push('reclassified');

    const confStr=item.conf!=null?item.conf.toFixed(2):'?';
    const confColor=(item.conf||1)<0.5?'#ff8800':(item.conf||1)<0.7?'#ffdd00':'#2ecc71';
    const newClassHtml=reclass!==undefined?`<span class="new-class">→ ${CAT_SHORT[reclass]||reclass}</span>`:'';

    return `<div class="${cls.join(' ')}" data-idx="${idx}" data-key="${key}" onclick="cardClick(event,${idx})">
      <img loading="lazy" src="/api/class_crop_image/${item.board}/${item.ann_id}" alt="crop">
      <div class="info">
        <span class="conf" style="color:${confColor}">${confStr}</span> ${newClassHtml}
        <span class="bname">${item.board}</span>
      </div>
    </div>`;
  }).join("");
}

function cardClick(e,idx){
  const item=displayItems[idx];
  if(!item)return;
  const key=`${item.board}__${item.ann_id}`;

  if(e.shiftKey && lastClickIdx>=0){
    // Range select
    const start=Math.min(lastClickIdx,idx), end=Math.max(lastClickIdx,idx);
    for(let i=start;i<=end;i++){
      const k=`${displayItems[i].board}__${displayItems[i].ann_id}`;
      selectedSet.add(k);
    }
  } else {
    if(selectedSet.has(key)) selectedSet.delete(key);
    else selectedSet.add(key);
  }
  lastClickIdx=idx;
  renderGrid();
  updateSelText();
}

function selectAllVisible(){
  displayItems.forEach(item=>{
    selectedSet.add(`${item.board}__${item.ann_id}`);
  });
  renderGrid();
  updateSelText();
}

function reclassifySelected(newCat){
  if(selectedSet.size===0)return;
  let count=0;
  for(const key of selectedSet){
    reclassMap[key]=newCat;
    count++;
  }
  document.getElementById("status-msg").textContent=`Reclassified ${count} to ${CAT_NAMES[newCat]||newCat}`;
  updateApplyBtn();
  renderGrid();
}

function updateApplyBtn(){
  const btn=document.getElementById("apply-btn");
  const n=Object.keys(reclassMap).length;
  btn.disabled=n===0;
  btn.textContent=n>0?`Apply ${n} Changes`:'Apply Changes';
}

async function applyChanges(){
  const items=[];
  for(const[key,newCat] of Object.entries(reclassMap)){
    const [board,ann_id_str]=key.split('__');
    items.push({board, ann_id:parseInt(ann_id_str), new_category_id:newCat});
  }
  if(items.length===0)return;

  document.getElementById("status-msg").textContent="Applying...";
  try{
    const r=await(await fetch("/api/class_reclassify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({items})})).json();
    if(r.ok){
      document.getElementById("status-msg").textContent=`✅ Applied ${r.changed} changes, ${r.corrections} corrections logged`;
      // Keep reclassified visual state, clear selection
      selectedSet.clear();
      renderGrid();
      updateSelText();
    } else {
      document.getElementById("status-msg").textContent="❌ Error applying changes";
    }
  }catch(e){
    document.getElementById("status-msg").textContent=`❌ ${e.message}`;
  }
}

function updateStats(){
  document.getElementById("stats-text").textContent=
    `Showing ${displayItems.length} of ${totalItems} ${CAT_NAMES[currentCat]||'?'} annotations | Page ${currentPage}/${totalPages}`;
}
function updateSelText(){
  document.getElementById("sel-text").textContent=
    selectedSet.size>0?`${selectedSet.size} selected`:'';
}
function updatePageNav(){
  document.getElementById("prev-page").disabled=currentPage<=1;
  document.getElementById("next-page").disabled=currentPage>=totalPages;
  document.getElementById("page-info").textContent=`Page ${currentPage}/${totalPages}`;
}
function changePage(dir){
  const np=currentPage+dir;
  if(np>=1&&np<=totalPages){currentPage=np;loadCrops();}
}

// Keyboard shortcuts
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  const k=e.key;
  // Category hotkeys
  if(KEY_TO_CAT[k]){reclassifySelected(KEY_TO_CAT[k]);return;}
  // A = select all
  if(k==='a'||k==='A'){e.preventDefault();selectAllVisible();return;}
  // Enter = apply
  if(k==='Enter'){e.preventDefault();applyChanges();return;}
  // Escape = clear selection
  if(k==='Escape'){selectedSet.clear();renderGrid();updateSelText();return;}
  // Arrow keys for page nav
  if(k==='ArrowLeft'){changePage(-1);return;}
  if(k==='ArrowRight'){changePage(1);return;}
});

initUI();
</script>
</body>
</html>
"""




# ======= Orientation Review =======
import math as _math
from collections import Counter as _Counter

_ORI_FINAL_DIR = ANNO_DIR
_ori_data = {}  # lazy loaded
_ori_loaded = False

def _load_ori_data():
    global _ori_data, _ori_loaded
    if _ori_loaded:
        return
    print("Loading orientation data from final_annotations...")
    for f in sorted(_ORI_FINAL_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        _ori_data[f.stem] = data
    _ori_loaded = True
    print(f"  Loaded {len(_ori_data)} boards")

@app.route("/orientation_review")
def ori_review_page():
    return ORI_REVIEW_HTML

@app.route("/api/ori/stats")
def api_ori_stats():
    _load_ori_data()
    angles = [0,45,90,135,180,225,270,315]
    angle_dist = _Counter()
    cat_dist = _Counter()
    cat_names = {1:"Resistor",2:"Capacitor",3:"Inductor",4:"Connector",5:"Diode",7:"Switch",8:"Transistor",9:"IC",10:"Oscillator"}
    for board, data in _ori_data.items():
        for ann in data.get("annotations",[]):
            ori = ann.get("orientation")
            if ori is not None:
                angle_dist[ori] += 1
                cat_dist[ann.get("category_id",0)] += 1
    total = sum(angle_dist.values())
    return jsonify({
        "total": total, "boards": len(_ori_data),
        "angles": {str(a): angle_dist.get(a,0) for a in angles},
        "categories": {cat_names.get(c,f"Cat{c}"): cnt for c,cnt in sorted(cat_dist.items())},
    })

@app.route("/api/ori/browse")
def api_ori_browse():
    _load_ori_data()
    angle = int(request.args.get("angle", 0))
    cat_filter = request.args.get("category", "")
    source_filter = request.args.get("source", "")
    page = int(request.args.get("page", 0))
    per_page = int(request.args.get("per_page", 100))
    cat_names = {1:"Resistor",2:"Capacitor",3:"Inductor",4:"Connector",5:"Diode",7:"Switch",8:"Transistor",9:"IC",10:"Oscillator"}
    cat_id_map = {v.lower():k for k,v in cat_names.items()}
    target_cat = cat_id_map.get(cat_filter.lower()) if cat_filter else None

    # Source lookup for filtering
    _src_cache = {}
    def _get_source(b):
        if b not in _src_cache:
            import os as _os
            fn = b + ".json"
            if _os.path.exists(str(BACKUP_DIR) + "/" + fn): _src_cache[b] = "backup"
            elif _os.path.exists(str(ANNO_DIR) + "/" + fn): _src_cache[b] = "v6-chen"
            elif _os.path.exists(str(ANNO_DIR) + "/" + fn): _src_cache[b] = "v6"
            else: _src_cache[b] = "original"
        return _src_cache[b]

    items = []
    for board, data in _ori_data.items():
        if source_filter and _get_source(board) != source_filter:
            continue
        for ann in data.get("annotations",[]):
            if ann.get("orientation") != angle:
                continue
            if target_cat and ann.get("category_id") != target_cat:
                continue
            items.append({"board":board,"ann":ann})

    total = len(items)
    start = page * per_page
    page_items = items[start:start+per_page]
    return jsonify({
        "angle":angle, "category":cat_filter, "page":page,
        "total":total, "total_pages":max(1,(total+per_page-1)//per_page),
        "items":[{
            "board":it["board"], "ann_id":it["ann"]["id"],
            "category":cat_names.get(it["ann"]["category_id"],"?"),
            "bbox":it["ann"]["bbox"], "orientation":it["ann"].get("orientation"),
        } for it in page_items],
    })

@app.route("/api/ori/crop/<board>/<int:ann_id>")
def api_ori_crop(board, ann_id):
    _load_ori_data()
    data = _ori_data.get(board)
    if not data:
        return "Not found", 404
    ann = next((a for a in data.get("annotations",[]) if a["id"]==ann_id), None)
    if not ann:
        return "Not found", 404
    img_file = find_image(board)
    if not img_file:
        return "Image not found", 404
    from PIL import Image as PILImage, ImageDraw as PILDraw
    img = PILImage.open(img_file).convert("RGB")
    bx,by,bw,bh = ann["bbox"]
    x1,y1 = max(0,int(bx)), max(0,int(by))
    x2,y2 = min(img.width,int(bx+bw)), min(img.height,int(by+bh))
    if x2<=x1 or y2<=y1:
        return "Bad bbox", 400
    crop = img.crop((x1,y1,x2,y2))
    import io as _io
    buf = _io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/api/ori/save", methods=["POST"])
def api_ori_save():
    _load_ori_data()
    data = request.json
    board = data.get("board")
    ann_id = data.get("ann_id")
    new_ori = data.get("orientation")
    if board is None or ann_id is None or new_ori is None:
        return jsonify({"error": "Missing fields"}), 400
    # Update in-memory
    board_data = _ori_data.get(board)
    if not board_data:
        return jsonify({"error": "Board not found"}), 404
    found = False
    for ann in board_data.get("annotations", []):
        if ann["id"] == ann_id:
            ann["orientation"] = int(new_ori)
            found = True
            break
    if not found:
        return jsonify({"error": "Annotation not found"}), 404
    # Save to disk
    from pathlib import Path as _P
    out_file = ANNO_DIR / f"{board}.json"
    out_file.write_text(json.dumps(board_data, indent=2))
    return jsonify({"ok": True, "board": board, "ann_id": ann_id, "orientation": int(new_ori)})

@app.route("/api/ori/delete", methods=["POST"])
def api_ori_delete():
    _load_ori_data()
    data = request.json
    board = data.get("board")
    ann_id = data.get("ann_id")
    if board is None or ann_id is None:
        return jsonify({"error": "Missing fields"}), 400
    board_data = _ori_data.get(board)
    if not board_data:
        return jsonify({"error": "Board not found"}), 404
    anns = board_data.get("annotations", [])
    new_anns = [a for a in anns if a["id"] != ann_id]
    if len(new_anns) == len(anns):
        return jsonify({"error": "Annotation not found"}), 404
    board_data["annotations"] = new_anns
    from pathlib import Path as _P
    out_file = ANNO_DIR / f"{board}.json"
    out_file.write_text(json.dumps(board_data, indent=2))
    return jsonify({"ok": True, "board": board, "ann_id": ann_id, "deleted": True})

@app.route("/api/ori/edit_category", methods=["POST"])
def api_ori_edit_category():
    _load_ori_data()
    data = request.json
    board = data.get("board")
    ann_id = data.get("ann_id")
    new_cat = data.get("category")
    VALID_CATS = {"Resistor","Capacitor","Inductor","Connector","Diode","Switch","Transistor","IC","Oscillator"}
    if board is None or ann_id is None or new_cat not in VALID_CATS:
        return jsonify({"error": "Missing/invalid fields"}), 400
    # Map category name to ID
    CAT_NAME_TO_ID = {"Resistor":1,"Capacitor":2,"Inductor":3,"Connector":4,"Diode":5,"Switch":7,"Transistor":8,"IC":9,"Oscillator":10}
    board_data = _ori_data.get(board)
    if not board_data:
        return jsonify({"error": "Board not found"}), 404
    found = False
    for ann in board_data.get("annotations", []):
        if ann["id"] == ann_id:
            ann["category_id"] = CAT_NAME_TO_ID[new_cat]
            ann["category_name"] = new_cat
            found = True
            break
    if not found:
        return jsonify({"error": "Annotation not found"}), 404
    from pathlib import Path as _P
    out_file = ANNO_DIR / f"{board}.json"
    out_file.write_text(json.dumps(board_data, indent=2))
    return jsonify({"ok": True, "board": board, "ann_id": ann_id, "category": new_cat})

@app.route("/api/ori/source")
def api_ori_source():
    import os as _os
    BACKUP_DIR_STR = str(BACKUP_DIR)
    CHEN_DIR = str(ANNO_DIR)
    V6_DIR = str(ANNO_DIR)
    FA_DIR = str(ANNO_DIR)
    sources = {}
    for fn in _os.listdir(FA_DIR):
        if not fn.endswith(".json"): continue
        b = fn[:-5]
        if _os.path.exists(_os.path.join(BACKUP_DIR, fn)): sources[b] = "backup"
        elif _os.path.exists(_os.path.join(CHEN_DIR, fn)): sources[b] = "v6-chen"
        elif _os.path.exists(_os.path.join(V6_DIR, fn)): sources[b] = "v6"
        else: sources[b] = "original"
    return jsonify(sources)

ORI_REVIEW_HTML = """<!DOCTYPE html>
<html><head><title>Orientation Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui;background:#1a1a2e;color:#eee;display:flex;flex-direction:column;height:100vh}
.toolbar{display:flex;gap:8px;padding:8px 12px;background:#16213e;align-items:center;flex-wrap:wrap}
.toolbar select,.toolbar button{padding:4px 10px;border-radius:4px;border:1px solid #555;background:#0f3460;color:#eee;cursor:pointer;font-size:13px}
.toolbar button:hover{background:#e94560}
.toolbar .info{color:#aaa;font-size:12px;margin-left:auto}
.grid{display:flex;flex-wrap:wrap;gap:4px;padding:8px;overflow-y:auto;flex:1}
.card{width:90px;height:90px;border:2px solid #333;border-radius:4px;overflow:hidden;cursor:pointer;position:relative}
.card.selected{border-color:#e94560;border-width:3px;box-shadow:0 0 8px #e94560}
.card img{width:100%;height:100%;object-fit:contain;background:#222}
.card .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.7);color:#fff;font-size:9px;text-align:center;padding:1px}
.card:hover{border-color:#e94560}
.card .src-badge{position:absolute;top:1px;right:1px;font-size:8px;padding:0 3px;border-radius:2px;color:#fff;font-weight:bold}
.card .src-v6{background:#e74c3c}
.card .src-v6chen{background:#9b59b6}
.card .src-backup{background:#27ae60}
.pager{display:flex;gap:8px;padding:8px 12px;background:#16213e;align-items:center;justify-content:center}
.pager button{padding:4px 12px;border-radius:4px;border:1px solid #555;background:#0f3460;color:#eee;cursor:pointer}
.stats{padding:6px 12px;background:#0f3460;font-size:12px;display:flex;gap:16px;flex-wrap:wrap}
.stats span{color:#aaa}
.hotkeys{padding:4px 12px;background:#1a1a2e;font-size:11px;color:#888;border-top:1px solid #333}
.hotkeys kbd{background:#333;padding:1px 5px;border-radius:3px;color:#eee;font-family:monospace}
.status-bar{padding:4px 12px;background:#27ae60;color:#fff;font-size:12px;text-align:center;display:none}
</style></head><body>
<div class="stats" id="stats"></div>
<div class="toolbar">
  <label>Angle: <select id="angle-sel" onchange="if(!_statsUpdating){curPage=0;browse();}"></select></label>
  <label>Category: <select id="cat-sel" onchange="curPage=0;browse()">
    <option value="">All</option>
    <option>Resistor</option><option>Capacitor</option><option>Inductor</option>
    <option>Connector</option><option>Diode</option><option>Switch</option>
    <option>Transistor</option><option>IC</option><option>Oscillator</option>
  </select></label>
  <label>Source: <select id="source-sel" onchange="curPage=0;browse()">
    <option value="">All</option>
    <option value="v6">v6 only</option>
    <option value="v6-chen">v6+Chen</option>
    <option value="backup">Reviewed</option>
  </select></label>
  <button onclick="browse()">Refresh</button>
  <button onclick="selectAll()" style="background:#8e44ad">Select All</button>
  <button onclick="clearSel()" style="background:#7f8c8d">Clear</button>
  <button onclick="deleteSelected()" style="background:#c0392b">🗑 Delete</button>
  <label>Re-cat: <select id="recat-sel">
    <option value="">--</option>
    <option>Resistor</option><option>Capacitor</option><option>Inductor</option>
    <option>Connector</option><option>Diode</option><option>Switch</option>
    <option>Transistor</option><option>IC</option><option>Oscillator</option>
  </select></label>
  <button onclick="recatSelected()" style="background:#2980b9">✏ Re-categorize</button>
  <div class="info" id="page-info"></div>
</div>
<div class="status-bar" id="status-bar"></div>
<div class="grid" id="grid"></div>
<div class="pager">
  <button onclick="prevPage()">&larr; Prev</button>
  <span id="pager-text">Page 1</span>
  <button onclick="nextPage()">Next &rarr;</button>
  <span style="margin-left:16px">Go to:</span>
  <input type="number" id="page-jump" min="1" style="width:60px;padding:3px 6px;border-radius:4px;border:1px solid #555;background:#0f3460;color:#eee;font-size:13px" onkeydown="if(event.key==='Enter')jumpPage()">
  <button onclick="jumpPage()">Go</button>
</div>
<div class="hotkeys">
  Select: click (shift+click range) | Relabel: <kbd>Q</kbd>=\u2196NW <kbd>W</kbd>=\u2191N <kbd>E</kbd>=\u2197NE <kbd>A</kbd>=\u2190W <kbd>D</kbd>=\u2192E <kbd>Z</kbd>=\u2199SW <kbd>S</kbd>=\u2193S <kbd>C</kbd>=\u2198SE | <kbd>X</kbd>/<kbd>Del</kbd>=delete | <kbd>a</kbd>=select all <kbd>Esc</kbd>=clear | <kbd>\u2190\u2192</kbd>=pages
</div>
<script>
let curPage=0,totalPages=1,curItems=[];
const selected=new Set();
const arrows={0:"\u2192",45:"\u2197",90:"\u2191",135:"\u2196",180:"\u2190",225:"\u2199",270:"\u2193",315:"\u2198"};
const keyMap={q:135,w:90,e:45,a:180,d:0,z:225,s:270,c:315};
const numMap={"1":135,"2":90,"3":45,"4":180,"6":0,"7":225,"8":270,"9":315};

function showStatus(msg,ms){
  const bar=document.getElementById("status-bar");
  bar.textContent=msg;bar.style.display="block";
  setTimeout(()=>{bar.style.display="none";},ms||2000);
}

let _statsUpdating=false;
async function loadStats(){
  _statsUpdating=true;
  const sel=document.getElementById("angle-sel");
  const prevVal=sel.value||"45";
  const r=await(await fetch("/api/ori/stats")).json();
  const el=document.getElementById("stats");
  let h=`<span>Total: ${r.total} | ${r.boards} boards</span><span>|</span>`;
  for(const a of [0,45,90,135,180,225,270,315])h+=`<span>${arrows[a]} ${a}\u00b0: ${r.angles[a]||0}</span>`;
  el.innerHTML=h;
  sel.innerHTML="";
  for(const a of [0,45,90,135,180,225,270,315]){
    const o=document.createElement("option");o.value=a;
    o.textContent=`${arrows[a]} ${a}\u00b0 (${r.angles[a]||0})`;sel.appendChild(o);
  }
  sel.value=prevVal;
  _statsUpdating=false;
}

function renderCards(){
  const grid=document.getElementById("grid");grid.innerHTML="";
  curItems.forEach((it,i)=>{
    const d=document.createElement("div");
    d.className="card"+(selected.has(i)?" selected":"");
    d.dataset.idx=i;
    const srcT=boardSources[it.board]||"";
    const srcB=srcT==="v6"?'<div class="src-badge src-v6">v6</div>':srcT==="v6-chen"?'<div class="src-badge src-v6chen">v6c</div>':srcT==="backup"?'<div class="src-badge src-backup">\u2713</div>':"";
    d.innerHTML=`<img src="/api/ori/crop/${it.board}/${it.ann_id}" loading="lazy">${srcB}<div class="label">${it.category} ${arrows[it.orientation]||""}</div>`;
    d.title=`${it.board} #${it.ann_id} ${it.orientation}\u00b0`;
    d.onclick=(e)=>toggleSelect(i,e);
    grid.appendChild(d);
  });
  document.getElementById("page-info").textContent=`${selected.size} selected | ${curItems.length} on page`;
}

let lastClickIdx=-1;
function toggleSelect(idx,e){
  if(e&&e.shiftKey&&lastClickIdx>=0){
    const lo=Math.min(lastClickIdx,idx),hi=Math.max(lastClickIdx,idx);
    for(let i=lo;i<=hi;i++)selected.add(i);
  } else {
    if(selected.has(idx))selected.delete(idx);else selected.add(idx);
  }
  lastClickIdx=idx;
  renderCards();
}
function selectAll(){curItems.forEach((_,i)=>selected.add(i));renderCards();}
function clearSel(){selected.clear();lastClickIdx=-1;renderCards();}

async function browse(page){
  if(page!==undefined)curPage=page;
  selected.clear();lastClickIdx=-1;
  const angle=document.getElementById("angle-sel").value;
  const cat=document.getElementById("cat-sel").value;
  const src=document.getElementById("source-sel")?document.getElementById("source-sel").value:"";
  const r=await(await fetch(`/api/ori/browse?angle=${angle}&category=${cat}&page=${curPage}&per_page=100${src?"&source="+src:""}`)).json();
  totalPages=r.total_pages;
  curItems=r.items;
  document.getElementById("pager-text").textContent=`Page ${curPage+1}/${totalPages}`;
  renderCards();
}

async function relabel(newAngle){
  if(selected.size===0){showStatus("Nothing selected",1500);return;}
  const items=[...selected].map(i=>curItems[i]);
  showStatus(`Relabeling ${items.length} items to ${arrows[newAngle]} ${newAngle}\u00b0...`);
  let ok=0;
  for(const it of items){
    const r=await(await fetch("/api/ori/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:it.board,ann_id:it.ann_id,orientation:newAngle})})).json();
    if(r.ok){ok++;it.orientation=newAngle;}
  }
  showStatus(`\u2705 Relabeled ${ok} items to ${arrows[newAngle]} ${newAngle}\u00b0`,2000);
  selected.clear();
  renderCards();
  loadStats();
}

function prevPage(){if(curPage>0){curPage--;browse();}}
function nextPage(){if(curPage<totalPages-1){curPage++;browse();}}

async function deleteSelected(){
  if(selected.size===0){showStatus("Nothing selected",1500);return;}
  if(!confirm(`Delete ${selected.size} component(s)? This cannot be undone.`))return;
  const items=[...selected].map(i=>curItems[i]);
  showStatus(`Deleting ${items.length} items...`);
  let ok=0;
  for(const it of items){
    const r=await(await fetch("/api/ori/delete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:it.board,ann_id:it.ann_id})})).json();
    if(r.ok)ok++;
  }
  showStatus(`\u2705 Deleted ${ok} items`,2000);
  selected.clear();
  browse();
  loadStats();
}

async function recatSelected(){
  const newCat=document.getElementById("recat-sel").value;
  if(!newCat){showStatus("Select a category first",1500);return;}
  if(selected.size===0){showStatus("Nothing selected",1500);return;}
  const items=[...selected].map(i=>curItems[i]);
  showStatus(`Re-categorizing ${items.length} items to ${newCat}...`);
  let ok=0;
  for(const it of items){
    const r=await(await fetch("/api/ori/edit_category",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:it.board,ann_id:it.ann_id,category:newCat})})).json();
    if(r.ok){ok++;it.category=newCat;}
  }
  showStatus(`\u2705 Re-categorized ${ok} items to ${newCat}`,2000);
  selected.clear();
  renderCards();
  loadStats();
}

document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  const k=e.key.toLowerCase();
  if(k in keyMap){e.preventDefault();relabel(keyMap[k]);return;}
  if(k in numMap){e.preventDefault();relabel(numMap[k]);return;}
  if(k==="x"||k==="delete"){e.preventDefault();deleteSelected();return;}
  if(k==="arrowleft")prevPage();
  else if(k==="arrowright")nextPage();
  else if(k==="escape")clearSel();
  else if(k==="a"&&selected.size===0)selectAll();
});

let boardSources={};
async function loadSources(){boardSources=await(await fetch("/api/ori/source")).json();}
function jumpPage(){
  const p=parseInt(document.getElementById("page-jump").value);
  if(p>=1&&p<=totalPages){curPage=p-1;browse();}
  else showStatus("Invalid page (1-"+totalPages+")",1500);
}
loadSources();
loadStats().then(()=>browse(0));
</script></body></html>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--anno_dir", type=str, default=None, help="Annotation JSON directory (pipeline output)")
    parser.add_argument("--image_dir", type=str, default=None, help="Board image directory")
    args = parser.parse_args()
    if args.anno_dir and args.image_dir:
        _init_paths(args.anno_dir, args.image_dir)

    load_custom_categories()
    if FLAG_FILE.exists():
        _flagged = json.loads(FLAG_FILE.read_text())
        flagged_anns.update(_flagged)
        total_flagged = sum(len(v) for v in flagged_anns.values())
        print(f"Loaded {total_flagged} flagged annotations across {len(flagged_anns)} boards")
    build_board_list()
    app.run(host=args.host, port=args.port, debug=False)
