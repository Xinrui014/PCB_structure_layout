#!/usr/bin/env python3
"""Extract board colors using k-means dominant cluster instead of mean."""
import json, os, glob
from PIL import Image
import numpy as np
from colorsys import rgb_to_hsv
from sklearn.cluster import MiniBatchKMeans

IMG_DIR = "/home/xinrui/projects/data/ti_pcb/images_top"
ANNO_DIR = "/home/xinrui/projects/PCB_structure_layout/re-annotation/final_annotations"
DEL_FILE = os.path.join(os.path.dirname(ANNO_DIR), "resolution_deleted_boards.json")
EXCL_FILE = "/home/xinrui/projects/PCB_structure_layout/component_pool/excluded_boards.json"

with open(DEL_FILE) as f:
    deleted = set(json.load(f))
with open(EXCL_FILE) as f:
    excluded = set(json.load(f))

def classify_rgb(r, g, b):
    # Black: all channels low
    if r < 30 and g < 30 and b < 30:
        return "black"
    # White: all channels high
    if r > 180 and g > 180 and b > 180:
        return "white"
    # Simple: largest channel wins
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"

results = []
check_boards = {"iwrl6432aopevm-top", "iwr6843levm-top", "trf37135evm-top", "ads42lb69evm-top"}

for i, path in enumerate(sorted(glob.glob(os.path.join(ANNO_DIR, "*.json")))):
    board = os.path.splitext(os.path.basename(path))[0]
    if board in deleted or board in excluded:
        continue
    img_path = os.path.join(IMG_DIR, board + ".png")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, board + ".jpg")
    if not os.path.exists(img_path):
        continue
    with open(path) as f:
        data = json.load(f)
    img = np.array(Image.open(img_path).convert("RGB"))
    ih, iw = img.shape[:2]
    anns = data.get("annotations", [])
    if not anns:
        continue

    # Collect all bbox corner points
    from scipy.spatial import ConvexHull
    corners = []
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        corners.extend([(x, y), (x+bw, y), (x, y+bh), (x+bw, y+bh)])
    corners = np.array(corners)

    # Build convex hull of all component corners = board boundary
    try:
        hull = ConvexHull(corners)
        hull_pts = corners[hull.vertices].astype(np.int32)
    except Exception:
        continue

    # Create mask from convex hull polygon
    import cv2
    mask = np.zeros((ih, iw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_pts, 1)
    mask = mask.astype(bool)

    # Exclude component bboxes
    for ann in anns:
        x, y, bw, bh = [int(v) for v in ann["bbox"]]
        mask[max(0, y):min(ih, y + bh), max(0, x):min(iw, x + bw)] = False

    bg = img[mask].astype(np.float32)
    if len(bg) < 500:
        continue
    if len(bg) > 20000:
        idx = np.random.RandomState(42).choice(len(bg), 20000, replace=False)
        bg = bg[idx]

    km = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=3)
    km.fit(bg)
    labels_u, counts = np.unique(km.labels_, return_counts=True)

    # Collect all clusters sorted by size (largest first)
    clusters = []
    for ci in range(len(labels_u)):
        c = km.cluster_centers_[labels_u[ci]]
        cr, cg, cb = float(c[0]), float(c[1]), float(c[2])
        clusters.append({"r": round(cr,1), "g": round(cg,1), "b": round(cb,1), "n": int(counts[ci])})
    clusters.sort(key=lambda x: -x["n"])

    # Pick dominant: largest cluster, skip very dark (<30) and very white (>220)
    dom = None
    dom_idx = 0
    for i, cl in enumerate(clusters):
        brightness = (cl["r"] + cl["g"] + cl["b"]) / 3
        if brightness < 30 or brightness > 220:
            continue
        dom = cl
        dom_idx = i
        break
    if dom is None:
        dom = clusters[0]
        dom_idx = 0

    r, g, b = dom["r"], dom["g"], dom["b"]
    hv, sv, vv = rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg, s_pct, v_pct = hv * 360, sv * 100, vv * 100
    color = classify_rgb(r, g, b)
    
    entry = {
        "board": board, "r": round(r, 1), "g": round(g, 1), "b": round(b, 1),
        "h": round(h_deg, 1), "s": round(s_pct, 1), "v": round(v_pct, 1), "color": color,
        "clusters": clusters[:5], "dom_idx": dom_idx,
    }
    results.append(entry)

    if board in check_boards:
        print(f"  CHECK {board}: {color} H={h_deg:.0f} S={s_pct:.0f} V={v_pct:.0f} RGB({r:.0f},{g:.0f},{b:.0f})")

    if (i + 1) % 500 == 0:
        print(f"  processed {i + 1}...", flush=True)

out = "/home/xinrui/projects/PCB_structure_layout/re-annotation/board_colors_v2.json"
with open(out, "w") as f:
    json.dump(results, f)

cc = {}
for d in results:
    cc[d["color"]] = cc.get(d["color"], 0) + 1
print(f"\nSaved {len(results)} boards to {out}")
print("=== Distribution (k-means dominant cluster) ===")
for c, n in sorted(cc.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n} ({100 * n / len(results):.1f}%)")
