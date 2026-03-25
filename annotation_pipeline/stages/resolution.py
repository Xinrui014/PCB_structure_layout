"""Stage 3: Resolution classification based on median Resistor/Capacitor body size."""
import json
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESOLUTION_BINS, CAT_ID_TO_NAME


def get_resolution_class(median_size):
    """Map median body size to resolution class R1–R7."""
    for name, lo, hi in RESOLUTION_BINS:
        if lo <= median_size < hi:
            return name
    return "R7"


def compute_resolution(annotations):
    """Compute resolution class from annotations using Resistor anchor (Capacitor fallback)."""
    # Collect min(w,h) per anchor category
    anchor_cats = ["Resistor", "Capacitor"]
    sizes_by_cat = {}

    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = CAT_ID_TO_NAME.get(cat_id, "")
        if cat_name in anchor_cats:
            _, _, w, h = ann["bbox"]
            body_size = min(w, h)
            if body_size > 0:
                sizes_by_cat.setdefault(cat_name, []).append(body_size)

    # Pick anchor: Resistor first, then Capacitor
    for anchor in anchor_cats:
        if anchor in sizes_by_cat and len(sizes_by_cat[anchor]) >= 3:
            median = statistics.median(sizes_by_cat[anchor])
            return get_resolution_class(median), round(median, 2), anchor

    # Fallback: use any available anchor
    for anchor in anchor_cats:
        if anchor in sizes_by_cat:
            median = statistics.median(sizes_by_cat[anchor])
            return get_resolution_class(median), round(median, 2), anchor

    return None, None, None


def run(output_dir):
    """Add resolution_class, resolution_median, resolution_anchor to all JSONs."""
    output_dir = Path(output_dir)
    jsons = sorted(output_dir.glob("*.json"))
    print(f"[Stage 3: Resolution] Processing {len(jsons)} boards...")

    stats = {"classified": 0, "no_anchor": 0}

    for i, json_path in enumerate(jsons):
        data = json.loads(json_path.read_text())
        anns = data.get("annotations", [])

        res_class, median, anchor = compute_resolution(anns)

        if res_class:
            data["resolution_class"] = res_class
            data["resolution_median"] = median
            data["resolution_anchor"] = anchor
            stats["classified"] += 1
        else:
            data["resolution_class"] = None
            data["resolution_median"] = None
            data["resolution_anchor"] = None
            stats["no_anchor"] += 1

        json_path.write_text(json.dumps(data, indent=2))

        if (i + 1) % 50 == 0 or i == len(jsons) - 1:
            print(f"  [{i+1}/{len(jsons)}] {json_path.stem}: {res_class or 'N/A'}")

    print(f"[Stage 3: Resolution] Done. Classified: {stats['classified']}, No anchor: {stats['no_anchor']}")
