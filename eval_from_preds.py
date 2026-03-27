#!/usr/bin/env python3
"""
Evaluate layout quality from existing _pred.txt files (no model loading needed).
Reads pred files + test.jsonl, computes all metrics.
"""

import json, re, argparse, os
from collections import Counter
from pathlib import Path

CANVAS_W = 1280
CANVAS_H = 720
CATEGORIES = {"RESISTOR","CAPACITOR","INDUCTOR","CONNECTOR","DIODE","LED","SWITCH","TRANSISTOR","IC","OSCILLATOR"}
RESOLUTION_TOKENS = {f"R{i}" for i in range(1, 8)}
COLOR_TOKENS = {"COLOR_GREEN", "COLOR_RED", "COLOR_BLUE", "COLOR_BLACK", "COLOR_WHITE", "COLOR_GREY", "COLOR_GRAY"}

# ── Parsing ──────────────────────────────────────────────────────────────────
def parse_layout(text):
    components = []
    meta = {"board_color": None, "resolution": None}
    tokens = re.findall(r'\[([^\[\]]+)\]', text)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in COLOR_TOKENS:
            meta["board_color"] = t
            i += 1; continue
        if t in RESOLUTION_TOKENS:
            meta["resolution"] = t
            i += 1; continue
        if t in CATEGORIES and i + 4 < len(tokens):
            try:
                x, y, w, h = int(tokens[i+1]), int(tokens[i+2]), int(tokens[i+3]), int(tokens[i+4])
                if w > 0 and h > 0:
                    components.append({"type": t, "x": x, "y": y, "w": w, "h": h})
                i += 5; continue
            except ValueError:
                pass
        i += 1
    return components, meta

# ── Hard Constraint Metrics ──────────────────────────────────────────────────
def boundary_rate(comps, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    if not comps: return 1.0
    ok = sum(1 for c in comps if c["x"] >= 0 and c["y"] >= 0
             and c["x"]+c["w"] <= canvas_w and c["y"]+c["h"] <= canvas_h)
    return ok / len(comps)

def overlap_rate(comps):
    if len(comps) <= 1: return 1.0
    def intersects(a, b):
        return not (a["x"]+a["w"] <= b["x"] or b["x"]+b["w"] <= a["x"] or
                    a["y"]+a["h"] <= b["y"] or b["y"]+b["h"] <= a["y"])
    no_overlap = 0
    for i, c in enumerate(comps):
        if not any(intersects(c, comps[j]) for j in range(len(comps)) if j != i):
            no_overlap += 1
    return no_overlap / len(comps)

# ── Quality Metrics ──────────────────────────────────────────────────────────
def parse_prompt_counts(prompt):
    name_map = {
        "capacitor": "CAPACITOR", "capacitors": "CAPACITOR",
        "resistor": "RESISTOR", "resistors": "RESISTOR",
        "inductor": "INDUCTOR", "inductors": "INDUCTOR",
        "connector": "CONNECTOR", "connectors": "CONNECTOR",
        "diode": "DIODE", "diodes": "DIODE",
        "led": "LED", "leds": "LED",
        "switch": "SWITCH", "switches": "SWITCH",
        "transistor": "TRANSISTOR", "transistors": "TRANSISTOR",
        "ic": "IC", "ics": "IC",
        "oscillator": "OSCILLATOR", "oscillators": "OSCILLATOR",
    }
    counts = {}
    for match in re.finditer(r'(\d+)\s+([a-zA-Z]+)', prompt):
        num = int(match.group(1))
        word = match.group(2).lower()
        if word in name_map:
            cat = name_map[word]
            counts[cat] = counts.get(cat, 0) + num
    return counts

def prompt_count_accuracy(prompt, pred):
    expected = parse_prompt_counts(prompt)
    pred_cnt = Counter(c["type"] for c in pred)
    per_class = {}
    matches = []
    for cat, exp_n in expected.items():
        pred_n = pred_cnt.get(cat, 0)
        match = (exp_n == pred_n)
        per_class[cat] = {"expected": exp_n, "predicted": pred_n, "match": match}
        matches.append(match)
    exact_match_rate = sum(matches) / len(matches) if matches else 1.0
    overall_exact = 1.0 if all(matches) else 0.0
    return per_class, exact_match_rate, overall_exact

def count_accuracy(gt, pred):
    return 1.0 if len(gt) == len(pred) else 0.0

def category_accuracy(gt, pred):
    gt_cnt = Counter(c["type"] for c in gt)
    pred_cnt = Counter(c["type"] for c in pred)
    scores = []
    for cat, gt_n in gt_cnt.items():
        pred_n = pred_cnt.get(cat, 0)
        scores.append(min(pred_n, gt_n) / gt_n)
    return sum(scores) / len(scores) if scores else 0.0

def iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a["w"]*a["h"] + b["w"]*b["h"] - inter
    return inter / union if union > 0 else 0.0

def mean_best_iou(gt, pred, iou_thresh=0.3):
    if not gt or not pred: return 0.0, 0.0
    best_ious = []
    for g in gt:
        best = max((iou(g, p) for p in pred), default=0.0)
        best_ious.append(best)
    mean_val = sum(best_ious) / len(best_ious)
    coverage = sum(1 for v in best_ious if v >= iou_thresh) / len(best_ious)
    return mean_val, coverage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory with *_pred.txt files")
    parser.add_argument("--test_jsonl", required=True, help="Path to test.jsonl")
    args = parser.parse_args()

    # Build GT map: image_name -> (prompt, gt_text)
    gt_map = {}
    with open(args.test_jsonl) as f:
        for line in f:
            sample = json.loads(line)
            msgs = sample["messages"]
            prompt = msgs[1]["content"]
            gt_text = msgs[2]["content"]
            image_name = sample.get("_meta", {}).get("image", "")
            if image_name:
                gt_map[image_name] = (prompt, gt_text)

    # Find pred files
    pred_files = sorted(Path(args.pred_dir).glob("*_pred.txt"))
    print(f"Found {len(pred_files)} pred files, {len(gt_map)} GT samples\n")

    metrics = {
        "boundary_rate": [],
        "overlap_rate": [],
        "prompt_class_exact_rate": [],
        "prompt_all_classes_exact": [],
        "count_accuracy": [],
        "category_accuracy": [],
        "mean_iou": [],
        "coverage": [],
    }

    matched = 0
    for pred_file in pred_files:
        # Extract image name: e.g. "boardname_pred.txt" -> "boardname"
        image_name = pred_file.stem.replace("_pred", "")
        if image_name not in gt_map:
            continue

        matched += 1
        prompt, gt_text = gt_map[image_name]
        pred_text = pred_file.read_text()

        gt, gt_meta = parse_layout(gt_text)
        pred, pred_meta = parse_layout(pred_text)

        br = boundary_rate(pred)
        or_ = overlap_rate(pred)
        ca = count_accuracy(gt, pred)
        cata = category_accuracy(gt, pred)
        miou, cov = mean_best_iou(gt, pred)
        per_class, cls_exact_rate, all_exact = prompt_count_accuracy(prompt, pred)

        metrics["boundary_rate"].append(br)
        metrics["overlap_rate"].append(or_)
        metrics["prompt_class_exact_rate"].append(cls_exact_rate)
        metrics["prompt_all_classes_exact"].append(all_exact)
        metrics["count_accuracy"].append(ca)
        metrics["category_accuracy"].append(cata)
        metrics["mean_iou"].append(miou)
        metrics["coverage"].append(cov)

        cls_detail = " | ".join(
            f"{cat}:{v['expected']}→{v['predicted']}{'✓' if v['match'] else '✗'}"
            for cat, v in per_class.items()
        )
        print(f"[{matched:3d}] {image_name}: GT={len(gt):3d} Pred={len(pred):3d} | "
              f"bound={br:.2f} overlap={or_:.2f} iou={miou:.3f} cov={cov:.2f} | meta={pred_meta}")
        if not all_exact:
            print(f"  Prompt counts: {cls_detail}")

    n = len(metrics["boundary_rate"])
    print(f"\n{'='*60}")
    print("LAYOUT EVALUATION SUMMARY")
    print(f"{'='*60}")
    print("  -- Hard Constraints --")
    print(f"  {'boundary_rate':<30s}: {sum(metrics['boundary_rate'])/n:.4f}  (fraction of comps inside canvas)")
    print(f"  {'overlap_rate':<30s}: {sum(metrics['overlap_rate'])/n:.4f}  (fraction of comps with no overlap)")
    print(f"  {'prompt_class_exact_rate':<30s}: {sum(metrics['prompt_class_exact_rate'])/n:.4f}  (fraction of classes matching prompt count)")
    print(f"  {'prompt_all_classes_exact':<30s}: {sum(metrics['prompt_all_classes_exact'])/n:.4f}  (fraction of samples where ALL classes match)")
    print("  -- Quality vs. GT --")
    print(f"  {'count_accuracy':<30s}: {sum(metrics['count_accuracy'])/n:.4f}  (exact total count match)")
    print(f"  {'category_accuracy':<30s}: {sum(metrics['category_accuracy'])/n:.4f}  (per-category count match)")
    print(f"  {'mean_iou':<30s}: {sum(metrics['mean_iou'])/n:.4f}  (mean best IoU vs GT)")
    print(f"  {'coverage':<30s}: {sum(metrics['coverage'])/n:.4f}  (fraction of GT comps matched IoU>0.3)")
    print(f"{'='*60}")
    print(f"  Evaluated on {n} / {len(pred_files)} pred files ({len(gt_map)} GT total)")

if __name__ == "__main__":
    main()
