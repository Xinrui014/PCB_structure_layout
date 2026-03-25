#!/usr/bin/env python3
"""
Layout Constraint Evaluation Script
Evaluates hard constraints and layout quality metrics for PCB layout generation.

Metrics:
  Hard constraints (binary per component):
    1. boundary_rate     — % components fully within 512x512
    2. overlap_rate      — % components with zero overlap with others
  
  Soft / quality metrics (vs. ground truth):
    3. count_accuracy    — exact match of total component count
    4. category_accuracy — per-category count match (macro avg)
    5. iou_score         — mean best IoU (each GT comp matched to nearest pred)
    6. coverage          — what fraction of GT components have a matching pred (IoU > 0.3)
"""

import json, re, torch, random, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ──────────────────────────────────────────────────────────────────
CKPT     = "/home/xinrui/projects/PCB_structure_layout/runs/qwen3b_lora/checkpoint-9000"
BASE     = "Qwen/Qwen2.5-3B-Instruct"
TEST_JSONL = "/home/xinrui/projects/data/ti_pcb/layout_data/test.jsonl"
CANVAS   = 512
CATEGORIES = {"RESISTOR","CAPACITOR","INDUCTOR","CONNECTOR","DIODE","LED","SWITCH","TRANSISTOR","IC"}

# ── Parsing ──────────────────────────────────────────────────────────────────
def parse_layout(text):
    components = []
    tokens = re.findall(r'\[(\w+)\]', text)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in CATEGORIES and i + 4 < len(tokens):
            try:
                x,y,w,h = int(tokens[i+1]),int(tokens[i+2]),int(tokens[i+3]),int(tokens[i+4])
                components.append({"type":t,"x":x,"y":y,"w":w,"h":h})
                i += 5; continue
            except ValueError: pass
        i += 1
    return components

# ── Hard Constraint Metrics ──────────────────────────────────────────────────
def boundary_rate(comps, canvas=CANVAS):
    """Fraction of components fully within canvas."""
    if not comps: return 1.0
    ok = sum(1 for c in comps if c["x"] >= 0 and c["y"] >= 0
             and c["x"]+c["w"] <= canvas and c["y"]+c["h"] <= canvas)
    return ok / len(comps)

def overlap_rate(comps):
    """Fraction of components that do NOT overlap with any other component."""
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
    """Parse expected per-class counts directly from the prompt string.
    e.g. 'Generate a PCB layout with 30 capacitors, 4 ICs, 1 connector'
    → {'CAPACITOR': 30, 'IC': 4, 'CONNECTOR': 1}
    """
    import re
    name_map = {
        "capacitor": "CAPACITOR", "capacitors": "CAPACITOR",
        "resistor": "RESISTOR",   "resistors": "RESISTOR",
        "inductor": "INDUCTOR",   "inductors": "INDUCTOR",
        "connector": "CONNECTOR", "connectors": "CONNECTOR",
        "diode": "DIODE",         "diodes": "DIODE",
        "led": "LED",             "leds": "LED",
        "switch": "SWITCH",       "switches": "SWITCH",
        "transistor": "TRANSISTOR","transistors": "TRANSISTOR",
        "ic": "IC",               "ics": "IC",
    }
    counts = {}
    # match patterns like "30 capacitors" or "4 ICs"
    for match in re.finditer(r'(\d+)\s+([a-zA-Z]+)', prompt):
        num  = int(match.group(1))
        word = match.group(2).lower()
        if word in name_map:
            cat = name_map[word]
            counts[cat] = counts.get(cat, 0) + num
    return counts

def prompt_count_accuracy(prompt, pred):
    """
    Hard constraint: does the predicted layout match the per-class counts
    specified in the prompt?
    Returns:
      - per_class_match: dict {cat: (expected, predicted, match)}
      - exact_match_rate: fraction of required classes with exact count match
      - overall_exact: 1.0 if ALL classes match exactly
    """
    expected = parse_prompt_counts(prompt)
    from collections import Counter
    pred_cnt = Counter(c["type"] for c in pred)

    per_class = {}
    matches = []
    for cat, exp_n in expected.items():
        pred_n = pred_cnt.get(cat, 0)
        match  = (exp_n == pred_n)
        per_class[cat] = {"expected": exp_n, "predicted": pred_n, "match": match}
        matches.append(match)

    exact_match_rate = sum(matches) / len(matches) if matches else 1.0
    overall_exact    = 1.0 if all(matches) else 0.0
    return per_class, exact_match_rate, overall_exact

def count_accuracy(gt, pred):
    """1 if exact total count match, else 0."""
    return 1.0 if len(gt) == len(pred) else 0.0

def category_accuracy(gt, pred):
    """Per-category count match, macro averaged over present GT categories."""
    from collections import Counter
    gt_cnt   = Counter(c["type"] for c in gt)
    pred_cnt = Counter(c["type"] for c in pred)
    scores = []
    for cat, gt_n in gt_cnt.items():
        pred_n = pred_cnt.get(cat, 0)
        scores.append(min(pred_n, gt_n) / gt_n)   # precision-like per category
    return sum(scores)/len(scores) if scores else 0.0

def iou(a, b):
    """IoU between two bboxes [x,y,w,h]."""
    ax1,ay1,ax2,ay2 = a["x"],a["y"],a["x"]+a["w"],a["y"]+a["h"]
    bx1,by1,bx2,by2 = b["x"],b["y"],b["x"]+b["w"],b["y"]+b["h"]
    inter_w = max(0, min(ax2,bx2)-max(ax1,bx1))
    inter_h = max(0, min(ay2,by2)-max(ay1,by1))
    inter   = inter_w * inter_h
    union   = a["w"]*a["h"] + b["w"]*b["h"] - inter
    return inter/union if union > 0 else 0.0

def mean_best_iou(gt, pred, iou_thresh=0.3):
    """For each GT comp, find best matching pred IoU. Returns mean IoU and coverage."""
    if not gt or not pred: return 0.0, 0.0
    best_ious = []
    for g in gt:
        best = max((iou(g, p) for p in pred), default=0.0)
        best_ious.append(best)
    mean_iou = sum(best_ious) / len(best_ious)
    coverage = sum(1 for v in best_ious if v >= iou_thresh) / len(best_ious)
    return mean_iou, coverage

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=CKPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.ckpt)
    model.eval()

    with open(TEST_JSONL) as f:
        test_data = [json.loads(l) for l in f]

    random.seed(args.seed)
    n = args.num_samples if args.num_samples > 0 else len(test_data)
    samples = random.sample(test_data, min(n, len(test_data)))

    system_msg = ("You are a PCB layout generator. Given a description of components, "
                  "output their placement as a sequence of tokens representing component "
                  "type, x-position, y-position, width, and height on a 512x512 board.")

    # Accumulators
    metrics = {
        "boundary_rate": [],
        "overlap_rate": [],
        "prompt_class_exact_rate": [],   # fraction of required classes with exact count
        "prompt_all_classes_exact": [],  # 1 if ALL classes match prompt exactly
        "count_accuracy": [],            # exact total count match vs GT
        "category_accuracy": [],         # per-category match vs GT
        "mean_iou": [],
        "coverage": [],
    }

    for i, sample in enumerate(samples):
        msgs     = sample["messages"]
        prompt   = msgs[1]["content"]
        gt_text  = msgs[2]["content"]

        chat_msgs = [{"role":"system","content":system_msg},
                     {"role":"user","content":prompt}]
        text   = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        pred_text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        gt   = parse_layout(gt_text)
        pred = parse_layout(pred_text)

        br   = boundary_rate(pred)
        or_  = overlap_rate(pred)
        ca   = count_accuracy(gt, pred)
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

        # Per-class detail string
        cls_detail = " | ".join(
            f"{cat}:{v['expected']}→{v['predicted']}{'✓' if v['match'] else '✗'}"
            for cat, v in per_class.items()
        )
        print(f"[{i+1:3d}/{len(samples)}] GT={len(gt):3d} Pred={len(pred):3d} | "
              f"bound={br:.2f} overlap={or_:.2f} iou={miou:.3f} cov={cov:.2f}")
        print(f"  Prompt counts: {cls_detail}")

    print("\n" + "="*60)
    print("LAYOUT EVALUATION SUMMARY")
    print("="*60)
    print("  -- Hard Constraints --")
    print(f"  {'boundary_rate':<30s}: {sum(metrics['boundary_rate'])/len(metrics['boundary_rate']):.4f}  (fraction of comps inside canvas)")
    print(f"  {'overlap_rate':<30s}: {sum(metrics['overlap_rate'])/len(metrics['overlap_rate']):.4f}  (fraction of comps with no overlap)")
    print(f"  {'prompt_class_exact_rate':<30s}: {sum(metrics['prompt_class_exact_rate'])/len(metrics['prompt_class_exact_rate']):.4f}  (fraction of classes matching prompt count)")
    print(f"  {'prompt_all_classes_exact':<30s}: {sum(metrics['prompt_all_classes_exact'])/len(metrics['prompt_all_classes_exact']):.4f}  (fraction of samples where ALL classes match)")
    print("  -- Quality vs. GT --")
    print(f"  {'count_accuracy':<30s}: {sum(metrics['count_accuracy'])/len(metrics['count_accuracy']):.4f}  (exact total count match)")
    print(f"  {'category_accuracy':<30s}: {sum(metrics['category_accuracy'])/len(metrics['category_accuracy']):.4f}  (per-category count match)")
    print(f"  {'mean_iou':<30s}: {sum(metrics['mean_iou'])/len(metrics['mean_iou']):.4f}  (mean best IoU vs GT)")
    print(f"  {'coverage':<30s}: {sum(metrics['coverage'])/len(metrics['coverage']):.4f}  (fraction of GT comps matched IoU>0.3)")
    print("="*60)
    print(f"  Evaluated on {len(samples)} samples")
    print(f"  Checkpoint: {args.ckpt}")

if __name__ == "__main__":
    main()
