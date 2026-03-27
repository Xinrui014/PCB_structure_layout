#!/usr/bin/env python3
"""
Layout Constraint Evaluation Script
Evaluates hard constraints and layout quality metrics for PCB layout generation.

Supports both:
- v1 format: [TYPE] [X] [Y] [W] [H] ... on 512x512
- v2 format: [COLOR_*] [R1-R7] [TYPE] [X] [Y] [W] [H] ... on 1280x720
"""

import json, re, torch, random, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CKPT = "/home/xinrui/projects/PCB_structure_layout/runs/qwen3b_lora/checkpoint-9000"
BASE = "Qwen/Qwen2.5-3B-Instruct"
TEST_JSONL = "/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/test.jsonl"
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
            i += 1
            continue
        if t in RESOLUTION_TOKENS:
            meta["resolution"] = t
            i += 1
            continue
        if t in CATEGORIES and i + 4 < len(tokens):
            try:
                x,y,w,h = int(tokens[i+1]), int(tokens[i+2]), int(tokens[i+3]), int(tokens[i+4])
                if w > 0 and h > 0:
                    components.append({"type":t,"x":x,"y":y,"w":w,"h":h})
                i += 5
                continue
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
        "oscillator": "OSCILLATOR", "oscillators": "OSCILLATOR",
    }
    counts = {}
    for match in re.finditer(r'(\d+)\s+([a-zA-Z]+)', prompt):
        num  = int(match.group(1))
        word = match.group(2).lower()
        if word in name_map:
            cat = name_map[word]
            counts[cat] = counts.get(cat, 0) + num
    return counts

def prompt_count_accuracy(prompt, pred):
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
    return 1.0 if len(gt) == len(pred) else 0.0

def category_accuracy(gt, pred):
    from collections import Counter
    gt_cnt   = Counter(c["type"] for c in gt)
    pred_cnt = Counter(c["type"] for c in pred)
    scores = []
    for cat, gt_n in gt_cnt.items():
        pred_n = pred_cnt.get(cat, 0)
        scores.append(min(pred_n, gt_n) / gt_n)
    return sum(scores)/len(scores) if scores else 0.0

def iou(a, b):
    ax1,ay1,ax2,ay2 = a["x"],a["y"],a["x"]+a["w"],a["y"]+a["h"]
    bx1,by1,bx2,by2 = b["x"],b["y"],b["x"]+b["w"],b["y"]+b["h"]
    inter_w = max(0, min(ax2,bx2)-max(ax1,bx1))
    inter_h = max(0, min(ay2,by2)-max(ay1,by1))
    inter   = inter_w * inter_h
    union   = a["w"]*a["h"] + b["w"]*b["h"] - inter
    return inter/union if union > 0 else 0.0

def mean_best_iou(gt, pred, iou_thresh=0.3):
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
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--test_jsonl", default=TEST_JSONL, help="Path to test.jsonl")
    parser.add_argument("--base_model", default=BASE, help="Base model name")
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.ckpt)
    model.eval()

    with open(args.test_jsonl) as f:
        test_data = [json.loads(l) for l in f]

    random.seed(args.seed)
    n = args.num_samples if args.num_samples > 0 else len(test_data)
    samples = random.sample(test_data, min(n, len(test_data)))

    system_msg = ("You are a PCB layout generator. Given a description of components, "
                  "output their placement as a sequence of tokens representing board-level "
                  "attributes followed by component type, x-position, y-position, width, "
                  "and height on a 1280x720 board.")

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

    for i, sample in enumerate(samples):
        msgs     = sample["messages"]
        prompt   = msgs[1]["content"]
        gt_text  = msgs[2]["content"]

        chat_msgs = [{"role":"system","content":system_msg},
                     {"role":"user","content":prompt}]
        text   = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred_text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        gt, gt_meta     = parse_layout(gt_text)
        pred, pred_meta = parse_layout(pred_text)

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

        cls_detail = " | ".join(
            f"{cat}:{v['expected']}→{v['predicted']}{'✓' if v['match'] else '✗'}"
            for cat, v in per_class.items()
        )
        print(f"[{i+1:3d}/{len(samples)}] GT={len(gt):3d} Pred={len(pred):3d} | "
              f"bound={br:.2f} overlap={or_:.2f} iou={miou:.3f} cov={cov:.2f} | meta={pred_meta}")
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
