#!/usr/bin/env python3
"""
PCB Layout Inference Script — Unified

Supports both:
- v1 format: [TYPE] [X] [Y] [W] [H] ... on 512x512 boards
- v2 format: [COLOR_*] [R1-R7] [TYPE] [X] [Y] [W] [H] ... on 1280x720 boards

Usage examples:
  CUDA_VISIBLE_DEVICES=0 python infer_layout.py \
      --backbone 3b \
      --ckpt runs/qwen3b_lora/checkpoint-9000 \
      --output_dir runs/qwen3b_lora/checkpoint-9000/vis_output \
      --num_samples 50

  CUDA_VISIBLE_DEVICES=0 python infer_layout.py \
      --backbone 3b \
      --ckpt runs/qwen3b_lora/checkpoint-9000 \
      --output_dir runs/qwen3b_lora/checkpoint-9000/vis_output \
      --paste \
      --num_samples 50

Backbone choices: 0.5b | 1.5b | 3b
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOARD_W = 1280
BOARD_H = 720
LEGACY_BOARD_SIZE = 512
CATEGORIES = {
    "RESISTOR", "CAPACITOR", "INDUCTOR", "CONNECTOR",
    "DIODE", "LED", "SWITCH", "TRANSISTOR", "IC", "OSCILLATOR",
}
CAT_NAME_TO_TOKEN = {
    "Resistor": "RESISTOR", "Capacitor": "CAPACITOR", "Inductor": "INDUCTOR",
    "Connector": "CONNECTOR", "Diode": "DIODE", "LED": "LED",
    "Switch": "SWITCH", "Transistor": "TRANSISTOR",
    "Integrated Circuit": "IC", "Integrated_Circuit": "IC", "IC": "IC",
    "Oscillator": "OSCILLATOR",
}
COLORS_RGB = {
    "RESISTOR": (255, 107, 107), "CAPACITOR": (78, 205, 196),
    "INDUCTOR": (69, 183, 209),  "CONNECTOR": (150, 206, 180),
    "DIODE": (255, 234, 167),    "LED": (221, 160, 221),
    "SWITCH": (240, 230, 140),   "TRANSISTOR": (255, 179, 71),
    "IC": (135, 206, 235),       "OSCILLATOR": (186, 85, 211),
}
SYSTEM_MSG = (
    "You are a PCB layout generator. Given a description of components, "
    "output their placement as a sequence of tokens representing board-level "
    "attributes followed by component type, x-position, y-position, width, "
    "and height on a 1280×720 board."
)

BACKBONE_REGISTRY = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
}

PROJECT_DIR   = Path("/home/xinrui/projects/PCB_structure_layout")
DATA_ROOT     = Path("/home/xinrui/projects/data/ti_pcb")
TEST_JSONL    = str(DATA_ROOT / "layout_data" / "v2_Color_Res_Class_xywh" / "test.jsonl")
METADATA_CSV  = str(DATA_ROOT / "components" / "metadata_train.csv")
TRAIN_IMG_DIR = str(DATA_ROOT / "COCO_label" / "cropped_512" / "images" / "train")
BOARD_IMG_DIR = str(DATA_ROOT / "layout_data" / "v2_Color_Res_Class_xywh" / "image" / "test")
EXCLUDE_FILE  = str(PROJECT_DIR / "component_pool" / "excluded_components.json")
RECLASS_FILE  = str(PROJECT_DIR / "component_pool" / "reclassified_components.json")

RESOLUTION_TOKENS = {f"R{i}" for i in range(1, 8)}
COLOR_TOKENS = {"COLOR_GREEN", "COLOR_RED", "COLOR_BLUE", "COLOR_BLACK", "COLOR_WHITE", "COLOR_GREY", "COLOR_GRAY"}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_layout(text):
    """Parse layout tokens from either v1 or v2 format."""
    components = []
    meta = {"board_color": None, "resolution": None}
    tokens = re.findall(r"\[([^\[\]]+)\]", text)
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
                x, y, w, h = (
                    int(tokens[i + 1]), int(tokens[i + 2]),
                    int(tokens[i + 3]), int(tokens[i + 4]),
                )
                if w > 0 and h > 0:
                    components.append((t, x, y, w, h))
                i += 5
                continue
            except ValueError:
                pass
        i += 1
    return components, meta

# ---------------------------------------------------------------------------
# Component Bank (used only when --paste is set)
# ---------------------------------------------------------------------------
class ComponentBank:
    def __init__(self, metadata_csv, board_img_dir, edge_margin=10,
                 exclude_file=None, reclass_file=None):
        self.board_img_dir = board_img_dir
        self.by_category = defaultdict(list)
        self._img_cache = {}

        excluded_ids = set()
        if exclude_file and os.path.exists(exclude_file):
            excluded_ids = set(json.load(open(exclude_file)))
            print(f"  Excluded crops: {len(excluded_ids)}")

        reclass_map = {}
        if reclass_file and os.path.exists(reclass_file):
            reclass_map = json.load(open(reclass_file))
            print(f"  Reclassified crops: {len(reclass_map)}")

        print(f"Loading component bank from {metadata_csv} ...")
        with open(metadata_csv) as f:
            for row in csv.DictReader(f):
                token_cat = CAT_NAME_TO_TOKEN.get(row["category_name"])
                if not token_cat:
                    continue
                x, y, w, h = (float(row["bbox_x"]), float(row["bbox_y"]),
                               float(row["bbox_w"]), float(row["bbox_h"]))
                if w <= 0 or h <= 0:
                    continue
                if row["id"] in excluded_ids:
                    continue
                if row["id"] in reclass_map:
                    token_cat = reclass_map[row["id"]]
                self.by_category[token_cat].append({
                    "source_image": row["source_image"],
                    "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                    "ar": w / h, "area": w * h,
                })
        for cat, entries in self.by_category.items():
            print(f"  {cat}: {len(entries)} crops")

    def _get_board_image(self, source_image):
        if source_image not in self._img_cache:
            path = os.path.join(self.board_img_dir, f"{source_image}.png")
            if not os.path.exists(path):
                return None
            self._img_cache[source_image] = Image.open(path).convert("RGB")
            if len(self._img_cache) > 200:
                del self._img_cache[next(iter(self._img_cache))]
        return self._img_cache.get(source_image)

    def find_match(self, cat, pred_w, pred_h, top_k=10):
        candidates = self.by_category.get(cat, [])
        if not candidates:
            return None
        pred_area = pred_w * pred_h
        pred_ar   = pred_w / pred_h if pred_h > 0 else 1.0
        pred_horiz = pred_w >= pred_h
        by_size = sorted(candidates, key=lambda e: abs(pred_area - e["area"]))
        size_filtered = by_size[:max(top_k * 5, 50)]
        same_orient = [e for e in size_filtered if (e["bbox_w"] >= e["bbox_h"]) == pred_horiz]
        pool = same_orient if len(same_orient) >= top_k else size_filtered
        pool.sort(key=lambda e: abs(pred_ar - e["ar"]))
        return random.choice(pool[:top_k])

    def load_crop(self, entry, target_w, target_h):
        board = self._get_board_image(entry["source_image"])
        if board is None:
            return None
        crop = board.crop((entry["bbox_x"], entry["bbox_y"],
                           entry["bbox_x"] + entry["bbox_w"],
                           entry["bbox_y"] + entry["bbox_h"]))
        return crop.resize((max(target_w, 1), max(target_h, 1)), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_bboxes(components, title=None, board_w=BOARD_W, board_h=BOARD_H):
    img = Image.new("RGB", (board_w, board_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for cat, x, y, w, h in components:
        color = COLORS_RGB.get(cat, (128, 128, 128))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x + 2, y + 2), cat[:3], fill=color)
    if title:
        draw.text((4, 4), title, fill=(0, 0, 0))
    return img

def paste_components(components, bank, top_k=10, board_w=BOARD_W, board_h=BOARD_H):
    img = Image.new("RGB", (board_w, board_h), (255, 255, 255))
    for cat, x, y, w, h in components:
        match = bank.find_match(cat, w, h, top_k=top_k)
        if match is None:
            continue
        crop = bank.load_crop(match, w, h)
        if crop is None:
            continue
        px = max(0, min(x, board_w - w))
        py = max(0, min(y, board_h - h))
        img.paste(crop, (px, py))
    return img

def make_side_by_side(*images, labels=None, gap=10):
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    total_w = sum(widths) + (len(images) - 1) * gap
    label_h = 20 if labels else 0
    canvas_h = max(heights) + label_h
    canvas = Image.new("RGB", (total_w, canvas_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    xoff = 0
    for i, img in enumerate(images):
        canvas.paste(img, (xoff, label_h))
        if labels and i < len(labels):
            draw.text((xoff + 4, 2), labels[i], fill=(220, 220, 220))
        xoff += img.width + gap
    return canvas

# ---------------------------------------------------------------------------
# HTML Gallery
# ---------------------------------------------------------------------------
def build_gallery_html(results, output_dir, paste=False):
    rows = []
    for r in results:
        name, prompt = r["name"], r["prompt"]
        gt_n, pred_n = len(r["gt_comps"]), len(r["pred_comps"])
        meta_txt = f"Pred meta: color={r['pred_meta'].get('board_color')} res={r['pred_meta'].get('resolution')}"
        paste_col = f'<div class="col"><h4>Pasted Components (Pred)</h4><img src="{name}_paste.png"></div>' if paste else ""
        rows.append(f"""
        <div class="sample">
          <h3>{name}</h3>
          <p class="prompt">{prompt}</p>
          <p class="stats">GT: {gt_n} | Pred: {pred_n} | {meta_txt}</p>
          <div class="row">
            <div class="col"><h4>BBox (GT | Pred)</h4><img src="{name}_bbox.png"></div>
            {paste_col}
            <div class="col"><h4>Original Board</h4><img src="{name}_orig.png"></div>
          </div>
        </div>""")
    legend = "".join(
        f'<span><span class="swatch" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{cat}</span>'
        for cat, c in COLORS_RGB.items()
    )
    html = f"""<!DOCTYPE html><html><head><meta charset=\"utf-8\">
<title>PCB Layout Inference</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#eee;margin:20px}}
h1{{text-align:center;color:#4ECDC4}} h3{{color:#FF6B6B;margin:0 0 4px}}
h4{{text-align:center;margin:4px 0;font-size:13px;color:#aaa}}
.sample{{background:#16213e;border-radius:12px;padding:16px;margin:16px auto;max-width:2800px}}
.prompt{{color:#888;font-size:13px;margin:4px 0}} .stats{{color:#666;font-size:12px}}
.row{{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}}
.col{{flex:1;min-width:300px;max-width:1280px}} .col img{{width:100%;border-radius:6px}}
.legend{{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:16px}}
.swatch{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:4px}}
</style></head><body>
<h1>PCB Layout Inference</h1>
<div class="legend">{legend}</div>
{"".join(rows)}
</body></html>"""
    path = os.path.join(output_dir, "results.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Gallery saved → {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PCB Layout Unified Inference")
    parser.add_argument("--backbone", choices=list(BACKBONE_REGISTRY.keys()), default="3b",
                        help="Backbone model size: 0.5b | 1.5b | 3b")
    parser.add_argument("--ckpt", default=str(PROJECT_DIR / "runs/qwen3b_lora/checkpoint-9000"),
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save results (default: <ckpt>/vis_output)")
    parser.add_argument("--paste", action="store_true",
                        help="Paste real component crops onto layout (Stage 2)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of test samples to run (-1 = all)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K candidates for component matching (used with --paste)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = str(Path(args.ckpt) / "vis_output")
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = BACKBONE_REGISTRY[args.backbone]
    print(f"Backbone : {args.backbone} → {base_model}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output dir: {args.output_dir}")
    print(f"Paste mode: {'ON' if args.paste else 'OFF (bbox only)'}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.ckpt)
    model.eval()

    bank = None
    if args.paste:
        bank = ComponentBank(METADATA_CSV, TRAIN_IMG_DIR,
                             exclude_file=EXCLUDE_FILE, reclass_file=RECLASS_FILE)

    with open(TEST_JSONL) as f:
        test_data = [json.loads(l) for l in f]
    n = args.num_samples if args.num_samples > 0 else len(test_data)
    samples = random.sample(test_data, min(n, len(test_data)))
    print(f"\nRunning inference on {len(samples)} samples ...\n")

    results = []
    for idx, sample in enumerate(samples):
        msgs       = sample["messages"]
        prompt     = msgs[1]["content"]
        gt_text    = msgs[2]["content"]
        image_name = sample.get("_meta", {}).get("image", f"sample_{idx:04d}")

        chat_msgs = [{"role": "system", "content": SYSTEM_MSG},
                     {"role": "user",   "content": prompt}]
        text   = tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gt_comps, gt_meta = parse_layout(gt_text)
        pred_comps, pred_meta = parse_layout(pred_text)
        print(f"  [{idx+1:3d}/{len(samples)}] {image_name}: GT={len(gt_comps)} Pred={len(pred_comps)} | meta={pred_meta}")

        gt_bbox   = draw_bboxes(gt_comps, title="GT")
        pred_bbox = draw_bboxes(pred_comps, title="Pred")
        bbox_img  = make_side_by_side(gt_bbox, pred_bbox, labels=["GT", "Pred"])
        bbox_img.save(os.path.join(args.output_dir, f"{image_name}_bbox.png"))

        if args.paste and bank is not None:
            paste_img = paste_components(pred_comps, bank, top_k=args.top_k)
            paste_img.save(os.path.join(args.output_dir, f"{image_name}_paste.png"))

        orig_path = os.path.join(BOARD_IMG_DIR, f"{image_name}.png")
        orig_out  = os.path.join(args.output_dir, f"{image_name}_orig.png")
        if os.path.exists(orig_path):
            Image.open(orig_path).save(orig_out)
        else:
            Image.new("RGB", (BOARD_W, BOARD_H), (30, 30, 30)).save(orig_out)

        with open(os.path.join(args.output_dir, f"{image_name}_pred.txt"), "w") as f:
            f.write(pred_text)

        results.append({"name": image_name, "prompt": prompt,
                        "gt_comps": gt_comps, "pred_comps": pred_comps,
                        "gt_meta": gt_meta, "pred_meta": pred_meta})

    build_gallery_html(results, args.output_dir, paste=args.paste)
    print(f"\nDone! {len(results)} samples → {args.output_dir}")

if __name__ == "__main__":
    main()
