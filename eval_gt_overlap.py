#!/usr/bin/env python3
"""Quick GT overlap/boundary stats from test.jsonl"""
import json, re
from pathlib import Path

CANVAS_W, CANVAS_H = 1280, 720
CATEGORIES = {"RESISTOR","CAPACITOR","INDUCTOR","CONNECTOR","DIODE","LED","SWITCH","TRANSISTOR","IC","OSCILLATOR"}
RESOLUTION_TOKENS = {f"R{i}" for i in range(1, 8)}
COLOR_TOKENS = {"COLOR_GREEN","COLOR_RED","COLOR_BLUE","COLOR_BLACK","COLOR_WHITE","COLOR_GREY","COLOR_GRAY"}

def parse_layout(text):
    components = []
    tokens = re.findall(r'\[([^\[\]]+)\]', text)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in COLOR_TOKENS or t in RESOLUTION_TOKENS:
            i += 1; continue
        if t in CATEGORIES and i + 4 < len(tokens):
            try:
                x,y,w,h = int(tokens[i+1]),int(tokens[i+2]),int(tokens[i+3]),int(tokens[i+4])
                if w > 0 and h > 0:
                    components.append({"type":t,"x":x,"y":y,"w":w,"h":h})
                i += 5; continue
            except ValueError: pass
        i += 1
    return components

def boundary_rate(comps):
    if not comps: return 1.0
    ok = sum(1 for c in comps if c["x"]>=0 and c["y"]>=0 and c["x"]+c["w"]<=CANVAS_W and c["y"]+c["h"]<=CANVAS_H)
    return ok/len(comps)

def overlap_rate(comps):
    if len(comps) <= 1: return 1.0
    def intersects(a, b):
        return not (a["x"]+a["w"]<=b["x"] or b["x"]+b["w"]<=a["x"] or a["y"]+a["h"]<=b["y"] or b["y"]+b["h"]<=a["y"])
    no_overlap = 0
    for i, c in enumerate(comps):
        if not any(intersects(c, comps[j]) for j in range(len(comps)) if j!=i):
            no_overlap += 1
    return no_overlap/len(comps)

import sys
test_jsonl = sys.argv[1]
with open(test_jsonl) as f:
    samples = [json.loads(l) for l in f]

brs, ors = [], []
for s in samples:
    gt_text = s["messages"][2]["content"]
    comps = parse_layout(gt_text)
    brs.append(boundary_rate(comps))
    ors.append(overlap_rate(comps))

print(f"GT stats over {len(samples)} test samples:")
print(f"  boundary_rate: {sum(brs)/len(brs):.4f}")
print(f"  overlap_rate (no-overlap): {sum(ors)/len(ors):.4f}")
print(f"  → {1-sum(ors)/len(ors):.4f} fraction overlapping")
