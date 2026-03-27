#!/usr/bin/env python3
"""Build a gallery.html from infer_output bbox/orig images."""
import os, sys, glob

infer_dir = sys.argv[1] if len(sys.argv) > 1 else "."

# Find all bbox images
bbox_files = sorted(glob.glob(os.path.join(infer_dir, "*_bbox.png")))
print(f"Found {len(bbox_files)} samples")

rows = []
for bbox_path in bbox_files:
    name = os.path.basename(bbox_path).replace("_bbox.png", "")
    orig = f"{name}_orig.png"
    bbox = f"{name}_bbox.png"
    rows.append(f"""
    <div class="sample">
      <h3>{name}</h3>
      <div class="imgs">
        <div><img src="{orig}" onerror="this.style.display='none'"><br><small>Original</small></div>
        <div><img src="{bbox}"><br><small>GT vs Pred</small></div>
      </div>
    </div>""")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>v2 Layout Inference Gallery ({len(bbox_files)} samples)</title>
<style>
body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 20px; }}
h1 {{ text-align: center; }}
.sample {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; }}
.sample h3 {{ margin: 0 0 10px; color: #8cf; }}
.imgs {{ display: flex; gap: 20px; align-items: flex-start; }}
.imgs img {{ max-width: 600px; max-height: 400px; border: 1px solid #555; }}
small {{ color: #aaa; }}
</style></head>
<body>
<h1>v2 Qwen 3B Layout — {len(bbox_files)} Samples</h1>
{"".join(rows)}
</body></html>"""

out_path = os.path.join(infer_dir, "gallery.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Written to {out_path}")
