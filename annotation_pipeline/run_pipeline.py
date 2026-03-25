#!/usr/bin/env python3
"""
PCB Annotation Pipeline — Orchestrator

Usage:
    python run_pipeline.py --image_dir /path/to/images --output_dir pipeline_output/
    python run_pipeline.py  # defaults to COCO test set

Stages:
    1. YOLO v6 tiled detection → COCO JSON per board
    2. Board color classification (k-means dominant cluster)
    3. Resolution classification (median Resistor/Cap body size)
    4. Orientation classification (auto-rule + ResNet18)

Output: One enriched COCO JSON per board in output_dir/
"""
import argparse
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_IMAGE_DIR


def main():
    parser = argparse.ArgumentParser(description="PCB Annotation Pipeline")
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR),
                        help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="pipeline_output",
                        help="Output directory for annotation JSONs")
    parser.add_argument("--gpu", type=int, default=2,
                        help="GPU device index")
    parser.add_argument("--stages", type=str, default="1,2,3,4",
                        help="Comma-separated stage numbers to run (e.g. '1,2,3,4' or '2,3')")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip boards that already have output JSONs (stage 1 only)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    stages = [int(s.strip()) for s in args.stages.split(",")]

    print(f"=" * 60)
    print(f"PCB Annotation Pipeline")
    print(f"  Image dir:  {image_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  GPU:        {args.gpu}")
    print(f"  Stages:     {stages}")
    print(f"=" * 60)

    t0 = time.time()

    if 1 in stages:
        print(f"\n{'='*60}")
        print(f"Stage 1: YOLO Detection")
        print(f"{'='*60}")
        from stages.detect import run as run_detect
        run_detect(image_dir, output_dir, gpu=args.gpu)

    if 2 in stages:
        print(f"\n{'='*60}")
        print(f"Stage 2: Color Classification")
        print(f"{'='*60}")
        from stages.color import run as run_color
        run_color(output_dir, image_dir)

    if 3 in stages:
        print(f"\n{'='*60}")
        print(f"Stage 3: Resolution Classification")
        print(f"{'='*60}")
        from stages.resolution import run as run_resolution
        run_resolution(output_dir)

    if 4 in stages:
        print(f"\n{'='*60}")
        print(f"Stage 4: Orientation Classification")
        print(f"{'='*60}")
        from stages.orientation import run as run_orientation
        run_orientation(output_dir, image_dir, gpu=args.gpu)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Summary stats
    jsons = list(output_dir.glob("*.json"))
    print(f"\nTotal boards: {len(jsons)}")

    import json
    from collections import Counter
    colors = Counter()
    res_classes = Counter()
    total_anns = 0
    for jp in jsons:
        d = json.loads(jp.read_text())
        colors[d.get("board_color", "unknown")] += 1
        res_classes[d.get("resolution_class", "unknown")] += 1
        total_anns += len(d.get("annotations", []))

    print(f"Total annotations: {total_anns}")
    print(f"Colors: {dict(colors.most_common())}")
    print(f"Resolution: {dict(res_classes.most_common())}")


if __name__ == "__main__":
    main()
