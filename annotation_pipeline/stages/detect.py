"""Stage 1: YOLO v6 tiled detection with NMS + contained-box filter."""
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    YOLO_MODEL, TILE_SIZE, OVERLAP, TILE_CONF, FULL_CONF, NMS_IOU,
    YOLO_TO_COCO, CATEGORIES,
)


def get_tiles(w, h):
    step = TILE_SIZE - OVERLAP
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)
            tiles.append((max(0, x2 - TILE_SIZE), max(0, y2 - TILE_SIZE), x2, y2))
    return list(set(tiles))


def nms_boxes(boxes, scores, classes, iou_t):
    """Agnostic NMS — suppresses overlapping boxes regardless of class."""
    if not boxes:
        return [], [], []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)
    order = scores.argsort()[::-1]
    boxes, scores, classes = boxes[order], scores[order], classes[order]
    keep = []
    suppressed = set()
    for i in range(len(boxes)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            a2 = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou >= iou_t:
                suppressed.add(j)
    return boxes[keep].tolist(), scores[keep].tolist(), classes[keep].tolist()


def filter_contained(boxes, scores, classes):
    """Remove smaller boxes fully contained inside larger boxes (shared-edge tolerance)."""
    if len(boxes) < 2:
        return boxes, scores, classes
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)
    n = len(boxes)
    remove = set()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in range(n):
        if i in remove:
            continue
        for j in range(n):
            if j == i or j in remove:
                continue
            if areas[j] >= areas[i]:
                continue
            tol = 5
            if (boxes[j, 0] >= boxes[i, 0] - tol and
                boxes[j, 1] >= boxes[i, 1] - tol and
                boxes[j, 2] <= boxes[i, 2] + tol and
                boxes[j, 3] <= boxes[i, 3] + tol):
                remove.add(j)
    keep = [i for i in range(n) if i not in remove]
    return boxes[keep].tolist(), scores[keep].tolist(), classes[keep].tolist()


def detect_board(model, img_path):
    """Run tiled + full-image detection on a single board image. Returns COCO annotations list."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    all_boxes, all_scores, all_classes = [], [], []

    # Tiled detection
    tiles = get_tiles(w, h)
    for tx1, ty1, tx2, ty2 in tiles:
        tile = img[ty1:ty2, tx1:tx2]
        results = model(tile, conf=TILE_CONF, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Map to global coords
                all_boxes.append([x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1])
                all_scores.append(conf)
                all_classes.append(cls)

    # Full-image detection
    results = model(img, conf=FULL_CONF, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_classes.append(cls)

    # NMS
    all_boxes, all_scores, all_classes = nms_boxes(all_boxes, all_scores, all_classes, NMS_IOU)

    # Filter contained boxes
    all_boxes, all_scores, all_classes = filter_contained(all_boxes, all_scores, all_classes)

    # Convert to COCO annotations
    annotations = []
    for i, (box, score, cls) in enumerate(zip(all_boxes, all_scores, all_classes)):
        coco_id = YOLO_TO_COCO.get(cls)
        if coco_id is None:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        annotations.append({
            "id": i + 1,
            "image_id": 0,
            "category_id": coco_id,
            "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
            "area": round(bw * bh, 2),
            "score": round(score, 4),
            "iscrowd": 0,
        })

    return annotations


def run(image_dir, output_dir, gpu=2):
    """Run detection on all images in image_dir, save COCO JSONs to output_dir."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1: Detection] Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(str(YOLO_MODEL))

    exts = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix in exts])
    print(f"[Stage 1: Detection] Processing {len(images)} images...")

    for i, img_path in enumerate(images):
        board = img_path.stem
        out_path = output_dir / f"{board}.json"

        annotations = detect_board(model, img_path)

        # Build COCO JSON
        h, w = cv2.imread(str(img_path)).shape[:2]
        coco = {
            "images": [{"id": 0, "file_name": img_path.name, "width": w, "height": h}],
            "categories": CATEGORIES,
            "annotations": annotations,
        }
        out_path.write_text(json.dumps(coco, indent=2))

        if (i + 1) % 50 == 0 or i == len(images) - 1:
            print(f"  [{i+1}/{len(images)}] {board}: {len(annotations)} detections")

    print(f"[Stage 1: Detection] Done. Output: {output_dir}")
    return output_dir
