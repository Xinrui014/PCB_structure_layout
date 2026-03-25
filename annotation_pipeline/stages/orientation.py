"""Stage 4: Orientation classification using auto-rule + ResNet18 classifier."""
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ORIENTATION_MODEL, ANGLES, MIN_CROP_SIZE, OVERRIDE_CONF


def find_image(board_name, image_dir):
    for ext in [".png", ".jpg", ".JPG", ".jpeg", ".JPEG", ".PNG"]:
        p = Path(image_dir) / f"{board_name}{ext}"
        if p.exists():
            return p
    return None


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ANGLES))
    ckpt = torch.load(str(ORIENTATION_MODEL), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    val_acc = ckpt.get("val_acc", "?")
    print(f"  Loaded orientation model (val_acc={val_acc})")
    return model


def classify_orientation(model, img_crop, transform, device):
    """Run orientation classifier on a crop. Returns (angle, confidence)."""
    tensor = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    return ANGLES[idx.item()], conf.item()


def run(output_dir, image_dir, gpu=2):
    """Add orientation to each annotation in all JSONs."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))

    output_dir = Path(output_dir)
    image_dir = Path(image_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 4: Orientation] Device: {device}")

    model = load_model(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    jsons = sorted(output_dir.glob("*.json"))
    print(f"[Stage 4: Orientation] Processing {len(jsons)} boards...")

    total_anns = 0
    overridden = 0

    for i, json_path in enumerate(jsons):
        data = json.loads(json_path.read_text())
        board = json_path.stem

        img_path = find_image(board, image_dir)
        if img_path is None:
            # Set all orientations to auto-rule
            for ann in data.get("annotations", []):
                w, h = ann["bbox"][2], ann["bbox"][3]
                ann["orientation"] = 0 if w >= h else 90
            json_path.write_text(json.dumps(data, indent=2))
            continue

        img = Image.open(str(img_path)).convert("RGB")
        img_w, img_h = img.size

        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]
            total_anns += 1

            # Auto-rule
            auto_angle = 0 if w >= h else 90

            # Crop for classifier
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                ann["orientation"] = auto_angle
                continue

            crop = img.crop((x1, y1, x2, y2))
            pred_angle, conf = classify_orientation(model, crop, transform, device)

            # Override auto-rule only for non-0/90 predictions with high confidence
            if pred_angle not in (0, 90) and conf > OVERRIDE_CONF:
                ann["orientation"] = pred_angle
                overridden += 1
            else:
                ann["orientation"] = auto_angle

        json_path.write_text(json.dumps(data, indent=2))

        if (i + 1) % 50 == 0 or i == len(jsons) - 1:
            print(f"  [{i+1}/{len(jsons)}] {board}")

    print(f"[Stage 4: Orientation] Done. Total annotations: {total_anns}, Classifier overrides: {overridden}")
