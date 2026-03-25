"""Centralized configuration for the annotation pipeline."""
from pathlib import Path

# ── Base Paths ──
PROJECT_ROOT = Path("/home/xinrui/projects/PCB_structure_layout")
DATA_ROOT = Path("/home/xinrui/projects/data/ti_pcb")

# ── Input ──
DEFAULT_IMAGE_DIR = DATA_ROOT / "COCO_label/images/test"
IMAGES_TOP_DIR = DATA_ROOT / "images_top"

# ── YOLO Detection (Stage 1) ──
YOLO_MODEL = PROJECT_ROOT / "re-annotation/runs/yolov8m_pcb_v6/weights/best.pt"
TILE_SIZE = 640
OVERLAP = 160
TILE_CONF = 0.4
FULL_CONF = 0.3
NMS_IOU = 0.4

# YOLO class index → COCO category ID (v6: 9 classes, no Fuse)
YOLO_TO_COCO = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10}

CATEGORIES = [
    {"id": 1, "name": "Resistor"},
    {"id": 2, "name": "Capacitor"},
    {"id": 3, "name": "Inductor"},
    {"id": 4, "name": "Connector"},
    {"id": 5, "name": "Diode"},
    {"id": 7, "name": "Switch"},
    {"id": 8, "name": "Transistor"},
    {"id": 9, "name": "Integrated Circuit"},
    {"id": 10, "name": "Oscillator"},
]

CAT_ID_TO_NAME = {c["id"]: c["name"] for c in CATEGORIES}
CAT_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}

# ── Color Classification (Stage 2) ──
# Uses k-means dominant cluster on background pixels (outside component bboxes convex hull)

# ── Resolution Classification (Stage 3) ──
# Anchor: median min(w,h) of Resistor bodies (Capacitor fallback)
RESOLUTION_BINS = [
    ("R1", 0, 6),
    ("R2", 6, 8),
    ("R3", 8, 12),
    ("R4", 12, 16),
    ("R5", 16, 22),
    ("R6", 22, 28),
    ("R7", 28, 99999),
]

# ── Orientation Classification (Stage 4) ──
ORIENTATION_MODEL = PROJECT_ROOT / "re-orientation/checkpoints/best.pt"
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
MIN_CROP_SIZE = 8
OVERRIDE_CONF = 0.7  # Override auto-rule only if classifier confidence > this
