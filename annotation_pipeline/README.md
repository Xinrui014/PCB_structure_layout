# Annotation Pipeline

Automated PCB board annotation pipeline: detect components, classify board color, resolution, and orientation — then review/correct via web UIs.

## Directory Structure

```
annotation_pipeline/
├── config.py                    # Centralized paths, thresholds, model configs
├── run_pipeline.py              # Orchestrator (4 stages), CLI flags
├── stages/
│   ├── detect.py                # Stage 1: YOLO v6 tiled detection
│   ├── color.py                 # Stage 2: K-means background color classification
│   ├── resolution.py            # Stage 3: Median R/C body size → R1–R7
│   └── orientation.py           # Stage 4: Auto-rule + ResNet18 classifier
├── annotator.py                 # Web UI: bbox + category editing
├── color_analysis.py            # Web UI: board color review + relabeling
├── resolution_analysis.py       # Web UI: resolution review + board deletion
├── orientation_review_server.py # Web UI: orientation review + relabel + delete
└── test_output/                 # Pipeline output (test set example)
    ├── annotation/              # 426 board annotation JSONs (COCO format)
    └── metadata/
        ├── board_colors_v2.json          # Board color labels + RGB + clusters
        └── resolution_deleted_boards.json # Flagged boards for exclusion
```

## Step 1: Run the Pipeline

Detect components, classify color/resolution/orientation for a set of board images.

```bash
cd /home/xinrui/projects/PCB_structure_layout/annotation_pipeline

CUDA_VISIBLE_DEVICES=2 conda run -n pcbgen python run_pipeline.py \
  --image_dir /path/to/board/images \
  --output_dir my_output/ \
  --gpu 2
```

**Stages:**
1. **Detect** — YOLO v6 tiled inference (640×640, overlap=160, agnostic NMS)
2. **Color** — K-means on background pixels → green/blue/red/black/white
3. **Resolution** — Median min(w,h) of Resistor bboxes → R1–R7
4. **Orientation** — Auto-rule (w≥h→0°, w<h→90°) + ResNet18 override for ambiguous cases

Run specific stages only:
```bash
python run_pipeline.py --output_dir my_output/ --stages 2,3
```

**Output:** One COCO-format JSON per board in `my_output/annotation/`, metadata in `my_output/metadata/`.

## Step 2: Review & Correct

After the pipeline runs, launch the web UIs to review and correct results.

### Setup

```bash
cd /home/xinrui/projects/PCB_structure_layout/annotation_pipeline
IMG=/path/to/board/images   # e.g. /home/xinrui/projects/data/ti_pcb/COCO_label/images/test
OUT=test_output             # pipeline output directory
```

### 2a. Annotator — Bbox & Category Editing (port 8899)

Review/edit component bounding boxes and categories.

```bash
conda run -n pcbgen python annotator.py \
  --anno_dir $OUT/annotation \
  --image_dir $IMG \
  --port 8899
```

**Keyboard shortcuts:**
- `N` = draw mode (drag to create bbox)
- `X` / `Delete` / `F` = delete selected bbox
- `1`–`9` = change category (1=Cap, 2=Con, 3=Dio, 4=Ind, 5=IC, 6=LED, 7=Res, 8=Swi, 9=Tra)
- `Ctrl+S` = save
- `←` / `→` or `A` / `D` = navigate boards
- `R` = reset zoom
- `\` / `/` = toggle sidebar/annotation panel

**Modifies:** `$OUT/annotation/<board>.json` directly.

### 2b. Color Analysis — Board Color Review (port 8896)

Review board color classification, pick correct cluster, relabel.

```bash
conda run -n pcbgen python color_analysis.py \
  --data_file $OUT/metadata/board_colors_v2.json \
  --image_dir $IMG \
  --port 8896
```

**Features:**
- Dashboard: color distribution + RGB histogram
- Gallery (`/gallery`): 3×5 grid, click cluster dots to change dominant color
- Hotkeys: `G`=green, `B`=blue, `R`=red, `K`=black, `W`=white, `O`=orange
- Select cards → press hotkey to relabel

**Modifies:** `$OUT/metadata/board_colors_v2.json` only.

### 2c. Resolution Analysis — R1–R7 Review (port 8895)

Review resolution classification based on median Resistor/Capacitor body size.

```bash
conda run -n pcbgen python resolution_analysis.py \
  --anno_dir $OUT/annotation \
  --image_dir $IMG \
  --port 8895
```

**Resolution bins:**
| Class | Body size (px) |
|-------|---------------|
| R1    | [0, 6)        |
| R2    | [6, 8)        |
| R3    | [8, 12)       |
| R4    | [12, 16)      |
| R5    | [16, 22)      |
| R6    | [22, 28)      |
| R7    | [28, +∞)      |

**Modifies:** `$OUT/metadata/resolution_deleted_boards.json` (flag file, boards not actually deleted).

### 2d. Orientation Review — Orientation Correction (port 8892)

Browse component crops by angle/category, relabel or delete.

```bash
conda run -n pcbgen python orientation_review_server.py \
  --anno_dir $OUT/annotation \
  --image_dir $IMG \
  --port 8892
```

**Selection:**
- Click = select/deselect
- Shift+Click = range select
- `A` = select all (when none selected)
- `Esc` = clear selection

**Relabel hotkeys (WASD grid):**
| Key | Angle  | Arrow |
|-----|--------|-------|
| Q   | 135°   | ↖     |
| W   | 90°    | ↑     |
| E   | 45°    | ↗     |
| A   | 180°   | ←     |
| D   | 0°     | →     |
| Z   | 225°   | ↙     |
| S   | 270°   | ↓     |
| C   | 315°   | ↘     |

**Delete:** `X` or `Delete` key (with confirmation).

**Navigation:** `←` / `→` = prev/next page.

**Modifies:** `$OUT/annotation/<board>.json` directly (removes annotations on delete, updates orientation on relabel).

## Data Source of Truth

| Data              | Source file                              | Modified by          |
|-------------------|------------------------------------------|----------------------|
| Bboxes/categories | `annotation/<board>.json`                | Annotator (8899)     |
| Orientation       | `annotation/<board>.json`                | Orientation (8892)   |
| Board color       | `metadata/board_colors_v2.json`          | Color Analysis (8896)|
| Resolution flags  | `metadata/resolution_deleted_boards.json`| Resolution (8895)    |
| Resolution class  | `annotation/<board>.json` (read-only)    | Pipeline stage 3     |

> **Note:** Color and resolution data exist in both `metadata/` files and inside annotation JSONs.
> The `metadata/` files are the source of truth for corrections made via web UIs.
> Sync them when building the final dataset.

## Models Used

- **YOLO v6:** `re-annotation/runs/yolov8m_pcb_v6/weights/best.pt` — 9 classes, mAP@0.5=0.885
- **Orientation ResNet18 v1:** `re-orientation/checkpoints/best.pt` — val_acc=0.944

## Conda Environment

```bash
# Create (if needed)
conda create -n pcbgen python=3.10
conda activate pcbgen
pip install ultralytics torch torchvision scikit-learn flask scipy pillow
```

Required packages: `ultralytics`, `torch`, `torchvision`, `scikit-learn`, `flask`, `scipy`, `pillow`
