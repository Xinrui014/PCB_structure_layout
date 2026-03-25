"""Stage 2: Board color classification using k-means dominant cluster on background."""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import ConvexHull


def classify_rgb(r, g, b):
    """Classify a dominant RGB color into one of 5 board colors."""
    if r < 30 and g < 30 and b < 30:
        return "black"
    if r > 180 and g > 180 and b > 180:
        return "white"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


def get_background_mask(img_array, annotations, margin=10):
    """Create a mask of background pixels (outside component bboxes convex hull)."""
    ih, iw = img_array.shape[:2]

    if not annotations:
        # No annotations — use entire image
        return np.ones((ih, iw), dtype=bool)

    # Collect bbox corners
    corners = []
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        corners.extend([(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)])
    corners = np.array(corners)

    if len(corners) < 3:
        return np.ones((ih, iw), dtype=bool)

    try:
        hull = ConvexHull(corners)
        hull_pts = corners[hull.vertices]
    except Exception:
        return np.ones((ih, iw), dtype=bool)

    # Create mask outside convex hull (with margin)
    from PIL import Image as PILImage, ImageDraw
    mask_img = PILImage.new("L", (iw, ih), 255)
    draw = ImageDraw.Draw(mask_img)
    expanded = []
    cx, cy = hull_pts.mean(axis=0)
    for px, py in hull_pts:
        dx, dy = px - cx, py - cy
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            expanded.append((px + dx / dist * margin, py + dy / dist * margin))
        else:
            expanded.append((px, py))
    draw.polygon(expanded, fill=0)
    mask = np.array(mask_img) > 0

    # Also exclude area very close to edges (often black borders)
    edge = 20
    mask[:edge, :] = False
    mask[-edge:, :] = False
    mask[:, :edge] = False
    mask[:, -edge:] = False

    return mask


def classify_board_color(img_path, annotations):
    """Classify board color from image + annotations."""
    img = np.array(Image.open(str(img_path)).convert("RGB"))
    mask = get_background_mask(img, annotations)

    bg_pixels = img[mask]
    if len(bg_pixels) < 100:
        # Fallback: use all pixels
        bg_pixels = img.reshape(-1, 3)

    # K-means to find dominant color
    n_clusters = min(3, len(bg_pixels) // 10)
    if n_clusters < 1:
        n_clusters = 1
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    kmeans.fit(bg_pixels)

    # Pick largest cluster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_idx = labels[counts.argmax()]
    dominant_rgb = kmeans.cluster_centers_[dominant_idx].astype(int)

    color = classify_rgb(*dominant_rgb)
    return color


def run(output_dir, image_dir):
    """Add board_color to all annotation JSONs in output_dir."""
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)

    jsons = sorted(output_dir.glob("*.json"))
    print(f"[Stage 2: Color] Processing {len(jsons)} boards...")

    for i, json_path in enumerate(jsons):
        data = json.loads(json_path.read_text())
        board = json_path.stem

        # Find image
        img_path = None
        for ext in [".png", ".jpg", ".JPG", ".jpeg", ".JPEG", ".PNG"]:
            p = image_dir / f"{board}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            data["board_color"] = "unknown"
        else:
            color = classify_board_color(img_path, data.get("annotations", []))
            data["board_color"] = color

        json_path.write_text(json.dumps(data, indent=2))

        if (i + 1) % 50 == 0 or i == len(jsons) - 1:
            print(f"  [{i+1}/{len(jsons)}] {board}: {data.get('board_color', '?')}")

    print(f"[Stage 2: Color] Done.")
