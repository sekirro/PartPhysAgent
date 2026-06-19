from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .types import BBox


def _as_bool_mask(mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(bool)


def load_rgb(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_rgb(image: Image.Image, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def save_mask(mask_bool_or_uint8, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask_bool_or_uint8)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    else:
        arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def read_mask(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_to_bbox(mask) -> BBox:
    arr = _as_bool_mask(mask)
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return BBox(0, 0, 0, 0)
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_area(bbox: BBox) -> int:
    return int(max(0, bbox.x2 - bbox.x1) * max(0, bbox.y2 - bbox.y1))


def mask_area(mask) -> int:
    return int(np.asarray(_as_bool_mask(mask)).sum())


def bbox_expand(bbox: BBox, pad_ratio: float, image_w: int, image_h: int) -> BBox:
    if bbox.is_empty:
        return BBox(0, 0, 0, 0)
    pad_x = int(round(bbox.width * pad_ratio))
    pad_y = int(round(bbox.height * pad_ratio))
    return BBox(
        max(0, bbox.x1 - pad_x),
        max(0, bbox.y1 - pad_y),
        min(int(image_w), bbox.x2 + pad_x),
        min(int(image_h), bbox.y2 + pad_y),
    )


def crop_image(image: Image.Image, bbox: BBox) -> Image.Image:
    return image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))


def crop_mask(mask, bbox: BBox) -> np.ndarray:
    arr = _as_bool_mask(mask)
    return arr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]


def apply_mask_white_bg(image: Image.Image, mask) -> Image.Image:
    image = image.convert("RGB")
    arr = np.asarray(image).copy()
    m = _as_bool_mask(mask)
    out = np.full_like(arr, 255)
    out[m] = arr[m]
    return Image.fromarray(out, mode="RGB")


def apply_mask_transparent_bg(image: Image.Image, mask) -> Image.Image:
    image = image.convert("RGBA")
    arr = np.asarray(image).copy()
    arr[..., 3] = _as_bool_mask(mask).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="RGBA")


def dim_non_mask_region(image: Image.Image, mask, dim_factor: float = 0.25) -> Image.Image:
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    m = _as_bool_mask(mask)
    dimmed = arr * float(dim_factor) + 255.0 * (1.0 - float(dim_factor))
    arr[~m] = dimmed[~m]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def overlay_mask(image: Image.Image, mask, alpha: float = 0.45, color=None) -> Image.Image:
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    m = _as_bool_mask(mask)
    if color is None:
        color = (255, 64, 64)
    color_arr = np.array(color, dtype=np.float32)
    arr[m] = (1.0 - alpha) * arr[m] + alpha * color_arr
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def overlay_multiple_masks(image: Image.Image, list_of_masks: Iterable, labels=None) -> Image.Image:
    colors = [
        (239, 83, 80),
        (66, 165, 245),
        (102, 187, 106),
        (255, 202, 40),
        (171, 71, 188),
        (38, 166, 154),
    ]
    out = image.convert("RGB")
    for idx, mask in enumerate(list_of_masks):
        out = overlay_mask(out, mask, alpha=0.42, color=colors[idx % len(colors)])
    if labels:
        draw = ImageDraw.Draw(out)
        for idx, mask in enumerate(list_of_masks):
            bbox = mask_to_bbox(mask)
            if bbox.is_empty:
                continue
            draw.text((bbox.x1 + 2, bbox.y1 + 2), str(labels[idx]), fill=colors[idx % len(colors)])
    return out


def mask_iou(mask_a, mask_b) -> float:
    a = _as_bool_mask(mask_a)
    b = _as_bool_mask(mask_b)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def mask_inside_ratio(mask, container_mask) -> float:
    m = _as_bool_mask(mask)
    c = _as_bool_mask(container_mask)
    area = m.sum()
    if area == 0:
        return 0.0
    return float(np.logical_and(m, c).sum() / area)


def _try_cv2_components(mask: np.ndarray, min_area: int) -> list[np.ndarray] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append(labels == idx)
    return comps


def connected_components_from_mask(mask, min_area: int) -> list[np.ndarray]:
    arr = _as_bool_mask(mask)
    if arr.sum() == 0:
        return []
    cv2_result = _try_cv2_components(arr, int(min_area))
    if cv2_result is not None:
        return cv2_result

    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    comps: list[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if not arr[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            coords = []
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            if len(coords) >= min_area:
                comp = np.zeros_like(arr, dtype=bool)
                ys, xs = zip(*coords)
                comp[np.array(ys), np.array(xs)] = True
                comps.append(comp)
    return comps


def resize_to_square_with_padding(image: Image.Image, size: int = 512, bg=(255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), bg)
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def bbox_from_xyxy(values, image_w: int | None = None, image_h: int | None = None) -> BBox:
    x1, y1, x2, y2 = [int(round(float(v))) for v in values]
    if image_w is not None:
        x1, x2 = max(0, x1), min(int(image_w), x2)
    if image_h is not None:
        y1, y2 = max(0, y1), min(int(image_h), y2)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return BBox(x1, y1, x2, y2)
