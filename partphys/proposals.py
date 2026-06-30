from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import (
    bbox_area,
    connected_components_from_mask,
    load_rgb,
    mask_area,
    mask_inside_ratio,
    mask_iou,
    mask_to_bbox,
    overlay_mask,
    read_mask,
    save_mask,
    save_rgb,
)
from .part_traits import is_collection_part, is_main_like_part, part_text
from .types import BBox, MaskCandidate, PartSpec


def _write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _candidate_from_mask(
    mask: np.ndarray,
    source: str,
    output_dir: Path,
    idx: int,
    image,
    object_mask: np.ndarray,
    prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
    stability_score=None,
    predicted_iou=None,
) -> MaskCandidate:
    h, w = object_mask.shape
    cid = f"candidate_{idx:03d}"
    mask_path = output_dir / f"{cid}_mask.png"
    overlay_path = output_dir / f"{cid}_overlay.png"
    save_mask(mask, mask_path)
    save_rgb(overlay_mask(image, mask), overlay_path)
    area = mask_area(mask)
    bbox = mask_to_bbox(mask)
    bbox_pixels = bbox_area(bbox)
    object_area = max(1, mask_area(object_mask))
    item_metadata = dict(metadata or {})
    item_metadata.setdefault("source", source)
    item_metadata.setdefault("object_area_ratio", float(area / object_area))
    item_metadata.setdefault("bbox_area_ratio", float(bbox_pixels / object_area))
    item_metadata.setdefault("boundary_quality", float(area / max(1, bbox_pixels)))
    if stability_score is not None:
        item_metadata.setdefault("stability_score", stability_score)
    if predicted_iou is not None:
        item_metadata.setdefault("predicted_iou", predicted_iou)
    cand = MaskCandidate(
        candidate_id=cid,
        source=source,
        mask_path=str(mask_path),
        bbox=bbox,
        area=area,
        area_ratio=float(area / max(1, h * w)),
        inside_object_ratio=mask_inside_ratio(mask, object_mask),
        stability_score=stability_score,
        predicted_iou=predicted_iou,
        prompt=prompt,
        part_scores={},
        metadata=item_metadata,
    )
    _write_json(output_dir / f"{cid}.json", cand.to_dict())
    return cand


def _dedup_candidates(candidates: list[MaskCandidate]) -> list[MaskCandidate]:
    source_priority = {
        "color_prior": 7,
        "text_box_sam": 6,
        "sam_auto": 5,
        "object_body": 4,
        "appearance_cluster": 3,
        "vlm_box_sam": 2,
        "schema_location": 1,
        "vlm_box": 0,
    }

    def better(a: MaskCandidate, b: MaskCandidate) -> MaskCandidate:
        a_pri = source_priority.get(a.source, 0)
        b_pri = source_priority.get(b.source, 0)
        if a_pri != b_pri:
            return a if a_pri > b_pri else b
        a_stab = a.stability_score if a.stability_score is not None else -1.0
        b_stab = b.stability_score if b.stability_score is not None else -1.0
        if a_stab != b_stab:
            return a if a_stab > b_stab else b
        if a.inside_object_ratio != b.inside_object_ratio:
            return a if a.inside_object_ratio > b.inside_object_ratio else b
        return a if a.area >= b.area else b

    kept: list[MaskCandidate] = []
    for cand in candidates:
        cand_mask = read_mask(cand.mask_path)
        replaced = False
        for i, existing in enumerate(list(kept)):
            if mask_iou(cand_mask, read_mask(existing.mask_path)) > 0.90:
                kept[i] = better(cand, existing)
                replaced = True
                break
        if not replaced:
            kept.append(cand)
    return kept


def choose_sam_mask_for_box(masks, bbox_rect_area: int, object_mask: np.ndarray, prefer_score_order: bool = True) -> np.ndarray | None:
    if not masks:
        return None
    object_area = max(1, int(mask_area(object_mask)))
    rect_area = max(1, int(bbox_rect_area))
    best = None
    best_score = -1e9
    for idx, raw in enumerate(masks):
        mask = np.asarray(raw, dtype=bool) & object_mask
        area = max(1, int(mask.sum()))
        area_ratio_to_box = area / rect_area
        score = -abs(float(np.log(max(area_ratio_to_box, 1e-6))))
        if area > object_area * 0.95 and rect_area < object_area * 0.70:
            score -= 2.0
        if area > rect_area * 4.0:
            score -= 1.0
        score += 0.5 * mask_inside_ratio(mask, object_mask)
        if prefer_score_order:
            score -= 0.03 * idx
        if score > best_score:
            best_score = score
            best = mask
    return best


def _select_auto_object_mask(auto_masks, width: int, height: int) -> np.ndarray | None:
    image_area = max(1, width * height)
    cx, cy = width / 2.0, height / 2.0
    rows = []
    min_area = max(64, int(0.001 * image_area))
    for item in auto_masks:
        mask = np.asarray(item.get("segmentation"), dtype=bool)
        area = int(mask.sum())
        if area < min_area:
            continue
        bbox = mask_to_bbox(mask)
        if bbox.is_empty:
            continue
        bw = max(1, bbox.x2 - bbox.x1)
        bh = max(1, bbox.y2 - bbox.y1)
        bbox_area_ratio = (bw * bh) / image_area
        area_ratio = area / image_area
        border_hits = int(bbox.x1 <= 2) + int(bbox.y1 <= 2) + int(bbox.x2 >= width - 2) + int(bbox.y2 >= height - 2)
        if border_hits >= 3 and (area_ratio > 0.35 or bbox_area_ratio > 0.85):
            continue
        bx = (bbox.x1 + bbox.x2) / 2.0
        by = (bbox.y1 + bbox.y2) / 2.0
        center_dist = min(
            1.0,
            float(np.sqrt(((bx - cx) / max(1.0, width / 2.0)) ** 2 + ((by - cy) / max(1.0, height / 2.0)) ** 2)),
        )
        quality_vals = [item.get("predicted_iou"), item.get("stability_score")]
        quality_vals = [float(x) for x in quality_vals if x is not None]
        quality = float(np.mean(quality_vals)) if quality_vals else 0.5
        area_score = min(1.0, area_ratio / 0.08)
        score = 0.50 * (1.0 - center_dist) + 0.30 * area_score + 0.20 * quality
        rows.append((score, area, mask))
    if not rows:
        return None
    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score = rows[0][0]
    selected = []
    for score, area, mask in rows[:24]:
        if score >= max(0.35, best_score * 0.45) or area >= int(0.005 * image_area):
            selected.append(mask)
    if not selected:
        selected = [rows[0][2]]
    return np.logical_or.reduce(selected).astype(bool)


def _foreground_from_corner_background(image) -> np.ndarray | None:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return None
    pad = max(4, min(h, w) // 32)
    samples = np.concatenate(
        [
            arr[:pad, :pad].reshape(-1, 3),
            arr[:pad, -pad:].reshape(-1, 3),
            arr[-pad:, :pad].reshape(-1, 3),
            arr[-pad:, -pad:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
    maxc = arr.max(axis=2)
    bg_brightness = float(bg.max())
    if bg_brightness < 35:
        foreground = (dist > 24) & (maxc > 18)
    elif bg_brightness > 220:
        foreground = (dist > 22) & (arr.min(axis=2) < 245)
    else:
        foreground = dist > 30
    min_area = max(64, int(0.0008 * h * w))
    try:
        import cv2  # type: ignore

        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    except Exception:
        pass
    comps = connected_components_from_mask(foreground, min_area)
    if not comps:
        return None
    return np.logical_or.reduce(comps).astype(bool)


def _keep_central_object_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    image_area = max(1, h * w)
    min_area = max(64, int(0.001 * image_area))
    comps = connected_components_from_mask(mask, min_area)
    if not comps:
        return mask.astype(bool)
    cx, cy = w / 2.0, h / 2.0
    rows = []
    for comp in comps:
        area = int(comp.sum())
        bbox = mask_to_bbox(comp)
        bx = (bbox.x1 + bbox.x2) / 2.0
        by = (bbox.y1 + bbox.y2) / 2.0
        center_dist = min(
            1.0,
            float(np.sqrt(((bx - cx) / max(1.0, w / 2.0)) ** 2 + ((by - cy) / max(1.0, h / 2.0)) ** 2)),
        )
        border_hits = int(bbox.x1 <= 1) + int(bbox.y1 <= 1) + int(bbox.x2 >= w - 1) + int(bbox.y2 >= h - 1)
        score = area * (1.0 + 0.35 * (1.0 - center_dist))
        if border_hits >= 2:
            score *= 0.35
        if bbox.y1 > int(0.55 * h):
            score *= 0.20
        if bbox.y2 < int(0.25 * h):
            score *= 0.40
        rows.append((score, area, comp))
    rows.sort(key=lambda x: x[0], reverse=True)
    best_score = rows[0][0]
    best_area = rows[0][1]
    kept = []
    for score, area, comp in rows:
        if score >= best_score * 0.20 or area >= max(int(0.015 * image_area), int(0.18 * best_area)):
            kept.append(comp)
    return np.logical_or.reduce(kept).astype(bool) if kept else rows[0][2].astype(bool)


def _refine_object_mask_with_background(image, selected_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    selected_mask = np.asarray(selected_mask, dtype=bool)
    h, w = selected_mask.shape
    image_area = max(1, h * w)
    warnings: list[str] = []
    foreground = _foreground_from_corner_background(image)
    foreground_object = None
    if foreground is not None:
        foreground_object = _keep_central_object_components(foreground)
        mask_area_ratio = float(selected_mask.sum() / image_area)
        bbox = mask_to_bbox(selected_mask)
        fg_bbox = mask_to_bbox(foreground_object)
        fg_area = int(foreground_object.sum())
        selected_area = max(1, int(selected_mask.sum()))
        fg_overlap = int((selected_mask & foreground_object).sum()) / max(1, fg_area)
        misses_upper_foreground = fg_bbox.y1 < bbox.y1 - int(0.06 * h)
        foreground_is_substantially_larger = fg_area > int(1.25 * selected_area)
        foreground_area_ratio = fg_area / image_area
        border_hits = int(bbox.x1 <= 2) + int(bbox.y1 <= 2) + int(bbox.x2 >= w - 2) + int(bbox.y2 >= h - 2)
        if (
            0.02 <= foreground_area_ratio <= 0.75
            and foreground_is_substantially_larger
            and (misses_upper_foreground or fg_overlap < 0.72)
        ):
            refined = foreground_object
            warnings.append("Object mask replaced by fuller corner-background foreground cleanup.")
        elif mask_area_ratio > 0.55 or border_hits >= 2:
            refined = foreground_object
            warnings.append("Object mask replaced by corner-background foreground cleanup.")
        else:
            refined = selected_mask & foreground
            if refined.sum() < max(64, 0.20 * selected_mask.sum()):
                refined = selected_mask
            elif refined.sum() < selected_mask.sum() * 0.95:
                warnings.append("Object mask intersected with corner-background foreground cleanup.")
        selected_mask = refined.astype(bool)
    cleaned = _keep_central_object_components(selected_mask)
    if cleaned.sum() >= max(64, int(0.02 * image_area)) and cleaned.sum() < selected_mask.sum() * 0.98:
        warnings.append("Object mask cleaned by central connected components.")
        selected_mask = cleaned
    if foreground_object is not None:
        bbox = mask_to_bbox(selected_mask)
        fg_bbox = mask_to_bbox(foreground_object)
        selected_area = max(1, int(selected_mask.sum()))
        fg_area = int(foreground_object.sum())
        foreground_area_ratio = fg_area / image_area
        bottom_only = bbox.y1 > int(0.55 * h)
        foreground_is_higher = fg_bbox.y1 < bbox.y1 - int(0.08 * h)
        if bottom_only and foreground_is_higher and foreground_object.sum() >= max(64, int(0.25 * selected_mask.sum())):
            selected_mask = foreground_object
            warnings.append("Object mask replaced by higher foreground component after bottom-only detection.")
        elif (
            0.02 <= foreground_area_ratio <= 0.75
            and fg_area > int(1.35 * selected_area)
            and fg_bbox.y1 < bbox.y1 - int(0.05 * h)
        ):
            selected_mask = foreground_object
            warnings.append("Object mask replaced by fuller foreground after post-cleanup completeness check.")
    return selected_mask.astype(bool), warnings


def generate_object_mask(
    image_path,
    object_name,
    detector,
    sam_tool,
    output_dir,
    fallback_to_full_image: bool = True,
) -> tuple[str, BBox, list[str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    w, h = image.size
    warnings: list[str] = []
    selected_mask = None

    if detector is not None and sam_tool is not None:
        boxes = detector.detect(image_path, object_name)
        if boxes:
            box = max(boxes, key=lambda x: float(x.get("score", 0.0))).get("bbox")
            try:
                masks = sam_tool.segment_from_box(image_path, box)
            except Exception as exc:
                masks = []
                warnings.append(f"Object detector box-to-SAM failed: {exc}")
            if masks:
                if isinstance(box, dict):
                    box = BBox.from_dict(box)
                rect_area = bbox_area(box) if isinstance(box, BBox) else max(1, int((box[2] - box[0]) * (box[3] - box[1])))
                selected_mask = choose_sam_mask_for_box(masks, rect_area, np.ones((h, w), dtype=bool))
                if selected_mask is not None:
                    warnings.append("Object mask generated from detector box + SAM.")

    if selected_mask is None and sam_tool is not None:
        try:
            auto = sam_tool.automatic_masks(image_path)
            selected_mask = _select_auto_object_mask(auto, w, h)
            if selected_mask is not None:
                warnings.append("Object mask generated from SAM automatic masks.")
        except Exception as exc:
            warnings.append(f"SAM automatic object mask failed: {exc}")

    if selected_mask is None:
        if not fallback_to_full_image:
            raise RuntimeError("Object mask unavailable and fallback_to_full_image is false.")
        selected_mask = np.ones((h, w), dtype=bool)
        warnings.append("Object tools unavailable; using full image as object mask.")

    selected_mask, cleanup_warnings = _refine_object_mask_with_background(image, selected_mask)
    warnings.extend(cleanup_warnings)
    bbox = mask_to_bbox(selected_mask)
    mask_path = output_dir / "object_mask.png"
    overlay_path = output_dir / "object_overlay.png"
    save_mask(selected_mask, mask_path)
    save_rgb(overlay_mask(image, selected_mask), overlay_path)
    _write_json(output_dir / "object_bbox.json", bbox.to_dict())
    return str(mask_path), bbox, warnings


def _appearance_cluster_masks(image, object_mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    arr = np.asarray(image.convert("RGB"))
    ys, xs = np.where(object_mask)
    if len(xs) == 0:
        return []
    pixels = arr[ys, xs].astype(np.float32)
    unique_target = min(6, max(3, int(np.sqrt(len(pixels)) // 40 + 3)))
    masks: list[np.ndarray] = []
    try:
        from sklearn.cluster import KMeans  # type: ignore

        labels = KMeans(n_clusters=unique_target, n_init=5, random_state=0).fit_predict(pixels)
    except Exception:
        bins = np.clip((pixels // 64).astype(np.int32), 0, 3)
        labels = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    for label in np.unique(labels):
        cluster = np.zeros(object_mask.shape, dtype=bool)
        cluster[ys[labels == label], xs[labels == label]] = True
        cluster &= object_mask
        if cluster.sum() < min_area:
            continue
        try:
            import cv2  # type: ignore

            kernel = np.ones((3, 3), np.uint8)
            cluster = cv2.morphologyEx(cluster.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            cluster = cv2.morphologyEx(cluster.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        except Exception:
            pass
        masks.extend(connected_components_from_mask(cluster, min_area))
    return masks


def _clean_color_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    except Exception:
        pass
    comps = connected_components_from_mask(mask, min_area)
    if not comps:
        return np.zeros_like(mask, dtype=bool)
    return np.logical_or.reduce(comps).astype(bool)


def _color_prior_masks(image, object_mask: np.ndarray, specs: list[PartSpec], min_area: int) -> list[tuple[PartSpec, np.ndarray, dict[str, Any]]]:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    sat = maxc - minc
    h, _ = object_mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None] / max(1, h - 1)
    out: list[tuple[PartSpec, np.ndarray, dict[str, Any]]] = []
    for spec in specs:
        text = part_text(spec)
        color_text = " ".join([spec.name, *list(spec.text_prompts or [])]).lower()
        pos = np.ones_like(object_mask, dtype=bool)
        if any(k in color_text for k in ("strawber", "berr", "topping")):
            pos &= yy < 0.50
        elif any(k in text for k in ("top", "upper", "frosting", "icing", "cream")):
            pos &= yy < 0.66
        if any(k in text for k in ("bottom", "lower", "plate", "dish", "stand", "tray")):
            pos &= yy > 0.50
        if any(k in text for k in ("middle", "cylinder", "base", "body", "cake layer")) and not any(k in text for k in ("plate", "dish", "stand")):
            pos &= (yy > 0.34) & (yy < 0.84)

        masks: list[tuple[str, np.ndarray]] = []
        if any(k in color_text for k in ("red", "strawber", "berr")):
            masks.append(("red", (r > 120) & (r > g * 1.12) & (r > b * 1.12) & (sat > 28)))
        if any(k in color_text for k in ("white", "cream", "frosting", "icing", "ceramic", "plate", "dish")):
            masks.append(("white", (maxc > 165) & (sat < 78)))
        if any(k in color_text for k in ("yellow", "golden", "orange", "sponge", "cake body", "cake layer", "cake_base")):
            masks.append(("yellow", (r > 115) & (g > 75) & (b < 155) & (r > b * 1.18) & (g > b * 1.02)))
        for color_name, color_mask in masks:
            mask = color_mask & pos & object_mask
            mask = _clean_color_mask(mask, min_area)
            if mask.sum() < min_area:
                continue
            if is_collection_part(spec):
                comps = connected_components_from_mask(mask, min_area)
                if comps:
                    mask = np.logical_or.reduce(comps).astype(bool)
            out.append((spec, mask, {"color_prior": color_name, "source": "color_prior"}))
    return out


def generate_part_candidates(
    image_path,
    object_mask_path,
    object_bbox,
    part_schema,
    detector,
    sam_tool,
    output_dir,
    min_part_area_ratio,
) -> list[MaskCandidate]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    object_mask = read_mask(object_mask_path)
    h, w = object_mask.shape
    min_area = max(1, int(float(min_part_area_ratio) * mask_area(object_mask)))
    raw_parts = part_schema if isinstance(part_schema, list) else part_schema.get("parts", [])
    specs = []
    for p in raw_parts:
        specs.append(p if isinstance(p, PartSpec) else PartSpec.from_dict(p))
    main_like_specs = [spec for spec in specs if is_main_like_part(spec)]
    allow_object_body = len([s for s in specs if s.visible]) <= 1

    raw: list[MaskCandidate] = []
    idx = 0
    if detector is not None and sam_tool is not None:
        for spec in specs:
            for prompt in spec.text_prompts:
                for det in detector.detect(image_path, prompt):
                    try:
                        masks = sam_tool.segment_from_box(image_path, det["bbox"])
                    except Exception:
                        masks = []
                    bbox = det.get("bbox")
                    if isinstance(bbox, dict):
                        rect_area = bbox_area(BBox.from_dict(bbox))
                    else:
                        rect_area = max(1, int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))) if bbox else 1
                    mask = choose_sam_mask_for_box(masks, rect_area, object_mask)
                    if mask is not None:
                        if mask.sum() < min_area or mask_inside_ratio(mask, object_mask) < 0.80:
                            continue
                        raw.append(
                            _candidate_from_mask(
                                mask,
                                "text_box_sam",
                                output_dir,
                                idx,
                                image,
                                object_mask,
                                prompt=prompt,
                                metadata={"detector": det},
                            )
                        )
                        idx += 1

    if sam_tool is not None:
        try:
            for item in sam_tool.automatic_masks(image_path):
                mask = np.asarray(item["segmentation"], dtype=bool) & object_mask
                if mask.sum() < min_area or mask_inside_ratio(mask, object_mask) < 0.80:
                    continue
                near_whole = mask.sum() > 0.95 * max(1, mask_area(object_mask))
                if near_whole and not allow_object_body:
                    continue
                source = "object_body" if near_whole else "sam_auto"
                raw.append(
                    _candidate_from_mask(
                        mask,
                        source,
                        output_dir,
                        idx,
                        image,
                        object_mask,
                        metadata={"crop_box": item.get("crop_box"), "bbox": item.get("bbox"), "area": item.get("area")},
                        stability_score=item.get("stability_score"),
                        predicted_iou=item.get("predicted_iou"),
                    )
                )
                idx += 1
        except Exception:
            pass

    for spec, mask, metadata in _color_prior_masks(image, object_mask, specs, min_area):
        if mask.sum() < min_area or mask_inside_ratio(mask, object_mask) < 0.80:
            continue
        raw.append(
            _candidate_from_mask(
                mask,
                "color_prior",
                output_dir,
                idx,
                image,
                object_mask,
                prompt=spec.name,
                metadata=metadata,
            )
        )
        idx += 1

    for mask in _appearance_cluster_masks(image, object_mask, min_area):
        if mask.sum() < min_area or mask_inside_ratio(mask, object_mask) < 0.80:
            continue
        raw.append(_candidate_from_mask(mask, "appearance_cluster", output_dir, idx, image, object_mask))
        idx += 1

    if allow_object_body:
        body_prompt = main_like_specs[0].name if main_like_specs else "object body"
        raw.append(
            _candidate_from_mask(
                object_mask.copy(),
                "object_body",
                output_dir,
                idx,
                image,
                object_mask,
                prompt=body_prompt,
                metadata={"reason": "main/body-like schema part or single visible part"},
            )
        )
        idx += 1

    _write_json(output_dir / "raw_candidates_summary.json", [c.to_dict() for c in raw])
    candidates = _dedup_candidates(raw)
    _write_json(output_dir / "candidates_summary.json", [c.to_dict() for c in candidates])
    return candidates
