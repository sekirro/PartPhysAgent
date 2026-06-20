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
from .part_traits import is_main_like_part
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
            masks = sam_tool.segment_from_box(image_path, box)
            if masks:
                if isinstance(box, dict):
                    box = BBox.from_dict(box)
                rect_area = bbox_area(box) if isinstance(box, BBox) else max(1, int((box[2] - box[0]) * (box[3] - box[1])))
                selected_mask = choose_sam_mask_for_box(masks, rect_area, np.ones((h, w), dtype=bool))

    if selected_mask is None and sam_tool is not None:
        try:
            auto = sam_tool.automatic_masks(image_path)
            if auto:
                cx, cy = w / 2.0, h / 2.0

                def score(item):
                    m = item["segmentation"]
                    bbox = mask_to_bbox(m)
                    bx = (bbox.x1 + bbox.x2) / 2.0
                    by = (bbox.y1 + bbox.y2) / 2.0
                    center_penalty = ((bx - cx) / max(1, w)) ** 2 + ((by - cy) / max(1, h)) ** 2
                    return mask_area(m) / max(1, w * h) - center_penalty

                selected_mask = max(auto, key=score)["segmentation"]
        except Exception as exc:
            warnings.append(f"SAM automatic object mask failed: {exc}")

    if selected_mask is None:
        if not fallback_to_full_image:
            raise RuntimeError("Object mask unavailable and fallback_to_full_image is false.")
        selected_mask = np.ones((h, w), dtype=bool)
        warnings.append("Object tools unavailable; using full image as object mask.")

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
    allow_object_body = bool(main_like_specs) or len([s for s in specs if s.visible]) <= 1

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
