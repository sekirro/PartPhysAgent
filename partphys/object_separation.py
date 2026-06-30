from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_utils import (
    connected_components_from_mask,
    load_rgb,
    mask_area,
    mask_inside_ratio,
    mask_iou,
    mask_to_bbox,
    overlay_mask,
    overlay_multiple_masks,
    read_mask,
    save_mask,
    save_rgb,
)
from .scene_builder import FRAME_NAMES, VIEW_LABELS, _candidate_multiview_paths
from .types import ObjectInstance, PartInstance


def _write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "object"


def _is_residual_part(part: PartInstance) -> bool:
    text = " ".join(
        [
            str(part.name),
            str(part.physics_group),
            str((part.metadata or {}).get("physics_group", "")),
        ]
    ).lower()
    return "unknown" in text or "residual" in text


def _candidate_score(mask: np.ndarray, object_mask: np.ndarray, source: str, metadata: dict[str, Any]) -> float:
    object_area = max(1, int(mask_area(object_mask)))
    area = int(mask_area(mask))
    area_ratio = area / object_area
    bbox = mask_to_bbox(mask)
    bbox_area = max(1, bbox.width * bbox.height)
    compactness = area / bbox_area
    quality_values = [metadata.get("predicted_iou"), metadata.get("stability_score")]
    quality_values = [float(x) for x in quality_values if x is not None]
    quality = float(np.mean(quality_values)) if quality_values else 0.55
    source_bonus = {
        "part_mask": 0.18,
        "sam_auto": 0.12,
        "connected_component": 0.08,
        "global_object": 0.0,
    }.get(source, 0.0)
    if source == "global_object":
        size_score = 0.20
    else:
        size_score = min(1.0, area_ratio / 0.25)
        if area_ratio > 0.92:
            size_score *= 0.25
    return float(0.38 * size_score + 0.22 * compactness + 0.22 * quality + source_bonus)


def _add_candidate(
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    object_mask: np.ndarray,
    source: str,
    min_area: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    mask = np.asarray(mask, dtype=bool) & object_mask
    area = int(mask_area(mask))
    if area < min_area:
        return
    inside = mask_inside_ratio(mask, object_mask)
    if inside < 0.70:
        return
    metadata = dict(metadata or {})
    score = _candidate_score(mask, object_mask, source, metadata)
    rows.append(
        {
            "mask": mask,
            "source": source,
            "area": area,
            "bbox": mask_to_bbox(mask),
            "inside_object_ratio": inside,
            "score": score,
            "metadata": metadata,
        }
    )


def _dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (float(item["score"]), int(item["area"])), reverse=True):
        duplicate_idx = None
        for idx, existing in enumerate(kept):
            if mask_iou(row["mask"], existing["mask"]) > 0.86:
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            kept.append(row)
        elif float(row["score"]) > float(kept[duplicate_idx]["score"]):
            kept[duplicate_idx] = row
    return kept


def _select_object_rows(rows: list[dict[str, Any]], object_mask: np.ndarray, max_objects: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    selected: list[dict[str, Any]] = []
    for row in _dedup(rows):
        if len(selected) >= max(1, int(max_objects)):
            break
        mask = row["mask"]
        reject = False
        for existing in selected:
            inter = int((mask & existing["mask"]).sum())
            smaller = max(1, min(int(mask.sum()), int(existing["mask"].sum())))
            if mask_iou(mask, existing["mask"]) > 0.58 or inter / smaller > 0.78:
                reject = True
                break
        if not reject:
            selected.append(row)
    if not selected:
        selected = [_dedup(rows)[0]]
    if len(selected) == 1:
        selected_area = int(selected[0]["mask"].sum())
        object_area = max(1, int(object_mask.sum()))
        if selected_area < int(0.35 * object_area):
            selected = []
    return selected


def _part_name_for_object(parts: list[PartInstance], fallback: str) -> str:
    if not parts:
        return fallback
    ordered = sorted(parts, key=lambda p: int(p.area), reverse=True)
    if len(ordered) == 1:
        return ordered[0].name
    tokens = [p.name for p in ordered[:3]]
    prefix = ""
    split_names = [re.split(r"[_\\s-]+", name.lower()) for name in tokens]
    if split_names and all(items for items in split_names):
        first = split_names[0][0]
        if all(items[0] == first for items in split_names):
            prefix = first + "_"
    return prefix + "object"


def _assign_parts_to_rows(parts: list[PartInstance], rows: list[dict[str, Any]], object_mask: np.ndarray) -> dict[int, list[PartInstance]]:
    assignments: dict[int, list[PartInstance]] = {idx: [] for idx in range(len(rows))}
    if not rows:
        return assignments
    for part in parts:
        if _is_residual_part(part):
            continue
        try:
            part_mask = read_mask(part.mask_path)
        except Exception:
            continue
        part_area = max(1, int(part_mask.sum()))
        best_idx = 0
        best_score = -1.0
        for idx, row in enumerate(rows):
            overlap = int((part_mask & row["mask"]).sum()) / part_area
            iou = mask_iou(part_mask, row["mask"])
            score = 0.75 * overlap + 0.25 * iou
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_score <= 0.02:
            best_idx = int(np.argmax([int(row["mask"].sum()) for row in rows]))
        assignments[best_idx].append(part)
    return assignments


def _drop_duplicate_empty_rows(
    parts: list[PartInstance],
    rows: list[dict[str, Any]],
    part_assignments: dict[int, list[PartInstance]],
    object_mask: np.ndarray,
    keep_empty_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[PartInstance]], int]:
    assigned_indices = [idx for idx, assigned in part_assignments.items() if assigned]
    if not assigned_indices:
        return rows, part_assignments, 0
    keep_empty_sources = keep_empty_sources or set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for idx, row in enumerate(rows):
        if part_assignments.get(idx):
            kept.append(row)
            continue
        if str(row.get("source", "")) in keep_empty_sources:
            kept.append(row)
            continue
        dropped += 1
    if dropped == 0:
        return rows, part_assignments, 0
    return kept, _assign_parts_to_rows(parts, kept, object_mask), dropped


def _union_part_view_masks(parts: list[PartInstance], label: str) -> np.ndarray | None:
    masks = []
    for part in parts:
        view_masks = (part.metadata or {}).get("view_masks") or {}
        path = view_masks.get(label)
        if not path:
            continue
        try:
            masks.append(read_mask(path))
        except Exception:
            continue
    if not masks:
        return None
    return np.logical_or.reduce(masks).astype(bool)


def _component_center_x(mask: np.ndarray) -> float:
    bbox = mask_to_bbox(mask)
    return float(bbox.x1 + bbox.x2) * 0.5


def _view_image_paths(image_path, multiview_dir: str | None) -> dict[str, Path]:
    paths = _candidate_multiview_paths(multiview_dir)
    if not paths:
        paths = [Path(image_path)] * len(FRAME_NAMES)
    return {label: paths[min(idx, len(paths) - 1)] for idx, label in enumerate(VIEW_LABELS)}


def _resolve_case_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _load_compose_object_rows(
    multiview_dir: str | None,
    object_mask: np.ndarray,
    min_area: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    if not multiview_dir:
        return [], {}
    root = Path(multiview_dir)
    metadata_path = root / "compose_metadata.json"
    if not metadata_path.exists():
        return [], {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}
    records = metadata.get("object_masks") or []
    rows: list[dict[str, Any]] = []
    view_masks_by_row: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        view_mask_paths = record.get("view_masks") or {}
        front_path = _resolve_case_path(root, view_mask_paths.get("front"))
        if front_path is None or not front_path.exists():
            continue
        try:
            front_mask = _resize_mask(read_mask(front_path), object_mask.shape) & object_mask
        except Exception:
            continue
        if int(front_mask.sum()) < min_area:
            continue
        row_idx = len(rows)
        view_masks: dict[str, np.ndarray] = {"front": front_mask}
        for label in VIEW_LABELS[1:]:
            path = _resolve_case_path(root, view_mask_paths.get(label))
            if path is None or not path.exists():
                continue
            try:
                view_masks[label] = read_mask(path)
            except Exception:
                continue
        rows.append(
            {
                "mask": front_mask,
                "source": "compose_metadata_object",
                "area": int(front_mask.sum()),
                "bbox": mask_to_bbox(front_mask),
                "inside_object_ratio": mask_inside_ratio(front_mask, object_mask),
                "score": 1.0,
                "metadata": {
                    "object_id": record.get("object_id"),
                    "object_name": record.get("name"),
                    "case_dir": record.get("case_dir"),
                    "source_metadata": str(metadata_path),
                },
            }
        )
        view_masks_by_row[row_idx] = view_masks
    return rows, view_masks_by_row


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shape:
        return mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    ) > 127


def _load_rgb_array(path) -> np.ndarray:
    image = load_rgb(path)
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    return np.asarray(image)


def _masked_colors(image: np.ndarray, mask: np.ndarray, max_samples: int = 12000) -> np.ndarray:
    mask = _resize_mask(mask, image.shape[:2])
    values = image[mask].astype(np.float32)
    if len(values) == 0:
        return values.reshape(0, 3)
    if len(values) > max_samples:
        idx = np.linspace(0, len(values) - 1, max_samples).round().astype(np.int64)
        values = values[idx]
    return values


def _kmeans(values: np.ndarray, k: int, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, values.shape[1] if values.ndim == 2 else 1), dtype=np.float32)
    k = max(1, min(int(k), len(values)))
    order = np.argsort(values[:, 0] * 0.30 + values[:, 1] * 0.59 + values[:, 2] * 0.11)
    init_idx = np.linspace(0, len(order) - 1, k).round().astype(np.int64)
    centers = values[order[init_idx]].copy()
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(iterations):
        dist = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(k):
            cluster = values[labels == cluster_idx]
            if len(cluster):
                centers[cluster_idx] = cluster.mean(axis=0)
    return labels, centers


def _object_reference(image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    colors = _masked_colors(image, mask)
    _, centers = _kmeans(colors, k=min(8, max(1, len(colors) // 400)), iterations=10)
    bbox = mask_to_bbox(mask)
    h, w = image.shape[:2]
    return {
        "centers": centers.astype(np.float32),
        "area": int(mask.sum()),
        "y1": float(bbox.y1) / max(1.0, float(h)),
        "y2": float(bbox.y2) / max(1.0, float(h)),
        "cy": float(bbox.y1 + bbox.y2) * 0.5 / max(1.0, float(h)),
        "height": float(bbox.y2 - bbox.y1) / max(1.0, float(h)),
        "cx": float(bbox.x1 + bbox.x2) * 0.5 / max(1.0, float(w)),
        "width": float(bbox.x2 - bbox.x1) / max(1.0, float(w)),
    }


def _appearance_score(image: np.ndarray, mask: np.ndarray, ref: dict[str, Any]) -> float:
    colors = _masked_colors(image, mask, max_samples=4000)
    centers = np.asarray(ref.get("centers"), dtype=np.float32)
    if len(colors) == 0 or len(centers) == 0:
        return -1e6
    sample_labels, sample_centers = _kmeans(colors, k=min(5, max(1, len(colors) // 600)), iterations=8)
    del sample_labels
    dists = np.sqrt(((sample_centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    color_score = -float(np.mean(np.min(dists, axis=1))) / 60.0
    bbox = mask_to_bbox(mask)
    h, w = image.shape[:2]
    cy = float(bbox.y1 + bbox.y2) * 0.5 / max(1.0, float(h))
    height = float(bbox.y2 - bbox.y1) / max(1.0, float(h))
    ref_y1 = float(ref.get("y1", cy))
    ref_y2 = float(ref.get("y2", cy))
    range_gap = max(ref_y1 - cy, cy - ref_y2, 0.0)
    y_score = -abs(cy - float(ref.get("cy", cy))) * 0.50
    y_range_score = -range_gap * 1.20
    height_score = -abs(height - float(ref.get("height", height))) * 0.20
    return float(color_score + y_score + y_range_score + height_score)


def _best_unique_assignment(scores: np.ndarray) -> dict[int, int]:
    rows, cols = scores.shape
    if rows == 0 or cols == 0:
        return {}
    if rows <= 7 and cols <= 9:
        import itertools

        best_score = -1e18
        best_perm: tuple[int, ...] | None = None
        for perm in itertools.permutations(range(cols), rows):
            value = float(sum(scores[row_idx, col_idx] for row_idx, col_idx in enumerate(perm)))
            if value > best_score:
                best_score = value
                best_perm = perm
        return {row_idx: int(col_idx) for row_idx, col_idx in enumerate(best_perm or ())}
    remaining = set(range(cols))
    assignment: dict[int, int] = {}
    for row_idx in range(rows):
        if not remaining:
            break
        col_idx = max(remaining, key=lambda idx: float(scores[row_idx, idx]))
        assignment[row_idx] = int(col_idx)
        remaining.remove(col_idx)
    return assignment


def _appearance_cluster_view_masks(
    image: np.ndarray,
    view_mask: np.ndarray,
    refs: list[dict[str, Any]],
    min_area: int,
) -> dict[int, np.ndarray]:
    view_mask = _resize_mask(view_mask, image.shape[:2])
    ys, xs = np.where(view_mask)
    if len(xs) < min_area or not refs:
        return {}
    rgb = image[ys, xs].astype(np.float32)
    h, w = image.shape[:2]
    xy = np.stack([xs / max(1.0, float(w)) * 32.0, ys / max(1.0, float(h)) * 32.0], axis=1).astype(np.float32)
    features = np.concatenate([rgb, xy], axis=1)
    labels, centers = _kmeans(features, k=min(10, max(len(refs) * 3, len(refs))), iterations=14)
    result = {idx: np.zeros(view_mask.shape, dtype=bool) for idx in range(len(refs))}
    for cluster_idx in range(len(centers)):
        cluster_pixels = labels == cluster_idx
        if int(cluster_pixels.sum()) < max(32, min_area // 12):
            continue
        cluster_mask = np.zeros(view_mask.shape, dtype=bool)
        cluster_mask[ys[cluster_pixels], xs[cluster_pixels]] = True
        scores = [_appearance_score(image, cluster_mask, ref) for ref in refs]
        best_idx = int(np.argmax(scores))
        result[best_idx] |= cluster_mask
    return {idx: mask for idx, mask in result.items() if int(mask.sum()) >= max(64, min_area // 5)}


def _clean_view_mask_components(image: np.ndarray, mask: np.ndarray, ref: dict[str, Any], min_area: int) -> np.ndarray:
    mask = _resize_mask(mask, image.shape[:2])
    comps = connected_components_from_mask(mask, max(32, min_area // 24))
    if len(comps) <= 1:
        return mask
    scored = []
    h = image.shape[0]
    ref_y1 = float(ref.get("y1", 0.0))
    ref_y2 = float(ref.get("y2", 1.0))
    ref_height = float(ref.get("height", 1.0))
    for comp in comps:
        bbox = mask_to_bbox(comp)
        cy = float(bbox.y1 + bbox.y2) * 0.5 / max(1.0, float(h))
        y_gap = max(ref_y1 - cy, cy - ref_y2, 0.0)
        comp_height = float(bbox.y2 - bbox.y1) / max(1.0, float(h))
        score = _appearance_score(image, comp, ref) - 1.5 * y_gap
        if ref_height > 0.38 and comp_height < 0.45 * ref_height:
            score -= 0.75
        scored.append((float(score), int(comp.sum()), comp))
    best_score = max(item[0] for item in scored)
    kept = [
        comp
        for score, area, comp in scored
        if score >= best_score - 0.35
    ]
    if not kept:
        return mask
    return np.logical_or.reduce(kept).astype(bool)


def _appearance_view_masks_by_row(
    image_path,
    multiview_dir: str | None,
    rows: list[dict[str, Any]],
    output_dir: Path,
    min_area: int,
) -> dict[int, dict[str, np.ndarray]]:
    if len(rows) < 2:
        return {}
    scene_dir = output_dir.parent
    paths = _view_image_paths(image_path, multiview_dir)
    front_image = _load_rgb_array(paths["front"])
    refs = [_object_reference(front_image, np.asarray(row["mask"], dtype=bool)) for row in rows]
    result: dict[int, dict[str, np.ndarray]] = {idx: {"front": np.asarray(row["mask"], dtype=bool)} for idx, row in enumerate(rows)}
    for label in VIEW_LABELS[1:]:
        mask_path = scene_dir / "multiview_object" / label / "object_mask.png"
        if not mask_path.exists():
            continue
        try:
            view_mask = read_mask(mask_path)
        except Exception:
            continue
        view_image = _load_rgb_array(paths[label])
        view_mask = _resize_mask(view_mask, view_image.shape[:2])
        comps = connected_components_from_mask(view_mask, min_area)
        assigned: dict[int, np.ndarray] = {}
        if len(comps) == len(rows):
            scores = np.asarray([[_appearance_score(view_image, comp, ref) for comp in comps] for ref in refs], dtype=np.float32)
            for row_idx, comp_idx in _best_unique_assignment(scores).items():
                if float(scores[row_idx, comp_idx]) > -3.5:
                    assigned[row_idx] = comps[comp_idx]
        if len(assigned) < len(rows):
            cluster_assigned = _appearance_cluster_view_masks(view_image, view_mask, refs, min_area)
            for row_idx, mask in cluster_assigned.items():
                if row_idx not in assigned or int(mask.sum()) > int(assigned[row_idx].sum()):
                    assigned[row_idx] = mask
        for row_idx, mask in assigned.items():
            result.setdefault(row_idx, {})[label] = _clean_view_mask_components(view_image, mask, refs[row_idx], min_area)
    return result


def _component_view_masks_by_row(
    rows: list[dict[str, Any]],
    output_dir: Path,
    min_area: int,
) -> dict[int, dict[str, np.ndarray]]:
    component_indices = [idx for idx, row in enumerate(rows) if row.get("source") == "connected_component"]
    if len(component_indices) < 2 or len(component_indices) != len(rows):
        return {}
    scene_dir = output_dir.parent
    front_order = sorted(component_indices, key=lambda idx: _component_center_x(rows[idx]["mask"]))
    result: dict[int, dict[str, np.ndarray]] = {idx: {} for idx in component_indices}
    for label in VIEW_LABELS[1:]:
        mask_path = scene_dir / "multiview_object" / label / "object_mask.png"
        if not mask_path.exists():
            continue
        try:
            view_mask = read_mask(mask_path)
        except Exception:
            continue
        comps = connected_components_from_mask(view_mask, min_area)
        if len(comps) != len(front_order):
            continue
        for row_idx, comp in zip(front_order, sorted(comps, key=_component_center_x)):
            result[row_idx][label] = comp
    return result


def separate_scene_objects(
    image_path,
    object_mask_path,
    parts: list[PartInstance],
    sam_tool,
    output_dir,
    object_name: str = "object",
    mode: str = "auto",
    max_objects: int = 6,
    min_object_area_ratio: float = 0.015,
    multiview_dir: str | None = None,
) -> tuple[list[ObjectInstance], dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    object_mask = read_mask(object_mask_path)
    object_area = max(1, int(mask_area(object_mask)))
    min_area = max(64, int(float(min_object_area_ratio) * object_area))
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    direct_view_masks: dict[int, dict[str, np.ndarray]] = {}

    if str(mode).lower() == "single":
        _add_candidate(rows, object_mask, object_mask, "global_object", 1, {"reason": "object_mode_single"})
    else:
        compose_rows, compose_view_masks = _load_compose_object_rows(multiview_dir, object_mask, min_area)
        if len(compose_rows) >= 2:
            rows = compose_rows[: max(1, int(max_objects))]
            direct_view_masks = {
                idx: masks
                for idx, masks in compose_view_masks.items()
                if idx < len(rows)
            }
        else:
            component_rows: list[dict[str, Any]] = []
            for comp in connected_components_from_mask(object_mask, min_area):
                _add_candidate(component_rows, comp, object_mask, "connected_component", min_area, {})
            rows.extend(component_rows)

            component_coverage = 0.0
            if component_rows:
                component_union = np.logical_or.reduce([row["mask"] for row in component_rows])
                component_coverage = int(component_union.sum()) / object_area
            prefer_components = len(component_rows) >= 2 and component_coverage >= 0.70

            if prefer_components:
                rows = component_rows
            else:
                for part in parts:
                    if _is_residual_part(part):
                        continue
                    try:
                        mask = read_mask(part.mask_path)
                    except Exception as exc:
                        warnings.append(f"Failed to read part mask for object proposal {part.name}: {exc}")
                        continue
                    _add_candidate(
                        rows,
                        mask,
                        object_mask,
                        "part_mask",
                        min_area,
                        {"part_id": int(part.part_id), "part_name": part.name},
                    )

                if sam_tool is not None:
                    try:
                        for item in sam_tool.automatic_masks(image_path):
                            mask = np.asarray(item.get("segmentation"), dtype=bool) & object_mask
                            area = int(mask.sum())
                            if area > int(0.96 * object_area):
                                continue
                            _add_candidate(
                                rows,
                                mask,
                                object_mask,
                                "sam_auto",
                                min_area,
                                {
                                    "predicted_iou": item.get("predicted_iou"),
                                    "stability_score": item.get("stability_score"),
                                    "bbox": item.get("bbox"),
                                    "crop_box": item.get("crop_box"),
                                },
                            )
                    except Exception as exc:
                        warnings.append(f"SAM automatic object proposals failed: {exc}")

            selected = _select_object_rows(rows, object_mask, max_objects)
            selected_coverage = int(np.logical_or.reduce([row["mask"] for row in selected]).sum()) / object_area if selected else 0.0
            if not selected or selected_coverage < 0.35:
                warnings.append("Object separation fell back to a single foreground object.")
                rows = []
                _add_candidate(rows, object_mask, object_mask, "global_object", 1, {"reason": "low_object_proposal_coverage"})
            else:
                rows = selected

    rows = _select_object_rows(rows, object_mask, max_objects)
    if not rows:
        _add_candidate(rows, object_mask, object_mask, "global_object", 1, {"reason": "empty_object_rows"})

    part_assignments = _assign_parts_to_rows(parts, rows, object_mask)
    rows, part_assignments, dropped_empty = _drop_duplicate_empty_rows(
        parts,
        rows,
        part_assignments,
        object_mask,
        keep_empty_sources={"connected_component", "compose_metadata_object"} if str(mode).lower() == "auto" else set(),
    )
    if dropped_empty:
        warnings.append(f"Dropped {dropped_empty} object proposals without assigned parts.")
    component_view_masks = direct_view_masks
    if not component_view_masks:
        component_view_masks = _appearance_view_masks_by_row(image_path, multiview_dir, rows, output_dir, min_area)
    if not component_view_masks:
        component_view_masks = _component_view_masks_by_row(rows, output_dir, min_area)
    objects: list[ObjectInstance] = []
    object_masks = []
    object_labels = []
    summary_rows = []
    for idx, row in enumerate(rows):
        assigned_parts = part_assignments.get(idx, [])
        metadata = row.get("metadata", {}) or {}
        fallback_name = str(metadata.get("object_name") or f"{object_name}_{idx}")
        name = _safe_name(_part_name_for_object(assigned_parts, fallback_name))
        obj_dir = output_dir / f"object_{idx:03d}_{name}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        mask = np.asarray(row["mask"], dtype=bool) & object_mask
        mask_path = obj_dir / "mask.png"
        overlay_path = obj_dir / "overlay.png"
        save_mask(mask, mask_path)
        save_rgb(overlay_mask(image, mask), overlay_path)
        view_masks: dict[str, str] = {}
        for label in VIEW_LABELS:
            view_mask = component_view_masks.get(idx, {}).get(label)
            if view_mask is None:
                view_mask = _union_part_view_masks(assigned_parts, label)
            if view_mask is None:
                continue
            view_path = obj_dir / f"{label}_mask.png"
            save_mask(view_mask, view_path)
            view_masks[label] = str(view_path)
        part_ids = [int(part.part_id) for part in assigned_parts]
        obj = ObjectInstance(
            object_id=idx,
            name=name,
            mask_path=str(mask_path),
            bbox=mask_to_bbox(mask),
            area=int(mask.sum()),
            confidence=float(row.get("score", 0.0)),
            part_ids=part_ids,
            warnings=[],
            metadata={
                "source": row.get("source"),
                "proposal_score": float(row.get("score", 0.0)),
                "view_masks": view_masks,
                "raw_metadata": row.get("metadata", {}),
            },
        )
        objects.append(obj)
        object_masks.append(mask)
        object_labels.append(f"{idx}:{name}")
        for part in assigned_parts:
            part.metadata["object_id"] = int(obj.object_id)
            part.metadata["object_name"] = obj.name
            part.metadata["object_mask_path"] = str(mask_path)
        summary_rows.append(obj.to_dict())
        _write_json(obj_dir / "object_summary.json", obj.to_dict())

    save_rgb(overlay_multiple_masks(image, object_masks, labels=object_labels), output_dir / "objects_overlay.png")
    summary = {
        "mode": str(mode),
        "object_count": len(objects),
        "objects": summary_rows,
        "warnings": warnings,
    }
    _write_json(output_dir / "objects_summary.json", summary)
    return objects, summary
