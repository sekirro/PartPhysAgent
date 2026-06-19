from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .image_utils import (
    connected_components_from_mask,
    load_rgb,
    mask_area,
    mask_iou,
    mask_to_bbox,
    overlay_mask,
    overlay_multiple_masks,
    read_mask,
    save_mask,
    save_rgb,
)
from .types import MaskCandidate, PartInstance, PartSpec


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip().lower())
    return value.strip("_") or "part"


def _write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _candidate_score(candidate: MaskCandidate, mask: np.ndarray, part: PartSpec, object_mask: np.ndarray, min_part_area_ratio: float) -> float:
    prompt_text = (candidate.prompt or "").lower()
    prompt_hits = part.name.lower() in prompt_text or any(p.lower() in prompt_text for p in part.text_prompts)
    semantic = 1.0 if prompt_hits else (0.5 if candidate.source == "sam_auto" else 0.3)
    material = 0.5
    bbox = candidate.bbox
    h, w = object_mask.shape
    cx = ((bbox.x1 + bbox.x2) / 2.0) / max(1, w)
    cy = ((bbox.y1 + bbox.y2) / 2.0) / max(1, h)
    loc = (part.location or "").lower()
    location = 0.6
    if any(k in loc for k in ["top", "upper"]):
        location = 1.0 if cy < 0.55 else 0.25
    if any(k in loc for k in ["bottom", "lower", "sole"]):
        location = max(location, 1.0 if cy > 0.45 else 0.25)
    if "left" in loc:
        location = max(location, 1.0 if cx < 0.60 else 0.35)
    if "right" in loc:
        location = max(location, 1.0 if cx > 0.40 else 0.35)
    if any(k in loc for k in ["center", "body", "main"]):
        location = max(location, 0.8)

    shape_text = f"{part.shape_prior} {part.name}".lower()
    aspect = bbox.width / max(1, bbox.height)
    fill = mask_area(mask) / max(1, bbox.width * bbox.height)
    shape = 0.55
    if any(k in shape_text for k in ["long", "thin", "handle", "lace"]):
        shape = 1.0 if aspect > 2.0 or aspect < 0.5 else 0.35
    elif any(k in shape_text for k in ["round", "wheel", "tire"]):
        shape = 1.0 if 0.55 <= aspect <= 1.8 and fill > 0.35 else 0.45
    elif any(k in shape_text for k in ["compact", "block", "head"]):
        shape = 1.0 if fill > 0.35 else 0.45
    elif any(k in shape_text for k in ["body", "shell", "main"]):
        shape = 0.8 if candidate.area > 0.08 * max(1, object_mask.sum()) else 0.5

    source = {"vlm_box_sam": 0.95, "vlm_box": 0.85, "text_box_sam": 1.0, "schema_location": 0.55, "sam_auto": 0.7, "appearance_cluster": 0.5}.get(candidate.source, 0.4)
    qualities = [x for x in [candidate.stability_score, candidate.predicted_iou] if x is not None]
    quality = float(np.mean(qualities)) if qualities else 0.5
    score = 0.30 * semantic + 0.20 * material + 0.15 * location + 0.15 * shape + 0.10 * source + 0.10 * quality

    area_ratio = candidate.area / max(1, object_mask.sum())
    if area_ratio < min_part_area_ratio:
        score -= 0.4
    if area_ratio > 0.95 and "body" not in part.name.lower():
        score -= 0.25
    comps = connected_components_from_mask(mask, max(1, int(0.05 * max(1, candidate.area))))
    if len(comps) > 4:
        score -= 0.15
    return float(max(0.0, min(1.0, score)))


def _save_part(image, mask: np.ndarray, part_id: int, name: str, confidence: float, source_ids: list[str], spec: PartSpec, output_dir: Path) -> PartInstance:
    part_dir = output_dir / "parts" / f"part_{part_id:03d}_{_safe_name(name)}"
    part_dir.mkdir(parents=True, exist_ok=True)
    mask_path = part_dir / "mask.png"
    overlay_path = part_dir / "overlay.png"
    save_mask(mask, mask_path)
    save_rgb(overlay_mask(image, mask), overlay_path)
    inst = PartInstance(
        part_id=part_id,
        name=name,
        mask_path=str(mask_path),
        bbox=mask_to_bbox(mask),
        area=mask_area(mask),
        confidence=float(confidence),
        candidate_ids=source_ids,
        expected_materials=spec.expected_materials,
        physics_group=spec.physics_group or spec.name,
        warnings=[],
        metadata={"part_spec": spec.to_dict()},
    )
    _write_json(part_dir / "part_summary.json", {"part": inst.to_dict()})
    return inst


def select_physical_parts(
    image_path,
    object_mask_path,
    candidates: list[MaskCandidate],
    part_schema,
    output_dir,
    vlm_client=None,
    clip_client=None,
    max_parts: int = 6,
    coverage_threshold: float = 0.75,
    min_part_area_ratio: float = 0.01,
) -> list[PartInstance]:
    output_dir = Path(output_dir)
    image = load_rgb(image_path)
    object_mask = read_mask(object_mask_path)
    raw_specs = part_schema if isinstance(part_schema, list) else part_schema.get("parts", [])
    specs = [p if isinstance(p, PartSpec) else PartSpec.from_dict(p) for p in raw_specs]
    specs = [p for p in specs if p.visible]
    if not specs:
        specs = [PartSpec(name="body", expected_materials=["Plastic"], physics_group="global_body")]

    selected_raw: list[dict] = []
    for spec in specs:
        best = None
        for cand in candidates:
            mask = read_mask(cand.mask_path) & object_mask
            score = _candidate_score(cand, mask, spec, object_mask, min_part_area_ratio)
            cand.part_scores[spec.name] = score
            if best is None or score > best["score"]:
                best = {"spec": spec, "candidate": cand, "mask": mask, "score": score}
        if best is not None and best["score"] >= 0.35:
            selected_raw.append(best)

    selected_raw.sort(key=lambda x: x["score"], reverse=True)
    resolved: list[dict] = []
    min_area = max(1, int(min_part_area_ratio * object_mask.sum()))
    for item in selected_raw:
        mask = item["mask"].copy()
        for kept in resolved:
            inter = np.logical_and(mask, kept["mask"])
            if inter.sum() > 0.20 * min(mask.sum(), kept["mask"].sum()):
                mask[kept["mask"]] = False
        if mask.sum() >= min_area:
            item = dict(item)
            item["mask"] = mask
            resolved.append(item)
        if len(resolved) >= max_parts:
            break

    if not resolved:
        spec = PartSpec(name="body", expected_materials=["Plastic"], physics_group="global_body")
        resolved.append({"spec": spec, "candidate": None, "mask": object_mask.copy(), "score": 0.5})

    union = np.zeros_like(object_mask, dtype=bool)
    for item in resolved:
        union |= item["mask"]
    coverage = float(union.sum() / max(1, object_mask.sum()))
    if coverage < coverage_threshold:
        residual = object_mask & ~union
        if residual.sum() >= min_area and len(resolved) < max_parts:
            spec = PartSpec(
                name="unknown_body",
                text_prompts=["object body"],
                expected_materials=["Plastic"],
                location="residual body",
                shape_prior="main body",
                physical_role="global residual material",
                physics_group="global_body",
            )
            resolved.append({"spec": spec, "candidate": None, "mask": residual, "score": 0.35})

    parts: list[PartInstance] = []
    for idx, item in enumerate(resolved[:max_parts]):
        cand = item.get("candidate")
        ids = [cand.candidate_id] if cand is not None else ["residual_or_object"]
        inst = _save_part(image, item["mask"], idx, item["spec"].name, item["score"], ids, item["spec"], output_dir)
        parts.append(inst)

    overlay = overlay_multiple_masks(image, [read_mask(p.mask_path) for p in parts], labels=[p.name for p in parts])
    save_rgb(overlay, output_dir / "parts" / "parts_overlay.png")
    pair_overlap = 0
    for i in range(len(parts)):
        mi = read_mask(parts[i].mask_path)
        for j in range(i + 1, len(parts)):
            pair_overlap += int(np.logical_and(mi, read_mask(parts[j].mask_path)).sum())
    summary = {
        "coverage": float((np.logical_or.reduce([read_mask(p.mask_path) for p in parts]) & object_mask).sum() / max(1, object_mask.sum())),
        "overlap_ratio": float(pair_overlap / max(1, object_mask.sum())),
        "parts": [p.to_dict() for p in parts],
    }
    _write_json(output_dir / "parts" / "selection_summary.json", summary)
    return parts
