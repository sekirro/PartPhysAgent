from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import load_rgb, mask_area, mask_inside_ratio, overlay_multiple_masks, read_mask, save_rgb
from .proposals import _candidate_from_mask, _dedup_candidates, generate_part_candidates
from .selector import _candidate_score, _save_part
from .types import BBox, MaskCandidate, PartInstance, PartSpec
from .vlm import part_specs_from_schema


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "part"


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _candidate_overlay_path(candidate: MaskCandidate) -> str:
    path = Path(candidate.mask_path)
    return str(path.with_name(f"{candidate.candidate_id}_overlay.png"))


def _normalized_bbox_to_pixels(bbox, width: int, height: int) -> BBox | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    vals = []
    try:
        vals = [float(x) for x in bbox]
    except Exception:
        return None
    if max(vals) <= 1.5:
        x1, y1, x2, y2 = vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height
    else:
        x1, y1, x2, y2 = vals
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2, y2)


def _rect_mask(shape, bbox: BBox, object_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] = True
    return mask & object_mask


def _heuristic_bbox_from_spec(spec: PartSpec, object_bbox, width: int, height: int) -> BBox:
    bbox = object_bbox if isinstance(object_bbox, BBox) else BBox.from_dict(object_bbox)
    ox1 = max(0, min(width - 1, int(bbox.x1)))
    oy1 = max(0, min(height - 1, int(bbox.y1)))
    ox2 = max(ox1 + 1, min(width, int(bbox.x2)))
    oy2 = max(oy1 + 1, min(height, int(bbox.y2)))
    ow = max(1, ox2 - ox1)
    oh = max(1, oy2 - oy1)
    text = " ".join([spec.name, spec.location, spec.shape_prior] + list(spec.text_prompts)).lower()

    x1, x2 = ox1, ox2
    y1, y2 = oy1, oy2
    if any(k in text for k in ("left", "handle", "grip")):
        x2 = ox1 + int(round(0.72 * ow))
    if any(k in text for k in ("right", "front", "head", "tip", "end")):
        x1 = ox1 + int(round(0.52 * ow))
    if any(k in text for k in ("top", "upper")):
        y2 = oy1 + int(round(0.58 * oh))
    elif any(k in text for k in ("bottom", "lower")):
        y1 = oy1 + int(round(0.42 * oh))
    elif any(k in text for k in ("long", "thin", "bar", "stick", "handle", "extension")):
        y1 = oy1 + int(round(0.12 * oh))
        y2 = oy1 + int(round(0.90 * oh))
    elif any(k in text for k in ("compact", "block", "head")):
        y1 = oy1 + int(round(0.10 * oh))
        y2 = oy1 + int(round(0.90 * oh))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return BBox(x1, y1, x2, y2)


class SegmentationAgent:
    def __init__(
        self,
        image_path,
        object_mask_path,
        object_bbox,
        part_schema,
        detector,
        sam_tool,
        vlm_client,
        output_dir,
        candidates_dir,
        max_parts: int = 6,
        min_part_area_ratio: float = 0.01,
        coverage_threshold: float = 0.75,
        max_retries: int = 2,
        vlm_weight: float = 0.55,
        min_accept_score: float = 0.45,
        max_vlm_candidates_per_part: int = 8,
    ):
        self.image_path = str(image_path)
        self.object_mask_path = str(object_mask_path)
        self.object_bbox = object_bbox
        self.part_schema = part_schema
        self.detector = detector
        self.sam_tool = sam_tool
        self.vlm_client = vlm_client
        self.output_dir = Path(output_dir)
        self.candidates_dir = Path(candidates_dir)
        self.max_parts = int(max_parts)
        self.min_part_area_ratio = float(min_part_area_ratio)
        self.coverage_threshold = float(coverage_threshold)
        self.max_retries = int(max_retries)
        self.vlm_weight = float(vlm_weight)
        self.min_accept_score = float(min_accept_score)
        self.max_vlm_candidates_per_part = int(max_vlm_candidates_per_part)
        self.logs: dict[str, Any] = {"iterations": [], "warnings": []}

    def run(self) -> tuple[list[PartInstance], list[MaskCandidate], dict[str, Any]]:
        if self.vlm_client is None or not getattr(self.vlm_client, "requires_remote_vlm", False):
            raise RuntimeError("Automatic segmentation requires --vlm-provider openai_compatible with a valid API key env var.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        image = load_rgb(self.image_path)
        object_mask = read_mask(self.object_mask_path)
        min_area = max(1, int(self.min_part_area_ratio * mask_area(object_mask)))
        specs = [p for p in part_specs_from_schema(self.part_schema) if p.visible]
        if not specs:
            raise RuntimeError("VLM part schema contained no visible parts.")

        candidates = generate_part_candidates(
            self.image_path,
            self.object_mask_path,
            self.object_bbox,
            self.part_schema,
            self.detector,
            self.sam_tool,
            self.candidates_dir,
            self.min_part_area_ratio,
        )
        candidates = self._append_vlm_box_candidates(candidates, specs, image, object_mask, min_area, reason="initial")
        candidates = _dedup_candidates(candidates)

        best_parts: list[PartInstance] = []
        quality: dict[str, Any] = {"ok": False, "reason": "not_run"}
        score_log: list[dict[str, Any]] = []
        for iteration in range(max(1, self.max_retries + 1)):
            parts, score_log = self._select_parts(candidates, specs, image, object_mask, min_area, iteration)
            quality = self._quality_report(parts, specs, object_mask, min_area)
            self.logs["iterations"].append(
                {
                    "iteration": iteration,
                    "candidate_count": len(candidates),
                    "selected_parts": [p.to_dict() for p in parts],
                    "quality": quality,
                }
            )
            best_parts = parts
            if quality["ok"]:
                break
            if iteration >= self.max_retries:
                break
            missing_specs = [s for s in specs if s.name in quality.get("missing_parts", [])]
            if not missing_specs:
                missing_specs = specs
            before = len(candidates)
            candidates = self._append_vlm_box_candidates(candidates, missing_specs, image, object_mask, min_area, reason="retry")
            candidates = _dedup_candidates(candidates)
            self.logs["iterations"][-1]["retry_added_candidates"] = len(candidates) - before

        _write_json(self.candidates_dir / "candidates_summary.json", [c.to_dict() for c in candidates])
        _write_json(self.output_dir / "agent_logs" / "candidate_scores.json", score_log)
        _write_json(self.output_dir / "agent_logs" / "quality_report.json", quality)
        _write_json(self.output_dir / "agent_logs" / "segmentation_agent_decisions.json", self.logs)
        if not quality.get("ok"):
            raise RuntimeError(f"Segmentation agent failed quality gate: {quality.get('reason')}")
        self._write_selection_outputs(best_parts, image, object_mask, quality)
        return best_parts, candidates, quality

    def _append_vlm_box_candidates(
        self,
        candidates: list[MaskCandidate],
        specs: list[PartSpec],
        image,
        object_mask: np.ndarray,
        min_area: int,
        reason: str,
    ) -> list[MaskCandidate]:
        h, w = object_mask.shape
        try:
            located = self.vlm_client.locate_parts(self.image_path, [s.to_dict() for s in specs])
        except Exception as exc:
            raise RuntimeError(f"VLM locate_parts failed during {reason}: {exc}") from exc
        if not isinstance(located, dict):
            return candidates
        existing_nums = []
        for cand in candidates:
            try:
                existing_nums.append(int(str(cand.candidate_id).split("_")[-1]))
            except Exception:
                pass
        idx = (max(existing_nums) + 1) if existing_nums else len(candidates)
        out = list(candidates)
        added_names: set[str] = set()
        for item in located.get("parts", []):
            name = str(item.get("name", "")).strip()
            spec = next((s for s in specs if s.name.lower() == name.lower()), None)
            if spec is None:
                continue
            bbox = _normalized_bbox_to_pixels(item.get("bbox"), w, h)
            if bbox is None:
                continue
            masks = []
            if self.sam_tool is not None:
                try:
                    masks = self.sam_tool.segment_from_box(self.image_path, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                except Exception as exc:
                    self.logs["warnings"].append(f"SAM box segmentation failed for {name}: {exc}")
                    masks = []
            rect = _rect_mask(object_mask.shape, bbox, object_mask)
            rect_area = max(1, int(rect.sum()))
            if masks:
                sam_mask = np.asarray(max(masks, key=mask_area), dtype=bool) & object_mask
                object_area = max(1, int(object_mask.sum()))
                too_large_for_box = sam_mask.sum() > max(rect_area * 3.0, object_area * 0.70)
                if too_large_for_box:
                    mask = rect
                    source = "vlm_box"
                    self.logs["warnings"].append(
                        f"SAM mask for {name} was too large for VLM bbox; used bbox mask instead."
                    )
                else:
                    mask = sam_mask
                    source = "vlm_box_sam"
            else:
                mask = rect
                source = "vlm_box"
            if mask.sum() < min_area or mask_inside_ratio(mask, object_mask) < 0.80:
                continue
            out.append(
                _candidate_from_mask(
                    mask,
                    source,
                    self.candidates_dir,
                    idx,
                    image,
                    object_mask,
                    prompt=spec.name,
                    metadata={
                        "vlm_bbox": item.get("bbox"),
                        "vlm_confidence": item.get("confidence"),
                        "vlm_reason": item.get("reason"),
                        "reason": reason,
                    },
                )
            )
            added_names.add(spec.name.lower())
            idx += 1

        existing_schema_names = {
            spec.name.lower()
            for spec in specs
            if any(
                cand.source == "schema_location" and spec.name.lower() in str(cand.prompt or "").lower()
                for cand in out
            )
        }
        for spec in specs:
            if spec.name.lower() in existing_schema_names:
                continue
            bbox = _heuristic_bbox_from_spec(spec, self.object_bbox, w, h)
            mask = _rect_mask(object_mask.shape, bbox, object_mask)
            if mask.sum() < min_area or mask_inside_ratio(mask, object_mask) < 0.80:
                continue
            out.append(
                _candidate_from_mask(
                    mask,
                    "schema_location",
                    self.candidates_dir,
                    idx,
                    image,
                    object_mask,
                    prompt=spec.name,
                    metadata={
                        "heuristic_bbox": bbox.to_dict(),
                        "reason": f"{reason}: schema-location fallback for visible part",
                    },
                )
            )
            self.logs["warnings"].append(f"Added schema-location fallback candidate for {spec.name}.")
            idx += 1
        return out

    def _select_parts(
        self,
        candidates: list[MaskCandidate],
        specs: list[PartSpec],
        image,
        object_mask: np.ndarray,
        min_area: int,
        iteration: int,
    ) -> tuple[list[PartInstance], list[dict[str, Any]]]:
        scored_by_spec: dict[str, list[dict[str, Any]]] = {}
        score_log: list[dict[str, Any]] = []
        object_area = max(1, int(object_mask.sum()))
        for spec in specs:
            raw_scores = []
            for cand in candidates:
                mask = read_mask(cand.mask_path) & object_mask
                rule_score = _candidate_score(cand, mask, spec, object_mask, self.min_part_area_ratio)
                raw_scores.append((rule_score, cand, mask))
            raw_scores.sort(key=lambda x: x[0], reverse=True)
            top = raw_scores[: max(1, self.max_vlm_candidates_per_part)]
            spec_scores = []
            for rule_score, cand, mask in top:
                overlay = _candidate_overlay_path(cand)
                try:
                    vlm_result = self.vlm_client.score_candidate_for_part(self.image_path, overlay, spec)
                except Exception as exc:
                    raise RuntimeError(f"VLM candidate scoring failed for {spec.name}/{cand.candidate_id}: {exc}") from exc
                vlm_score = float(vlm_result.get("score", 0.0) or 0.0)
                vlm_score = max(0.0, min(1.0, vlm_score))
                combined = (1.0 - self.vlm_weight) * rule_score + self.vlm_weight * vlm_score
                area_ratio = float(mask.sum() / object_area)
                if len(specs) > 1 and area_ratio > 0.80 and cand.source in {"appearance_cluster", "sam_auto"}:
                    combined = min(combined, 0.42)
                item = {
                    "iteration": iteration,
                    "part_name": spec.name,
                    "candidate_id": cand.candidate_id,
                    "source": cand.source,
                    "rule_score": rule_score,
                    "vlm_score": vlm_score,
                    "combined_score": float(combined),
                    "area_ratio": area_ratio,
                    "vlm_reason": vlm_result.get("reason"),
                }
                score_log.append(item)
                spec_scores.append({"spec": spec, "candidate": cand, "mask": mask, "score": float(combined), "log": item})
            spec_scores.sort(key=lambda x: x["score"], reverse=True)
            scored_by_spec[spec.name] = spec_scores

        spec_order = sorted(
            specs,
            key=lambda s: (
                scored_by_spec.get(s.name, [{}])[0].get("mask", object_mask).sum()
                if scored_by_spec.get(s.name)
                else object_area
            ),
        )
        resolved = []
        used = np.zeros_like(object_mask, dtype=bool)
        for spec in spec_order:
            chosen = None
            for item in scored_by_spec.get(spec.name, []):
                if item["score"] < self.min_accept_score:
                    continue
                mask = item["mask"].copy()
                mask[used] = False
                if mask.sum() < min_area:
                    continue
                chosen = dict(item)
                chosen["mask"] = mask
                break
            if chosen is not None:
                used |= chosen["mask"]
                resolved.append(chosen)
            if len(resolved) >= self.max_parts:
                break

        expected_names = {spec.name for spec in specs}
        selected_names = {item["spec"].name for item in resolved}
        if expected_names.issubset(selected_names):
            self._fill_residual_by_nearest_part(resolved, object_mask, min_area)

        parts = []
        for idx, item in enumerate(resolved):
            cand = item["candidate"]
            part = _save_part(
                image,
                item["mask"],
                idx,
                item["spec"].name,
                item["score"],
                [cand.candidate_id],
                item["spec"],
                self.output_dir,
            )
            part.metadata.setdefault("selection", {})
            part.metadata["selection"].update(item["log"])
            _write_json(Path(part.mask_path).parent / "part_summary.json", {"part": part.to_dict()})
            parts.append(part)
        return parts, score_log

    def _fill_residual_by_nearest_part(self, resolved: list[dict[str, Any]], object_mask: np.ndarray, min_area: int) -> None:
        union = np.zeros_like(object_mask, dtype=bool)
        for item in resolved:
            union |= item["mask"]
        residual = object_mask & ~union
        if residual.sum() < min_area:
            return

        centers = []
        for item in resolved:
            ys, xs = np.where(item["mask"])
            if len(xs) == 0:
                bbox = item["candidate"].bbox
                centers.append(((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0))
            else:
                centers.append((float(xs.mean()), float(ys.mean())))
        rys, rxs = np.where(residual)
        center_arr = np.asarray(centers, dtype=np.float32)
        dx = rxs[:, None].astype(np.float32) - center_arr[None, :, 0]
        dy = rys[:, None].astype(np.float32) - center_arr[None, :, 1]
        labels = np.argmin(dx * dx + dy * dy, axis=1)
        for idx, item in enumerate(resolved):
            take = labels == idx
            count = int(take.sum())
            if count == 0:
                continue
            item["mask"][rys[take], rxs[take]] = True
            item["log"]["residual_fill_pixels"] = count
        self.logs["warnings"].append("Filled residual object-mask pixels by nearest selected part.")

    def _quality_report(self, parts: list[PartInstance], specs: list[PartSpec], object_mask: np.ndarray, min_area: int) -> dict[str, Any]:
        names = {p.name for p in parts}
        expected = [s.name for s in specs if s.visible]
        missing = [name for name in expected if name not in names]
        masks = [read_mask(p.mask_path) & object_mask for p in parts]
        union = np.logical_or.reduce(masks) if masks else np.zeros_like(object_mask, dtype=bool)
        coverage = float(union.sum() / max(1, object_mask.sum()))
        pair_overlap = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                pair_overlap += int(np.logical_and(masks[i], masks[j]).sum())
        overlap_ratio = float(pair_overlap / max(1, object_mask.sum()))
        tiny = [p.name for p in parts if p.area < min_area]
        ok = not missing and not tiny and coverage >= min(self.coverage_threshold, 0.98)
        reason = "ok"
        if missing:
            reason = "missing_visible_parts"
        elif tiny:
            reason = "tiny_selected_parts"
        elif coverage < min(self.coverage_threshold, 0.98):
            reason = "low_coverage"
        return {
            "ok": bool(ok),
            "reason": reason,
            "expected_parts": expected,
            "selected_parts": [p.name for p in parts],
            "missing_parts": missing,
            "tiny_parts": tiny,
            "coverage": coverage,
            "overlap_ratio": overlap_ratio,
        }

    def _write_selection_outputs(self, parts: list[PartInstance], image, object_mask: np.ndarray, quality: dict[str, Any]):
        masks = [read_mask(p.mask_path) for p in parts]
        if masks:
            overlay = overlay_multiple_masks(image, masks, labels=[p.name for p in parts])
            save_rgb(overlay, self.output_dir / "parts" / "parts_overlay.png")
        summary = dict(quality)
        summary["parts"] = [p.to_dict() for p in parts]
        _write_json(self.output_dir / "parts" / "selection_summary.json", summary)
