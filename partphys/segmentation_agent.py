from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .contact_sheet import build_candidate_contact_sheet
from .image_utils import bbox_area, load_rgb, mask_area, mask_inside_ratio, mask_to_bbox, overlay_multiple_masks, read_mask, save_rgb
from .part_traits import is_main_like_part, is_specific_part, specificity_rank
from .proposals import _candidate_from_mask, _dedup_candidates, choose_sam_mask_for_box, generate_part_candidates
from .selector import _candidate_score, _save_part
from .types import BBox, MaskCandidate, PartInstance, PartSpec
from .vlm import part_specs_from_schema


RECT_SOURCES = {"vlm_box", "schema_location"}
SAM_LIKE_SOURCES = {"text_box_sam", "sam_auto", "object_body", "appearance_cluster", "vlm_box_sam"}


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "part"


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _normalized_bbox_to_pixels(bbox, width: int, height: int) -> BBox | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
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


def _candidate_quality(candidate: MaskCandidate) -> float:
    values = [candidate.inside_object_ratio]
    for value in (candidate.stability_score, candidate.predicted_iou):
        if value is not None:
            values.append(float(value))
    values.append(float(candidate.metadata.get("boundary_quality", 0.5)))
    return float(max(0.0, min(1.0, np.mean(values))))


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
        min_part_area_ratio: float = 0.002,
        coverage_threshold: float = 0.75,
        max_retries: int = 2,
        vlm_weight: float = 0.55,
        min_accept_score: float = 0.45,
        max_vlm_candidates_per_part: int = 12,
        segmentation_mode: str = "candidate_pool",
        use_vlm_bbox_proposals: bool = False,
        use_schema_location_proposals: bool = False,
        strict_segmentation: bool = False,
        residual_policy: str = "unknown",
        candidate_top_k: int = 40,
        candidate_contact_sheet_top_k: int = 24,
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
        self.segmentation_mode = segmentation_mode
        self.use_vlm_bbox_proposals = bool(use_vlm_bbox_proposals or segmentation_mode == "legacy_vlm_bbox")
        self.use_schema_location_proposals = bool(use_schema_location_proposals or segmentation_mode == "legacy_vlm_bbox")
        self.strict_segmentation = bool(strict_segmentation)
        self.residual_policy = residual_policy
        self.candidate_top_k = int(candidate_top_k)
        self.candidate_contact_sheet_top_k = int(candidate_contact_sheet_top_k)
        self.has_remote_vlm = self.vlm_client is not None and getattr(self.vlm_client, "requires_remote_vlm", False)
        self.logs: dict[str, Any] = {"iterations": [], "warnings": []}

    def run(self) -> tuple[list[PartInstance], list[MaskCandidate], dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "agent_logs").mkdir(parents=True, exist_ok=True)
        image = load_rgb(self.image_path)
        object_mask = read_mask(self.object_mask_path)
        min_area = max(1, int(self.min_part_area_ratio * mask_area(object_mask)))
        specs = [p for p in part_specs_from_schema(self.part_schema) if p.visible]
        if not specs:
            specs = [PartSpec(name="body", text_prompts=["object body"], expected_materials=["Plastic"], location="main object body", shape_prior="main body")]

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
        if self.use_vlm_bbox_proposals:
            candidates = self._append_vlm_box_candidates(candidates, specs, image, object_mask, min_area, reason="explicit")
        if self.use_schema_location_proposals:
            candidates = self._append_schema_location_candidates(candidates, specs, image, object_mask, min_area, reason="explicit")
        candidates = self._rank_general_candidates(_dedup_candidates(candidates))[: max(1, self.candidate_top_k)]
        _write_json(self.candidates_dir / "candidates_summary.json", [c.to_dict() for c in candidates])

        contact_candidates = candidates[: max(1, self.candidate_contact_sheet_top_k)]
        contact_sheet = build_candidate_contact_sheet(
            self.image_path,
            contact_candidates,
            self.output_dir / "agent_logs" / "contact_sheet_all.png",
            max_candidates=self.candidate_contact_sheet_top_k,
        )
        vlm_rankings = self._rank_with_vlm(contact_sheet, specs, contact_candidates)
        parts, score_log = self._select_parts(candidates, specs, image, object_mask, min_area, vlm_rankings)
        quality = self._quality_report(parts, specs, object_mask, min_area)
        quality["candidate_count"] = len(candidates)
        self.logs["iterations"].append(
            {
                "iteration": 0,
                "candidate_count": len(candidates),
                "selected_parts": [p.to_dict() for p in parts],
                "quality": quality,
            }
        )
        if not quality.get("ok"):
            self.logs["warnings"].append(f"Segmentation quality warning: {quality.get('reason')}")

        _write_json(self.output_dir / "agent_logs" / "candidate_scores.json", score_log)
        _write_json(self.output_dir / "agent_logs" / "quality_report.json", quality)
        _write_json(self.output_dir / "agent_logs" / "segmentation_agent_decisions.json", self.logs)
        self._write_selection_outputs(parts, image, object_mask, quality)
        if self.strict_segmentation and not quality.get("ok"):
            raise RuntimeError(f"Segmentation agent failed quality gate: {quality.get('reason')}")
        return parts, candidates, quality

    def _rank_general_candidates(self, candidates: list[MaskCandidate]) -> list[MaskCandidate]:
        priority = {
            "text_box_sam": 6,
            "sam_auto": 5,
            "object_body": 4,
            "appearance_cluster": 3,
            "vlm_box_sam": 2,
            "schema_location": 1,
            "vlm_box": 0,
        }
        return sorted(candidates, key=lambda c: (priority.get(c.source, 0), _candidate_quality(c), c.area), reverse=True)

    def _rank_with_vlm(self, contact_sheet: str, specs: list[PartSpec], candidates: list[MaskCandidate]) -> dict[str, dict[str, dict[str, Any]]]:
        rankings: dict[str, dict[str, dict[str, Any]]] = {}
        if not self.has_remote_vlm or self.vlm_client is None:
            self.logs["warnings"].append("No remote VLM candidate ranking available; using rule-only scores.")
            return rankings
        try:
            result = self.vlm_client.rank_candidates_for_parts(
                self.image_path,
                contact_sheet,
                [s.to_dict() for s in specs],
                [c.to_dict() for c in candidates],
            )
        except Exception as exc:
            self.logs["warnings"].append(f"VLM candidate ranking failed: {exc}")
            return rankings
        for warning in result.get("warnings", []) if isinstance(result, dict) else []:
            self.logs["warnings"].append(str(warning))
        valid_ids = {c.candidate_id for c in candidates}
        for item in result.get("rankings", []) if isinstance(result, dict) else []:
            part_name = str(item.get("part_name", ""))
            if item.get("missing"):
                rankings.setdefault(part_name, {})
                continue
            for cand in item.get("candidates", []) or []:
                cid = str(cand.get("candidate_id", ""))
                if cid not in valid_ids:
                    self.logs["warnings"].append(f"VLM returned unknown candidate_id {cid} for {part_name}; ignored.")
                    continue
                score = max(0.0, min(1.0, float(cand.get("score", 0.0) or 0.0)))
                rankings.setdefault(part_name, {})[cid] = {"score": score, "reason": cand.get("reason")}
        return rankings

    def _append_vlm_box_candidates(
        self,
        candidates: list[MaskCandidate],
        specs: list[PartSpec],
        image,
        object_mask: np.ndarray,
        min_area: int,
        reason: str,
    ) -> list[MaskCandidate]:
        if self.vlm_client is None:
            return candidates
        h, w = object_mask.shape
        try:
            located = self.vlm_client.locate_parts(self.image_path, [s.to_dict() for s in specs])
        except Exception as exc:
            self.logs["warnings"].append(f"VLM locate_parts failed during {reason}: {exc}")
            return candidates
        if not isinstance(located, dict):
            return candidates
        idx = self._next_candidate_idx(candidates)
        out = list(candidates)
        for item in located.get("parts", []):
            name = str(item.get("name", "")).strip()
            spec = next((s for s in specs if s.name.lower() == name.lower()), None)
            if spec is None:
                continue
            bbox = _normalized_bbox_to_pixels(item.get("bbox"), w, h)
            if bbox is None:
                continue
            rect = _rect_mask(object_mask.shape, bbox, object_mask)
            if rect.sum() < min_area or mask_inside_ratio(rect, object_mask) < 0.80:
                continue
            mask = None
            source = "vlm_box"
            if self.sam_tool is not None:
                try:
                    masks = self.sam_tool.segment_from_box(self.image_path, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                    mask = choose_sam_mask_for_box(masks, bbox_area(bbox), object_mask)
                except Exception as exc:
                    self.logs["warnings"].append(f"SAM box segmentation failed for VLM bbox {name}: {exc}")
            if mask is not None and mask.sum() >= min_area and mask_inside_ratio(mask, object_mask) >= 0.80:
                source = "vlm_box_sam"
            else:
                mask = rect
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
            idx += 1
        return out

    def _append_schema_location_candidates(
        self,
        candidates: list[MaskCandidate],
        specs: list[PartSpec],
        image,
        object_mask: np.ndarray,
        min_area: int,
        reason: str,
    ) -> list[MaskCandidate]:
        h, w = object_mask.shape
        out = list(candidates)
        idx = self._next_candidate_idx(out)
        for spec in specs:
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
                    metadata={"heuristic_bbox": bbox.to_dict(), "reason": reason},
                )
            )
            idx += 1
        return out

    def _next_candidate_idx(self, candidates: list[MaskCandidate]) -> int:
        nums = []
        for cand in candidates:
            try:
                nums.append(int(str(cand.candidate_id).split("_")[-1]))
            except Exception:
                pass
        return max(nums) + 1 if nums else len(candidates)

    def _select_parts(
        self,
        candidates: list[MaskCandidate],
        specs: list[PartSpec],
        image,
        object_mask: np.ndarray,
        min_area: int,
        vlm_rankings: dict[str, dict[str, dict[str, Any]]],
    ) -> tuple[list[PartInstance], list[dict[str, Any]]]:
        object_area = max(1, int(object_mask.sum()))
        score_log: list[dict[str, Any]] = []
        scored_by_spec: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            spec_scores = []
            rule_rows = []
            for cand in candidates:
                mask = read_mask(cand.mask_path) & object_mask
                rule_score, breakdown = _candidate_score(cand, mask, spec, object_mask, self.min_part_area_ratio, return_breakdown=True)
                rule_rows.append((rule_score, cand, mask, breakdown))
            rule_rows.sort(key=lambda x: x[0], reverse=True)
            non_rect_available = any(c.source in SAM_LIKE_SOURCES and score >= self.min_accept_score * 0.70 for score, c, _, _ in rule_rows)
            for rule_score, cand, mask, breakdown in rule_rows[: max(1, self.max_vlm_candidates_per_part)]:
                quality = _candidate_quality(cand)
                vlm_item = vlm_rankings.get(spec.name, {}).get(cand.candidate_id)
                vlm_score = None if vlm_item is None else float(vlm_item.get("score", 0.0))
                if vlm_score is None:
                    final_score = 0.80 * rule_score + 0.20 * quality
                else:
                    final_score = 0.45 * rule_score + 0.45 * vlm_score + 0.10 * quality
                if cand.source in {"vlm_box", "schema_location"}:
                    final_score = min(final_score, 0.35)
                if cand.source == "vlm_box_sam" and not self.use_vlm_bbox_proposals:
                    final_score = min(final_score, 0.60)
                if cand.source in RECT_SOURCES and non_rect_available:
                    final_score = min(final_score, 0.30)
                area_ratio = float(mask.sum() / object_area)
                if area_ratio > 0.95 and not is_main_like_part(spec):
                    final_score = min(final_score, 0.30)
                elif area_ratio > 0.70 and is_specific_part(spec):
                    final_score = min(final_score, 0.45)
                item = {
                    "part_name": spec.name,
                    "candidate_id": cand.candidate_id,
                    "source": cand.source,
                    "rule_score": float(rule_score),
                    "rule_breakdown": breakdown,
                    "vlm_rank_score": vlm_score,
                    "candidate_quality": float(quality),
                    "final_score": float(max(0.0, min(1.0, final_score))),
                    "area_ratio": area_ratio,
                    "reason": (vlm_item or {}).get("reason"),
                    "rect_suppressed_by_sam": bool(cand.source in RECT_SOURCES and non_rect_available),
                }
                score_log.append(item)
                spec_scores.append({"spec": spec, "candidate": cand, "mask": mask, "score": item["final_score"], "log": item})
            spec_scores.sort(key=lambda x: x["score"], reverse=True)
            scored_by_spec[spec.name] = spec_scores

        resolved: list[dict[str, Any]] = []
        used = np.zeros_like(object_mask, dtype=bool)
        for spec in sorted(specs, key=specificity_rank):
            chosen = None
            for item in scored_by_spec.get(spec.name, []):
                if item["score"] < self.min_accept_score:
                    continue
                mask = item["mask"].copy()
                if not is_specific_part(spec):
                    mask[used] = False
                else:
                    mask[used] = False
                if mask.sum() < min_area:
                    continue
                chosen = dict(item)
                chosen["mask"] = mask
                break
            if chosen is None:
                self.logs["warnings"].append(f"No reliable candidate selected for visible part {spec.name}.")
                continue
            used |= chosen["mask"]
            resolved.append(chosen)
            if len(resolved) >= self.max_parts:
                break

        self._apply_residual_policy(resolved, object_mask, min_area)

        parts: list[PartInstance] = []
        for idx, item in enumerate(resolved[: self.max_parts]):
            cand = item.get("candidate")
            ids = [cand.candidate_id] if cand is not None else [item.get("candidate_id", "residual_or_object")]
            part = _save_part(
                image,
                item["mask"],
                idx,
                item["spec"].name,
                item["score"],
                ids,
                item["spec"],
                self.output_dir,
            )
            part.metadata.setdefault("selection", {})
            part.metadata["selection"].update(item.get("log", {}))
            _write_json(Path(part.mask_path).parent / "part_summary.json", {"part": part.to_dict()})
            parts.append(part)
        return parts, score_log

    def _apply_residual_policy(self, resolved: list[dict[str, Any]], object_mask: np.ndarray, min_area: int) -> None:
        if not resolved:
            return
        union = np.zeros_like(object_mask, dtype=bool)
        for item in resolved:
            union |= item["mask"]
        residual = object_mask & ~union
        if residual.sum() < min_area:
            return
        if self.residual_policy == "ignore":
            return
        if self.residual_policy == "fill_nearest":
            self._fill_residual_by_nearest_part(resolved, object_mask, min_area)
            return
        if self.residual_policy == "unknown" and len(resolved) < self.max_parts:
            spec = PartSpec(
                name="unknown_body",
                text_prompts=["unknown object residual"],
                expected_materials=["Plastic"],
                location="unassigned residual object pixels",
                shape_prior="residual body",
                physical_role="unassigned residual material",
                should_simulate_separately=False,
                visible=True,
                physics_group="global_body",
            )
            resolved.append(
                {
                    "spec": spec,
                    "candidate": None,
                    "candidate_id": "unknown_residual",
                    "mask": residual,
                    "score": 0.25,
                    "log": {"source": "residual_unknown", "residual_policy": "unknown"},
                }
            )
            self.logs["warnings"].append("Added unknown_body residual part.")

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
            if len(xs) == 0 and item.get("candidate") is not None:
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
            item.setdefault("log", {})["residual_fill_pixels"] = count
        self.logs["warnings"].append("Filled residual object-mask pixels by nearest selected part.")

    def _quality_report(self, parts: list[PartInstance], specs: list[PartSpec], object_mask: np.ndarray, min_area: int) -> dict[str, Any]:
        names = {p.name for p in parts if p.name != "unknown_body"}
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
        ok = coverage >= self.coverage_threshold and not tiny and overlap_ratio <= 0.20
        reason = "ok"
        if tiny:
            reason = "tiny_selected_parts"
        elif overlap_ratio > 0.20:
            reason = "high_overlap"
        elif coverage < self.coverage_threshold:
            reason = "low_coverage"
        selected_candidate_ids = {p.name: list(p.candidate_ids) for p in parts}
        return {
            "ok": bool(ok),
            "reason": reason,
            "expected_parts": expected,
            "selected_parts": [p.name for p in parts],
            "missing_parts": missing,
            "coverage": coverage,
            "overlap_ratio": overlap_ratio,
            "tiny_parts": tiny,
            "accepted_with_warnings": bool(not ok or missing),
            "strict_segmentation": self.strict_segmentation,
            "residual_policy": self.residual_policy,
            "candidate_count": 0,
            "selected_candidate_ids": selected_candidate_ids,
        }

    def _write_selection_outputs(self, parts: list[PartInstance], image, object_mask: np.ndarray, quality: dict[str, Any]):
        masks = [read_mask(p.mask_path) for p in parts]
        if masks:
            overlay = overlay_multiple_masks(image, masks, labels=[p.name for p in parts])
            save_rgb(overlay, self.output_dir / "parts" / "parts_overlay.png")
        summary = dict(quality)
        summary["parts"] = [p.to_dict() for p in parts]
        _write_json(self.output_dir / "parts" / "selection_summary.json", summary)
