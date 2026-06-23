from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .image_utils import load_rgb, overlay_multiple_masks, read_mask, resize_to_square_with_padding, save_rgb
from .scene_builder import VIEW_LABELS


ACTION_SPACE = [
    "generate_part_candidates",
    "rank_candidates",
    "rerank_candidates",
    "rerun_with_schema_location_proposals",
    "repair_object_mask",
    "repair_part_mask",
    "compile_layout",
    "fill_unknown_by_nearest_part",
    "increase_candidate_pool",
    "multiview_align",
    "knn_cleanup",
]


def _get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _part_dict(part) -> dict[str, Any]:
    return part.to_dict() if hasattr(part, "to_dict") else dict(part)


def _part_name(part) -> str:
    if hasattr(part, "name"):
        return str(part.name)
    return str(part.get("name", "part"))


def _is_residual_part(part) -> bool:
    name = _part_name(part).lower()
    metadata = getattr(part, "metadata", {}) or {}
    group = str(getattr(part, "physics_group", "") or metadata.get("physics_group", "")).lower()
    return name in {"unknown_body", "unknown", "residual", "residual_body"} or "residual" in name or group in {"global_body", "unknown", "residual"}


def _action_name(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("action") or action.get("name") or "").lower()
    return str(action).lower()


class PartSegAgentController:
    """Planner/critic controller around the deterministic segmentation tools."""

    def __init__(self, config, scene_dir, vlm_client, object_name: str, image_path: str, view_paths=None):
        self.config = config
        self.scene_dir = Path(scene_dir)
        self.vlm_client = vlm_client
        self.object_name = object_name
        self.image_path = str(image_path)
        self.view_paths = [str(Path(p).expanduser()) for p in (view_paths or []) if p]
        self.enabled = str(_get(config, "agent_mode", "pipeline")).lower() == "agent"
        self.max_rounds = max(1, int(_get(config, "agent_rounds", 2)))
        self.log_dir = self.scene_dir / "agent_logs"
        self.state_path = self.scene_dir / "agent_logs" / "agent_state.json"
        self.state: dict[str, Any] = {
            "enabled": self.enabled,
            "object": object_name,
            "image_path": self.image_path,
            "views": [
                {"label": label, "image_path": path}
                for label, path in zip(VIEW_LABELS, self.view_paths)
            ],
            "mode": _get(config, "agent_mode", "pipeline"),
            "max_rounds": self.max_rounds,
            "action_space": list(ACTION_SPACE),
            "planner": None,
            "planner_evidence_path": None,
            "rounds": [],
            "action_trace": [],
            "multiview": None,
            "final": {},
        }

    def save(self) -> None:
        _write_json(self.state_path, self.state)

    def _make_grid(self, images: list[Image.Image], labels: list[str], output_path: Path, tile_size: int = 512) -> str:
        tiles = []
        for image, label in zip(images, labels):
            tile = resize_to_square_with_padding(image.convert("RGB"), tile_size)
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, min(tile_size, 220), 30), fill=(255, 255, 255))
            draw.text((8, 8), label, fill=(0, 0, 0))
            tiles.append(tile)
        while len(tiles) < 4:
            tiles.append(Image.new("RGB", (tile_size, tile_size), (255, 255, 255)))
        canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), (255, 255, 255))
        for idx, tile in enumerate(tiles[:4]):
            canvas.paste(tile, ((idx % 2) * tile_size, (idx // 2) * tile_size))
        save_rgb(canvas, output_path)
        return str(output_path)

    def _planner_evidence(self) -> str:
        if len(self.view_paths) < 2:
            return self.image_path
        images = []
        labels = []
        for label, path in zip(VIEW_LABELS, self.view_paths):
            p = Path(path)
            if not p.exists():
                continue
            images.append(load_rgb(p))
            labels.append(label)
        if len(images) < 2:
            return self.image_path
        path = self.log_dir / "planner_multiview_evidence.png"
        evidence_path = self._make_grid(images, labels, path)
        self.state["planner_evidence_path"] = evidence_path
        return evidence_path

    def plan(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            self.state["planner"] = {"source": "disabled", "schema": schema}
            self.save()
            return {}
        evidence_path = self._planner_evidence()
        try:
            plan = self.vlm_client.plan_part_segmentation(evidence_path, self.object_name)
            if not isinstance(plan, dict):
                plan = {}
            plan.setdefault("object", self.object_name)
            plan.setdefault("parts", schema.get("parts", []))
            plan.setdefault("tool_plan", [])
            plan["source"] = "vlm_or_fallback"
            plan["visual_evidence"] = evidence_path
        except Exception as exc:
            plan = {
                "object": self.object_name,
                "parts": schema.get("parts", []),
                "tool_plan": [],
                "warnings": [f"Planner failed: {exc}"],
                "source": "exception_fallback",
                "visual_evidence": evidence_path,
            }
        self.state["planner"] = plan
        self.save()
        return plan

    def _planned_repairs(self, previous_critique: dict[str, Any] | None, round_idx: int) -> list[dict[str, Any]]:
        if not previous_critique or round_idx == 0:
            return []
        repairs = []
        for action in previous_critique.get("repair_actions") or []:
            if isinstance(action, dict):
                name = _action_name(action)
                if name:
                    repairs.append(dict(action, action=name))
            else:
                name = _action_name(action)
                if name:
                    repairs.append({"action": name})
        if repairs:
            return repairs
        text = json.dumps(previous_critique.get("failure_modes") or [], ensure_ascii=False).lower()
        if any(k in text for k in ("missing", "tiny", "small", "part")):
            repairs.append({"action": "rerun_with_schema_location_proposals", "reason": "critic reported missing or tiny parts"})
            repairs.append({"action": "increase_candidate_pool", "reason": "critic reported missing or tiny parts"})
        if any(k in text for k in ("unknown", "residual", "coverage")):
            repairs.append({"action": "fill_unknown_by_nearest_part", "reason": "critic reported unknown residual or low coverage"})
        if any(k in text for k in ("overlap", "leak", "plate", "background")):
            repairs.append({"action": "compile_layout", "reason": "critic reported overlap or leakage"})
        return repairs

    def _overrides_from_repairs(self, repairs: list[dict[str, Any]]) -> dict[str, Any]:
        if not repairs:
            return {}
        overrides: dict[str, Any] = {
            "candidate_top_k": max(int(_get(self.config, "candidate_top_k", 40)), 60),
            "candidate_contact_sheet_top_k": max(int(_get(self.config, "candidate_contact_sheet_top_k", 24)), 36),
            "segmentation_min_accept_score": max(0.30, float(_get(self.config, "segmentation_min_accept_score", 0.45)) - 0.05),
        }
        for repair in repairs:
            name = _action_name(repair)
            if name in {"rerun_with_schema_location_proposals", "generate_part_candidates", "repair_part_mask"}:
                overrides["use_schema_location_proposals"] = True
            if name in {"increase_candidate_pool", "generate_part_candidates"}:
                overrides["candidate_top_k"] = max(int(overrides.get("candidate_top_k", 0)), 80)
                overrides["candidate_contact_sheet_top_k"] = max(int(overrides.get("candidate_contact_sheet_top_k", 0)), 48)
                overrides["max_vlm_candidates_per_part"] = max(int(_get(self.config, "max_vlm_candidates_per_part", 12)), 20)
            if name in {"rank_candidates", "rerank_candidates"}:
                overrides["candidate_contact_sheet_top_k"] = max(int(overrides.get("candidate_contact_sheet_top_k", 0)), 48)
                overrides["segmentation_vlm_weight"] = max(float(_get(self.config, "segmentation_vlm_weight", 0.55)), 0.70)
            if name in {"fill_unknown_by_nearest_part", "knn_cleanup"}:
                overrides["residual_policy"] = "fill_nearest"
                overrides["coverage_threshold"] = min(float(_get(self.config, "coverage_threshold", 0.75)), 0.70)
            if name in {"compile_layout"}:
                overrides["strict_segmentation"] = True
                overrides["segmentation_min_accept_score"] = max(float(overrides.get("segmentation_min_accept_score", 0.0)), 0.40)
            if name in {"repair_object_mask"} and getattr(self.vlm_client, "requires_remote_vlm", False):
                overrides["use_vlm_bbox_proposals"] = True
        if getattr(self.vlm_client, "requires_remote_vlm", False):
            overrides.setdefault("use_vlm_bbox_proposals", True)
        return overrides

    def _planned_tool_actions(self, round_idx: int, overrides: dict[str, Any]) -> list[dict[str, Any]]:
        actions = [
            {"action": "generate_part_candidates", "source": "default_executor"},
            {"action": "rank_candidates", "source": "default_executor"},
            {"action": "compile_layout", "source": "default_executor"},
        ]
        if overrides.get("use_schema_location_proposals"):
            actions.append({"action": "rerun_with_schema_location_proposals", "source": "repair_policy"})
        if overrides.get("use_vlm_bbox_proposals"):
            actions.append({"action": "repair_part_mask", "source": "repair_policy", "method": "vlm_bbox_proposals"})
        if overrides.get("residual_policy") == "fill_nearest":
            actions.append({"action": "fill_unknown_by_nearest_part", "source": "repair_policy"})
        if round_idx > 0:
            applied = [key for key in sorted(overrides) if not str(key).startswith("_")]
            for action in actions:
                action.setdefault("parameters", {})["overrides"] = applied
        return actions

    def round_overrides(self, previous_critique: dict[str, Any] | None, round_idx: int, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled or round_idx == 0:
            return {}
        repairs = self._planned_repairs(previous_critique, round_idx)
        overrides = self._overrides_from_repairs(repairs)
        self.state["action_trace"].append(
            {
                "round": round_idx,
                "context": context or {"stage": "canonical"},
                "repairs": repairs,
                "overrides": overrides,
            }
        )
        self.save()
        return overrides

    def _deterministic_critique(self, quality: dict[str, Any]) -> dict[str, Any]:
        failure_modes = []
        repair_actions = []
        reason = quality.get("reason")
        if reason and reason != "ok":
            failure_modes.append(str(reason))
        for part_name in quality.get("missing_parts", []) or []:
            failure_modes.append(f"missing part: {part_name}")
            repair_actions.append({"action": "rerun_with_schema_location_proposals", "target": part_name})
        unknown_ratio = float(quality.get("unknown_ratio", 0.0) or 0.0)
        if unknown_ratio > 0.05:
            failure_modes.append(f"unknown residual too large: {unknown_ratio:.3f}")
            repair_actions.append({"action": "fill_unknown_by_nearest_part", "target": "unknown_body"})
        for issue in quality.get("semantic_issues", []) or []:
            failure_modes.append(str(issue))
            if "tiny" in str(issue):
                repair_actions.append({"action": "increase_candidate_pool", "target": str(issue).split(":")[0]})
            if any(k in str(issue).lower() for k in ("leak", "plate", "background", "overlap")):
                repair_actions.append({"action": "compile_layout", "target": str(issue).split(":")[0]})
        overlap = float(quality.get("overlap_ratio", 0.0) or 0.0)
        overlap_limit = float(_get(self.config, "agent_max_overlap_ratio", 0.10))
        if overlap > overlap_limit:
            failure_modes.append(f"part overlap too high: {overlap:.3f}")
            repair_actions.append({"action": "compile_layout", "target": "all_parts"})
        ok = bool(quality.get("ok", False)) and not failure_modes
        return {"ok": ok, "failure_modes": failure_modes, "repair_actions": repair_actions, "notes": ["deterministic quality critique"]}

    def critique(self, parts, quality: dict[str, Any], overlay_path, round_idx: int) -> dict[str, Any]:
        deterministic = self._deterministic_critique(quality)
        if not self.enabled:
            return deterministic
        if overlay_path and Path(overlay_path).exists():
            try:
                vlm_critique = self.vlm_client.critique_part_segmentation(str(overlay_path), parts, quality)
                if not isinstance(vlm_critique, dict):
                    vlm_critique = {}
            except Exception as exc:
                vlm_critique = {"ok": deterministic["ok"], "failure_modes": [], "repair_actions": [], "notes": [f"VLM critic failed: {exc}"]}
        else:
            vlm_critique = {"ok": deterministic["ok"], "failure_modes": [], "repair_actions": [], "notes": ["No overlay for VLM critic."]}

        failure_modes = list(dict.fromkeys((vlm_critique.get("failure_modes") or []) + deterministic["failure_modes"]))
        repair_actions = list(vlm_critique.get("repair_actions") or []) + deterministic["repair_actions"]
        return {
            "ok": bool(vlm_critique.get("ok", deterministic["ok"])) and not failure_modes,
            "failure_modes": failure_modes,
            "repair_actions": repair_actions,
            "notes": list(vlm_critique.get("notes") or []) + deterministic["notes"],
            "round": round_idx,
        }

    def record_round(self, round_idx: int, parts, quality: dict[str, Any], critique: dict[str, Any], overrides: dict[str, Any]) -> None:
        tool_actions = self._planned_tool_actions(round_idx, overrides)
        self.state["rounds"].append(
            {
                "round": round_idx,
                "tool_actions": tool_actions,
                "overrides": overrides,
                "quality": quality,
                "critique": critique,
                "parts": [_part_dict(p) for p in parts],
            }
        )
        self.state["action_trace"].append(
            {
                "stage": "canonical",
                "round": round_idx,
                "event": "execute_and_observe",
                "actions": tool_actions,
                "quality": {
                    "ok": bool(quality.get("ok", False)),
                    "reason": quality.get("reason"),
                    "coverage": quality.get("coverage"),
                    "unknown_ratio": quality.get("unknown_ratio"),
                    "overlap_ratio": quality.get("overlap_ratio"),
                    "missing_parts": quality.get("missing_parts", []),
                },
                "accepted": bool(critique.get("ok", quality.get("ok", False))),
            }
        )
        self.state["final"] = {
            "round": round_idx,
            "accepted": bool(critique.get("ok", quality.get("ok", False))),
            "selected_parts": [getattr(p, "name", "") for p in parts],
        }
        self.save()

    def should_retry(self, critique: dict[str, Any], quality: dict[str, Any], round_idx: int) -> bool:
        if not self.enabled:
            return False
        if round_idx + 1 >= self.max_rounds:
            return False
        if bool(critique.get("ok", False)) and bool(quality.get("ok", False)):
            return False
        return bool(critique.get("repair_actions") or critique.get("failure_modes"))

    def _build_multiview_overlay(self, parts) -> str | None:
        if len(self.view_paths) < 2:
            return None
        images = []
        labels = []
        for label, path in zip(VIEW_LABELS, self.view_paths):
            p = Path(path)
            if not p.exists():
                continue
            image = load_rgb(p)
            masks = []
            mask_labels = []
            for part in parts:
                metadata = getattr(part, "metadata", {}) or {}
                mask_path = (metadata.get("view_masks") or {}).get(label)
                if not mask_path or not Path(mask_path).exists():
                    continue
                try:
                    masks.append(read_mask(mask_path))
                    mask_labels.append(_part_name(part))
                except Exception:
                    continue
            if masks:
                image = overlay_multiple_masks(image, masks, labels=mask_labels)
            images.append(image)
            labels.append(label)
        if len(images) < 2:
            return None
        return self._make_grid(images, labels, self.log_dir / "multiview_part_overlay.png")

    def _multiview_quality(self, summary: dict[str, Any], parts) -> dict[str, Any]:
        if not summary or not summary.get("enabled"):
            return {"ok": True, "reason": "multiview_disabled", "view_coverage": None, "missing_parts": []}
        view_count = max(1, len(summary.get("views") or []))
        semantic_parts = [part for part in parts if not _is_residual_part(part)]
        part_count = max(1, len(semantic_parts))
        missing_by_view = {}
        for view in summary.get("views") or []:
            missing = [name for name in (view.get("missing_parts") or []) if "unknown" not in str(name).lower() and "residual" not in str(name).lower()]
            if missing:
                missing_by_view[view.get("label", "view")] = missing
        support = {}
        for part in semantic_parts:
            view_masks = (getattr(part, "metadata", {}) or {}).get("view_masks") or {}
            support[_part_name(part)] = len([p for p in view_masks.values() if p and Path(p).exists()])
        view_coverage = float(sum(support.values()) / max(1, part_count * view_count))
        missing_parts = [name for name, count in support.items() if count == 0]
        weak_parts = [name for name, count in support.items() if count / view_count < 0.5]
        ok = not missing_parts and view_coverage >= 0.5
        reason = "ok" if ok else "insufficient_multiview_part_support"
        return {
            "ok": ok,
            "reason": reason,
            "view_coverage": view_coverage,
            "view_count": view_count,
            "part_view_support": support,
            "missing_parts": missing_parts,
            "weak_multiview_parts": weak_parts,
            "missing_by_view": missing_by_view,
            "warnings": summary.get("warnings", []),
        }

    def record_multiview_summary(self, summary: dict[str, Any] | None, parts) -> dict[str, Any]:
        if not self.enabled:
            return {}
        summary = summary or {"enabled": False}
        quality = self._multiview_quality(summary, parts)
        overlay_path = self._build_multiview_overlay(parts)
        critique = self._deterministic_critique(quality)
        if overlay_path:
            try:
                vlm_critique = self.vlm_client.critique_part_segmentation(overlay_path, parts, quality)
                if isinstance(vlm_critique, dict):
                    failure_modes = list(dict.fromkeys((vlm_critique.get("failure_modes") or []) + critique["failure_modes"]))
                    critique = {
                        "ok": bool(vlm_critique.get("ok", critique["ok"])) and not failure_modes,
                        "failure_modes": failure_modes,
                        "repair_actions": list(vlm_critique.get("repair_actions") or []) + critique["repair_actions"],
                        "notes": list(vlm_critique.get("notes") or []) + critique["notes"],
                    }
            except Exception as exc:
                critique["notes"].append(f"VLM multiview critic failed: {exc}")
        self.state["multiview"] = {
            "summary": summary,
            "quality": quality,
            "critique": critique,
            "overlay_path": overlay_path,
        }
        self.state["action_trace"].append(
            {
                "stage": "multiview",
                "event": "align_and_verify",
                "actions": [
                    {"action": "multiview_align", "source": "default_executor"},
                    {"action": "compile_layout", "source": "default_executor"},
                ],
                "quality": {
                    "ok": bool(quality.get("ok", True)),
                    "reason": quality.get("reason"),
                    "view_coverage": quality.get("view_coverage"),
                    "missing_parts": quality.get("missing_parts", []),
                    "weak_multiview_parts": quality.get("weak_multiview_parts", []),
                },
                "accepted": bool(critique.get("ok", quality.get("ok", True))),
            }
        )
        self.state.setdefault("final", {})["multiview_ok"] = bool(critique.get("ok", quality.get("ok", True)))
        self.save()
        return critique
