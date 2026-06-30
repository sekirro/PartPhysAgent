from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .detectors import GroundingDINODetector, HuggingFaceGroundingDINODetector, NoDetector
from .gaussian_assign import (
    assign_by_aabb_heuristic,
    assign_by_projection,
    build_part_aabbs,
    load_ply_positions,
    save_assignment_outputs,
)
from .image_utils import (
    apply_mask_white_bg,
    bbox_expand,
    crop_image,
    crop_mask,
    load_rgb,
    mask_area,
    mask_to_bbox,
    overlay_mask,
    overlay_multiple_masks,
    read_mask,
    resize_to_square_with_padding,
    save_mask,
    save_rgb,
)
from .material_table import (
    clamp_physics_to_material,
    default_E_for_material,
    default_nu_for_material,
    density_for_material,
    normalize_material_name,
)
from .multi_object_physgm import run_multi_object_physgm
from .object_separation import separate_scene_objects
from .physgm_runner import PhysGMRunner, _find_physgm_root, _resolve_path
from .part_seg_agent import PartSegAgentController
from .proposals import generate_object_mask, generate_part_candidates
from .report import write_json, write_warnings
from .sam_tool import create_sam_tool
from .scene_builder import VIEW_LABELS, _candidate_multiview_paths
from .selector import select_physical_parts
from .sim_config_builder import build_part_aware_sim_config
from .segmentation_agent import SegmentationAgent
from .types import BBox, ObjectInstance, PartInstance, PartPhysResult, PartSpec, PhysGMResult, PhysicsParams
from .vlm import NoVLMClient, OpenAICompatibleVLMClient, normalize_part_schema, part_specs_from_schema


def _get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "part"


def _part_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _is_residual_part(part: PartInstance) -> bool:
    name = str(getattr(part, "name", "")).lower()
    group = str(getattr(part, "physics_group", "") or (getattr(part, "metadata", {}) or {}).get("physics_group", "")).lower()
    return name in {"unknown_body", "unknown", "residual", "residual_body"} or "residual" in name or group in {"global_body", "unknown", "residual"}


def _write_json(path, data):
    write_json(path, data)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_manual_path(path: str, base_dir: Path) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def weighted_median(values, weights=None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("weighted_median requires at least one valid value.")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() * 0.5
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _weighted_std(values, weights) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        return None
    values = values[valid]
    weights = weights[valid]
    mean = float(np.average(values, weights=weights))
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def aggregate_physics_outputs(outputs: list[dict[str, Any]], expected_materials=None, part_confidence: float = 1.0) -> PhysicsParams:
    expected = [normalize_material_name(x) for x in (expected_materials or [])]
    if not outputs:
        material = expected[0] if expected else "Plastic"
        return PhysicsParams(
            material=material,
            material_confidence=0.0,
            E=default_E_for_material(material),
            nu=default_nu_for_material(material),
            density=density_for_material(material),
            confidence=0.0,
            source_outputs=[],
            warnings=["No PhysGM crop outputs; used material defaults."],
        )
    weights_by_variant = {"tight": 0.8, "padded": 1.0, "context_dim": 1.2, "isolated_full": 0.9}
    material_votes: dict[str, float] = {}
    logE_values = []
    nu_values = []
    densities = []
    weights = []
    warnings: list[str] = []
    for item in outputs:
        weight = float(weights_by_variant.get(item.get("variant"), 1.0))
        material = normalize_material_name(item.get("material"))
        material_votes[material] = material_votes.get(material, 0.0) + weight
        E = float(item.get("E", 0.0) or 0.0)
        nu = float(item.get("nu", 0.0) or 0.0)
        if E > 0 and math.isfinite(E):
            logE_values.append(math.log10(E))
            nu_values.append(max(0.01, min(0.49, nu)))
            weights.append(weight)
        density = item.get("density")
        if density is not None and float(density) > 0:
            densities.append(float(density))
    material = max(material_votes, key=material_votes.get)
    material_conf = float(material_votes[material] / max(1e-9, sum(material_votes.values())))
    if logE_values:
        E = float(10 ** weighted_median(logE_values, weights))
        nu = float(weighted_median(nu_values, weights))
        logE_std = _weighted_std(logE_values, weights)
        nu_std = _weighted_std(nu_values, weights)
    else:
        E = default_E_for_material(material)
        nu = default_nu_for_material(material)
        logE_std = None
        nu_std = None
        warnings.append("No valid E/nu values; used material defaults.")
    density = float(np.median(densities)) if densities else density_for_material(material)
    E, nu, clamp_warnings = clamp_physics_to_material(material, E, nu)
    warnings.extend(clamp_warnings)
    if expected:
        prior_agreement = 1.0 if material in expected else 0.5
    else:
        prior_agreement = 0.8
    consistency = 1.0
    if logE_std is not None:
        consistency = max(0.2, 1.0 - min(logE_std / 2.0, 0.8))
    confidence = float(max(0.0, min(1.0, part_confidence * consistency * prior_agreement * material_conf)))
    return PhysicsParams(
        material=material,
        material_confidence=material_conf,
        E=E,
        nu=nu,
        density=density,
        logE_std=logE_std,
        nu_std=nu_std,
        confidence=confidence,
        source_outputs=outputs,
        warnings=warnings,
    )


def build_part_crops(
    image_path,
    object_mask_path,
    part_mask_path,
    part_output_dir,
    pad_ratios=(0.10, 0.30),
) -> dict[str, str]:
    part_output_dir = Path(part_output_dir)
    part_output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    w, h = image.size
    obj_mask = read_mask(object_mask_path)
    part_mask = read_mask(part_mask_path) & obj_mask
    part_bbox = mask_to_bbox(part_mask)
    obj_bbox = mask_to_bbox(obj_mask)
    if part_bbox.is_empty:
        part_bbox = obj_bbox if not obj_bbox.is_empty else BBox(0, 0, w, h)
    if obj_bbox.is_empty:
        obj_bbox = BBox(0, 0, w, h)

    arr = np.asarray(image.convert("RGB"))
    part_only = np.full_like(arr, 255)
    part_only[part_mask] = arr[part_mask]
    part_only_img = Image.fromarray(part_only, mode="RGB")

    tight = crop_image(part_only_img, part_bbox)
    padded_bbox = bbox_expand(part_bbox, pad_ratios[-1], w, h)
    padded = crop_image(part_only_img, padded_bbox)

    context = np.full_like(arr, 255)
    context[obj_mask] = arr[obj_mask]
    non_part = obj_mask & ~part_mask
    context[non_part] = (context[non_part].astype(np.float32) * 0.35 + 255.0 * 0.65).astype(np.uint8)
    context_dim = crop_image(Image.fromarray(context, mode="RGB"), obj_bbox)

    isolated = np.full_like(arr, 255)
    isolated[part_mask] = arr[part_mask]
    isolated_full = crop_image(Image.fromarray(isolated, mode="RGB"), obj_bbox)

    crops = {
        "tight": resize_to_square_with_padding(tight, 512),
        "padded": resize_to_square_with_padding(padded, 512),
        "context_dim": resize_to_square_with_padding(context_dim, 512),
        "isolated_full": resize_to_square_with_padding(isolated_full, 512),
    }
    out = {}
    file_names = {
        "tight": "crop_tight.png",
        "padded": "crop_padded.png",
        "context_dim": "crop_context_dim.png",
        "isolated_full": "crop_isolated_full.png",
    }
    for name, crop in crops.items():
        path = part_output_dir / file_names[name]
        save_rgb(crop, path)
        out[name] = str(path)
    return out


def infer_part_physics(
    part_instance: PartInstance,
    crop_paths: dict[str, str],
    physgm_runner: PhysGMRunner,
    material_prior,
    output_dir,
) -> PhysicsParams:
    output_dir = Path(output_dir)
    outputs = []
    warnings: list[str] = []
    for run_idx, (variant, crop_path) in enumerate(crop_paths.items()):
        run_dir = output_dir / "physgm_outputs" / f"run_{run_idx:02d}_{variant}"
        try:
            result = physgm_runner.infer_image(
                crop_path,
                scene_name=f"part_{part_instance.part_id:03d}_{_safe_name(part_instance.name)}_{variant}",
                output_dir=run_dir,
                save_gaussian=False,
            )
            item = dict(result.raw)
            item.update(
                {
                    "variant": variant,
                    "material": result.material,
                    "E": result.E,
                    "nu": result.nu,
                    "density": result.density,
                    "predicted_phys_path": result.predicted_phys_path,
                }
            )
            outputs.append(item)
        except Exception as exc:
            warnings.append(f"Part {part_instance.name} crop {variant} PhysGM failed: {exc}")
    params = aggregate_physics_outputs(outputs, part_instance.expected_materials, part_instance.confidence)
    params.warnings.extend(warnings)
    summary_path = output_dir / "part_summary.json"
    existing = {}
    if summary_path.exists():
        try:
            existing = _load_json(summary_path)
        except Exception:
            existing = {}
    existing["part"] = part_instance.to_dict()
    existing["aggregated_physics"] = params.to_dict()
    existing["material_prior"] = material_prior or {}
    _write_json(summary_path, existing)
    return params


def run_simulation(
    repo_root,
    model_path,
    output_path,
    config_path,
    render_img: bool = True,
    compile_video: bool = True,
    white_bg: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "gs_simulation.py", "--model_path", str(model_path), "--output_path", str(output_path), "--config", str(config_path)]
    if render_img:
        cmd.append("--render_img")
    if compile_video:
        cmd.append("--compile_video")
    if white_bg:
        cmd.append("--white_bg")
    command_text = " ".join(cmd)
    (output_path / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output_path / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output_path / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return {"returncode": proc.returncode, "command": command_text, "stdout": str(output_path / "stdout.txt"), "stderr": str(output_path / "stderr.txt")}


def _part_object_id(part: PartInstance) -> int:
    try:
        return int((part.metadata or {}).get("object_id", 0))
    except Exception:
        return 0


def _write_ascii_object_ply(path: Path, positions: np.ndarray, indices: np.ndarray) -> None:
    pts = np.asarray(positions)[indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def save_object_assignment_outputs(
    output_dir,
    gaussian_part_ids,
    parts: list[PartInstance],
    objects: list[ObjectInstance],
    positions=None,
    object_ids_override=None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ids = np.asarray(gaussian_part_ids, dtype=np.int32)
    part_to_object = {int(part.part_id): _part_object_id(part) for part in parts}
    object_by_id = {int(obj.object_id): obj for obj in objects}
    if object_ids_override is not None and len(object_ids_override) == len(ids):
        object_ids = np.asarray(object_ids_override, dtype=np.int32)
    else:
        object_ids = np.full(len(ids), -1, dtype=np.int32)
        for part_id, object_id in part_to_object.items():
            object_ids[ids == int(part_id)] = int(object_id)
    np.save(output_dir / "gaussian_object_ids.npy", object_ids)

    per_object_dir = output_dir / "per_object_gaussians"
    per_object_dir.mkdir(parents=True, exist_ok=True)
    index = {"objects": [], "unassigned_count": int((object_ids < 0).sum())}
    positions_arr = np.asarray(positions) if positions is not None else None
    indexed_object_ids = set(int(x) for x in np.unique(object_ids[object_ids >= 0]))
    indexed_object_ids.update(int(obj.object_id) for obj in objects)
    for object_id in sorted(indexed_object_ids):
        obj = object_by_id.get(object_id)
        name = obj.name if obj is not None else f"object_{object_id}"
        indices = np.where(object_ids == object_id)[0].astype(np.int64)
        stem = f"object_{object_id:03d}_{_safe_name(name)}"
        indices_path = output_dir / f"{stem}_indices.npy"
        np.save(indices_path, indices)
        ply_path = per_object_dir / f"{stem}.ply"
        ply_value = None
        if positions_arr is not None and len(indices) > 0:
            _write_ascii_object_ply(ply_path, positions_arr, indices)
            ply_value = str(ply_path)
        if obj is not None:
            obj.metadata["gaussian_count"] = int(len(indices))
        index["objects"].append(
            {
                "object_id": object_id,
                "object_name": name,
                "count": int(len(indices)),
                "part_ids": [int(pid) for pid, oid in part_to_object.items() if int(oid) == object_id],
                "indices_path": str(indices_path),
                "ply_path": ply_value,
            }
        )
    with (output_dir / "object_gaussian_index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return {
        "gaussian_object_ids": str(output_dir / "gaussian_object_ids.npy"),
        "object_gaussian_index": str(output_dir / "object_gaussian_index.json"),
        "per_object_gaussians_dir": str(per_object_dir),
        "per_object_gaussian_counts": {str(item["object_id"]): int(item["count"]) for item in index["objects"]},
    }


def _part_view_mask_path(part: PartInstance, label: str) -> str | None:
    if label == "front":
        return part.mask_path
    view_masks = (part.metadata or {}).get("view_masks") or {}
    return view_masks.get(label)


def _crop_mask_like_object_input(mask_path: str, crop_row: dict[str, Any], output_path: Path) -> str:
    source = Image.open(mask_path).convert("L")
    source_w = int(crop_row.get("source_width", source.width) or source.width)
    source_h = int(crop_row.get("source_height", source.height) or source.height)
    if source.size != (source_w, source_h):
        source = source.resize((source_w, source_h), Image.Resampling.NEAREST)
    output_size = int(crop_row.get("output_size", 512) or 512)
    if "crop_x0" in crop_row and "crop_y0" in crop_row and "crop_size" in crop_row:
        x0 = int(math.floor(float(crop_row["crop_x0"])))
        y0 = int(math.floor(float(crop_row["crop_y0"])))
        size = int(math.ceil(float(crop_row["crop_size"])))
        canvas = np.zeros((size, size), dtype=np.uint8)
        src = np.asarray(source)
        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(source.width, x0 + size)
        sy1 = min(source.height, y0 + size)
        if sx1 > sx0 and sy1 > sy0:
            dx0 = sx0 - x0
            dy0 = sy0 - y0
            canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = src[sy0:sy1, sx0:sx1]
        out = Image.fromarray(canvas, mode="L").resize((output_size, output_size), Image.Resampling.NEAREST)
    else:
        out = source.resize((output_size, output_size), Image.Resampling.NEAREST)
    arr = (np.asarray(out) > 0).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output_path)
    return str(output_path)


def assign_per_object_projection(
    positions: np.ndarray,
    parts: list[PartInstance],
    direct_object_ids: np.ndarray | None,
    whole_result: PhysGMResult,
    output_dir: Path,
) -> dict[str, Any] | None:
    raw = whole_result.raw or {}
    if raw.get("geometry_source") != "per_object_physgm":
        return None
    object_results = raw.get("object_results") or []
    if direct_object_ids is None or len(direct_object_ids) != len(positions) or not object_results:
        return None
    output_dir = Path(output_dir)
    ids = np.full(len(positions), -1, dtype=np.int32)
    object_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for object_row in object_results:
        try:
            object_id = int(object_row["object_id"])
        except Exception:
            continue
        object_indices = np.where(np.asarray(direct_object_ids, dtype=np.int32) == object_id)[0]
        if len(object_indices) == 0:
            continue
        object_parts = [part for part in parts if _part_object_id(part) == object_id]
        if not object_parts:
            warnings.append(f"Object {object_id} has no local parts for projection assignment.")
            continue
        input_dir = Path(object_row.get("input_dir", ""))
        physgm_dir = Path(object_row.get("physgm_dir", ""))
        metadata_path = input_dir / "object_input_metadata.json"
        camera_meta = physgm_dir / "input_batch_meta.npz"
        if not metadata_path.exists() or not camera_meta.exists():
            warnings.append(f"Object {object_id} missing crop metadata or camera metadata.")
            continue
        try:
            input_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Object {object_id} failed to read crop metadata: {exc}")
            continue
        view_rows = input_metadata.get("view_rows") or []
        local_part_masks = []
        local_mask_dir = output_dir / "object_local_part_masks" / f"object_{object_id:03d}_{_safe_name(object_row.get('object_name', 'object'))}"
        for part in object_parts:
            view_masks: dict[str, str] = {}
            for label, crop_row in zip(VIEW_LABELS, view_rows):
                source_mask = _part_view_mask_path(part, label)
                if not source_mask:
                    continue
                try:
                    view_masks[label] = _crop_mask_like_object_input(
                        source_mask,
                        crop_row,
                        local_mask_dir / f"part_{int(part.part_id):03d}_{_safe_name(part.name)}_{label}.png",
                    )
                except Exception as exc:
                    warnings.append(f"Object {object_id} part {part.name} {label} mask crop failed: {exc}")
            mask_path = view_masks.get("front") or part.mask_path
            local_part_masks.append(
                {
                    "part_id": part.part_id,
                    "name": part.name,
                    "mask_path": mask_path,
                    "view_masks": view_masks,
                    "area": part.area,
                    "confidence": part.confidence,
                    "physics_group": part.physics_group,
                }
            )
        local_assign = assign_by_projection(positions[object_indices], local_part_masks, camera_meta, (512, 512))
        local_ids = np.asarray(local_assign.get("gaussian_part_ids"), dtype=np.int32)
        if len(local_ids) == len(object_indices):
            ids[object_indices] = local_ids
        object_summaries.append(
            {
                "object_id": object_id,
                "object_name": object_row.get("object_name"),
                "gaussian_count": int(len(object_indices)),
                "assigned_ratio": float(local_assign.get("assigned_ratio", 0.0)),
                "per_part_counts": local_assign.get("per_part_counts", {}),
                "projection_views": local_assign.get("view_labels", []),
                "projection_view_hits": local_assign.get("per_view_hits", {}),
                "warnings": local_assign.get("warnings", []),
            }
        )
        warnings.extend([f"object {object_id}: {item}" for item in local_assign.get("warnings", [])])
    assigned = ids >= 0
    counts = {str(pid): int((ids == int(pid)).sum()) for pid in np.unique(ids[assigned])}
    return {
        "gaussian_part_ids": ids,
        "assigned_ratio": float(assigned.mean()) if len(ids) else 0.0,
        "per_part_counts": counts,
        "warnings": warnings,
        "object_local_projection": object_summaries,
        "view_labels": [],
        "per_view_hits": {},
        "projection_image_size": [512, 512],
        "mean_view_support": 0.0,
        "view_support_counts": {},
        "margin_ratio": {},
        "low_confidence_count": 0,
        "smoothed_count": 0,
        "knn_unknown_reassigned_count": 0,
        "knn_island_reassigned_count": 0,
    }


class PartPhysAgent:
    def __init__(self, config):
        self.config = config
        self.warnings: list[str] = []
        self.physgm_root = _find_physgm_root(_get(config, "physgm_root"))

    def _resolve_physgm_path(self, path: str | None) -> str | None:
        return _resolve_path(path, self.physgm_root)

    def _init_vlm(self, required: bool = False):
        provider = "none" if _get(self.config, "no_vlm", False) else _get(self.config, "vlm_provider", "none")
        if provider == "openai_compatible":
            return OpenAICompatibleVLMClient(
                model=_get(self.config, "vlm_model"),
                api_base=_get(self.config, "vlm_api_base"),
                api_key_env=_get(self.config, "vlm_api_key_env", "OPENAI_API_KEY"),
                timeout=int(_get(self.config, "vlm_timeout", 180)),
                required=required,
            )
        if required:
            raise RuntimeError("Automatic segmentation requires --vlm-provider openai_compatible.")
        return NoVLMClient()

    def _init_detector(self):
        model_id = _get(self.config, "groundingdino_model")
        box_threshold = float(_get(self.config, "groundingdino_box_threshold", 0.25))
        text_threshold = float(_get(self.config, "groundingdino_text_threshold", 0.25))
        if model_id:
            detector = HuggingFaceGroundingDINODetector(
                model_id,
                device=_get(self.config, "device", "cuda"),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if detector.warning:
                self.warnings.append(detector.warning)
            return detector
        config = _get(self.config, "groundingdino_config")
        weights = _get(self.config, "groundingdino_weights")
        if config and weights:
            detector = GroundingDINODetector(
                config,
                weights,
                device=_get(self.config, "device", "cuda"),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if detector.warning:
                self.warnings.append(detector.warning)
            return detector
        return NoDetector()

    def _init_sam(self):
        checkpoint = _get(self.config, "sam_checkpoint")
        if not checkpoint:
            return None
        try:
            return create_sam_tool(
                checkpoint,
                backend=_get(self.config, "sam_backend", "sam2"),
                config=_get(self.config, "sam_config"),
                device=_get(self.config, "device", "cuda"),
                sam2_root=_get(self.config, "sam2_root"),
                model_type=_get(self.config, "sam_model_type", "vit_b"),
                points_per_side=int(_get(self.config, "sam_points_per_side", 16)),
                pred_iou_thresh=float(_get(self.config, "sam_pred_iou_thresh", 0.88)),
                stability_score_thresh=float(_get(self.config, "sam_stability_score_thresh", 0.92)),
                crop_n_layers=int(_get(self.config, "sam_crop_n_layers", 0)),
                min_mask_region_area=int(_get(self.config, "sam_min_mask_region_area", 100)),
            )
        except RuntimeError as exc:
            self.warnings.append(str(exc))
            return None

    def _load_manual_masks(self, masks_json: str | None, image_size) -> dict[str, Any] | None:
        if not masks_json:
            return None
        path = Path(masks_json).expanduser().resolve()
        data = _load_json(path)
        base = path.parent
        if data.get("object_mask"):
            data["object_mask"] = _resolve_manual_path(data["object_mask"], base)
        for part in data.get("parts", []):
            part["mask"] = _resolve_manual_path(part["mask"], base)
        for mask_path in [data.get("object_mask"), *[p.get("mask") for p in data.get("parts", [])]]:
            if mask_path:
                mask = read_mask(mask_path)
                if mask.shape[::-1] != tuple(image_size):
                    raise ValueError(f"Mask size {mask.shape[::-1]} does not match input image {image_size}: {mask_path}")
        return data

    def _manual_part_instances(self, manual, object_mask_path, scene_dir, image_path, schema) -> list[PartInstance]:
        image = load_rgb(image_path)
        parts = []
        specs_by_name = {p.name: p for p in part_specs_from_schema(schema)}
        for idx, raw in enumerate(manual.get("parts", [])):
            name = raw.get("name") or f"part_{idx}"
            spec = specs_by_name.get(name) or PartSpec(
                name=name,
                expected_materials=raw.get("expected_materials") or ["Plastic"],
                physics_group=raw.get("physics_group") or name,
            )
            mask = read_mask(raw["mask"])
            part_dir = scene_dir / "parts" / f"part_{idx:03d}_{_safe_name(name)}"
            part_dir.mkdir(parents=True, exist_ok=True)
            mask_path = part_dir / "mask.png"
            save_mask(mask, mask_path)
            save_rgb(overlay_mask(image, mask), part_dir / "overlay.png")
            inst = PartInstance(
                part_id=idx,
                name=name,
                mask_path=str(mask_path),
                bbox=mask_to_bbox(mask),
                area=mask_area(mask),
                confidence=float(raw.get("confidence", 1.0)),
                candidate_ids=["manual"],
                expected_materials=[normalize_material_name(x) for x in raw.get("expected_materials", spec.expected_materials)],
                physics_group=raw.get("physics_group") or spec.physics_group or name,
                warnings=[],
                metadata={"part_spec": spec.to_dict(), "manual": True},
            )
            _write_json(part_dir / "part_summary.json", {"part": inst.to_dict()})
            parts.append(inst)
        if not parts:
            mask = read_mask(object_mask_path)
            spec = PartSpec(name="body", expected_materials=["Plastic"], physics_group="global_body")
            part_dir = scene_dir / "parts" / "part_000_body"
            part_dir.mkdir(parents=True, exist_ok=True)
            mask_path = part_dir / "mask.png"
            save_mask(mask, mask_path)
            save_rgb(overlay_mask(image, mask), part_dir / "overlay.png")
            parts.append(
                PartInstance(0, "body", str(mask_path), mask_to_bbox(mask), mask_area(mask), 0.5, ["object_mask"], ["Plastic"], "global_body", [], {"part_spec": spec.to_dict()})
            )
        return parts

    def _schema_from_manual_or_file(self, manual, object_name: str) -> dict[str, Any]:
        schema_json = _get(self.config, "part_schema_json")
        if schema_json:
            return normalize_part_schema(_load_json(schema_json), object_name)
        if manual and manual.get("parts"):
            return normalize_part_schema(
                {
                    "object": object_name,
                    "parts": [
                        {
                            "name": p.get("name") or f"part_{i}",
                            "text_prompts": [p.get("name") or f"part_{i}"],
                            "expected_materials": p.get("expected_materials") or ["Plastic"],
                            "location": p.get("location", ""),
                            "shape_prior": p.get("shape_prior", ""),
                            "physical_role": p.get("physical_role", ""),
                            "should_simulate_separately": True,
                            "visible": True,
                            "physics_group": p.get("physics_group") or p.get("name") or f"part_{i}",
                        }
                        for i, p in enumerate(manual["parts"])
                    ],
                    "relations": [],
                },
                object_name,
            )
        return {}

    def _save_object_inputs(self, image_path, object_mask_path, object_bbox, input_dir):
        input_dir = Path(input_dir)
        image = load_rgb(image_path)
        mask = read_mask(object_mask_path)
        save_rgb(image, input_dir / "input.png")
        bbox = object_bbox if isinstance(object_bbox, BBox) else mask_to_bbox(mask)
        if bbox.is_empty:
            bbox = BBox(0, 0, image.size[0], image.size[1])
        crop = crop_image(image, bbox)
        save_rgb(crop, input_dir / "object_crop.png")
        white_full = apply_mask_white_bg(image, mask)
        save_rgb(crop_image(white_full, bbox), input_dir / "object_crop_white_bg.png")
        save_rgb(white_full, input_dir / "object_isolated_full.png")
        return str(input_dir / "object_isolated_full.png")

    def _run_segmentation_agent(
        self,
        image_path,
        object_mask_path,
        object_bbox,
        schema,
        detector,
        sam,
        vlm,
        output_dir,
        candidates_dir,
        overrides: dict[str, Any] | None = None,
    ):
        overrides = overrides or {}

        def cfg(key: str, default=None):
            return overrides[key] if key in overrides else _get(self.config, key, default)

        return SegmentationAgent(
            image_path=image_path,
            object_mask_path=object_mask_path,
            object_bbox=object_bbox,
            part_schema=schema,
            detector=detector,
            sam_tool=sam,
            vlm_client=vlm,
            output_dir=output_dir,
            candidates_dir=candidates_dir,
            max_parts=int(cfg("max_parts", 6)),
            min_part_area_ratio=float(cfg("min_part_area_ratio", 0.01)),
            coverage_threshold=float(cfg("coverage_threshold", 0.75)),
            max_retries=int(cfg("segmentation_max_retries", 2)),
            vlm_weight=float(cfg("segmentation_vlm_weight", 0.55)),
            min_accept_score=float(cfg("segmentation_min_accept_score", 0.45)),
            max_vlm_candidates_per_part=int(cfg("max_vlm_candidates_per_part", 12)),
            segmentation_mode=cfg("segmentation_mode", "candidate_pool"),
            use_vlm_bbox_proposals=bool(cfg("use_vlm_bbox_proposals", False)),
            use_schema_location_proposals=bool(cfg("use_schema_location_proposals", False)),
            strict_segmentation=bool(cfg("strict_segmentation", False)),
            residual_policy=cfg("residual_policy", "unknown"),
            candidate_top_k=int(cfg("candidate_top_k", 40)),
            candidate_contact_sheet_top_k=int(cfg("candidate_contact_sheet_top_k", 24)),
        ).run()

    def _refresh_part_summaries(self, scene_dir: Path, parts: list[PartInstance]) -> None:
        for part in parts:
            _write_json(Path(part.mask_path).parent / "part_summary.json", {"part": part.to_dict()})
        summary_path = scene_dir / "parts" / "selection_summary.json"
        try:
            summary = _load_json(summary_path) if summary_path.exists() else {}
        except Exception:
            summary = {}
        summary["parts"] = [p.to_dict() for p in parts]
        _write_json(summary_path, summary)

    def _attach_multiview_part_masks(
        self,
        parts: list[PartInstance],
        schema: dict[str, Any],
        image_path: str,
        object_name: str,
        detector,
        sam,
        vlm,
        scene_dir: Path,
        part_agent=None,
    ) -> dict[str, Any]:
        multiview_dir = _get(self.config, "multiview_dir")
        view_paths = _candidate_multiview_paths(multiview_dir)
        if not view_paths:
            return {"enabled": False, "reason": "no_multiview_dir"}

        for part in parts:
            part.metadata.setdefault("view_masks", {})
            part.metadata.setdefault("view_part_summaries", {})

        parts_by_name = {_part_key(part.name): part for part in parts}
        summary: dict[str, Any] = {
            "enabled": True,
            "multiview_dir": str(multiview_dir),
            "view_labels": list(VIEW_LABELS),
            "views": [],
            "alignment": "part_schema_name",
            "warnings": [],
        }
        canonical_path = Path(image_path).expanduser().resolve()

        for label, view_path in zip(VIEW_LABELS, view_paths):
            view_path = Path(view_path).expanduser().resolve()
            if view_path == canonical_path:
                for part in parts:
                    part.metadata["view_masks"][label] = part.mask_path
                    part.metadata["view_part_summaries"][label] = {
                        "source": "canonical",
                        "area": part.area,
                        "bbox": part.bbox.to_dict(),
                        "confidence": part.confidence,
                    }
                summary["views"].append({"label": label, "image_path": str(view_path), "source": "canonical", "matched_parts": [p.name for p in parts]})
                continue

            view_object_dir = scene_dir / "multiview_object" / label
            view_candidates_dir = scene_dir / "multiview_candidates" / label
            view_output_dir = scene_dir / "multiview_parts" / label
            try:
                view_object_mask_path, view_object_bbox, object_warnings = generate_object_mask(
                    view_path,
                    object_name,
                    detector,
                    sam,
                    view_object_dir,
                    fallback_to_full_image=bool(_get(self.config, "fallback_to_full_image", True)),
                    keep_multi_components=str(_get(self.config, "object_mode", "single")).lower() == "auto",
                )
                for warning in object_warnings:
                    summary["warnings"].append(f"{label}: {warning}")
                    self.warnings.append(f"{label}: {warning}")
            except Exception as exc:
                warning = f"Multiview segmentation failed for {label}: {exc}"
                summary["warnings"].append(warning)
                self.warnings.append(warning)
                continue

            view_parts = []
            view_quality = {}
            view_critique = None
            view_round_records = []
            view_rounds = part_agent.max_rounds if part_agent and getattr(part_agent, "enabled", False) else 1
            final_view_output_dir = view_output_dir
            for round_idx in range(view_rounds):
                overrides = part_agent.round_overrides(view_critique, round_idx, {"stage": "multiview", "view": label}) if part_agent else {}
                round_output_dir = view_output_dir if round_idx == 0 else scene_dir / "agent_rounds" / "multiview" / label / f"round_{round_idx:02d}"
                round_candidates_dir = view_candidates_dir if round_idx == 0 else round_output_dir / "candidates"
                try:
                    view_parts, _, raw_quality = self._run_segmentation_agent(
                        image_path=str(view_path),
                        object_mask_path=view_object_mask_path,
                        object_bbox=view_object_bbox,
                        schema=schema,
                        detector=detector,
                        sam=sam,
                        vlm=vlm,
                        output_dir=round_output_dir,
                        candidates_dir=round_candidates_dir,
                        overrides=overrides,
                    )
                    final_view_output_dir = round_output_dir
                except Exception as exc:
                    view_quality = {"ok": False, "reason": f"view segmentation failed: {exc}", "missing_parts": [p.name for p in parts]}
                    view_critique = {"ok": False, "failure_modes": [view_quality["reason"]], "repair_actions": [{"action": "increase_candidate_pool", "target": label}], "notes": []}
                    view_round_records.append({"round": round_idx, "ok": False, "quality": view_quality, "critique": view_critique, "overrides": overrides})
                    break

                view_by_name = {_part_key(part.name): part for part in view_parts}
                missing = [part.name for part in parts if not _is_residual_part(part) and _part_key(part.name) not in view_by_name]
                view_quality = dict(raw_quality or {})
                if missing:
                    view_quality["ok"] = False
                    view_quality["reason"] = "missing_aligned_parts"
                    view_quality["missing_parts"] = sorted(set((view_quality.get("missing_parts") or []) + missing))
                overlay_path = round_output_dir / "parts" / "parts_overlay.png"
                view_critique = part_agent.critique(view_parts, view_quality, overlay_path, round_idx) if part_agent else {"ok": bool(view_quality.get("ok", True))}
                view_round_records.append(
                    {
                        "round": round_idx,
                        "output_dir": str(round_output_dir),
                        "candidates_dir": str(round_candidates_dir),
                        "overrides": overrides,
                        "quality": view_quality,
                        "critique": view_critique,
                    }
                )
                if not (part_agent and part_agent.should_retry(view_critique, view_quality, round_idx)):
                    break

            view_by_name = {_part_key(part.name): part for part in view_parts}
            matched = []
            missing = []
            for part in parts:
                view_part = view_by_name.get(_part_key(part.name))
                if view_part is None:
                    if _is_residual_part(part):
                        continue
                    missing.append(part.name)
                    continue
                part.metadata["view_masks"][label] = view_part.mask_path
                part.metadata["view_part_summaries"][label] = {
                    "source": "multiview_segmentation",
                    "area": view_part.area,
                    "bbox": view_part.bbox.to_dict(),
                    "confidence": view_part.confidence,
                    "candidate_ids": list(view_part.candidate_ids),
                }
                matched.append(part.name)
            if missing:
                warning = f"{label}: missing aligned masks for {', '.join(missing)}"
                summary["warnings"].append(warning)
                self.warnings.append(warning)
            if not view_quality.get("ok"):
                warning = f"{label}: multiview segmentation accepted with warnings: {view_quality.get('reason')}"
                summary["warnings"].append(warning)
                self.warnings.append(warning)
            summary["views"].append(
                {
                    "label": label,
                    "image_path": str(view_path),
                    "source": "multiview_segmentation",
                    "quality": view_quality,
                    "agent_rounds": view_round_records,
                    "matched_parts": matched,
                    "missing_parts": missing,
                    "view_parts_dir": str(final_view_output_dir / "parts"),
                }
            )

        _write_json(scene_dir / "agent_logs" / "multiview_segmentation_summary.json", summary)
        self._refresh_part_summaries(scene_dir, parts)
        return summary

    def _attach_object_multiview_part_masks(
        self,
        parts: list[PartInstance],
        schema: dict[str, Any],
        obj: ObjectInstance,
        image_path: str,
        object_name: str,
        detector,
        sam,
        vlm,
        object_work_dir: Path,
    ) -> dict[str, Any]:
        multiview_dir = _get(self.config, "multiview_dir")
        view_paths = _candidate_multiview_paths(multiview_dir)
        if not view_paths:
            for part in parts:
                part.metadata.setdefault("view_masks", {})["front"] = part.mask_path
            return {"enabled": False, "reason": "no_multiview_dir"}

        for part in parts:
            part.metadata.setdefault("view_masks", {})["front"] = part.mask_path
            part.metadata.setdefault("view_part_summaries", {})

        parts_by_name = {_part_key(part.name): part for part in parts}
        summary: dict[str, Any] = {
            "enabled": True,
            "object_id": int(obj.object_id),
            "object_name": object_name,
            "multiview_dir": str(multiview_dir),
            "view_labels": list(VIEW_LABELS),
            "views": [],
            "alignment": "object_local_part_schema_name",
            "warnings": [],
        }
        canonical_path = Path(image_path).expanduser().resolve()
        object_view_masks = (obj.metadata or {}).get("view_masks") or {}

        for label, view_path in zip(VIEW_LABELS, view_paths):
            view_path = Path(view_path).expanduser().resolve()
            if label == "front" or view_path == canonical_path:
                summary["views"].append({"label": label, "image_path": str(view_path), "source": "canonical", "matched_parts": [p.name for p in parts]})
                continue
            view_object_mask_path = object_view_masks.get(label)
            if not view_object_mask_path:
                warning = f"{label}: missing object view mask for object {obj.object_id}"
                summary["warnings"].append(warning)
                continue
            try:
                view_object_mask = read_mask(view_object_mask_path)
                view_object_bbox = mask_to_bbox(view_object_mask)
            except Exception as exc:
                warning = f"{label}: failed to read object view mask for object {obj.object_id}: {exc}"
                summary["warnings"].append(warning)
                continue

            view_output_dir = object_work_dir / "multiview_parts" / label
            view_candidates_dir = object_work_dir / "multiview_candidates" / label
            try:
                view_parts, _, view_quality = self._run_segmentation_agent(
                    image_path=str(view_path),
                    object_mask_path=view_object_mask_path,
                    object_bbox=view_object_bbox,
                    schema=schema,
                    detector=detector,
                    sam=sam,
                    vlm=vlm,
                    output_dir=view_output_dir,
                    candidates_dir=view_candidates_dir,
                    overrides={},
                )
            except Exception as exc:
                warning = f"{label}: object-local multiview part segmentation failed for object {obj.object_id}: {exc}"
                summary["warnings"].append(warning)
                self.warnings.append(warning)
                continue

            view_by_name = {_part_key(part.name): part for part in view_parts}
            matched = []
            missing = []
            for key, part in parts_by_name.items():
                view_part = view_by_name.get(key)
                if view_part is None:
                    if not _is_residual_part(part):
                        missing.append(part.name)
                    continue
                part.metadata["view_masks"][label] = view_part.mask_path
                part.metadata["view_part_summaries"][label] = {
                    "source": "object_local_multiview_segmentation",
                    "area": view_part.area,
                    "bbox": view_part.bbox.to_dict(),
                    "confidence": view_part.confidence,
                    "candidate_ids": list(view_part.candidate_ids),
                }
                matched.append(part.name)
            if missing:
                warning = f"{label}: missing object-local aligned parts for object {obj.object_id}: {', '.join(missing)}"
                summary["warnings"].append(warning)
                self.warnings.append(warning)
            if not (view_quality or {}).get("ok", True):
                warning = f"{label}: object-local multiview segmentation accepted with warnings: {(view_quality or {}).get('reason')}"
                summary["warnings"].append(warning)
                self.warnings.append(warning)
            summary["views"].append(
                {
                    "label": label,
                    "image_path": str(view_path),
                    "source": "object_local_multiview_segmentation",
                    "quality": view_quality,
                    "matched_parts": matched,
                    "missing_parts": missing,
                    "view_parts_dir": str(view_output_dir / "parts"),
                }
            )

        _write_json(object_work_dir / "multiview_part_summary.json", summary)
        return summary

    def _run_object_centric_part_segmentation(
        self,
        objects: list[ObjectInstance],
        image_path: str,
        object_name: str,
        detector,
        sam,
        vlm,
        scene_dir: Path,
    ) -> tuple[list[PartInstance], list[ObjectInstance], dict[str, Any]]:
        object_parts_root = scene_dir / "object_parts"
        object_parts_root.mkdir(parents=True, exist_ok=True)
        all_parts: list[PartInstance] = []
        summary: dict[str, Any] = {"enabled": True, "objects": [], "warnings": []}
        next_part_id = 0
        view_paths = _candidate_multiview_paths(_get(self.config, "multiview_dir"))

        for obj in objects:
            object_work_dir = object_parts_root / f"object_{int(obj.object_id):03d}_{_safe_name(obj.name)}"
            object_work_dir.mkdir(parents=True, exist_ok=True)
            object_input = self._save_object_inputs(image_path, obj.mask_path, obj.bbox, object_work_dir / "input")
            object_label = obj.name
            try:
                identified = vlm.identify_object(object_input).get("object")
                if identified and str(identified).strip().lower() not in {"object", "foreground"}:
                    object_label = str(identified).strip()
            except Exception as exc:
                summary["warnings"].append(f"Object {obj.object_id}: VLM object identification failed: {exc}")

            schema = self._schema_from_manual_or_file(None, object_label)
            if not schema:
                schema = vlm.generate_part_schema(object_input, object_label, obj.mask_path)
            schema = normalize_part_schema(schema, object_label)
            _write_json(object_work_dir / "schema" / "part_schema.json", schema)

            object_agent = PartSegAgentController(self.config, object_work_dir, vlm, object_label, object_input, view_paths)
            agent_plan = object_agent.plan(schema)
            if (
                object_agent.enabled
                and not _get(self.config, "part_schema_json")
                and isinstance(agent_plan.get("parts"), list)
                and agent_plan.get("parts")
            ):
                schema = normalize_part_schema(
                    {
                        "object": agent_plan.get("object") or object_label,
                        "parts": agent_plan.get("parts", []),
                        "relations": agent_plan.get("relations", []),
                    },
                    object_label,
                )
                _write_json(object_work_dir / "schema" / "part_schema.json", schema)
                _write_json(object_work_dir / "schema" / "agent_plan.json", agent_plan)

            object_parts: list[PartInstance] = []
            quality: dict[str, Any] = {}
            critique = None
            rounds = object_agent.max_rounds if object_agent.enabled else 1
            for round_idx in range(rounds):
                overrides = object_agent.round_overrides(critique, round_idx, {"stage": "object_part", "object_id": int(obj.object_id)})
                round_output_dir = object_work_dir if round_idx == 0 else object_work_dir / "agent_rounds" / f"round_{round_idx:02d}"
                round_candidates_dir = round_output_dir / "candidates"
                object_parts, _, quality = self._run_segmentation_agent(
                    image_path=image_path,
                    object_mask_path=obj.mask_path,
                    object_bbox=obj.bbox,
                    schema=schema,
                    detector=detector,
                    sam=sam,
                    vlm=vlm,
                    output_dir=round_output_dir,
                    candidates_dir=round_candidates_dir,
                    overrides=overrides,
                )
                overlay_path = round_output_dir / "parts" / "parts_overlay.png"
                critique = object_agent.critique(object_parts, quality, overlay_path, round_idx)
                object_agent.record_round(round_idx, object_parts, quality, critique, overrides)
                if not object_agent.should_retry(critique, quality, round_idx):
                    break
            if not object_parts:
                warning = f"Object {obj.object_id}: object-local part segmentation produced no parts."
                summary["warnings"].append(warning)
                self.warnings.append(warning)
                continue

            multiview_summary = self._attach_object_multiview_part_masks(
                object_parts,
                schema,
                obj,
                image_path,
                object_label,
                detector,
                sam,
                vlm,
                object_work_dir,
            )
            object_agent.record_multiview_summary(multiview_summary, object_parts)

            remapped_part_ids = []
            for part in object_parts:
                local_part_id = int(part.part_id)
                part.part_id = next_part_id
                next_part_id += 1
                part.metadata["object_id"] = int(obj.object_id)
                part.metadata["object_name"] = object_label
                part.metadata["object_mask_path"] = obj.mask_path
                part.metadata["object_local_part_id"] = local_part_id
                part.metadata["object_part_schema"] = schema
                part.physics_group = f"object_{int(obj.object_id)}:{part.physics_group or part.name}"
                remapped_part_ids.append(int(part.part_id))
                all_parts.append(part)
                _write_json(Path(part.mask_path).parent / "part_summary.json", {"part": part.to_dict()})

            obj.name = _safe_name(object_label)
            obj.part_ids = remapped_part_ids
            obj.metadata["object_centric_part_schema"] = schema
            obj.metadata["object_centric_parts_dir"] = str(object_work_dir / "parts")
            summary["objects"].append(
                {
                    "object_id": int(obj.object_id),
                    "object_name": obj.name,
                    "part_ids": remapped_part_ids,
                    "part_count": len(remapped_part_ids),
                    "quality": quality,
                    "parts_dir": str(object_work_dir / "parts"),
                    "multiview_summary": multiview_summary,
                }
            )

        if all_parts:
            masks = []
            labels = []
            image = load_rgb(image_path)
            for part in all_parts:
                try:
                    masks.append(read_mask(part.mask_path))
                    labels.append(f"{part.part_id}:{part.metadata.get('object_name', '')}/{part.name}")
                except Exception:
                    continue
            parts_dir = scene_dir / "parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            if masks:
                save_rgb(overlay_multiple_masks(image, masks, labels=labels), parts_dir / "parts_overlay.png")
            _write_json(parts_dir / "selection_summary.json", {"source": "object_centric", "parts": [p.to_dict() for p in all_parts], "warnings": summary["warnings"]})
        _write_json(object_parts_root / "object_centric_part_summary.json", summary)
        return all_parts, objects, summary


    def _load_whole_physgm_result(self, whole_dir) -> PhysGMResult:
        whole_dir = Path(whole_dir).expanduser().resolve()
        point_cloud = whole_dir / "point_clouds.ply"
        predicted = whole_dir / "predicted_phys.json"
        if not point_cloud.exists():
            raise RuntimeError(f"Existing whole PhysGM dir has no point_clouds.ply: {whole_dir}")
        raw = {}
        material = "Plastic"
        E = default_E_for_material(material)
        nu = default_nu_for_material(material)
        density = density_for_material(material)
        if predicted.exists():
            try:
                raw = _load_json(predicted)
                material = normalize_material_name(raw.get("material", material))
                E = float(raw.get("E", E) or E)
                nu = float(raw.get("nu", nu) or nu)
                density = float(raw.get("density", density_for_material(material)) or density_for_material(material))
            except Exception as exc:
                self.warnings.append(f"Failed to read existing whole PhysGM physics: {exc}")
        return PhysGMResult(str(whole_dir), str(point_cloud), str(predicted), material, E, nu, density, raw)

    def _fallback_part_physics(self, part: PartInstance) -> PhysicsParams:
        material = normalize_material_name(part.expected_materials[0] if part.expected_materials else "Plastic")
        return PhysicsParams(
            material=material,
            material_confidence=0.3,
            E=default_E_for_material(material),
            nu=default_nu_for_material(material),
            density=density_for_material(material),
            confidence=0.3 * part.confidence,
            source_outputs=[],
            warnings=["Skipped part PhysGM; used material table defaults."],
        )

    def run(self, image_path, scene_name, object_hint=None) -> PartPhysResult:
        image_path = str(Path(image_path).expanduser().resolve())
        image = load_rgb(image_path)
        scene_dir = (Path(_get(self.config, "output_dir", "results_partphys")).expanduser() / scene_name).resolve()
        scene_dir.mkdir(parents=True, exist_ok=True)
        input_dir = scene_dir / "input"
        object_dir = scene_dir / "object"
        schema_dir = scene_dir / "schema"
        candidates_dir = scene_dir / "candidates"
        simulation_dir = scene_dir / "simulation"
        for d in [input_dir, object_dir, schema_dir, candidates_dir, simulation_dir, scene_dir / "agent_logs"]:
            d.mkdir(parents=True, exist_ok=True)

        manual = self._load_manual_masks(_get(self.config, "masks_json"), image.size)
        manual_parts = bool(manual and manual.get("parts"))
        require_vlm = bool(_get(self.config, "require_vlm", False))
        vlm = self._init_vlm(required=require_vlm)
        detector = self._init_detector()
        sam = self._init_sam()

        object_name = object_hint or _get(self.config, "object") or None
        if object_name is None:
            object_name = vlm.identify_object(image_path).get("object", "object")

        if manual and manual.get("object_mask"):
            object_mask = read_mask(manual["object_mask"])
            object_mask_path = object_dir / "object_mask.png"
            save_mask(object_mask, object_mask_path)
            save_rgb(overlay_mask(image, object_mask), object_dir / "object_overlay.png")
            object_bbox = mask_to_bbox(object_mask)
            _write_json(object_dir / "object_bbox.json", object_bbox.to_dict())
        elif manual and manual.get("parts"):
            masks = [read_mask(p["mask"]) for p in manual["parts"]]
            object_mask = np.logical_or.reduce(masks)
            object_mask_path = object_dir / "object_mask.png"
            save_mask(object_mask, object_mask_path)
            save_rgb(overlay_mask(image, object_mask), object_dir / "object_overlay.png")
            object_bbox = mask_to_bbox(object_mask)
            _write_json(object_dir / "object_bbox.json", object_bbox.to_dict())
            self.warnings.append("masks_json had no object_mask; used union of part masks.")
        else:
            object_mask_path, object_bbox, object_warnings = generate_object_mask(
                image_path,
                object_name,
                detector,
                sam,
                object_dir,
                fallback_to_full_image=bool(_get(self.config, "fallback_to_full_image", True)),
                keep_multi_components=str(_get(self.config, "object_mode", "single")).lower() == "auto",
            )
            object_mask_path = Path(object_mask_path)
            self.warnings.extend(object_warnings)

        whole_input_image = self._save_object_inputs(image_path, object_mask_path, object_bbox, input_dir)

        schema = self._schema_from_manual_or_file(manual, object_name)
        if not schema:
            schema = vlm.generate_part_schema(image_path, object_name, str(object_mask_path))
        schema = normalize_part_schema(schema, object_name)
        _write_json(schema_dir / "part_schema.json", schema)
        agent_view_paths = _candidate_multiview_paths(_get(self.config, "multiview_dir"))
        part_agent = PartSegAgentController(self.config, scene_dir, vlm, object_name, image_path, agent_view_paths)
        agent_plan = part_agent.plan(schema)
        if (
            part_agent.enabled
            and not manual_parts
            and not _get(self.config, "part_schema_json")
            and isinstance(agent_plan.get("parts"), list)
            and agent_plan.get("parts")
        ):
            schema = normalize_part_schema(
                {
                    "object": agent_plan.get("object") or object_name,
                    "parts": agent_plan.get("parts", []),
                    "relations": agent_plan.get("relations", []),
                },
                object_name,
            )
            _write_json(schema_dir / "part_schema.json", schema)
            _write_json(schema_dir / "agent_plan.json", agent_plan)

        if manual_parts:
            parts = self._manual_part_instances(manual, object_mask_path, scene_dir, image_path, schema)
        else:
            quality = {}
            critique = None
            rounds = part_agent.max_rounds if part_agent.enabled else 1
            parts = []
            for round_idx in range(rounds):
                overrides = part_agent.round_overrides(critique, round_idx)
                round_candidates_dir = candidates_dir if round_idx == 0 else scene_dir / "agent_rounds" / f"round_{round_idx:02d}" / "candidates"
                parts, _, quality = self._run_segmentation_agent(
                    image_path=image_path,
                    object_mask_path=object_mask_path,
                    object_bbox=object_bbox,
                    schema=schema,
                    detector=detector,
                    sam=sam,
                    vlm=vlm,
                    output_dir=scene_dir,
                    candidates_dir=round_candidates_dir,
                    overrides=overrides,
                )
                overlay_path = scene_dir / "parts" / "parts_overlay.png"
                critique = part_agent.critique(parts, quality, overlay_path, round_idx)
                part_agent.record_round(round_idx, parts, quality, critique, overrides)
                if not part_agent.should_retry(critique, quality, round_idx):
                    break
            if not quality.get("ok"):
                self.warnings.append(f"Segmentation accepted with warnings: {quality.get('reason')}")

        multiview_summary = self._attach_multiview_part_masks(parts, schema, image_path, object_name, detector, sam, vlm, scene_dir, part_agent)
        multiview_critique = part_agent.record_multiview_summary(multiview_summary, parts)
        if multiview_critique and not multiview_critique.get("ok", True):
            self.warnings.append("Agent multiview critic accepted result with warnings.")

        objects, object_summary = separate_scene_objects(
            image_path=image_path,
            object_mask_path=object_mask_path,
            parts=parts,
            sam_tool=sam,
            output_dir=scene_dir / "objects",
            object_name=object_name,
            mode=_get(self.config, "object_mode", "single"),
            max_objects=int(_get(self.config, "max_objects", 6)),
            min_object_area_ratio=float(_get(self.config, "min_object_area_ratio", 0.015)),
            multiview_dir=_get(self.config, "multiview_dir"),
        )
        self.warnings.extend(object_summary.get("warnings", []))
        if (
            str(_get(self.config, "object_mode", "single")).lower() == "auto"
            and len(objects) > 1
            and not manual_parts
        ):
            object_centric_parts, objects, object_centric_summary = self._run_object_centric_part_segmentation(
                objects=objects,
                image_path=image_path,
                object_name=object_name,
                detector=detector,
                sam=sam,
                vlm=vlm,
                scene_dir=scene_dir,
            )
            if object_centric_parts:
                parts = object_centric_parts
                object_summary["object_centric_parts"] = object_centric_summary
                object_summary["objects"] = [obj.to_dict() for obj in objects]
                _write_json(scene_dir / "objects" / "objects_summary.json", object_summary)
            else:
                self.warnings.append("Object-centric part segmentation produced no usable parts; kept global part segmentation.")
        self._refresh_part_summaries(scene_dir, parts)

        if bool(_get(self.config, "mask_only", False)):
            self.warnings.append("Mask-only mode; stopped after object mask and part masks.")
            result = PartPhysResult(
                scene_name=scene_name,
                object_name=object_name,
                object_mask_path=str(object_mask_path),
                parts=parts,
                objects=objects,
                part_physics={},
                whole_physgm=PhysGMResult("", None, "", "", 0.0, 0.0, None, {}),
                assignment_summary={"mode": "mask_only", "warnings": ["Mask-only mode skipped PhysGM, assignment, and simulation config."]},
                sim_config_path=None,
                simulation_output_dir=None,
                warnings=self.warnings,
            )
            _write_json(scene_dir / "partphys_summary.json", result)
            write_warnings(scene_dir / "warnings.txt", self.warnings)
            return result

        runner = None

        def get_runner() -> PhysGMRunner:
            nonlocal runner
            if runner is None:
                runner = PhysGMRunner(
                    config_path=_get(self.config, "physgm_config"),
                    checkpoint_path=_get(self.config, "checkpoint"),
                    template_config_path=_get(self.config, "template_config"),
                    device=_get(self.config, "device", "cpu"),
                    output_base_dir=scene_dir,
                    mock=bool(_get(self.config, "mock_physgm", False)),
                    physgm_root=_get(self.config, "physgm_root"),
                    amp_dtype=_get(self.config, "amp_dtype", "bf16"),
                    mvadapter_root=_get(self.config, "mvadapter_root"),
                    mvadapter_variant=_get(self.config, "mvadapter_variant", "sd"),
                    mvadapter_device=_get(self.config, "mvadapter_device"),
                    mvadapter_prompt=_get(self.config, "mvadapter_prompt", "high quality object, clean background"),
                    mvadapter_num_views=int(_get(self.config, "mvadapter_num_views", 6)),
                    mvadapter_steps=int(_get(self.config, "mvadapter_steps", 50)),
                    mvadapter_guidance_scale=float(_get(self.config, "mvadapter_guidance_scale", 3.0)),
                    mvadapter_seed=int(_get(self.config, "mvadapter_seed", 1234)),
                    mvadapter_timeout=int(_get(self.config, "mvadapter_timeout", 1800)),
                    mvadapter_adapter_path=_get(self.config, "mvadapter_adapter_path"),
                    mvadapter_required=bool(_get(self.config, "require_mvadapter", False)),
                )
            return runner

        part_physics: dict[int, PhysicsParams] = {}
        segmentation_only = bool(_get(self.config, "segmentation_only", False))
        if segmentation_only:
            self.warnings.append("Segmentation-only mode; skipped per-part PhysGM and simulation config.")
        elif bool(_get(self.config, "skip_part_physgm", False)):
            for part in parts:
                part_physics[part.part_id] = self._fallback_part_physics(part)
                self.warnings.extend(part_physics[part.part_id].warnings)
        else:
            physgm_runner = get_runner()
            for part in parts:
                part_dir = Path(part.mask_path).parent
                part_object_mask_path = (part.metadata or {}).get("object_mask_path") or object_mask_path
                crops = build_part_crops(image_path, part_object_mask_path, part.mask_path, part_dir)
                try:
                    material_prior = vlm.infer_material_prior(crops.get("padded") or image_path, part.name)
                except Exception as exc:
                    material_prior = {}
                    self.warnings.append(f"VLM material prior failed for {part.name}; continuing with PhysGM outputs: {exc}")
                params = infer_part_physics(part, crops, physgm_runner, material_prior, part_dir)
                part_physics[part.part_id] = params
                self.warnings.extend(params.warnings)

        whole_dir_arg = _get(self.config, "whole_physgm_dir")
        direct_object_ids = None
        physgm_scene_mode = "single_object"
        if whole_dir_arg:
            whole_result = self._load_whole_physgm_result(whole_dir_arg)
            whole_dir = Path(whole_result.scene_dir)
        else:
            whole_dir = scene_dir / "physgm_whole"
            should_run_multi_object = (
                str(_get(self.config, "object_mode", "auto")).lower() == "auto"
                and len(objects) > 1
            )
            if should_run_multi_object:
                physgm_scene_mode = "multi_object"
                geometry_source = str(_get(self.config, "multi_object_geometry_source", "whole_scene")).lower()
                multi_object_result = None
                if geometry_source == "per_object":
                    try:
                        multi_object_result, direct_object_ids = run_multi_object_physgm(
                            get_runner(),
                            whole_input_image,
                            scene_name=f"{scene_name}_multi_object",
                            output_dir=whole_dir / "per_object_physgm",
                            objects=objects,
                            multiview_dir=_get(self.config, "multiview_dir"),
                            use_mvadapter=bool(_get(self.config, "use_mvadapter", False)),
                        )
                    except Exception as exc:
                        self.warnings.append(f"Per-object PhysGM failed; continuing with whole-scene geometry and projection labels: {exc}")
                else:
                    self.warnings.append("Using whole-scene PhysGM geometry for multi-object scene; object and part labels are assigned by multi-view projection.")
                if multi_object_result is not None:
                    whole_result = multi_object_result
                    whole_dir = Path(whole_result.scene_dir)
                    whole_result.raw["mode"] = "multi_object_per_object_merged"
                    whole_result.raw["geometry_source"] = "per_object_physgm"
                else:
                    whole_result = get_runner().infer_image(
                        whole_input_image,
                        scene_name=f"{scene_name}_whole",
                        output_dir=whole_dir,
                        save_gaussian=True,
                        use_mvadapter=bool(_get(self.config, "use_mvadapter", False)),
                        multiview_dir=_get(self.config, "multiview_dir"),
                    )
                    whole_result.raw["mode"] = "multi_object_whole_scene"
                    whole_result.raw["geometry_source"] = "whole_scene_physgm"
                    whole_result.raw["per_object_physgm_dir"] = None
                    whole_result.raw["per_object_physgm"] = None
                _write_json(whole_result.predicted_phys_path, whole_result.raw)
            else:
                whole_result = get_runner().infer_image(
                    whole_input_image,
                    scene_name=f"{scene_name}_whole",
                    output_dir=whole_dir,
                    save_gaussian=True,
                    use_mvadapter=bool(_get(self.config, "use_mvadapter", False)),
                    multiview_dir=_get(self.config, "multiview_dir"),
                )
        if not whole_result.point_cloud_path:
            self.warnings.append("Whole-object PhysGM did not produce point_clouds.ply.")

        assignment_summary: dict[str, Any] = {"mode": _get(self.config, "assignment_mode", "projection"), "warnings": []}
        part_aabbs: list[dict[str, Any]] = []
        assignment_mode = _get(self.config, "assignment_mode", "projection")
        if assignment_mode != "none" and whole_result.point_cloud_path and Path(whole_result.point_cloud_path).exists():
            positions = load_ply_positions(whole_result.point_cloud_path)
            if assignment_mode == "projection":
                assign = assign_per_object_projection(positions, parts, direct_object_ids, whole_result, scene_dir / "assignment")
                if assign is not None:
                    assignment_summary["mode"] = "object_local_projection"
                else:
                    part_masks = [
                        {
                            "part_id": p.part_id,
                            "name": p.name,
                            "mask_path": p.mask_path,
                            "view_masks": p.metadata.get("view_masks", {}),
                            "area": p.area,
                            "confidence": p.confidence,
                            "physics_group": p.physics_group,
                        }
                        for p in parts
                    ]
                    assign = assign_by_projection(positions, part_masks, whole_dir / "input_batch_meta.npz", image.size)
                if assign.get("assigned_ratio", 0.0) < 0.05 and bool(_get(self.config, "fallback_to_aabb_heuristic", True)):
                    self.warnings.append("Projection assignment ratio too low; falling back to AABB heuristic.")
                    fallback = assign_by_aabb_heuristic(positions, parts, image.size)
                    fallback["warnings"] = assign.get("warnings", []) + fallback.get("warnings", [])
                    assign = fallback
                    assignment_summary["mode"] = "aabb_heuristic"
            else:
                assign = assign_by_aabb_heuristic(positions, parts, image.size)
                assignment_summary["mode"] = "aabb_heuristic"
            ids = assign["gaussian_part_ids"]
            part_aabbs = build_part_aabbs(
                positions,
                ids,
                parts,
                min_count=int(_get(self.config, "min_gaussian_count_per_part", 20)),
                padding_ratio=float(_get(self.config, "padding_ratio", 0.15)),
                min_half_size=float(_get(self.config, "min_half_size", 0.02)),
            )
            assignment_summary.update(
                {
                    "assigned_ratio": assign.get("assigned_ratio", 0.0),
                    "per_part_counts": assign.get("per_part_counts", {}),
                    "projection_views": assign.get("view_labels", []),
                    "projection_view_hits": assign.get("per_view_hits", {}),
                    "projection_image_size": assign.get("projection_image_size"),
                    "projection_mean_view_support": assign.get("mean_view_support"),
                    "projection_view_support_counts": assign.get("view_support_counts", {}),
                    "projection_margin_ratio": assign.get("margin_ratio", {}),
                    "projection_low_confidence_count": assign.get("low_confidence_count", 0),
                    "projection_smoothed_count": assign.get("smoothed_count", 0),
                    "projection_knn_unknown_reassigned_count": assign.get("knn_unknown_reassigned_count", 0),
                    "projection_knn_island_reassigned_count": assign.get("knn_island_reassigned_count", 0),
                    "object_local_projection": assign.get("object_local_projection", []),
                    "aabb_count": len(part_aabbs),
                    "physgm_scene_mode": physgm_scene_mode,
                    "warnings": assign.get("warnings", []),
                }
            )
            save_assignment_outputs(scene_dir / "assignment", ids, part_aabbs, assignment_summary, parts, whole_result.point_cloud_path)
            projected_object_ids = None
            object_projection_summary = None
            if direct_object_ids is not None and len(direct_object_ids) == len(ids):
                projected_object_ids = direct_object_ids
                assignment_summary["object_assignment_source"] = "multi_object_physgm_direct"
            elif assignment_mode == "projection" and objects:
                object_masks = [
                    {
                        "part_id": obj.object_id,
                        "name": obj.name,
                        "mask_path": obj.mask_path,
                        "view_masks": obj.metadata.get("view_masks", {}),
                        "area": obj.area,
                        "confidence": obj.confidence,
                    }
                    for obj in objects
                ]
                object_assign = assign_by_projection(positions, object_masks, whole_dir / "input_batch_meta.npz", image.size)
                projected_object_ids = object_assign.get("gaussian_part_ids")
                object_projection_summary = {
                    "assigned_ratio": object_assign.get("assigned_ratio", 0.0),
                    "per_object_counts": object_assign.get("per_part_counts", {}),
                    "projection_views": object_assign.get("view_labels", []),
                    "projection_view_hits": object_assign.get("per_view_hits", {}),
                    "warnings": object_assign.get("warnings", []),
                }
                assignment_summary["object_assignment_source"] = "object_mask_projection"
            object_assignment = save_object_assignment_outputs(
                scene_dir / "assignment",
                ids,
                parts,
                objects,
                positions,
                object_ids_override=projected_object_ids,
            )
            if object_projection_summary is not None:
                assignment_summary["object_projection_assignment"] = object_projection_summary
            assignment_summary.update(object_assignment)
            _write_json(scene_dir / "assignment" / "assignment_summary.json", assignment_summary)
            _write_json(
                scene_dir / "objects" / "objects_summary.json",
                {
                    "mode": _get(self.config, "object_mode", "single"),
                    "object_count": len(objects),
                    "objects": [obj.to_dict() for obj in objects],
                    "warnings": object_summary.get("warnings", []),
                },
            )
            self.warnings.extend(assign.get("warnings", []))
            if not part_aabbs:
                self.warnings.append("No valid part AABBs built; simulation will use only global physics.")
        else:
            assignment_summary["warnings"].append("Assignment skipped.")
            save_assignment_outputs(scene_dir / "assignment", np.array([], dtype=np.int32), [], assignment_summary, parts, whole_result.point_cloud_path)
            object_assignment = save_object_assignment_outputs(scene_dir / "assignment", np.array([], dtype=np.int32), parts, objects, None)
            assignment_summary.update(object_assignment)
            _write_json(scene_dir / "assignment" / "assignment_summary.json", assignment_summary)

        if segmentation_only:
            result = PartPhysResult(
                scene_name=scene_name,
                object_name=object_name,
                object_mask_path=str(object_mask_path),
                parts=parts,
                objects=objects,
                part_physics=part_physics,
                whole_physgm=whole_result,
                assignment_summary=assignment_summary,
                sim_config_path=None,
                simulation_output_dir=None,
                warnings=self.warnings,
            )
            _write_json(scene_dir / "partphys_summary.json", result)
            write_warnings(scene_dir / "warnings.txt", self.warnings)
            return result

        template_config = self._resolve_physgm_path(_get(self.config, "template_config"))
        if not template_config or not Path(template_config).exists():
            raise RuntimeError(f"Template config not found: {_get(self.config, 'template_config')}")
        sim_config_path = simulation_dir / "sim_config_partphys.json"
        sim_part_physics = part_physics
        sim_part_aabbs = part_aabbs
        if physgm_scene_mode == "multi_object" and bool(_get(self.config, "skip_part_physgm", False)):
            sim_part_physics = {}
            sim_part_aabbs = []
            self.warnings.append("Multi-object simulation used global physics only because per-part PhysGM was skipped.")
        _, sim_warnings = build_part_aware_sim_config(
            template_config,
            sim_config_path,
            whole_result,
            parts,
            sim_part_physics,
            sim_part_aabbs,
        )
        self.warnings.extend(sim_warnings)

        cmd_preview = [
            sys.executable,
            "gs_simulation.py",
            "--model_path",
            str(Path(whole_dir).resolve()),
            "--output_path",
            str(simulation_dir.resolve()),
            "--config",
            str(sim_config_path.resolve()),
            "--render_img",
            "--compile_video",
        ]
        if bool(_get(self.config, "white_bg", False)):
            cmd_preview.append("--white_bg")
        (simulation_dir / "command.txt").write_text(" ".join(cmd_preview) + "\n", encoding="utf-8")
        simulation_output_dir = None
        if bool(_get(self.config, "simulate", False)):
            if self.physgm_root is None:
                self.warnings.append("Simulation skipped: PhysGM root not found.")
            else:
                sim_result = run_simulation(
                    self.physgm_root,
                    Path(whole_dir).resolve(),
                    simulation_dir.resolve(),
                    sim_config_path.resolve(),
                    render_img=bool(_get(self.config, "render_img", True)),
                    compile_video=bool(_get(self.config, "compile_video", True)),
                    white_bg=bool(_get(self.config, "white_bg", False)),
                )
                _write_json(simulation_dir / "run_result.json", sim_result)
                simulation_output_dir = str(simulation_dir)
                if sim_result["returncode"] != 0:
                    self.warnings.append(f"Simulation failed with return code {sim_result['returncode']}.")

        result = PartPhysResult(
            scene_name=scene_name,
            object_name=object_name,
            object_mask_path=str(object_mask_path),
            parts=parts,
            objects=objects,
            part_physics=part_physics,
            whole_physgm=whole_result,
            assignment_summary=assignment_summary,
            sim_config_path=str(sim_config_path),
            simulation_output_dir=simulation_output_dir,
            warnings=self.warnings,
        )
        _write_json(scene_dir / "partphys_summary.json", result)
        write_warnings(scene_dir / "warnings.txt", self.warnings)
        return result
