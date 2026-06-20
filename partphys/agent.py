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
from .physgm_runner import PhysGMRunner, _find_physgm_root, _resolve_path
from .proposals import generate_object_mask, generate_part_candidates
from .report import write_json, write_warnings
from .sam_tool import SAMTool
from .selector import select_physical_parts
from .sim_config_builder import build_part_aware_sim_config
from .segmentation_agent import SegmentationAgent
from .types import BBox, PartInstance, PartPhysResult, PartSpec, PhysGMResult, PhysicsParams
from .vlm import NoVLMClient, OpenAICompatibleVLMClient, normalize_part_schema, part_specs_from_schema


def _get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "part"


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
            return SAMTool(
                checkpoint,
                config=_get(self.config, "sam_config"),
                device=_get(self.config, "device", "cuda"),
                sam2_root=_get(self.config, "sam2_root"),
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
        require_vlm = not manual_parts
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
            )
            object_mask_path = Path(object_mask_path)
            self.warnings.extend(object_warnings)

        whole_input_image = self._save_object_inputs(image_path, object_mask_path, object_bbox, input_dir)

        schema = self._schema_from_manual_or_file(manual, object_name)
        if not schema:
            schema = vlm.generate_part_schema(image_path, object_name, str(object_mask_path))
        schema = normalize_part_schema(schema, object_name)
        _write_json(schema_dir / "part_schema.json", schema)

        if manual_parts:
            parts = self._manual_part_instances(manual, object_mask_path, scene_dir, image_path, schema)
        else:
            parts, _, quality = SegmentationAgent(
                image_path=image_path,
                object_mask_path=object_mask_path,
                object_bbox=object_bbox,
                part_schema=schema,
                detector=detector,
                sam_tool=sam,
                vlm_client=vlm,
                output_dir=scene_dir,
                candidates_dir=candidates_dir,
                max_parts=int(_get(self.config, "max_parts", 6)),
                min_part_area_ratio=float(_get(self.config, "min_part_area_ratio", 0.01)),
                coverage_threshold=float(_get(self.config, "coverage_threshold", 0.75)),
                max_retries=int(_get(self.config, "segmentation_max_retries", 2)),
                vlm_weight=float(_get(self.config, "segmentation_vlm_weight", 0.55)),
                min_accept_score=float(_get(self.config, "segmentation_min_accept_score", 0.45)),
            ).run()
            if not quality.get("ok"):
                raise RuntimeError(f"Segmentation agent failed: {quality}")

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
                crops = build_part_crops(image_path, object_mask_path, part.mask_path, part_dir)
                try:
                    material_prior = vlm.infer_material_prior(crops.get("padded") or image_path, part.name)
                except Exception as exc:
                    material_prior = {}
                    self.warnings.append(f"VLM material prior failed for {part.name}; continuing with PhysGM outputs: {exc}")
                params = infer_part_physics(part, crops, physgm_runner, material_prior, part_dir)
                part_physics[part.part_id] = params
                self.warnings.extend(params.warnings)

        whole_dir_arg = _get(self.config, "whole_physgm_dir")
        if whole_dir_arg:
            whole_result = self._load_whole_physgm_result(whole_dir_arg)
            whole_dir = Path(whole_result.scene_dir)
        else:
            whole_dir = scene_dir / "physgm_whole"
            whole_result = get_runner().infer_image(
                whole_input_image,
                scene_name=f"{scene_name}_whole",
                output_dir=whole_dir,
                save_gaussian=True,
                use_mvadapter=bool(_get(self.config, "use_mvadapter", False)),
            )
        if not whole_result.point_cloud_path:
            self.warnings.append("Whole-object PhysGM did not produce point_clouds.ply.")

        assignment_summary: dict[str, Any] = {"mode": _get(self.config, "assignment_mode", "projection"), "warnings": []}
        part_aabbs: list[dict[str, Any]] = []
        assignment_mode = _get(self.config, "assignment_mode", "projection")
        if assignment_mode != "none" and whole_result.point_cloud_path and Path(whole_result.point_cloud_path).exists():
            positions = load_ply_positions(whole_result.point_cloud_path)
            if assignment_mode == "projection":
                part_masks = [{"part_id": p.part_id, "mask_path": p.mask_path, "area": p.area, "confidence": p.confidence} for p in parts]
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
                    "aabb_count": len(part_aabbs),
                    "warnings": assign.get("warnings", []),
                }
            )
            save_assignment_outputs(scene_dir / "assignment", ids, part_aabbs, assignment_summary, parts, whole_result.point_cloud_path)
            self.warnings.extend(assign.get("warnings", []))
            if not part_aabbs:
                self.warnings.append("No valid part AABBs built; simulation will use only global physics.")
        else:
            assignment_summary["warnings"].append("Assignment skipped.")
            save_assignment_outputs(scene_dir / "assignment", np.array([], dtype=np.int32), [], assignment_summary, parts, whole_result.point_cloud_path)

        if segmentation_only:
            result = PartPhysResult(
                scene_name=scene_name,
                object_name=object_name,
                object_mask_path=str(object_mask_path),
                parts=parts,
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
        _, sim_warnings = build_part_aware_sim_config(
            template_config,
            sim_config_path,
            whole_result,
            parts,
            part_physics,
            part_aabbs,
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
