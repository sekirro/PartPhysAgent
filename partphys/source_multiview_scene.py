from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .material_table import (
    density_for_material,
    default_E_for_material,
    default_nu_for_material,
    material_to_solver_material,
    normalize_material_name,
)
from .multi_object_physgm import (
    _header_signature,
    _raise_low_object_opacity,
    _read_ply_header,
    _safe_name,
    _select_vertex_indices,
    _vertex_dtype,
)
from .physgm_runner import PhysGMRunner
from .scene_builder import FRAME_NAMES
from .types import PhysGMResult


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("multi-object source manifest must contain a non-empty 'objects' list")
    for idx, obj in enumerate(objects):
        if not obj.get("views_dir"):
            raise ValueError(f"manifest object {idx} is missing views_dir")
        obj.setdefault("name", f"object_{idx:03d}")
        obj.setdefault("object_id", idx)
    return data


def _check_source_views(views_dir: str | Path) -> Path:
    root = Path(views_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"views_dir does not exist: {root}")
    missing = [name for name in [*FRAME_NAMES, "pose.json"] if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"views_dir {root} is missing: {', '.join(missing)}")
    return root


def _placement_from_object(obj: dict[str, Any]) -> dict[str, Any]:
    translation = obj.get("translation", [0.0, 0.0, 0.0])
    if len(translation) != 3:
        raise ValueError(f"Object {obj.get('name')} translation must have 3 values")
    scale = float(obj.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Object {obj.get('name')} scale must be positive")
    return {
        "object_id": int(obj.get("object_id", 0)),
        "object_name": str(obj.get("name", "object")),
        "translation": [float(x) for x in translation],
        "scale": scale,
        "align_bottom_axis": str(obj.get("align_bottom_axis", "z")),
        "center_axes": list(obj.get("center_axes", ["x", "y"])),
    }


def _read_binary_vertices(path: Path) -> tuple[list[str], np.ndarray]:
    header, count, header_end, fmt = _read_ply_header(path)
    if "binary_" not in fmt:
        raise ValueError(f"Only binary PLY is supported for source-view multi-object merge: {path}")
    dtype = _vertex_dtype(header, fmt)
    with path.open("rb") as f:
        f.seek(header_end)
        vertices = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count).copy()
    return header, vertices


def _transform_vertices(vertices: np.ndarray, placement: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    names = vertices.dtype.names or ()
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY vertex layout must contain x/y/z fields")
    transformed = vertices.copy()
    xyz = np.stack(
        [
            transformed["x"].astype(np.float32),
            transformed["y"].astype(np.float32),
            transformed["z"].astype(np.float32),
        ],
        axis=1,
    )
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    scale = float(placement["scale"])
    translation = np.asarray(placement["translation"], dtype=np.float32)
    axes = {"x": 0, "y": 1, "z": 2}
    origin = bbox_center.copy()
    for axis in ("x", "y", "z"):
        if axis not in placement.get("center_axes", []):
            origin[axes[axis]] = 0.0
    bottom_axis = placement.get("align_bottom_axis")
    if bottom_axis in axes:
        origin[axes[bottom_axis]] = bbox_min[axes[bottom_axis]]
    xyz = (xyz - origin.reshape(1, 3)) * scale + translation.reshape(1, 3)
    transformed["x"] = xyz[:, 0].astype(transformed.dtype["x"])
    transformed["y"] = xyz[:, 1].astype(transformed.dtype["y"])
    transformed["z"] = xyz[:, 2].astype(transformed.dtype["z"])
    aabb_min = xyz.min(axis=0)
    aabb_max = xyz.max(axis=0)
    metadata = {
        "source_bbox_min": bbox_min.astype(float).tolist(),
        "source_bbox_max": bbox_max.astype(float).tolist(),
        "source_origin": origin.astype(float).tolist(),
        "merged_bbox_min": aabb_min.astype(float).tolist(),
        "merged_bbox_max": aabb_max.astype(float).tolist(),
        "merged_center": (0.5 * (aabb_min + aabb_max)).astype(float).tolist(),
        "merged_half_size": (0.5 * (aabb_max - aabb_min)).astype(float).tolist(),
    }
    return transformed, metadata


def merge_source_object_plys(
    object_rows: list[dict[str, Any]],
    output_path: Path,
    max_vertices_per_object: int | None = None,
) -> tuple[list[int], list[dict[str, Any]], np.ndarray, np.ndarray | None, dict[str, int]]:
    if not object_rows:
        raise ValueError("No object rows to merge")
    first_signature = None
    merged_vertices: list[np.ndarray] = []
    counts: list[int] = []
    object_aabbs: list[dict[str, Any]] = []
    output_header: list[str] | None = None
    object_id_chunks: list[np.ndarray] = []
    global_part_id_chunks: list[np.ndarray] = []
    part_id_map: dict[str, int] = {}
    next_global_part_id = 0
    saw_part_ids = False

    for row in object_rows:
        object_id = int(row["object_id"])
        ply_path = Path(row["point_cloud_path"])
        header, vertices = _read_binary_vertices(ply_path)
        signature = _header_signature(header)
        if first_signature is None:
            first_signature = signature
            output_header = header
        elif signature != first_signature:
            raise ValueError(f"Cannot merge PLY files with different vertex layouts: {ply_path}")
        selected = _select_vertex_indices(vertices, max_vertices_per_object)
        selected_vertices = vertices[selected].copy()
        transformed, aabb = _transform_vertices(selected_vertices, row["placement"])
        _raise_low_object_opacity(transformed)
        merged_vertices.append(transformed)
        count = int(len(transformed))
        counts.append(count)
        object_id_chunks.append(np.full(count, object_id, dtype=np.int32))

        part_ids_path = row.get("gaussian_part_ids_path")
        if part_ids_path and Path(part_ids_path).exists():
            local_ids = np.load(part_ids_path).astype(np.int32)
            if len(local_ids) == len(vertices):
                saw_part_ids = True
                selected_local = local_ids[selected]
                selected_global = np.full(len(selected_local), -1, dtype=np.int32)
                for local_pid in sorted(int(x) for x in np.unique(selected_local) if int(x) >= 0):
                    key = f"{object_id}:{local_pid}"
                    if key not in part_id_map:
                        part_id_map[key] = next_global_part_id
                        next_global_part_id += 1
                    selected_global[selected_local == local_pid] = part_id_map[key]
                global_part_id_chunks.append(selected_global)
            else:
                row.setdefault("warnings", []).append(
                    f"Skipped part-id merge: local part ids count {len(local_ids)} does not match PLY vertex count {len(vertices)}."
                )
                global_part_id_chunks.append(np.full(count, -1, dtype=np.int32))
        else:
            global_part_id_chunks.append(np.full(count, -1, dtype=np.int32))

        aabb.update(
            {
                "object_id": object_id,
                "object_name": row["object_name"],
                "count": count,
                "placement": row["placement"],
            }
        )
        object_aabbs.append(aabb)

    total = int(sum(counts))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert output_header is not None
    header = [f"element vertex {total}" if line.startswith("element vertex ") else line for line in output_header]
    with output_path.open("wb") as out:
        out.write(("\n".join(header) + "\n").encode("ascii"))
        for vertices in merged_vertices:
            out.write(vertices.tobytes())
    object_ids = np.concatenate(object_id_chunks) if object_id_chunks else np.zeros(0, dtype=np.int32)
    global_part_ids = np.concatenate(global_part_id_chunks) if saw_part_ids else None
    return counts, object_aabbs, object_ids, global_part_ids, part_id_map


def _global_physics(manifest: dict[str, Any], object_rows: list[dict[str, Any]]) -> dict[str, Any]:
    explicit = manifest.get("global_physics") or {}
    if explicit:
        material = normalize_material_name(str(explicit.get("material", "Plastic")))
        return {
            "material": material,
            "solver_material": material_to_solver_material(material),
            "E": float(explicit.get("E", default_E_for_material(material))),
            "nu": float(explicit.get("nu", default_nu_for_material(material))),
            "density": float(explicit.get("density", density_for_material(material))),
            "source": "manifest.global_physics",
        }
    if object_rows:
        row = max(object_rows, key=lambda item: item.get("gaussian_count", 0))
        material = normalize_material_name(str(row.get("material", "Plastic")))
        return {
            "material": material,
            "solver_material": material_to_solver_material(material),
            "E": float(row.get("E") or default_E_for_material(material)),
            "nu": float(row.get("nu") or default_nu_for_material(material)),
            "density": float(row.get("density") or density_for_material(material)),
            "source": "largest_object_physgm_prediction",
        }
    material = "Plastic"
    return {
        "material": material,
        "solver_material": material_to_solver_material(material),
        "E": default_E_for_material(material),
        "nu": default_nu_for_material(material),
        "density": density_for_material(material),
        "source": "default",
    }



def _rotation_matrix_np(degree: float, axis: int) -> np.ndarray:
    rad = np.deg2rad(float(degree))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    if int(axis) == 0:
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if int(axis) == 1:
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if int(axis) == 2:
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Invalid rotation axis: {axis}")


def _normalized_mpm_bbox_from_ply(ply_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    _, vertices = _read_binary_vertices(ply_path)
    names = vertices.dtype.names or ()
    xyz = np.stack(
        [vertices["x"].astype(np.float64), vertices["y"].astype(np.float64), vertices["z"].astype(np.float64)],
        axis=1,
    )
    if "opacity" in names:
        threshold = float(config.get("opacity_threshold", 0.02))
        # PLY opacities are stored as logits in PhysGM/3DGS outputs.
        opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
        mask = opacity > threshold
        if int(mask.sum()) > 0:
            xyz = xyz[mask]
    manual_rot = _rotation_matrix_np(90.0, 0)
    xyz = xyz @ manual_rot.T
    for degree, axis in zip(config.get("rotation_degree", []), config.get("rotation_axis", [])):
        xyz = xyz @ _rotation_matrix_np(float(degree), int(axis)).T
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    max_diff = float(np.max(bbox_max - bbox_min))
    if max_diff <= 0 or not np.isfinite(max_diff):
        raise ValueError("Cannot compute normalized MPM bbox for an empty or degenerate scene")
    scale = float(config.get("scale", 1.0)) / max_diff
    center = 0.5 * (bbox_min + bbox_max)
    normalized = (xyz - center.reshape(1, 3)) * scale + np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    return {
        "source_bbox_min": bbox_min.astype(float).tolist(),
        "source_bbox_max": bbox_max.astype(float).tolist(),
        "normalized_bbox_min": normalized.min(axis=0).astype(float).tolist(),
        "normalized_bbox_max": normalized.max(axis=0).astype(float).tolist(),
        "normalization_max_diff": max_diff,
        "normalization_scale": scale,
        "particle_count_used": int(len(xyz)),
    }


def _target_floor_clearance(manifest: dict[str, Any]) -> float:
    simulation = manifest.get("simulation") if isinstance(manifest.get("simulation"), dict) else {}
    value = simulation.get("floor_clearance", manifest.get("floor_clearance", 0.25))
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("floor_clearance must be a non-negative finite number")
    return value


def _apply_initial_floor_alignment(config: dict[str, Any], scene_ply_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    bbox = _normalized_mpm_bbox_from_ply(scene_ply_path, config)
    clearance = _target_floor_clearance(manifest)
    bottom_y = float(bbox["normalized_bbox_min"][1])
    floor_y = None
    for condition in config.get("boundary_conditions", []):
        if condition.get("type") != "surface_collider":
            continue
        normal = condition.get("normal", [])
        if len(normal) == 3 and abs(float(normal[1]) - 1.0) < 1e-6:
            point = condition.get("point", [1.0, 0.25, 1.0])
            if len(point) == 3:
                floor_y = float(point[1])
                break
    if floor_y is None:
        bbox.update({"enabled": False, "reason": "no upward surface_collider found", "target_floor_clearance": clearance})
        config["initial_floor_alignment"] = bbox
        return bbox
    target_bottom_y = floor_y + clearance
    shift_y = target_bottom_y - bottom_y
    alignment = dict(bbox)
    alignment.update(
        {
            "enabled": True,
            "mode": "post_normalization_particle_shift",
            "floor_y": float(floor_y),
            "target_floor_clearance": float(clearance),
            "normalized_bottom_y_before_shift": float(bottom_y),
            "mpm_position_shift": [0.0, float(shift_y), 0.0],
            "normalized_bottom_y_after_shift": float(bottom_y + shift_y),
            "actual_floor_clearance_after_shift": float(bottom_y + shift_y - floor_y),
        }
    )
    config["initial_floor_alignment"] = alignment
    return alignment



def _elastic_stability_score(E: float, nu: float, density: float) -> float:
    rho = max(float(density), 1.0)
    young = max(float(E), 0.0)
    poisson = min(max(float(nu), -0.95), 0.49)
    mu = young / (2.0 * (1.0 + poisson))
    denom = max((1.0 + poisson) * (1.0 - 2.0 * poisson), 1.0e-6)
    lam = young * poisson / denom
    return (lam + 2.0 * mu) / rho


def _apply_merged_material_stability(config: dict[str, Any], part_materials_json: str | Path | None) -> dict[str, Any] | None:
    if not part_materials_json or not Path(part_materials_json).exists():
        return None
    data = _read_json(Path(part_materials_json), {}) or {}
    items = []
    fallback = data.get("fallback") if isinstance(data.get("fallback"), dict) else {}
    if fallback:
        items.append({"name": fallback.get("name", "fallback"), **fallback})
    for key, value in (data.get("parts") or {}).items():
        if isinstance(value, dict):
            items.append({"name": value.get("name", f"part_{key}"), **value})
    if not items:
        return None
    reference = {
        "E": 5.0e6,
        "nu": 0.35,
        "density": 2500.0,
        "score": _elastic_stability_score(5.0e6, 0.35, 2500.0),
    }
    scored = []
    for item in items:
        try:
            E = float(item.get("E", config.get("E", 1.0e5)))
            nu = float(item.get("nu", config.get("nu", 0.35)))
            density = float(item.get("density", config.get("density", 1000.0)))
        except Exception:
            continue
        scored.append({"name": str(item.get("name", "part")), "E": E, "nu": nu, "density": density, "score": _elastic_stability_score(E, nu, density)})
    if not scored:
        return None
    max_item = max(scored, key=lambda item: item["score"])
    metadata = {
        "enabled": True,
        "adjusted": False,
        "reference_score": reference["score"],
        "max_score": max_item["score"],
        "max_part": max_item["name"],
        "max_E": max_item["E"],
        "max_nu": max_item["nu"],
        "max_density": max_item["density"],
    }
    try:
        frame_dt = float(config["frame_dt"])
        base_substep_dt = float(config["substep_dt"])
    except Exception:
        metadata["reason"] = "missing_frame_or_substep_dt"
        config.setdefault("source_multiview_solver_stability", metadata)
        return metadata
    if max_item["score"] <= reference["score"]:
        metadata["reason"] = "within_reference_stability"
        config.setdefault("source_multiview_solver_stability", metadata)
        return metadata
    safety = 0.70
    target_dt = base_substep_dt * safety * np.sqrt(reference["score"] / max_item["score"])
    base_steps = max(1, int(np.ceil(frame_dt / base_substep_dt)))
    target_steps = max(base_steps, int(np.ceil(frame_dt / target_dt)))
    min_dt = 5.0e-5
    max_steps_by_min_dt = int(np.floor(frame_dt / min_dt))
    max_steps = max(base_steps, max_steps_by_min_dt)
    step_per_frame = min(target_steps, max_steps)
    new_substep_dt = frame_dt / float(step_per_frame)
    if new_substep_dt < base_substep_dt:
        config["substep_dt"] = float(new_substep_dt)
        metadata.update(
            {
                "adjusted": True,
                "original_substep_dt": base_substep_dt,
                "new_substep_dt": float(new_substep_dt),
                "frame_dt": frame_dt,
                "step_per_frame": int(step_per_frame),
                "target_substep_dt": float(target_dt),
                "reason": "merged_per_particle_materials_require_smaller_explicit_mpm_dt",
            }
        )
    else:
        metadata["reason"] = "computed_step_not_smaller_than_base"
    config["source_multiview_solver_stability"] = metadata
    return metadata

def _write_sim_config(template_config: str | Path, output_path: Path, manifest: dict[str, Any], object_rows: list[dict[str, Any]], scene_ply_path: Path | None = None, part_materials_json: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    with Path(template_config).open("r", encoding="utf-8") as f:
        config = json.load(f)
    physics = _global_physics(manifest, object_rows)
    config["material"] = physics["solver_material"]
    config["E"] = float(physics["E"])
    config["nu"] = float(physics["nu"])
    config["density"] = float(physics["density"])
    solver_stability = _apply_merged_material_stability(config, part_materials_json)
    floor_adjustment = None
    if scene_ply_path is not None:
        floor_adjustment = _apply_initial_floor_alignment(config, Path(scene_ply_path), manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return physics, floor_adjustment, solver_stability



def _run_logged_command(cmd: list[str], cwd: Path, output_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    result = {
        "returncode": int(proc.returncode),
        "command": cmd,
        "cwd": str(cwd),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    _write_json(output_dir / "run_result.json", result)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}: {' '.join(cmd)}; see {output_dir}")
    return result


def _manifest_list(manifest: dict[str, Any], key: str) -> list[str]:
    value = manifest.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _run_single_object_partphys(
    partphys_root: Path,
    manifest: dict[str, Any],
    obj: dict[str, Any],
    views_dir: Path,
    scene_name: str,
    output_root: Path,
    physgm_root: str | Path,
    physgm_config: str | Path,
    checkpoint: str | Path,
    template_config: str | Path,
    device: str,
    amp_dtype: str,
) -> tuple[Path, dict[str, Any]]:
    object_id = int(obj.get("object_id", 0))
    object_name = str(obj.get("name", f"object_{object_id:03d}"))
    safe = _safe_name(object_name)
    per_scene_name = f"{scene_name}_object_{object_id:03d}_{safe}"
    cmd = [
        sys.executable,
        str(partphys_root / "partphys_pipeline.py"),
        "--image",
        str(views_dir / FRAME_NAMES[0]),
        "--multiview-dir",
        str(views_dir),
        "--scene-name",
        per_scene_name,
        "--output-dir",
        str(output_root),
        "--physgm-root",
        str(physgm_root),
        "--physgm-config",
        str(physgm_config),
        "--checkpoint",
        str(checkpoint),
        "--template-config",
        str(template_config),
        "--device",
        str(device),
        "--amp-dtype",
        str(amp_dtype),
        "--object-mode",
        "single",
        "--assignment-mode",
        "projection",
    ]
    if bool(manifest.get("white_bg", True)):
        cmd.append("--white-bg")
    cmd.extend(_manifest_list(manifest, "partphys_extra_args"))
    if isinstance(obj.get("partphys_extra_args"), list):
        cmd.extend(str(x) for x in obj["partphys_extra_args"])
    log_dir = output_root / per_scene_name / "orchestrator_logs" / "partphys"
    result = _run_logged_command(cmd, cwd=partphys_root, output_dir=log_dir)
    return output_root / per_scene_name, result


def _run_material_agent(
    material_agent_root: Path,
    partphys_root: Path,
    manifest: dict[str, Any],
    partphys_scene: Path,
    physgm_root: str | Path,
    physgm_config: str | Path,
    checkpoint: str | Path,
    template_config: str | Path,
    device: str,
    amp_dtype: str,
) -> tuple[Path, dict[str, Any]]:
    output_dir = partphys_scene / "material_agent_multiobject"
    cmd = [
        sys.executable,
        "-m",
        "material_agent.cli",
        "--partphys-scene",
        str(partphys_scene),
        "--output-dir",
        str(output_dir),
        "--partphys-root",
        str(partphys_root),
        "--physgm-root",
        str(physgm_root),
        "--physgm-config",
        str(physgm_config),
        "--checkpoint",
        str(checkpoint),
        "--template-config",
        str(template_config),
        "--backend",
        "part_id",
        "--device",
        str(device),
        "--amp-dtype",
        str(amp_dtype),
    ]
    if bool(manifest.get("material_agent_simulate", False)):
        cmd.append("--simulate")
    if bool(manifest.get("material_agent_skip_physgm_distribution", False)):
        cmd.append("--skip-physgm-distribution")
    cmd.extend(_manifest_list(manifest, "material_agent_extra_args"))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{material_agent_root}:{partphys_root}:{physgm_root}:{env.get('PYTHONPATH', '')}"
    result = _run_logged_command(cmd, cwd=material_agent_root, output_dir=output_dir / "orchestrator_log", env=env)
    return output_dir, result


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_from_partphys_scene(
    partphys_scene: Path,
    object_id: int,
    object_name: str,
    views_dir: Path,
    placement: dict[str, Any],
    material_agent_dir: Path | None = None,
) -> dict[str, Any]:
    summary = _read_json(partphys_scene / "partphys_summary.json", {}) or {}
    whole_info = summary.get("whole_physgm") or {}
    whole_dir = partphys_scene / "physgm_whole"
    point_cloud = whole_dir / "point_clouds.ply"
    if not point_cloud.exists() and whole_info.get("point_cloud_path"):
        point_cloud = Path(whole_info["point_cloud_path"])
    if not point_cloud.exists():
        raise RuntimeError(f"Missing single-object PhysGM point cloud for {object_name}: {point_cloud}")
    predicted = whole_dir / "predicted_phys.json"
    if not predicted.exists() and whole_info.get("predicted_phys_path"):
        predicted = Path(whole_info["predicted_phys_path"])
    raw_phys = _read_json(predicted, {}) or {}
    raw_whole = whole_info.get("raw") if isinstance(whole_info.get("raw"), dict) else {}
    material = raw_phys.get("material") or whole_info.get("material") or raw_whole.get("material") or "Plastic"
    density = raw_phys.get("density") or whole_info.get("density") or raw_whole.get("density") or density_for_material(material)
    materials_json = None
    if material_agent_dir is not None:
        candidate = material_agent_dir / "selected_part_materials.json"
        if candidate.exists():
            materials_json = str(candidate)
    return {
        "object_id": int(object_id),
        "object_name": object_name,
        "views_dir": str(views_dir),
        "partphys_scene": str(partphys_scene),
        "physgm_dir": str(point_cloud.parent),
        "point_cloud_path": str(point_cloud),
        "predicted_phys_path": str(predicted) if predicted.exists() else None,
        "gaussian_part_ids_path": str(partphys_scene / "assignment" / "gaussian_part_ids.npy"),
        "selected_part_materials_path": materials_json,
        "material_agent_dir": str(material_agent_dir) if material_agent_dir else None,
        "material": material,
        "E": float(raw_phys.get("E") or whole_info.get("E") or raw_whole.get("E") or default_E_for_material(material)),
        "nu": float(raw_phys.get("nu") or whole_info.get("nu") or raw_whole.get("nu") or default_nu_for_material(material)),
        "density": float(density),
        "placement": placement,
    }


def _whole_object_material_params(row: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    visual_material = normalize_material_name(str(row.get("material", "Plastic")))
    return {
        "name": name or "whole_object_fallback",
        "material": material_to_solver_material(visual_material),
        "visual_material": visual_material,
        "E": float(row.get("E", default_E_for_material(visual_material))),
        "nu": float(row.get("nu", default_nu_for_material(visual_material))),
        "density": float(row.get("density", density_for_material(visual_material))),
    }


def _part_lacks_quantitative_material_evidence(row: dict[str, Any], local_part_id: int) -> bool:
    material_agent_dir = row.get("material_agent_dir")
    if not material_agent_dir:
        return True
    posterior_path = Path(material_agent_dir) / "part_posteriors.json"
    posterior = _read_json(posterior_path, {}) or {}
    info = posterior.get(str(local_part_id)) or posterior.get(local_part_id)
    if not isinstance(info, dict):
        return True
    return info.get("E_mean") is None or info.get("nu_mean") is None


def _repair_weak_part_material(row: dict[str, Any], local_part_id: int, params: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(params)
    selected_E = float(repaired.get("E", 0.0) or 0.0)
    whole_E = float(row.get("E", 0.0) or 0.0)
    if whole_E <= 0 or selected_E <= max(whole_E * 10.0, 1.0e7):
        return repaired
    if not _part_lacks_quantitative_material_evidence(row, local_part_id):
        return repaired

    whole = _whole_object_material_params(row, name=repaired.get("name"))
    repaired.update({k: whole[k] for k in ["material", "visual_material", "E", "nu", "density"]})
    repaired["material_repair"] = {
        "reason": "part_material_lacked_quantitative_distribution_and_exceeded_whole_object_E",
        "selected_E_before_repair": selected_E,
        "whole_object_E": whole_E,
        "whole_object_material": row.get("material"),
    }
    return repaired


def _merge_part_material_tables(object_rows: list[dict[str, Any]], part_id_map: dict[str, int], output_path: Path) -> str | None:
    fallback = None
    parts: dict[str, Any] = {}
    mapping: list[dict[str, Any]] = []
    for row in object_rows:
        materials_path = row.get("selected_part_materials_path")
        if not materials_path or not Path(materials_path).exists():
            continue
        data = _read_json(Path(materials_path), {}) or {}
        if fallback is None and isinstance(data.get("fallback"), dict):
            fallback = dict(data["fallback"])
        local_parts = data.get("parts", data)
        for local_pid_text, params in local_parts.items():
            if str(local_pid_text) == "fallback":
                continue
            try:
                local_pid = int(local_pid_text)
            except Exception:
                continue
            key = f"{int(row['object_id'])}:{local_pid}"
            if key not in part_id_map:
                continue
            global_pid = int(part_id_map[key])
            merged = _repair_weak_part_material(row, local_pid, dict(params))
            merged["name"] = f"{row['object_name']}/{merged.get('name', f'part_{local_pid}')}"
            merged["object_id"] = int(row["object_id"])
            merged["object_name"] = row["object_name"]
            merged["local_part_id"] = local_pid
            parts[str(global_pid)] = merged
            mapping.append({"object_id": int(row["object_id"]), "local_part_id": local_pid, "global_part_id": global_pid})
    if not parts:
        return None
    if object_rows:
        first_fallback = _whole_object_material_params(object_rows[0], name="fallback_global")
        if fallback is None:
            fallback = first_fallback
        else:
            fallback_E = float(fallback.get("E", 0.0) or 0.0)
            first_E = float(first_fallback.get("E", 0.0) or 0.0)
            if first_E > 0 and fallback_E > max(first_E * 10.0, 1.0e7):
                first_fallback["material_repair"] = {
                    "reason": "fallback_material_exceeded_first_whole_object_E",
                    "selected_E_before_repair": fallback_E,
                    "whole_object_E": first_E,
                }
                fallback = first_fallback
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, {"fallback": fallback, "parts": parts, "part_id_map": mapping})
    return str(output_path)



def _run_simulation(
    physgm_root: str | Path,
    scene_dir: Path,
    config_path: Path,
    output_path: Path,
    white_bg: bool,
    partphys_root: str | Path | None = None,
    part_ids_path: str | Path | None = None,
    part_materials_json: str | Path | None = None,
) -> dict[str, Any]:
    physgm_root = Path(physgm_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    use_part_materials = bool(part_ids_path and part_materials_json and Path(part_ids_path).exists() and Path(part_materials_json).exists())
    env = os.environ.copy()
    if use_part_materials:
        root = Path(partphys_root).expanduser().resolve() if partphys_root else Path(__file__).resolve().parents[1]
        script = root / "tools" / "gs_simulation_partid_materials.py"
        cmd = [
            sys.executable,
            str(script),
            "--model_path",
            str(scene_dir),
            "--output_path",
            str(output_path),
            "--config",
            str(config_path),
            "--part_ids",
            str(part_ids_path),
            "--part_materials_json",
            str(part_materials_json),
            "--render_img",
            "--compile_video",
        ]
        env["PHYSGM_ROOT"] = str(physgm_root)
        env["PYTHONPATH"] = f"{physgm_root}:{root}:{env.get('PYTHONPATH', '')}"
    else:
        cmd = [
            sys.executable,
            "gs_simulation.py",
            "--model_path",
            str(scene_dir),
            "--output_path",
            str(output_path),
            "--config",
            str(config_path),
            "--render_img",
            "--compile_video",
        ]
    if white_bg:
        cmd.append("--white_bg")
    (output_path / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, cwd=str(physgm_root), env=env, text=True, capture_output=True)
    (output_path / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    (output_path / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    result = {
        "returncode": int(proc.returncode),
        "command": cmd,
        "backend": "part_id" if use_part_materials else "global",
        "part_ids_path": str(part_ids_path) if use_part_materials else None,
        "part_materials_json": str(part_materials_json) if use_part_materials else None,
        "output_path": str(output_path),
        "video_path": str(output_path / "output.mp4") if (output_path / "output.mp4").exists() else None,
    }
    _write_json(output_path / "run_result.json", result)
    if proc.returncode != 0:
        raise RuntimeError(f"multi-object simulation failed with return code {proc.returncode}; see {output_path}")
    return result


def run_source_multiview_scene(
    manifest_path: str | Path,
    output_dir: str | Path,
    scene_name: str,
    physgm_config: str | Path,
    checkpoint: str | Path,
    template_config: str | Path,
    physgm_root: str | Path,
    device: str = "cuda",
    amp_dtype: str = "fp16",
    simulate: bool = False,
    white_bg: bool = False,
    max_vertices_per_object: int | None = None,
    partphys_root: str | Path = "/root/PartPhysAgent",
    material_agent_root: str | Path = "/root/MaterialAgent",
    object_flow: str = "partphys_material",
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    scene_dir = Path(output_dir).expanduser().resolve() / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    _write_json(scene_dir / "source_multiview_manifest.json", manifest)

    partphys_root_path = Path(partphys_root).expanduser().resolve()
    material_agent_root_path = Path(material_agent_root).expanduser().resolve()
    flow = str(manifest.get("object_flow", object_flow)).lower()
    if flow not in {"physgm_only", "partphys", "partphys_material"}:
        raise ValueError("object_flow must be one of: physgm_only, partphys, partphys_material")

    runner = None
    if flow == "physgm_only":
        runner = PhysGMRunner(
            config_path=str(physgm_config),
            checkpoint_path=str(checkpoint),
            template_config_path=str(template_config),
            device=device,
            output_base_dir=str(scene_dir / "physgm_objects"),
            mock=False,
            save_gaussian_default=True,
            physgm_root=str(physgm_root),
            amp_dtype=amp_dtype,
        )

    object_rows: list[dict[str, Any]] = []
    for idx, obj in enumerate(manifest["objects"]):
        views_dir = _check_source_views(obj["views_dir"])
        object_id = int(obj.get("object_id", idx))
        object_name = str(obj.get("name", f"object_{object_id:03d}"))
        placement = _placement_from_object(obj)
        if flow in {"partphys", "partphys_material"}:
            partphys_scene, partphys_run = _run_single_object_partphys(
                partphys_root=partphys_root_path,
                manifest=manifest,
                obj=obj,
                views_dir=views_dir,
                scene_name=scene_name,
                output_root=scene_dir / "per_object_partphys",
                physgm_root=physgm_root,
                physgm_config=physgm_config,
                checkpoint=checkpoint,
                template_config=template_config,
                device=device,
                amp_dtype=amp_dtype,
            )
            material_agent_dir = None
            material_agent_run = None
            if flow == "partphys_material" and not bool(manifest.get("skip_material_agent", False)):
                material_agent_dir, material_agent_run = _run_material_agent(
                    material_agent_root=material_agent_root_path,
                    partphys_root=partphys_root_path,
                    manifest=manifest,
                    partphys_scene=partphys_scene,
                    physgm_root=physgm_root,
                    physgm_config=physgm_config,
                    checkpoint=checkpoint,
                    template_config=template_config,
                    device=device,
                    amp_dtype=amp_dtype,
                )
            row = _row_from_partphys_scene(
                partphys_scene=partphys_scene,
                object_id=object_id,
                object_name=object_name,
                views_dir=views_dir,
                placement=placement,
                material_agent_dir=material_agent_dir,
            )
            row["partphys_run"] = partphys_run
            row["material_agent_run"] = material_agent_run
        else:
            assert runner is not None
            safe = _safe_name(object_name)
            obj_out = scene_dir / "physgm_objects" / f"object_{object_id:03d}_{safe}"
            result: PhysGMResult = runner.infer_image(
                str(views_dir / FRAME_NAMES[0]),
                scene_name=f"{scene_name}_object_{object_id:03d}_{safe}",
                output_dir=obj_out,
                save_gaussian=True,
                use_mvadapter=False,
                multiview_dir=str(views_dir),
            )
            if not result.point_cloud_path or not Path(result.point_cloud_path).exists():
                raise RuntimeError(f"PhysGM did not produce point_clouds.ply for object {object_id}:{object_name}")
            row = {
                "object_id": object_id,
                "object_name": object_name,
                "views_dir": str(views_dir),
                "physgm_dir": result.scene_dir,
                "point_cloud_path": result.point_cloud_path,
                "predicted_phys_path": result.predicted_phys_path,
                "material": result.material,
                "E": result.E,
                "nu": result.nu,
                "density": result.density,
                "placement": placement,
            }
        object_rows.append(row)

    counts, object_aabbs, object_ids, global_part_ids, part_id_map = merge_source_object_plys(
        object_rows,
        scene_dir / "point_clouds.ply",
        max_vertices_per_object=max_vertices_per_object,
    )
    for row, count in zip(object_rows, counts):
        row["gaussian_count"] = int(count)
    np.save(scene_dir / "gaussian_object_ids_direct.npy", object_ids)
    global_part_ids_path = None
    if global_part_ids is not None:
        global_part_ids_path = scene_dir / "global_gaussian_part_ids.npy"
        np.save(global_part_ids_path, global_part_ids)
    merged_part_materials_path = None
    if global_part_ids_path is not None:
        merged_part_materials_path = _merge_part_material_tables(
            object_rows,
            part_id_map,
            scene_dir / "merged_part_materials.json",
        )
    _write_json(scene_dir / "object_aabbs.json", object_aabbs)
    _write_json(scene_dir / "global_part_id_map.json", part_id_map)

    physics, floor_adjustment, solver_stability = _write_sim_config(
        template_config,
        scene_dir / "simulation" / "sim_config_source_multiview.json",
        manifest,
        object_rows,
        scene_dir / "point_clouds.ply",
        merged_part_materials_path,
    )
    raw = {
        "mode": "source_multiview_multi_object_after_single_object_agents" if flow != "physgm_only" else "source_multiview_multi_object",
        "object_flow": flow,
        "geometry_source": "per_object_partphys_whole_physgm" if flow != "physgm_only" else "per_object_physgm_from_source_views",
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "object_results": object_rows,
        "object_aabbs": object_aabbs,
        "object_gaussian_counts": {str(row["object_id"]): int(row["gaussian_count"]) for row in object_rows},
        "gaussian_object_ids_direct": str(scene_dir / "gaussian_object_ids_direct.npy"),
        "global_gaussian_part_ids": str(global_part_ids_path) if global_part_ids_path else None,
        "global_part_id_map": part_id_map,
        "merged_part_materials": merged_part_materials_path,
        "global_physics": physics,
        "simulation_initial_floor_alignment": floor_adjustment,
        "simulation_solver_stability": solver_stability,
        "simulation_config": str(scene_dir / "simulation" / "sim_config_source_multiview.json"),
    }
    _write_json(scene_dir / "predicted_phys.json", raw)
    _write_json(scene_dir / "raw_output_summary.json", raw)

    sim_result = None
    if simulate:
        sim_result = _run_simulation(
            physgm_root=physgm_root,
            scene_dir=scene_dir,
            config_path=scene_dir / "simulation" / "sim_config_source_multiview.json",
            output_path=scene_dir / "simulation",
            white_bg=white_bg,
            partphys_root=partphys_root_path,
            part_ids_path=global_part_ids_path,
            part_materials_json=merged_part_materials_path,
        )
        raw["simulation_result"] = sim_result
        _write_json(scene_dir / "raw_output_summary.json", raw)
        _write_json(scene_dir / "predicted_phys.json", raw)
    return raw
