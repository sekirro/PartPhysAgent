from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .material_table import density_for_material, default_E_for_material, default_nu_for_material, material_to_solver_material, normalize_material_name
from .types import PhysicsParams, PhysGMResult, PartInstance



def _solver_safe_global_values(material: str, E: float, nu: float, density: float, warnings: list[str]):
    solver_material = material_to_solver_material(material)
    safe_E = float(E)
    safe_nu = float(nu)
    safe_density = float(density)
    return solver_material, safe_E, safe_nu, safe_density


def _solver_safe_local_values(part_name: str, E: float, nu: float, density: float, warnings: list[str]):
    return float(E), float(nu), float(density)


def _stabilize_time_step(config: dict[str, Any], E: float, warnings: list[str]) -> None:
    if not math.isfinite(E) or E <= 0:
        return
    current_dt = float(config.get("substep_dt", 2e-4))
    reference_E = 2.5e6
    target_dt = current_dt * math.sqrt(reference_E / max(float(E), reference_E))
    if target_dt < current_dt:
        config["substep_dt"] = float(target_dt)
        warnings.append(f"Simulation substep_dt adjusted from {current_dt:g} to {target_dt:g} for E={E:g}.")


def _phys_values(phys: PhysicsParams | PhysGMResult | None):
    if phys is None:
        material = "Plastic"
        return material, default_E_for_material(material), default_nu_for_material(material), density_for_material(material)
    material = normalize_material_name(getattr(phys, "material", "Plastic"))
    E = float(getattr(phys, "E", default_E_for_material(material)) or default_E_for_material(material))
    nu = float(getattr(phys, "nu", default_nu_for_material(material)) or default_nu_for_material(material))
    density = getattr(phys, "density", None)
    if density is None:
        density = density_for_material(material)
    return material, E, nu, float(density)


def _sanitize_global_physics(material: str, E: float, nu: float, density: float, warnings: list[str]):
    if not math.isfinite(E) or E <= 0:
        warnings.append(f"Invalid global E replaced with {default_E_for_material(material):g} for {material}.")
        E = default_E_for_material(material)
    if not math.isfinite(nu) or not (0 <= nu < 0.5):
        replacement = min(default_nu_for_material(material), 0.49)
        warnings.append(f"Invalid global nu {nu:g} replaced with {replacement:g} for {material}.")
        nu = replacement
    if not math.isfinite(density) or density <= 0:
        warnings.append(f"Invalid global density replaced with {density_for_material(material):g} for {material}.")
        density = density_for_material(material)
    return material, float(E), float(nu), float(density)





def _valid_param(E: float, nu: float, density: float, point, size) -> bool:
    return (
        math.isfinite(E)
        and E > 0
        and math.isfinite(nu)
        and 0 <= nu < 0.5
        and math.isfinite(density)
        and density > 0
        and len(point) == 3
        and len(size) == 3
        and all(float(x) > 0 for x in size)
    )


def build_part_aware_sim_config(
    template_config_path,
    output_config_path,
    whole_physics: PhysicsParams | PhysGMResult | None,
    part_instances: list[PartInstance],
    part_physics: dict[int, PhysicsParams],
    part_aabbs: list[dict[str, Any]],
    global_policy: str = "whole_or_dominant",
):
    warnings: list[str] = []
    with open(template_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    material, E, nu, density = _phys_values(whole_physics)
    material, E, nu, density = _sanitize_global_physics(material, E, nu, density, warnings)
    if E <= 0 or not (0 <= nu < 0.5) or density <= 0:
        largest = max(part_instances, key=lambda p: p.area, default=None)
        if largest is not None and largest.part_id in part_physics:
            material, E, nu, density = _phys_values(part_physics[largest.part_id])
            warnings.append("Whole-object physics invalid; used largest part as global base.")
        else:
            material = "Plastic"
            E, nu, density = default_E_for_material(material), default_nu_for_material(material), density_for_material(material)
            warnings.append("Whole-object physics invalid; used Plastic defaults.")
    solver_material, sim_E, sim_nu, sim_density = _solver_safe_global_values(material, E, nu, density, warnings)
    config["material"] = solver_material
    config["E"] = float(sim_E)
    config["nu"] = float(sim_nu)
    config["density"] = float(sim_density)
    _stabilize_time_step(config, sim_E, warnings)

    part_by_id = {p.part_id: p for p in part_instances}
    additional = []
    metadata = []
    for aabb in part_aabbs:
        pid = int(aabb["part_id"])
        phys = part_physics.get(pid)
        if phys is None:
            warnings.append(f"Skipping part {pid}: no physics params.")
            continue
        _, pE, pnu, pdensity = _phys_values(phys)
        point = [float(x) for x in aabb.get("center", [])]
        size = [float(x) for x in aabb.get("half_size", [])]
        if not _valid_param(pE, pnu, pdensity, point, size):
            warnings.append(f"Skipping part {pid}: invalid local material params.")
            continue
        part = part_by_id.get(pid)
        part_name = part.name if part else aabb.get("part_name", str(pid))
        sim_pE, sim_pnu, sim_pdensity = _solver_safe_local_values(part_name, pE, pnu, pdensity, warnings)
        additional.append({"point": point, "size": size, "E": float(sim_pE), "nu": float(sim_pnu), "density": float(sim_pdensity)})
        metadata.append(
            {
                "part_id": pid,
                "part_name": part_name,
                "material_label": phys.material,
                "confidence": phys.confidence,
                "point": point,
                "size": size,
                "count": aabb.get("count"),
                "coordinate_space": aabb.get("coordinate_space", "unknown"),
                "raw_E": float(pE),
                "raw_nu": float(pnu),
                "raw_density": float(pdensity),
                "simulation_E": float(sim_pE),
                "simulation_nu": float(sim_pnu),
                "simulation_density": float(sim_pdensity),
            }
        )
    config["additional_material_params"] = additional
    output_config_path = Path(output_config_path)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    metadata_path = output_config_path.parent / "part_aabb_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"parts": metadata, "warnings": warnings}, f, indent=2)
    return config, warnings
