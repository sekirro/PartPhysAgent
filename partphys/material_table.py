from __future__ import annotations

import math


PHYSGM_MATERIALS = [
    "Wood",
    "Metal",
    "Plastic",
    "Glass",
    "Fabric",
    "Leather",
    "Ceramic",
    "Stone",
    "Rubber",
    "Paper",
    "Sand",
    "Snow",
    "Plasticine",
    "Foam",
]

MATERIAL_TO_DENSITY = {
    "Wood": 700.0,
    "Metal": 7800.0,
    "Plastic": 1200.0,
    "Glass": 2500.0,
    "Fabric": 500.0,
    "Leather": 900.0,
    "Ceramic": 2500.0,
    "Stone": 2600.0,
    "Rubber": 1100.0,
    "Paper": 800.0,
    "Sand": 1600.0,
    "Snow": 300.0,
    "Plasticine": 2000.0,
    "Foam": 100.0,
}

MATERIAL_TO_E_RANGE = {
    "Wood": (1e8, 2e10),
    "Metal": (1e9, 3e11),
    "Plastic": (1e6, 5e9),
    "Glass": (1e9, 1e11),
    "Fabric": (1e4, 1e8),
    "Leather": (1e5, 1e9),
    "Ceramic": (1e8, 1e11),
    "Stone": (1e8, 1e11),
    "Rubber": (1e4, 1e8),
    "Paper": (1e5, 1e9),
    "Sand": (1e3, 1e7),
    "Snow": (1e3, 1e7),
    "Plasticine": (1e3, 1e7),
    "Foam": (1e3, 1e7),
}

MATERIAL_TO_NU_RANGE = {
    "Wood": (0.25, 0.45),
    "Metal": (0.20, 0.35),
    "Plastic": (0.30, 0.45),
    "Glass": (0.18, 0.30),
    "Fabric": (0.20, 0.45),
    "Leather": (0.30, 0.49),
    "Ceramic": (0.15, 0.35),
    "Stone": (0.10, 0.35),
    "Rubber": (0.40, 0.499),
    "Paper": (0.20, 0.45),
    "Sand": (0.20, 0.45),
    "Snow": (0.10, 0.35),
    "Plasticine": (0.25, 0.49),
    "Foam": (0.10, 0.45),
}

MATERIAL_TO_SOLVER_MATERIAL = {
    "Wood": "metal",
    "Metal": "metal",
    "Plastic": "metal",
    "Glass": "metal",
    "Fabric": "foam",
    "Leather": "foam",
    "Ceramic": "metal",
    "Stone": "metal",
    "Rubber": "jelly",
    "Paper": "foam",
    "Sand": "sand",
    "Snow": "snow",
    "Plasticine": "plasticine",
    "Foam": "foam",
}

_ALIASES = {
    "wooden": "Wood",
    "timber": "Wood",
    "steel": "Metal",
    "iron": "Metal",
    "aluminum": "Metal",
    "aluminium": "Metal",
    "metallic": "Metal",
    "plasticine": "Plasticine",
    "clay": "Plasticine",
    "rubbery": "Rubber",
    "cloth": "Fabric",
    "textile": "Fabric",
    "paperboard": "Paper",
    "cardboard": "Paper",
    "ceramics": "Ceramic",
}


def normalize_material_name(name: str | None) -> str:
    if not name:
        return "Plastic"
    text = str(name).strip()
    if not text:
        return "Plastic"
    lower = text.lower().replace("_", " ").replace("-", " ").strip()
    if lower in _ALIASES:
        return _ALIASES[lower]
    for material in PHYSGM_MATERIALS:
        if lower == material.lower():
            return material
    for material in PHYSGM_MATERIALS:
        if material.lower() in lower:
            return material
    return "Plastic"


def density_for_material(material: str | None) -> float:
    return MATERIAL_TO_DENSITY[normalize_material_name(material)]


def default_E_for_material(material: str | None) -> float:
    lo, hi = MATERIAL_TO_E_RANGE[normalize_material_name(material)]
    return math.sqrt(lo * hi)


def default_nu_for_material(material: str | None) -> float:
    lo, hi = MATERIAL_TO_NU_RANGE[normalize_material_name(material)]
    return (lo + hi) * 0.5


def clamp_physics_to_material(material: str | None, E: float, nu: float):
    material = normalize_material_name(material)
    warnings: list[str] = []
    e_lo, e_hi = MATERIAL_TO_E_RANGE[material]
    nu_lo, nu_hi = MATERIAL_TO_NU_RANGE[material]
    e_value = float(E)
    nu_value = float(nu)
    if not math.isfinite(e_value) or e_value <= 0:
        e_value = default_E_for_material(material)
        warnings.append(f"Invalid E replaced with {e_value:g} for {material}.")
    if e_value < e_lo:
        warnings.append(f"E clamped from {E:g} to {e_lo:g} for {material}.")
        e_value = e_lo
    elif e_value > e_hi:
        warnings.append(f"E clamped from {E:g} to {e_hi:g} for {material}.")
        e_value = e_hi
    if not math.isfinite(nu_value):
        nu_value = default_nu_for_material(material)
        warnings.append(f"Invalid nu replaced with {nu_value:g} for {material}.")
    if nu_value < nu_lo:
        warnings.append(f"nu clamped from {nu:g} to {nu_lo:g} for {material}.")
        nu_value = nu_lo
    elif nu_value > nu_hi:
        warnings.append(f"nu clamped from {nu:g} to {nu_hi:g} for {material}.")
        nu_value = nu_hi
    return e_value, nu_value, warnings


def material_to_solver_material(material: str | None) -> str:
    return MATERIAL_TO_SOLVER_MATERIAL[normalize_material_name(material)]
