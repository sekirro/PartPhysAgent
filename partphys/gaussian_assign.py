from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import read_mask
from .types import PartInstance


def load_ply_positions(ply_path) -> np.ndarray:
    path = Path(ply_path)
    try:
        from plyfile import PlyData  # type: ignore

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    except ImportError:
        pass
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        fmt = next((x for x in header_lines if x.startswith("format ")), "")
        if "ascii" not in fmt:
            raise RuntimeError("Binary PLY requires plyfile. Install plyfile or use ASCII PLY.")
        count = 0
        props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                props.append(line.split()[-1])
        data = []
        for _ in range(count):
            values = f.readline().decode("ascii").strip().split()
            if not values:
                continue
            row = {name: float(values[i]) for i, name in enumerate(props[: len(values)])}
            data.append([row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)])
    return np.asarray(data, dtype=np.float64)


def assign_by_projection(positions, part_masks, camera_meta_npz, image_size) -> dict[str, Any]:
    warnings: list[str] = []
    positions = np.asarray(positions, dtype=np.float64)
    ids = np.full(len(positions), -1, dtype=np.int32)
    if len(positions) == 0:
        return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": ["No Gaussian positions."]}
    if not camera_meta_npz or not Path(camera_meta_npz).exists():
        return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": ["Projection metadata unavailable."]}
    try:
        meta = np.load(camera_meta_npz)
        intr = np.asarray(meta["input_intr"])
        c2ws = np.asarray(meta["input_c2ws"])
        while intr.ndim > 2:
            intr = intr[0]
        while c2ws.ndim > 3:
            c2ws = c2ws[0]
        fx, fy, cx, cy = [float(x) for x in intr[0]]
        c2w = c2ws[0]
        w2c = np.linalg.inv(c2w)
        homo = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        cam = (w2c @ homo.T).T[:, :3]
        valid = cam[:, 2] > 1e-6
        u = fx * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + cx
        v = fy * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + cy
        width, height = image_size
        u_i = np.round(u).astype(np.int64)
        v_i = np.round(v).astype(np.int64)
        for part in sorted(part_masks, key=lambda item: item["area"], reverse=True):
            mask = read_mask(part["mask_path"]) if isinstance(part["mask_path"], str) else np.asarray(part["mask_path"], dtype=bool)
            mh, mw = mask.shape
            uu = np.clip((u_i * mw / max(1, width)).astype(np.int64), 0, mw - 1)
            vv = np.clip((v_i * mh / max(1, height)).astype(np.int64), 0, mh - 1)
            hit = valid & (u_i >= 0) & (u_i < width) & (v_i >= 0) & (v_i < height) & mask[vv, uu]
            ids[hit] = int(part["part_id"])
        assigned = ids >= 0
        counts = {str(pid): int((ids == int(pid)).sum()) for pid in np.unique(ids[assigned])}
        return {"gaussian_part_ids": ids, "assigned_ratio": float(assigned.mean()), "per_part_counts": counts, "warnings": warnings}
    except Exception as exc:
        return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": [f"Projection assignment failed: {exc}"]}


def assign_by_aabb_heuristic(positions, part_instances: list[PartInstance], image_size) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.float64)
    ids = np.full(len(positions), -1, dtype=np.int32)
    if len(positions) == 0:
        return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": ["No Gaussian positions."]}
    width, height = image_size
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    norm = (positions - mins) / span
    warnings = ["Used approximate AABB heuristic for 2D mask to 3D assignment."]
    ordered = sorted(part_instances, key=lambda p: p.area, reverse=True)
    for part in ordered:
        bx1 = part.bbox.x1 / max(1, width)
        bx2 = part.bbox.x2 / max(1, width)
        by1 = part.bbox.y1 / max(1, height)
        by2 = part.bbox.y2 / max(1, height)
        x_lo = max(0.0, bx1 - 0.08)
        x_hi = min(1.0, bx2 + 0.08)
        y_lo = max(0.0, 1.0 - by2 - 0.08)
        y_hi = min(1.0, 1.0 - by1 + 0.08)
        loc = str(part.metadata.get("part_spec", {}).get("location", "")).lower()
        if "top" in loc or "upper" in loc:
            y_lo = max(y_lo, 0.45)
        if "bottom" in loc or "lower" in loc or "sole" in loc:
            y_hi = min(y_hi, 0.55)
        if "left" in loc:
            x_hi = min(x_hi, 0.65)
        if "right" in loc:
            x_lo = max(x_lo, 0.35)
        hit = (norm[:, 0] >= x_lo) & (norm[:, 0] <= x_hi) & (norm[:, 1] >= y_lo) & (norm[:, 1] <= y_hi)
        ids[hit] = part.part_id
    if np.any(ids < 0) and part_instances:
        largest = max(part_instances, key=lambda p: p.area)
        ids[ids < 0] = largest.part_id
    assigned = ids >= 0
    counts = {str(pid): int((ids == int(pid)).sum()) for pid in np.unique(ids[assigned])}
    return {"gaussian_part_ids": ids, "assigned_ratio": float(assigned.mean()), "per_part_counts": counts, "warnings": warnings}


def _normalize_to_mpm_space(positions: np.ndarray) -> np.ndarray:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    return 0.4 + (positions - mins) / span * 1.2


def build_part_aabbs(
    positions,
    gaussian_part_ids,
    part_instances: list[PartInstance],
    min_count: int = 20,
    padding_ratio: float = 0.15,
    min_half_size: float = 0.02,
) -> list[dict[str, Any]]:
    positions = np.asarray(positions, dtype=np.float64)
    ids = np.asarray(gaussian_part_ids)
    if len(positions) == 0:
        return []
    mpm_pos = _normalize_to_mpm_space(positions)
    out = []
    for part in part_instances:
        pts = mpm_pos[ids == part.part_id]
        if len(pts) < min_count:
            continue
        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        half = (max_xyz - min_xyz) / 2.0
        half = np.maximum(half * (1.0 + padding_ratio), float(min_half_size))
        out.append(
            {
                "part_id": int(part.part_id),
                "part_name": part.name,
                "count": int(len(pts)),
                "center": center.tolist(),
                "half_size": half.tolist(),
                "min_xyz": min_xyz.tolist(),
                "max_xyz": max_xyz.tolist(),
                "coordinate_space": "mpm_normalized_approx",
            }
        )
    return out


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "part"


def _write_ascii_part_ply(path: Path, positions: np.ndarray, indices: np.ndarray):
    pts = positions[indices]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def _write_filtered_part_ply(source_ply_path, path: Path, indices: np.ndarray, fallback_positions: np.ndarray):
    try:
        from plyfile import PlyData, PlyElement  # type: ignore

        ply = PlyData.read(str(source_ply_path))
        vertex = ply["vertex"].data
        subset = vertex[indices]
        PlyData([PlyElement.describe(subset, "vertex")], text=ply.text).write(str(path))
    except Exception:
        _write_ascii_part_ply(path, fallback_positions, indices)


def save_part_gaussian_outputs(output_dir, gaussian_part_ids, part_instances=None, source_ply_path=None):
    output_dir = Path(output_dir)
    ids = np.asarray(gaussian_part_ids, dtype=np.int32)
    index_path = output_dir / "part_gaussian_index.json"
    ply_dir = output_dir / "per_part_gaussians"
    ply_dir.mkdir(parents=True, exist_ok=True)
    positions = None
    if source_ply_path and Path(source_ply_path).exists():
        try:
            positions = load_ply_positions(source_ply_path)
        except Exception:
            positions = None
    part_by_id = {int(p.part_id): p for p in (part_instances or [])}
    part_ids = sorted(int(x) for x in np.unique(ids[ids >= 0]))
    index = {"parts": [], "unassigned_count": int((ids < 0).sum())}
    for pid in part_ids:
        part = part_by_id.get(pid)
        name = part.name if part is not None else f"part_{pid}"
        indices = np.where(ids == pid)[0].astype(np.int64)
        stem = f"part_{pid:03d}_{_safe_name(name)}"
        indices_path = output_dir / f"{stem}_indices.npy"
        np.save(indices_path, indices)
        ply_path = ply_dir / f"{stem}.ply"
        if positions is not None and len(indices) > 0:
            _write_filtered_part_ply(source_ply_path, ply_path, indices, positions)
            ply_value = str(ply_path)
        else:
            ply_value = None
        index["parts"].append(
            {
                "part_id": pid,
                "part_name": name,
                "count": int(len(indices)),
                "indices_path": str(indices_path),
                "ply_path": ply_value,
            }
        )
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index


def save_assignment_outputs(output_dir, gaussian_part_ids, part_aabbs, summary, part_instances=None, source_ply_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gaussian_part_ids.npy", np.asarray(gaussian_part_ids, dtype=np.int32))
    with open(output_dir / "per_part_aabb.json", "w", encoding="utf-8") as f:
        json.dump(part_aabbs, f, indent=2)
    part_index = save_part_gaussian_outputs(output_dir, gaussian_part_ids, part_instances, source_ply_path)
    summary = dict(summary)
    summary["part_gaussian_index"] = str(output_dir / "part_gaussian_index.json")
    summary["per_part_gaussians_dir"] = str(output_dir / "per_part_gaussians")
    summary["per_part_gaussian_counts"] = {str(item["part_id"]): int(item["count"]) for item in part_index.get("parts", [])}
    with open(output_dir / "assignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
