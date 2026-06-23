from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import read_mask
from .scene_builder import VIEW_LABELS
from .types import PartInstance


PLY_NUMPY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


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
        count = 0
        scalar_props = []
        props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                tokens = line.split()
                if len(tokens) >= 3 and tokens[1] != "list":
                    scalar_props.append((tokens[2], tokens[1]))
                    props.append(tokens[-1])
                elif len(tokens) >= 5 and tokens[1] == "list":
                    scalar_props.append((tokens[-1], "list"))
        if "ascii" not in fmt:
            endian = "<" if "binary_little_endian" in fmt else ">" if "binary_big_endian" in fmt else None
            if endian is None:
                raise RuntimeError(f"Unsupported PLY format: {fmt}")
            dtype_fields = []
            for name, type_name in scalar_props:
                if type_name == "list":
                    raise RuntimeError("Binary PLY vertex list properties require plyfile.")
                np_type = PLY_NUMPY_TYPES.get(type_name)
                if not np_type:
                    raise RuntimeError(f"Unsupported binary PLY property type: {type_name}")
                dtype_fields.append((name, np.dtype(endian + np_type)))
            dtype = np.dtype(dtype_fields)
            data_bytes = f.read(dtype.itemsize * count)
            vertex = np.frombuffer(data_bytes, dtype=dtype, count=count)
            if not {"x", "y", "z"}.issubset(vertex.dtype.names or ()):
                raise RuntimeError("PLY vertex data has no x/y/z fields.")
            return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
        data = []
        for _ in range(count):
            values = f.readline().decode("ascii").strip().split()
            if not values:
                continue
            row = {name: float(values[i]) for i, name in enumerate(props[: len(values)])}
            data.append([row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)])
    return np.asarray(data, dtype=np.float64)


def _normalize_c2ws(c2ws: np.ndarray) -> np.ndarray:
    c2ws = np.asarray(c2ws, dtype=np.float64)
    while c2ws.ndim > 3:
        c2ws = c2ws[0]
    if c2ws.ndim == 2:
        c2ws = c2ws[None, :, :]
    return c2ws


def _intrinsics_for_view(intr: np.ndarray, view_idx: int) -> tuple[float, float, float, float]:
    intr = np.asarray(intr, dtype=np.float64)
    while intr.ndim > 3:
        intr = intr[0]
    if intr.ndim == 3:
        item = intr[min(view_idx, intr.shape[0] - 1)]
    elif intr.ndim == 2 and intr.shape == (3, 3):
        item = intr
    elif intr.ndim == 2 and intr.shape[1] >= 4:
        item = intr[min(view_idx, intr.shape[0] - 1)]
    elif intr.ndim == 1:
        item = intr
    else:
        item = np.ravel(intr)
    item = np.asarray(item, dtype=np.float64)
    if item.ndim == 2 and item.shape[0] >= 3 and item.shape[1] >= 3:
        return float(item[0, 0]), float(item[1, 1]), float(item[0, 2]), float(item[1, 2])
    flat = item.reshape(-1)
    if flat.size < 4:
        raise ValueError(f"Unsupported intrinsics shape: {intr.shape}")
    return float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])


def _meta_scalar(meta, key: str, default: int) -> int:
    if key not in meta.files:
        return int(default)
    values = np.asarray(meta[key]).reshape(-1)
    if values.size == 0:
        return int(default)
    return int(values[0])


def _view_mask_value(part: dict[str, Any], label: str, view_idx: int):
    view_masks = part.get("view_masks") or {}
    if isinstance(view_masks, dict) and view_masks.get(label):
        return view_masks[label]
    if view_idx == 0:
        return part.get("mask_path")
    return None


def _read_mask_cached(value, cache: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(value, str):
        if value not in cache:
            cache[value] = read_mask(value)
        return cache[value]
    return np.asarray(value, dtype=bool)


def _mask_weight_map(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    try:
        import cv2  # type: ignore

        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        positive = dist[mask]
        if positive.size:
            scale = float(np.percentile(positive, 90))
        else:
            scale = 1.0
        norm = np.clip(dist / max(scale, 1e-6), 0.0, 1.0)
        weights = 0.45 + 0.55 * norm
        weights[~mask] = 0.0
        return weights.astype(np.float32)
    except Exception:
        return mask.astype(np.float32)


def _mask_weight_cached(value, mask_cache: dict[str, np.ndarray], weight_cache: dict[str, np.ndarray]) -> np.ndarray:
    key = value if isinstance(value, str) else str(id(value))
    if key not in weight_cache:
        weight_cache[key] = _mask_weight_map(_read_mask_cached(value, mask_cache))
    return weight_cache[key]


def _is_residual_part(part: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(part.get("name", "")),
            str(part.get("part_name", "")),
            str(part.get("physics_group", "")),
        ]
    ).lower()
    return "unknown" in text or "residual" in text


def _smooth_low_confidence_assignments(
    positions: np.ndarray,
    ids: np.ndarray,
    low_confidence: np.ndarray,
    high_confidence: np.ndarray,
    k: int = 12,
) -> tuple[np.ndarray, int]:
    low_idx = np.where(low_confidence)[0]
    high_idx = np.where(high_confidence & (ids >= 0))[0]
    if len(low_idx) == 0 or len(high_idx) < max(3, k // 2):
        return ids, 0
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return ids, 0

    tree = cKDTree(positions[high_idx])
    query_k = min(int(k), len(high_idx))
    _, nn = tree.query(positions[low_idx], k=query_k)
    if query_k == 1:
        nn = nn[:, None]
    neighbor_labels = ids[high_idx[nn]]
    new_ids = ids.copy()
    changed = 0
    max_label = int(max(0, ids[ids >= 0].max(initial=0)))
    for row_idx, labels in zip(low_idx, neighbor_labels):
        labels = labels[labels >= 0]
        if labels.size < 3:
            continue
        counts = np.bincount(labels.astype(np.int64), minlength=max_label + 1)
        label = int(np.argmax(counts))
        support = int(counts[label])
        if support >= max(3, int(np.ceil(0.55 * labels.size))) and label != int(new_ids[row_idx]):
            new_ids[row_idx] = label
            changed += 1
    return new_ids, changed


def _knn_label_consistency_cleanup(
    positions: np.ndarray,
    ids: np.ndarray,
    residual_part_ids: set[int],
    k: int = 16,
) -> tuple[np.ndarray, dict[str, int]]:
    assigned_idx = np.where(ids >= 0)[0]
    if len(assigned_idx) < max(8, k):
        return ids, {"unknown_reassigned": 0, "island_reassigned": 0}
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return ids, {"unknown_reassigned": 0, "island_reassigned": 0}

    tree = cKDTree(positions[assigned_idx])
    query_k = min(int(k) + 1, len(assigned_idx))
    _, nn = tree.query(positions[assigned_idx], k=query_k)
    if query_k == 1:
        nn = nn[:, None]
    neighbor_ids = ids[assigned_idx[nn]]
    max_label = int(max(0, ids[ids >= 0].max(initial=0)))
    new_ids = ids.copy()
    unknown_reassigned = 0
    island_reassigned = 0

    for row_pos, row in zip(assigned_idx, neighbor_ids):
        current = int(ids[row_pos])
        labels = row[1:] if row.size > 1 else row
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        non_residual = np.asarray([x for x in labels if int(x) not in residual_part_ids], dtype=np.int64)
        if non_residual.size < 4:
            continue
        counts = np.bincount(non_residual, minlength=max_label + 1)
        majority = int(np.argmax(counts))
        majority_count = int(counts[majority])
        majority_frac = majority_count / max(1, int(non_residual.size))

        if current in residual_part_ids:
            if majority_frac >= 0.50 and majority_count >= 5:
                new_ids[row_pos] = majority
                unknown_reassigned += 1
            continue

        own_count = int(np.sum(non_residual == current))
        own_frac = own_count / max(1, int(non_residual.size))
        if majority != current and own_frac <= 0.18 and majority_frac >= 0.68 and majority_count >= 7:
            new_ids[row_pos] = majority
            island_reassigned += 1

    return new_ids, {"unknown_reassigned": unknown_reassigned, "island_reassigned": island_reassigned}


def assign_by_projection(positions, part_masks, camera_meta_npz, image_size) -> dict[str, Any]:
    warnings: list[str] = []
    positions = np.asarray(positions, dtype=np.float64)
    ids = np.full(len(positions), -1, dtype=np.int32)
    if len(positions) == 0:
        return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": ["No Gaussian positions."]}
    if not camera_meta_npz or not Path(camera_meta_npz).exists():
        return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": ["Projection metadata unavailable."]}
    try:
        with np.load(camera_meta_npz) as meta:
            intr = np.asarray(meta["input_intr"])
            c2ws = _normalize_c2ws(np.asarray(meta["input_c2ws"]))
            width = _meta_scalar(meta, "width", image_size[0])
            height = _meta_scalar(meta, "height", image_size[1])
        if not part_masks:
            return {"gaussian_part_ids": ids, "assigned_ratio": 0.0, "per_part_counts": {}, "warnings": ["No part masks."]}
        parts = sorted(part_masks, key=lambda item: float(item.get("area", 0)))
        residual_part_ids = {int(part["part_id"]) for part in parts if _is_residual_part(part)}
        scores = np.zeros((len(positions), len(parts)), dtype=np.float32)
        view_support = np.zeros((len(positions), len(parts)), dtype=np.uint8)
        per_view_hits: dict[str, dict[str, int]] = {}
        used_labels: list[str] = []
        homo = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        mask_cache: dict[str, np.ndarray] = {}
        weight_cache: dict[str, np.ndarray] = {}
        view_count = min(len(c2ws), len(VIEW_LABELS))
        for view_idx in range(view_count):
            label = VIEW_LABELS[view_idx]
            available = any(_view_mask_value(part, label, view_idx) is not None for part in parts)
            if not available:
                continue
            fx, fy, cx, cy = _intrinsics_for_view(intr, view_idx)
            c2w = c2ws[view_idx]
            w2c = np.linalg.inv(c2w)
            cam = (w2c @ homo.T).T[:, :3]
            valid = cam[:, 2] > 1e-6
            u = fx * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + cx
            v = fy * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + cy
            u_i = np.round(u).astype(np.int64)
            v_i = np.round(v).astype(np.int64)
            in_frame = valid & (u_i >= 0) & (u_i < width) & (v_i >= 0) & (v_i < height)
            label_hits: dict[str, int] = {}
            for part_idx, part in enumerate(parts):
                value = _view_mask_value(part, label, view_idx)
                if value is None:
                    continue
                try:
                    mask = _read_mask_cached(value, mask_cache)
                except Exception as exc:
                    warnings.append(f"Failed to read {label} mask for part {part.get('part_id')}: {exc}")
                    continue
                if mask.size == 0:
                    continue
                weight_map = _mask_weight_cached(value, mask_cache, weight_cache)
                mh, mw = mask.shape
                uu = np.clip((u_i * mw / max(1, width)).astype(np.int64), 0, mw - 1)
                vv = np.clip((v_i * mh / max(1, height)).astype(np.int64), 0, mh - 1)
                hit = in_frame & mask[vv, uu]
                hit_count = int(hit.sum())
                if hit_count == 0:
                    continue
                pid = str(int(part["part_id"]))
                label_hits[pid] = hit_count
                confidence = max(0.05, float(part.get("confidence", 1.0) or 1.0))
                if _is_residual_part(part):
                    confidence = min(confidence, 0.20)
                area_tie_break = 1e-6 / max(1.0, float(part.get("area", 1.0) or 1.0))
                scores[hit, part_idx] += confidence * weight_map[vv[hit], uu[hit]] + area_tie_break
                view_support[hit, part_idx] += 1
            if label_hits:
                used_labels.append(label)
                per_view_hits[label] = label_hits

        low_confidence_count = 0
        smoothed_count = 0
        knn_unknown_reassigned_count = 0
        knn_island_reassigned_count = 0
        mean_view_support = 0.0
        margin_stats = {"mean": 0.0, "p10": 0.0}
        if used_labels:
            best = np.argmax(scores, axis=1)
            best_score = scores[np.arange(len(positions)), best]
            assigned_mask = best_score > 0
            sorted_scores = np.sort(scores, axis=1)
            second_score = sorted_scores[:, -2] if scores.shape[1] > 1 else np.zeros_like(best_score)
            margin = best_score - second_score
            margin_ratio = margin / np.maximum(best_score, 1e-6)
            best_support = view_support[np.arange(len(positions)), best]
            if np.any(assigned_mask):
                mean_view_support = float(best_support[assigned_mask].mean())
                margin_stats = {
                    "mean": float(margin_ratio[assigned_mask].mean()),
                    "p10": float(np.percentile(margin_ratio[assigned_mask], 10)),
                }
            multi_view_available = len(used_labels) >= 3
            low_confidence = assigned_mask & (
                (margin_ratio < 0.18)
                | (multi_view_available & (best_support <= 1))
            )
            high_confidence = assigned_mask & (
                (margin_ratio >= 0.30)
                & ((best_support >= 2) | ~multi_view_available)
            )
            low_confidence_count = int(low_confidence.sum())
            for part_idx, part in enumerate(parts):
                ids[assigned_mask & (best == part_idx)] = int(part["part_id"])
            if low_confidence_count:
                ids, smoothed_count = _smooth_low_confidence_assignments(positions, ids, low_confidence, high_confidence)
            ids, knn_stats = _knn_label_consistency_cleanup(positions, ids, residual_part_ids)
            knn_unknown_reassigned_count = int(knn_stats.get("unknown_reassigned", 0))
            knn_island_reassigned_count = int(knn_stats.get("island_reassigned", 0))
        else:
            warnings.append("No readable projection masks for any camera view.")
        assigned = ids >= 0
        counts = {str(pid): int((ids == int(pid)).sum()) for pid in np.unique(ids[assigned])}
        support_counts: dict[str, dict[str, int]] = {}
        if used_labels:
            best_for_support = np.argmax(scores, axis=1)
            best_support = view_support[np.arange(len(positions)), best_for_support]
            for support in range(int(best_support.max(initial=0)) + 1):
                count = int((assigned & (best_support == support)).sum())
                if count:
                    support_counts[str(support)] = {"count": count}
        return {
            "gaussian_part_ids": ids,
            "assigned_ratio": float(assigned.mean()),
            "per_part_counts": counts,
            "warnings": warnings,
            "view_labels": used_labels,
            "per_view_hits": per_view_hits,
            "projection_image_size": [int(width), int(height)],
            "mean_view_support": mean_view_support,
            "view_support_counts": support_counts,
            "margin_ratio": margin_stats,
            "low_confidence_count": low_confidence_count,
            "smoothed_count": int(smoothed_count),
            "knn_unknown_reassigned_count": knn_unknown_reassigned_count,
            "knn_island_reassigned_count": knn_island_reassigned_count,
        }
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
