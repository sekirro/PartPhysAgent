from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .gaussian_assign import _intrinsics_for_view, _meta_scalar, _normalize_c2ws, load_ply_positions
from .image_utils import read_mask
from .material_table import density_for_material, default_E_for_material, default_nu_for_material
from .scene_builder import FRAME_NAMES, VIEW_LABELS, _candidate_multiview_paths
from .types import ObjectInstance, PhysGMResult

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


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return value.strip("_") or "object"


def _write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _source_view_paths(image_path, multiview_dir: str | None) -> list[Path]:
    paths = _candidate_multiview_paths(multiview_dir)
    if paths:
        return paths
    return [Path(image_path)] * len(FRAME_NAMES)


def _mask_for_view(obj: ObjectInstance, label: str) -> str | None:
    if label == "front":
        return obj.mask_path
    view_masks = (obj.metadata or {}).get("view_masks") or {}
    return view_masks.get(label)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty object mask.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _square_crop_from_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float = 0.18,
) -> tuple[float, float, float]:
    x1, y1, x2, y2 = bbox
    w, h = image_size
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    crop_size = max(bw, bh) * (1.0 + 2.0 * float(padding_ratio))
    crop_size = max(8.0, min(crop_size, float(max(w, h))))
    cx = 0.5 * float(x1 + x2)
    cy = 0.5 * float(y1 + y2)
    crop_x0 = cx - 0.5 * crop_size
    crop_y0 = cy - 0.5 * crop_size
    return crop_x0, crop_y0, crop_size


def _crop_rgba_square(rgba: Image.Image, crop_x0: float, crop_y0: float, crop_size: float, output_size: int = 512) -> Image.Image:
    src = np.asarray(rgba.convert("RGBA"))
    x0 = int(np.floor(crop_x0))
    y0 = int(np.floor(crop_y0))
    size = int(np.ceil(crop_size))
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    canvas[..., :3] = 255
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(rgba.width, x0 + size)
    sy1 = min(rgba.height, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = src[sy0:sy1, sx0:sx1]
    cropped = Image.fromarray(canvas, mode="RGBA")
    return cropped.resize((output_size, output_size), Image.Resampling.BICUBIC)


def _save_alpha_mask(rgba: Image.Image, image_path: Path) -> str:
    mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
    alpha = np.asarray(rgba.convert("RGBA"))[..., 3]
    Image.fromarray((alpha > 0).astype(np.uint8) * 255, mode="L").save(mask_path)
    return str(mask_path)


def _masked_crop_to_view(
    image_path: Path,
    mask_path: str | None,
    output_path: Path,
    output_size: int = 512,
) -> dict[str, Any]:
    if not mask_path:
        raise ValueError(f"Missing object mask for view image {image_path}")
    image = Image.open(image_path).convert("RGBA")
    mask = read_mask(mask_path)
    if mask.shape[::-1] != image.size:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 127
    bbox = _mask_bbox(mask)
    crop_x0, crop_y0, crop_size = _square_crop_from_bbox(bbox, image.size)
    arr = np.asarray(image).copy()
    arr[~mask] = np.array([255, 255, 255, 0], dtype=np.uint8)
    out = _crop_rgba_square(Image.fromarray(arr, mode="RGBA"), crop_x0, crop_y0, crop_size, output_size=output_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    output_mask_path = _save_alpha_mask(out, output_path)
    return {
        "source_path": str(image_path),
        "mask_path": str(mask_path),
        "output_mask_path": output_mask_path,
        "bbox": [int(x) for x in bbox],
        "mask_area": int(mask.sum()),
        "alpha_nonzero": int((np.asarray(out.convert("RGBA"))[..., 3] > 0).sum()),
        "crop_x0": float(crop_x0),
        "crop_y0": float(crop_y0),
        "crop_size": float(crop_size),
        "output_size": int(output_size),
        "source_width": int(image.width),
        "source_height": int(image.height),
    }


def _masked_full_frame_to_view(
    image_path: Path,
    mask_path: str | None,
    output_path: Path,
    output_size: int = 512,
) -> dict[str, Any]:
    if not mask_path:
        raise ValueError(f"Missing object mask for view image {image_path}")
    image = Image.open(image_path).convert("RGBA")
    mask = read_mask(mask_path)
    if mask.shape[::-1] != image.size:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 127
    arr = np.asarray(image).copy()
    arr[~mask] = np.array([255, 255, 255, 0], dtype=np.uint8)
    out = Image.fromarray(arr, mode="RGBA")
    if out.size != (output_size, output_size):
        out = out.resize((output_size, output_size), Image.Resampling.BICUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    output_mask_path = _save_alpha_mask(out, output_path)
    return {
        "source_path": str(image_path),
        "mask_path": str(mask_path),
        "output_mask_path": output_mask_path,
        "bbox": [int(x) for x in _mask_bbox(mask)],
        "mask_area": int(mask.sum()),
        "alpha_nonzero": int((np.asarray(out.convert("RGBA"))[..., 3] > 0).sum()),
        "output_size": int(output_size),
        "source_width": int(image.width),
        "source_height": int(image.height),
    }


def _rgba_preview(path: Path, size: int = 192) -> Image.Image:
    rgba = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(rgba)
    checker = np.full((size, size, 3), 240, dtype=np.uint8)
    yy, xx = np.indices((size, size))
    checker[((xx // 12 + yy // 12) % 2) == 0] = 210
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    comp = (arr[..., :3].astype(np.float32) * alpha + checker.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    alpha_vis = np.repeat(arr[..., 3:4], 3, axis=2)
    pane = Image.new("RGB", (size, size * 2), "white")
    pane.paste(Image.fromarray(comp, mode="RGB"), (0, 0))
    pane.paste(Image.fromarray(alpha_vis, mode="RGB"), (0, size))
    return pane


def _write_input_debug_sheet(output_dir: Path, object_name: str, rows: list[dict[str, Any]]) -> str:
    panes = []
    size = 192
    for frame_name, label, row in zip(FRAME_NAMES, VIEW_LABELS, rows):
        pane = _rgba_preview(output_dir / frame_name, size=size)
        draw = ImageDraw.Draw(pane)
        draw.rectangle([0, 0, size - 1, 18], fill=(255, 255, 255))
        draw.text((3, 3), f"{label} area={int(row.get('mask_area', 0))}", fill=(0, 0, 0))
        panes.append(pane)
    if not panes:
        return ""
    sheet = Image.new("RGB", (size * len(panes), size * 2), "white")
    for idx, pane in enumerate(panes):
        sheet.paste(pane, (idx * size, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((3, size * 2 - 15), object_name[:80], fill=(0, 0, 0))
    debug_path = output_dir / "object_input_alpha_debug.png"
    sheet.save(debug_path)
    return str(debug_path)


def _write_combined_debug_sheet(paths: list[Path], output_path: Path) -> str | None:
    images = [Image.open(path).convert("RGB") for path in paths if path.exists()]
    if not images:
        return None
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _load_pose_template(multiview_dir: str | None) -> dict[str, Any] | None:
    if not multiview_dir:
        return None
    path = Path(multiview_dir) / "pose.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_cropped_pose(
    output_dir: Path,
    source_pose: dict[str, Any] | None,
    crop_rows: list[dict[str, Any]],
) -> None:
    if not source_pose or not source_pose.get("frames"):
        return
    frames = []
    for idx, (frame_name, crop) in enumerate(zip(FRAME_NAMES, crop_rows)):
        src = dict(source_pose["frames"][min(idx, len(source_pose["frames"]) - 1)])
        output_size = float(crop["output_size"])
        src["file_path"] = frame_name
        src["w"] = int(output_size)
        src["h"] = int(output_size)
        # The object has already been recentered into a canonical square image.
        # Keep the PhysGM camera canonical here; crop offsets are scene-placement
        # metadata for the later merge step, not camera principal-point shifts.
        src["fx"] = float(output_size)
        src["fy"] = float(output_size)
        src["cx"] = float(output_size) * 0.5
        src["cy"] = float(output_size) * 0.5
        frames.append(src)
    with (output_dir / "pose.json").open("w", encoding="utf-8") as f:
        json.dump({"scene_name": output_dir.name, "frames": frames}, f, indent=2)


def _write_full_frame_pose(
    output_dir: Path,
    source_pose: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> None:
    if not source_pose or not source_pose.get("frames"):
        return
    frames = []
    for idx, (frame_name, row) in enumerate(zip(FRAME_NAMES, rows)):
        src = dict(source_pose["frames"][min(idx, len(source_pose["frames"]) - 1)])
        source_w = float(src.get("w", row["source_width"]) or row["source_width"])
        source_h = float(src.get("h", row["source_height"]) or row["source_height"])
        out_size = float(row["output_size"])
        src["file_path"] = frame_name
        src["w"] = int(row["output_size"])
        src["h"] = int(row["output_size"])
        src["fx"] = float(src.get("fx", max(source_w, source_h))) * (out_size / source_w)
        src["fy"] = float(src.get("fy", max(source_w, source_h))) * (out_size / source_h)
        src["cx"] = float(src.get("cx", source_w / 2.0)) * (out_size / source_w)
        src["cy"] = float(src.get("cy", source_h / 2.0)) * (out_size / source_h)
        frames.append(src)
    with (output_dir / "pose.json").open("w", encoding="utf-8") as f:
        json.dump({"scene_name": output_dir.name, "frames": frames}, f, indent=2)


def build_object_multiview_input(
    image_path,
    multiview_dir: str | None,
    obj: ObjectInstance,
    output_dir,
    crop_objects: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _source_view_paths(image_path, multiview_dir)
    rows = []
    for idx, (frame_name, label) in enumerate(zip(FRAME_NAMES, VIEW_LABELS)):
        source = source_paths[min(idx, len(source_paths) - 1)]
        if crop_objects:
            rows.append(_masked_crop_to_view(source, _mask_for_view(obj, label), output_dir / frame_name))
        else:
            rows.append(_masked_full_frame_to_view(source, _mask_for_view(obj, label), output_dir / frame_name))
    if crop_objects:
        _write_cropped_pose(output_dir, _load_pose_template(multiview_dir), rows)
    else:
        _write_full_frame_pose(output_dir, _load_pose_template(multiview_dir), rows)
    metadata = {
        "object_id": int(obj.object_id),
        "object_name": obj.name,
        "input_mode": "crop" if crop_objects else "full_frame_masked",
        "source_multiview_dir": str(multiview_dir) if multiview_dir else None,
        "view_labels": VIEW_LABELS,
        "frame_names": FRAME_NAMES,
        "view_rows": rows,
    }
    metadata["debug_alpha_sheet"] = _write_input_debug_sheet(output_dir, obj.name, rows)
    _write_json(output_dir / "object_input_metadata.json", metadata)
    return output_dir


def _read_ply_header(path: Path) -> tuple[list[str], int, int, str]:
    with path.open("rb") as f:
        header_bytes = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
    header = header_bytes.decode("ascii", errors="replace").splitlines()
    fmt = next((line for line in header if line.startswith("format ")), "")
    count = None
    for line in header:
        if line.startswith("element vertex "):
            count = int(line.split()[-1])
            break
    if count is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    return header, count, len(header_bytes), fmt


def _header_signature(header: list[str]) -> list[str]:
    return ["element vertex <count>" if line.startswith("element vertex ") else line for line in header]


def _vertex_dtype(header: list[str], fmt: str) -> np.dtype:
    endian = "<" if "binary_little_endian" in fmt else ">" if "binary_big_endian" in fmt else None
    if endian is None:
        raise ValueError(f"Unsupported PLY format for filtered merge: {fmt}")
    fields = []
    in_vertex = False
    for line in header:
        if line.startswith("element vertex "):
            in_vertex = True
            continue
        if line.startswith("element ") and not line.startswith("element vertex "):
            in_vertex = False
        if not in_vertex or not line.startswith("property "):
            continue
        tokens = line.split()
        if len(tokens) < 3 or tokens[1] == "list":
            raise ValueError("PLY list properties are not supported for filtered merge.")
        np_type = PLY_NUMPY_TYPES.get(tokens[1])
        if not np_type:
            raise ValueError(f"Unsupported PLY property type: {tokens[1]}")
        fields.append((tokens[2], np.dtype(endian + np_type)))
    return np.dtype(fields)


def _select_vertex_indices(vertex: np.ndarray, max_vertices: int | None) -> np.ndarray:
    count = len(vertex)
    if max_vertices is None or count <= int(max_vertices):
        return np.arange(count, dtype=np.int64)
    keep = max(1, int(max_vertices))
    names = vertex.dtype.names or ()
    if "opacity" in names:
        opacity = np.asarray(vertex["opacity"], dtype=np.float32)
        return np.sort(np.argpartition(opacity, -keep)[-keep:]).astype(np.int64)
    return np.linspace(0, count - 1, keep).round().astype(np.int64)


def _write_filtered_ply(input_path: Path, output_path: Path, keep_indices: np.ndarray) -> int:
    header, count, header_end, fmt = _read_ply_header(input_path)
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < count)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_header = [
        f"element vertex {len(keep_indices)}" if line.startswith("element vertex ") else line
        for line in header
    ]
    with output_path.open("wb") as out:
        out.write(("\n".join(output_header) + "\n").encode("ascii"))
        if "binary_" in fmt:
            dtype = _vertex_dtype(header, fmt)
            with input_path.open("rb") as f:
                f.seek(header_end)
                vertex = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count).copy()
            out.write(vertex[keep_indices].tobytes())
        elif "ascii" in fmt:
            keep = set(int(idx) for idx in keep_indices.tolist())
            with input_path.open("rb") as f:
                f.seek(header_end)
                for idx in range(count):
                    line = f.readline()
                    if idx in keep:
                        out.write(line)
        else:
            raise ValueError(f"Unsupported PLY format for filtering: {fmt}")
    return int(len(keep_indices))


def _projective_object_keep_indices(
    ply_path: Path,
    camera_meta_npz: Path,
    view_masks: dict[str, str],
    min_view_support: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    positions = load_ply_positions(ply_path)
    if len(positions) == 0 or not camera_meta_npz.exists() or not view_masks:
        return np.arange(len(positions), dtype=np.int64), {
            "enabled": False,
            "reason": "missing_positions_camera_meta_or_masks",
            "input_count": int(len(positions)),
            "kept_count": int(len(positions)),
        }
    with np.load(camera_meta_npz) as meta:
        intr = np.asarray(meta["input_intr"])
        c2ws = _normalize_c2ws(np.asarray(meta["input_c2ws"]))
        width = _meta_scalar(meta, "width", 512)
        height = _meta_scalar(meta, "height", 512)
    support = np.zeros(len(positions), dtype=np.uint8)
    per_view_hits: dict[str, int] = {}
    homo = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    view_count = min(len(c2ws), len(VIEW_LABELS))
    for view_idx in range(view_count):
        label = VIEW_LABELS[view_idx]
        mask_path = view_masks.get(label)
        if not mask_path:
            continue
        mask = read_mask(mask_path)
        if mask.size == 0:
            continue
        fx, fy, cx, cy = _intrinsics_for_view(intr, view_idx)
        w2c = np.linalg.inv(c2ws[view_idx])
        cam = (w2c @ homo.T).T[:, :3]
        valid = cam[:, 2] > 1e-6
        u = fx * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + cx
        v = fy * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + cy
        u_i = np.round(u).astype(np.int64)
        v_i = np.round(v).astype(np.int64)
        in_frame = valid & (u_i >= 0) & (u_i < width) & (v_i >= 0) & (v_i < height)
        mh, mw = mask.shape
        uu = np.clip((u_i * mw / max(1, width)).astype(np.int64), 0, mw - 1)
        vv = np.clip((v_i * mh / max(1, height)).astype(np.int64), 0, mh - 1)
        hit = in_frame & mask[vv, uu]
        support[hit] += 1
        per_view_hits[label] = int(hit.sum())
    available_views = sum(1 for label in VIEW_LABELS[:view_count] if view_masks.get(label))
    required = min(int(min_view_support), max(1, int(available_views)))
    keep = np.where(support >= required)[0].astype(np.int64)
    min_keep = max(256, int(0.15 * len(positions)))
    if len(keep) < min_keep:
        return np.arange(len(positions), dtype=np.int64), {
            "enabled": False,
            "reason": "projection_filter_would_remove_too_many_points",
            "input_count": int(len(positions)),
            "kept_count": int(len(positions)),
            "candidate_kept_count": int(len(keep)),
            "required_view_support": int(required),
            "per_view_hits": per_view_hits,
        }
    return keep, {
        "enabled": True,
        "input_count": int(len(positions)),
        "kept_count": int(len(keep)),
        "removed_count": int(len(positions) - len(keep)),
        "required_view_support": int(required),
        "available_views": int(available_views),
        "per_view_hits": per_view_hits,
    }


def _logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _raise_low_object_opacity(vertex: np.ndarray, target_p99_opacity: float = 0.08) -> float:
    names = vertex.dtype.names or ()
    if len(vertex) == 0 or "opacity" not in names:
        return 0.0
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    current_p99 = float(np.percentile(opacity, 99))
    target_p99 = _logit(target_p99_opacity)
    if current_p99 >= target_p99:
        return 0.0
    offset = target_p99 - current_p99
    vertex["opacity"] = (opacity + offset).astype(vertex.dtype["opacity"])
    return float(offset)


def _object_placement(obj: ObjectInstance, pixel_to_world: float = 0.003) -> dict[str, Any]:
    metadata = dict(obj.metadata or {})
    raw_metadata = metadata.get("raw_metadata") if isinstance(metadata.get("raw_metadata"), dict) else {}
    source_metadata = metadata.get("source_metadata") or raw_metadata.get("source_metadata")
    object_record = None
    if source_metadata:
        try:
            data = json.loads(Path(source_metadata).read_text(encoding="utf-8"))
            objects = data.get("objects") or []
            object_id = int(obj.object_id)
            if 0 <= object_id < len(objects):
                object_record = objects[object_id]
        except Exception:
            object_record = None
    object_record = object_record or {}
    x_px = float(object_record.get("x", metadata.get("x", raw_metadata.get("x", 0.0))) or 0.0)
    z_px = float(object_record.get("z", metadata.get("z", raw_metadata.get("z", 0.0))) or 0.0)
    y_px = float(object_record.get("y", object_record.get("dy", metadata.get("y", raw_metadata.get("y", 0.0)))) or 0.0)
    scale = float(object_record.get("scale", metadata.get("scale", raw_metadata.get("scale", 1.0))) or 1.0)
    return {
        "object_id": int(obj.object_id),
        "object_name": obj.name,
        "x_px": x_px,
        "y_px": y_px,
        "z_px": z_px,
        "scale": scale,
        "pixel_to_world": float(pixel_to_world),
        "translation": [float(x_px * pixel_to_world), float(-y_px * pixel_to_world), float(z_px * pixel_to_world)],
        "source_metadata": source_metadata,
    }


def merge_ply_files(
    ply_paths: list[Path],
    output_path: Path,
    max_vertices_per_object: int | None = 60000,
    placements: list[dict[str, Any]] | None = None,
) -> list[int]:
    if not ply_paths:
        raise ValueError("No PLY files to merge.")
    records: list[tuple[Path, list[str], int, int, str]] = []
    first_sig = None
    for path in ply_paths:
        header, count, header_end, fmt = _read_ply_header(path)
        sig = _header_signature(header)
        if first_sig is None:
            first_sig = sig
        elif sig != first_sig:
            raise ValueError(f"Cannot merge PLY files with different vertex layouts: {path}")
        records.append((path, header, count, header_end, fmt))

    filtered_vertices = []
    selected_counts = []
    for path, header, count, header_end, fmt in records:
        if "binary_" not in fmt:
            if max_vertices_per_object is not None and count > int(max_vertices_per_object):
                raise ValueError("Filtered merge for ASCII PLY is not supported.")
            selected_counts.append(int(count))
            filtered_vertices.append(None)
            continue
        dtype = _vertex_dtype(header, fmt)
        with path.open("rb") as f:
            f.seek(header_end)
            vertex = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count).copy()
        selected = _select_vertex_indices(vertex, max_vertices_per_object)
        selected_vertex = vertex[selected].copy()
        if placements and len(placements) > len(filtered_vertices):
            placement = placements[len(filtered_vertices)]
            translation = np.asarray(placement.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            names = selected_vertex.dtype.names or ()
            if all(axis in names for axis in ("x", "y", "z")):
                selected_vertex["x"] = (selected_vertex["x"].astype(np.float32) + translation[0]).astype(selected_vertex.dtype["x"])
                selected_vertex["y"] = (selected_vertex["y"].astype(np.float32) + translation[1]).astype(selected_vertex.dtype["y"])
                selected_vertex["z"] = (selected_vertex["z"].astype(np.float32) + translation[2]).astype(selected_vertex.dtype["z"])
        _raise_low_object_opacity(selected_vertex)
        filtered_vertices.append(selected_vertex)
        selected_counts.append(int(len(selected)))

    total = int(sum(selected_counts))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"element vertex {total}" if line.startswith("element vertex ") else line
        for line in records[0][1]
    ]
    with output_path.open("wb") as out:
        out.write(("\n".join(header) + "\n").encode("ascii"))
        for vertices, (path, _, _, header_end, _) in zip(filtered_vertices, records):
            if vertices is None:
                with path.open("rb") as f:
                    f.seek(header_end)
                    out.write(f.read())
            else:
                out.write(vertices.tobytes())
    return selected_counts


def run_multi_object_physgm(
    runner,
    image_path,
    scene_name: str,
    output_dir,
    objects: list[ObjectInstance],
    multiview_dir: str | None = None,
    use_mvadapter: bool = False,
    crop_objects: bool = True,
    filter_by_projection: bool = True,
) -> tuple[PhysGMResult, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_root = output_dir / "objects"
    ply_paths: list[Path] = []
    result_rows: list[dict[str, Any]] = []
    debug_paths: list[Path] = []
    ordered_objects = sorted(objects, key=lambda obj: int(obj.object_id))
    for obj in ordered_objects:
        safe = _safe_name(obj.name)
        obj_root = object_root / f"object_{int(obj.object_id):03d}_{safe}"
        input_dir = build_object_multiview_input(
            image_path,
            multiview_dir,
            obj,
            obj_root / "input_views",
            crop_objects=bool(crop_objects),
        )
        input_debug_path = input_dir / "object_input_alpha_debug.png"
        if input_debug_path.exists():
            debug_paths.append(input_debug_path)
        phys_dir = obj_root / "physgm"
        result = runner.infer_image(
            str(input_dir / "000.png"),
            scene_name=f"{scene_name}_object_{int(obj.object_id):03d}_{safe}",
            output_dir=phys_dir,
            save_gaussian=True,
            use_mvadapter=use_mvadapter and not multiview_dir,
            multiview_dir=str(input_dir),
        )
        if not result.point_cloud_path or not Path(result.point_cloud_path).exists():
            raise RuntimeError(f"PhysGM did not produce point_clouds.ply for object {obj.object_id}:{obj.name}")
        point_cloud_path = Path(result.point_cloud_path)
        projection_filter: dict[str, Any] = {"enabled": False, "reason": "disabled"}
        if filter_by_projection:
            try:
                input_metadata = json.loads((input_dir / "object_input_metadata.json").read_text(encoding="utf-8"))
                view_rows = input_metadata.get("view_rows") or []
                view_masks = {
                    label: str(row["output_mask_path"])
                    for label, row in zip(VIEW_LABELS, view_rows)
                    if row.get("output_mask_path")
                }
                keep_indices, projection_filter = _projective_object_keep_indices(
                    point_cloud_path,
                    Path(result.scene_dir) / "input_batch_meta.npz",
                    view_masks,
                    min_view_support=1,
                )
                if projection_filter.get("enabled"):
                    filtered_path = obj_root / "filtered_point_clouds.ply"
                    _write_filtered_ply(point_cloud_path, filtered_path, keep_indices)
                    point_cloud_path = filtered_path
            except Exception as exc:
                projection_filter = {"enabled": False, "reason": f"projection_filter_failed: {exc}"}
        ply_paths.append(point_cloud_path)
        result_rows.append(
            {
                "object_id": int(obj.object_id),
                "object_name": obj.name,
                "input_dir": str(input_dir),
                "input_mode": "crop" if crop_objects else "full_frame_masked",
                "input_debug_path": str(input_debug_path) if input_debug_path.exists() else None,
                "physgm_dir": result.scene_dir,
                "point_cloud_path": result.point_cloud_path,
                "filtered_point_cloud_path": str(point_cloud_path),
                "projection_filter": projection_filter,
                "predicted_phys_path": result.predicted_phys_path,
                "material": result.material,
                "E": result.E,
                "nu": result.nu,
                "density": result.density,
            }
        )

    placements = [_object_placement(obj) for obj in ordered_objects]
    counts = merge_ply_files(ply_paths, output_dir / "point_clouds.ply", placements=placements)
    object_ids = np.concatenate(
        [np.full(count, int(obj.object_id), dtype=np.int32) for obj, count in zip(ordered_objects, counts)]
    )
    np.save(output_dir / "gaussian_object_ids_direct.npy", object_ids)

    first_phys_dir = Path(result_rows[0]["physgm_dir"])
    first_meta = first_phys_dir / "input_batch_meta.npz"
    if first_meta.exists():
        shutil.copy2(first_meta, output_dir / "input_batch_meta.npz")
    combined_debug = _write_combined_debug_sheet(debug_paths, output_dir / "object_inputs_alpha_debug.png")
    largest_idx = int(np.argmax([obj.area for obj in ordered_objects]))
    largest = result_rows[largest_idx]
    material = largest.get("material") or "Plastic"
    density = largest.get("density") or density_for_material(material)
    raw = {
        "mode": "multi_object_physgm",
        "material": material,
        "E": float(largest.get("E") or default_E_for_material(material)),
        "nu": float(largest.get("nu") or default_nu_for_material(material)),
        "density": float(density),
        "object_results": result_rows,
        "object_placements": placements,
        "object_inputs_alpha_debug": combined_debug,
        "object_gaussian_counts": {
            str(int(obj.object_id)): int(count)
            for obj, count in zip(ordered_objects, counts)
        },
        "gaussian_object_ids_direct": str(output_dir / "gaussian_object_ids_direct.npy"),
    }
    _write_json(output_dir / "predicted_phys.json", raw)
    _write_json(output_dir / "raw_output_summary.json", raw)
    return (
        PhysGMResult(
            str(output_dir),
            str(output_dir / "point_clouds.ply"),
            str(output_dir / "predicted_phys.json"),
            str(material),
            float(raw["E"]),
            float(raw["nu"]),
            float(raw["density"]),
            raw,
        ),
        object_ids,
    )
