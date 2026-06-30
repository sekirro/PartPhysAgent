from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


FRAME_NAMES = ["000.png", "006.png", "012.png", "018.png"]
VIEW_LABELS = ["front", "right", "rear", "left"]


def _shift_for_view(label: str, x: float, z: float) -> float:
    if label == "front":
        return x
    if label == "right":
        return -z
    if label == "rear":
        return -x
    if label == "left":
        return z
    return x


def _depth_for_view(label: str, x: float, z: float) -> float:
    if label == "front":
        return z
    if label == "right":
        return x
    if label == "rear":
        return -z
    if label == "left":
        return -x
    return z


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("objects"):
        raise ValueError("Manifest must contain a non-empty 'objects' list.")
    return data


def _load_layer(case_dir: Path, frame_name: str, size: int) -> Image.Image:
    path = case_dir / frame_name
    if not path.exists():
        raise FileNotFoundError(f"Missing frame: {path}")
    image = Image.open(path).convert("RGBA")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _shifted_layer_and_mask(layer: Image.Image, size: int, dx: float, dy: float, scale: float) -> tuple[Image.Image, Image.Image]:
    if scale != 1.0:
        width = max(1, int(round(layer.width * scale)))
        height = max(1, int(round(layer.height * scale)))
        layer = layer.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    x0 = (size - layer.width) // 2 + int(round(dx))
    y0 = (size - layer.height) // 2 + int(round(dy))
    canvas.alpha_composite(layer, (x0, y0))
    alpha = np.asarray(canvas.getchannel("A"))
    mask = Image.fromarray((alpha > 0).astype(np.uint8) * 255, mode="L")
    return canvas, mask


def _write_contact_sheet(output_dir: Path, size: int) -> None:
    thumb = 256
    sheet = Image.new("RGB", (thumb * len(FRAME_NAMES), thumb + 30), "white")
    draw = ImageDraw.Draw(sheet)
    yy, xx = np.indices((thumb, thumb))
    checker = np.full((thumb, thumb, 3), 240, dtype=np.uint8)
    checker[((xx // 16 + yy // 16) % 2) == 0] = 215
    for idx, (frame_name, label) in enumerate(zip(FRAME_NAMES, VIEW_LABELS)):
        rgba = np.asarray(Image.open(output_dir / frame_name).convert("RGBA").resize((thumb, thumb), Image.Resampling.LANCZOS))
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        preview = (rgba[..., :3].astype(np.float32) * alpha + checker.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        sheet.paste(Image.fromarray(preview, mode="RGB"), (idx * thumb, 30))
        draw.text((idx * thumb + 6, 8), label, fill=(0, 0, 0))
    sheet.save(output_dir / "contact_sheet.png")


def compose_case(manifest_path: Path, output_dir: Path, size: int) -> None:
    manifest = _load_manifest(manifest_path)
    objects = manifest["objects"]
    output_dir.mkdir(parents=True, exist_ok=True)
    object_mask_records: list[dict[str, Any]] = [
        {
            "object_id": int(idx),
            "name": str(obj.get("name", f"object_{idx}")),
            "case_dir": str(obj["case_dir"]),
            "view_masks": {},
        }
        for idx, obj in enumerate(objects)
    ]

    for frame_name, label in zip(FRAME_NAMES, VIEW_LABELS):
        base = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        ordered = sorted(
            objects,
            key=lambda obj: _depth_for_view(label, float(obj.get("x", 0.0)), float(obj.get("z", 0.0))),
        )
        for obj in ordered:
            layer = _load_layer(Path(obj["case_dir"]), frame_name, size)
            dx = _shift_for_view(label, float(obj.get("x", 0.0)), float(obj.get("z", 0.0)))
            dy = float(obj.get("y", obj.get("dy", 0.0)))
            scale = float(obj.get("scale", 1.0))
            object_idx = objects.index(obj)
            shifted, mask = _shifted_layer_and_mask(layer, size, dx, dy, scale)
            base.alpha_composite(shifted)
            mask_path = Path("object_masks") / f"object_{object_idx:03d}_{label}.png"
            (output_dir / mask_path).parent.mkdir(parents=True, exist_ok=True)
            mask.save(output_dir / mask_path)
            object_mask_records[object_idx]["view_masks"][label] = str(mask_path)
        base.save(output_dir / frame_name)

    pose_src = Path(manifest.get("pose_json") or Path(objects[0]["case_dir"]) / "pose.json")
    if pose_src.exists():
        shutil.copy2(pose_src, output_dir / "pose.json")

    metadata = {
        "method": "rgba_alpha_composite_view_dependent_offsets",
        "manifest": str(manifest_path),
        "objects": objects,
        "object_masks": object_mask_records,
        "frame_names": FRAME_NAMES,
        "view_labels": VIEW_LABELS,
        "background": "transparent_rgba_white_rgb",
        "size": int(size),
    }
    with (output_dir / "compose_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    _write_contact_sheet(output_dir, size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a clean RGBA multi-object four-view case.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    compose_case(args.manifest, args.output_dir, args.size)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
