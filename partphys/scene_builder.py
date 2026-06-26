from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

FRAME_NAMES = ["000.png", "006.png", "012.png", "018.png"]
VIEW_LABELS = ["front", "right", "rear", "left"]


def _canonical_w2c(angle_deg: float, radius: float = 1.7) -> list[list[float]]:
    angle = math.radians(angle_deg)
    c = np.array([radius * math.sin(angle), 0.0, radius * math.cos(angle)], dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    forward = target - c
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = c
    return np.linalg.inv(c2w).tolist()


def _load_template(template_pose_json: str | None):
    if not template_pose_json:
        return None
    path = Path(template_pose_json)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    if "A" in image.getbands():
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
        alpha = rgba[..., 3] > 0
        if alpha.any():
            rgb = rgba[..., :3]
            visible = rgb[alpha]
            return tuple(int(x) for x in np.clip(visible.mean(axis=0), 0, 255))
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    samples = np.concatenate([rgb[:4, :4].reshape(-1, 3), rgb[:4, -4:].reshape(-1, 3), rgb[-4:, :4].reshape(-1, 3), rgb[-4:, -4:].reshape(-1, 3)], axis=0)
    return tuple(int(x) for x in np.clip(samples.mean(axis=0), 0, 255))


def _load_physgm_image(path) -> Image.Image:
    image = Image.open(path)
    if "A" in image.getbands():
        return image.convert("RGBA")
    return image.convert("RGB")


def _resize_to_square_for_physgm(image: Image.Image, size: int = 512, bg=(255, 255, 255)) -> Image.Image:
    has_alpha = "A" in image.getbands()
    mode = "RGBA" if has_alpha else "RGB"
    image = image.convert(mode)
    w, h = image.size
    if w == 0 or h == 0:
        color = (*bg, 0) if has_alpha else bg
        return Image.new(mode, (size, size), color)
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    color = (*bg, 0) if has_alpha else bg
    canvas = Image.new(mode, (size, size), color)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _save_physgm_image(image: Image.Image, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "RGBA" if "A" in image.getbands() else "RGB"
    image.convert(mode).save(path)


def _side_proxy(image: Image.Image, label: str) -> Image.Image:
    w, h = image.size
    has_alpha = "A" in image.getbands()
    bg = _background_color(image)
    canvas = Image.new("RGBA" if has_alpha else "RGB", (w, h), (*bg, 0) if has_alpha else bg)
    scale = 0.78
    side = image.resize((max(1, int(w * scale)), h), Image.Resampling.LANCZOS)
    if label == "left":
        side = ImageOps.mirror(side)
        x = int(w * 0.07)
    else:
        x = w - side.width - int(w * 0.07)
    canvas.paste(side, (x, 0))
    return canvas


def _rear_proxy(image: Image.Image) -> Image.Image:
    rear = ImageOps.mirror(image)
    rear = ImageEnhance.Contrast(rear).enhance(0.92)
    return ImageEnhance.Color(rear).enhance(0.9)


def _single_image_proxy_views(image: Image.Image) -> list[Image.Image]:
    return [image.copy(), _side_proxy(image, "right"), _rear_proxy(image), _side_proxy(image, "left")]


def _candidate_multiview_paths(multiview_dir: str | None) -> list[Path]:
    if not multiview_dir:
        return []
    root = Path(multiview_dir)
    if not root.exists():
        return []
    paths = [root / name for name in FRAME_NAMES]
    if all(p.exists() for p in paths):
        return paths
    label_paths = [root / f"{label}.png" for label in VIEW_LABELS]
    if all(p.exists() for p in label_paths):
        return label_paths
    return []


def _load_multiview_images(multiview_dir: str | None, size: int | None) -> tuple[list[Image.Image], list[str]] | None:
    paths = _candidate_multiview_paths(multiview_dir)
    if not paths:
        return None
    images = []
    for path in paths:
        image = _load_physgm_image(path)
        if size is not None:
            image = _resize_to_square_for_physgm(image, int(size))
        images.append(image)
    return images, [str(p) for p in paths]


def _write_pose_frames(template, frame_names: list[str], width: int, height: int) -> tuple[list[dict], bool]:
    frames = []
    if template and template.get("frames"):
        for idx, frame_name in enumerate(frame_names):
            src = template["frames"][min(idx, len(template["frames"]) - 1)]
            frame = dict(src)
            frame["file_path"] = frame_name
            frame["h"] = height
            frame["w"] = width
            scale = max(width, height)
            frame["fx"] = float(scale)
            frame["fy"] = float(scale)
            frame["cx"] = float(width) / 2.0
            frame["cy"] = float(height) / 2.0
            frames.append(frame)
        return frames, True

    for frame_name, angle in zip(frame_names, [0, 90, 180, 270]):
        frames.append(
            {
                "file_path": frame_name,
                "w2c": _canonical_w2c(angle),
                "h": height,
                "w": width,
                "fx": float(max(width, height)),
                "fy": float(max(width, height)),
                "cx": float(width) / 2.0,
                "cy": float(height) / 2.0,
            }
        )
    return frames, False


def build_physgm_input_scene(
    image_path,
    scene_root,
    scene_name,
    template_pose_json=None,
    duplicate_single_image: bool = True,
    size: int | None = None,
    multiview_dir: str | None = None,
    multiview_source: str = "provided_multiview",
) -> dict:
    scene_root = Path(scene_root)
    scene_dir = scene_root / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    loaded_multiview = _load_multiview_images(multiview_dir, size)
    source_paths = []
    if loaded_multiview is not None:
        view_images, source_paths = loaded_multiview
        view_source = multiview_source
    else:
        image = _load_physgm_image(image_path)
        if size is not None:
            image = _resize_to_square_for_physgm(image, int(size))
        if duplicate_single_image:
            view_images = _single_image_proxy_views(image)
            view_source = "single_image_proxy"
        else:
            view_images = [image.copy() for _ in FRAME_NAMES]
            view_source = "single_image_repeat"
        source_paths = [str(image_path)]

    w, h = view_images[0].size
    frame_paths = []
    frame_records = []
    for name, label, image in zip(FRAME_NAMES, VIEW_LABELS, view_images):
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        out = scene_dir / name
        _save_physgm_image(image, out)
        frame_paths.append(str(out))
        frame_records.append({"file_name": name, "view_label": label, "path": str(out)})

    template = _load_template(template_pose_json)
    frames, template_used = _write_pose_frames(template, FRAME_NAMES, w, h)
    pose = {"scene_name": scene_name, "frames": frames}
    pose_json = scene_dir / "pose.json"
    with open(pose_json, "w", encoding="utf-8") as f:
        json.dump(pose, f, indent=2)

    data_txt = scene_root / "data.txt"
    with open(data_txt, "w", encoding="utf-8") as f:
        f.write(f"./{scene_name}/pose.json\n")

    metadata = {
        "scene_name": scene_name,
        "view_source": view_source,
        "view_labels": VIEW_LABELS,
        "frame_names": FRAME_NAMES,
        "source_paths": source_paths,
        "frame_records": frame_records,
        "pose_template": str(template_pose_json) if template_pose_json else None,
        "camera_template_used": template_used,
        "width": w,
        "height": h,
    }
    metadata_path = scene_dir / "view_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "scene_root": str(scene_root),
        "scene_dir": str(scene_dir),
        "pose_json": str(pose_json),
        "data_txt": str(data_txt),
        "frame_paths": frame_paths,
        "camera_template_used": template_used,
        "view_source": view_source,
        "view_metadata": str(metadata_path),
    }
