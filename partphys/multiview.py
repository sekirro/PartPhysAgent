from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .scene_builder import FRAME_NAMES, VIEW_LABELS


DEFAULT_MVADAPTER_ROOTS = [
    "/root/autodl-tmp/MV-Adapter",
]
DEFAULT_CARDINAL_INDICES = [0, 2, 3, 4]


class MVAdapterError(RuntimeError):
    pass


def find_mvadapter_root(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_root = os.getenv("MVADAPTER_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(DEFAULT_MVADAPTER_ROOTS)
    for item in candidates:
        root = Path(item).expanduser()
        if (root / "scripts" / "inference_i2mv_sd.py").exists() or (root / "scripts" / "inference_i2mv_sdxl.py").exists():
            return root.resolve()
    return None


def _script_path(root: Path, variant: str) -> Path:
    value = (variant or "sd").lower()
    if value == "sdxl":
        script = root / "scripts" / "inference_i2mv_sdxl.py"
    elif value == "sd":
        script = root / "scripts" / "inference_i2mv_sd.py"
    else:
        raise MVAdapterError(f"Unsupported MV-Adapter variant: {variant}")
    if not script.exists():
        raise MVAdapterError(f"MV-Adapter script not found: {script}")
    return script


def _grid_cells(image: Image.Image) -> list[Image.Image]:
    w, h = image.size
    if w >= h * 2:
        count = max(1, round(w / h))
        cell_w = w // count
        return [image.crop((i * cell_w, 0, (i + 1) * cell_w, h)) for i in range(count)]
    if h >= w * 2:
        count = max(1, round(h / w))
        cell_h = h // count
        return [image.crop((0, i * cell_h, w, (i + 1) * cell_h)) for i in range(count)]
    if w == h and w >= 2:
        cell_w = w // 2
        cell_h = h // 2
        return [
            image.crop((0, 0, cell_w, cell_h)),
            image.crop((cell_w, 0, w, cell_h)),
            image.crop((0, cell_h, cell_w, h)),
            image.crop((cell_w, cell_h, w, h)),
        ]
    return [image]


def split_mvadapter_grid(
    grid_path,
    output_dir,
    selected_indices: list[int] | None = None,
    size: int | None = 512,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = Image.open(grid_path).convert("RGB")
    cells = _grid_cells(grid)
    if selected_indices is None:
        selected_indices = DEFAULT_CARDINAL_INDICES if len(cells) >= 6 else list(range(4))
    if len(selected_indices) != len(FRAME_NAMES):
        raise MVAdapterError(f"Expected four selected MV-Adapter views, got {selected_indices}")
    if max(selected_indices) >= len(cells):
        raise MVAdapterError(f"MV-Adapter grid has {len(cells)} cells; cannot select {selected_indices}")

    frame_paths = []
    for frame_name, idx in zip(FRAME_NAMES, selected_indices):
        cell = cells[idx]
        if size is not None and cell.size != (size, size):
            cell = cell.resize((int(size), int(size)), Image.Resampling.LANCZOS)
        out = output_dir / frame_name
        cell.save(out)
        frame_paths.append(str(out))
    return {
        "view_dir": str(output_dir),
        "frame_paths": frame_paths,
        "selected_indices": selected_indices,
        "grid_cell_count": len(cells),
        "frame_names": FRAME_NAMES,
        "view_labels": VIEW_LABELS,
    }


def generate_mvadapter_views(
    image_path,
    work_dir,
    *,
    root: str | None = None,
    variant: str = "sd",
    prompt: str = "high quality object, clean background",
    device: str = "cuda",
    num_views: int = 6,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    seed: int = 1234,
    timeout: int = 1800,
    adapter_path: str | None = None,
    selected_indices: list[int] | None = None,
    size: int = 512,
) -> dict[str, Any]:
    mvadapter_root = find_mvadapter_root(root)
    if mvadapter_root is None:
        checked = [root, os.getenv("MVADAPTER_ROOT"), *DEFAULT_MVADAPTER_ROOTS]
        checked = [str(x) for x in checked if x]
        raise MVAdapterError(f"MV-Adapter repo not found. Checked: {checked}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    script = _script_path(mvadapter_root, variant)
    grid_path = work_dir / "mvadapter_grid.png"
    stdout_path = work_dir / "mvadapter_stdout.txt"
    stderr_path = work_dir / "mvadapter_stderr.txt"

    cmd = [
        sys.executable,
        str(script),
        "--image",
        str(image_path),
        "--text",
        prompt,
        "--num_views",
        str(int(num_views)),
        "--num_inference_steps",
        str(int(num_inference_steps)),
        "--guidance_scale",
        str(float(guidance_scale)),
        "--seed",
        str(int(seed)),
        "--output",
        str(grid_path),
        "--device",
        device,
    ]
    base_model = os.getenv("MVADAPTER_BASE_MODEL")
    if base_model:
        cmd.extend(["--base_model", base_model])
    if adapter_path:
        cmd.extend(["--adapter_path", adapter_path])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(mvadapter_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command_path = work_dir / "mvadapter_command.json"
    command_path.write_text(json.dumps({"cmd": cmd, "cwd": str(mvadapter_root)}, indent=2), encoding="utf-8")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(mvadapter_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout),
        )
    except subprocess.TimeoutExpired as exc:
        raise MVAdapterError(f"MV-Adapter timed out after {timeout}s") from exc

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:]
        detail = tail[0] if tail else f"return code {proc.returncode}"
        raise MVAdapterError(f"MV-Adapter failed: {detail}")
    if not grid_path.exists():
        raise MVAdapterError(f"MV-Adapter did not write output grid: {grid_path}")

    split = split_mvadapter_grid(grid_path, work_dir / "views", selected_indices=selected_indices, size=size)
    metadata = {
        "view_source": "mvadapter",
        "root": str(mvadapter_root),
        "variant": variant,
        "script": str(script),
        "grid_path": str(grid_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": str(command_path),
        "num_views": int(num_views),
        "prompt": prompt,
        "device": device,
        **split,
    }
    metadata_path = work_dir / "mvadapter_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata
