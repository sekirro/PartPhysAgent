#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PARTPHY_ROOT = Path(__file__).resolve().parents[1]
if str(PARTPHY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARTPHY_ROOT))

from partphys.gaussian_assign import (  # noqa: E402
    assign_by_projection,
    build_part_aabbs,
    load_ply_positions,
    save_assignment_outputs,
)
from partphys.types import PartInstance  # noqa: E402


DEFAULT_MATERIAL_SOLVER_YAML = """sampling:
  candidate_budget: 2
posterior:
  schema_expected_material_weight: 0.30
  physgm_crop_weight: 0.20
  vlm_material_weight: 0.0
  global_physgm_weight_for_multi_part: 0.05
  skill_memory_weight: 0.10
"""

SCHEMA_PROMPT = """You are designing a part-level physical simulation agent for a 3D object.
Look at all provided views and output a complete part schema for simulation.
Important rules:
- You will receive four ordered views: front, right, rear, and left. Use all views, not only the first one.
- Do not only segment the main object body.
- Include visible support/contact objects that physically interact with the object, such as a plate, tray, stand, base, holder, or table-contact support.
- Do not omit thin support objects just because they are visually small.
- Small decorations can be separate parts if visible, including tiny stems, candles, berries, signs, ribbons, cream swirls, labels, or toppings.
- Prefer an exhaustive simulation schema: every visually distinct physical component should either become a part or be intentionally merged into a named parent part.
- Return strict JSON only, no markdown.
- Use only these material names: Wood, Metal, Plastic, Glass, Fabric, Leather, Ceramic, Stone, Rubber, Paper, Sand, Snow, Plasticine, Foam.
Required schema:
{
  "object": "object_name",
  "parts": [
    {
      "name": "snake_case_part_name",
      "text_prompts": ["short visual prompt", "synonym"],
      "expected_materials": ["Plasticine"],
      "location": "where the part is in the image/views",
      "shape_prior": "short shape description",
      "physical_role": "what physical role it has",
      "should_simulate_separately": true,
      "visible": true,
      "physics_group": "stable semantic group"
    }
  ],
  "relations": [{"type": "supports|on_top_of|attached_to", "parent": "part_name", "child": "part_name"}]
}
"""

ALLOWED_MATERIALS = {
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
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    log("RUN " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}: {' '.join(cmd)}")
    return proc


def extract_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_vlm_schema(schema: dict[str, Any], object_hint: str | None = None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError("VLM schema must be a JSON object.")
    schema.setdefault("object", object_hint or "object")
    parts = schema.get("parts") or []
    if not isinstance(parts, list) or not parts:
        raise ValueError("VLM schema contains no parts.")
    normalized_parts = []
    seen: set[str] = set()
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        name = str(part.get("name") or f"part_{idx}").strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_") or f"part_{idx}"
        if name in seen:
            suffix = 2
            base = name
            while f"{base}_{suffix}" in seen:
                suffix += 1
            name = f"{base}_{suffix}"
        seen.add(name)
        prompts = part.get("text_prompts") or [name.replace("_", " ")]
        if isinstance(prompts, str):
            prompts = [prompts]
        materials = [str(x) for x in (part.get("expected_materials") or []) if str(x) in ALLOWED_MATERIALS]
        normalized_parts.append(
            {
                "name": name,
                "text_prompts": [str(x) for x in prompts if str(x).strip()] or [name.replace("_", " ")],
                "expected_materials": materials or ["Plasticine"],
                "location": str(part.get("location") or "visible part"),
                "shape_prior": str(part.get("shape_prior") or "visible shape"),
                "physical_role": str(part.get("physical_role") or "physical part"),
                "should_simulate_separately": bool(part.get("should_simulate_separately", True)),
                "visible": bool(part.get("visible", True)),
                "physics_group": str(part.get("physics_group") or name),
            }
        )
    schema["parts"] = normalized_parts
    relations = schema.get("relations") or []
    schema["relations"] = relations if isinstance(relations, list) else []
    return schema


def make_multiview_evidence_sheet(frame_paths: list[Path], labels: list[str], output_path: Path) -> Path:
    cell = 512
    label_h = 34
    sheet = Image.new("RGB", (cell * 2, (cell + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (frame_path, label) in enumerate(zip(frame_paths, labels)):
        row = idx // 2
        col = idx % 2
        x0 = col * cell
        y0 = row * (cell + label_h)
        draw.rectangle([x0, y0, x0 + cell - 1, y0 + label_h - 1], fill=(245, 245, 245), outline=(180, 180, 180))
        draw.text((x0 + 12, y0 + 9), label, fill=(0, 0, 0))
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        px = x0 + (cell - image.width) // 2
        py = y0 + label_h + (cell - image.height) // 2
        sheet.paste(image, (px, py))
        draw.rectangle([x0, y0 + label_h, x0 + cell - 1, y0 + label_h + cell - 1], outline=(180, 180, 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def call_vlm_schema(args: argparse.Namespace, scene_dir: Path) -> Path:
    key = os.environ.get(args.vlm_api_key_env)
    if not key:
        raise RuntimeError(f"VLM API key env var {args.vlm_api_key_env} is not set.")
    multiview_dir = Path(args.multiview_dir).expanduser().resolve()
    frame_names = candidate_frame_names(multiview_dir)
    view_labels = ["front", "right", "rear", "left"]
    frame_paths = [multiview_dir / name for name in frame_names]
    schema_dir = scene_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    make_multiview_evidence_sheet(frame_paths, view_labels, schema_dir / "vlm_multiview_evidence.png")
    prompt = SCHEMA_PROMPT
    if args.object:
        prompt += f"\nObject hint: {args.object}\n"
    prompt += "\nImages are provided below in this exact order: front, right, rear, left."
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, frame_path in zip(view_labels, frame_paths):
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"View: {label}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    payload = {
        "model": args.vlm_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    if args.vlm_disable_thinking:
        payload["extra_body"] = {"enable_thinking": False}
    request = urllib.request.Request(
        args.vlm_api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    log("Generating part schema with VLM")
    started = time.time()
    with urllib.request.urlopen(request, timeout=args.vlm_timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    write_json(schema_dir / "vlm_raw_response.json", raw)
    message = raw["choices"][0]["message"]
    schema = extract_json_from_text(message.get("content") or "")
    schema = normalize_vlm_schema(schema, args.object)
    schema_path = schema_dir / "vlm_generated_schema.json"
    write_json(schema_path, schema)
    log(f"VLM schema ready in {time.time() - started:.1f}s: {schema_path}")
    log("VLM parts: " + ", ".join(part["name"] for part in schema.get("parts", [])))
    return schema_path


def candidate_frame_names(multiview_dir: Path) -> list[str]:
    canonical = ["000.png", "006.png", "012.png", "018.png"]
    labels = ["front.png", "right.png", "rear.png", "left.png"]
    if all((multiview_dir / name).exists() for name in canonical):
        return canonical
    if all((multiview_dir / name).exists() for name in labels):
        return labels
    raise FileNotFoundError(f"Could not find 4-view images in {multiview_dir}; expected 000/006/012/018.png or front/right/rear/left.png")


def run_partphys_masks(args: argparse.Namespace, schema_path: Path, scene_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(PARTPHY_ROOT / "partphys_pipeline.py"),
        "--image",
        str(Path(args.image).expanduser().resolve()),
        "--multiview-dir",
        str(Path(args.multiview_dir).expanduser().resolve()),
        "--scene-name",
        args.scene_name,
        "--object",
        args.object or "object",
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--part-schema-json",
        str(schema_path),
        "--no-vlm",
        "--agent-mode",
        "pipeline",
        "--groundingdino-model",
        args.groundingdino_model,
        "--groundingdino-box-threshold",
        str(args.groundingdino_box_threshold),
        "--groundingdino-text-threshold",
        str(args.groundingdino_text_threshold),
        "--segmentation-mode",
        "candidate_pool",
        "--residual-policy",
        args.residual_policy,
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--candidate-contact-sheet-top-k",
        str(args.candidate_contact_sheet_top_k),
        "--max-vlm-candidates-per-part",
        "12",
        "--sam-backend",
        "sam2",
        "--sam-checkpoint",
        args.sam_checkpoint,
        "--sam-config",
        args.sam_config,
        "--sam2-root",
        args.sam2_root,
        "--sam-points-per-side",
        str(args.sam_points_per_side),
        "--sam-pred-iou-thresh",
        str(args.sam_pred_iou_thresh),
        "--sam-stability-score-thresh",
        str(args.sam_stability_score_thresh),
        "--sam-crop-n-layers",
        str(args.sam_crop_n_layers),
        "--sam-min-mask-region-area",
        str(args.sam_min_mask_region_area),
        "--min-part-area-ratio",
        str(args.min_part_area_ratio),
        "--segmentation-max-retries",
        "2",
        "--segmentation-vlm-weight",
        "0.0",
        "--segmentation-min-accept-score",
        str(args.segmentation_min_accept_score),
        "--mask-only",
    ]
    log("Running PartPhysAgent mask stage with VLM-generated schema")
    run_cmd(cmd, cwd=PARTPHY_ROOT, timeout=args.partphys_timeout)
    summary = read_json(scene_dir / "parts" / "selection_summary.json", {})
    log("Selected parts: " + ", ".join(summary.get("selected_parts", [])))


def make_physgm_input_scene(args: argparse.Namespace, scene_dir: Path) -> Path:
    import yaml

    source = Path(args.multiview_dir).expanduser().resolve()
    frame_names = candidate_frame_names(source)
    input_root = scene_dir / "physgm_input_scene"
    input_scene = input_root / f"{args.scene_name}_whole_input"
    input_scene.mkdir(parents=True, exist_ok=True)
    target_names = ["000.png", "006.png", "012.png", "018.png"]
    for src_name, dst_name in zip(frame_names, target_names):
        shutil.copy2(source / src_name, input_scene / dst_name)
    pose_src = source / "pose.json"
    if not pose_src.exists():
        pose_src = Path(args.physgm_root).expanduser().resolve() / "example_data" / "cake" / "pose.json"
    pose = json.loads(pose_src.read_text(encoding="utf-8"))
    pose["scene_name"] = f"{args.scene_name}_whole_input"
    for frame, dst_name in zip(pose.get("frames", []), target_names):
        frame["file_path"] = dst_name
    write_json(input_scene / "pose.json", pose)
    (input_root / "data.txt").write_text(f"./{input_scene.name}/pose.json\n", encoding="utf-8")

    physgm_root = Path(args.physgm_root).expanduser().resolve()
    with open(args.physgm_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"]["data_path"] = str(input_root / "data.txt")
    infer_config = scene_dir / "physgm_infer_from_multiview.yaml"
    with open(infer_config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    log(f"PhysGM input scene: {input_scene}")
    return infer_config


def physgm_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    physgm_root = str(Path(args.physgm_root).expanduser().resolve())
    partphys_root = str(PARTPHY_ROOT)
    material_root = str(Path(args.material_agent_root).expanduser().resolve())
    env["PYTHONPATH"] = f"{physgm_root}:{partphys_root}:{material_root}:{env.get('PYTHONPATH', '')}"
    physgm_bin = str(Path(args.physgm_python).expanduser().resolve().parent)
    env["PATH"] = f"{physgm_bin}:{env.get('PATH', '')}"
    return env


def run_physgm_whole(args: argparse.Namespace, scene_dir: Path, infer_config: Path) -> None:
    physgm_root = Path(args.physgm_root).expanduser().resolve()
    whole_dir = scene_dir / "physgm_whole"
    if whole_dir.exists() and any(whole_dir.iterdir()):
        raise RuntimeError(f"Refusing to reuse existing PhysGM whole dir: {whole_dir}")
    cmd = [
        str(Path(args.physgm_python).expanduser().resolve()),
        str(physgm_root / "pipeline.py"),
        "--config",
        str(infer_config),
        "--checkpoint",
        args.checkpoint,
        "--template-config",
        args.template_config,
        "--scene-name",
        "physgm_whole",
        "--output-dir",
        str(scene_dir),
    ]
    log("Running original PhysGM whole-object pipeline")
    run_cmd(cmd, cwd=physgm_root, env=physgm_env(args), timeout=args.physgm_timeout)


def write_input_batch_meta(args: argparse.Namespace, scene_dir: Path, infer_config: Path) -> None:
    code = r'''
from pathlib import Path
import sys
import yaml
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from data.dataset_infer import Dataset
config_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
with open(config_path, 'r', encoding='utf-8') as handle:
    config = edict(yaml.safe_load(handle))
config.evaluation = True
batch = next(iter(DataLoader(Dataset(config), batch_size=1, shuffle=False)))
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out_path,
    input_intr=batch['input_intr'].detach().cpu().numpy(),
    input_c2ws=batch['input_c2ws'].detach().cpu().numpy(),
    scene_scale=batch['scene_scale'].detach().cpu().numpy(),
    pos_avg_inv=batch['pos_avg_inv'].detach().cpu().numpy(),
)
print(out_path)
'''
    out = scene_dir / "physgm_whole" / "input_batch_meta.npz"
    cmd = [str(Path(args.physgm_python).expanduser().resolve()), "-", str(infer_config), str(out)]
    log("Writing PhysGM camera projection metadata")
    proc = subprocess.run(
        cmd,
        input=code,
        text=True,
        cwd=str(Path(args.physgm_root).expanduser().resolve()),
        env=physgm_env(args),
        timeout=args.physgm_timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError("Failed to write input_batch_meta.npz")


def run_assignment(scene_dir: Path) -> dict[str, Any]:
    whole = scene_dir / "physgm_whole"
    selection = read_json(scene_dir / "parts" / "selection_summary.json", {})
    parts: list[PartInstance] = []
    part_masks: list[dict[str, Any]] = []
    for item in selection.get("parts", []):
        name = str(item.get("name", "")).lower()
        if "unknown" in name or "residual" in name:
            continue
        part = PartInstance(
            part_id=int(item["part_id"]),
            name=item["name"],
            mask_path=item["mask_path"],
            bbox=item.get("bbox", {}),
            area=int(item.get("area", 0)),
            confidence=float(item.get("confidence", 1.0) or 1.0),
            candidate_ids=item.get("candidate_ids", []),
            expected_materials=item.get("expected_materials", []),
            physics_group=item.get("physics_group"),
            warnings=item.get("warnings", []),
            metadata=item.get("metadata", {}),
        )
        parts.append(part)
        part_masks.append(
            {
                "part_id": part.part_id,
                "name": part.name,
                "mask_path": part.mask_path,
                "view_masks": part.metadata.get("view_masks", {}),
                "area": part.area,
                "confidence": part.confidence,
                "physics_group": part.physics_group,
            }
        )
    if not parts:
        raise RuntimeError("No non-residual parts found for assignment.")
    positions = load_ply_positions(whole / "point_clouds.ply")
    image = Image.open(scene_dir / "input" / "input.png")
    assign = assign_by_projection(positions, part_masks, whole / "input_batch_meta.npz", image.size)
    ids = assign["gaussian_part_ids"]
    aabbs = build_part_aabbs(positions, ids, parts, min_count=20, padding_ratio=0.15, min_half_size=0.02)
    summary = {
        "mode": "projection_vlm_schema_end_to_end",
        "whole_physgm_dir": str(whole),
        "assigned_ratio": assign.get("assigned_ratio", 0.0),
        "per_part_counts": assign.get("per_part_counts", {}),
        "projection_views": assign.get("view_labels", []),
        "projection_view_hits": assign.get("per_view_hits", {}),
        "projection_image_size": assign.get("projection_image_size"),
        "projection_mean_view_support": assign.get("mean_view_support"),
        "projection_view_support_counts": assign.get("view_support_counts", {}),
        "projection_margin_ratio": assign.get("margin_ratio", {}),
        "projection_low_confidence_count": assign.get("low_confidence_count", 0),
        "projection_smoothed_count": assign.get("smoothed_count", 0),
        "projection_knn_unknown_reassigned_count": assign.get("knn_unknown_reassigned_count", 0),
        "projection_knn_island_reassigned_count": assign.get("knn_island_reassigned_count", 0),
        "aabb_count": len(aabbs),
        "warnings": assign.get("warnings", []),
    }
    log("Saving Gaussian-to-part assignment")
    save_assignment_outputs(scene_dir / "assignment", ids, aabbs, summary, parts, whole / "point_clouds.ply")
    partphys_summary = read_json(scene_dir / "partphys_summary.json", {})
    predicted = whole / "predicted_phys.json"
    partphys_summary["whole_physgm"] = {
        "scene_dir": str(whole),
        "point_cloud_path": str(whole / "point_clouds.ply"),
        "predicted_phys_path": str(predicted),
        "raw": read_json(predicted, {}) or {},
    }
    partphys_summary["assignment_summary"] = summary
    warnings = partphys_summary.get("warnings", []) or []
    warnings.append("Part schema was generated by VLM; candidate ranking VLM was disabled to avoid slow contact-sheet calls.")
    warnings.append("Whole PhysGM and Gaussian assignment were run fresh in this end-to-end script.")
    partphys_summary["warnings"] = warnings
    write_json(scene_dir / "partphys_summary.json", partphys_summary)
    log(f"Assignment ratio: {summary['assigned_ratio']:.4f}; counts: {summary['per_part_counts']}")
    return summary


def run_material_agent(args: argparse.Namespace, scene_dir: Path) -> Path:
    material_root = Path(args.material_agent_root).expanduser().resolve()
    output_dir = scene_dir / args.material_output_name
    solver_config = scene_dir / "material_agent_solver.yaml"
    solver_config.write_text(DEFAULT_MATERIAL_SOLVER_YAML, encoding="utf-8")
    cmd = [
        str(Path(args.physgm_python).expanduser().resolve()),
        "-m",
        "material_agent.cli",
        "--partphys-scene",
        str(scene_dir),
        "--output-dir",
        str(output_dir),
        "--config",
        str(solver_config),
        "--backend",
        "part_id",
        "--candidate-budget",
        str(args.material_candidate_budget),
        "--simulate",
        "--physgm-root",
        args.physgm_root,
        "--partphys-root",
        str(PARTPHY_ROOT),
        "--timeout-sec",
        str(args.sim_timeout),
    ]
    if args.select_candidate:
        cmd.extend(["--selection", "human", "--select-candidate", args.select_candidate])
    log("Running MaterialAgent per-particle simulation")
    run_cmd(cmd, cwd=material_root, env=physgm_env(args), timeout=args.material_timeout)
    return output_dir


def count_ply_vertices(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def final_report(scene_dir: Path, material_output: Path) -> None:
    part_selection = read_json(scene_dir / "parts" / "selection_summary.json", {}) or {}
    material_selection = read_json(material_output / "selection.json", {}) or {}
    assignment = read_json(scene_dir / "assignment" / "assignment_summary.json", {}) or {}
    selected_candidate = material_selection.get("candidate_id")
    run_result = read_json(material_output / "candidate_outputs" / str(selected_candidate) / "run_result.json", {}) if selected_candidate else {}
    if not run_result:
        candidates = sorted((material_output / "candidate_outputs").glob("*/run_result.json"))
        if candidates:
            run_result = read_json(candidates[0], {}) or {}
            selected_candidate = run_result.get("candidate_id", selected_candidate)
    report = {
        "scene_dir": str(scene_dir),
        "schema_path": str(scene_dir / "schema" / "vlm_generated_schema.json"),
        "selected_parts": part_selection.get("selected_parts"),
        "selection_ok": part_selection.get("ok"),
        "selection_reason": part_selection.get("reason"),
        "unknown_ratio": part_selection.get("unknown_ratio"),
        "whole_gaussians": count_ply_vertices(scene_dir / "physgm_whole" / "point_clouds.ply"),
        "assignment_ratio": assignment.get("assigned_ratio"),
        "per_part_counts": assignment.get("per_part_counts"),
        "material_output_dir": str(material_output),
        "material_selected_candidate": selected_candidate,
        "material_selection_score": material_selection.get("score"),
        "simulation_status": run_result.get("status"),
        "simulation_video": run_result.get("video_path"),
    }
    write_json(scene_dir / "end_to_end_summary.json", report)
    log("End-to-end summary:")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete VLM-schema PartPhysAgent + PhysGM + MaterialAgent pipeline.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--multiview-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--object", default="object")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/results_partphys")
    parser.add_argument("--force", action="store_true", help="Delete an existing scene dir before running. Use carefully.")

    parser.add_argument("--vlm-model", default="qwen3.7-plus")
    parser.add_argument("--vlm-api-base", default="https://llm-jrkem52i075alacx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--vlm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--vlm-timeout", type=int, default=300)
    parser.add_argument("--vlm-disable-thinking", action="store_true", default=False)

    parser.add_argument("--physgm-root", default="/root/PhysGM")
    parser.add_argument("--physgm-python", default="/root/miniconda3/envs/physgm/bin/python")
    parser.add_argument("--physgm-config", default="/root/PhysGM/configs/infer.yaml")
    parser.add_argument("--checkpoint", default="/root/PhysGM/checkpoints/checkpoint.pt")
    parser.add_argument("--template-config", default="/root/PhysGM/configs/physical/down_template.json")

    parser.add_argument("--material-agent-root", default="/root/MaterialAgent")
    parser.add_argument("--material-output-name", default="material_agent_partid_full")
    parser.add_argument("--material-candidate-budget", type=int, default=2)
    parser.add_argument("--select-candidate", default=None, help="Optional manual candidate id; omitted uses generic auto selection.")

    parser.add_argument("--groundingdino-model", default="/root/autodl-tmp/models/grounding-dino-base")
    parser.add_argument("--groundingdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-text-threshold", type=float, default=0.25)
    parser.add_argument("--sam-checkpoint", default="/root/autodl-tmp/models/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-root", default="/root/autodl-tmp/repos/sam2")
    parser.add_argument("--sam-points-per-side", type=int, default=16)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--sam-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=100)
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--candidate-contact-sheet-top-k", type=int, default=30)
    parser.add_argument("--min-part-area-ratio", type=float, default=0.001)
    parser.add_argument("--segmentation-min-accept-score", type=float, default=0.40)
    parser.add_argument("--residual-policy", choices=["ignore", "unknown", "fill_nearest"], default="fill_nearest")

    parser.add_argument("--partphys-timeout", type=int, default=2400)
    parser.add_argument("--physgm-timeout", type=int, default=2400)
    parser.add_argument("--material-timeout", type=int, default=2400)
    parser.add_argument("--sim-timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    scene_dir = output_root / args.scene_name
    if scene_dir.exists():
        if not args.force:
            raise RuntimeError(f"Scene already exists; choose a new --scene-name or pass --force: {scene_dir}")
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    schema_path = call_vlm_schema(args, scene_dir)
    run_partphys_masks(args, schema_path, scene_dir)
    infer_config = make_physgm_input_scene(args, scene_dir)
    run_physgm_whole(args, scene_dir, infer_config)
    write_input_batch_meta(args, scene_dir, infer_config)
    run_assignment(scene_dir)
    material_output = run_material_agent(args, scene_dir)
    final_report(scene_dir, material_output)
    log(f"Complete pipeline finished in {(time.time() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
