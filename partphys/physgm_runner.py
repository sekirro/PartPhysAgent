from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .material_table import density_for_material, default_E_for_material, default_nu_for_material, normalize_material_name
from .multiview import generate_mvadapter_views
from .scene_builder import build_physgm_input_scene
from .types import PhysGMResult


CLASS_TO_MATERIAL = {
    0: "Wood",
    1: "Metal",
    2: "Plastic",
    3: "Glass",
    4: "Fabric",
    5: "Leather",
    6: "Ceramic",
    7: "Stone",
    8: "Rubber",
    9: "Paper",
    10: "Sand",
    11: "Snow",
    12: "Plasticine",
    13: "Foam",
}

E_MEAN = 7.387210
E_STD = 2.456477
NU_MEAN = 0.398
NU_STD = 0.111


def _write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _find_physgm_root(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "pipeline.py").exists():
            return p
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "PhysGM"
        if (candidate / "pipeline.py").exists():
            return candidate
        if (parent / "pipeline.py").exists() and (parent / "model").exists():
            return parent
    sibling = Path.cwd().parent / "PhysGM"
    if (sibling / "pipeline.py").exists():
        return sibling.resolve()
    return None


def _resolve_path(path: str | None, physgm_root: Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.exists():
        return str(p.resolve())
    if physgm_root is not None:
        q = physgm_root / path
        if q.exists():
            return str(q.resolve())
    return str(p)


def _mock_material_from_name(name: str) -> str:
    text = name.lower()
    if any(k in text for k in ["head", "metal", "steel", "iron"]):
        return "Metal"
    if any(k in text for k in ["handle", "wood", "wooden"]):
        return "Wood"
    if any(k in text for k in ["sole", "rubber", "tire", "wheel"]):
        return "Rubber"
    if any(k in text for k in ["upper", "fabric", "cloth", "lace"]):
        return "Fabric"
    return "Plastic"


def _write_mock_ply(path: str, grid: int = 9):
    pts = []
    for z in np.linspace(0.25, 1.75, grid):
        for y in np.linspace(0.25, 1.75, grid):
            for x in np.linspace(0.25, 1.75, grid):
                pts.append((x, y, z))
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


class PhysGMRunner:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        template_config_path,
        device,
        output_base_dir,
        mock: bool = False,
        save_gaussian_default: bool = True,
        physgm_root: str | None = None,
        amp_dtype: str = "bf16",
        mvadapter_root: str | None = None,
        mvadapter_variant: str = "sd",
        mvadapter_device: str | None = None,
        mvadapter_prompt: str = "high quality object, clean background",
        mvadapter_num_views: int = 6,
        mvadapter_steps: int = 50,
        mvadapter_guidance_scale: float = 3.0,
        mvadapter_seed: int = 1234,
        mvadapter_timeout: int = 1800,
        mvadapter_adapter_path: str | None = None,
        mvadapter_required: bool = False,
    ):
        self.mock = bool(mock)
        self.device = device
        self.output_base_dir = str(output_base_dir)
        self.save_gaussian_default = save_gaussian_default
        self.amp_dtype = amp_dtype
        self.mvadapter_root = mvadapter_root
        self.mvadapter_variant = mvadapter_variant
        self.mvadapter_device = mvadapter_device
        self.mvadapter_prompt = mvadapter_prompt
        self.mvadapter_num_views = int(mvadapter_num_views)
        self.mvadapter_steps = int(mvadapter_steps)
        self.mvadapter_guidance_scale = float(mvadapter_guidance_scale)
        self.mvadapter_seed = int(mvadapter_seed)
        self.mvadapter_timeout = int(mvadapter_timeout)
        self.mvadapter_adapter_path = mvadapter_adapter_path
        self.mvadapter_required = bool(mvadapter_required)
        self.physgm_root = _find_physgm_root(physgm_root)
        self.config_path = _resolve_path(config_path, self.physgm_root)
        self.checkpoint_path = _resolve_path(checkpoint_path, self.physgm_root)
        self.template_config_path = _resolve_path(template_config_path, self.physgm_root)
        self.model = None
        self.base_config = None
        self.Dataset = None
        self.DataLoader = None
        self.torch = None
        self.template_pose_json = None
        if self.physgm_root is not None:
            pose = self.physgm_root / "example_data" / "cake" / "pose.json"
            if pose.exists():
                self.template_pose_json = str(pose)
        if not self.mock:
            self._load_model()

    def _load_model(self):
        if self.physgm_root is None:
            raise RuntimeError("PhysGM root not found. Provide --physgm-root.")
        if not self.config_path or not Path(self.config_path).exists():
            raise RuntimeError(f"PhysGM config not found: {self.config_path}")
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise RuntimeError(f"PhysGM checkpoint not found: {self.checkpoint_path}. Use --mock-physgm for tests.")
        sys.path.insert(0, str(self.physgm_root))
        import torch  # type: ignore
        import yaml  # type: ignore
        from easydict import EasyDict as edict  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
        from data.dataset_infer import Dataset  # type: ignore
        from model.physgm import PhysGM  # type: ignore

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.base_config = edict(yaml.safe_load(f))
        self.base_config.evaluation = True
        self.torch = torch
        self.Dataset = Dataset
        self.DataLoader = DataLoader
        self.model = PhysGM(self.base_config, device=self.device).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        self.model.eval()

    def _attach_scene_extra(self, scene_info: dict, key: str, value: Any | None) -> dict:
        if value is None:
            return scene_info
        scene_info[key] = value
        metadata_path = scene_info.get("view_metadata")
        if metadata_path:
            try:
                path = Path(metadata_path)
                metadata = json.loads(path.read_text(encoding="utf-8"))
                metadata[key] = value
                path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            except Exception:
                pass
        return scene_info

    def _maybe_generate_mvadapter(self, image_path, output_dir, use_mvadapter: bool) -> dict[str, Any] | None:
        if not use_mvadapter:
            return None
        work_dir = Path(output_dir) / "mvadapter"
        try:
            return generate_mvadapter_views(
                image_path,
                work_dir,
                root=self.mvadapter_root,
                variant=self.mvadapter_variant,
                prompt=self.mvadapter_prompt,
                device=self.mvadapter_device or self.device,
                num_views=self.mvadapter_num_views,
                num_inference_steps=self.mvadapter_steps,
                guidance_scale=self.mvadapter_guidance_scale,
                seed=self.mvadapter_seed,
                timeout=self.mvadapter_timeout,
                adapter_path=self.mvadapter_adapter_path,
                selected_indices=[0, 2, 3, 4] if self.mvadapter_num_views >= 6 else None,
                size=512,
            )
        except Exception as exc:
            if self.mvadapter_required:
                raise RuntimeError(f"MV-Adapter multiview generation failed: {exc}") from exc
            return {"view_source": "mvadapter_failed", "error": str(exc), "fallback": "single_image_proxy"}

    def _build_input_scene(self, image_path, scene_name, output_dir, use_mvadapter: bool = False, multiview_dir: str | None = None) -> dict:
        mvadapter_info = self._maybe_generate_mvadapter(image_path, output_dir, use_mvadapter)
        multiview_source = "provided_multiview"
        if mvadapter_info and mvadapter_info.get("view_source") == "mvadapter" and mvadapter_info.get("view_dir"):
            multiview_dir = mvadapter_info["view_dir"]
            multiview_source = "mvadapter"
        scene_info = build_physgm_input_scene(
            image_path,
            Path(output_dir) / "input_scene",
            scene_name,
            template_pose_json=self.template_pose_json,
            duplicate_single_image=True,
            size=512,
            multiview_dir=multiview_dir,
            multiview_source=multiview_source,
        )
        return self._attach_scene_extra(scene_info, "mvadapter", mvadapter_info)

    def _infer_mock(self, image_path, scene_name, output_dir, save_gaussian: bool) -> PhysGMResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_info = self._build_input_scene(image_path, scene_name, output_dir, use_mvadapter=False)
        material = _mock_material_from_name(f"{scene_name} {image_path}")
        E = default_E_for_material(material)
        nu = default_nu_for_material(material)
        density = density_for_material(material)
        raw = {
            "material": material,
            "E": E,
            "nu": nu,
            "density": density,
            "mock": True,
            "input_scene": scene_info,
        }
        predicted = output_dir / "predicted_phys.json"
        _write_json(predicted, raw)
        point_cloud = None
        if save_gaussian:
            point_cloud = output_dir / "point_clouds.ply"
            _write_mock_ply(str(point_cloud))
        meta = output_dir / "input_batch_meta.npz"
        intr = np.array([[512.0, 512.0, 256.0, 256.0]] * 4, dtype=np.float32)
        c2ws = np.stack([np.eye(4, dtype=np.float32) for _ in range(4)])
        np.savez(meta, input_intr=intr, input_c2ws=c2ws, height=512, width=512, mock=True)
        _write_json(output_dir / "raw_output_summary.json", raw)
        return PhysGMResult(str(output_dir), str(point_cloud) if point_cloud else None, str(predicted), material, E, nu, density, raw)

    def infer_image(
        self,
        image_path,
        scene_name,
        output_dir,
        save_gaussian: bool | None = None,
        use_mvadapter: bool = False,
        multiview_dir: str | None = None,
    ) -> PhysGMResult:
        if save_gaussian is None:
            save_gaussian = self.save_gaussian_default
        if self.mock:
            return self._infer_mock(image_path, scene_name, output_dir, bool(save_gaussian))
        if self.model is None or self.base_config is None or self.torch is None:
            self._load_model()
        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_info = self._build_input_scene(
            image_path,
            scene_name,
            output_dir,
            use_mvadapter=use_mvadapter,
            multiview_dir=multiview_dir,
        )

        config = copy.deepcopy(self.base_config)
        config.data.data_path = scene_info["data_txt"]
        dataset = self.Dataset(config)
        dataloader = self.DataLoader(dataset, batch_size=1, shuffle=False)
        amp = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
        autocast_cm = torch.autocast(device_type="cuda", dtype=amp) if "cuda" in str(self.device) else nullcontext()
        with torch.no_grad():
            batch = next(iter(dataloader))
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            with autocast_cm:
                ret_dict = self.model(batch)

            E_mu_norm = ret_dict["E_mu"][0].float().item() if "E_mu" in ret_dict else 0.0
            E_value = float(round((10 ** (E_mu_norm * E_STD + E_MEAN)) * 0.1, 2))
            nu_mu_norm = ret_dict["nu_mu"][0].float().item() if "nu_mu" in ret_dict else 0.0
            nu_value = float(round(nu_mu_norm * NU_STD + NU_MEAN, 4))
            mat_idx = int(torch.argmax(ret_dict["phys_logits"][0].float()).item())
            material = normalize_material_name(CLASS_TO_MATERIAL.get(mat_idx, "Plastic"))
            density = density_for_material(material)

            raw = {"E": E_value, "nu": nu_value, "material": material, "density": density, "input_scene": scene_info}
            predicted = output_dir / "predicted_phys.json"
            _write_json(predicted, raw)

            point_cloud = None
            if save_gaussian:
                self.model.save_visualization(batch, ret_dict, str(output_dir), save_gaussian=True, save_video=False)
                for root, _, files in os.walk(output_dir):
                    for file_name in files:
                        src = Path(root) / file_name
                        if file_name.endswith(".ply") and file_name != "point_clouds.ply":
                            dst = output_dir / "point_clouds.ply"
                            if src.resolve() != dst.resolve():
                                shutil.move(str(src), str(dst))
                point_cloud = output_dir / "point_clouds.ply"

            meta = output_dir / "input_batch_meta.npz"
            meta_data = {}
            for key in ["input_intr", "input_c2ws", "scene_scale", "pos_avg_inv"]:
                if key in batch:
                    value = batch[key]
                    if isinstance(value, torch.Tensor):
                        meta_data[key] = value.detach().cpu().numpy()
            if meta_data:
                np.savez(meta, **meta_data)
            _write_json(output_dir / "raw_output_summary.json", raw)
        return PhysGMResult(str(output_dir), str(point_cloud) if point_cloud and point_cloud.exists() else None, str(predicted), material, E_value, nu_value, density, raw)
