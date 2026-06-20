from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .types import BBox


DEFAULT_SAM2_ROOT = "/root/autodl-tmp/repos/sam2"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

_CKPT_TO_CONFIG = {
    "sam2_hiera_tiny.pt": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small.pt": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus.pt": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": DEFAULT_SAM2_CONFIG,
}


def _resolve_sam2_config(checkpoint: str, config: str | None) -> str:
    if config:
        return config
    return _CKPT_TO_CONFIG.get(Path(checkpoint).name, DEFAULT_SAM2_CONFIG)


def _prepare_sam2_import(sam2_root: str | None) -> None:
    root = Path(sam2_root or os.environ.get("SAM2_REPO_ROOT") or DEFAULT_SAM2_ROOT).expanduser()
    if root.exists():
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def _coerce_bbox(bbox: BBox | dict | list | tuple) -> BBox:
    if isinstance(bbox, dict):
        return BBox.from_dict(bbox)
    if isinstance(bbox, BBox):
        return bbox
    return BBox(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


class BaseSAMTool:
    def _load_image_np(self, image_path) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(image_path).convert("RGB"))

    def segment_from_box(self, image_path, bbox: BBox | dict | list | tuple) -> list[np.ndarray]:
        raise NotImplementedError

    def segment_from_points(self, image_path, points, labels) -> list[np.ndarray]:
        raise NotImplementedError

    def automatic_masks(self, image_path) -> list[dict[str, Any]]:
        raise NotImplementedError


class SAM2Tool(BaseSAMTool):
    def __init__(
        self,
        checkpoint: str | None,
        config: str | None = None,
        device: str = "cuda",
        sam2_root: str | None = None,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 100,
    ):
        if not checkpoint or not Path(checkpoint).exists():
            raise RuntimeError("SAM2 checkpoint missing. Provide --sam-checkpoint or use --masks-json.")
        self.checkpoint = checkpoint
        self.config = _resolve_sam2_config(checkpoint, config)
        self.device = device
        try:
            _prepare_sam2_import(sam2_root)
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            sam = build_sam2(self.config, ckpt_path=checkpoint, device=device, apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(sam)
            generator_kwargs = dict(
                points_per_side=int(points_per_side),
                points_per_batch=64,
                pred_iou_thresh=float(pred_iou_thresh),
                stability_score_thresh=float(stability_score_thresh),
                crop_n_layers=int(crop_n_layers),
                min_mask_region_area=int(min_mask_region_area),
                output_mode="binary_mask",
            )
            try:
                self.generator = SAM2AutomaticMaskGenerator(sam, **generator_kwargs)
            except TypeError:
                generator_kwargs.pop("min_mask_region_area", None)
                self.generator = SAM2AutomaticMaskGenerator(sam, **generator_kwargs)
        except Exception as exc:
            raise RuntimeError(f"SAM2 unavailable: {exc}") from exc

    def segment_from_box(self, image_path, bbox: BBox | dict | list | tuple) -> list[np.ndarray]:
        bbox = _coerce_bbox(bbox)
        image = self._load_image_np(image_path)
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32),
            multimask_output=True,
            normalize_coords=False,
        )
        order = np.argsort(scores)[::-1]
        return [masks[i].astype(bool) for i in order]

    def segment_from_points(self, image_path, points, labels) -> list[np.ndarray]:
        image = self._load_image_np(image_path)
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
            normalize_coords=False,
        )
        order = np.argsort(scores)[::-1]
        return [masks[i].astype(bool) for i in order]

    def automatic_masks(self, image_path) -> list[dict[str, Any]]:
        image = self._load_image_np(image_path)
        return _normalize_auto_masks(self.generator.generate(image))


class SAM1Tool(BaseSAMTool):
    def __init__(
        self,
        checkpoint: str | None,
        model_type: str = "vit_b",
        device: str = "cuda",
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 100,
    ):
        if not checkpoint or not Path(checkpoint).exists():
            raise RuntimeError("SAM1 checkpoint missing. Provide --sam-checkpoint or use --masks-json.")
        try:
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry  # type: ignore

            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            self.generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=int(points_per_side),
                pred_iou_thresh=float(pred_iou_thresh),
                stability_score_thresh=float(stability_score_thresh),
                crop_n_layers=int(crop_n_layers),
                min_mask_region_area=int(min_mask_region_area),
                output_mode="binary_mask",
            )
        except Exception as exc:
            raise RuntimeError(f"SAM1 unavailable: {exc}") from exc

    def segment_from_box(self, image_path, bbox: BBox | dict | list | tuple) -> list[np.ndarray]:
        bbox = _coerce_bbox(bbox)
        image = self._load_image_np(image_path)
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32),
            multimask_output=True,
        )
        order = np.argsort(scores)[::-1]
        return [masks[i].astype(bool) for i in order]

    def segment_from_points(self, image_path, points, labels) -> list[np.ndarray]:
        image = self._load_image_np(image_path)
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
        order = np.argsort(scores)[::-1]
        return [masks[i].astype(bool) for i in order]

    def automatic_masks(self, image_path) -> list[dict[str, Any]]:
        image = self._load_image_np(image_path)
        return _normalize_auto_masks(self.generator.generate(image))


def _normalize_auto_masks(raw) -> list[dict[str, Any]]:
    out = []
    for item in raw:
        out.append(
            {
                "segmentation": np.asarray(item["segmentation"]).astype(bool),
                "predicted_iou": item.get("predicted_iou"),
                "stability_score": item.get("stability_score"),
                "bbox": item.get("bbox"),
                "area": item.get("area"),
                "crop_box": item.get("crop_box"),
            }
        )
    return out


def create_sam_tool(
    checkpoint: str | None,
    backend: str = "sam2",
    config: str | None = None,
    device: str = "cuda",
    sam2_root: str | None = None,
    model_type: str = "vit_b",
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.92,
    crop_n_layers: int = 0,
    min_mask_region_area: int = 100,
) -> BaseSAMTool:
    if backend == "sam1":
        return SAM1Tool(
            checkpoint,
            model_type=model_type,
            device=device,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            min_mask_region_area=min_mask_region_area,
        )
    return SAM2Tool(
        checkpoint,
        config=config,
        device=device,
        sam2_root=sam2_root,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=min_mask_region_area,
    )


SAMTool = SAM2Tool
