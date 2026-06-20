from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .image_utils import bbox_from_xyxy


def _ensure_groundingdino_repo_on_path(config_path: str | None) -> Path | None:
    if not config_path:
        return None

    config = Path(config_path).expanduser().resolve()
    start = config if config.is_dir() else config.parent
    for candidate in (start, *start.parents):
        if (candidate / "groundingdino" / "__init__.py").exists():
            repo_root = str(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return candidate
    return None


def _groundingdino_custom_ops_available() -> bool:
    try:
        from groundingdino import _C  # type: ignore

        return hasattr(_C, "ms_deform_attn_forward")
    except Exception:
        return False


class BaseDetector:
    def detect(self, image_path, text_prompt) -> list[dict[str, Any]]:
        raise NotImplementedError


class NoDetector(BaseDetector):
    def detect(self, image_path, text_prompt) -> list[dict[str, Any]]:
        return []


def _format_groundingdino_prompt(text_prompt) -> str:
    prompt = str(text_prompt).strip().lower()
    if prompt and prompt[-1] not in ".!?":
        prompt = f"{prompt}."
    return prompt


def _item_at(values, idx: int, default):
    if values is None:
        return default
    try:
        value = values[idx]
    except Exception:
        return default
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return default
    return value


class HuggingFaceGroundingDINODetector(BaseDetector):
    def __init__(
        self,
        model_id: str = "/root/autodl-tmp/models/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.model_id = model_id
        self.requested_device = device
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = None
        self.model = None
        self.torch = None
        self.available = False
        self.warning: str | None = None

        try:
            import torch  # type: ignore
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore

            self.torch = torch
            if "cuda" in str(device).lower() and not torch.cuda.is_available():
                self.device = "cpu"
                self.warning = "CUDA requested for Hugging Face GroundingDINO but unavailable; using CPU."
            local_files_only = Path(model_id).expanduser().exists()
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as exc:
            prefix = f"{self.warning} " if self.warning else ""
            self.warning = f"{prefix}Hugging Face GroundingDINO unavailable: {exc}"

    def detect(self, image_path, text_prompt) -> list[dict[str, Any]]:
        if not self.available or self.processor is None or self.model is None or self.torch is None:
            return []
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = _format_groundingdino_prompt(text_prompt)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            input_ids = inputs.get("input_ids")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = [image.size[::-1]]
            if hasattr(self.processor, "post_process_grounded_object_detection"):
                try:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        input_ids,
                        threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=target_sizes,
                    )
                except TypeError:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        input_ids,
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=target_sizes,
                    )
            else:
                results = self.processor.image_processor.post_process_object_detection(
                    outputs,
                    threshold=self.box_threshold,
                    target_sizes=target_sizes,
                )

            result = results[0] if results else {}
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("text_labels", result.get("labels"))
            width, height = image.size
            out = []
            for idx, box in enumerate(boxes):
                vals = [float(x) for x in box.tolist()]
                bbox = bbox_from_xyxy(vals, width, height)
                score = float(_item_at(scores, idx, 0.0))
                label = _item_at(labels, idx, prompt)
                if not isinstance(label, str):
                    label = prompt
                out.append({"bbox": bbox.to_dict(), "score": score, "label": label})
            return out
        except Exception as exc:
            self.warning = f"Hugging Face GroundingDINO detect failed: {exc}"
            return []


class GroundingDINODetector(BaseDetector):
    def __init__(
        self,
        config_path: str | None,
        weights_path: str | None,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.config_path = config_path
        self.weights_path = weights_path
        self.requested_device = device
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.available = False
        self.warning: str | None = None
        self.repo_root: Path | None = None
        if not config_path or not weights_path:
            self.warning = "GroundingDINO config/weights not provided."
            return
        if not Path(config_path).exists() or not Path(weights_path).exists():
            self.warning = "GroundingDINO config or weights path does not exist."
            return
        self.repo_root = _ensure_groundingdino_repo_on_path(config_path)
        if "cuda" in str(self.device).lower() and not _groundingdino_custom_ops_available():
            self.device = "cpu"
            self.warning = (
                "GroundingDINO custom C++ ops unavailable; using CPU fallback. "
                "Compile groundingdino._C for CUDA inference."
            )
        try:
            from groundingdino.util.inference import load_model  # type: ignore

            self._predict = None
            self._load_image = None
            self.model = load_model(config_path, weights_path, device=self.device)
            self.available = True
        except Exception as exc:
            self.warning = f"GroundingDINO unavailable: {exc}"

    def detect(self, image_path, text_prompt) -> list[dict[str, Any]]:
        if not self.available or self.model is None:
            return []
        try:
            from groundingdino.util.inference import load_image, predict  # type: ignore

            image_source, image = load_image(str(image_path))
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=str(text_prompt),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            width, height = Image.open(image_path).size
            out = []
            for box, score, phrase in zip(boxes, logits, phrases):
                vals = [float(x) for x in box.tolist()]
                if max(vals) <= 1.5:
                    cx, cy, bw, bh = vals
                    xyxy = [
                        (cx - bw / 2.0) * width,
                        (cy - bh / 2.0) * height,
                        (cx + bw / 2.0) * width,
                        (cy + bh / 2.0) * height,
                    ]
                else:
                    xyxy = vals
                bbox = bbox_from_xyxy(xyxy, width, height)
                out.append({"bbox": bbox.to_dict(), "score": float(score), "label": str(phrase)})
            return out
        except Exception as exc:
            self.warning = f"GroundingDINO detect failed: {exc}"
            return []
