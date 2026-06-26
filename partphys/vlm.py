from __future__ import annotations

import base64
import json
import os
import re
import socket
import urllib.error
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from .material_table import normalize_material_name
from .prompts import (
    CANDIDATE_RANK_PROMPT,
    MATERIAL_VERIFY_PROMPT,
    MASK_SCORE_PROMPT,
    PART_AGENT_CRITIQUE_PROMPT,
    PART_AGENT_PLAN_PROMPT,
    PART_SCHEMA_PROMPT,
    PART_VERIFY_PROMPT,
)
from .types import PartSpec


def _template_schema(object_name: str) -> dict[str, Any]:
    name = (object_name or "object").lower().strip()
    if name in {"hammer", "mallet"}:
        parts = [
            {
                "name": "head",
                "text_prompts": ["hammer head", "metal hammer head"],
                "expected_materials": ["Metal"],
                "location": "top/front compact region",
                "shape_prior": "compact block",
                "physical_role": "stiff impact part",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "head",
            },
            {
                "name": "handle",
                "text_prompts": ["hammer handle", "wooden handle", "rubber handle"],
                "expected_materials": ["Wood", "Rubber", "Plastic"],
                "location": "long lower part",
                "shape_prior": "long thin bar",
                "physical_role": "grip/support",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "handle",
            },
        ]
    elif name in {"shoe", "sneaker", "boot"}:
        parts = [
            {
                "name": "sole",
                "text_prompts": ["shoe sole", "rubber sole"],
                "expected_materials": ["Rubber"],
                "location": "bottom lower region",
                "shape_prior": "long base",
                "physical_role": "contact/friction",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "sole",
            },
            {
                "name": "upper",
                "text_prompts": ["shoe upper", "fabric upper", "leather upper"],
                "expected_materials": ["Fabric", "Leather"],
                "location": "upper body",
                "shape_prior": "main shell",
                "physical_role": "flexible body",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "upper",
            },
            {
                "name": "lace",
                "text_prompts": ["shoe lace", "thin lace"],
                "expected_materials": ["Fabric"],
                "location": "top thin strings",
                "shape_prior": "thin strings",
                "physical_role": "small flexible tie",
                "should_simulate_separately": False,
                "visible": True,
                "physics_group": "upper",
            },
        ]
    elif name in {"car", "toy_car", "toy car"}:
        parts = [
            {
                "name": "body",
                "text_prompts": ["car body", "main shell"],
                "expected_materials": ["Plastic", "Metal"],
                "location": "main center body",
                "shape_prior": "large shell",
                "physical_role": "main rigid body",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "body",
            },
            {
                "name": "tire",
                "text_prompts": ["car tire", "rubber wheel"],
                "expected_materials": ["Rubber"],
                "location": "bottom wheels",
                "shape_prior": "round wheel",
                "physical_role": "soft contact wheels",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "tire",
            },
            {
                "name": "window",
                "text_prompts": ["car window", "transparent window"],
                "expected_materials": ["Glass", "Plastic"],
                "location": "upper transparent regions",
                "shape_prior": "flat panels",
                "physical_role": "stiffer transparent panels",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "window",
            },
        ]
    elif name in {"chair", "stool"}:
        parts = [
            {
                "name": "frame",
                "text_prompts": ["chair frame", "chair legs", "chair back"],
                "expected_materials": ["Wood", "Metal", "Plastic"],
                "location": "main structural frame",
                "shape_prior": "thin structural members",
                "physical_role": "support frame",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "frame",
            },
            {
                "name": "cushion",
                "text_prompts": ["chair cushion", "seat cushion"],
                "expected_materials": ["Fabric", "Foam", "Leather"],
                "location": "seat/back soft areas",
                "shape_prior": "padded block",
                "physical_role": "soft deformable cushion",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "cushion",
            },
        ]
    elif name in {"cup", "mug"}:
        parts = [
            {
                "name": "body",
                "text_prompts": ["cup body", "mug body"],
                "expected_materials": ["Ceramic", "Glass", "Plastic"],
                "location": "main body",
                "shape_prior": "hollow cylinder",
                "physical_role": "main rigid container",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "body",
            },
            {
                "name": "handle",
                "text_prompts": ["cup handle", "mug handle"],
                "expected_materials": ["Ceramic", "Glass", "Plastic"],
                "location": "side handle",
                "shape_prior": "curved loop",
                "physical_role": "attached grip",
                "should_simulate_separately": False,
                "visible": True,
                "physics_group": "body",
            },
        ]
    elif name in {"cake", "cake_slice", "cake slice", "pastry", "dessert"}:
        parts = [
            {
                "name": "cake_body",
                "text_prompts": ["cake body", "main cake", "cake base"],
                "expected_materials": ["Foam", "Plasticine"],
                "location": "main cake body",
                "shape_prior": "main body/base",
                "physical_role": "soft main dessert body",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "cake_body",
            },
            {
                "name": "icing",
                "text_prompts": ["icing", "cream", "frosting"],
                "expected_materials": ["Foam", "Plasticine", "Snow"],
                "location": "top or outer cream/icing",
                "shape_prior": "soft layer",
                "physical_role": "soft decorative layer",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "icing",
            },
            {
                "name": "topping",
                "text_prompts": ["cake topping", "small visible toppings", "strawberry topping"],
                "expected_materials": ["Plasticine", "Foam"],
                "location": "small visible toppings",
                "shape_prior": "small pieces",
                "physical_role": "small decorative pieces",
                "should_simulate_separately": False,
                "visible": True,
                "physics_group": "icing",
            },
        ]
    else:
        parts = [
            {
                "name": "body",
                "text_prompts": [object_name or "object"],
                "expected_materials": ["Plastic"],
                "location": "main object body",
                "shape_prior": "main body",
                "physical_role": "global fallback body",
                "should_simulate_separately": True,
                "visible": True,
                "physics_group": "body",
            }
        ]
    return {"object": object_name or "object", "parts": parts, "relations": []}


def normalize_part_schema(schema: dict[str, Any], object_hint: str | None = None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        schema = _template_schema(object_hint or "object")
    parts = []
    for raw in schema.get("parts", []):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["name"] = str(item.get("name") or "body").strip().replace(" ", "_")
        item["text_prompts"] = [str(x) for x in item.get("text_prompts") or [item["name"]]]
        item["expected_materials"] = [normalize_material_name(x) for x in item.get("expected_materials") or ["Plastic"]]
        item["location"] = str(item.get("location") or "")
        item["shape_prior"] = str(item.get("shape_prior") or "")
        item["physical_role"] = str(item.get("physical_role") or "")
        item["should_simulate_separately"] = bool(item.get("should_simulate_separately", True))
        item["visible"] = bool(item.get("visible", True))
        item["physics_group"] = str(item.get("physics_group") or item["name"])
        parts.append(item)
    if not parts:
        parts = _template_schema(object_hint or schema.get("object") or "object")["parts"]
    return {
        "object": schema.get("object") or object_hint or "object",
        "parts": parts,
        "relations": schema.get("relations", []),
    }


def part_specs_from_schema(schema: dict[str, Any]) -> list[PartSpec]:
    schema = normalize_part_schema(schema, schema.get("object"))
    return [PartSpec.from_dict(p) for p in schema.get("parts", []) if p.get("visible", True)]


class BaseVLMClient:
    def identify_object(self, image_path, object_hint=None) -> dict[str, Any]:
        raise NotImplementedError

    def generate_part_schema(self, image_path, object_name, object_mask_path=None) -> dict[str, Any]:
        raise NotImplementedError

    def score_candidate_for_part(self, image_path, candidate_overlay_path, part_spec) -> dict[str, Any]:
        raise NotImplementedError

    def rank_candidates_for_parts(self, image_path: str, contact_sheet_path: str, parts: list[dict], candidates: list[dict]) -> dict[str, Any]:
        raise NotImplementedError

    def verify_selected_parts(self, image_path, overlay_path, parts) -> dict[str, Any]:
        raise NotImplementedError

    def locate_parts(self, image_path, parts) -> dict[str, Any]:
        raise NotImplementedError

    def infer_material_prior(self, image_path, part_name) -> dict[str, Any]:
        raise NotImplementedError

    def plan_part_segmentation(self, image_path, object_name, previous_critique=None) -> dict[str, Any]:
        raise NotImplementedError

    def critique_part_segmentation(self, overlay_path, parts, quality_report) -> dict[str, Any]:
        raise NotImplementedError


class NoVLMClient(BaseVLMClient):
    def identify_object(self, image_path, object_hint=None) -> dict[str, Any]:
        return {"object": object_hint or "object", "confidence": 0.0, "source": "fallback"}

    def generate_part_schema(self, image_path, object_name, object_mask_path=None) -> dict[str, Any]:
        return normalize_part_schema(_template_schema(object_name or "object"), object_name)

    def score_candidate_for_part(self, image_path, candidate_overlay_path, part_spec) -> dict[str, Any]:
        return {"score": 0.5, "reason": "No VLM available."}

    def rank_candidates_for_parts(self, image_path: str, contact_sheet_path: str, parts: list[dict], candidates: list[dict]) -> dict[str, Any]:
        return {"rankings": [], "warnings": ["No VLM candidate ranking available."]}

    def verify_selected_parts(self, image_path, overlay_path, parts) -> dict[str, Any]:
        return {"ok": True, "warnings": ["No VLM verification available."]}

    def locate_parts(self, image_path, parts) -> dict[str, Any]:
        return {"parts": []}

    def infer_material_prior(self, image_path, part_name) -> dict[str, Any]:
        return {"material": "Plastic", "confidence": 0.0, "source": "fallback"}

    def plan_part_segmentation(self, image_path, object_name, previous_critique=None) -> dict[str, Any]:
        schema = normalize_part_schema(_template_schema(object_name or "object"), object_name)
        return {
            "object": schema["object"],
            "parts": schema["parts"],
            "tool_plan": [
                {"action": "generate_object_mask", "reason": "fallback planner"},
                {"action": "generate_part_candidates", "reason": "fallback planner"},
                {"action": "rank_candidates", "reason": "fallback planner"},
                {"action": "multiview_align", "reason": "fallback planner"},
                {"action": "gaussian_assign", "reason": "fallback planner"},
                {"action": "knn_cleanup", "reason": "fallback planner"},
            ],
            "quality_gates": {"max_unknown_ratio": 0.05, "max_overlap_ratio": 0.03, "require_multiview_alignment": True},
            "warnings": ["No VLM planner available; used template plan."],
        }

    def critique_part_segmentation(self, overlay_path, parts, quality_report) -> dict[str, Any]:
        return {"ok": bool(quality_report.get("ok", True)), "failure_modes": [], "repair_actions": [], "notes": ["No VLM critic available."]}


class OpenAICompatibleVLMClient(BaseVLMClient):
    def __init__(
        self,
        model: str | None,
        api_base: str | None,
        api_key_env: str = "OPENAI_API_KEY",
        fallback: BaseVLMClient | None = None,
        timeout: int = 180,
        required: bool = False,
    ):
        self.model = model or "gpt-4o-mini"
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = api_key_env
        self.fallback = fallback or NoVLMClient()
        self.timeout = timeout
        self.required = bool(required)
        self.requires_remote_vlm = True
        self.warnings: list[str] = []

    def _image_url(self, image_path: str) -> str:
        data = Path(image_path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
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

    def _call_json(self, prompt: str, image_path: str | None = None) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"API key env var {self.api_key_env} is not set.")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            content.append({"type": "image_url", "image_url": {"url": self._image_url(image_path)}})
        client = OpenAI(api_key=key, base_url=self.api_base, timeout=self.timeout)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }
        if self.model.lower().startswith("qwen") or "aliyuncs.com" in self.api_base:
            kwargs["extra_body"] = {"enable_thinking": False}
        try:
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content or ""
        except (OpenAIError, AttributeError, IndexError) as exc:
            raise RuntimeError(f"VLM OpenAI-compatible call failed: {exc}") from exc
        return self._extract_json(text)

    def _fallback_on_error(self, method: str, error: Exception, *args):
        msg = f"VLM {method} failed: {error}. Falling back to NoVLMClient."
        self.warnings.append(msg)
        if self.required:
            raise RuntimeError(msg) from error
        return getattr(self.fallback, method)(*args)

    def identify_object(self, image_path, object_hint=None) -> dict[str, Any]:
        if object_hint:
            return {"object": object_hint, "confidence": 1.0, "source": "hint"}
        try:
            return self._call_json("Identify the main physical object. Return strict JSON: {\"object\": string, \"confidence\": number}.", image_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("identify_object", exc, image_path, object_hint)

    def generate_part_schema(self, image_path, object_name, object_mask_path=None) -> dict[str, Any]:
        prompt = PART_SCHEMA_PROMPT + f"\nObject name: {object_name}\n"
        try:
            return normalize_part_schema(self._call_json(prompt, image_path), object_name)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("generate_part_schema", exc, image_path, object_name, object_mask_path)

    def score_candidate_for_part(self, image_path, candidate_overlay_path, part_spec) -> dict[str, Any]:
        prompt = MASK_SCORE_PROMPT + f"\nPart: {part_spec.to_dict() if hasattr(part_spec, 'to_dict') else part_spec}"
        try:
            return self._call_json(prompt, candidate_overlay_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("score_candidate_for_part", exc, image_path, candidate_overlay_path, part_spec)

    def rank_candidates_for_parts(self, image_path: str, contact_sheet_path: str, parts: list[dict], candidates: list[dict]) -> dict[str, Any]:
        candidate_brief = [
            {
                "candidate_id": c.get("candidate_id"),
                "source": c.get("source"),
                "area_ratio": c.get("metadata", {}).get("object_area_ratio", c.get("area_ratio")),
                "prompt": c.get("prompt"),
            }
            for c in candidates
        ]
        prompt = (
            CANDIDATE_RANK_PROMPT
            + "\nRequested parts:\n"
            + json.dumps(parts, ensure_ascii=False)
            + "\nCandidate metadata:\n"
            + json.dumps(candidate_brief, ensure_ascii=False)
        )
        try:
            return self._call_json(prompt, contact_sheet_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("rank_candidates_for_parts", exc, image_path, contact_sheet_path, parts, candidates)

    def verify_selected_parts(self, image_path, overlay_path, parts) -> dict[str, Any]:
        prompt = PART_VERIFY_PROMPT + f"\nParts: {[p.to_dict() if hasattr(p, 'to_dict') else p for p in parts]}"
        try:
            return self._call_json(prompt, overlay_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("verify_selected_parts", exc, image_path, overlay_path, parts)

    def locate_parts(self, image_path, parts) -> dict[str, Any]:
        prompt = (
            "Locate each requested visible object part in the image. Return strict JSON only. "
            "Return one entry for every requested part name; if uncertain, give the best visible bbox with lower confidence. "
            "Use normalized bbox coordinates [x1, y1, x2, y2] in the range 0..1. "
            "Do not include the whole object bbox unless the requested part is the whole body.\n"
            f"Parts: {parts}\n"
            "Required schema: {\"parts\": [{\"name\": \"handle\", \"bbox\": [0.1, 0.4, 0.8, 0.7], \"confidence\": 0.8, \"reason\": \"short reason\"}]}"
        )
        try:
            return self._call_json(prompt, image_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("locate_parts", exc, image_path, parts)

    def infer_material_prior(self, image_path, part_name) -> dict[str, Any]:
        prompt = MATERIAL_VERIFY_PROMPT + f"\nPart name: {part_name}"
        try:
            return self._call_json(prompt, image_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("infer_material_prior", exc, image_path, part_name)

    def plan_part_segmentation(self, image_path, object_name, previous_critique=None) -> dict[str, Any]:
        prompt = (
            PART_AGENT_PLAN_PROMPT
            + f"\nObject name: {object_name}\n"
            + "Previous critique:\n"
            + json.dumps(previous_critique or {}, ensure_ascii=False)
        )
        try:
            return self._call_json(prompt, image_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("plan_part_segmentation", exc, image_path, object_name, previous_critique)

    def critique_part_segmentation(self, overlay_path, parts, quality_report) -> dict[str, Any]:
        prompt = (
            PART_AGENT_CRITIQUE_PROMPT
            + "\nParts:\n"
            + json.dumps([p.to_dict() if hasattr(p, "to_dict") else p for p in parts], ensure_ascii=False)
            + "\nQuality report:\n"
            + json.dumps(quality_report or {}, ensure_ascii=False)
        )
        try:
            return self._call_json(prompt, overlay_path)
        except (RuntimeError, TimeoutError, socket.timeout, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_on_error("critique_part_segmentation", exc, overlay_path, parts, quality_report)
