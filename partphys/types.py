from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional


def _coerce_bbox(value: Any) -> "BBox":
    if isinstance(value, BBox):
        return value
    if isinstance(value, dict):
        return BBox.from_dict(value)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return BBox(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    raise TypeError(f"Cannot convert {value!r} to BBox")


def _from_dict_dataclass(cls, data: dict[str, Any]):
    kwargs = {}
    names = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in names:
            continue
        kwargs[key] = value
    return cls(**kwargs)


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox":
        return cls(int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))

    @property
    def width(self) -> int:
        return max(0, int(self.x2) - int(self.x1))

    @property
    def height(self) -> int:
        return max(0, int(self.y2) - int(self.y1))

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0


@dataclass
class MaskCandidate:
    candidate_id: str
    source: str
    mask_path: str
    bbox: BBox
    area: int
    area_ratio: float
    inside_object_ratio: float
    stability_score: Optional[float] = None
    predicted_iou: Optional[float] = None
    prompt: Optional[str] = None
    part_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.bbox = _coerce_bbox(self.bbox)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskCandidate":
        item = _from_dict_dataclass(cls, data)
        item.bbox = _coerce_bbox(item.bbox)
        return item


@dataclass
class PartSpec:
    name: str
    text_prompts: list[str] = field(default_factory=list)
    expected_materials: list[str] = field(default_factory=list)
    location: str = ""
    shape_prior: str = ""
    physical_role: str = ""
    should_simulate_separately: bool = True
    visible: bool = True
    physics_group: Optional[str] = None

    def __post_init__(self):
        if self.physics_group is None:
            self.physics_group = self.name
        if not self.text_prompts:
            self.text_prompts = [self.name]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartSpec":
        return _from_dict_dataclass(cls, data)


@dataclass
class PartInstance:
    part_id: int
    name: str
    mask_path: str
    bbox: BBox
    area: int
    confidence: float
    candidate_ids: list[str] = field(default_factory=list)
    expected_materials: list[str] = field(default_factory=list)
    physics_group: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.bbox = _coerce_bbox(self.bbox)
        if not self.physics_group:
            self.physics_group = self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartInstance":
        item = _from_dict_dataclass(cls, data)
        item.bbox = _coerce_bbox(item.bbox)
        return item


@dataclass
class PhysicsParams:
    material: str
    material_confidence: float
    E: float
    nu: float
    density: float
    logE_std: Optional[float] = None
    nu_std: Optional[float] = None
    confidence: float = 0.0
    source_outputs: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicsParams":
        return _from_dict_dataclass(cls, data)


@dataclass
class PhysGMResult:
    scene_dir: str
    point_cloud_path: Optional[str]
    predicted_phys_path: str
    material: str
    E: float
    nu: float
    density: Optional[float]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysGMResult":
        return _from_dict_dataclass(cls, data)


@dataclass
class PartPhysResult:
    scene_name: str
    object_name: str
    object_mask_path: str
    parts: list[PartInstance]
    part_physics: dict[int, PhysicsParams]
    whole_physgm: PhysGMResult
    assignment_summary: dict[str, Any]
    sim_config_path: Optional[str]
    simulation_output_dir: Optional[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["part_physics"] = {
            str(k): v.to_dict() if is_dataclass(v) else v
            for k, v in self.part_physics.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartPhysResult":
        parts = [PartInstance.from_dict(p) for p in data.get("parts", [])]
        part_physics = {
            int(k): PhysicsParams.from_dict(v)
            for k, v in data.get("part_physics", {}).items()
        }
        whole = data.get("whole_physgm")
        if isinstance(whole, dict):
            whole = PhysGMResult.from_dict(whole)
        return cls(
            scene_name=data["scene_name"],
            object_name=data["object_name"],
            object_mask_path=data["object_mask_path"],
            parts=parts,
            part_physics=part_physics,
            whole_physgm=whole,
            assignment_summary=data.get("assignment_summary", {}),
            sim_config_path=data.get("sim_config_path"),
            simulation_output_dir=data.get("simulation_output_dir"),
            warnings=data.get("warnings", []),
        )
