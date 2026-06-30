from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from partphys.agent import PartPhysAgent, aggregate_physics_outputs, build_part_crops, weighted_median
from partphys.gaussian_assign import assign_by_aabb_heuristic, assign_by_projection, build_part_aabbs, load_ply_positions, save_assignment_outputs
from partphys.image_utils import bbox_expand, mask_iou, mask_to_bbox, read_mask, save_mask
from partphys.material_table import clamp_physics_to_material, normalize_material_name
from partphys.multiview import split_mvadapter_grid
from partphys.physgm_runner import PhysGMRunner
from partphys.scene_builder import build_physgm_input_scene
from partphys.segmentation_agent import SegmentationAgent
from partphys.sim_config_builder import build_part_aware_sim_config
from partphys.types import BBox, PartInstance
from partphys.vlm import NoVLMClient


def _write_template(path: Path):
    path.write_text(
        json.dumps(
            {
                "density": 5000,
                "n_grid": 50,
                "substep_dt": 2e-4,
                "frame_dt": 4e-2,
                "frame_num": 1,
                "boundary_conditions": [{"type": "bounding_box"}],
            }
        ),
        encoding="utf-8",
    )


def test_mask_to_bbox_iou_and_bbox_expand():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:6, 3:9] = True
    assert mask_to_bbox(mask) == BBox(3, 2, 9, 6)
    other = np.zeros_like(mask)
    other[4:8, 6:10] = True
    assert 0 < mask_iou(mask, other) < 1
    assert bbox_expand(BBox(3, 2, 9, 6), 0.5, 12, 10) == BBox(0, 0, 12, 8)


def test_crop_generation(tmp_path):
    img = Image.new("RGB", (32, 32), "white")
    arr = np.asarray(img).copy()
    arr[8:24, 6:26] = [120, 80, 40]
    image_path = tmp_path / "input.png"
    Image.fromarray(arr).save(image_path)
    obj = np.zeros((32, 32), dtype=bool)
    obj[8:24, 6:26] = True
    part = np.zeros((32, 32), dtype=bool)
    part[10:18, 8:16] = True
    obj_path = tmp_path / "object.png"
    part_path = tmp_path / "part.png"
    save_mask(obj, obj_path)
    save_mask(part, part_path)
    crops = build_part_crops(image_path, obj_path, part_path, tmp_path / "part_out")
    assert set(crops) == {"tight", "padded", "context_dim", "isolated_full"}
    for path in crops.values():
        assert Image.open(path).size == (512, 512)


def test_material_normalization_and_clamping():
    assert normalize_material_name("steel") == "Metal"
    E, nu, warnings = clamp_physics_to_material("Rubber", 1e12, 0.2)
    assert E <= 1e8
    assert nu >= 0.40
    assert warnings


def test_weighted_median_and_aggregation():
    assert weighted_median([1, 10, 100], [1, 10, 1]) == 10
    params = aggregate_physics_outputs(
        [
            {"variant": "tight", "material": "Metal", "E": 1e9, "nu": 0.3},
            {"variant": "context_dim", "material": "Metal", "E": 1e10, "nu": 0.32},
            {"variant": "padded", "material": "Wood", "E": 1e8, "nu": 0.35},
        ],
        expected_materials=["Metal"],
        part_confidence=0.9,
    )
    assert params.material == "Metal"
    assert params.E > 0
    assert 0 <= params.nu < 0.5


def test_scene_builder_writes_data_and_pose(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 48), "white").save(image_path)
    info = build_physgm_input_scene(image_path, tmp_path / "scene_root", "scene")
    assert Path(info["data_txt"]).read_text(encoding="utf-8").strip() == "./scene/pose.json"
    pose = json.loads(Path(info["pose_json"]).read_text(encoding="utf-8"))
    meta = json.loads(Path(info["view_metadata"]).read_text(encoding="utf-8"))
    assert len(pose["frames"]) == 4
    assert [frame["file_path"] for frame in pose["frames"]] == ["000.png", "006.png", "012.png", "018.png"]
    assert meta["view_source"] == "single_image_proxy"
    assert meta["view_labels"] == ["front", "right", "rear", "left"]
    assert (tmp_path / "scene_root" / "scene" / "000.png").exists()


def test_split_mvadapter_grid_selects_cardinal_views(tmp_path):
    colors = [
        (255, 0, 0),
        (255, 128, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
    ]
    grid = Image.new("RGB", (60, 10))
    for idx, color in enumerate(colors):
        grid.paste(Image.new("RGB", (10, 10), color), (idx * 10, 0))
    grid_path = tmp_path / "grid.png"
    grid.save(grid_path)
    info = split_mvadapter_grid(grid_path, tmp_path / "views", size=None)
    assert info["selected_indices"] == [0, 2, 3, 4]
    expected = [colors[i] for i in info["selected_indices"]]
    for frame_path, color in zip(info["frame_paths"], expected):
        assert Image.open(frame_path).getpixel((0, 0)) == color


def test_sim_config_builder_additional_material_params(tmp_path):
    template = tmp_path / "template.json"
    _write_template(template)
    part = PartInstance(0, "head", "mask.png", BBox(0, 0, 5, 5), 25, 1.0, [], ["Metal"], "head", [], {})
    phys = aggregate_physics_outputs([{"variant": "tight", "material": "Metal", "E": 2e9, "nu": 0.3}], ["Metal"], 1.0)
    config, warnings = build_part_aware_sim_config(
        template,
        tmp_path / "sim_config_partphys.json",
        phys,
        [part],
        {0: phys},
        [{"part_id": 0, "part_name": "head", "center": [1, 1, 1], "half_size": [0.2, 0.2, 0.2], "count": 30}],
    )
    assert len(config["additional_material_params"]) == 1
    assert (tmp_path / "part_aabb_metadata.json").exists()


def test_mock_physgm_runner_outputs(tmp_path):
    image_path = tmp_path / "metal_head.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    runner = PhysGMRunner(None, None, None, "cpu", tmp_path, mock=True)
    result = runner.infer_image(image_path, "head", tmp_path / "physgm", save_gaussian=True)
    assert result.material == "Metal"
    assert Path(result.predicted_phys_path).exists()
    assert result.point_cloud_path and Path(result.point_cloud_path).exists()
    assert load_ply_positions(result.point_cloud_path).shape[1] == 3


def test_cli_help_works():
    proc = subprocess.run(
        [sys.executable, "partphys_pipeline.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0
    assert "--image" in proc.stdout


def test_partphys_agent_mock_smoke(tmp_path):
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[8:28, 12:52] = [150, 150, 160]
    image[30:56, 24:40] = [120, 70, 25]
    image_path = tmp_path / "hammer.png"
    Image.fromarray(image).save(image_path)

    object_mask = np.zeros((64, 64), dtype=bool)
    object_mask[8:56, 12:52] = True
    head_mask = np.zeros((64, 64), dtype=bool)
    head_mask[8:28, 12:52] = True
    handle_mask = np.zeros((64, 64), dtype=bool)
    handle_mask[30:56, 24:40] = True
    save_mask(object_mask, tmp_path / "object_mask.png")
    save_mask(head_mask, tmp_path / "head_mask.png")
    save_mask(handle_mask, tmp_path / "handle_mask.png")
    masks_json = tmp_path / "masks.json"
    masks_json.write_text(
        json.dumps(
            {
                "object_mask": "object_mask.png",
                "parts": [
                    {"name": "head", "mask": "head_mask.png", "expected_materials": ["Metal"], "physics_group": "head"},
                    {"name": "handle", "mask": "handle_mask.png", "expected_materials": ["Wood"], "physics_group": "handle"},
                ],
            }
        ),
        encoding="utf-8",
    )
    template = tmp_path / "template.json"
    _write_template(template)
    cfg = SimpleNamespace(
        output_dir=str(tmp_path / "results"),
        masks_json=str(masks_json),
        part_schema_json=None,
        object="hammer",
        no_vlm=True,
        vlm_provider="none",
        groundingdino_config=None,
        groundingdino_weights=None,
        sam_checkpoint=None,
        sam_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_root="/root/autodl-tmp/repos/sam2",
        device="cpu",
        physgm_config=None,
        checkpoint=None,
        template_config=str(template),
        physgm_root=None,
        amp_dtype="fp32",
        mock_physgm=True,
        skip_part_physgm=False,
        assignment_mode="aabb_heuristic",
        min_gaussian_count_per_part=1,
        padding_ratio=0.15,
        min_half_size=0.02,
        simulate=False,
        max_parts=6,
        min_part_area_ratio=0.01,
        coverage_threshold=0.75,
        fallback_to_aabb_heuristic=True,
    )
    result = PartPhysAgent(cfg).run(str(image_path), "mock_hammer", object_hint="hammer")
    scene_dir = Path(cfg.output_dir) / "mock_hammer"
    assert (scene_dir / "partphys_summary.json").exists()
    sim_config = json.loads((scene_dir / "simulation" / "sim_config_partphys.json").read_text(encoding="utf-8"))
    assert result.sim_config_path
    assert len(sim_config["additional_material_params"]) >= 1


class _FakeRequiredVLM:
    requires_remote_vlm = True

    def locate_parts(self, image_path, parts):
        return {
            "parts": [
                {"name": "head", "bbox": [0.10, 0.10, 0.90, 0.42], "confidence": 0.95},
                {"name": "handle", "bbox": [0.40, 0.45, 0.62, 0.95], "confidence": 0.95},
            ]
        }

    def score_candidate_for_part(self, image_path, candidate_overlay_path, part_spec):
        path = str(candidate_overlay_path)
        name = part_spec.name if hasattr(part_spec, "name") else part_spec.get("name")
        if name == "head" and "candidate_001" in path:
            return {"score": 0.95, "reason": "head bbox candidate"}
        if name == "handle" and "candidate_002" in path:
            return {"score": 0.95, "reason": "handle bbox candidate"}
        return {"score": 0.10, "reason": "wrong part"}

    def rank_candidates_for_parts(self, image_path, contact_sheet_path, parts, candidates):
        return {"rankings": [], "warnings": []}


class _FailingLocateVLM(NoVLMClient):
    requires_remote_vlm = True

    def locate_parts(self, image_path, parts):
        raise AssertionError("locate_parts should not be called in candidate_pool mode")

    def rank_candidates_for_parts(self, image_path, contact_sheet_path, parts, candidates):
        return {"rankings": [], "warnings": []}


class _FakeSAM:
    def __init__(self, masks):
        self.masks = masks

    def automatic_masks(self, image_path):
        return [
            {
                "segmentation": mask,
                "predicted_iou": 0.95,
                "stability_score": 0.95,
                "bbox": mask_to_bbox(mask).to_dict(),
                "area": int(mask.sum()),
                "crop_box": None,
            }
            for mask in self.masks
        ]

    def segment_from_box(self, image_path, bbox):
        return []

    def segment_from_points(self, image_path, points, labels):
        return []


def _seg_image_and_object(tmp_path):
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[8:56, 8:56] = [130, 130, 130]
    image_path = tmp_path / "object.png"
    Image.fromarray(image).save(image_path)
    object_mask = np.zeros((64, 64), dtype=bool)
    object_mask[8:56, 8:56] = True
    object_mask_path = tmp_path / "object_mask.png"
    save_mask(object_mask, object_mask_path)
    return image_path, object_mask, object_mask_path


def _hammer_schema():
    return {
        "object": "hammer",
        "parts": [
            {"name": "head", "text_prompts": ["hammer head"], "expected_materials": ["Metal"], "location": "top", "shape_prior": "compact block", "visible": True},
            {"name": "handle", "text_prompts": ["hammer handle"], "expected_materials": ["Wood"], "location": "lower", "shape_prior": "long thin bar", "visible": True},
        ],
    }


def _head_handle_masks():
    head = np.zeros((64, 64), dtype=bool)
    head[8:28, 10:54] = True
    handle = np.zeros((64, 64), dtype=bool)
    handle[30:56, 26:38] = True
    return head, handle


def test_segmentation_agent_no_longer_requires_remote_vlm(tmp_path):
    image_path, object_mask, object_mask_path = _seg_image_and_object(tmp_path)
    head, handle = _head_handle_masks()
    parts, candidates, quality = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=_hammer_schema(),
        detector=None,
        sam_tool=_FakeSAM([head, handle]),
        vlm_client=NoVLMClient(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.40,
        min_accept_score=0.35,
    ).run()
    assert {"head", "handle"}.issubset({p.name for p in parts})
    assert candidates
    assert quality["reason"] != "requires_remote_vlm"


def test_vlm_bbox_disabled_by_default(tmp_path):
    image_path, object_mask, object_mask_path = _seg_image_and_object(tmp_path)
    head, handle = _head_handle_masks()
    parts, _, _ = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=_hammer_schema(),
        detector=None,
        sam_tool=_FakeSAM([head, handle]),
        vlm_client=_FailingLocateVLM(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.40,
        min_accept_score=0.35,
        use_vlm_bbox_proposals=False,
    ).run()
    assert {"head", "handle"}.issubset({p.name for p in parts})


def test_wrong_vlm_bbox_does_not_override_sam_candidates(tmp_path):
    image_path, object_mask, object_mask_path = _seg_image_and_object(tmp_path)
    head, handle = _head_handle_masks()
    parts, candidates, _ = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=_hammer_schema(),
        detector=None,
        sam_tool=_FakeSAM([head, handle]),
        vlm_client=_FakeRequiredVLM(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.40,
        min_accept_score=0.35,
    ).run()
    by_id = {c.candidate_id: c for c in candidates}
    selected_sources = {by_id[p.candidate_ids[0]].source for p in parts if p.candidate_ids and p.candidate_ids[0] in by_id}
    assert selected_sources <= {"sam_auto", "text_box_sam", "appearance_cluster", "object_body"}
    assert "vlm_box" not in selected_sources


def test_no_rectangle_final_mask_when_sam_candidate_exists(tmp_path):
    image_path, object_mask, object_mask_path = _seg_image_and_object(tmp_path)
    head, handle = _head_handle_masks()
    parts, candidates, _ = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=_hammer_schema(),
        detector=None,
        sam_tool=_FakeSAM([head, handle]),
        vlm_client=NoVLMClient(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.40,
        min_accept_score=0.35,
        use_schema_location_proposals=True,
    ).run()
    by_id = {c.candidate_id: c for c in candidates}
    selected_sources = {by_id[p.candidate_ids[0]].source for p in parts if p.candidate_ids and p.candidate_ids[0] in by_id}
    assert "schema_location" not in selected_sources
    assert "vlm_box" not in selected_sources


def test_main_body_large_sam_mask_allowed(tmp_path):
    image_path, object_mask, object_mask_path = _seg_image_and_object(tmp_path)
    cake_body = object_mask.copy()
    icing = np.zeros((64, 64), dtype=bool)
    icing[8:18, 16:48] = True
    schema = {
        "object": "cake",
        "parts": [
            {"name": "cake_body", "text_prompts": ["cake body"], "expected_materials": ["Foam"], "location": "main cake body", "shape_prior": "main body/base", "visible": True},
            {"name": "icing", "text_prompts": ["icing"], "expected_materials": ["Foam"], "location": "top", "shape_prior": "soft layer", "visible": True},
        ],
    }
    parts, _, _ = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=schema,
        detector=None,
        sam_tool=_FakeSAM([cake_body, icing]),
        vlm_client=NoVLMClient(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.55,
        min_accept_score=0.35,
    ).run()
    assert "cake_body" in {p.name for p in parts}
    scores = json.loads((tmp_path / "scene" / "agent_logs" / "candidate_scores.json").read_text(encoding="utf-8"))
    body_scores = [s for s in scores if s["part_name"] == "cake_body"]
    assert max(s["final_score"] for s in body_scores) >= 0.45


def test_residual_policy_unknown_does_not_distort_masks(tmp_path):
    image_path, object_mask, object_mask_path = _seg_image_and_object(tmp_path)
    head, handle = _head_handle_masks()
    parts, _, _ = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=_hammer_schema(),
        detector=None,
        sam_tool=_FakeSAM([head, handle]),
        vlm_client=NoVLMClient(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.80,
        min_accept_score=0.35,
        residual_policy="unknown",
    ).run()
    by_name = {p.name: p for p in parts}
    assert "unknown_body" in by_name
    assert np.array_equal(read_mask(by_name["head"].mask_path), head)
    assert np.array_equal(read_mask(by_name["handle"].mask_path), handle)


def test_segmentation_agent_recovers_parts_from_vlm_boxes(tmp_path):
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[8:56, 8:56] = [130, 130, 130]
    image_path = tmp_path / "hammer.png"
    Image.fromarray(image).save(image_path)
    object_mask = np.zeros((64, 64), dtype=bool)
    object_mask[8:56, 8:56] = True
    object_mask_path = tmp_path / "object_mask.png"
    save_mask(object_mask, object_mask_path)
    schema = {
        "object": "hammer",
        "parts": [
            {"name": "head", "text_prompts": ["hammer head"], "expected_materials": ["Metal"], "location": "top", "shape_prior": "compact block", "visible": True},
            {"name": "handle", "text_prompts": ["hammer handle"], "expected_materials": ["Wood"], "location": "lower", "shape_prior": "long thin bar", "visible": True},
        ],
    }
    parts, candidates, quality = SegmentationAgent(
        image_path=image_path,
        object_mask_path=object_mask_path,
        object_bbox=mask_to_bbox(object_mask),
        part_schema=schema,
        detector=None,
        sam_tool=None,
        vlm_client=_FakeRequiredVLM(),
        output_dir=tmp_path / "scene",
        candidates_dir=tmp_path / "scene" / "candidates",
        coverage_threshold=0.45,
        min_accept_score=0.30,
        segmentation_mode="legacy_vlm_bbox",
        residual_policy="ignore",
    ).run()
    assert quality["ok"]
    assert {p.name for p in parts} == {"head", "handle"}
    assert len(candidates) >= 2
    assert (tmp_path / "scene" / "agent_logs" / "candidate_scores.json").exists()


def test_gaussian_assignment_small_overwrites_body(tmp_path):
    body = np.ones((10, 10), dtype=bool)
    small = np.zeros((10, 10), dtype=bool)
    small[4:7, 4:7] = True
    body_path = tmp_path / "body.png"
    small_path = tmp_path / "small.png"
    save_mask(body, body_path)
    save_mask(small, small_path)
    meta_path = tmp_path / "input_batch_meta.npz"
    np.savez(
        meta_path,
        input_intr=np.asarray([[1.0, 1.0, 5.0, 5.0]], dtype=np.float32),
        input_c2ws=np.eye(4, dtype=np.float32)[None, ...],
    )
    result = assign_by_projection(
        np.asarray([[0.0, 0.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float32),
        [
            {"part_id": 0, "mask_path": str(body_path), "area": int(body.sum())},
            {"part_id": 1, "mask_path": str(small_path), "area": int(small.sum())},
        ],
        meta_path,
        (10, 10),
    )
    assert result["gaussian_part_ids"][0] == 1

    parts = [
        PartInstance(0, "body", str(body_path), BBox(0, 0, 10, 10), int(body.sum()), 1.0),
        PartInstance(1, "small", str(small_path), BBox(4, 4, 7, 7), int(small.sum()), 1.0),
    ]
    heuristic = assign_by_aabb_heuristic(np.asarray([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32), parts, (10, 10))
    assert heuristic["gaussian_part_ids"][0] == 1


def test_assignment_outputs_part_gaussian_index_and_ply(tmp_path):
    ply = tmp_path / "point_clouds.ply"
    ply.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "0 0 0\n1 0 0\n0 1 0\n1 1 0\n",
        encoding="utf-8",
    )
    parts = [
        PartInstance(0, "head", "head.png", BBox(0, 0, 1, 1), 1, 1.0),
        PartInstance(1, "handle", "handle.png", BBox(0, 0, 1, 1), 1, 1.0),
    ]
    save_assignment_outputs(
        tmp_path / "assignment",
        np.array([0, 0, 1, -1], dtype=np.int32),
        [],
        {"mode": "test", "warnings": []},
        parts,
        ply,
    )
    index = json.loads((tmp_path / "assignment" / "part_gaussian_index.json").read_text(encoding="utf-8"))
    assert index["unassigned_count"] == 1
    assert {p["part_name"] for p in index["parts"]} == {"head", "handle"}
    assert (tmp_path / "assignment" / "per_part_gaussians" / "part_000_head.ply").exists()
