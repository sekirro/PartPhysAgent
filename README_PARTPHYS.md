# PartPhysAgent

PartPhysAgent is an inference-only extension around the existing PhysGM project.
It does not train PhysGM, does not modify PhysGM weights, and does not change the original PhysGM entrypoints.

The pipeline decomposes one input image into physically meaningful parts, estimates per-part physical parameters with PhysGM crop inference, runs whole-object PhysGM once to get a unified 3DGS, assigns part regions to the whole Gaussian geometry, and builds a PhysGM MPM simulation config with local `E`, `nu`, and `density`.

MVP behavior:

- Whole-object PhysGM is the only final geometry source.
- Part-crop PhysGM is used only to estimate per-part physical parameters.
- Local parameters are injected through `additional_material_params`.
- Per-part material labels are saved in JSON metadata.
- The solver constitutive material class remains global unless a later Phase 2 adds per-particle material kernels.

## Setup

Install and verify the original PhysGM first, following `../PhysGM/README.md`.

Optional tools:

- SAM2 repository and checkpoints for masks
- GroundingDINO dependencies for text-grounded boxes
- `opencv-python` for better mask cleanup
- `plyfile` for binary PLY loading
- `scikit-learn` for KMeans appearance proposals

SAM2 and GroundingDINO are optional if you provide `--masks-json`. This project uses SAM2 only; SAM1 `segment-anything` is not supported.


SAM2 paths used by the automatic pipeline:

```text
/root/autodl-tmp/repos/sam2
/root/autodl-tmp/models/sam2/sam2.1_hiera_large.pt
```

## Commands

Full automatic:

```bash
python partphys_pipeline.py \
  --image examples/hammer.png \
  --scene-name hammer_001 \
  --object hammer \
  --output-dir /root/autodl-tmp/results_partphys \
  --sam-checkpoint /root/autodl-tmp/models/sam2/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-root /root/autodl-tmp/repos/sam2 \
  --groundingdino-config path/to/GroundingDINO_SwinT_OGC.py \
  --groundingdino-weights path/to/groundingdino_swint_ogc.pth \
  --physgm-root ../PhysGM \
  --physgm-config configs/infer.yaml \
  --checkpoint checkpoints/checkpoint.pt \
  --template-config configs/physical/down_template.json \
  --simulate
```

Manual masks:

```bash
python partphys_pipeline.py \
  --image examples/hammer.png \
  --scene-name hammer_manual \
  --part-schema-json examples/hammer_schema.json \
  --masks-json examples/hammer_masks.json \
  --physgm-root ../PhysGM \
  --physgm-config configs/infer.yaml \
  --checkpoint checkpoints/checkpoint.pt \
  --template-config configs/physical/down_template.json \
  --simulate
```

Mock test:

```bash
python partphys_pipeline.py \
  --image tests/fixtures/hammer.png \
  --scene-name mock_hammer \
  --masks-json tests/fixtures/hammer_masks.json \
  --mock-physgm \
  --no-simulate
```

## Manual JSON

`--part-schema-json`:

```json
{
  "object": "hammer",
  "parts": [
    {
      "name": "head",
      "text_prompts": ["hammer head", "metal hammer head"],
      "expected_materials": ["Metal"],
      "location": "top/front",
      "shape_prior": "compact block",
      "physical_role": "stiff impact part",
      "should_simulate_separately": true,
      "visible": true,
      "physics_group": "head"
    }
  ],
  "relations": []
}
```

`--masks-json` paths may be absolute or relative to the JSON file:

```json
{
  "object_mask": "object_mask.png",
  "parts": [
    {
      "name": "head",
      "mask": "head_mask.png",
      "expected_materials": ["Metal"],
      "physics_group": "head"
    },
    {
      "name": "handle",
      "mask": "handle_mask.png",
      "expected_materials": ["Wood"],
      "physics_group": "handle"
    }
  ]
}
```

Recommended outputs go to:

```text
/root/autodl-tmp/results_partphys/<scene_name>/
```

Run tests:

```bash
python -m pytest tests/test_partphys_utils.py
```

## Segmentation Agent Outputs

Automatic part segmentation now requires a VLM:

```bash
python partphys_pipeline.py \
  --image examples/hammer.png \
  --scene-name hammer_agent \
  --object hammer \
  --output-dir /root/autodl-tmp/results_partphys \
  --vlm-provider openai_compatible \
  --vlm-model <model> \
  --vlm-api-base <openai-compatible-base-url> \
  --vlm-api-key-env OPENAI_API_KEY \
  --physgm-root /root/PhysGM \
  --segmentation-only
```

The segmentation agent writes:

```text
<scene>/parts/part_XXX_<name>/mask.png
<scene>/parts/parts_overlay.png
<scene>/agent_logs/segmentation_agent_decisions.json
<scene>/agent_logs/candidate_scores.json
<scene>/agent_logs/quality_report.json
<scene>/assignment/gaussian_part_ids.npy
<scene>/assignment/part_gaussian_index.json
<scene>/assignment/per_part_gaussians/part_XXX_<name>.ply
```

Use `--whole-physgm-dir <dir>` to reuse an existing whole-object PhysGM output directory containing `point_clouds.ply` and optional `input_batch_meta.npz`. Without it, the pipeline runs whole-object PhysGM and then assigns the resulting Gaussians to the selected parts.
