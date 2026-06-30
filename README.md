# PartPhysAgent

PartPhysAgent is an inference-only extension around the existing PhysGM project.
It does not train PhysGM, does not modify PhysGM weights, and does not replace
the original PhysGM entrypoints.

The current code takes one canonical input image and can optionally use an
existing four-view directory. It generates an object mask, builds or loads a
part schema, segments physical parts, optionally separates a foreground into
object instances, aligns part/object masks across four views, runs PhysGM to
obtain 3DGS geometry, assigns Gaussians to objects and parts, estimates per-part
physical parameters from PhysGM crop inference, and writes a PhysGM simulation
config.

## Current Behavior

- Single-object final geometry comes from whole-object PhysGM, not from per-part
  crops.
- Multi-object final geometry uses per-object PhysGM when `--object-mode auto`
  finds more than one object. Each object is masked in the same four-view camera
  frame, reconstructed separately, then merged into one PLY with direct
  `gaussian_object_ids.npy`.
- Per-part PhysGM crop inference is used only as physical-parameter evidence.
- The default simulation config uses PhysGM-compatible `additional_material_params`
  over part AABBs.
- The main `partphys_pipeline.py` simulation path calls PhysGM `gs_simulation.py`.
- `tools/gs_simulation_partid_materials.py` is available as a separate per-part-id
  simulation tool, but it is not the default `partphys_pipeline.py` simulation
  entrypoint.
- A four-view directory can be used for both whole-object PhysGM input and
  multi-view object/part-mask evidence.
- If no VLM schema is available, built-in object templates are used as fallback.
- `--object-mode auto` exports object instances, runs object-local part
  segmentation, and writes object/part Gaussian assignments. The default is
  `single` for compatibility with older single-object runs.

## Setup

Install and verify the original PhysGM first, following `/root/PhysGM/README.md`.

Optional dependencies:

- SAM2 or SAM1 checkpoints for mask generation
- GroundingDINO dependencies for text-grounded boxes
- `opencv-python` for mask cleanup
- `plyfile` for PLY loading
- `scikit-learn` for KMeans appearance proposals
- `openai` for OpenAI-compatible VLM calls

SAM and GroundingDINO are optional if `--masks-json` is provided.

## Main Command

```bash
python partphys_pipeline.py \
  --image examples/cake.png \
  --scene-name cake_partphys \
  --object cake \
  --output-dir /root/autodl-tmp/results_partphys \
  --multiview-dir /path/to/four_views \
  --sam-backend sam2 \
  --sam-checkpoint /path/to/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --vlm-provider openai_compatible \
  --vlm-model qwen3.7-plus \
  --vlm-api-base https://llm-jrkem52i075alacx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --vlm-api-key-env DASHSCOPE_API_KEY \
  --physgm-root /root/PhysGM \
  --physgm-config /root/PhysGM/configs/infer.yaml \
  --checkpoint /root/PhysGM/checkpoints/checkpoint.pt \
  --template-config /root/PhysGM/configs/physical/down_template.json \
  --agent-mode agent \
  --object-mode auto \
  --simulate \
  --white-bg
```

## Useful Modes

Mask-only:

```bash
python partphys_pipeline.py \
  --image examples/cake.png \
  --scene-name cake_masks \
  --object cake \
  --output-dir /root/autodl-tmp/results_partphys \
  --mask-only
```

Stop after Gaussian assignment:

```bash
python partphys_pipeline.py \
  --image examples/cake.png \
  --scene-name cake_assignment \
  --object cake \
  --output-dir /root/autodl-tmp/results_partphys \
  --segmentation-only
```

Multi-object Gaussian-only check:

```bash
python partphys_pipeline.py \
  --image /path/to/four_views/000.png \
  --multiview-dir /path/to/four_views \
  --scene-name multi_object_assignment \
  --object "object collection" \
  --output-dir /root/autodl-tmp/results_partphys \
  --physgm-root /root/PhysGM \
  --object-mode auto \
  --max-objects 4 \
  --segmentation-only \
  --vlm-provider openai_compatible \
  --vlm-model qwen3.7-plus \
  --vlm-api-base https://llm-jrkem52i075alacx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --vlm-api-key-env DASHSCOPE_API_KEY \
  --use-schema-location-proposals
```

Reuse an existing whole-object PhysGM result:

```bash
python partphys_pipeline.py \
  --image examples/cake.png \
  --scene-name cake_reuse_whole \
  --object cake \
  --output-dir /root/autodl-tmp/results_partphys \
  --whole-physgm-dir /path/to/physgm_whole
```

Manual masks:

```bash
python partphys_pipeline.py \
  --image examples/hammer.png \
  --scene-name hammer_manual \
  --part-schema-json examples/hammer_schema.json \
  --masks-json examples/hammer_masks.json \
  --physgm-root /root/PhysGM \
  --physgm-config /root/PhysGM/configs/infer.yaml \
  --checkpoint /root/PhysGM/checkpoints/checkpoint.pt \
  --template-config /root/PhysGM/configs/physical/down_template.json \
  --simulate
```

## Inputs

`--image` is required and is used as the canonical view.

`--multiview-dir` is optional. The directory is recognized when it contains either:

```text
000.png
006.png
012.png
018.png
```

or:

```text
front.png
right.png
rear.png
left.png
```

`--masks-json` may provide an `object_mask` and part masks. If only part masks are
provided, their union is used as the object mask.

`--part-schema-json` may provide the part schema. Otherwise the code uses VLM
schema generation when available, then normalizes or falls back to built-in
templates.

## Segmentation

The default segmentation mode is `candidate_pool`.

The implemented candidate sources include text-box SAM proposals, SAM automatic
masks, color priors, appearance clusters, object-body fallback, optional VLM box
proposals, and optional schema-location proposals.

The VLM does not draw final masks in the default path. It can identify the object,
generate or revise a schema, rank existing candidate masks by `candidate_id`, and
provide mask/material judgments.

With `--agent-mode agent`, `PartSegAgentController` adds a bounded
planner/critic/repair loop over the existing segmentation tools. Its action space
is fixed in code and includes candidate generation, ranking, schema-location
reruns, object/mask repair, layout compilation, nearest-part residual fill,
candidate-pool expansion, multi-view alignment, and KNN cleanup.

For multi-object scenes, the recommended default is to let the VLM identify each
object crop and generate each object-local schema, while keeping segmentation in
`pipeline` mode. The VLM planner/critic path can be slower because it may call
the remote model once per object and repair round.

When a structural object has a valid schema but most pixels remain residual, the
segmentation agent applies a conservative structural residual repair. It is
schema-driven, not object-name-specific: high residual plus parts described as
top compact blocks and long handles/shafts/rods can be repartitioned into those
parts. The repair only triggers for very large residuals so normal
`unknown_body` behavior is preserved.

## Four-View Handling

When `--multiview-dir` is provided, the code:

- uses the four images as PhysGM input views when running whole-object PhysGM;
- runs part segmentation on non-canonical views;
- aligns cross-view parts by normalized part name within each object;
- aligns cross-view objects by appearance/shape evidence instead of x-order,
  because object left/right ordering can change under orbit cameras;
- stores per-view mask paths in each part's `metadata["view_masks"]`;
- uses available view masks during projection-based Gaussian assignment.

## Object Separation

The default object mode is:

```text
--object-mode single
```

This keeps the foreground as one object and preserves earlier single-object
behavior.

For multi-object scenes, use:

```text
--object-mode auto
```

In auto mode, the code builds object proposals from connected foreground
components, existing part masks, and SAM automatic masks when SAM is available.
It then attaches cross-view object masks, identifies each object crop when VLM is
enabled, generates an object-local part schema, and runs part segmentation for
each object.

Object outputs include:

```text
objects/object_XXX_<name>/mask.png
objects/object_XXX_<name>/overlay.png
objects/object_XXX_<name>/object_summary.json
objects/objects_overlay.png
objects/objects_summary.json
object_parts/object_XXX_<object>/schema/part_schema.json
object_parts/object_XXX_<object>/parts/parts_overlay.png
object_parts/object_centric_part_summary.json
```

For multi-object PhysGM input debugging, each object also writes:

```text
physgm_whole/per_object_physgm/object_inputs_alpha_debug.png
physgm_whole/per_object_physgm/objects/object_XXX_<object>/input_views/object_input_alpha_debug.png
```

## Gaussian Assignment

The default assignment mode is `projection`.

For single-object scenes, the code projects whole-object 3DGS positions into
available camera views from `input_batch_meta.npz`, checks the projected pixels
against part masks, combines multi-view votes, applies confidence and KNN
cleanup, and writes per-Gaussian part ids.

For multi-object scenes, the merged per-object PLY provides direct object ids.
Part ids are still assigned with projection voting against the object-local
part masks, then object ids are written from the per-object merge metadata.

When object separation is enabled, the assignment stage writes:

```text
assignment/gaussian_object_ids.npy
assignment/object_gaussian_index.json
assignment/per_object_gaussians/object_XXX_<name>.ply
```

If projection fails or has no valid support, the code can fall back to
`aabb_heuristic`.

## Main Outputs

Outputs are written under:

```text
<output-dir>/<scene-name>/
```

Main files include:

```text
input/
object/
schema/part_schema.json
candidates/
parts/part_XXX_<name>/mask.png
parts/parts_overlay.png
parts/selection_summary.json
agent_logs/
assignment/gaussian_part_ids.npy
assignment/assignment_summary.json
assignment/part_gaussian_index.json
assignment/per_part_gaussians/part_XXX_<name>.ply
assignment/per_part_aabb.json
assignment/gaussian_object_ids.npy
assignment/object_gaussian_index.json
assignment/per_object_gaussians/object_XXX_<name>.ply
objects/
physgm_whole/
simulation/sim_config_partphys.json
simulation/part_aabb_metadata.json
simulation/command.txt
partphys_summary.json
warnings.txt
```

If `--simulate` is used, simulation stdout/stderr and `run_result.json` are also
written under `simulation/`.

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
    }
  ]
}
```

## Tests

```bash
python -m pytest tests/test_partphys_utils.py
```
