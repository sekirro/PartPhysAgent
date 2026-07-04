# PartPhysAgent

PartPhysAgent is a single-object part decomposition layer around PhysGM. It
segments object parts, runs or reuses whole-object PhysGM reconstruction, assigns
the reconstructed 3D Gaussians to parts, and writes part-aware simulation
metadata for downstream material selection or direct PhysGM simulation.

It does not train PhysGM and does not modify PhysGM model weights.

## Current Behavior

- Input is one primary image plus an optional four-view directory.
- `--multiview-dir` can provide PhysGM-style views such as
  `000.png/006.png/012.png/018.png` or `front/right/rear/left`.
- Object masks are read from `--masks-json` when provided, otherwise generated
  with GroundingDINO/SAM/SAM2 fallbacks.
- Part schema is read from `--part-schema-json`, from `--masks-json`, from an
  optional OpenAI-compatible VLM, or from built-in fallback templates.
- Default part segmentation uses a candidate pool, rule scores, optional VLM
  candidate-id ranking, non-overlap layout compilation, and optional
  agent-mode repair rounds.
- Whole-object PhysGM is the geometry source. Part-crop PhysGM is used only as
  evidence for part-level physical parameters.
- Gaussian-to-part assignment writes `assignment/gaussian_part_ids.npy`,
  per-part Gaussian index files, and per-part AABB metadata.
- The built-in PartPhysAgent simulation config uses PhysGM
  `additional_material_params` over part AABBs.
- Downstream tools such as MaterialAgent can use `gaussian_part_ids.npy` with
  `tools/gs_simulation_partid_materials.py` to assign solver material,
  `E`, `nu`, and density by particle part id.

## Command

```bash
conda activate physgm
cd /root/PartPhysAgent
PYTHONPATH=/root/PartPhysAgent:/root/PhysGM python partphys_pipeline.py \
  --image /path/to/front.png \
  --multiview-dir /path/to/four_views \
  --scene-name example_partphys \
  --output-dir /root/autodl-tmp/results_partphys \
  --physgm-root /root/PhysGM \
  --physgm-config /root/PhysGM/configs/infer.yaml \
  --checkpoint /root/PhysGM/checkpoints/checkpoint.pt \
  --template-config /root/PhysGM/configs/physical/down_template.json \
  --agent-mode agent \
  --simulate \
  --white-bg
```

Useful options:

```text
--mask-only              stop after object and part masks
--segmentation-only      stop after masks and Gaussian assignment
--whole-physgm-dir DIR   reuse an existing whole-object PhysGM output
--skip-part-physgm       use fallback part physics instead of crop inference
--assignment-mode        projection, aabb_heuristic, or none
--no-vlm                 disable VLM calls
--require-vlm            fail if VLM is unavailable
```

## Main Outputs

```text
<scene>/input/
<scene>/object/
<scene>/schema/
<scene>/candidates/
<scene>/parts/
<scene>/agent_logs/
<scene>/physgm_whole/
<scene>/assignment/gaussian_part_ids.npy
<scene>/assignment/part_gaussian_index.json
<scene>/assignment/per_part_gaussians/
<scene>/assignment/per_part_aabb.json
<scene>/simulation/sim_config_partphys.json
<scene>/partphys_summary.json
<scene>/warnings.txt
```

## Tests

```bash
cd /root/PartPhysAgent
PYTHONPATH=/root/PartPhysAgent:/root/PhysGM python -m pytest tests/test_partphys_utils.py
```
