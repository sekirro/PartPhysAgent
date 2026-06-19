#!/usr/bin/env bash
set -euo pipefail

python partphys_pipeline.py \
  --image examples/hammer.png \
  --scene-name hammer_001 \
  --object hammer \
  --output-dir /root/autodl-tmp/results_partphys \
  --physgm-config ../PhysGM/configs/infer.yaml \
  --checkpoint ../PhysGM/checkpoints/checkpoint.pt \
  --template-config ../PhysGM/configs/physical/down_template.json \
  --sam-checkpoint /root/autodl-tmp/models/sam2/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-root /root/autodl-tmp/repos/sam2 \
  --no-simulate
