#!/usr/bin/env bash
set -euo pipefail

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before running this script}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export MVADAPTER_BASE_MODEL="/root/autodl-tmp/models/sd21-base-diffusers"

python3 partphys_pipeline.py \
  --image examples/hammer.png \
  --scene-name hammer_qwen_part_agent \
  --object hammer \
  --output-dir /root/autodl-tmp/results_partphys \
  --vlm-provider openai_compatible \
  --vlm-model qwen3.7-plus \
  --vlm-api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --vlm-api-key-env DASHSCOPE_API_KEY \
  --vlm-timeout 180 \
  --physgm-root /root/PhysGM \
  --physgm-config configs/infer.yaml \
  --checkpoint /root/PhysGM/checkpoints/checkpoint.pt \
  --template-config /root/PhysGM/configs/physical/down_template.json \
  --use-mvadapter \
  --mvadapter-root /root/autodl-tmp/MV-Adapter \
  --mvadapter-adapter-path /root/autodl-tmp/models/mv-adapter \
  --mvadapter-variant sd \
  --mvadapter-num-views 6 \
  --mvadapter-steps 50 \
  --groundingdino-model /root/autodl-tmp/models/grounding-dino-base \
  --groundingdino-box-threshold 0.25 \
  --groundingdino-text-threshold 0.25 \
  --sam-checkpoint /root/autodl-tmp/models/sam2/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-root /root/autodl-tmp/repos/sam2 \
  --simulate \
  --assignment-mode projection \
  --segmentation-max-retries 2 \
  --segmentation-vlm-weight 0.55 \
  --segmentation-min-accept-score 0.45
