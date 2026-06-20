from __future__ import annotations

import argparse
from pathlib import Path

from partphys.agent import PartPhysAgent


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_amp_dtype(device: str) -> str:
    return "bf16" if "cuda" in device else "fp32"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PartPhysAgent inference-only extension for PhysGM.")
    parser.add_argument("--image", type=str, required=True, help="Single RGB input image.")
    parser.add_argument("--scene-name", type=str, required=True, help="Scene name used under output-dir.")
    parser.add_argument("--output-dir", type=str, default="results_partphys", help="Output root directory.")
    parser.add_argument("--physgm-config", type=str, default="../PhysGM/configs/infer.yaml", help="PhysGM infer.yaml path.")
    parser.add_argument("--checkpoint", type=str, default="../PhysGM/checkpoints/checkpoint.pt", help="PhysGM checkpoint path.")
    parser.add_argument("--template-config", type=str, default="../PhysGM/configs/physical/down_template.json", help="PhysGM physical template JSON.")
    parser.add_argument("--physgm-root", type=str, default=None, help="Path to the original PhysGM repository.")
    parser.add_argument("--object", type=str, default=None, help="Optional object name hint.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--multiview-dir", type=str, default=None, help="Existing four-view image directory for whole-object PhysGM input.")
    parser.add_argument("--use-mvadapter", action="store_true", help="Use MV-Adapter to synthesize four views for the whole-object PhysGM input.")
    parser.add_argument("--require-mvadapter", action="store_true", help="Fail instead of falling back when MV-Adapter is unavailable.")
    parser.add_argument("--mvadapter-root", type=str, default="/root/autodl-tmp/MV-Adapter", help="Path to the MV-Adapter repository.")
    parser.add_argument("--mvadapter-variant", choices=["sd", "sdxl"], default="sd", help="MV-Adapter image-to-multiview script variant.")
    parser.add_argument("--mvadapter-device", type=str, default=None, help="Device used by MV-Adapter. Defaults to --device.")
    parser.add_argument("--mvadapter-prompt", type=str, default="high quality object, clean background")
    parser.add_argument("--mvadapter-num-views", type=int, default=6, help="Number of views requested from MV-Adapter; 6 selects the cardinal 0/90/180/270 views.")
    parser.add_argument("--mvadapter-steps", type=int, default=50)
    parser.add_argument("--mvadapter-guidance-scale", type=float, default=3.0)
    parser.add_argument("--mvadapter-seed", type=int, default=1234)
    parser.add_argument("--mvadapter-timeout", type=int, default=1800)
    parser.add_argument("--mvadapter-adapter-path", type=str, default=None, help="Optional local or Hugging Face adapter path override.")
    parser.add_argument("--sam-backend", choices=["sam2", "sam1"], default="sam2")
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="SAM checkpoint path.")
    parser.add_argument("--sam-config", type=str, default=None, help="SAM2 config name, e.g. configs/sam2.1/sam2.1_hiera_l.yaml. Inferred from checkpoint if omitted.")
    parser.add_argument("--sam2-root", type=str, default="/root/autodl-tmp/repos/sam2", help="Path to the SAM2 repository root.")
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--sam-points-per-side", type=int, default=16)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--sam-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=100)
    parser.add_argument("--groundingdino-model", type=str, default=None, help="Hugging Face GroundingDINO model id or local directory.")
    parser.add_argument("--groundingdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-text-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-config", type=str, default=None)
    parser.add_argument("--groundingdino-weights", type=str, default=None)
    parser.add_argument("--part-schema-json", type=str, default=None)
    parser.add_argument("--masks-json", type=str, default=None)
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--require-vlm", action="store_true", default=False)
    parser.add_argument("--vlm-provider", choices=["none", "openai_compatible"], default="none")
    parser.add_argument("--vlm-model", type=str, default=None)
    parser.add_argument("--vlm-api-base", type=str, default=None)
    parser.add_argument("--vlm-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--vlm-timeout", type=int, default=180, help="VLM request timeout in seconds.")
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--simulate", dest="simulate", action="store_true")
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.set_defaults(simulate=False)
    parser.add_argument("--white-bg", dest="white_bg", action="store_true", help="Render simulation on a white background.")
    parser.add_argument("--no-white-bg", dest="white_bg", action="store_false", help="Render simulation on a black background.")
    parser.set_defaults(white_bg=False)
    parser.add_argument("--skip-part-physgm", action="store_true")
    parser.add_argument("--assignment-mode", choices=["projection", "aabb_heuristic", "none"], default="projection")
    parser.add_argument("--max-parts", type=int, default=6)
    parser.add_argument("--min-part-area-ratio", type=float, default=0.002)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--mock-physgm", action="store_true")
    parser.add_argument("--segmentation-only", action="store_true", help="Stop after part masks and Gaussian-part assignment outputs.")
    parser.add_argument("--mask-only", action="store_true", help="Stop after object mask, part masks, and segmentation reports.")
    parser.add_argument("--whole-physgm-dir", type=str, default=None, help="Existing whole-object PhysGM output dir containing point_clouds.ply.")
    parser.add_argument("--segmentation-mode", choices=["candidate_pool", "legacy_vlm_bbox"], default="candidate_pool")
    parser.add_argument("--use-vlm-bbox-proposals", action="store_true", default=False)
    parser.add_argument("--use-schema-location-proposals", action="store_true", default=False)
    parser.add_argument("--strict-segmentation", action="store_true", default=False)
    parser.add_argument("--residual-policy", choices=["ignore", "unknown", "fill_nearest"], default="unknown")
    parser.add_argument("--candidate-top-k", type=int, default=40)
    parser.add_argument("--candidate-contact-sheet-top-k", type=int, default=24)
    parser.add_argument("--max-vlm-candidates-per-part", type=int, default=12)
    parser.add_argument("--segmentation-max-retries", type=int, default=2)
    parser.add_argument("--segmentation-vlm-weight", type=float, default=0.55)
    parser.add_argument("--segmentation-min-accept-score", type=float, default=0.45)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.device = _resolve_device(args.device)
    if args.amp_dtype == "auto":
        args.amp_dtype = _default_amp_dtype(args.device)
    agent = PartPhysAgent(args)
    result = agent.run(args.image, args.scene_name, object_hint=args.object)
    print(f"PartPhysAgent finished: {Path(result.sim_config_path).parent.parent if result.sim_config_path else args.output_dir}")
    print(f"Summary: {Path(args.output_dir) / args.scene_name / 'partphys_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
