PART_SCHEMA_PROMPT = """You are a physical reasoning assistant.
Given an object image, decompose the visible object into parts that may have different physical properties for MPM simulation.

Rules:
1. Split parts if they likely have different material, stiffness, density, friction, or deformation behavior.
2. Merge semantic parts if they likely share the same material and physical behavior.
3. Ignore tiny decorative details unless their material strongly affects simulation.
4. Only include visible parts unless the user asks for hidden parts.
5. For each part, output name, text_prompts, expected_materials, location, shape_prior, physical_role, should_simulate_separately, visible, physics_group.
6. Output strict JSON only.

Required JSON schema:
{
  "object": "hammer",
  "parts": [
    {
      "name": "head",
      "text_prompts": ["hammer head", "metal hammer head"],
      "expected_materials": ["Metal"],
      "location": "top/front compact region",
      "shape_prior": "compact block",
      "physical_role": "stiff impact part",
      "should_simulate_separately": true,
      "visible": true,
      "physics_group": "head"
    }
  ],
  "relations": [
    {
      "part_a": "head",
      "part_b": "handle",
      "relation": "rigidly_attached"
    }
  ]
}
"""

MASK_SCORE_PROMPT = """Score whether the highlighted mask corresponds to the requested physical part. Return strict JSON with keys score, reason."""

CANDIDATE_RANK_PROMPT = """Rank existing candidate masks for each requested physical part.

Rules:
1. You are not allowed to produce bounding boxes.
2. You are not allowed to output coordinates.
3. You must select only from the candidate_id values visible in the contact sheet and listed in the candidate metadata.
4. If none of the candidates match a part, set missing=true.
5. Do not invent masks or candidate IDs.
6. Return strict JSON only.

Required JSON schema:
{
  "rankings": [
    {
      "part_name": "cake_body",
      "candidates": [
        {
          "candidate_id": "candidate_003",
          "score": 0.92,
          "reason": "Mask covers the main cake body without plate."
        }
      ],
      "missing": false
    }
  ],
  "warnings": []
}
"""

PART_VERIFY_PROMPT = """Verify whether the selected part masks are physically meaningful, visible, and non-overlapping enough for part-aware MPM simulation. Return strict JSON."""

MATERIAL_VERIFY_PROMPT = """Infer the likely PhysGM material class for the highlighted object part. Use only: Wood, Metal, Plastic, Glass, Fabric, Leather, Ceramic, Stone, Rubber, Paper, Sand, Snow, Plasticine, Foam. Return strict JSON."""

PART_AGENT_PLAN_PROMPT = """You are the planner of a physical part segmentation agent.
Given an object image, or a labeled 2x2 multi-view sheet, and optional previous critique, decide how to decompose the object into physically meaningful parts and which segmentation tools should be used.

Rules:
1. Parts should be split by physical/material behavior, not only visual semantics.
2. Prefer a small number of simulation-useful parts.
3. When multiple labeled views are visible, use all views to decide the part schema, but align the same physical part by name/physics_group rather than by 2D shape.
4. Output only deterministic tool actions from this list: generate_object_mask, repair_object_mask, generate_part_candidates, rank_candidates, compile_layout, multiview_align, gaussian_assign, knn_cleanup.
5. Return strict JSON only.

Required JSON schema:
{
  "object": "cake",
  "parts": [
    {
      "name": "cake_base",
      "text_prompts": ["cake base", "main cake body"],
      "expected_materials": ["Foam", "Plasticine"],
      "location": "main cylindrical body",
      "shape_prior": "large cylinder",
      "physical_role": "soft main body",
      "should_simulate_separately": true,
      "visible": true,
      "physics_group": "cake_base"
    }
  ],
  "tool_plan": [
    {
      "action": "generate_object_mask",
      "reason": "obtain full visible object before part segmentation",
      "parameters": {"fallback": "foreground_union_if_sam_incomplete"}
    }
  ],
  "quality_gates": {
    "max_unknown_ratio": 0.05,
    "max_overlap_ratio": 0.03,
    "require_multiview_alignment": true
  },
  "warnings": []
}
"""

PART_AGENT_CRITIQUE_PROMPT = """You are the critic of a physical part segmentation agent.
Given a selected-parts overlay or labeled multi-view overlay, part metadata, and quality metrics, decide whether the segmentation is acceptable for part-aware 3DGS simulation.

For multi-view overlays, check that each named part is consistently identified across front/right/rear/left views when visible. Do not reject a part merely because it is partly occluded in one view; reject it when the same physical region is assigned to different part names or a required visible part is missing in most views.

Return strict JSON only.

Required JSON schema:
{
  "ok": true,
  "failure_modes": [
    "front object mask misses visible top decorations"
  ],
  "repair_actions": [
    {
      "action": "repair_object_mask",
      "target": "front",
      "parameters": {"method": "foreground_union"},
      "reason": "SAM object mask is incomplete"
    }
  ],
  "notes": []
}
"""
