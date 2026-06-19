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

PART_VERIFY_PROMPT = """Verify whether the selected part masks are physically meaningful, visible, and non-overlapping enough for part-aware MPM simulation. Return strict JSON."""

MATERIAL_VERIFY_PROMPT = """Infer the likely PhysGM material class for the highlighted object part. Use only: Wood, Metal, Plastic, Glass, Fabric, Leather, Ceramic, Stone, Rubber, Paper, Sand, Snow, Plasticine, Foam. Return strict JSON."""
