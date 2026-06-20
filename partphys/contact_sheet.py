from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .image_utils import overlay_mask, read_mask


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_candidate_contact_sheet(
    image_path,
    candidates,
    output_path,
    max_candidates: int = 24,
    cell_size: int = 256,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.open(image_path).convert("RGB")
    selected = list(candidates)[: max(0, int(max_candidates))]
    if not selected:
        canvas = Image.new("RGB", (cell_size, cell_size), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 16), "no candidates", fill=(0, 0, 0), font=_font(20))
        canvas.save(output_path)
        return str(output_path)

    cols = min(4, max(1, int(math.ceil(math.sqrt(len(selected))))))
    rows = int(math.ceil(len(selected) / cols))
    label_h = 58
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + label_h)), "white")
    id_font = _font(22)
    small_font = _font(15)

    for idx, candidate in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_size
        y0 = row * (cell_size + label_h)
        mask = read_mask(candidate.mask_path)
        thumb = overlay_mask(base, mask, alpha=0.48).resize((cell_size, cell_size), Image.BICUBIC)
        canvas.paste(thumb, (x0, y0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x0, y0, x0 + cell_size, y0 + 34), fill=(255, 255, 255))
        draw.text((x0 + 8, y0 + 4), candidate.candidate_id, fill=(0, 0, 0), font=id_font)
        label_y = y0 + cell_size + 4
        ratio = candidate.metadata.get("object_area_ratio", candidate.area_ratio)
        text = f"{candidate.source}  area={float(ratio):.3f}"
        draw.text((x0 + 8, label_y), text, fill=(0, 0, 0), font=small_font)
        if candidate.prompt:
            draw.text((x0 + 8, label_y + 22), str(candidate.prompt)[:30], fill=(60, 60, 60), font=small_font)

    canvas.save(output_path)
    return str(output_path)
