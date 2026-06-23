from __future__ import annotations

from .types import PartSpec


MAIN_LIKE_KEYWORDS = ("body", "main", "base", "shell", "cake", "object")
SPECIFIC_KEYWORDS = (
    "wheel",
    "tire",
    "handle",
    "head",
    "lace",
    "sole",
    "window",
    "strawberry",
    "topping",
    "icing",
    "cream",
    "plate",
    "grip",
)


def part_text(part: PartSpec) -> str:
    return " ".join(
        [
            part.name,
            part.location,
            part.shape_prior,
            *list(part.text_prompts or []),
        ]
    ).lower()


def is_main_like_part(part: PartSpec) -> bool:
    text = part_text(part)
    strong_main = any(key in text for key in ("body", "main", "base", "shell", "object"))
    if any(key in text for key in SPECIFIC_KEYWORDS) and not strong_main:
        return False
    return any(key in text for key in MAIN_LIKE_KEYWORDS)


def is_specific_part(part: PartSpec) -> bool:
    text = part_text(part)
    return any(key in text for key in SPECIFIC_KEYWORDS)


def is_collection_part(part: PartSpec) -> bool:
    text = part_text(part)
    if any(key in text for key in ("frosting", "icing", "cream")):
        return False
    return any(
        key in text
        for key in (
            "strawberries",
            "berries",
            "toppings",
            "decorations",
            "pieces",
            "multiple",
            "arranged",
        )
    )


def is_plate_like_part(part: PartSpec) -> bool:
    text = part_text(part)
    return any(key in text for key in ("plate", "dish", "stand", "disc", "disk", "tray"))


def is_frosting_like_part(part: PartSpec) -> bool:
    text = part_text(part)
    return any(key in text for key in ("frosting", "icing", "cream", "drip"))


def specificity_rank(part: PartSpec) -> tuple[int, int, str]:
    if is_specific_part(part):
        return (0, 0, part.name)
    if is_main_like_part(part):
        return (2, 0, part.name)
    return (1, 0, part.name)
