from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path, data: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "to_dict"):
        data = data.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_warnings(path, warnings: list[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for warning in warnings:
            f.write(str(warning).rstrip() + "\n")
