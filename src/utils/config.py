from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_pixel_expr(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        node = ast.parse(value, mode="eval")
    except SyntaxError:
        return value
    allowed = (ast.Expression, ast.BinOp, ast.Mult, ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Constant)
    if not all(isinstance(child, allowed) for child in ast.walk(node)):
        return value
    return int(eval(compile(node, "<pixel_expr>", "eval"), {"__builtins__": {}}, {}))


def flatten_config(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        merged.update(config)
    return merged

