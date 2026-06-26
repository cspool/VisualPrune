from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_dispatch_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_dispatch_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty dispatch rows to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_json_field(value: str) -> Any:
    if value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, dict) and value.get("type") == "Tensor":
        shape = value.get("shape")
        if isinstance(shape, list) and all(isinstance(dim, int) for dim in shape):
            return shape
    return None


def output_tensor_shape(row: dict[str, str]) -> list[int] | None:
    return tensor_shape(parse_json_field(row.get("outputs", "")))


def iter_arg_tensors(row: dict[str, str]) -> list[dict[str, Any]]:
    args = parse_json_field(row.get("args", ""))
    if not isinstance(args, list):
        return []
    return [item for item in args if isinstance(item, dict) and item.get("type") == "Tensor"]


def event_ids(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["event_id"] for row in rows if row.get("event_id")})


def filter_event(rows: list[dict[str, str]], event_id: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("event_id") == event_id]
    return sorted(selected, key=lambda row: int(row.get("event_op_index") or 0))


def summarize_ops(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row.get("op_name", "") for row in rows))
