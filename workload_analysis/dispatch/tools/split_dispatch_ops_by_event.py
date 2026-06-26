#!/usr/bin/env python3
"""Split filtered dispatch ops into one CSV per event_id."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


WORKLOAD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    WORKLOAD_DIR
    / "dispatch/profiles/filtered_dispatch_visipruner_full_32tok/dispatch_ops.csv"
)
DEFAULT_OUTPUT_DIR = WORKLOAD_DIR / "dispatch/visualize"
DEFAULT_KEEP_COLUMNS_1_BASED = (1, 2, 6, 7, 9, 10, 11, 12, 15, 16)


def safe_dir_name(value: str) -> str:
    """Keep event IDs usable as directory names and prevent path traversal."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._/")
    if not safe:
        raise ValueError(f"Cannot derive a safe directory name from event_id={value!r}")
    return safe


def selected_fieldnames(fieldnames: list[str], columns_1_based: tuple[int, ...]) -> list[str]:
    selected: list[str] = []
    for col in columns_1_based:
        index = col - 1
        if index < 0 or index >= len(fieldnames):
            raise ValueError(
                f"Selected column {col} is outside CSV column range 1..{len(fieldnames)}"
            )
        selected.append(fieldnames[index])
    return selected


def parse_column_selection(value: str) -> tuple[int, ...]:
    columns = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not columns:
        raise ValueError("Column selection cannot be empty")
    return columns


def read_dispatch_ops(path: Path, event_column: str) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if event_column not in reader.fieldnames:
            raise ValueError(
                f"CSV is missing required column {event_column!r}; "
                f"available columns: {reader.fieldnames}"
            )
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            event_id = row.get(event_column, "")
            if not event_id:
                raise ValueError(f"Found row with empty {event_column!r}: {row}")
            groups[event_id].append(row)
    return list(reader.fieldnames), dict(groups)


def write_group(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_dispatch_ops(
    input_csv: Path,
    output_dir: Path,
    event_column: str,
    keep_columns_1_based: tuple[int, ...],
) -> list[dict[str, str | int]]:
    fieldnames, groups = read_dispatch_ops(input_csv, event_column)
    output_fieldnames = selected_fieldnames(fieldnames, keep_columns_1_based)
    if event_column not in output_fieldnames:
        raise ValueError(
            f"event column {event_column!r} must be included in output columns; "
            f"selected columns resolve to {output_fieldnames}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str | int]] = []
    for event_id in sorted(groups, key=lambda value: (len(value), value)):
        rows = groups[event_id]
        event_dir = output_dir / safe_dir_name(event_id)
        output_csv = event_dir / "dispatch_ops.csv"
        write_group(output_csv, output_fieldnames, rows)
        index_rows.append(
            {
                "event_id": event_id,
                "row_count": len(rows),
                "output_csv": str(output_csv.relative_to(output_dir)),
            }
        )

    index_csv = output_dir / "dispatch_ops_by_event_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "row_count", "output_csv"])
        writer.writeheader()
        writer.writerows(index_rows)

    index_json = output_dir / "dispatch_ops_by_event_index.json"
    index_json.write_text(json.dumps(index_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--event-column", default="event_id")
    parser.add_argument(
        "--keep-columns",
        default=",".join(str(col) for col in DEFAULT_KEEP_COLUMNS_1_BASED),
        help=(
            "Comma-separated 1-based CSV column numbers to keep in each split file. "
            "Default keeps columns 1,2,6,7,9,10,11,12,15,16."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    keep_columns = parse_column_selection(args.keep_columns)
    index_rows = split_dispatch_ops(input_csv, output_dir, args.event_column, keep_columns)
    print(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_dir": str(output_dir),
                "event_count": len(index_rows),
                "row_count": sum(int(row["row_count"]) for row in index_rows),
                "keep_columns_1_based": list(keep_columns),
                "index_csv": str(output_dir / "dispatch_ops_by_event_index.csv"),
                "index_json": str(output_dir / "dispatch_ops_by_event_index.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
