#!/usr/bin/env python3
"""Audit same-input three-way layer retest artifacts and reports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path("/workspace/VisiPrune")
EXPERIMENT_DIR = ROOT_DIR / "autoresearch/experiments/e2_single_request_latency"
OUTPUT_DIR = EXPERIMENT_DIR / "output"

EXPECTED_IMAGE = str(
    ROOT_DIR / "autoresearch/data/benchmark_images/002901d9d194c4fb.jpg"
)
EXPECTED_PROMPT = "Describe the image briefly."
EXPECTED_TOKENS = 32

VARIANTS = [
    {
        "name": "dense-FA2",
        "config": "dense-fa2",
        "tag": "sameinput_dense_fa2",
        "report": "SAME_INPUT_DENSE_FA2_LAYER_PERFORMANCE_REPORT.md",
        "use_flash_attn": True,
        "use_visipruner": False,
    },
    {
        "name": "eager VisiPruner full",
        "config": "visipruner-full",
        "tag": "sameinput_visipruner_full_eager",
        "report": "SAME_INPUT_VISIPRUNER_FULL_EAGER_LAYER_PERFORMANCE_REPORT.md",
        "use_flash_attn": False,
        "use_visipruner": True,
    },
    {
        "name": "VisiPruner-FA2",
        "config": "visipruner-full-fa2",
        "tag": "sameinput_visipruner_full_fa2",
        "report": "SAME_INPUT_VISIPRUNER_FULL_FA2_LAYER_PERFORMANCE_REPORT.md",
        "use_flash_attn": True,
        "use_visipruner": True,
    },
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require_file(path: Path, failures: list[str]) -> bool:
    if not path.exists():
        failures.append(f"missing file: {path}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        failures.append(f"empty file: {path}")
        return False
    return True


def layer_coverage(rows: list[dict[str, str]], phase: str) -> set[int]:
    covered: set[int] = set()
    for row in rows:
        if row.get("phase") != phase:
            continue
        try:
            covered.add(int(row["layer_idx"]))
        except Exception:
            continue
    return covered


def decode_occurrences(rows: list[dict[str, str]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in rows:
        if row.get("phase") != "decode":
            continue
        try:
            counts[int(row["layer_idx"])] += 1
        except Exception:
            continue
    return counts


def markdown_table_layers(text: str, heading_prefix: str) -> set[int]:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(heading_prefix):
            start = idx
            break
    if start is None:
        return set()

    layers: set[int] = set()
    in_table = False
    for line in lines[start + 1 :]:
        if line.startswith("## ") and in_table:
            break
        if line.startswith("| layer |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if layers:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or not cells[0].isdigit():
            continue
        layers.add(int(cells[0]))
    return layers


def report_has_placeholders(text: str) -> bool:
    bad_fragments = [
        "no prefill layer events found",
        "no decode layer events found",
        "| 0 | - |",
        "| 31 | - |",
        "| - | - |",
    ]
    return any(fragment in text for fragment in bad_fragments)


def git_human_draft_status(root_dir: Path) -> str:
    path = "workload_analysis/human_draft.md"
    try:
        proc = subprocess.run(
            ["git", "status", "--short", "--", path],
            cwd=root_dir,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return f"git status unavailable: {exc}"
    return proc.stdout.strip() or "clean-or-untracked-clean"


def audit_variant(
    variant: dict[str, Any],
    *,
    output_dir: Path,
    report_dir: Path,
    tokens: int,
    failures: list[str],
) -> dict[str, Any]:
    clock_tag = f"clock_{variant['tag']}_{tokens}tok"
    nsys_tag = f"nsys_{variant['tag']}_{tokens}tok"
    report_path = report_dir / variant["report"]

    required = [
        output_dir / f"{clock_tag}.json",
        output_dir / f"{clock_tag}.log",
        output_dir / f"{clock_tag}_ranges.csv",
        output_dir / f"{clock_tag}_layer_events.csv",
        output_dir / f"{nsys_tag}.json",
        output_dir / f"{nsys_tag}.log",
        output_dir / f"{nsys_tag}.nsys-rep",
        output_dir / f"{nsys_tag}.sqlite",
        output_dir / f"{nsys_tag}_ranges.csv",
        output_dir / f"{nsys_tag}_layer_events.csv",
        output_dir / f"{nsys_tag}_layer_kernel_breakdown.csv",
        output_dir / f"{nsys_tag}_layer_kernel_breakdown.json",
        output_dir / f"{nsys_tag}_stats_cuda_gpu_kern_sum.csv",
        output_dir / f"{nsys_tag}_stats_nvtx_gpu_proj_sum.csv",
        output_dir / f"{nsys_tag}_stats_nvtx_kern_sum.csv",
        output_dir / f"{nsys_tag}_stats_nvtx_sum.csv",
        report_path,
    ]
    existing = {str(path): require_file(path, failures) for path in required}

    if not existing.get(str(output_dir / f"{clock_tag}.json")):
        return {"name": variant["name"], "ok": False, "clock_tag": clock_tag, "nsys_tag": nsys_tag}

    clock = read_json(output_dir / f"{clock_tag}.json")
    config = clock.get("config")
    if config != variant["config"]:
        failures.append(f"{variant['name']}: config mismatch {config!r} != {variant['config']!r}")
    if clock.get("use_flash_attn") is not variant["use_flash_attn"]:
        failures.append(f"{variant['name']}: use_flash_attn mismatch")
    if clock.get("use_visipruner") is not variant["use_visipruner"]:
        failures.append(f"{variant['name']}: use_visipruner mismatch")

    layer_events_path = output_dir / f"{clock_tag}_layer_events.csv"
    if layer_events_path.exists():
        rows = read_csv(layer_events_path)
        prefill = layer_coverage(rows, "prefill")
        decode = layer_coverage(rows, "decode")
        if prefill != set(range(32)):
            failures.append(f"{variant['name']}: clock prefill layer coverage {sorted(prefill)}")
        if decode != set(range(32)):
            failures.append(f"{variant['name']}: clock decode layer coverage {sorted(decode)}")
        decode_counts = decode_occurrences(rows)
        missing_decode_repeats = [layer for layer in range(32) if decode_counts[layer] < 1]
        if missing_decode_repeats:
            failures.append(
                f"{variant['name']}: missing clock decode occurrences for layers {missing_decode_repeats}"
            )

    nsys_layer_path = output_dir / f"{nsys_tag}_layer_kernel_breakdown.csv"
    if nsys_layer_path.exists():
        rows = read_csv(nsys_layer_path)
        total_rows = [row for row in rows if row.get("component") == "total"]
        prefill = layer_coverage(total_rows, "prefill")
        decode = layer_coverage(total_rows, "decode")
        if prefill != set(range(32)):
            failures.append(f"{variant['name']}: Nsight prefill layer coverage {sorted(prefill)}")
        if decode != set(range(32)):
            failures.append(f"{variant['name']}: Nsight decode layer coverage {sorted(decode)}")

    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        prefill_layers = markdown_table_layers(text, "## Forward 1: Prefill")
        decode_layers = markdown_table_layers(text, "## Decode forwards:")
        if prefill_layers != set(range(32)):
            failures.append(f"{variant['name']}: report prefill layer coverage {sorted(prefill_layers)}")
        if decode_layers != set(range(32)):
            failures.append(f"{variant['name']}: report decode layer coverage {sorted(decode_layers)}")
        if report_has_placeholders(text):
            failures.append(f"{variant['name']}: report contains placeholder-looking fields")

    return {
        "name": variant["name"],
        "clock_tag": clock_tag,
        "nsys_tag": nsys_tag,
        "clock_json": str(output_dir / f"{clock_tag}.json"),
        "report": str(report_path),
        "image_path": clock.get("image_path"),
        "prompt": clock.get("prompt"),
        "max_new_tokens": clock.get("max_new_tokens"),
        "config": clock.get("config"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(ROOT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(EXPERIMENT_DIR))
    parser.add_argument("--tokens", type=int, default=EXPECTED_TOKENS)
    parser.add_argument("--expected-image", default=EXPECTED_IMAGE)
    parser.add_argument("--expected-prompt", default=EXPECTED_PROMPT)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    failures: list[str] = []
    summaries = [
        audit_variant(
            variant,
            output_dir=output_dir,
            report_dir=report_dir,
            tokens=args.tokens,
            failures=failures,
        )
        for variant in VARIANTS
    ]

    inputs = {
        (item.get("image_path"), item.get("prompt"), item.get("max_new_tokens"))
        for item in summaries
        if item.get("image_path") is not None
    }
    if len(inputs) != 1:
        failures.append(f"same-input check failed: observed inputs={sorted(inputs)}")
    elif inputs:
        image_path, prompt, max_new_tokens = next(iter(inputs))
        if image_path != args.expected_image:
            failures.append(f"image_path mismatch: {image_path!r} != {args.expected_image!r}")
        if prompt != args.expected_prompt:
            failures.append(f"prompt mismatch: {prompt!r} != {args.expected_prompt!r}")
        if int(max_new_tokens) != args.tokens:
            failures.append(f"max_new_tokens mismatch: {max_new_tokens!r} != {args.tokens!r}")

    human_status = git_human_draft_status(root_dir)
    if human_status.startswith(" M") or human_status.startswith("M "):
        failures.append(f"workload_analysis/human_draft.md modified according to git: {human_status}")

    payload = {
        "ok": not failures,
        "variant_count": len(VARIANTS),
        "summaries": summaries,
        "human_draft_git_status": human_status,
        "failures": failures,
    }
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if failures:
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
