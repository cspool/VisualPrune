#!/usr/bin/env python3
"""Compare two algorithmic trace JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
WORKLOAD_DIR = ROOT_DIR / "workload_analysis"
DEFAULT_OUTPUT_DIR = WORKLOAD_DIR / "algorithmic_trace/comparisons"


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(saved: int | float, base: int | float) -> float:
    return 100.0 * float(saved) / float(base) if base else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Dense or reference trace JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate trace JSON, e.g. VisiPrune.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tag", default="fresh_visipruner_vs_dense_32tok")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bsum = baseline["summary"]
    csum = candidate["summary"]
    b_total = int(bsum["total_flops_actual_model_path"])
    c_total = int(csum["total_flops_actual_model_path"])
    total_saved = b_total - c_total

    phase_rows = []
    phases = sorted(set(bsum["by_phase_flops"]) | set(csum["by_phase_flops"]))
    for phase in phases:
        b = int(bsum["by_phase_flops"].get(phase, 0))
        c = int(csum["by_phase_flops"].get(phase, 0))
        phase_rows.append({
            "phase": phase,
            "baseline_flops": b,
            "candidate_flops": c,
            "saved_flops": b - c,
            "saved_pct_of_baseline_phase": pct(b - c, b),
        })

    op_rows = []
    ops = sorted(set(bsum["by_op_flops"]) | set(csum["by_op_flops"]))
    for op in ops:
        b = int(bsum["by_op_flops"].get(op, 0))
        c = int(csum["by_op_flops"].get(op, 0))
        op_rows.append({
            "op_name": op,
            "baseline_flops": b,
            "candidate_flops": c,
            "saved_flops": b - c,
            "saved_pct_of_baseline_op": pct(b - c, b),
        })
    op_rows.sort(key=lambda row: row["saved_flops"], reverse=True)

    payload = {
        "baseline_trace": str(Path(args.baseline).resolve()),
        "candidate_trace": str(Path(args.candidate).resolve()),
        "baseline_config": baseline.get("config"),
        "candidate_config": candidate.get("config"),
        "baseline_total_flops_actual_model_path": b_total,
        "candidate_total_flops_actual_model_path": c_total,
        "saved_flops_actual_model_path": total_saved,
        "saved_pct_actual_model_path": pct(total_saved, b_total),
        "baseline_total_flops_with_ideal_lm_head": int(bsum["total_flops_with_ideal_lm_head"]),
        "candidate_total_flops_with_ideal_lm_head": int(csum["total_flops_with_ideal_lm_head"]),
        "phase_comparison": phase_rows,
        "top_saved_ops": op_rows[:20],
    }

    json_path = out_dir / f"{args.tag}.json"
    phase_csv = out_dir / f"{args.tag}_phase.csv"
    op_csv = out_dir / f"{args.tag}_ops.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with phase_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    with op_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(op_rows[0]))
        writer.writeheader()
        writer.writerows(op_rows)

    print(json.dumps({
        "comparison": str(json_path),
        "phase_csv": str(phase_csv),
        "op_csv": str(op_csv),
        "saved_pct_actual_model_path": payload["saved_pct_actual_model_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
