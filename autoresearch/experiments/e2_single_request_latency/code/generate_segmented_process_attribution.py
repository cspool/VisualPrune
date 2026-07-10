#!/usr/bin/env python3
"""Generate full-layer segmented process attribution for VisiPruner full eager."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VARIANT = "visipruner-full-eager"

PROCESS_ORDER = [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention_scores",
    "attention_output",
    "output_projection",
    "post_attention_rmsnorm",
    "mlp",
    "visual_process",
]

PROCESS_COST_TYPE = {
    "input_rmsnorm": "norm",
    "post_attention_rmsnorm": "norm",
    "qkv_projection": "projection",
    "output_projection": "projection",
    "mlp": "mlp",
    "rope": "position_encoding",
    "attention_scores": "attention",
    "attention_output": "attention",
    "visual_process": "pruning_selection",
}

METRICS = [
    {
        "metric": "CUPTI kernel ms",
        "target_field": "kernel_total_ms",
        "representative_field": "CUPTI kernel ms",
    },
    {
        "metric": "NVTX CPU ms",
        "target_field": "range_ms",
        "representative_field": "NVTX CPU ms",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return default
        return int(float(value))
    except Exception:
        return default


def fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def pct(numer: float, denom: float) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{100.0 * numer / denom:.2f}%"


def compact_unique(values: list[Any], *, limit: int = 6) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.append(text)
    if len(seen) > limit:
        return "; ".join(seen[:limit]) + f"; ... (+{len(seen) - limit})"
    return "; ".join(seen)


def compact_numeric_values(values: list[Any]) -> str:
    numbers = sorted({inum(value) for value in values if str(value or "").strip()})
    if not numbers:
        return ""
    if len(numbers) == 1:
        return str(numbers[0])
    return f"{numbers[0]}-{numbers[-1]}"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(md_escape(v) for v in row) + " |")
    return out


def compact_ranges(values: list[int]) -> str:
    if not values:
        return "-"
    values = sorted(set(values))
    groups: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in groups)


def event_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["phase"]), inum(row["forward_id"]), inum(row["layer"]))


def component_key(row: dict[str, str]) -> tuple[str, int, int, str]:
    phase = row.get("phase", "")
    occurrence = inum(row.get("occurrence"))
    layer = inum(row.get("layer_idx"))
    component = row.get("component", "")
    return (phase, occurrence, layer, component)


def family_values(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in row.items():
        if not key.endswith("_ms"):
            continue
        if key in {"range_ms", "kernel_total_ms"}:
            continue
        family = key[: -len("_ms")]
        amount = fnum(value)
        if amount > 0:
            values[family] = amount
    return values


def dominant_family_from_text(value: str) -> str:
    value = value.strip()
    if not value:
        return "-"
    return value.split("(", 1)[0].strip()


def process_sort_key(process_id: str) -> tuple[int, str]:
    try:
        return (PROCESS_ORDER.index(process_id), process_id)
    except ValueError:
        return (len(PROCESS_ORDER), process_id)


def infer_role(phase: str, layer: int, occurrence: int, workload: str) -> str:
    if phase == "prefill":
        if "layer0_mass_fold" in workload:
            return "prefill_layer0_mass_fold"
        if "shallow_text_to_vision_mask" in workload:
            return "prefill_shallow_text_to_vision_mask"
        if "last_query_proxy_or_selection" in workload:
            if layer == 6:
                return "prefill_last_query_proxy_without_visual_process"
            return "prefill_last_query_proxy_with_visual_process"
        if "middle_pruned_compact_prefill" in workload:
            return "prefill_middle_pruned_compact"
        if "deep_removed_prefill" in workload:
            return "prefill_deep_removed"
        return f"prefill_unknown_{workload}"
    if layer <= 18:
        return "decode_full_kv_cache"
    if layer <= 27:
        return "decode_middle_pruned_kv_cache"
    return "decode_deep_removed_kv_cache"


def expected_retained_tokens(phase: str, layer: int, occurrence: int, kv_len: int) -> str:
    if phase == "prefill":
        return str(kv_len)
    if layer <= 18:
        return f"full_kv:{kv_len}"
    if layer <= 27:
        return f"middle_pruned_kv:{kv_len}"
    return f"deep_removed_kv:{kv_len}"


def feature_signature(row: dict[str, Any], process_set: list[str] | None = None) -> str:
    parts = [
        f"phase={row.get('phase')}",
        f"forward={row.get('forward_id')}",
        f"occurrence={row.get('occurrence')}",
        f"layer={row.get('layer')}",
        f"role={row.get('role')}",
        f"q_len={row.get('q_len')}",
        f"kv_len={row.get('kv_len')}",
        f"retained={row.get('retained_tokens')}",
        f"workload={row.get('workload_type')}",
        f"dominant_family={row.get('dominant_family')}",
    ]
    if process_set is not None:
        parts.append("processes=" + ",".join(process_set))
    return "; ".join(parts)


def target_complexity(row: dict[str, Any], process_id: str) -> tuple[str, float]:
    q_len = max(1, inum(row.get("q_len"), 1))
    kv_len = max(1, inum(row.get("kv_len"), 1))
    if process_id in {"attention_scores", "attention_output", "visual_process"}:
        return "q_len*kv_len", float(q_len * kv_len)
    return "q_len", float(q_len)


def validation_status(score: float, exact: bool) -> tuple[str, str]:
    if exact:
        return ("pass", "high")
    if score >= 0.82:
        return ("pass", "high")
    if score >= 0.68:
        return ("warning", "medium")
    return ("fail", "low")


@dataclass
class RepresentativeMethod:
    key: tuple[str, int, int]
    representative_layer_id: str
    phase: str
    forward_id: int
    occurrence: int
    layer: int
    q_len: int
    kv_len: int
    workload_type: str
    role: str
    retained_tokens: str
    dominant_family: str
    process_set: list[str]
    rows_by_process: dict[str, dict[str, str]]
    feature_signature: str
    expected_op_families: str
    expected_kernel_families: str
    boundary_role: str
    confidence: str


def representative_id(phase: str, forward_id: int, occurrence: int, layer: int) -> str:
    return f"fwd{forward_id:02d}_{phase}_occ{occurrence:02d}_layer{layer:02d}"


def build_full_rows(
    full_input_rows: list[dict[str, str]],
    layer_breakdown_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    totals = {
        component_key(row): row
        for row in layer_breakdown_rows
        if row.get("component") == "total"
    }
    rows: list[dict[str, Any]] = []
    for row in sorted(full_input_rows, key=lambda r: inum(r.get("event_id"))):
        phase = row.get("phase", "")
        forward_id = inum(row.get("forward_id"))
        occurrence = inum(row.get("occurrence"))
        layer = inum(row.get("layer"))
        total = totals.get((phase, occurrence, layer, "total"), {})
        q_len = inum(row.get("q_len") or total.get("q_len"))
        kv_len = inum(row.get("kv_len") or total.get("kv_len"))
        workload = row.get("workload_type") or total.get("workload_type", "")
        role = infer_role(phase, layer, occurrence, workload)
        family = total.get("dominant_family") or dominant_family_from_text(row.get("dominant_kernel_family", ""))
        out = {
            "variant": VARIANT,
            "event_id": inum(row.get("event_id")),
            "forward_id": forward_id,
            "phase": phase,
            "occurrence": occurrence,
            "layer": layer,
            "q_len": q_len,
            "kv_len": kv_len,
            "workload_type": workload,
            "operator_path": row.get("operator_path", ""),
            "role": role,
            "retained_tokens": expected_retained_tokens(phase, layer, occurrence, kv_len),
            "dominant_family": family,
            "family_values": family_values(total),
            "range_ms": fnum(total.get("range_ms") or row.get("nvtx_cpu_range_ms")),
            "kernel_total_ms": fnum(total.get("kernel_total_ms") or row.get("cupti_launch_owned_kernel_sum_ms")),
        }
        out["target_feature_signature"] = feature_signature(out)
        rows.append(out)
    return rows


def build_representatives(
    process_rows: list[dict[str, str]],
    full_by_key: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[tuple[str, int, int], RepresentativeMethod]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in process_rows:
        phase = row.get("phase", "")
        forward_id = inum(row.get("forward_id"))
        layer = inum(row.get("layer"))
        grouped[(phase, forward_id, layer)].append(row)

    representatives: dict[tuple[str, int, int], RepresentativeMethod] = {}
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        phase, forward_id, layer = key
        occurrence = 0 if phase == "prefill" else max(0, forward_id - 2)
        full = full_by_key.get(key, {})
        q_len = inum(rows[0].get("q_len") or full.get("q_len"))
        kv_len = inum(rows[0].get("kv_len") or full.get("kv_len"))
        workload = str(full.get("workload_type") or "")
        if not workload:
            workload = "representative_process_trace"
        role = infer_role(phase, layer, occurrence, workload)
        process_set = sorted({row["process_id"] for row in rows}, key=process_sort_key)
        row_by_process = {row["process_id"]: row for row in rows}
        method_row = {
            "phase": phase,
            "forward_id": forward_id,
            "occurrence": occurrence,
            "layer": layer,
            "q_len": q_len,
            "kv_len": kv_len,
            "workload_type": workload,
            "role": role,
            "retained_tokens": expected_retained_tokens(phase, layer, occurrence, kv_len),
            "dominant_family": str(full.get("dominant_family") or rows[0].get("dominant kernel family", "")),
        }
        expected_op_families = "; ".join(
            f"{process}:{row_by_process[process].get('fx_op_families', '')}"
            for process in process_set
        )
        expected_kernel_families = "; ".join(
            f"{process}:{row_by_process[process].get('expected_kernel_families', '')}"
            for process in process_set
        )
        representatives[key] = RepresentativeMethod(
            key=key,
            representative_layer_id=representative_id(phase, forward_id, occurrence, layer),
            phase=phase,
            forward_id=forward_id,
            occurrence=occurrence,
            layer=layer,
            q_len=q_len,
            kv_len=kv_len,
            workload_type=workload,
            role=role,
            retained_tokens=method_row["retained_tokens"],
            dominant_family=method_row["dominant_family"],
            process_set=process_set,
            rows_by_process=row_by_process,
            feature_signature=feature_signature(method_row, process_set),
            expected_op_families=expected_op_families,
            expected_kernel_families=expected_kernel_families,
            boundary_role=role,
            confidence="high",
        )
    return representatives


def compatibility_score(target: dict[str, Any], rep: RepresentativeMethod, exact: bool) -> float:
    if exact:
        return 1.0
    score = 0.0
    if target["phase"] == rep.phase:
        score += 0.15
    if target["role"] == rep.role:
        score += 0.35
    if target["workload_type"] == rep.workload_type:
        score += 0.12
    if dominant_family_from_text(str(target["dominant_family"])) == dominant_family_from_text(rep.dominant_family):
        score += 0.08
    tq, tk = max(1, target["q_len"]), max(1, target["kv_len"])
    rq, rk = max(1, rep.q_len), max(1, rep.kv_len)
    q_ratio = min(tq, rq) / max(tq, rq)
    kv_ratio = min(tk, rk) / max(tk, rk)
    score += 0.15 * ((q_ratio + kv_ratio) / 2.0)
    layer_distance = abs(target["layer"] - rep.layer) / 31.0
    occ_distance = abs(target["occurrence"] - rep.occurrence) / 30.0 if target["phase"] == "decode" else 0.0
    score += 0.15 * max(0.0, 1.0 - ((layer_distance + occ_distance) / 2.0))
    return min(score, 0.99)


def representative_distance(target: dict[str, Any], rep: RepresentativeMethod) -> tuple[float, float, float]:
    q = abs(target["q_len"] - rep.q_len) / max(1, target["q_len"], rep.q_len)
    kv = abs(target["kv_len"] - rep.kv_len) / max(1, target["kv_len"], rep.kv_len)
    layer = abs(target["layer"] - rep.layer) / 31.0
    occurrence = abs(target["occurrence"] - rep.occurrence) / 30.0 if target["phase"] == "decode" else 0.0
    return (q + kv + occurrence, layer, occurrence)


def choose_representative(
    target: dict[str, Any],
    representatives: dict[tuple[str, int, int], RepresentativeMethod],
) -> tuple[RepresentativeMethod | None, bool, float, str]:
    key = event_key(target)
    if key in representatives:
        rep = representatives[key]
        return (
            rep,
            True,
            1.0,
            "direct process-level NVTX/CUPTI evidence for this exact input-layer",
        )

    candidates = [
        rep
        for rep in representatives.values()
        if rep.phase == target["phase"] and rep.role == target["role"]
    ]
    if not candidates:
        return (None, False, 0.0, "no representative with matching phase and feature role")

    candidates.sort(key=lambda rep: representative_distance(target, rep))
    rep = candidates[0]
    score = compatibility_score(target, rep, exact=False)
    basis = (
        "same phase, inferred feature role, workload/process set; "
        "nearest representative selected by q_len/kv_len, occurrence, and layer distance"
    )
    return (rep, False, score, basis)


def assignment_source(exact: bool, status: str) -> str:
    if exact:
        return "observed_fx_op"
    if status == "fail":
        return "unknown"
    return "template_scaled"


def boundary_delta(prev: dict[str, Any] | None, cur: dict[str, Any]) -> tuple[str, str]:
    if prev is None:
        return ("start of full input-layer sequence", "start")
    reasons = []
    deltas = []
    for field, reason in [
        ("phase", "prefill/decode switch"),
        ("role", "feature role changed"),
        ("workload_type", "workload type changed"),
        ("representative_layer_id", "representative changed"),
        ("attribution_source", "evidence source changed"),
    ]:
        if prev.get(field) != cur.get(field):
            reasons.append(reason)
            deltas.append(f"{field}:{prev.get(field)}->{cur.get(field)}")
    if prev.get("kv_len") != cur.get("kv_len"):
        deltas.append(f"kv_len:{prev.get('kv_len')}->{cur.get('kv_len')}")
    if prev.get("q_len") != cur.get("q_len"):
        deltas.append(f"q_len:{prev.get('q_len')}->{cur.get('q_len')}")
    if not reasons:
        reasons.append("contiguous compatible interval continues")
    return ("; ".join(dict.fromkeys(reasons)), "; ".join(deltas) or "none")


def build_assignments(
    full_rows: list[dict[str, Any]],
    representatives: dict[tuple[str, int, int], RepresentativeMethod],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provisional: list[dict[str, Any]] = []
    for target in full_rows:
        rep, exact, score, basis = choose_representative(target, representatives)
        if rep is None:
            status = "fail"
            confidence = "low"
            source = "unknown"
            rep_id = ""
            rep_sig = ""
            processes = []
            validation_reason = "No compatible representative method exists; row must remain unknown."
        else:
            status, confidence = validation_status(score, exact)
            source = assignment_source(exact, status)
            rep_id = rep.representative_layer_id
            rep_sig = rep.feature_signature
            processes = rep.process_set
            if exact:
                validation_reason = (
                    "Exact representative process trace exists; process weights are normalized "
                    "to this run's measured layer-wise total for conservation."
                )
            else:
                validation_reason = (
                    "Feature-compatible representative selected; process set is migrated by "
                    "complexity-scaled weights and normalized to the target layer metric."
                )
        provisional.append(
            {
                **target,
                "representative": rep,
                "representative_layer_id": rep_id,
                "representative_forward_id": rep.forward_id if rep else "",
                "representative_occurrence": rep.occurrence if rep else "",
                "representative_layer": rep.layer if rep else "",
                "representative_feature_signature": rep_sig,
                "feature_match_basis": basis,
                "feature_match_score": score,
                "template_validation_status": status,
                "template_confidence": confidence,
                "validation_reason": validation_reason,
                "attribution_source": source,
                "process_set": processes,
            }
        )

    assignments: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    interval_index = 0

    def flush_interval(rows: list[dict[str, Any]], prev_row: dict[str, Any] | None) -> None:
        nonlocal interval_index
        if not rows:
            return
        first = rows[0]
        rep = first.get("representative")
        interval_index += 1
        attr_type = f"T{interval_index:04d}_{first['phase']}_{first['role']}_{first['attribution_source']}"
        reason, delta = boundary_delta(prev_row, first)
        for row in rows:
            row["attribution_type_id"] = attr_type
            assignments.append(row)

        layers = [inum(row["layer"]) for row in rows]
        events = [inum(row["event_id"]) for row in rows]
        forwards = [inum(row["forward_id"]) for row in rows]
        occurrences = [inum(row["occurrence"]) for row in rows]
        target_feature_summary = (
            f"events={compact_ranges(events)}; forwards={compact_ranges(forwards)}; "
            f"occurrences={compact_ranges(occurrences)}; layers={compact_ranges(layers)}; "
            f"role={first['role']}; q={compact_ranges([inum(row['q_len']) for row in rows])}; "
            f"kv={compact_ranges([inum(row['kv_len']) for row in rows])}"
        )
        intervals.append(
            {
                "variant": VARIANT,
                "phase": first["phase"],
                "attribution_type_id": attr_type,
                "layer_interval": target_feature_summary,
                "boundary_reason": reason,
                "boundary_feature_delta": delta,
                "representative_layer_id": first.get("representative_layer_id", ""),
                "representative_layer_method": rep.boundary_role if rep else "",
                "representative_feature_signature": first.get("representative_feature_signature", ""),
                "target_feature_summary": target_feature_summary,
                "template_match_basis": first.get("feature_match_basis", ""),
                "expected_processes": ",".join(rep.process_set) if rep else "",
                "expected_op_families": rep.expected_op_families if rep else "",
                "complexity_features": "attention_scores/attention_output/visual_process=q_len*kv_len; others=q_len",
                "normalization_scope": "layer_total_metric",
                "confidence": first.get("template_confidence", ""),
                "fallback_policy": "downgrade to unknown when no same-role representative exists",
            }
        )

    prev_interval_last: dict[str, Any] | None = None
    for row in provisional:
        key = (
            row["phase"],
            row["role"],
            row["representative_layer_id"],
            row["attribution_source"],
            row["template_validation_status"],
        )
        if current_key is None:
            current_key = key
        if key != current_key:
            flush_interval(current_rows, prev_interval_last)
            prev_interval_last = current_rows[-1] if current_rows else prev_interval_last
            current_rows = []
            current_key = key
        current_rows.append(row)
    flush_interval(current_rows, prev_interval_last)

    return assignments, intervals


def build_process_attribution(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_group_indices: dict[tuple[str, str, int, int, str], list[int]] = defaultdict(list)

    for assignment in assignments:
        rep: RepresentativeMethod | None = assignment.get("representative")
        if rep is None:
            for metric_def in METRICS:
                source_ms = fnum(assignment.get(metric_def["target_field"]))
                row = {
                    "variant": VARIANT,
                    "phase": assignment["phase"],
                    "layer": assignment["layer"],
                    "occurrence": assignment["occurrence"],
                    "attribution_type_id": assignment["attribution_type_id"],
                    "attribution_source": "unknown",
                    "representative_layer_id": "",
                    "process": "unknown",
                    "cost_type": "unknown",
                    "metric": metric_def["metric"],
                    "ms": fmt(source_ms),
                    "pct_of_layer_metric": "100.00%",
                    "source_layer_metric_ms": fmt(source_ms),
                    "representative_process_ms": "",
                    "complexity_formula": "",
                    "target_complexity": "",
                    "representative_complexity": "",
                    "complexity_ratio": "",
                    "raw_weight_ms": "",
                    "normalization_scope": "layer_total_metric",
                    "process_title": "",
                    "fx_nodes": "",
                    "fx_op_families": "",
                    "matched_sample_families": "",
                    "representative_matched_kernel_families": "",
                    "representative_dominant_kernel_family": "",
                    "representative_kernel_family_ms": "",
                    "representative_runtime_api_calls": "",
                    "representative_kernel_instances": "",
                    "representative_validation_status": "",
                    "attribution_method": "unknown",
                    "fallback_reason": assignment["validation_reason"],
                    "conservation_error_ms": "0.000000",
                    "expected_kernel_families": "",
                    "template_validation_status": assignment["template_validation_status"],
                }
                by_group_indices[
                    (VARIANT, assignment["phase"], assignment["layer"], assignment["occurrence"], metric_def["metric"])
                ].append(len(output))
                output.append(row)
            continue

        target_families = ",".join(assignment.get("family_values", {}).keys())
        for metric_def in METRICS:
            metric = metric_def["metric"]
            source_ms = fnum(assignment.get(metric_def["target_field"]))
            raw_rows: list[tuple[str, dict[str, str], float, str, float, float, float]] = []
            for process in rep.process_set:
                rep_process_row = rep.rows_by_process[process]
                rep_process_ms = fnum(rep_process_row.get(metric_def["representative_field"]))
                formula, target_c = target_complexity(assignment, process)
                _, rep_c = target_complexity(
                    {
                        "q_len": rep.q_len,
                        "kv_len": rep.kv_len,
                    },
                    process,
                )
                ratio = target_c / rep_c if rep_c > 0 else 0.0
                raw = rep_process_ms * ratio
                raw_rows.append((process, rep_process_row, rep_process_ms, formula, target_c, rep_c, raw))

            raw_total = sum(item[-1] for item in raw_rows)
            if raw_total <= 0:
                raw_rows = []

            for process, rep_process_row, rep_process_ms, formula, target_c, rep_c, raw in raw_rows:
                ms = source_ms * raw / raw_total if raw_total > 0 else 0.0
                method = (
                    "observed_process_nvtx_cupti"
                    if assignment["attribution_source"] == "observed_fx_op"
                    else "complexity_scaled_layer_conserved"
                )
                row = {
                    "variant": VARIANT,
                    "phase": assignment["phase"],
                    "layer": assignment["layer"],
                    "occurrence": assignment["occurrence"],
                    "attribution_type_id": assignment["attribution_type_id"],
                    "attribution_source": assignment["attribution_source"],
                    "representative_layer_id": rep.representative_layer_id,
                    "process": process,
                    "cost_type": PROCESS_COST_TYPE.get(process, "other"),
                    "metric": metric,
                    "ms": fmt(ms),
                    "pct_of_layer_metric": pct(ms, source_ms),
                    "source_layer_metric_ms": fmt(source_ms),
                    "representative_process_ms": fmt(rep_process_ms),
                    "complexity_formula": formula,
                    "target_complexity": fmt(target_c),
                    "representative_complexity": fmt(rep_c),
                    "complexity_ratio": fmt(target_c / rep_c if rep_c > 0 else 0.0),
                    "raw_weight_ms": fmt(raw),
                    "normalization_scope": "layer_total_metric",
                    "process_title": rep_process_row.get("process_title", ""),
                    "fx_nodes": rep_process_row.get("fx_nodes", ""),
                    "fx_op_families": rep_process_row.get("fx_op_families", ""),
                    "matched_sample_families": target_families or rep_process_row.get("matched_kernel_families", ""),
                    "representative_matched_kernel_families": rep_process_row.get("matched_kernel_families", ""),
                    "representative_dominant_kernel_family": rep_process_row.get("dominant kernel family", ""),
                    "representative_kernel_family_ms": rep_process_row.get("kernel_family_ms", ""),
                    "representative_runtime_api_calls": rep_process_row.get("runtime API calls", ""),
                    "representative_kernel_instances": rep_process_row.get("kernel instances", ""),
                    "representative_validation_status": rep_process_row.get("validation status", ""),
                    "attribution_method": method,
                    "fallback_reason": "",
                    "conservation_error_ms": "0.000000",
                    "expected_kernel_families": rep_process_row.get("expected_kernel_families", ""),
                    "template_validation_status": assignment["template_validation_status"],
                }
                by_group_indices[
                    (VARIANT, assignment["phase"], assignment["layer"], assignment["occurrence"], metric)
                ].append(len(output))
                output.append(row)

    for key, indices in by_group_indices.items():
        total = sum(fnum(output[i]["ms"]) for i in indices)
        source = fnum(output[indices[0]]["source_layer_metric_ms"]) if indices else 0.0
        err = abs(total - source)
        for i in indices:
            output[i]["conservation_error_ms"] = fmt(err)
    return output


def build_assignment_rows(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in assignments:
        rows.append(
            {
                "variant": VARIANT,
                "phase": row["phase"],
                "layer": row["layer"],
                "occurrence": row["occurrence"],
                "q_len": row["q_len"],
                "kv_len": row["kv_len"],
                "workload_type": row["workload_type"],
                "target_feature_signature": row["target_feature_signature"],
                "attribution_type_id": row["attribution_type_id"],
                "attribution_source": row["attribution_source"],
                "fx_template_source_layer": row["representative_layer_id"],
                "representative_feature_signature": row["representative_feature_signature"],
                "feature_match_basis": row["feature_match_basis"],
                "feature_match_score": fmt(row["feature_match_score"]),
                "template_validation_status": row["template_validation_status"],
                "template_confidence": row["template_confidence"],
                "validation_reason": row["validation_reason"],
                "event_id": row["event_id"],
                "forward_id": row["forward_id"],
                "representative_forward_id": row["representative_forward_id"],
                "representative_occurrence": row["representative_occurrence"],
                "representative_layer": row["representative_layer"],
            }
        )
    return rows


def build_aggregation(process_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    denominators: dict[str, float] = {}
    layer_metric_seen: set[tuple[str, str, str, str, str]] = set()
    for row in process_rows:
        key = (row["variant"], row["phase"], str(row["layer"]), str(row["occurrence"]), row["metric"])
        if key in layer_metric_seen:
            continue
        layer_metric_seen.add(key)
        denominators[row["metric"]] = denominators.get(row["metric"], 0.0) + fnum(row["source_layer_metric_ms"])

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in process_rows:
        grouped[(row["variant"], row["process"], row["cost_type"], row["metric"])].append(row)

    output = []
    for (variant, process, cost_type, metric), rows in sorted(grouped.items()):
        total = sum(fnum(row["ms"]) for row in rows)
        by_source = defaultdict(float)
        covered = set()
        for row in rows:
            by_source[row["attribution_source"]] += fnum(row["ms"])
            covered.add((row["phase"], row["layer"], row["occurrence"]))
        denom = denominators.get(metric, 0.0)
        representative_layers = compact_unique([row.get("representative_layer_id") for row in rows], limit=8)
        runtime_calls = compact_numeric_values([row.get("representative_runtime_api_calls") for row in rows])
        kernel_instances = compact_numeric_values([row.get("representative_kernel_instances") for row in rows])
        launch_evidence = []
        if runtime_calls:
            launch_evidence.append(f"runtime API calls {runtime_calls}")
        if kernel_instances:
            launch_evidence.append(f"kernel instances {kernel_instances}")
        output.append(
            {
                "variant": variant,
                "process": process,
                "cost_type": cost_type,
                "metric": metric,
                "sum_ms": fmt(total),
                "global_metric_pct": pct(total, denom),
                "observed_ms": fmt(by_source["observed_fx_op"]),
                "template_scaled_ms": fmt(by_source["template_scaled"]),
                "fallback_ms": fmt(by_source["fallback_component"]),
                "unknown_ms": fmt(by_source["unknown"]),
                "observed_pct": pct(by_source["observed_fx_op"], total),
                "template_scaled_pct": pct(by_source["template_scaled"], total),
                "fallback_pct": pct(by_source["fallback_component"], total),
                "unknown_pct": pct(by_source["unknown"], total),
                "covered_layers": str(len(covered)),
                "coverage_note": "covered_layers counts input-layer rows for this process and metric",
                "process_title": compact_unique([row.get("process_title") for row in rows], limit=3),
                "representative_layers": representative_layers,
                "fx_nodes": compact_unique([row.get("fx_nodes") for row in rows], limit=4),
                "fx_op_families": compact_unique([row.get("fx_op_families") for row in rows], limit=4),
                "expected_kernel_families": compact_unique(
                    [row.get("expected_kernel_families") for row in rows], limit=4
                ),
                "matched_kernel_families": compact_unique(
                    [row.get("representative_matched_kernel_families") for row in rows], limit=4
                ),
                "dominant_kernel_families": compact_unique(
                    [row.get("representative_dominant_kernel_family") for row in rows], limit=4
                ),
                "sample_kernel_family_ms": compact_unique(
                    [row.get("representative_kernel_family_ms") for row in rows], limit=4
                ),
                "sample_launch_evidence": "; ".join(launch_evidence),
                "sample_validation_status": compact_unique(
                    [row.get("representative_validation_status") for row in rows], limit=4
                ),
            }
        )
    return output


def build_coverage(process_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assignment_by_layer_metric: dict[tuple[str, str, str, str], str] = {}
    source_by_layer_metric: dict[tuple[str, str, str, str], float] = {}
    for row in process_rows:
        key = (row["variant"], row["phase"], str(row["layer"]), str(row["occurrence"]), row["metric"])
        assignment_by_layer_metric[key] = row["attribution_source"]
        source_by_layer_metric[key] = fnum(row["source_layer_metric_ms"])

    by_metric: dict[str, list[tuple[tuple[str, str, str, str], str, float]]] = defaultdict(list)
    for key, source in source_by_layer_metric.items():
        metric = key[-1]
        layer_key = key[:4]
        by_metric[metric].append((layer_key, assignment_by_layer_metric[key], source))

    output = []
    for metric, rows in sorted(by_metric.items()):
        counts = defaultdict(int)
        sums = defaultdict(float)
        for _, source, ms in rows:
            counts[source] += 1
            sums[source] += ms
        output.append(
            {
                "variant": VARIANT,
                "metric": metric,
                "total_layers": str(len(rows)),
                "observed_fx_layers": str(counts["observed_fx_op"]),
                "template_scaled_layers": str(counts["template_scaled"]),
                "fallback_layers": str(counts["fallback_component"]),
                "unknown_layers": str(counts["unknown"]),
                "observed_ms": fmt(sums["observed_fx_op"]),
                "template_scaled_ms": fmt(sums["template_scaled"]),
                "fallback_ms": fmt(sums["fallback_component"]),
                "unknown_ms": fmt(sums["unknown"]),
                "strict_claim_allowed": "false",
                "risk_note": (
                    "Layer totals are conserved, but most input-layers are template_scaled; "
                    "this is an estimate, not a full direct process-level trace."
                ),
            }
        )
    return output


def role_summary(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        grouped[(row["phase"], row["role"], row["attribution_source"])].append(row)
    output = []
    for (phase, role, source), rows in sorted(grouped.items()):
        reps = sorted({row["representative_layer_id"] for row in rows if row.get("representative_layer_id")})
        output.append(
            {
                "phase": phase,
                "role": role,
                "source": source,
                "rows": str(len(rows)),
                "layers": compact_ranges([inum(row["layer"]) for row in rows]),
                "occurrences": compact_ranges([inum(row["occurrence"]) for row in rows]),
                "representatives": ", ".join(reps[:8]) + (" ..." if len(reps) > 8 else ""),
            }
        )
    return output


def validation_stats(assignments: list[dict[str, Any]], process_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_assignment = [row for row in assignments if not row.get("attribution_source") or not row.get("attribution_type_id")]
    failed_strict = [
        row
        for row in assignments
        if row.get("template_validation_status") == "fail"
        and row.get("attribution_source") in {"observed_fx_op", "template_scaled"}
    ]
    by_group = defaultdict(float)
    source = {}
    for row in process_rows:
        key = (row["variant"], row["phase"], row["layer"], row["occurrence"], row["metric"])
        by_group[key] += fnum(row["ms"])
        source[key] = fnum(row["source_layer_metric_ms"])
    errors = [abs(total - source[key]) for key, total in by_group.items()]
    return {
        "assignment_rows": len(assignments),
        "missing_assignment": len(missing_assignment),
        "failed_strict_rows": len(failed_strict),
        "metric_groups": len(by_group),
        "max_conservation_err": max(errors, default=0.0),
    }


def top_rows(rows: list[dict[str, Any]], metric: str, limit: int = 12) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["metric"] == metric]
    return sorted(filtered, key=lambda row: fnum(row["sum_ms"]), reverse=True)[:limit]


def write_report(
    path: Path,
    args: argparse.Namespace,
    assignments: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    aggregation: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    role_rows = role_summary(assignments)
    out: list[str] = []
    out.append("# VisiPruner Full Eager Full-Layer Process Attribution Report")
    out.append("")
    out.append("This report estimates end-to-end process-wise performance for all VisiPruner full eager input-layers using representative process traces and layer-conserved normalization.")
    out.append("")
    out.append("## Sources")
    out.append("")
    out.extend(
        [
            f"- representative process-wise report: `{args.representative_report}`",
            f"- representative process CSV: `{args.representative_process_csv}`",
            f"- full layer-wise report: `{args.layer_report}`",
            f"- full input-layer CSV: `{args.full_input_layer_csv}`",
            f"- layer kernel breakdown CSV: `{args.layer_kernel_csv}`",
            f"- attribution type map: `{args.type_map_output}`",
            f"- template assignment: `{args.assignment_output}`",
            f"- process attribution: `{args.attribution_output}`",
            f"- process aggregation: `{args.aggregation_output}`",
            f"- coverage and risk: `{args.coverage_output}`",
        ]
    )
    out.append("")
    out.append("## Method")
    out.append("")
    out.append("- Target coverage is the full layer-wise SAME_INPUT run: each `(phase, forward_id, layer, occurrence)` row has its own measured NVTX CPU range and CUPTI launch-owned kernel sum.")
    out.append("- Representative rows are strict process-level NVTX/CUPTI traces. They provide process sets, FX/op families, kernel-family validation, and metric-specific process weights.")
    out.append("- Interval assignment is inferred from feature signatures: phase, q_len, kv_len, workload type, retained-token/KV regime, layer role, dominant kernel family, and process set.")
    out.append("- For each target input-layer, representative process weights are scaled by theoretical complexity and normalized to that target layer's measured metric, so layer totals are conserved.")
    out.append("- Component-level normalization is intentionally not used here because the process definitions cross `attn`/`mlp` hook boundaries; the safer scope is the total layer metric.")
    out.append("")
    out.append("## Validation Summary")
    out.append("")
    out.extend(
        md_table(
            ["check", "value"],
            [
                ["assignment rows", stats["assignment_rows"]],
                ["missing assignment", stats["missing_assignment"]],
                ["failed strict rows", stats["failed_strict_rows"]],
                ["metric groups", stats["metric_groups"]],
                ["max conservation error ms", fmt(stats["max_conservation_err"])],
                ["interval rows", len(intervals)],
            ],
        )
    )
    out.append("")
    out.append("## Coverage And Risk")
    out.append("")
    out.extend(
        md_table(
            [
                "metric",
                "total input-layers",
                "observed",
                "template_scaled",
                "fallback",
                "unknown",
                "strict claim allowed",
            ],
            [
                [
                    row["metric"],
                    row["total_layers"],
                    row["observed_fx_layers"],
                    row["template_scaled_layers"],
                    row["fallback_layers"],
                    row["unknown_layers"],
                    row["strict_claim_allowed"],
                ]
                for row in coverage
            ],
        )
    )
    out.append("")
    out.append("## Feature-Inferred Assignment Summary")
    out.append("")
    out.extend(
        md_table(
            ["phase", "feature role", "source", "rows", "layers", "occurrences", "representatives"],
            [
                [
                    row["phase"],
                    row["role"],
                    row["source"],
                    row["rows"],
                    row["layers"],
                    row["occurrences"],
                    row["representatives"],
                ]
                for row in role_rows
            ],
        )
    )
    out.append("")
    for metric in ["CUPTI kernel ms", "NVTX CPU ms"]:
        out.append(f"## Global Process Aggregation: {metric}")
        out.append("")
        out.extend(
            md_table(
                [
                    "process",
                    "process title",
                    "cost_type",
                    "sum_ms",
                    "global_pct",
                    "observed_ms",
                    "template_scaled_ms",
                    "evidence mix",
                    "covered input-layers",
                    "representative layers",
                    "FX nodes",
                    "FX op families",
                    "expected kernels",
                    "matched sample kernels",
                    "sample launch evidence",
                    "sample validation",
                ],
                [
                    [
                        row["process"],
                        row["process_title"],
                        row["cost_type"],
                        row["sum_ms"],
                        row["global_metric_pct"],
                        row["observed_ms"],
                        row["template_scaled_ms"],
                        f"obs {row['observed_pct']} / tmpl {row['template_scaled_pct']}",
                        row["covered_layers"],
                        row["representative_layers"],
                        row["fx_nodes"],
                        row["fx_op_families"],
                        row["expected_kernel_families"],
                        row["matched_kernel_families"],
                        row["sample_launch_evidence"],
                        row["sample_validation_status"],
                    ]
                    for row in top_rows(aggregation, metric)
                ],
            )
        )
        out.append("")
    out.append("## Interpretation Boundaries")
    out.append("")
    out.append("- `observed_fx_op` rows use direct process-level NVTX/CUPTI evidence for the same input-layer, then normalize to the full layer-wise run's measured metric.")
    out.append("- `template_scaled` rows are estimates: the representative process template is complexity-scaled and then layer-conserved.")
    out.append("- Global aggregation rows include representative-layer FX op families and launch-owned kernel families from the strict process-wise CSV; these columns explain the process template, not a new full-layer direct trace.")
    out.append("- CUPTI kernel ms and NVTX CPU ms are separate metric scopes and should not be subtracted from each other.")
    out.append("- Because most decode input-layers are template-scaled, global process percentages are suitable for optimization guidance but not a replacement for tracing every process-level range.")
    out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def write_breakdown(
    path: Path,
    args: argparse.Namespace,
    assignments: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    aggregation: list[dict[str, Any]],
) -> None:
    out: list[str] = []
    out.append("# SAME_INPUT Full-Layer Process Attribution Breakdown")
    out.append("")
    out.append("This companion report records the detailed feature-driven segmentation used before process attribution.")
    out.append("")
    out.append("## Output Files")
    out.append("")
    out.extend(
        [
            f"- `{args.type_map_output}`",
            f"- `{args.assignment_output}`",
            f"- `{args.attribution_output}`",
            f"- `{args.aggregation_output}`",
            f"- `{args.coverage_output}`",
        ]
    )
    out.append("")
    out.append("## Interval Map Preview")
    out.append("")
    preview = intervals[:80]
    out.extend(
        md_table(
            [
                "type",
                "phase",
                "interval",
                "representative",
                "confidence",
                "boundary reason",
            ],
            [
                [
                    row["attribution_type_id"],
                    row["phase"],
                    row["layer_interval"],
                    row["representative_layer_id"],
                    row["confidence"],
                    row["boundary_reason"],
                ]
                for row in preview
            ],
        )
    )
    if len(intervals) > len(preview):
        out.append("")
        out.append(f"Only the first {len(preview)} of {len(intervals)} intervals are shown. Use the CSV for the complete interval map.")
    out.append("")
    out.append("## Process Totals By Metric")
    out.append("")
    for metric in ["CUPTI kernel ms", "NVTX CPU ms"]:
        rows = [row for row in aggregation if row["metric"] == metric]
        rows = sorted(rows, key=lambda row: fnum(row["sum_ms"]), reverse=True)
        out.append(f"### {metric}")
        out.append("")
        out.extend(
            md_table(
                ["process", "cost_type", "sum_ms", "global_pct", "observed_ms", "template_scaled_ms", "covered input-layers"],
                [
                    [
                        row["process"],
                        row["cost_type"],
                        row["sum_ms"],
                        row["global_metric_pct"],
                        row["observed_ms"],
                        row["template_scaled_ms"],
                        row["covered_layers"],
                    ]
                    for row in rows
                ],
            )
        )
        out.append("")
    out.append("## Assignment Source Counts")
    out.append("")
    counts = defaultdict(int)
    for row in assignments:
        counts[row["attribution_source"]] += 1
    out.extend(
        md_table(
            ["source", "input-layer rows"],
            [[source, count] for source, count in sorted(counts.items())],
        )
    )
    out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    base = Path("autoresearch/experiments/e2_single_request_latency")
    output = base / "output"
    package = output / "visipruner_full_eager_full_layer_process_attribution"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-input-layer-csv", type=Path, default=output / "visipruner_full_eager_layer_wise/nsys_fxsameinput_visipruner_full_eager_32tok_all_input_layer_performance.csv")
    parser.add_argument("--layer-kernel-csv", type=Path, default=output / "visipruner_full_eager_layer_wise/nsys_fxsameinput_visipruner_full_eager_32tok_layer_kernel_breakdown.csv")
    parser.add_argument("--representative-process-csv", type=Path, default=output / "visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv")
    parser.add_argument("--representative-report", type=Path, default=output / "visipruner_full_eager_process_wise/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md")
    parser.add_argument("--layer-report", type=Path, default=output / "visipruner_full_eager_layer_wise/SAME_INPUT_VISIPRUNER_FULL_EAGER_LAYER_PERFORMANCE_REPORT.md")
    parser.add_argument("--type-map-output", type=Path, default=package / "full_layer_attribution_type_map.csv")
    parser.add_argument("--assignment-output", type=Path, default=package / "full_layer_template_assignment.csv")
    parser.add_argument("--attribution-output", type=Path, default=package / "full_layer_process_attribution.csv")
    parser.add_argument("--aggregation-output", type=Path, default=package / "full_layer_process_aggregation.csv")
    parser.add_argument("--coverage-output", type=Path, default=package / "full_layer_coverage_and_risk.csv")
    parser.add_argument("--report-output", type=Path, default=package / "SAME_INPUT_VISIPRUNER_FULL_EAGER_FULL_LAYER_PROCESS_ATTRIBUTION_REPORT.md")
    parser.add_argument("--breakdown-output", type=Path, default=package / "SAME_INPUT_FULL_LAYER_PROCESS_ATTRIBUTION_BREAKDOWN.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_input_rows = read_csv(args.full_input_layer_csv)
    layer_breakdown_rows = read_csv(args.layer_kernel_csv)
    process_rows = read_csv(args.representative_process_csv)

    full_rows = build_full_rows(full_input_rows, layer_breakdown_rows)
    full_by_key = {event_key(row): row for row in full_rows}
    representatives = build_representatives(process_rows, full_by_key)
    assignments, intervals = build_assignments(full_rows, representatives)
    process_attribution = build_process_attribution(assignments)
    assignment_rows = build_assignment_rows(assignments)
    aggregation = build_aggregation(process_attribution)
    coverage = build_coverage(process_attribution)
    stats = validation_stats(assignments, process_attribution)

    write_csv(
        args.type_map_output,
        intervals,
        [
            "variant",
            "phase",
            "attribution_type_id",
            "layer_interval",
            "boundary_reason",
            "boundary_feature_delta",
            "representative_layer_id",
            "representative_layer_method",
            "representative_feature_signature",
            "target_feature_summary",
            "template_match_basis",
            "expected_processes",
            "expected_op_families",
            "complexity_features",
            "normalization_scope",
            "confidence",
            "fallback_policy",
        ],
    )
    write_csv(
        args.assignment_output,
        assignment_rows,
        [
            "variant",
            "phase",
            "layer",
            "occurrence",
            "q_len",
            "kv_len",
            "workload_type",
            "target_feature_signature",
            "attribution_type_id",
            "attribution_source",
            "fx_template_source_layer",
            "representative_feature_signature",
            "feature_match_basis",
            "feature_match_score",
            "template_validation_status",
            "template_confidence",
            "validation_reason",
            "event_id",
            "forward_id",
            "representative_forward_id",
            "representative_occurrence",
            "representative_layer",
        ],
    )
    write_csv(
        args.attribution_output,
        process_attribution,
        [
            "variant",
            "phase",
            "layer",
            "occurrence",
            "attribution_type_id",
            "attribution_source",
            "representative_layer_id",
            "process",
            "cost_type",
            "metric",
            "ms",
            "pct_of_layer_metric",
            "source_layer_metric_ms",
            "representative_process_ms",
            "complexity_formula",
            "target_complexity",
            "representative_complexity",
            "complexity_ratio",
            "raw_weight_ms",
            "normalization_scope",
            "process_title",
            "fx_nodes",
            "fx_op_families",
            "matched_sample_families",
            "representative_matched_kernel_families",
            "representative_dominant_kernel_family",
            "representative_kernel_family_ms",
            "representative_runtime_api_calls",
            "representative_kernel_instances",
            "representative_validation_status",
            "attribution_method",
            "fallback_reason",
            "conservation_error_ms",
            "expected_kernel_families",
            "template_validation_status",
        ],
    )
    write_csv(
        args.aggregation_output,
        aggregation,
        [
            "variant",
            "process",
            "cost_type",
            "metric",
            "sum_ms",
            "global_metric_pct",
            "observed_ms",
            "template_scaled_ms",
            "fallback_ms",
            "unknown_ms",
            "observed_pct",
            "template_scaled_pct",
            "fallback_pct",
            "unknown_pct",
            "covered_layers",
            "coverage_note",
            "process_title",
            "representative_layers",
            "fx_nodes",
            "fx_op_families",
            "expected_kernel_families",
            "matched_kernel_families",
            "dominant_kernel_families",
            "sample_kernel_family_ms",
            "sample_launch_evidence",
            "sample_validation_status",
        ],
    )
    write_csv(
        args.coverage_output,
        coverage,
        [
            "variant",
            "metric",
            "total_layers",
            "observed_fx_layers",
            "template_scaled_layers",
            "fallback_layers",
            "unknown_layers",
            "observed_ms",
            "template_scaled_ms",
            "fallback_ms",
            "unknown_ms",
            "strict_claim_allowed",
            "risk_note",
        ],
    )
    write_report(args.report_output, args, assignments, intervals, aggregation, coverage, stats)
    write_breakdown(args.breakdown_output, args, assignments, intervals, aggregation)

    print(f"assignments={len(assignments)}")
    print(f"intervals={len(intervals)}")
    print(f"process_rows={len(process_attribution)}")
    print(f"max_conservation_err={stats['max_conservation_err']:.9f}")
    print(f"report={args.report_output}")


if __name__ == "__main__":
    main()
