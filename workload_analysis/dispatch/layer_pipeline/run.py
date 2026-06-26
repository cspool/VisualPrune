#!/usr/bin/env python3
"""Run dispatch analysis and small tensor ONNX export for selected layers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze import build_layer_summary, infer_original_dimensions, infer_small_config, write_process_markdown
from dispatch_io import event_ids, filter_event, read_dispatch_rows, write_dispatch_rows
from flow_codegen import copy_flow_template, export_stage_onnx, write_onnx_code_review, write_process_code_outputs
from review_reconstruction import infer_dimensions_from_dispatch, infer_dispatch_features, stage_evidence


WORKLOAD_ROOT = Path(__file__).resolve().parents[2]
DISPATCH_ROOT = WORKLOAD_ROOT / "dispatch"
DEFAULT_SOURCE_CSV = DISPATCH_ROOT / "profiles/filtered_dispatch_visipruner_full_32tok/dispatch_ops.csv"
DEFAULT_OUT_DIR = DISPATCH_ROOT / "visualize"
DEFAULT_CODE_DIR = DEFAULT_OUT_DIR


def normalize_layers(layer_args: list[str], default_input: int) -> list[str]:
    normalized: list[str] = []
    for raw in layer_args:
        for item in raw.split(","):
            token = item.strip()
            if not token:
                continue
            if re.fullmatch(r"input\d+_layer\d+", token):
                normalized.append(token)
            elif re.fullmatch(r"\d+", token):
                normalized.append(f"input{default_input}_layer{token}")
            else:
                raise ValueError(
                    f"Unsupported layer selector {token!r}; use event_id like input1_layer0 or layer number like 0"
                )
    return list(dict.fromkeys(normalized))


def process_layer(
    rows: list[dict[str, str]],
    event_id: str,
    out_dir: Path,
    code_dir: Path,
    small_seq: int,
    small_hidden: int,
    small_heads: int,
    small_ffn: int,
    skip_onnx: bool,
) -> dict[str, object]:
    event_rows = filter_event(rows, event_id)
    if not event_rows:
        raise ValueError(f"No dispatch rows found for {event_id}")

    layer_dir = out_dir / event_id
    review_dir = layer_dir / "dispatch_review"
    flow_dir = code_dir / event_id / "torch_flow"
    onnx_dir = layer_dir / "onnx"
    review_dir.mkdir(parents=True, exist_ok=True)

    original = infer_original_dimensions(event_rows)
    small_config = infer_small_config(
        original=original,
        small_seq=small_seq,
        small_hidden=small_hidden,
        small_heads=small_heads,
        small_ffn=small_ffn,
    )
    summary = build_layer_summary(event_rows, event_id, small_config)
    dispatch_dims = infer_dimensions_from_dispatch(event_rows)
    dispatch_features = infer_dispatch_features(event_rows)
    summary["dispatch_features"] = dispatch_features
    core_evidence = {
        stage: stage_evidence(stage, event_rows, dispatch_features, dispatch_dims)
        for stage in dispatch_features["expected_stages"]
    }
    write_dispatch_rows(review_dir / "dispatch_ops.csv", event_rows)
    (review_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_process_markdown(review_dir / "tensor_compute_process.md", summary)

    copy_flow_template(flow_dir, event_id, small_config, dispatch_features)
    process_manifest = write_process_code_outputs(
        flow_dir=flow_dir,
        analysis_dir=review_dir,
        event_id=event_id,
        summary=summary,
        dims=dispatch_dims,
        features=dispatch_features,
        core_evidence=core_evidence,
    )
    if not skip_onnx:
        export_stage_onnx(flow_dir, onnx_dir)

    manifest_path = onnx_dir / "manifest.json"
    onnx_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    onnx_code_review = write_onnx_code_review(
        flow_dir=flow_dir,
        analysis_dir=review_dir,
        event_id=event_id,
        onnx_manifest=onnx_manifest,
        core_evidence=core_evidence,
    )
    if onnx_code_review is not None:
        process_manifest["onnx_code_review"] = onnx_code_review
        (review_dir / "process_manifest.json").write_text(
            json.dumps(process_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    layer_manifest = {
        "event_id": event_id,
        "layer_dir": str(layer_dir),
        "dispatch_review": {
            "dispatch_ops": str(review_dir / "dispatch_ops.csv"),
            "summary": str(review_dir / "summary.json"),
            "process": str(review_dir / "tensor_compute_process.md"),
            "process_manifest": str(review_dir / "process_manifest.json"),
            "process_code_index": str(review_dir / "process_code_index.md"),
            "onnx_code_index": str(review_dir / "onnx_code_index.md") if onnx_code_review is not None else None,
            "onnx_code_map": str(review_dir / "onnx_code_map.json") if onnx_code_review is not None else None,
        },
        "torch_flow": str(flow_dir),
        "process_outputs": process_manifest,
        "onnx_code_review": onnx_code_review,
        "onnx_manifest": str(manifest_path) if manifest_path.exists() else None,
        "onnx_dir": str(onnx_dir),
        "onnx_files": onnx_manifest,
    }
    (layer_dir / "layer_manifest.json").write_text(json.dumps(layer_manifest, indent=2) + "\n", encoding="utf-8")
    return layer_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--layers", nargs="+", required=True, help="Event ids or layer ids, e.g. input1_layer0 input1_layer5 or 0,5")
    parser.add_argument("--default-input", type=int, default=1, help="Input id used when --layers contains plain layer numbers.")
    parser.add_argument("--small-seq", type=int, default=16)
    parser.add_argument("--small-hidden", type=int, default=32)
    parser.add_argument("--small-heads", type=int, default=4)
    parser.add_argument("--small-ffn", type=int, default=64)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--list-events", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_dispatch_rows(args.source_csv)
    if args.list_events:
        print("\n".join(event_ids(rows)))
        return
    selected = normalize_layers(args.layers, args.default_input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.code_dir.mkdir(parents=True, exist_ok=True)
    manifests = [
        process_layer(
            rows=rows,
            event_id=event_id,
            out_dir=args.out_dir,
            code_dir=args.code_dir,
            small_seq=args.small_seq,
            small_hidden=args.small_hidden,
            small_heads=args.small_heads,
            small_ffn=args.small_ffn,
            skip_onnx=args.skip_onnx,
        )
        for event_id in selected
    ]
    top_manifest = {
        "source_csv": str(args.source_csv),
        "out_dir": str(args.out_dir),
        "code_dir": str(args.code_dir),
        "layers": selected,
        "manifests": manifests,
    }
    manifest_path = args.out_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(top_manifest, indent=2))


if __name__ == "__main__":
    main()
