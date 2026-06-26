#!/usr/bin/env python3
"""Run open-tool dense baselines and compare them with algorithmic trace output."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
WORKLOAD_DIR = ROOT_DIR / "workload_analysis"
VENDOR_DIR = WORKLOAD_DIR / "vendor/python"
LLM_ANALYSIS_DIR = WORKLOAD_DIR / "external/llm-analysis"
LLM_VIEWER_DIR = WORKLOAD_DIR / "external/llm-viewer"
DEFAULT_TRACE_POINTER = WORKLOAD_DIR / "algorithmic_trace/traces/latest_algorithmic_trace_path.txt"
DEFAULT_OUTPUT_DIR = WORKLOAD_DIR / "open_tool_dense_baseline/dense_baseline"

for path in [str(VENDOR_DIR), str(LLM_VIEWER_DIR), str(LLM_ANALYSIS_DIR)]:
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def load_trace(path: Path | None) -> dict[str, Any]:
    if path is None:
        pointer = DEFAULT_TRACE_POINTER
        if not pointer.exists():
            raise FileNotFoundError(f"missing algorithmic trace pointer: {pointer}")
        path = Path(pointer.read_text(encoding="utf-8").strip())
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dense_equivalent_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    prefill_layers = [row for row in trace["layer_events"] if row.get("phase") == "prefill"]
    initial_prefill = min((int(row["q_len"]) for row in prefill_layers), default=trace["request"]["prompt_token_count_before_image_expansion"])
    generated = int(trace["request"]["output_token_count"])
    dims = trace["model_dims"]
    return {
        "language_model": {
            "name": "llava-v1.5-7b-language-backbone",
            "num_layers": dims["num_hidden_layers"],
            "n_head": dims["num_attention_heads"],
            "hidden_dim": dims["hidden_size"],
            "vocab_size": dims["vocab_size"],
            "max_seq_len": None,
            "num_key_value_heads": dims["num_key_value_heads"],
            "ffn_embed_dim": dims["intermediate_size"],
            "model_type": "llama",
            "mlp_gated_linear_units": True,
        },
        "input": {
            "batch_size": 1,
            "dense_prefill_seq_len": initial_prefill,
            "num_tokens_to_generate": generated,
        },
    }


def run_llm_analysis(dense_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from llm_analysis.analysis import infer

    model_config_path = output_dir / "llm_analysis_model_config.json"
    write_json(model_config_path, dense_cfg["language_model"])
    summary = infer(
        model_name=str(model_config_path),
        gpu_name="a6000-48gb",
        dtype_name="w16a16e16",
        log_level="ERROR",
        batch_size_per_gpu=dense_cfg["input"]["batch_size"],
        seq_len=dense_cfg["input"]["dense_prefill_seq_len"],
        num_tokens_to_generate=dense_cfg["input"]["num_tokens_to_generate"],
        use_kv_cache=True,
        flops_efficiency=1.0,
        hbm_memory_efficiency=1.0,
        output_dir=None,
    )
    write_json(output_dir / "llm_analysis_dense_summary.json", summary)
    return summary


def run_llm_viewer_dense(dense_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import configs.Llama as llama_config
    from model_analyzer import ModelAnalyzer

    params = SimpleNamespace(
        num_attention_heads=dense_cfg["language_model"]["n_head"],
        hidden_size=dense_cfg["language_model"]["hidden_dim"],
        num_key_value_heads=dense_cfg["language_model"]["num_key_value_heads"],
        num_hidden_layers=dense_cfg["language_model"]["num_layers"],
        intermediate_size=dense_cfg["language_model"]["ffn_embed_dim"],
        vocab_size=dense_cfg["language_model"]["vocab_size"],
    )
    analyzer = ModelAnalyzer.__new__(ModelAnalyzer)
    analyzer.model_id = "llava-v1.5-7b-language-backbone"
    analyzer.hardware = "nvidia_A6000_Ada"
    analyzer.config = llama_config
    analyzer.model_params = params
    result = analyzer.analyze(
        seqlen=dense_cfg["input"]["dense_prefill_seq_len"],
        batchsize=dense_cfg["input"]["batch_size"],
        w_bit=16,
        a_bit=16,
        kv_bit=16,
        use_flashattention=False,
        tp_size=1,
    )
    serializable = json.loads(json.dumps(result, default=float))
    write_json(output_dir / "llm_viewer_dense_summary.json", serializable)
    return serializable


def write_dense_comparison_csv(trace: dict[str, Any], llm_viewer: dict[str, Any], output_dir: Path) -> None:
    custom_summary = trace["summary"]
    rows = [
        {
            "source": "algorithmic_trace_custom_visipruner",
            "scope": "full_vlm_actual_model_path",
            "flops_or_ops": custom_summary["total_flops_actual_model_path"],
            "notes": "Includes CLIP vision tower, projector, dynamic VisiPrune LLM schedule, and actual lm_head path.",
        },
        {
            "source": "algorithmic_trace_custom_visipruner",
            "scope": "full_vlm_ideal_last_token_lm_head",
            "flops_or_ops": custom_summary["total_flops_with_ideal_lm_head"],
            "notes": "Same as above but counts lm_head only for last token in each forward.",
        },
        {
            "source": "llm_viewer_dense",
            "scope": "language_model_prefill_one_layer_repeated",
            "flops_or_ops": llm_viewer["total_results"]["prefill"]["OPs"],
            "notes": "Dense LLaMA language-backbone estimate; excludes CLIP/projector and dynamic pruning.",
        },
        {
            "source": "llm_viewer_dense",
            "scope": "language_model_decode_one_step",
            "flops_or_ops": llm_viewer["total_results"]["decode"]["OPs"],
            "notes": "Dense LLaMA decode step at prompt length; excludes per-token growth and VisiPrune pruning.",
        },
    ]
    path = output_dir / "dense_tool_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=None, help="Path to algorithmic_trace.json. Defaults to latest pointer.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = load_trace(Path(args.trace) if args.trace else None)
    dense_cfg = dense_equivalent_from_trace(trace)
    write_json(out_dir / "dense_equivalent_config.json", dense_cfg)

    llm_analysis_summary = run_llm_analysis(dense_cfg, out_dir)
    llm_viewer_summary = run_llm_viewer_dense(dense_cfg, out_dir)
    write_dense_comparison_csv(trace, llm_viewer_summary, out_dir)

    report = {
        "scheme": "open_tool_dense_baseline_comparison",
        "external_tools": {
            "llm-analysis": {
                "path": str(LLM_ANALYSIS_DIR),
                "role": "Dense LLaMA inference lower-bound latency/memory model.",
                "output": str(out_dir / "llm_analysis_dense_summary.json"),
            },
            "LLM-Viewer": {
                "path": str(LLM_VIEWER_DIR),
                "role": "Dense LLaMA operator/roofline-style layer analysis.",
                "output": str(out_dir / "llm_viewer_dense_summary.json"),
            },
            "calflops": {
                "path": str(WORKLOAD_DIR / "external/calculate-flops.pytorch"),
                "role": "Installed as a local optional FLOP counter for future module-level sanity checks.",
            },
        },
        "dense_equivalent_config": dense_cfg,
        "limitations_for_visipruner": [
            "Open tools model dense transformer layers with one global sequence length.",
            "They do not represent VisiPrune's data-dependent middle selection or deep visual-token exit per layer.",
            "They do not include LLaVA's CLIP vision tower and multimodal projector unless manually extended.",
            "Use algorithmic trace as the authoritative VisiPrune workload trace; use open-tool dense baseline as dense-language-backbone context.",
        ],
        "llm_analysis_summary_keys": sorted(llm_analysis_summary.keys()),
        "llm_viewer_total_results": llm_viewer_summary["total_results"],
    }
    write_json(out_dir / "open_tool_fit_report.json", report)
    print(json.dumps({
        "report": str(out_dir / "open_tool_fit_report.json"),
        "comparison_csv": str(out_dir / "dense_tool_comparison.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
