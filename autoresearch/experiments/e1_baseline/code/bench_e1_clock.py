#!/usr/bin/env python3
"""
E1: Clock-Based Performance Comparison — GPU-clock-synchronized wall-clock timing.

Measures end-to-end inference latency using torch.cuda.Event for precise
GPU-synchronized timing across all E1 configs. Produces comparison tables
and structured JSON/CSV output.

This is a PRACTICAL timing tool for:
  1. Quick validation before expensive nsys/ncu profiling
  2. Cross-validation against nsys GPU time measurements
  3. Understanding real user-perceived latency (includes Python overhead)

For SCIENTIFIC analysis (kernel-level breakdown, bottleneck classification),
use nsys/ncu profiling via bench_e1_baseline.py + e1_runner.py.

Usage:
    # Run all configs with default settings
    python bench_e1_clock.py

    # Run specific configs
    python bench_e1_clock.py --configs dense-fa2 visipruner-full

    # Custom iterations and token count
    python bench_e1_clock.py --iterations 20 --max-new-tokens 256

    # Save results to specific directory
    python bench_e1_clock.py --output-dir /path/to/results
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════
# Environment setup — MUST run before any PyTorch / HF import
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_GPU = 1


def _parse_gpu_early(argv):
    for i, arg in enumerate(argv):
        if arg == '--gpu' and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
        elif arg.startswith('--gpu='):
            try:
                return int(arg.split('=', 1)[1])
            except ValueError:
                pass
    return _DEFAULT_GPU


os.environ['CUDA_VISIBLE_DEVICES'] = str(_parse_gpu_early(sys.argv))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/VisPrune/models'

# ═══════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════

_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', 'repo')
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (process_images, get_model_name_from_path,
                             tokenizer_image_token)
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX

# ═══════════════════════════════════════════════════════════════════════════
# Config definitions (same as bench_e1_baseline.py)
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "dense-fa2": {
        "use_flash_attn": True,
        "pruning_config": None,
        "description": "Dense + Flash Attention 2 (fastest baseline)",
    },
    "dense-eager": {
        "use_flash_attn": False,
        "pruning_config": None,
        "description": "Dense + Eager attention (no pruning)",
    },
    "dense-sdpa": {
        "use_flash_attn": False,
        "pruning_config": None,
        "description": "Dense + SDPA (PyTorch native, optional reference)",
    },
    "visipruner-full": {
        "use_flash_attn": False,
        "pruning_config": {
            "mode": ["middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "VisiPruner full pipeline (middle+deep pruning)",
    },
    "visipruner-shallow-only": {
        "use_flash_attn": False,
        "pruning_config": {
            "mode": ["shallow"],
            "shallow_mid_layer": 6,
        },
        "description": "VisiPruner shallow-only pruning",
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_PATH = "/workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg"
DEFAULT_PROMPT = "Describe the image briefly."
DEFAULT_OUTPUT_DIR = "/workspace/VisPrune/autoresearch/profiling/results/e1_baseline"
DEFAULT_MAX_TOKENS = 128
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 2
DEFAULT_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Model & inference helpers (minimal — no dependency on bench_e1_baseline)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(config_key, model_path, model_base):
    cfg = CONFIGS[config_key]
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    use_visipruner = cfg["pruning_config"] is not None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        device_map="cuda:0",
        use_flash_attn=cfg["use_flash_attn"],
        use_visipruner=use_visipruner,
    )
    model.eval()
    return tokenizer, model, image_processor


def load_image(image_path, image_processor, model_config):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    image_tensor = process_images([image], image_processor, model_config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    return image_tensor, image_size


def run_inference(model, tokenizer, image_processor, image_tensor, image_size,
                  prompt, max_new_tokens, pruning_config):
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda',
                                   non_blocking=True),
            image_sizes=[image_size],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pruning_config=pruning_config,
            use_cache=True,
        )

    output_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return output_text


# ═══════════════════════════════════════════════════════════════════════════
# Core: clock-based timing
# ═══════════════════════════════════════════════════════════════════════════

def time_one_config(config_key, model_path, model_base, image_path,
                    prompt, max_new_tokens, n_iter, n_warm, seed):
    """Run wall-clock timing for a single config.

    Returns:
        dict with timing results, or {"error": "..."} on failure.
    """
    cfg = CONFIGS[config_key]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model
    t0 = time.time()
    tokenizer, model, image_processor = load_model(
        config_key, model_path, model_base)
    t_load = time.time() - t0

    # Load image
    image_tensor, image_size = load_image(image_path, image_processor, model.config)

    # Warmup
    for i in range(n_warm):
        _ = run_inference(
            model, tokenizer, image_processor, image_tensor, image_size,
            prompt, max_new_tokens=max_new_tokens,
            pruning_config=cfg["pruning_config"],
        )
        torch.cuda.synchronize()

    # Timed iterations
    times_ms = []
    output_token_counts = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for i in range(n_iter):
        torch.cuda.synchronize()
        start_ev.record()
        output = run_inference(
            model, tokenizer, image_processor, image_tensor, image_size,
            prompt, max_new_tokens=max_new_tokens,
            pruning_config=cfg["pruning_config"],
        )
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(start_ev.elapsed_time(end_ev))
        output_token_counts.append(
            len(tokenizer.encode(output, add_special_tokens=False))
        )

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    tps = max_new_tokens / (mean_ms / 1000) if mean_ms > 0 else 0.0
    actual_tokens_mean = (
        statistics.mean(output_token_counts) if output_token_counts else 0.0
    )
    actual_tps = (
        actual_tokens_mean / (mean_ms / 1000) if mean_ms > 0 else 0.0
    )

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()

    return {
        "config": config_key,
        "description": cfg["description"],
        "use_flash_attn": cfg["use_flash_attn"],
        "has_pruning": cfg["pruning_config"] is not None,
        "model_load_s": round(t_load, 1),
        "max_new_tokens": max_new_tokens,
        "iterations": n_iter,
        "warmup_iters": n_warm,
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "tokens_per_sec": round(tps, 2),
        "actual_output_tokens_mean": round(actual_tokens_mean, 2),
        "actual_tokens_per_sec": round(actual_tps, 2),
        "output_tokens": output_token_counts,
        "raw_times_ms": [round(t, 1) for t in times_ms],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(results, baseline_key="dense-fa2"):
    """Print formatted comparison table."""
    valid = {k: v for k, v in results.items() if "error" not in v}
    if len(valid) < 2:
        return

    baseline = valid.get(baseline_key)
    if baseline is None:
        baseline_key = list(valid.keys())[0]
        baseline = valid[baseline_key]

    baseline_ms = baseline["mean_ms"]

    print(f"\n{'═'*80}")
    print(f"  E1 Clock-Based Performance Comparison")
    print(f"  Baseline: {baseline_key} ({baseline['description']})")
    print(f"  Theoretical pruning speedup: 2.17×")
    print(f"{'═'*80}")
    header = (f"  {'Config':<28s} {'Mean':>8s}  {'Std':>8s}  "
              f"{'Median':>8s}  {'Tok/s':>8s}  {'Speedup':>8s}  "
              f"{'vs 2.17×':>10s}")
    print(header)
    print(f"  {'-'*78}")

    # Sort by mean time (fastest first)
    for k, v in sorted(valid.items(), key=lambda x: x[1]["mean_ms"]):
        speedup = baseline_ms / v["mean_ms"] if v["mean_ms"] > 0 else 0
        if speedup >= 1.0:
            vs_theory = f"{(2.17 / speedup):.2f}× slower"
        else:
            vs_theory = f"slower than baseline"
        print(f"  {k:<28s} {v['mean_ms']:7.1f}ms "
              f"{v['std_ms']:7.1f}ms {v['median_ms']:7.1f}ms "
              f"{v['tokens_per_sec']:7.1f}  {speedup:7.3f}×  "
              f"{vs_theory:>10s}",
              flush=True)
    print(f"{'═'*80}")

    # Key findings
    visi_full = valid.get("visipruner-full")
    if visi_full and baseline:
        actual_su = baseline_ms / visi_full["mean_ms"] if visi_full["mean_ms"] > 0 else 0
        print(f"  Key findings:")
        print(f"    visipruner-full vs {baseline_key}: {actual_su:.3f}× "
              f"(vs theoretical 2.17×, GAP = {2.17 - actual_su:.3f}×)")
        if actual_su < 1.0:
            print(f"    ⚠  VisiPruner is actually {1/actual_su:.1f}× SLOWER "
                  f"than Dense-FA2!")

    visi_shallow = valid.get("visipruner-shallow-only")
    if visi_shallow and baseline:
        su_shallow = baseline_ms / visi_shallow["mean_ms"] if visi_shallow["mean_ms"] > 0 else 0
        print(f"    visipruner-shallow-only vs {baseline_key}: {su_shallow:.3f}×")

    dense_eager = valid.get("dense-eager")
    if dense_eager and baseline:
        su_eager = baseline_ms / dense_eager["mean_ms"] if dense_eager["mean_ms"] > 0 else 0
        print(f"    dense-eager vs {baseline_key}: {su_eager:.3f}× "
              f"(FA2 is {1/su_eager:.1f}× faster than Eager)")

    print(f"{'═'*80}")

    # Print errors if any
    errors = {k: v for k, v in results.items() if "error" in v}
    if errors:
        print(f"\n  Configs with errors:")
        for k, v in errors.items():
            print(f"    {k}: {v['error']}")


def save_results(results, output_dir, args):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # System info
    sys_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "e1_clock_comparison",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "model_path": args.model_path,
        "image_path": args.image_path,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "iterations": args.iterations,
        "warmup": args.warmup_iters,
        "seed": args.seed,
    }

    # Compute speedups
    valid = {k: v for k, v in results.items() if "error" not in v}
    baseline_key = "dense-fa2"
    baseline = valid.get(baseline_key)
    speedups = {}
    if baseline and baseline["mean_ms"] > 0:
        for k, v in valid.items():
            if k != baseline_key:
                speedups[f"{k}_vs_{baseline_key}"] = {
                    "actual_speedup": round(
                        baseline["mean_ms"] / v["mean_ms"], 3
                    ) if v["mean_ms"] > 0 else 0,
                    "theoretical_speedup": 2.17,
                    "gap": round(
                        2.17 - baseline["mean_ms"] / v["mean_ms"], 3
                    ) if v["mean_ms"] > 0 else 2.17,
                }

    full_results = {
        **sys_info,
        "speedups": speedups,
        "configs": results,
    }

    # JSON
    json_path = os.path.join(output_dir, f"e1_clock_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Results JSON → {json_path}")

    # Stable symlink target
    stable_path = os.path.join(output_dir, "e1_clock_latest.json")
    with open(stable_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Latest JSON   → {stable_path}")

    # CSV
    csv_path = os.path.join(output_dir, f"e1_clock_{ts}.csv")
    with open(csv_path, "w") as f:
        f.write("config,description,use_flash_attn,has_pruning,"
                "mean_ms,std_ms,median_ms,min_ms,max_ms,"
                "tokens_per_sec,actual_output_tokens_mean,"
                "actual_tokens_per_sec,model_load_s\n")
        for k, v in sorted(results.items()):
            if "error" in v:
                f.write(f"{k},ERROR: {v['error']},,,,,,,,\n")
            else:
                f.write(f"{v['config']},{v['description']},"
                        f"{v['use_flash_attn']},{v['has_pruning']},"
                        f"{v['mean_ms']},{v['std_ms']},{v['median_ms']},"
                        f"{v['min_ms']},{v['max_ms']},"
                        f"{v['tokens_per_sec']},"
                        f"{v.get('actual_output_tokens_mean', 0)},"
                        f"{v.get('actual_tokens_per_sec', 0)},"
                        f"{v['model_load_s']}\n")
    print(f"  Results CSV  → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="E1: Clock-Based Performance Comparison"
    )
    p.add_argument("--configs", type=str, nargs="+",
                   default=list(CONFIGS.keys()),
                   help=f"Configs to benchmark (default: all {len(CONFIGS)})")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--image-path", type=str, default=DEFAULT_IMAGE_PATH)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"Tokens to generate per iteration (default: {DEFAULT_MAX_TOKENS})")
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                   help=f"Timed iterations per config (default: {DEFAULT_ITERATIONS})")
    p.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP,
                   help=f"Warmup iterations before timing (default: {DEFAULT_WARMUP})")
    p.add_argument("--gpu", type=int, default=_DEFAULT_GPU,
                   help=f"GPU device ID (default: {_DEFAULT_GPU})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving results to disk")
    p.add_argument("--baseline", type=str, default="dense-fa2",
                   choices=list(CONFIGS.keys()),
                   help="Config to use as speedup baseline (default: dense-fa2)")
    p.add_argument("--skip-errors", action="store_true",
                   help="Continue to next config if one fails")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Validate configs
    for c in args.configs:
        if c not in CONFIGS:
            sys.exit(f"ERROR: Unknown config '{c}'. "
                     f"Valid configs: {list(CONFIGS.keys())}")

    # Validate image
    if not os.path.exists(args.image_path):
        sys.exit(f"ERROR: Image not found: {args.image_path}")

    print(f"{'═'*80}")
    print(f"  E1: Clock-Based Performance Comparison")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Configs: {args.configs}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Iterations: {args.iterations} (+ {args.warmup_iters} warmup)")
    print(f"  Image: {os.path.basename(args.image_path)}")
    print(f"  Prompt: {args.prompt[:60]}...")
    print(f"{'═'*80}", flush=True)

    results = {}
    total_start = time.time()

    for config in args.configs:
        cfg = CONFIGS[config]
        print(f"\n{'─'*60}")
        print(f"  [{config}] {cfg['description']}")
        print(f"  Flash Attn: {cfg['use_flash_attn']} | "
              f"Pruning: {cfg['pruning_config'] is not None}")
        print(f"{'─'*60}", flush=True)

        try:
            result = time_one_config(
                config, args.model_path, args.model_base,
                args.image_path, args.prompt,
                args.max_new_tokens, args.iterations,
                args.warmup_iters, args.seed,
            )
            results[config] = result
            print(f"  ✓ {config}: {result['mean_ms']:.1f}ms ± {result['std_ms']:.1f}ms "
                  f"({result['tokens_per_sec']:.1f} requested tok/s, "
                  f"{result['actual_tokens_per_sec']:.1f} actual tok/s)",
                  flush=True)
        except Exception as e:
            print(f"  ✗ {config} FAILED: {e}", flush=True)
            if args.skip_errors:
                results[config] = {"error": str(e), "config": config}
                continue
            else:
                # Save partial results before exiting
                if not args.no_save and results:
                    save_results(results, args.output_dir, args)
                sys.exit(f"ERROR: {config} failed. "
                         f"Use --skip-errors to continue past failures.")

    total_elapsed = time.time() - total_start
    print(f"\n  Total wall time: {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)", flush=True)

    # Print comparison
    print_comparison_table(results, baseline_key=args.baseline)

    # Save
    if not args.no_save:
        save_results(results, args.output_dir, args)

    # Exit code
    errors = {k: v for k, v in results.items() if "error" in v}
    if errors:
        print(f"\n  ⚠  {len(errors)} config(s) had errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
