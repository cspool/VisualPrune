#!/usr/bin/env python3
"""
E1: Baseline Gap — theoretical 2.17× vs actual GPU speedup.

Hardware profiling pipeline: nsys (locate hotspots) → ncu (deep analysis).
NO wall-time timing. NO CUPTI. All performance data from nsys/ncu.

Model loading follows the pattern from repo/scripts/v1_5/visiPruner_eval/
working eval scripts: device_map="cuda:0", no cache_dir, no attn_implementation
kwarg (builder sets it internally via use_flash_attn).

Usage:
    # Step 0 — smoke test (no profiler needed, validates setup)
    python bench_e1_baseline.py --config dense-fa2 --mode smoke

    # Step 1 — nsys full-timeline sampling
    nsys profile --trace cuda,nvtx,cublas --stats=true \
        --output e1_nsys_dense_fa2 \
        python bench_e1_baseline.py --config dense-fa2 --mode nsys-profile

    # Step 2 — ncu deep analysis on hotspots identified by nsys
    ncu --set full --kernel-name regex:"<pattern>" --launch-count 50 --launch-skip 5 \
        --output e1_ncu_dense_fa2_matmul \
        python bench_e1_baseline.py --config dense-fa2 --mode ncu-profile

    # List all configs
    python bench_e1_baseline.py --mode list-configs

Configs:
    dense-fa2              Dense + Flash Attention 2 (fastest baseline)
    dense-eager            Dense + Eager attention (no pruning)
    dense-sdpa             Dense + SDPA (PyTorch native reference)
    visipruner-full        VisiPruner full pipeline (middle+deep pruning)
    visipruner-shallow-only VisiPruner shallow-only pruning
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════
# Environment setup — MUST run before any PyTorch / HF import
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_GPU = 1


def _parse_gpu_early(argv):
    """Extract --gpu value from argv before full argparse parse."""
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

# Force HF to use ONLY local cache.  Model files live under HF_HOME.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/VisPrune/models'

# ═══════════════════════════════════════════════════════════════════════════
# Imports (after env vars are set)
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
# Config definitions
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
DEFAULT_NSYS_TOKENS = 128
DEFAULT_NCU_TOKENS = 32
DEFAULT_SMOKE_TOKENS = 16
DEFAULT_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="E1: Baseline Gap — nsys/ncu hardware profiling harness"
    )
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--image-path", type=str, default=DEFAULT_IMAGE_PATH)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--config", type=str, default=None,
                   choices=list(CONFIGS.keys()) + ["all"],
                   help="Benchmark config (use 'all' for all configs)")
    p.add_argument("--mode", type=str, required=True,
                   choices=["nsys-profile", "ncu-profile", "wall-clock", "smoke", "list-configs"])
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override default token count")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpu", type=int, default=_DEFAULT_GPU,
                   help=f"GPU device ID (default: {_DEFAULT_GPU})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip warmup run")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing system_info.json")
    p.add_argument("--no-nvtx", action="store_true",
                   help="Disable NVTX range markers")
    p.add_argument("--iterations", type=int, default=10,
                   help="Number of inference iterations for wall-clock mode (default: 10)")
    p.add_argument("--warmup-iters", type=int, default=2,
                   help="Number of warmup iterations before timing (default: 2)")
    p.add_argument("--compare", action="store_true",
                   help="Run wall-clock mode across ALL configs and produce comparison")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_image_exists(image_path):
    if not os.path.exists(image_path):
        return False, f"Image not found: {image_path}"
    if not os.path.isfile(image_path):
        return False, f"Path is not a file: {image_path}"
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        return False, f"Cannot open/verify image: {image_path} — {e}"
    return True, None


# ═══════════════════════════════════════════════════════════════════════════
# Model loading (follows eval script pattern: device_map="cuda:0")
# ═══════════════════════════════════════════════════════════════════════════

def load_model(config_key, model_path, model_base):
    """Load LLaVA model following the working eval script pattern.

    Key differences from old broken approach:
    - device_map="cuda:0" (not "auto") — matches visiPruner_eval scripts
    - No cache_dir kwarg — HF_HOME env var controls cache location
    - use_flash_attn parameter — builder sets attn_implementation internally
    - use_visipruner parameter — builder injects into config so
      LlamaDecoderLayer selects the correct attention class
    """
    cfg = CONFIGS[config_key]
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)

    use_visipruner = cfg["pruning_config"] is not None

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map="cuda:0",
        use_flash_attn=cfg["use_flash_attn"],
        use_visipruner=use_visipruner,
    )
    model.eval()
    return tokenizer, model, image_processor


# ═══════════════════════════════════════════════════════════════════════════
# Image loading (follows eval script pattern)
# ═══════════════════════════════════════════════════════════════════════════

def load_image(image_path, image_processor, model_config):
    """Load and process a single image. Follows model_vqa_loader.py pattern."""
    ok, err = _check_image_exists(image_path)
    if not ok:
        sys.exit(f"ERROR: {err}")

    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    image_tensor = process_images([image], image_processor, model_config)
    # process_images returns a list; take first element
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    return image_tensor, image_size


# ═══════════════════════════════════════════════════════════════════════════
# Inference (follows model_vqa_loader.py generate() call pattern)
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(model, tokenizer, image_processor, image_tensor, image_size,
                  prompt, max_new_tokens, pruning_config, use_nvtx):
    """Run end-to-end inference. NO wall-time measurement.

    Matches the model.generate() call pattern from model_vqa_loader.py.
    """
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    if use_nvtx:
        torch.cuda.nvtx.range_push("e1_inference")

    try:
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
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    output_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return output_text


def run_warmup(model, tokenizer, image_processor, image_tensor, image_size,
               prompt, pruning_config):
    """Single short warmup forward pass."""
    _ = run_inference(
        model, tokenizer, image_processor, image_tensor, image_size,
        prompt, max_new_tokens=4,
        pruning_config=pruning_config, use_nvtx=False,
    )
    torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════
# Mode handlers
# ═══════════════════════════════════════════════════════════════════════════

def run_nsys_profile(args):
    """Step 1: nsys full-timeline sampling. Single forward pass + NVTX."""
    cfg = CONFIGS[args.config]
    max_tokens = (args.max_new_tokens if args.max_new_tokens is not None
                  else DEFAULT_NSYS_TOKENS)

    tokenizer, model, image_processor = load_model(
        args.config, args.model_path, args.model_base)
    image_tensor, image_size = load_image(
        args.image_path, image_processor, model.config)

    use_nvtx = not args.no_nvtx
    if use_nvtx:
        torch.cuda.nvtx.range_push("e1_inference")

    output = run_inference(
        model, tokenizer, image_processor, image_tensor, image_size,
        args.prompt, max_new_tokens=max_tokens,
        pruning_config=cfg["pruning_config"],
        use_nvtx=False,  # already pushed at this level
    )

    if use_nvtx:
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    n_output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
    print(f"[E1 nsys-profile] config={args.config} tokens={max_tokens} done. "
          f"output_tokens={n_output_tokens} output={output[:60]}...",
          flush=True)

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()


def run_ncu_profile(args):
    """Step 2: ncu deep kernel analysis. Minimal tokens, optional warmup."""
    cfg = CONFIGS[args.config]
    max_tokens = (args.max_new_tokens if args.max_new_tokens is not None
                  else DEFAULT_NCU_TOKENS)

    tokenizer, model, image_processor = load_model(
        args.config, args.model_path, args.model_base)
    image_tensor, image_size = load_image(
        args.image_path, image_processor, model.config)

    if not args.no_warmup:
        run_warmup(model, tokenizer, image_processor, image_tensor, image_size,
                   args.prompt, cfg["pruning_config"])
        print("[E1 ncu-profile] warmup complete, starting profiled run...",
              flush=True)

    output = run_inference(
        model, tokenizer, image_processor, image_tensor, image_size,
        args.prompt, max_new_tokens=max_tokens,
        pruning_config=cfg["pruning_config"],
        use_nvtx=False,
    )
    torch.cuda.synchronize()
    n_output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
    print(f"[E1 ncu-profile] config={args.config} tokens={max_tokens} done. "
          f"output_tokens={n_output_tokens} output={output[:60]}...",
          flush=True)

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()


def run_smoke_test(args):
    """Smoke test: verify model loads and runs inference end-to-end."""
    cfg = CONFIGS[args.config]
    max_tokens = (args.max_new_tokens if args.max_new_tokens is not None
                  else DEFAULT_SMOKE_TOKENS)

    print("━━━ E1 Smoke Test ━━━", flush=True)
    print(f"  Config:       {args.config}", flush=True)
    print(f"  Description:  {cfg['description']}", flush=True)
    print(f"  Flash Attn:   {cfg['use_flash_attn']}", flush=True)
    print(f"  Pruning:      {cfg['pruning_config'] is not None}", flush=True)
    print(f"  Model:        {args.model_path}", flush=True)
    print(f"  GPU:          {args.gpu}", flush=True)
    print(f"  Max tokens:   {max_tokens}", flush=True)
    print(f"  Image:        {os.path.basename(args.image_path)}", flush=True)
    print(flush=True)

    # Pre-flight
    ok, err = _check_image_exists(args.image_path)
    if not ok:
        sys.exit(f"ERROR: {err}")
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU detected: {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print(flush=True)

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    tokenizer, model, image_processor = load_model(
        args.config, args.model_path, args.model_base)
    t_load = time.time() - t0
    print(f"  Model loaded in {t_load:.1f}s", flush=True)

    # Load image
    print("Loading image...", flush=True)
    image_tensor, image_size = load_image(
        args.image_path, image_processor, model.config)
    print(f"  Image size: {image_size[0]}×{image_size[1]}", flush=True)

    # Warmup
    if not args.no_warmup:
        print("Warmup run (4 tokens)...", flush=True)
        t0 = time.time()
        run_warmup(model, tokenizer, image_processor, image_tensor, image_size,
                   args.prompt, cfg["pruning_config"])
        t_warm = time.time() - t0
        print(f"  Warmup complete in {t_warm:.1f}s", flush=True)

    # Timed inference (diagnostic only — not a performance metric)
    print(f"Running inference (max_new_tokens={max_tokens})...", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    output = run_inference(
        model, tokenizer, image_processor, image_tensor, image_size,
        args.prompt, max_new_tokens=max_tokens,
        pruning_config=cfg["pruning_config"],
        use_nvtx=False,
    )
    torch.cuda.synchronize()
    t_infer = time.time() - t0

    n_output_tokens = len(tokenizer.encode(output))
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    print(flush=True)
    print("━━━ Results ━━━", flush=True)
    print(f"  Wall time:       {t_infer:.3f}s  (diagnostic only)", flush=True)
    print(f"  Output tokens:   {n_output_tokens}", flush=True)
    if t_infer > 0:
        print(f"  Tokens/sec:      {n_output_tokens / t_infer:.1f}", flush=True)
    print(f"  Peak GPU mem:    {peak_mem:.2f} GB", flush=True)
    print(f"  Output:          {output[:150]}{'...' if len(output) > 150 else ''}",
          flush=True)
    print(flush=True)
    print(f"[E1 smoke] config={args.config} PASSED ✓", flush=True)

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()


def run_wall_clock(_args):
    """Wall-clock timing: measure end-to-end inference latency.

    Runs N iterations with warmup, reports mean ± std.
    Uses torch.cuda.Event for precise GPU-synchronized timing.
    This is NOT nsys/ncu — it's standard Python-accessible wall-clock.

    If --compare is set or --config "all", runs ALL configs sequentially
    and produces a comparison table + JSON result file.
    """
    import statistics

    # Determine which configs to run
    if _args.config == "all" or getattr(_args, 'compare', False):
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [_args.config]

    all_results = {}

    for config_key in configs_to_run:
        cfg = CONFIGS[config_key]
        max_tokens = (_args.max_new_tokens if _args.max_new_tokens is not None
                      else DEFAULT_NSYS_TOKENS)
        n_iter = _args.iterations
        n_warm = _args.warmup_iters

        print(f"\n{'━'*60}", flush=True)
        print(f"[E1 wall-clock] config={config_key} "
              f"({cfg['description']})", flush=True)
        print(f"[E1 wall-clock] Loading model...", flush=True)

        try:
            tokenizer, model, image_processor = load_model(
                config_key, _args.model_path, _args.model_base)
        except Exception as e:
            print(f"[E1 wall-clock] ERROR loading model for {config_key}: {e}",
                  flush=True)
            all_results[config_key] = {"error": f"model_load_failed: {e}"}
            continue

        try:
            image_tensor, image_size = load_image(
                _args.image_path, image_processor, model.config)
        except Exception as e:
            print(f"[E1 wall-clock] ERROR loading image for {config_key}: {e}",
                  flush=True)
            all_results[config_key] = {"error": f"image_load_failed: {e}"}
            del model, tokenizer, image_processor
            torch.cuda.empty_cache()
            continue

        # Warmup
        print(f"[E1 wall-clock] Warmup ({n_warm} iters)...", flush=True)
        for i in range(n_warm):
            try:
                _ = run_inference(
                    model, tokenizer, image_processor, image_tensor, image_size,
                    _args.prompt, max_new_tokens=max_tokens,
                    pruning_config=cfg["pruning_config"], use_nvtx=False,
                )
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[E1 wall-clock] ERROR during warmup iter {i} "
                      f"for {config_key}: {e}", flush=True)
                all_results[config_key] = {"error": f"warmup_failed: {e}"}
                del model, tokenizer, image_processor
                torch.cuda.empty_cache()
                break
        else:
            # Timed iterations using CUDA events
            times_ms = []
            output_token_counts = []
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

            print(f"[E1 wall-clock] Timed iterations ({n_iter} × "
                  f"max_new_tokens={max_tokens})...", flush=True)
            for i in range(n_iter):
                try:
                    torch.cuda.synchronize()
                    start_ev.record()
                    output = run_inference(
                        model, tokenizer, image_processor,
                        image_tensor, image_size,
                        _args.prompt, max_new_tokens=max_tokens,
                        pruning_config=cfg["pruning_config"], use_nvtx=False,
                    )
                    end_ev.record()
                    torch.cuda.synchronize()
                    elapsed = start_ev.elapsed_time(end_ev)
                    times_ms.append(elapsed)
                    output_token_counts.append(
                        len(tokenizer.encode(output, add_special_tokens=False))
                    )
                    print(f"  iter {i+1}/{n_iter}: {elapsed:.1f} ms", flush=True)
                except Exception as e:
                    print(f"[E1 wall-clock] ERROR during timed iter {i} "
                          f"for {config_key}: {e}", flush=True)
                    break

            mean_ms = statistics.mean(times_ms) if times_ms else 0.0
            std_ms = (statistics.stdev(times_ms)
                      if len(times_ms) > 1 else 0.0)

            # Tokens-per-second
            tps = (max_tokens / (mean_ms / 1000)) if mean_ms > 0 else 0.0
            actual_tokens_mean = (
                statistics.mean(output_token_counts)
                if output_token_counts else 0.0
            )
            actual_tps = (
                actual_tokens_mean / (mean_ms / 1000)
                if mean_ms > 0 else 0.0
            )

            all_results[config_key] = {
                "config": config_key,
                "description": cfg["description"],
                "use_flash_attn": cfg["use_flash_attn"],
                "has_pruning": cfg["pruning_config"] is not None,
                "max_new_tokens": max_tokens,
                "iterations": n_iter,
                "warmup_iters": n_warm,
                "mean_ms": round(mean_ms, 2),
                "std_ms": round(std_ms, 2),
                "min_ms": round(min(times_ms), 2) if times_ms else 0,
                "max_ms": round(max(times_ms), 2) if times_ms else 0,
                "tokens_per_sec": round(tps, 2),
                "actual_output_tokens_mean": round(actual_tokens_mean, 2),
                "actual_tokens_per_sec": round(actual_tps, 2),
                "output_tokens": output_token_counts,
                "raw_times_ms": [round(t, 1) for t in times_ms],
            }

            print(f"\n  Results for {config_key}:", flush=True)
            print(f"    Mean:  {mean_ms:.1f} ms", flush=True)
            print(f"    Std:   {std_ms:.1f} ms", flush=True)
            if times_ms:
                print(f"    Min:   {min(times_ms):.1f} ms", flush=True)
                print(f"    Max:   {max(times_ms):.1f} ms", flush=True)
            print(f"    Speed: {tps:.1f} tok/s (requested token budget)", flush=True)
            print(f"    Actual speed: {actual_tps:.1f} tok/s "
                  f"(mean output tokens={actual_tokens_mean:.1f})",
                  flush=True)
            print(f"    Raw:   {[round(t, 1) for t in times_ms]}", flush=True)
            print(f"    Output tokens: {output_token_counts}", flush=True)

            del model, tokenizer, image_processor
            torch.cuda.empty_cache()

    # ── Comparison table (multi-config only) ───────────────────────────
    if len(all_results) >= 2 and any("error" not in v for v in all_results.values()):
        valid = {k: v for k, v in all_results.items() if "error" not in v}
        if len(valid) >= 2:
            baseline_key = "dense-fa2"
            baseline = valid.get(baseline_key)
            if baseline is None:
                baseline_key = list(valid.keys())[0]
                baseline = valid[baseline_key]

            print(f"\n{'═'*72}")
            print(f"  E1 Wall-Clock Comparison (baseline={baseline_key})")
            print(f"{'═'*72}")
            header = (f"  {'Config':<28s} {'Mean':>8s} {'Std':>8s} "
                      f"{'Tok/s':>8s} {'Speedup':>8s} {'vs Theory':>10s}")
            print(header)
            print(f"  {'-'*68}")

            baseline_ms = baseline["mean_ms"]
            for k, v in sorted(valid.items(),
                               key=lambda x: x[1].get("mean_ms", 999999)):
                speedup = baseline_ms / v["mean_ms"] if v["mean_ms"] > 0 else 0
                vs_theory = (f"{2.17 / speedup:.2f}× slower"
                             if speedup > 0 else "N/A")
                print(f"  {k:<28s} {v['mean_ms']:7.1f}ms "
                      f"{v['std_ms']:7.1f}ms {v['tokens_per_sec']:7.1f} "
                      f"{speedup:7.3f}× {vs_theory:>10s}",
                      flush=True)
            print(f"{'═'*72}")

            # Speedup summary
            visi_full = valid.get("visipruner-full")
            if visi_full and baseline:
                actual_su = baseline_ms / visi_full["mean_ms"] if visi_full["mean_ms"] > 0 else 0
                print(f"  Actual speedup (visipruner-full vs {baseline_key}): "
                      f"{actual_su:.3f}×")
                print(f"  Theoretical speedup:                              2.17×")
                print(f"  GAP: {2.17 - actual_su:.3f}×")
            print(f"{'═'*72}")

    # ── Save results JSON ─────────────────────────────────────────────
    if not _args.no_save and all_results:
        os.makedirs(_args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        wc_path = os.path.join(_args.output_dir, f"e1_wall_clock_{ts}.json")
        with open(wc_path, "w") as f:
            json.dump({
                "experiment": "e1_wall_clock",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configs": all_results,
            }, f, indent=2)
        print(f"\n[E1 wall-clock] Results saved to {wc_path}", flush=True)

    # Also save a stable symlink-friendly copy
    if not _args.no_save and all_results:
        stable_path = os.path.join(_args.output_dir, "e1_wall_clock_latest.json")
        with open(stable_path, "w") as f:
            json.dump({
                "experiment": "e1_wall_clock",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configs": all_results,
            }, f, indent=2)
        print(f"[E1 wall-clock] Latest results → {stable_path}", flush=True)


def run_list_configs(_args):
    """List all available benchmark configurations."""
    print("━━━ E1 Baseline Configurations ━━━")
    print()
    for key, cfg in CONFIGS.items():
        has_pruning = cfg["pruning_config"] is not None
        pruning_info = ""
        if has_pruning:
            modes = cfg["pruning_config"].get("mode", [])
            pruning_info = f" | pruning modes={modes}"
        fa_str = "flash_attn" if cfg["use_flash_attn"] else "eager"
        print(f"  {key:30s}  attn={fa_str:12s}  "
              f"{'PRUNING' if has_pruning else 'no-pruning':12s}{pruning_info}")
        print(f"  {'':30s}  {cfg['description']}")
        print()
    print(f"Total: {len(CONFIGS)} configs")
    print()
    script = os.path.basename(__file__)
    print(f"  # Smoke test")
    print(f"  python {script} --config dense-fa2 --mode smoke")
    print(f"  # nsys profiling")
    print(f"  nsys profile --trace cuda,nvtx,cublas --stats=true \\")
    print(f"      --output e1_nsys_dense_fa2 \\")
    print(f"      python {script} --config dense-fa2 --mode nsys-profile")


# ═══════════════════════════════════════════════════════════════════════════
# System info
# ═══════════════════════════════════════════════════════════════════════════

def collect_system_info(args):
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = round(
        torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)

    cfg_info = {}
    if args.config and args.config != "all":
        cfg_info = {
            "use_flash_attn": CONFIGS[args.config]["use_flash_attn"],
            "has_pruning": CONFIGS[args.config]["pruning_config"] is not None,
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_id": args.gpu,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem,
        "config": args.config,
        "mode": args.mode,
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "script": os.path.basename(__file__),
        **cfg_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.mode == "list-configs":
        run_list_configs(args)
        return

    # --config "all" only valid for wall-clock mode
    if args.config == "all":
        if args.mode not in ("wall-clock", "smoke"):
            sys.exit(
                "ERROR: --config all is only supported for --mode wall-clock "
                "or --mode smoke."
            )
    elif args.config is None:
        sys.exit(
            f"ERROR: --config is required for --mode {args.mode}. "
            f"Use --mode list-configs to see available configs."
        )

    _set_seed(args.seed)

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Collect and save system info
    sys_info = collect_system_info(args)
    if not args.no_save:
        info_path = os.path.join(args.output_dir, "system_info.json")
        if os.path.exists(info_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            info_path = os.path.join(args.output_dir, f"system_info_{ts}.json")
        with open(info_path, "w") as f:
            json.dump(sys_info, f, indent=2)
        print(f"[E1] System info saved to {info_path}", flush=True)

    # Dispatch
    if args.mode == "nsys-profile":
        run_nsys_profile(args)
    elif args.mode == "ncu-profile":
        run_ncu_profile(args)
    elif args.mode == "wall-clock":
        run_wall_clock(args)
    elif args.mode == "smoke":
        if args.config == "all":
            # Run smoke test for all configs sequentially
            results = {}
            for c in CONFIGS:
                print(f"\n{'═'*60}")
                print(f"  Smoke testing: {c}")
                print(f"{'═'*60}", flush=True)
                try:
                    args_c = argparse.Namespace(**vars(args))
                    args_c.config = c
                    run_smoke_test(args_c)
                    results[c] = "PASSED"
                except SystemExit:
                    results[c] = "FAILED (exit)"
                except Exception as e:
                    results[c] = f"FAILED: {e}"
                    print(f"  [E1 smoke] {c} FAILED: {e}", flush=True)
                torch.cuda.empty_cache()
            print(f"\n{'═'*60}")
            print(f"  Smoke Test Summary")
            print(f"{'═'*60}")
            for c, status in results.items():
                print(f"  {c:<30s} {status}")
            failed = [c for c, s in results.items() if "FAILED" in str(s)]
            if failed:
                sys.exit(f"ERROR: {len(failed)} config(s) failed: {failed}")
        else:
            run_smoke_test(args)
    else:
        sys.exit(f"ERROR: Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()
