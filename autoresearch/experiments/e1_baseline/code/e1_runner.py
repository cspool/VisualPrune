#!/usr/bin/env python3
"""
E1: Experiment Runner — orchestrates the full nsys → ncu pipeline.

Usage:
    # Step 1 only: nsys profiling for all configs
    python e1_runner.py --step 1

    # Step 2 only: ncu deep analysis on hotspots (requires Step 1 results)
    python e1_runner.py --step 2

    # Step 3 only: summarize and generate deliverables
    python e1_runner.py --step 3

    # Full pipeline (all steps)
    python e1_runner.py --step all

    # Dry-run: print commands without executing
    python e1_runner.py --step 1 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_E1_DIR = os.path.dirname(_CODE_DIR)
_PROFILING_DIR = "/workspace/VisPrune/autoresearch/profiling"
_RESULTS_DIR = os.path.join(_PROFILING_DIR, "results", "e1_baseline")
_OUTPUT_DIR = os.path.join(_E1_DIR, "output")

# ── nsys/ncu PATH (must be in PATH for subprocess calls) ──────────────
_NSYS_SYSTEMS_DIR = "/opt/nvidia/nsight-systems/2026.3.1/bin"
_NCU_DIR = "/usr/local/cuda-13.2/bin"
_EXTRA_PATH = f"{_NSYS_SYSTEMS_DIR}:{_NCU_DIR}"
if _EXTRA_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_EXTRA_PATH}:{os.environ.get('PATH', '/usr/bin')}"

# Import our analysis module
sys.path.insert(0, _CODE_DIR)
import e1_analysis as ana

# ── Defaults ────────────────────────────────────────────────────────────
BENCH_SCRIPT = os.path.join(_CODE_DIR, "bench_e1_baseline.py")
ALL_CONFIGS = ["dense-fa2", "dense-eager", "visipruner-full", "visipruner-shallow-only"]
DEFAULT_GPU = 1
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_CACHE_DIR = "/workspace/VisPrune/models"
DEFAULT_IMAGE_PATH = "/workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg"
DEFAULT_NSYS_TOKENS = 128
DEFAULT_NCU_TOKENS = 32
DEFAULT_NCU_LAUNCH_COUNT = 50
DEFAULT_NCU_LAUNCH_SKIP = 5


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="E1: Baseline Gap — experiment runner (nsys → ncu pipeline)"
    )
    p.add_argument("--step", type=str, required=True,
                   choices=["0", "1", "2", "3", "wc", "all"],
                   help="Which step to run (0/wc=wall-clock, 1=nsys, 2=ncu, "
                        "3=summarize, all=full pipeline)")
    p.add_argument("--configs", type=str, nargs="+", default=ALL_CONFIGS,
                   help=f"Configs to profile (default: {' '.join(ALL_CONFIGS)})")
    p.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                   help=f"GPU device ID (default: {DEFAULT_GPU})")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                   help=f"HF cache directory (default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--image-path", type=str, default=DEFAULT_IMAGE_PATH)
    p.add_argument("--output-dir", type=str, default=_OUTPUT_DIR)
    p.add_argument("--nsys-tokens", type=int, default=DEFAULT_NSYS_TOKENS)
    p.add_argument("--ncu-tokens", type=int, default=DEFAULT_NCU_TOKENS)
    p.add_argument("--ncu-launch-count", type=int, default=DEFAULT_NCU_LAUNCH_COUNT)
    p.add_argument("--ncu-launch-skip", type=int, default=DEFAULT_NCU_LAUNCH_SKIP)
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands but don't execute")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--no-ncu-warmup", action="store_true",
                   help="Skip warmup before ncu profiling")
    p.add_argument("--hotspot-min-pct", type=float, default=3.0,
                   help="Minimum GPU time %% for hotspot selection (default: 3.0)")
    p.add_argument("--hotspot-max-kernels", type=int, default=5,
                   help="Max hotspot kernels per config for ncu (default: 5)")
    # Wall-clock step options
    p.add_argument("--wc-iterations", type=int, default=10,
                   help="Wall-clock timed iterations per config (default: 10)")
    p.add_argument("--wc-warmup", type=int, default=2,
                   help="Wall-clock warmup iterations (default: 2)")
    p.add_argument("--wc-tokens", type=int, default=128,
                   help="Wall-clock max_new_tokens (default: 128)")
    return p.parse_args()


# ── Shell command helpers ───────────────────────────────────────────────

def run_cmd(cmd: list[str], description: str, dry_run: bool = False,
            timeout: int = 3600, env: dict | None = None) -> tuple[int, str, str]:
    """Run a shell command. Returns (returncode, stdout, stderr)."""
    print(f"\n{'─'*72}")
    print(f"[e1_runner] {description}")
    print(f"[e1_runner] CMD: {' '.join(cmd)}")
    print(f"{'─'*72}", flush=True)

    if dry_run:
        print("[e1_runner] DRY-RUN — skipping execution", flush=True)
        return 0, "", ""

    try:
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
        )
        if result.stdout:
            # Print last few lines of stdout for progress
            stdout_lines = result.stdout.strip().split("\n")
            for line in stdout_lines[-10:]:
                print(f"  stdout: {line}", flush=True)
        if result.returncode != 0:
            print(f"[e1_runner] FAILED (exit={result.returncode})", flush=True)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"  stderr: {line}", flush=True)
        else:
            print(f"[e1_runner] ✓ Done", flush=True)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[e1_runner] TIMEOUT after {timeout}s", flush=True)
        return -1, "", f"Timeout after {timeout}s"
    except FileNotFoundError as e:
        print(f"[e1_runner] COMMAND NOT FOUND: {e}", flush=True)
        return -2, "", str(e)


# ── Step 0: Wall-clock timing ───────────────────────────────────────────

def step0_wall_clock(args) -> dict:
    """Run GPU-clock-synchronized wall-clock timing for all configs.

    Uses torch.cuda.Event for precise end-to-end latency measurement.
    This is a practical supplement to nsys/ncu profiling.

    Returns:
        dict mapping config_name → timing results
    """
    import statistics

    os.makedirs(args.output_dir, exist_ok=True)

    # We import bench_e1_baseline's timing functions, but use our own
    # subprocess-free approach for direct model access.
    bench_script = os.path.join(_CODE_DIR, "bench_e1_baseline.py")

    all_results = {}
    for config in args.configs:
        print(f"\n{'─'*60}")
        print(f"[e1_runner] Step 0 wall-clock: {config}")
        print(f"{'─'*60}", flush=True)

        # Use bench_e1_baseline.py in wall-clock mode for each config
        cmd = [
            "python", bench_script,
            "--config", config,
            "--mode", "wall-clock",
            "--max-new-tokens", str(args.wc_tokens),
            "--iterations", str(args.wc_iterations),
            "--warmup-iters", str(args.wc_warmup),
            "--gpu", str(args.gpu),
            "--image-path", args.image_path,
            "--model-path", args.model_path,
            "--output-dir", args.output_dir,
            "--no-save",
        ]
        rc, stdout, stderr = run_cmd(
            cmd, f"Step 0 wall-clock: {config}", args.dry_run, timeout=3600)

        if rc == 0 and not args.dry_run:
            # Parse timing from stdout: "Mean:  XXX ms"
            mean_ms = 0.0
            std_ms = 0.0
            tps = 0.0
            raw_times = []
            output_tokens = []
            for line in stdout.split("\n"):
                line = line.strip()
                if line.startswith("Mean:"):
                    try:
                        mean_ms = float(line.split()[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Std:"):
                    try:
                        std_ms = float(line.split()[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Speed:"):
                    try:
                        tps = float(line.split()[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Raw:") and "[" in line:
                    # Parse [t1, t2, ...]
                    try:
                        arr_str = line[line.index("["):line.index("]")+1]
                        raw_times = [float(x.strip()) for x in
                                     arr_str.strip("[]").split(",")
                                     if x.strip()]
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("iter ") and ":" in line and line.endswith(" ms"):
                    try:
                        raw_times.append(float(line.rsplit(" ", 1)[0].split(":")[-1]))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Output tokens:") and "[" in line:
                    try:
                        arr_str = line[line.index("["):line.index("]")+1]
                        output_tokens = [int(float(x.strip())) for x in
                                         arr_str.strip("[]").split(",")
                                         if x.strip()]
                    except (ValueError, IndexError):
                        pass

            if mean_ms <= 0 and raw_times:
                mean_ms = statistics.mean(raw_times)
            if std_ms <= 0 and len(raw_times) > 1:
                std_ms = statistics.stdev(raw_times)
            if tps <= 0 and mean_ms > 0:
                tps = args.wc_tokens / (mean_ms / 1000)
            actual_output_tokens_mean = (
                statistics.mean(output_tokens) if output_tokens else 0.0
            )
            actual_tps = (
                actual_output_tokens_mean / (mean_ms / 1000)
                if mean_ms > 0 else 0.0
            )

            all_results[config] = {
                "config": config,
                "max_new_tokens": args.wc_tokens,
                "iterations": args.wc_iterations,
                "warmup_iters": args.wc_warmup,
                "completed_iterations": len(raw_times),
                "mean_ms": round(mean_ms, 2),
                "std_ms": round(std_ms, 2),
                "tokens_per_sec": round(tps, 2),
                "actual_output_tokens_mean": round(actual_output_tokens_mean, 2),
                "actual_tokens_per_sec": round(actual_tps, 2),
                "output_tokens": output_tokens,
                "raw_times_ms": raw_times,
            }
        else:
            all_results[config] = {"error": f"rc={rc}", "config": config}

    # Save wall-clock results
    if not args.dry_run and all_results:
        wc_path = os.path.join(args.output_dir, "e1_wall_clock_results.json")
        with open(wc_path, "w") as f:
            json.dump({
                "experiment": "e1_wall_clock",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configs": all_results,
            }, f, indent=2)
        print(f"\n[e1_runner] Wall-clock results → {wc_path}", flush=True)

        # Print comparison
        valid = {k: v for k, v in all_results.items() if "error" not in v}
        if len(valid) >= 2:
            baseline_key = "dense-fa2"
            baseline = valid.get(baseline_key, list(valid.values())[0])
            print(f"\n  Wall-Clock Comparison (baseline={baseline_key}):")
            for k, v in sorted(valid.items(), key=lambda x: x[1].get("mean_ms", 999999)):
                if baseline["mean_ms"] > 0 and v["mean_ms"] > 0:
                    su = baseline["mean_ms"] / v["mean_ms"]
                    print(f"    {k:<28s} {v['mean_ms']:7.1f}ms  "
                          f"{su:6.3f}× vs baseline", flush=True)

    return all_results


# ── Step 1: nsys profiling ──────────────────────────────────────────────

def step1_nsys(args) -> dict[str, list[dict]]:
    """Run nsys profiling for all configs. Parse results.

    Returns:
        dict mapping config_name → list of kernel dicts
    """
    os.makedirs(args.output_dir, exist_ok=True)

    nsys_data = {}
    nsys_timing = {}
    for config in args.configs:
        output_base = os.path.join(args.output_dir, f"e1_nsys_{config}")
        sqlite_file = f"{output_base}.sqlite"
        # nsys stats appends report name to -o output:
        #   -o /path/kernels  →  /path/kernels_cuda_gpu_kern_sum.csv
        kernels_csv = f"{output_base}_kernels_cuda_gpu_kern_sum.csv"

        if os.path.exists(sqlite_file) and not args.force:
            print(f"[e1_runner] Step 1: {config} — sqlite already exists, "
                  f"use --force to overwrite", flush=True)
        else:
            # Run nsys profile
            # --stats=true creates <output>.sqlite automatically (nsys ≥2024)
            # --trace cuda,nvtx,cublas captures GPU kernels + NVTX markers
            cmd = [
                "nsys", "profile",
                "--trace", "cuda,nvtx,cublas",
                "--stats=true",
                "--force-overwrite=true",
                f"--output={output_base}",
                "python", BENCH_SCRIPT,
                "--config", config,
                "--mode", "nsys-profile",
                "--max-new-tokens", str(args.nsys_tokens),
                "--gpu", str(args.gpu),
                "--image-path", args.image_path,
                "--model-path", args.model_path,
            ]
            rc, stdout, stderr = run_cmd(
                cmd, f"Step 1 nsys: {config}", args.dry_run, timeout=3600)

            if rc != 0 and not args.dry_run:
                print(f"[e1_runner] ERROR: nsys failed for {config}, skipping...",
                      flush=True)
                continue

        # Export kernel summary CSV from sqlite
        if os.path.exists(sqlite_file) and (not os.path.exists(kernels_csv) or args.force):
            if args.force and os.path.exists(kernels_csv):
                os.remove(kernels_csv)
            cmd = [
                "nsys", "stats",
                "--report", "cuda_gpu_kern_sum",
                "--format", "csv",
                "-o", os.path.join(args.output_dir, f"e1_nsys_{config}_kernels"),
                sqlite_file,
            ]
            rc, stdout, stderr = run_cmd(
                cmd, f"Step 1 export CSV: {config}", args.dry_run, timeout=300)

        # Parse the CSV
        if os.path.exists(kernels_csv):
            kernels = ana.parse_nsys_kernel_csv(kernels_csv)
            nsys_data[config] = kernels
            t_total_ms = ana.compute_total_gpu_time(kernels) / 1e6

            print(f"\n[e1_runner] {config}: {len(kernels)} unique kernels, "
                  f"T_total={t_total_ms:.1f} ms", flush=True)

            # Print top-5 kernels
            top5 = ana.extract_top_kernels(kernels, 5)
            for i, k in enumerate(top5, 1):
                print(f"  {i}. [{k['category']:22s}] {k['name'][:60]:60s} "
                      f"{k['total_time_ns']/1e6:7.2f} ms ({k['time_pct']:5.1f}%)",
                      flush=True)

            # Write top-10 CSV for this config
            top10_path = os.path.join(args.output_dir, f"e1_nsys_{config}_top10.csv")
            ana.write_top_kernels_csv(kernels, top10_path, n=10)
            print(f"  → Top-10 kernels saved to {top10_path}", flush=True)

            # Write category CSV
            cats = ana.aggregate_by_category(kernels)
            cat_path = os.path.join(args.output_dir, f"e1_nsys_{config}_categories.csv")
            ana.write_kernel_categories_csv(cats, cat_path)
            print(f"  → Category breakdown saved to {cat_path}", flush=True)
        else:
            print(f"[e1_runner] WARNING: No kernel CSV for {config}", flush=True)

        if os.path.exists(sqlite_file):
            timing = ana.extract_nsys_timing_summary(
                sqlite_file, nsys_data.get(config, [])
            )
            nsys_timing[config] = timing
            nvtx_ms = timing.get("nvtx_inference_ms", 0)
            kernel_sum_ms = timing.get("gpu_kernel_sum_ms", 0)
            kernel_span_ms = timing.get("gpu_kernel_span_ms", 0)
            print(f"  → nsys timing: kernel_sum={kernel_sum_ms:.1f} ms, "
                  f"kernel_span={kernel_span_ms:.1f} ms, "
                  f"nvtx_e2e={nvtx_ms:.1f} ms", flush=True)

    if nsys_timing and not args.dry_run:
        timing_path = os.path.join(args.output_dir, "e1_nsys_timing_summary.json")
        with open(timing_path, "w") as f:
            json.dump(nsys_timing, f, indent=2)
        print(f"\n[e1_runner] nsys timing summary saved → {timing_path}",
              flush=True)

    return nsys_data


# ── Step 2: ncu deep analysis ───────────────────────────────────────────

def step2_ncu(args, nsys_data: dict[str, list[dict]]) -> list[dict]:
    """Run ncu deep analysis on hotspot kernels identified in Step 1.

    Args:
        nsys_data: dict mapping config_name → list of kernel dicts

    Returns:
        list of per-kernel ncu bottleneck results
    """
    os.makedirs(args.output_dir, exist_ok=True)
    ncu_results = []

    for config in args.configs:
        kernels = nsys_data.get(config, [])
        config_ncu_dir = os.path.join(args.output_dir, f"ncu_{config}")
        os.makedirs(config_ncu_dir, exist_ok=True)

        # Select hotspot kernels
        if kernels:
            hotspot_names = ana.select_hotspot_kernels(
                kernels, min_time_pct=args.hotspot_min_pct,
                max_kernels=args.hotspot_max_kernels)
            print(f"\n[e1_runner] {config} hotspots: {hotspot_names}", flush=True)
        else:
            # Use fallback patterns
            hotspot_names = ana.FALLBACK_NCU_PATTERNS.get(config, ["matmul"])
            print(f"\n[e1_runner] {config} fallback patterns: {hotspot_names}",
                  flush=True)

        for pattern in hotspot_names:
            # Sanitize pattern for filename and regex
            safe_name = pattern.replace(" ", "_").replace("/", "_")[:40]
            ncu_output = os.path.join(config_ncu_dir, f"e1_ncu_{config}_{safe_name}")
            csv_path = f"{ncu_output}.csv"

            if os.path.exists(csv_path) and not args.force:
                print(f"[e1_runner] Step 2: {config}/{safe_name} — "
                      f"CSV exists, skip", flush=True)
            else:
                cmd = [
                    "ncu",
                    "--set", "full",
                    "-k", f"regex:{pattern}",
                    "--launch-count", str(args.ncu_launch_count),
                    "--launch-skip", str(args.ncu_launch_skip),
                    "--csv",
                    "--log-file", f"{ncu_output}.csv",
                    "-f",
                    "--target-processes", "all",
                    "python", BENCH_SCRIPT,
                    "--config", config,
                    "--mode", "ncu-profile",
                    "--max-new-tokens", str(args.ncu_tokens),
                    "--gpu", str(args.gpu),
                    "--image-path", args.image_path,
                    "--model-path", args.model_path,
                    "--cache-dir", args.cache_dir,
                ]
                if args.no_ncu_warmup:
                    cmd.append("--no-warmup")

                warmup_flag = "--no-warmup" if args.no_ncu_warmup else ""
                rc, stdout, stderr = run_cmd(
                    cmd, f"Step 2 ncu: {config}/{safe_name} {warmup_flag}",
                    args.dry_run, timeout=1800)

                if rc != 0 and not args.dry_run:
                    print(f"[e1_runner] WARNING: ncu failed for {config}/{safe_name}",
                          flush=True)

            # Parse ncu CSV if it exists
            if os.path.exists(csv_path):
                ncu_rows = ana.parse_ncu_csv(csv_path)
                if ncu_rows:
                    # Classify bottleneck for each unique kernel
                    seen_kernels = set()
                    for row in ncu_rows:
                        kname = row.get("Kernel Name", "").strip('"')
                        if not kname or kname in seen_kernels:
                            continue
                        seen_kernels.add(kname)

                        bottleneck = ana.classify_bottleneck(row)
                        cache_metrics = ana.compute_cache_metrics([row])

                        result = {
                            "config": config,
                            "pattern": pattern,
                            "kernel_name": kname,
                            **bottleneck,
                            "gpu_time_ns": _safe_float_metric(row, "gpu__time_duration.sum"),
                            "l1_bytes": cache_metrics["l1_bytes_read"] + cache_metrics["l1_bytes_write"],
                            "l2_bytes": cache_metrics["l2_bytes_read"] + cache_metrics["l2_bytes_write"],
                            "dram_bytes": cache_metrics["dram_bytes_read"] + cache_metrics["dram_bytes_write"],
                        }
                        ncu_results.append(result)
                        print(f"  [{result['bottleneck_class']:15s}] "
                              f"U_c={result['u_compute_pct']:5.1f}% "
                              f"U_bw={result['u_bw_pct']:5.1f}% "
                              f"O_SM={result['o_sm_pct']:5.1f}% | "
                              f"{kname[:60]}", flush=True)

            # Save intermediate results
            if ncu_results:
                inter_path = os.path.join(args.output_dir, "e1_ncu_results.json")
                with open(inter_path, "w") as f:
                    json.dump(ncu_results, f, indent=2)

    # Write bottleneck CSV
    bottleneck_path = os.path.join(args.output_dir, "e1_ncu_per_kernel_bottleneck.csv")
    ana.write_ncu_bottleneck_csv(ncu_results, bottleneck_path)
    print(f"\n[e1_runner] ncu bottleneck results → {bottleneck_path}", flush=True)

    return ncu_results


def _safe_float_metric(row: dict, key: str) -> float:
    """Extract float metric from ncu row, trying various key formats."""
    try:
        return float(row.get(key, 0))
    except (ValueError, TypeError):
        pass
    # Try partial match
    for k, v in row.items():
        if key in k or k in key:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


# ── Step 3: Summarize ───────────────────────────────────────────────────

def step3_summarize(args, nsys_data: dict[str, list[dict]],
                    ncu_results: list[dict]):
    """Generate final deliverables: CSVs, charts, gap_summary.json.

    Delegates chart generation to e1_summarize.py.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    nsys_timing = {}
    timing_path = os.path.join(args.output_dir, "e1_nsys_timing_summary.json")
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            nsys_timing = json.load(f)
    else:
        for config in args.configs:
            sqlite_file = os.path.join(args.output_dir, f"e1_nsys_{config}.sqlite")
            if os.path.exists(sqlite_file):
                nsys_timing[config] = ana.extract_nsys_timing_summary(
                    sqlite_file, nsys_data.get(config, [])
                )

    # --- Compute speedups for all config pairs ---
    speedups = []
    nvtx_speedups = []
    for target in args.configs:
        if target == "dense-fa2":
            continue
        su = ana.compute_speedup(nsys_data, "dense-fa2", target)
        speedups.append(su)
        nvtx_su = ana.compute_timing_speedup(
            nsys_timing, "nvtx_inference_ms", "dense-fa2", target
        )
        if "error" not in nvtx_su:
            nvtx_speedups.append(nvtx_su)
        print(f"[e1_runner] {target} vs dense-fa2: "
              f"kernel_sum_speedup={su['actual_speedup']}× "
              f"(gap={su['gap']}×); "
              f"nvtx_e2e_speedup={nvtx_su['actual_speedup']}× "
              f"(gap={nvtx_su['gap']}×)", flush=True)

    # --- Compute cache metrics summary ---
    cache_summary = {}
    for config in args.configs:
        config_ncu = [r for r in ncu_results if r["config"] == config]
        if config_ncu:
            total_l1 = sum(r.get("l1_bytes", 0) for r in config_ncu)
            total_l2 = sum(r.get("l2_bytes", 0) for r in config_ncu)
            total_dram = sum(r.get("dram_bytes", 0) for r in config_ncu)
            total = total_l1 + total_l2 + total_dram
            cache_summary[config] = {
                "l1_bytes": total_l1,
                "l2_bytes": total_l2,
                "dram_bytes": total_dram,
                "l1_pct": round(total_l1 / total * 100, 1) if total > 0 else 0,
                "l2_pct": round(total_l2 / total * 100, 1) if total > 0 else 0,
                "dram_pct": round(total_dram / total * 100, 1) if total > 0 else 0,
            }

    # --- Build comprehensive summary ---
    summary = {
        "experiment": "e1_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric_note": (
            "speedups uses summed CUDA GPU kernel durations from "
            "cuda_gpu_kern_sum. speedups_nvtx_end_to_end uses the NVTX "
            "e1_inference range from nsys SQLite and is closer to profiled "
            "end-to-end inference latency."
        ),
        "t_total": {},
        "speedups": speedups,
        "speedups_nvtx_end_to_end": nvtx_speedups,
        "nsys_timing": nsys_timing,
        "cache_summary": cache_summary,
        "top_kernels": {},
        "kernel_categories": {},
        "ncu_bottleneck_summary": {},
    }

    for config in args.configs:
        kernels = nsys_data.get(config, [])
        t_total_ns = ana.compute_total_gpu_time(kernels)
        summary["t_total"][config] = {
            "total_time_ns": int(t_total_ns),
            "total_time_ms": round(t_total_ns / 1e6, 3),
        }

        top10 = ana.extract_top_kernels(kernels, 10)
        summary["top_kernels"][config] = [
            {"name": k["name"], "category": k["category"],
             "total_time_ms": round(k["total_time_ns"] / 1e6, 3),
             "time_pct": round(k["time_pct"], 1)}
            for k in top10
        ]

        cats = ana.aggregate_by_category(kernels)
        summary["kernel_categories"][config] = [
            {"category": c["category"], "display": c["display"],
             "total_time_ms": round(c["total_time_ns"] / 1e6, 3),
             "time_pct": round(c["time_pct"], 1)}
            for c in cats
        ]

    # Count bottleneck types per config
    for config in args.configs:
        config_ncu = [r for r in ncu_results if r["config"] == config]
        bottleneck_counts = defaultdict(int)
        for r in config_ncu:
            bottleneck_counts[r.get("bottleneck_class", "unknown")] += 1
        summary["ncu_bottleneck_summary"][config] = dict(bottleneck_counts)

    # Write gap_summary.json
    summary_path = os.path.join(args.output_dir, "e1_gap_summary.json")
    ana.write_gap_summary_json(summary, summary_path)
    print(f"\n[e1_runner] Summary → {summary_path}", flush=True)

    # --- Generate figures via e1_summarize.py ---
    summarize_script = os.path.join(_CODE_DIR, "e1_summarize.py")
    if os.path.exists(summarize_script):
        print(f"\n[e1_runner] Generating figures via e1_summarize.py...", flush=True)
        cmd = [
            "python", summarize_script,
            "--summary-json", summary_path,
            "--nsys-dir", args.output_dir,
            "--output-dir", args.output_dir,
        ]
        rc, stdout, stderr = run_cmd(cmd, "Step 3: figures", args.dry_run, timeout=300)
    else:
        print(f"[e1_runner] NOTE: e1_summarize.py not found at {summarize_script}, "
              f"skipping figures", flush=True)

    # Print final summary table
    print(f"\n{'='*72}")
    print(f"  E1 Baseline Gap — Final Summary")
    print(f"{'='*72}")
    print(f"  {'Config':<28s} {'T_total':>10s} {'L1%':>6s} {'L2%':>6s} {'DRAM%':>6s} {'Bottleneck':>15s}")
    print(f"  {'-'*72}")
    for config in args.configs:
        t_ms = summary["t_total"][config]["total_time_ms"]
        cs = cache_summary.get(config, {})
        l1 = cs.get("l1_pct", 0)
        l2 = cs.get("l2_pct", 0)
        dram = cs.get("dram_pct", 0)
        bn = summary["ncu_bottleneck_summary"].get(config, {})
        dominant = max(bn, key=bn.get) if bn else "unknown"
        print(f"  {config:<28s} {t_ms:8.1f}ms {l1:5.1f}% {l2:5.1f}% {dram:5.1f}% {dominant:>15s}")
    print(f"{'='*72}")

    if speedups:
        main_su = next((s for s in speedups if s["target_config"] == "visipruner-full"), speedups[0])
        print(f"  Kernel-sum speedup (visipruner-full vs dense-fa2): {main_su['actual_speedup']}×")
        print(f"  Theoretical speedup:                          2.17×")
        print(f"  GAP:                                           {main_su['gap']}×")
    if nvtx_speedups:
        main_nvtx = next((s for s in nvtx_speedups if s["target_config"] == "visipruner-full"), nvtx_speedups[0])
        print(f"  NVTX e2e speedup (visipruner-full vs dense-fa2): {main_nvtx['actual_speedup']}×")
        print(f"  NVTX e2e GAP:                                   {main_nvtx['gap']}×")
    print(f"{'='*72}")

    return summary


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Step 0: Wall-clock timing
    wc_data = {}
    if args.step in ("0", "wc", "all"):
        print(f"\n{'#'*72}")
        print(f"# E1 Step 0: Wall-Clock Timing")
        print(f"# Configs: {args.configs}")
        print(f"{'#'*72}", flush=True)
        wc_data = step0_wall_clock(args)

    # Step 1: nsys
    nsys_data = {}
    if args.step in ("1", "all"):
        print(f"\n{'#'*72}")
        print(f"# E1 Step 1: nsys full-timeline profiling")
        print(f"# Configs: {args.configs}")
        print(f"{'#'*72}", flush=True)
        nsys_data = step1_nsys(args)

        # Save intermediate nsys data
        nsys_data_path = os.path.join(args.output_dir, "e1_nsys_data.json")
        if not args.dry_run:
            # Convert to serializable form
            serializable = {
                config: [
                    {**k, "total_time_ns": int(k["total_time_ns"])}
                    for k in kernels
                ]
                for config, kernels in nsys_data.items()
            }
            with open(nsys_data_path, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\n[e1_runner] nsys data saved → {nsys_data_path}", flush=True)

    # Step 2: ncu
    ncu_results = []
    if args.step in ("2", "all"):
        # Load nsys data if only running step 2
        if not nsys_data:
            nsys_data_path = os.path.join(args.output_dir, "e1_nsys_data.json")
            if os.path.exists(nsys_data_path):
                with open(nsys_data_path) as f:
                    nsys_data = json.load(f)
                print(f"[e1_runner] Loaded nsys data from {nsys_data_path}", flush=True)
            else:
                print(f"[e1_runner] WARNING: No nsys data found. "
                      f"Run Step 1 first or use --step all.", flush=True)
                # Continue with fallback patterns

        print(f"\n{'#'*72}")
        print(f"# E1 Step 2: ncu deep kernel analysis")
        print(f"# Configs: {args.configs}")
        print(f"{'#'*72}", flush=True)
        ncu_results = step2_ncu(args, nsys_data)

    # Step 3: Summarize
    if args.step in ("3", "all"):
        # Load data if only running step 3
        if not nsys_data:
            nsys_data_path = os.path.join(args.output_dir, "e1_nsys_data.json")
            if os.path.exists(nsys_data_path):
                with open(nsys_data_path) as f:
                    nsys_data_raw = json.load(f)
                # Reconstruct kernel dicts with float values
                nsys_data = {
                    config: [{**k, "total_time_ns": float(k["total_time_ns"])}
                             for k in kernels]
                    for config, kernels in nsys_data_raw.items()
                }
        if not ncu_results:
            ncu_path = os.path.join(args.output_dir, "e1_ncu_results.json")
            if os.path.exists(ncu_path):
                with open(ncu_path) as f:
                    ncu_results = json.load(f)

        print(f"\n{'#'*72}")
        print(f"# E1 Step 3: Summarize and generate deliverables")
        print(f"{'#'*72}", flush=True)
        step3_summarize(args, nsys_data, ncu_results)

    elapsed = time.time() - start_time
    print(f"\n[e1_runner] Total elapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
