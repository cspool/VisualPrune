#!/usr/bin/env python3
"""Microbench LLaVA-7B decode GEMV concurrency options.

This is a profiling aid, not a production runtime implementation.
It compares serial batch-1 GEMV against larger batched GEMM for the projection
shapes that dominate decode: attention projections and MLP projections.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


DEFAULT_OUTPUT = (
    "/workspace/VisPrune/autoresearch/experiments/e2_single_request_latency/output/"
    "gemv_concurrency_bench.json"
)


@dataclass(frozen=True)
class Shape:
    name: str
    in_features: int
    out_features: int


SHAPES = [
    Shape("attn_square_4096x4096", 4096, 4096),
    Shape("mlp_up_or_gate_4096x11008", 4096, 11008),
    Shape("mlp_down_11008x4096", 11008, 4096),
]


def parse_batches(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def sync() -> None:
    torch.cuda.synchronize()


def measure_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    sync()
    # Keep a reference until after synchronization.
    _ = out
    return float(start.elapsed_time(end)) / iters


def serial_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    outs = []
    for idx in range(x.shape[0]):
        outs.append(F.linear(x[idx : idx + 1], weight))
    return outs[-1]


def serial_projections(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    outs = []
    for weight in weights:
        outs.append(F.linear(x, weight))
    return outs[-1]


def bench_single_shape(shape: Shape, batches: list[int], warmup: int, iters: int) -> list[dict]:
    results = []
    weight = torch.randn(
        shape.out_features, shape.in_features, device="cuda", dtype=torch.float16
    )

    for batch in batches:
        x = torch.randn(batch, shape.in_features, device="cuda", dtype=torch.float16)
        serial_ms = measure_ms(
            lambda: serial_rows(x, weight),
            warmup=warmup,
            iters=iters,
        )
        batched_ms = measure_ms(
            lambda: F.linear(x, weight),
            warmup=warmup,
            iters=iters,
        )
        flops = 2.0 * batch * shape.in_features * shape.out_features
        results.append(
            {
                "kind": "single_projection",
                "shape": shape.name,
                "batch_rows": batch,
                "serial_ms": serial_ms,
                "batched_ms": batched_ms,
                "speedup": serial_ms / batched_ms if batched_ms else 0.0,
                "batched_tflops": flops / (batched_ms / 1000.0) / 1e12,
            }
        )
        del x

    del weight
    torch.cuda.empty_cache()
    return results


def bench_concat(
    name: str,
    in_features: int,
    out_features: int,
    count: int,
    batches: list[int],
    warmup: int,
    iters: int,
) -> list[dict]:
    results = []
    weights = [
        torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
        for _ in range(count)
    ]
    fused_weight = torch.cat(weights, dim=0).contiguous()

    for batch in batches:
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.float16)
        serial_ms = measure_ms(
            lambda: serial_projections(x, weights),
            warmup=warmup,
            iters=iters,
        )
        fused_ms = measure_ms(
            lambda: F.linear(x, fused_weight),
            warmup=warmup,
            iters=iters,
        )
        flops = 2.0 * batch * in_features * out_features * count
        results.append(
            {
                "kind": "projection_concat",
                "shape": name,
                "batch_rows": batch,
                "serial_ms": serial_ms,
                "fused_ms": fused_ms,
                "speedup": serial_ms / fused_ms if fused_ms else 0.0,
                "fused_tflops": flops / (fused_ms / 1000.0) / 1e12,
            }
        )
        del x

    del weights, fused_weight
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    batches = parse_batches(args.batches)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for shape in SHAPES:
        results.extend(bench_single_shape(shape, batches, args.warmup, args.iters))

    results.extend(
        bench_concat(
            "qkv_concat_3x4096x4096",
            in_features=4096,
            out_features=4096,
            count=3,
            batches=batches,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    results.extend(
        bench_concat(
            "mlp_gate_up_concat_2x4096x11008",
            in_features=4096,
            out_features=11008,
            count=2,
            batches=batches,
            warmup=args.warmup,
            iters=args.iters,
        )
    )

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batches": batches,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    fieldnames = sorted({key for row in results for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(payload, indent=2))
    print(f"JSON: {output}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
