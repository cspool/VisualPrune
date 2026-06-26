#!/usr/bin/env python3
"""Run a LLaVA-style VQA JSONL workload against an SGLang OpenAI endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Sample:
    index: int
    raw: dict[str, Any]
    image_path: Path
    prompt: str
    image_ref: str


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class Result:
    index: int
    success: bool
    latency_s: float
    output_text: str
    response: dict[str, Any] | None
    error: str | None
    sample: Sample


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_arg(value: str | None, field_name: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for {field_name}: {exc}") from exc


def bounded_rglob(root: Path, max_depth: int) -> list[Path]:
    if max_depth <= 0:
        return []
    matches: list[Path] = []
    root_depth = len(root.parts)
    for path in root.rglob("*"):
        if len(path.parts) - root_depth > max_depth:
            continue
        if path.is_file():
            matches.append(path)
    return matches


def resolve_image_path(image_folder: Path, image_ref: str, max_depth: int) -> Path:
    direct = (image_folder / image_ref).resolve()
    if direct.is_file():
        return direct

    candidates: list[Path] = []
    ref_path = Path(image_ref)
    if ref_path.suffix:
        candidates.append(ref_path)
    else:
        candidates.extend(Path(image_ref + ext) for ext in IMAGE_EXTENSIONS)

    for candidate in candidates:
        direct_candidate = (image_folder / candidate).resolve()
        if direct_candidate.is_file():
            return direct_candidate

    if max_depth <= 0:
        raise FileNotFoundError(str(direct))

    candidate_names = {candidate.name for candidate in candidates}
    candidate_suffixes = {str(candidate) for candidate in candidates}
    hits = []
    for path in bounded_rglob(image_folder, max_depth):
        rel = path.relative_to(image_folder)
        if path.name in candidate_names or str(rel) in candidate_suffixes:
            hits.append(path.resolve())

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        exact_suffix_hits = [
            path for path in hits if any(str(path).endswith(str(s)) for s in candidates)
        ]
        if len(exact_suffix_hits) == 1:
            return exact_suffix_hits[0]
        raise FileNotFoundError(
            f"Ambiguous image {image_ref!r}; matches: {', '.join(map(str, hits[:8]))}"
        )

    raise FileNotFoundError(str(direct))


def load_samples(args: argparse.Namespace) -> list[Sample]:
    question_file = Path(args.question_file).expanduser().resolve()
    image_folder = Path(args.image_folder).expanduser().resolve()
    if not question_file.is_file():
        raise SystemExit(f"Question file not found: {question_file}")
    if not image_folder.is_dir():
        raise SystemExit(f"Image folder not found: {image_folder}")

    samples: list[Sample] = []
    with question_file.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            if line_index < args.offset:
                continue
            if args.limit and len(samples) >= args.limit:
                break
            obj = json.loads(line)
            image_rel = obj.get(args.image_field)
            prompt = obj.get(args.question_field)
            if not image_rel:
                raise SystemExit(
                    f"Missing image field {args.image_field!r} at line {line_index + 1}"
                )
            if prompt is None:
                raise SystemExit(
                    f"Missing question field {args.question_field!r} at line {line_index + 1}"
                )
            image_ref = str(image_rel)
            try:
                image_path = resolve_image_path(
                    image_folder, image_ref, args.image_search_depth
                )
            except FileNotFoundError as exc:
                raise SystemExit(
                    f"Image not found for line {line_index + 1}: {image_ref} "
                    f"under {image_folder}"
                ) from exc
            samples.append(
                Sample(
                    index=line_index,
                    raw=obj,
                    image_path=image_path,
                    prompt=str(prompt),
                    image_ref=image_ref,
                )
            )

    if not samples:
        raise SystemExit("No samples selected.")
    return samples


def image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_payload(sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(sample.image_path)},
                        "modalities": "image",
                    },
                    {"type": "text", "text": sample.prompt},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_completion_tokens": args.max_new_tokens,
        "stream": False,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.stop:
        payload["stop"] = args.stop
    payload.update(args.extra_request_body or {})
    return payload


async def send_one(
    client: httpx.AsyncClient,
    sample: Sample,
    args: argparse.Namespace,
) -> Result:
    payload = build_payload(sample, args)
    start = time.perf_counter()
    try:
        response = await client.post(args.chat_completions_url, json=payload)
        latency = time.perf_counter() - start
        if response.status_code >= 400:
            return Result(
                index=sample.index,
                success=False,
                latency_s=latency,
                output_text="",
                response=None,
                error=f"HTTP {response.status_code}: {response.text[:1000]}",
                sample=sample,
            )
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content)
        return Result(
            index=sample.index,
            success=True,
            latency_s=latency,
            output_text=str(content).strip(),
            response=data,
            error=None,
            sample=sample,
        )
    except Exception as exc:  # noqa: BLE001 - record benchmark failures per request.
        latency = time.perf_counter() - start
        return Result(
            index=sample.index,
            success=False,
            latency_s=latency,
            output_text="",
            response=None,
            error=repr(exc),
            sample=sample,
        )


async def run_requests(samples: list[Sample], args: argparse.Namespace) -> list[Result]:
    limits = httpx.Limits(
        max_connections=max(args.concurrency, 1) + 4,
        max_keepalive_connections=max(args.concurrency, 1) + 4,
    )
    timeout = httpx.Timeout(args.request_timeout)
    results: list[Result] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if args.warmup:
            warmup_samples = samples[: min(args.warmup, len(samples))]
            for sample in warmup_samples:
                await send_one(client, sample, args)

        semaphore = asyncio.Semaphore(max(args.concurrency, 1))

        async def bounded(sample: Sample) -> Result:
            async with semaphore:
                return await send_one(client, sample, args)

        tasks = [asyncio.create_task(bounded(sample)) for sample in samples]
        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            if args.progress and (completed == len(samples) or completed % args.progress == 0):
                print(f"completed {completed}/{len(samples)}", flush=True)

    return sorted(results, key=lambda item: item.index)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def token_count(results: list[Result], key: str) -> int | None:
    total = 0
    found = False
    for result in results:
        usage = (result.response or {}).get("usage") or {}
        value = usage.get(key)
        if isinstance(value, int):
            total += value
            found = True
    return total if found else None


def write_answers(results: list[Result], args: argparse.Namespace) -> None:
    answers_file = Path(args.answers_file).expanduser().resolve()
    answers_file.parent.mkdir(parents=True, exist_ok=True)
    with answers_file.open("w", encoding="utf-8") as f:
        for result in results:
            if not result.success and not args.write_failures:
                continue
            usage = (result.response or {}).get("usage")
            metadata = {
                "backend": args.backend_label,
                "latency_s": result.latency_s,
                "success": result.success,
                "usage": usage,
                "image": result.sample.image_ref,
                "resolved_image_path": str(result.sample.image_path),
                "pruning_config": args.pruning_config_json,
                "pruning_config_applied_in_sglang": False,
            }
            if result.error:
                metadata["error"] = result.error
            line = {
                "question_id": result.sample.raw.get("question_id", result.index),
                "prompt": result.sample.prompt,
                "text": result.output_text,
                "answer_id": uuid.uuid4().hex,
                "model_id": args.model_id,
                "metadata": metadata,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_summary(
    results: list[Result],
    args: argparse.Namespace,
    started_at: str,
    ended_at: str,
    wall_time_s: float,
) -> dict[str, Any]:
    metrics_file = Path(args.metrics_file).expanduser().resolve()
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    success_results = [result for result in results if result.success]
    failed_results = [result for result in results if not result.success]
    latencies = [result.latency_s for result in success_results]
    completion_tokens = token_count(success_results, "completion_tokens")
    prompt_tokens = token_count(success_results, "prompt_tokens")
    total_tokens = token_count(success_results, "total_tokens")

    summary: dict[str, Any] = {
        "backend": args.backend_label,
        "dataset": args.dataset_name,
        "model": args.model,
        "model_id": args.model_id,
        "question_file": str(Path(args.question_file).expanduser().resolve()),
        "image_folder": str(Path(args.image_folder).expanduser().resolve()),
        "answers_file": str(Path(args.answers_file).expanduser().resolve()),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_s": wall_time_s,
        "selected": len(results),
        "completed": len(success_results),
        "failed": len(failed_results),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "request_throughput_rps": len(success_results) / wall_time_s
        if wall_time_s > 0
        else None,
        "mean_latency_s": safe_mean(latencies),
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "p90_latency_s": percentile(latencies, 90),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "min_latency_s": min(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_s": completion_tokens / wall_time_s
        if completion_tokens is not None and wall_time_s > 0
        else None,
        "pruning_config": args.pruning_config_json,
        "pruning_config_applied_in_sglang": False,
        "failures": [
            {
                "question_id": result.sample.raw.get("question_id", result.index),
                "image": result.sample.raw.get(args.image_field),
                "error": result.error,
            }
            for result in failed_results[: args.max_failure_records]
        ],
    }
    metrics_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a LLaVA-style VQA JSONL workload on SGLang."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--model-id", default="llava-v1.5-7b-sglang")
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--dataset-name", default="vqa")
    parser.add_argument("--backend-label", default="sglang")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--question-field", default="text")
    parser.add_argument(
        "--image-search-depth",
        type=int,
        default=0,
        help="Search under image folder up to this depth when direct image path is missing.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to run. 0 means all selected samples.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--progress", type=int, default=1)
    parser.add_argument("--write-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-request-body", default=None)
    parser.add_argument("--pruning-config", default=None)
    parser.add_argument("--max-failure-records", type=int, default=20)
    args = parser.parse_args()
    args.chat_completions_url = args.base_url.rstrip("/") + "/chat/completions"
    args.extra_request_body = load_json_arg(
        args.extra_request_body, "--extra-request-body"
    )
    args.pruning_config_json = load_json_arg(args.pruning_config, "--pruning-config")
    return args


def main() -> None:
    args = parse_args()
    samples = load_samples(args)

    if args.dry_run:
        first = samples[0]
        payload = build_payload(first, args)
        payload["messages"][0]["content"][0]["image_url"]["url"] = (
            f"<data-url bytes={first.image_path.stat().st_size}>"
        )
        print(
            json.dumps(
                {
                    "selected": len(samples),
                    "first_question_id": first.raw.get("question_id", first.index),
                    "first_image": str(first.image_path),
                    "payload": payload,
                    "note": "Dry run validates files and payload shape only.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    started_at = utc_now()
    wall_start = time.perf_counter()
    results = asyncio.run(run_requests(samples, args))
    wall_time_s = time.perf_counter() - wall_start
    ended_at = utc_now()

    write_answers(results, args)
    write_summary(results, args, started_at, ended_at, wall_time_s)

    failures = [result for result in results if not result.success]
    if failures and args.write_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
