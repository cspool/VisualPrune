#!/usr/bin/env python3
"""Single-request latency breakdown for native VisPrune/LLaVA inference.

The script is intentionally single-request oriented:
  - no throughput calculation
  - optional warmup outside the measured request
  - wall-clock timing with CUDA synchronization for clock runs
  - NVTX ranges for Nsight Systems runs

Use `/workspace/VisPrune/venv_profiling/bin/python` for this checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GPU = 1
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_PATH = (
    "/workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg"
)
DEFAULT_PROMPT = "Describe the image briefly."
DEFAULT_OUTPUT_DIR = (
    "/workspace/VisPrune/autoresearch/experiments/e2_single_request_latency/output"
)

VISIPRUNER_CONFIGS: dict[str, dict[str, Any]] = {
    "dense-fa": {
        "use_flash_attn": False,
        "use_visipruner": False,
        "pruning_config": None,
        "description": "Dense full-attention LLaVA reference; no VisPrune pruning.",
    },
    "dense-fa2": {
        "use_flash_attn": True,
        "use_visipruner": False,
        "pruning_config": None,
        "description": "Dense LLaVA reference with FlashAttention2; no VisPrune pruning.",
    },
    "visipruner-full": {
        "use_flash_attn": False,
        "use_visipruner": True,
        "pruning_config": {
            "mode": ["shallow", "middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "Native VisPrune full path: shallow + middle + deep.",
    },
    "visipruner-middle-deep": {
        "use_flash_attn": False,
        "use_visipruner": True,
        "pruning_config": {
            "mode": ["middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "Native VisPrune middle + deep path.",
    },
    "dense-eager": {
        "use_flash_attn": False,
        "use_visipruner": False,
        "pruning_config": None,
        "description": "Alias for dense-fa; dense eager full attention.",
    },
}


def get_flash_attn_info() -> dict[str, str | bool | None]:
    try:
        import flash_attn

        return {
            "available": True,
            "version": getattr(flash_attn, "__version__", None),
            "path": getattr(flash_attn, "__file__", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_gpu_early(argv: list[str]) -> int:
    for idx, arg in enumerate(argv):
        if arg == "--gpu" and idx + 1 < len(argv):
            return int(argv[idx + 1])
        if arg.startswith("--gpu="):
            return int(arg.split("=", 1)[1])
    return DEFAULT_GPU


os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_parse_gpu_early(sys.argv)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/VisPrune/models")

REPO_DIR = Path(__file__).resolve().parents[4] / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


@dataclass
class RangeStats:
    values_ms: list[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        self.values_ms.append(value_ms)

    def as_dict(self) -> dict[str, float | int]:
        total = sum(self.values_ms)
        return {
            "count": len(self.values_ms),
            "total_ms": total,
            "mean_ms": statistics.mean(self.values_ms) if self.values_ms else 0.0,
            "min_ms": min(self.values_ms) if self.values_ms else 0.0,
            "max_ms": max(self.values_ms) if self.values_ms else 0.0,
        }


class LatencyRecorder:
    def __init__(self, sync_cuda: bool, enable_nvtx: bool):
        self.sync_cuda = sync_cuda
        self.enable_nvtx = enable_nvtx and torch.cuda.is_available()
        self.ranges: dict[str, RangeStats] = {}

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def range(self, name: str):
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(name)
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.ranges.setdefault(name, RangeStats()).add(elapsed_ms)
            if self.enable_nvtx:
                torch.cuda.nvtx.range_pop()

    def summarized(self) -> dict[str, dict[str, float | int]]:
        return {name: stats.as_dict() for name, stats in sorted(self.ranges.items())}


def build_prompt(raw_prompt: str, conv_mode: str) -> str:
    prompt = raw_prompt
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def patch_model_for_ranges(model, recorder: LatencyRecorder, tracker: dict[str, Any]) -> None:
    tracker["active_prefix"] = "visprune"
    model._visprune_profile_tracker = tracker

    original_prepare = model.prepare_inputs_labels_for_multimodal
    original_encode_images = model.encode_images
    original_forward = model.forward

    def wrapped_encode_images(images):
        prefix = tracker.get("active_prefix", "visprune")
        with recorder.range(f"{prefix}.vision_encode_project"):
            return original_encode_images(images)

    def wrapped_prepare(*args, **kwargs):
        prefix = tracker.get("active_prefix", "visprune")
        with recorder.range(f"{prefix}.prepare_multimodal"):
            return original_prepare(*args, **kwargs)

    def wrapped_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is None and args:
            input_ids = args[0]

        if inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])
        elif input_ids is not None:
            seq_len = int(input_ids.shape[1])
        else:
            seq_len = -1

        if seq_len > 1:
            range_suffix = "forward_prefill"
            if tracker.get("active_prefix", "visprune") == "visprune":
                tracker["prefill_seq_lens"].append(seq_len)
        else:
            range_suffix = "forward_decode"
            if tracker.get("active_prefix", "visprune") == "visprune":
                tracker["decode_seq_lens"].append(seq_len)

        range_name = f"{tracker.get('active_prefix', 'visprune')}.{range_suffix}"
        with recorder.range(range_name):
            return original_forward(*args, **kwargs)

    model.encode_images = wrapped_encode_images
    model.prepare_inputs_labels_for_multimodal = wrapped_prepare
    model.forward = wrapped_forward

    base_model = model.get_model()
    for layer_idx, layer in enumerate(base_model.layers):
        attn = getattr(layer, "self_attn", None)
        if not hasattr(attn, "value_aware_token_selection"):
            continue
        original_select = attn.value_aware_token_selection

        def make_wrapped_select(orig, idx):
            def wrapped_select(*args, **kwargs):
                prefix = tracker.get("active_prefix", "visprune")
                with recorder.range(f"{prefix}.value_aware_token_selection"):
                    result = orig(*args, **kwargs)
                if prefix != "visprune":
                    return result
                if torch.is_tensor(result):
                    tracker["value_select_tensor_layers"].append(idx)
                    tracker["selected_visual_token_counts"].append(int(result.numel()))
                elif isinstance(result, bool):
                    tracker["deep_exit_checks"].append({"layer": idx, "exit": bool(result)})
                return result

            return wrapped_select

        attn.value_aware_token_selection = make_wrapped_select(original_select, layer_idx)


def load_model(config: dict[str, Any], model_path: str, model_base: str | None):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map="cuda:0",
        use_flash_attn=config["use_flash_attn"],
        use_visipruner=config["use_visipruner"],
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def run_request(
    *,
    recorder: LatencyRecorder,
    model,
    tokenizer,
    image_processor,
    image_path: str,
    prompt: str,
    conv_mode: str,
    max_new_tokens: int,
    temperature: float,
    pruning_config: dict[str, Any] | None,
    measured: bool,
) -> dict[str, Any]:
    prefix = "visprune" if measured else "warmup"
    previous_prefix = None
    if hasattr(model, "_visprune_profile_tracker"):
        previous_prefix = model._visprune_profile_tracker.get("active_prefix")
        model._visprune_profile_tracker["active_prefix"] = prefix

    try:
        with recorder.range(f"{prefix}.build_prompt_tokenize"):
            prompt_text = build_prompt(prompt, conv_mode)
            input_ids = tokenizer_image_token(
                prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        with recorder.range(f"{prefix}.load_image"):
            image = Image.open(image_path).convert("RGB")
            image_size = image.size

        with recorder.range(f"{prefix}.image_preprocess_cpu"):
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]

        with recorder.range(f"{prefix}.input_h2d"):
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            attention_mask = attention_mask.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        with recorder.range(f"{prefix}.generate_total"):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=temperature > 0,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    pruning_config=pruning_config,
                    use_cache=True,
                )
    finally:
        if hasattr(model, "_visprune_profile_tracker"):
            model._visprune_profile_tracker["active_prefix"] = previous_prefix or "visprune"

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if output_ids.shape[1] > input_ids.shape[1]:
        output_token_ids = output_ids[0, input_ids.shape[1] :]
    else:
        output_token_ids = output_ids[0]
    output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()

    return {
        "prompt_token_count": int(input_ids.shape[1]),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "output_token_count": int(output_token_ids.numel()),
        "output_text": output_text,
    }


def derived_breakdown(summary: dict[str, dict[str, float | int]]) -> dict[str, float]:
    def total(name: str) -> float:
        return float(summary.get(name, {}).get("total_ms", 0.0))

    request_total = total("visprune.request")
    generate_total = total("visprune.generate_total")
    prepare = total("visprune.prepare_multimodal")
    vision = total("visprune.vision_encode_project")
    prefill = total("visprune.forward_prefill")
    decode = total("visprune.forward_decode")

    return {
        "request_total_ms": request_total,
        "non_generate_ms": max(0.0, request_total - generate_total),
        "generate_total_ms": generate_total,
        "prepare_multimodal_ms": prepare,
        "vision_encode_project_ms": vision,
        "prepare_without_vision_ms": max(0.0, prepare - vision),
        "forward_prefill_ms": prefill,
        "forward_decode_sum_ms": decode,
        "generate_other_ms": max(0.0, generate_total - prepare - prefill - decode),
        "value_aware_token_selection_ms": total("visprune.value_aware_token_selection"),
    }


def write_outputs(payload: dict[str, Any], output_dir: str, tag: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    csv_path = out_dir / f"{tag}_ranges.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    rows = []
    request_total = payload["derived_ms"].get("request_total_ms", 0.0) or 0.0
    generate_total = payload["derived_ms"].get("generate_total_ms", 0.0) or 0.0
    for name, data in payload["ranges"].items():
        total_ms = float(data["total_ms"])
        rows.append(
            {
                "range": name,
                "count": data["count"],
                "total_ms": total_ms,
                "mean_ms": data["mean_ms"],
                "min_ms": data["min_ms"],
                "max_ms": data["max_ms"],
                "pct_request": (100.0 * total_ms / request_total) if request_total else 0.0,
                "pct_generate": (100.0 * total_ms / generate_total) if generate_total else 0.0,
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "range",
                "count",
                "total_ms",
                "mean_ms",
                "min_ms",
                "max_ms",
                "pct_request",
                "pct_generate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    latest_json = out_dir / "clock_latest.json"
    latest_csv = out_dir / "clock_latest_ranges.csv"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(VISIPRUNER_CONFIGS), default="visipruner-full")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--sync-timing", choices=["on", "off"], default="on")
    parser.add_argument("--nvtx", choices=["on", "off"], default="on")
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Call torch.cuda.profiler.start/stop around the measured request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VISIPRUNER_CONFIGS[args.config]
    recorder = LatencyRecorder(
        sync_cuda=args.sync_timing == "on",
        enable_nvtx=args.nvtx == "on",
    )
    tracker: dict[str, Any] = {
        "prefill_seq_lens": [],
        "decode_seq_lens": [],
        "value_select_tensor_layers": [],
        "selected_visual_token_counts": [],
        "deep_exit_checks": [],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"visprune_single_{args.config}_{timestamp}"

    with recorder.range("visprune.model_load"):
        tokenizer, model, image_processor, context_len = load_model(
            config, args.model_path, args.model_base
        )
    patch_model_for_ranges(model, recorder, tracker)

    for _ in range(args.warmup_iters):
        run_request(
            recorder=recorder,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=args.image_path,
            prompt=args.prompt,
            conv_mode=args.conv_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            pruning_config=config["pruning_config"],
            measured=False,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if args.cuda_profiler_api:
        torch.cuda.profiler.start()

    try:
        with recorder.range("visprune.request"):
            request = run_request(
                recorder=recorder,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=args.image_path,
                prompt=args.prompt,
                conv_mode=args.conv_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pruning_config=config["pruning_config"],
                measured=True,
            )
    finally:
        if args.cuda_profiler_api:
            torch.cuda.profiler.stop()

    ranges = recorder.summarized()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "config": args.config,
        "description": config["description"],
        "use_visipruner": config["use_visipruner"],
        "use_flash_attn": config["use_flash_attn"],
        "pruning_config": config["pruning_config"],
        "model_path": args.model_path,
        "model_base": args.model_base,
        "image_path": args.image_path,
        "prompt": args.prompt,
        "conv_mode": args.conv_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "warmup_iters": args.warmup_iters,
        "sync_timing": args.sync_timing,
        "nvtx": args.nvtx,
        "cuda_profiler_api": args.cuda_profiler_api,
        "context_len": context_len,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_home": os.environ.get("HF_HOME"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "flash_attn": get_flash_attn_info(),
        },
        "request": request,
        "tracker": tracker,
        "ranges": ranges,
        "derived_ms": derived_breakdown(ranges),
    }

    json_path, csv_path = write_outputs(payload, args.output_dir, tag)

    print(json.dumps(payload["derived_ms"], indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
