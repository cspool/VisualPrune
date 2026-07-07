#!/usr/bin/env python3
"""Run a torch.profiler trace for one VisiPrune/LLaVA generate request.

This tool intentionally tests `with_stack=True` and `with_modules=True` and
then renders a profiler-derived process sketch for comparison with the existing
FX and TorchDispatch visualizations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[3]
WORKLOAD_DIR = ROOT_DIR / "workload_analysis"
REPO_DIR = ROOT_DIR / "repo"
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_PATH = str(ROOT_DIR / "autoresearch/data/benchmark_images/002901d9d194c4fb.jpg")
DEFAULT_OUTPUT_DIR = str(WORKLOAD_DIR / "torch_profile/traces")
DEFAULT_PROMPT = "Describe the image briefly."
DEFAULT_GPU = 1


CONFIGS: dict[str, dict[str, Any]] = {
    "dense-eager": {
        "use_flash_attn": False,
        "use_visipruner": False,
        "visipruner_decode_backend": "off",
        "pruning_config": None,
        "description": "Dense LLaVA eager attention baseline.",
    },
    "dense-fa2": {
        "use_flash_attn": True,
        "use_visipruner": False,
        "visipruner_decode_backend": "off",
        "pruning_config": None,
        "description": "Dense LLaVA FlashAttention2 baseline.",
    },
    "visipruner-full": {
        "use_flash_attn": False,
        "use_visipruner": True,
        "visipruner_decode_backend": "eager",
        "pruning_config": {
            "mode": ["shallow", "middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "Native VisiPrune full path: shallow + middle + deep.",
    },
    "visipruner-full-fa2": {
        "use_flash_attn": True,
        "use_visipruner": True,
        "visipruner_decode_backend": "auto",
        "pruning_config": {
            "mode": ["shallow", "middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "VisiPrune full path with optimized backend auto-selection.",
    },
    "visipruner-full-vp-fa": {
        "use_flash_attn": True,
        "use_visipruner": True,
        "visipruner_decode_backend": "vp-fa",
        "pruning_config": {
            "mode": ["shallow", "middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "VisiPrune full path with Triton VP-FA prefill.",
    },
}


def parse_gpu_early(argv: list[str]) -> int:
    for idx, arg in enumerate(argv):
        if arg == "--gpu" and idx + 1 < len(argv):
            return int(argv[idx + 1])
        if arg.startswith("--gpu="):
            return int(arg.split("=", 1)[1])
    return DEFAULT_GPU


os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(parse_gpu_early(sys.argv)))
os.environ.setdefault("HF_HOME", str(ROOT_DIR / "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


@dataclass
class ProfilerState:
    forward_id: int = 0
    current_forward_id: int | None = None
    current_phase: str | None = None
    active_prefix: str = "warmup"
    forward_events: list[dict[str, Any]] = field(default_factory=list)
    layer_events: list[dict[str, Any]] = field(default_factory=list)
    selection_events: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def record_scope(name: str):
    with torch.profiler.record_function(name):
        yield


def int_shape(value: Any) -> list[int] | None:
    if value is None or not hasattr(value, "shape"):
        return None
    return [int(item) for item in value.shape]


def tensor_count(value: Any) -> int | None:
    if torch.is_tensor(value):
        return int(value.numel())
    return None


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_cache_len(past_key_value: Any, q_len: int, layer_idx: int) -> int:
    if past_key_value is None:
        return 0
    if hasattr(past_key_value, "get_usable_length"):
        try:
            return int(past_key_value.get_usable_length(q_len, layer_idx=layer_idx))
        except TypeError:
            return int(past_key_value.get_usable_length(q_len))
        except Exception:
            return 0
    try:
        return int(past_key_value[layer_idx][0].shape[-2])
    except Exception:
        return 0


def build_prompt(raw_prompt: str, conv_mode: str) -> str:
    prompt = raw_prompt
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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
        visipruner_decode_backend=config["visipruner_decode_backend"],
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def patch_model_for_profiler(model: Any, state: ProfilerState, *, layer_profile: bool) -> None:
    original_forward = model.forward

    def wrapped_model_forward(*args: Any, **kwargs: Any) -> Any:
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

        state.forward_id += 1
        forward_id = state.forward_id
        phase = "prefill" if seq_len > 1 else "decode"
        previous_forward_id = state.current_forward_id
        previous_phase = state.current_phase
        state.current_forward_id = forward_id
        state.current_phase = phase
        event = {
            "forward_id": forward_id,
            "prefix": state.active_prefix,
            "phase": phase,
            "input_seq_len": seq_len,
            "input_ids_shape": int_shape(input_ids),
            "inputs_embeds_shape": int_shape(inputs_embeds),
        }
        state.forward_events.append(event)
        scope_name = f"{state.active_prefix}.forward_{phase}"
        try:
            with record_scope(scope_name):
                output = original_forward(*args, **kwargs)
            logits = getattr(output, "logits", None)
            if logits is None and isinstance(output, tuple) and output:
                logits = output[0]
            event["logits_shape"] = int_shape(logits)
            return output
        finally:
            state.current_forward_id = previous_forward_id
            state.current_phase = previous_phase

    model.forward = wrapped_model_forward

    base_model = model.get_model()
    for layer_idx, layer in enumerate(base_model.layers):
        original_layer_forward = layer.forward

        def make_layer_forward(orig: Any, idx: int):
            def wrapped_layer_forward(*args: Any, **kwargs: Any) -> Any:
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                position_ids = kwargs.get("position_ids")
                past_key_value = kwargs.get("past_key_value")
                if hidden_states is None:
                    return orig(*args, **kwargs)
                q_len = int(hidden_states.shape[1])
                past_len = safe_cache_len(past_key_value, q_len, idx)
                phase = state.current_phase or ("prefill" if q_len > 1 else "decode")
                event = {
                    "prefix": state.active_prefix,
                    "forward_id": state.current_forward_id,
                    "phase": phase,
                    "layer_idx": idx,
                    "q_len": q_len,
                    "past_len": past_len,
                    "kv_len": past_len + q_len,
                    "hidden_shape_in": int_shape(hidden_states),
                    "position_shape": int_shape(position_ids),
                    "important_visual_tokens_in": tensor_count(kwargs.get("important_vis_tokens")),
                    "exit_indicator_in": int(kwargs.get("exit_indicator", 0) or 0),
                }
                scope_name = f"{state.active_prefix}.layer{idx:02d}.{phase}"
                with record_scope(scope_name):
                    output = orig(*args, **kwargs)
                if isinstance(output, tuple) and output:
                    out_hidden = output[0]
                    if isinstance(out_hidden, tuple) and out_hidden:
                        out_hidden = out_hidden[0]
                    event["hidden_shape_out"] = int_shape(out_hidden)
                    try:
                        event["exit_indicator_out"] = int(output[-1] or 0)
                    except Exception:
                        event["exit_indicator_out"] = None
                state.layer_events.append(event)
                return output

            return wrapped_layer_forward

        layer.forward = make_layer_forward(original_layer_forward, layer_idx)

        if layer_profile:
            for component in ("self_attn", "mlp"):
                module = getattr(layer, component, None)
                if module is None or not hasattr(module, "forward"):
                    continue
                original_component_forward = module.forward
                label = "attn" if component == "self_attn" else "mlp"

                def make_component_forward(orig: Any, idx: int, label_name: str):
                    def wrapped_component_forward(*args: Any, **kwargs: Any) -> Any:
                        hidden_states = kwargs.get("hidden_states")
                        if hidden_states is None and args:
                            hidden_states = args[0]
                        if torch.is_tensor(hidden_states) and hidden_states.dim() >= 2:
                            phase = "prefill" if int(hidden_states.shape[1]) > 1 else "decode"
                        else:
                            phase = state.current_phase or "unknown"
                        scope_name = f"{state.active_prefix}.layer{idx:02d}.{phase}.{label_name}"
                        with record_scope(scope_name):
                            return orig(*args, **kwargs)

                    return wrapped_component_forward

                module.forward = make_component_forward(original_component_forward, layer_idx, label)

        attn = getattr(layer, "self_attn", None)
        if hasattr(attn, "value_aware_token_selection"):
            original_select = attn.value_aware_token_selection

            def make_wrapped_select(orig: Any, idx: int):
                def wrapped_select(*args: Any, **kwargs: Any) -> Any:
                    scope_name = f"{state.active_prefix}.value_aware_token_selection.layer{idx:02d}"
                    with record_scope(scope_name):
                        result = orig(*args, **kwargs)
                    event = {
                        "prefix": state.active_prefix,
                        "forward_id": state.current_forward_id,
                        "phase": state.current_phase,
                        "layer_idx": idx,
                        "value_states_shape": int_shape(args[0]) if len(args) > 0 else None,
                        "attn_output_shape": int_shape(args[1]) if len(args) > 1 else None,
                        "attn_weights_shape": int_shape(args[2]) if len(args) > 2 else None,
                        "input_important_visual_tokens": tensor_count(args[3]) if len(args) > 3 else None,
                    }
                    if torch.is_tensor(result):
                        event["result_type"] = "tensor"
                        event["selected_visual_token_count"] = int(result.numel())
                    elif isinstance(result, bool):
                        event["result_type"] = "bool"
                        event["deep_exit"] = bool(result)
                    elif result is None:
                        event["result_type"] = "none"
                        event["selected_visual_token_count"] = 0
                    else:
                        event["result_type"] = type(result).__name__
                    state.selection_events.append(event)
                    return result

                return wrapped_select

            attn.value_aware_token_selection = make_wrapped_select(original_select, layer_idx)


def run_request(
    *,
    state: ProfilerState,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    image_path: str,
    prompt: str,
    conv_mode: str,
    max_new_tokens: int,
    temperature: float,
    pruning_config: dict[str, Any] | None,
    prefix: str,
) -> dict[str, Any]:
    previous_prefix = state.active_prefix
    state.active_prefix = prefix
    try:
        with record_scope(f"{prefix}.build_prompt_tokenize"):
            prompt_text = build_prompt(prompt, conv_mode)
            input_ids = tokenizer_image_token(
                prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        with record_scope(f"{prefix}.load_image"):
            image = Image.open(image_path).convert("RGB")
            image_size = image.size

        with record_scope(f"{prefix}.image_preprocess_cpu"):
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]

        with record_scope(f"{prefix}.input_h2d"):
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            attention_mask = attention_mask.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        with record_scope(f"{prefix}.generate_total"):
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
        state.active_prefix = previous_prefix

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


def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def time_range_value(event: Any, name: str) -> Any:
    time_range = safe_attr(event, "time_range")
    if time_range is None:
        return None
    return safe_attr(time_range, name)


def stack_summary(stack: Any, *, max_items: int = 4) -> str:
    if not stack:
        return ""
    try:
        items = list(stack)
    except TypeError:
        return str(stack)
    compact = []
    for item in items[-max_items:]:
        compact.append(str(item).replace(str(ROOT_DIR), "$ROOT"))
    return " | ".join(compact)


def module_hierarchy(event: Any) -> str:
    value = safe_attr(event, "module_hierarchy")
    if value:
        return str(value)
    metadata = safe_attr(event, "event_metadata") or {}
    if isinstance(metadata, dict):
        for key in ("Module Hierarchy", "module_hierarchy", "Python module hierarchy"):
            if metadata.get(key):
                return str(metadata[key])
    return ""


def input_shapes(event: Any) -> str:
    shapes = safe_attr(event, "input_shapes")
    if shapes:
        return json_dumps(shapes)
    structured = safe_attr(event, "structured_input_shapes")
    if structured:
        return json_dumps(structured)
    return ""


def event_row(event: Any) -> dict[str, Any]:
    return {
        "id": safe_attr(event, "id"),
        "name": safe_attr(event, "name"),
        "key": safe_attr(event, "key"),
        "parent_id": safe_attr(safe_attr(event, "cpu_parent"), "id"),
        "parent_name": safe_attr(safe_attr(event, "cpu_parent"), "name"),
        "is_user_annotation": bool(safe_attr(event, "is_user_annotation", False)),
        "is_async": bool(safe_attr(event, "is_async", False)),
        "scope": safe_attr(event, "scope"),
        "cpu_start_us": time_range_value(event, "start"),
        "cpu_end_us": time_range_value(event, "end"),
        "cpu_time_total_us": safe_attr(event, "cpu_time_total", 0.0),
        "self_cpu_time_total_us": safe_attr(event, "self_cpu_time_total", 0.0),
        "device_time_total_us": safe_attr(event, "device_time_total", 0.0),
        "self_device_time_total_us": safe_attr(event, "self_device_time_total", 0.0),
        "cpu_memory_usage": safe_attr(event, "cpu_memory_usage", 0),
        "device_memory_usage": safe_attr(event, "device_memory_usage", 0),
        "input_shapes": input_shapes(event),
        "module_hierarchy": module_hierarchy(event),
        "stack": stack_summary(safe_attr(event, "stack")),
    }


def key_average_row(event: Any) -> dict[str, Any]:
    return {
        "key": safe_attr(event, "key"),
        "count": safe_attr(event, "count", 0),
        "cpu_time_total_us": safe_attr(event, "cpu_time_total", 0.0),
        "self_cpu_time_total_us": safe_attr(event, "self_cpu_time_total", 0.0),
        "device_time_total_us": safe_attr(event, "device_time_total", 0.0),
        "self_device_time_total_us": safe_attr(event, "self_device_time_total", 0.0),
        "cpu_memory_usage": safe_attr(event, "cpu_memory_usage", 0),
        "device_memory_usage": safe_attr(event, "device_memory_usage", 0),
        "input_shapes": json_dumps(safe_attr(event, "input_shapes", "")),
        "stack": stack_summary(safe_attr(event, "stack")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = {key for row in rows for key in row}
    if field_order is None:
        fields = sorted(keys)
    else:
        fields = [field for field in field_order if field in keys]
        fields.extend(sorted(keys - set(fields)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_descendant_of(event: Any, parent: Any) -> bool:
    cursor = safe_attr(event, "cpu_parent")
    parent_id = safe_attr(parent, "id")
    while cursor is not None:
        if safe_attr(cursor, "id") == parent_id:
            return True
        cursor = safe_attr(cursor, "cpu_parent")
    return False


def aggregate_child_ops(events: list[Any], scope_event: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        name = str(safe_attr(event, "name", ""))
        if not name.startswith("aten::"):
            continue
        if not is_descendant_of(event, scope_event):
            continue
        row = grouped.setdefault(
            name,
            {
                "op": name,
                "count": 0,
                "cpu_time_total_us": 0.0,
                "device_time_total_us": 0.0,
                "example_input_shapes": "",
                "example_stack": "",
            },
        )
        row["count"] += 1
        row["cpu_time_total_us"] += float(safe_attr(event, "cpu_time_total", 0.0) or 0.0)
        row["device_time_total_us"] += float(safe_attr(event, "device_time_total", 0.0) or 0.0)
        if not row["example_input_shapes"]:
            row["example_input_shapes"] = input_shapes(event)
        if not row["example_stack"]:
            row["example_stack"] = stack_summary(safe_attr(event, "stack"))
    return sorted(
        grouped.values(),
        key=lambda item: (float(item["device_time_total_us"]), float(item["cpu_time_total_us"])),
        reverse=True,
    )


def record_scope_rows(events: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        name = str(safe_attr(event, "name", ""))
        if not name.startswith(("profile.", "warmup.")):
            continue
        child_ops = aggregate_child_ops(events, event)
        rows.append({
            **event_row(event),
            "child_aten_op_count": sum(int(row["count"]) for row in child_ops),
            "top_child_aten_ops": "; ".join(
                f"{row['op']} x{row['count']}" for row in child_ops[:8]
            ),
        })
    return rows


def module_stack_rows(events: list[Any]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for event in events:
        name = str(safe_attr(event, "name", ""))
        if not name.startswith("aten::"):
            continue
        counter[(module_hierarchy(event), stack_summary(safe_attr(event, "stack"), max_items=2))] += 1
    rows = [
        {
            "module_hierarchy": module,
            "stack_tail": stack,
            "aten_op_count": count,
        }
        for (module, stack), count in counter.most_common()
    ]
    return rows


def find_scope(events: list[Any], name: str) -> Any | None:
    for event in events:
        if safe_attr(event, "name") == name:
            return event
    return None


def choose_layer_event(layer_events: list[dict[str, Any]], target_layer: int) -> dict[str, Any] | None:
    measured = [row for row in layer_events if row.get("prefix") == "profile"]
    for row in measured:
        if int(row.get("layer_idx", -1)) == target_layer and row.get("phase") == "prefill":
            return row
    return measured[0] if measured else None


def write_process_view(
    path: Path,
    *,
    metadata: dict[str, Any],
    request: dict[str, Any],
    state: ProfilerState,
    events: list[Any],
    target_layer: int,
) -> None:
    layer_event = choose_layer_event(state.layer_events, target_layer)
    lines: list[str] = []
    lines.append("# Torch Profiler Process View")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- tag: `{metadata['tag']}`")
    lines.append(f"- config: `{metadata['config']}`")
    lines.append(f"- torch: `{metadata['environment']['torch']}`")
    lines.append("- profiler options: `record_shapes=True`, `with_stack=True`, `with_modules=True`")
    lines.append(f"- output text: `{request.get('output_text', '')}`")
    lines.append("")
    lines.append("## Evidence Boundary")
    lines.append("")
    lines.append("This file is generated from `torch.profiler` events plus explicit `record_function` scopes.")
    lines.append("It is a process sketch, not an FX graph and not a TorchDispatch tensor-id reconstruction.")
    lines.append("The profiler can show event timing, shapes, stacks, and Chrome trace structure, but it does not")
    lines.append("provide producer-consumer tensor ids or complete alias/inplace evidence.")
    lines.append("")

    measured_forwards = [row for row in state.forward_events if row.get("prefix") == "profile"]
    measured_layers = [row for row in state.layer_events if row.get("prefix") == "profile"]
    measured_selections = [row for row in state.selection_events if row.get("prefix") == "profile"]
    lines.append("## Observed High-Level Schedule")
    lines.append("")
    lines.append(f"- measured forward events: `{len(measured_forwards)}`")
    lines.append(f"- measured layer events: `{len(measured_layers)}`")
    lines.append(f"- measured selection events: `{len(measured_selections)}`")
    if measured_layers:
        prefill = [row for row in measured_layers if row.get("phase") == "prefill"]
        decode = [row for row in measured_layers if row.get("phase") == "decode"]
        lines.append(f"- prefill layers: `{len(prefill)}`")
        lines.append(f"- decode layer calls: `{len(decode)}`")
    lines.append("")

    scope_rows = record_scope_rows(events)
    lines.append("## Top `record_function` Scopes")
    lines.append("")
    lines.append("| scope | cpu total ms | device total ms | child ATen ops | top child ops |")
    lines.append("|---|---:|---:|---:|---|")
    for row in sorted(scope_rows, key=lambda item: float(item.get("cpu_time_total_us") or 0.0), reverse=True)[:20]:
        lines.append(
            "| {scope} | {cpu:.3f} | {device:.3f} | {count} | {ops} |".format(
                scope=row.get("name", ""),
                cpu=float(row.get("cpu_time_total_us") or 0.0) / 1000.0,
                device=float(row.get("device_time_total_us") or 0.0) / 1000.0,
                count=row.get("child_aten_op_count", 0),
                ops=str(row.get("top_child_aten_ops", "")).replace("|", "\\|"),
            )
        )
    lines.append("")

    if layer_event is not None:
        layer_idx = int(layer_event["layer_idx"])
        phase = str(layer_event["phase"])
        q_len = int(layer_event["q_len"])
        kv_len = int(layer_event["kv_len"])
        hidden_shape = layer_event.get("hidden_shape_in") or []
        hidden = hidden_shape[-1] if hidden_shape else "Hidden"
        layer_scope_name = f"profile.layer{layer_idx:02d}.{phase}"
        attn_scope_name = f"profile.layer{layer_idx:02d}.{phase}.attn"
        mlp_scope_name = f"profile.layer{layer_idx:02d}.{phase}.mlp"
        lines.append(f"## Layer {layer_idx} `{phase}` Profiler-Derived Process Sketch")
        lines.append("")
        lines.append(f"- layer scope: `{layer_scope_name}`")
        lines.append(f"- observed q_len: `{q_len}`")
        lines.append(f"- observed kv_len: `{kv_len}`")
        lines.append(f"- hidden shape in: `{hidden_shape}`")
        lines.append("")
        lines.append("### Tensor-Axis Sketch")
        lines.append("")
        lines.append("```text")
        lines.append(f"Token axis S={q_len} (compressed)                     Hidden dimension")
        lines.append(f"                                                       0                                      {hidden}")
        lines.append("                                                       ▲                                        ▲")
        lines.append("hidden input      profile.layer scope            ──▶   ┌────────────────────────────────────────┐")
        lines.append("                                                       │ HIDDEN_ROWS_BEFORE_LAYER               │")
        lines.append("                                                       │ HIDDEN_ROWS_BEFORE_LAYER               │")
        lines.append("                                                       │ HIDDEN_ROWS_BEFORE_LAYER               │")
        lines.append("                                                       └────────────────────────────────────────┘")
        lines.append("self attention    profile.layer.attn             ──▶   ┌────────────────────────────────────────┐")
        lines.append("                                                       │ QKV_ROPE_ATTENTION_OUTPUT              │")
        lines.append("                                                       │ scope groups profiler events, not DAG   │")
        lines.append("                                                       └────────────────────────────────────────┘")
        lines.append("residual + norm   inferred layer body             ──▶   [RESIDUAL_ADD_AND_POST_ATTN_RMSNORM]")
        lines.append("mlp               profile.layer.mlp              ──▶   ┌────────────────────────────────────────┐")
        lines.append("                                                       │ GATE_UP_SILU_DOWN_ROWS                 │")
        lines.append("                                                       │ profiler gives events, not data edges   │")
        lines.append("                                                       └────────────────────────────────────────┘")
        lines.append("layer output      layer return                    ──▶   [HIDDEN_ROWS_AFTER_LAYER]")
        lines.append("```")
        lines.append("")

        for scope_name, title in (
            (layer_scope_name, "Layer Total"),
            (attn_scope_name, "Attention Scope"),
            (mlp_scope_name, "MLP Scope"),
        ):
            scope = find_scope(events, scope_name)
            if scope is None:
                continue
            child_ops = aggregate_child_ops(events, scope)
            lines.append(f"### {title}: Top Child ATen Ops")
            lines.append("")
            lines.append("| op | count | cpu total ms | device total ms | example input shapes |")
            lines.append("|---|---:|---:|---:|---|")
            for row in child_ops[:20]:
                lines.append(
                    "| {op} | {count} | {cpu:.3f} | {device:.3f} | `{shapes}` |".format(
                        op=str(row["op"]).replace("|", "\\|"),
                        count=row["count"],
                        cpu=float(row["cpu_time_total_us"]) / 1000.0,
                        device=float(row["device_time_total_us"]) / 1000.0,
                        shapes=str(row.get("example_input_shapes", "")).replace("`", "'"),
                    )
                )
            lines.append("")

    lines.append("## What `with_stack` Produced")
    lines.append("")
    nonempty_stack_events = [event for event in events if stack_summary(safe_attr(event, "stack"))]
    nonempty_aten_stack_events = [
        event
        for event in events
        if str(safe_attr(event, "name", "")).startswith("aten::") and stack_summary(safe_attr(event, "stack"))
    ]
    lines.append(f"- profiler events with non-empty stack metadata: `{len(nonempty_stack_events)}`")
    lines.append(f"- ATen events with non-empty stack metadata: `{len(nonempty_aten_stack_events)}`")
    lines.append(f"- total profiler events: `{len(events)}`")
    lines.append("")
    if nonempty_stack_events:
        lines.append("| event | stack tail |")
        lines.append("|---|---|")
        for event in nonempty_stack_events[:20]:
            lines.append(
                "| {name} | {stack} |".format(
                    name=str(safe_attr(event, "name", "")).replace("|", "\\|"),
                    stack=stack_summary(safe_attr(event, "stack")).replace("|", "\\|"),
                )
            )
    else:
        lines.append("No non-empty Python stack metadata was observed in this eager run.")
        lines.append("The option was enabled, but it did not provide a usable source-location")
        lines.append("mapping for the ATen events exported by this local profiler path.")
    lines.append("")

    lines.append("## What `with_modules` Produced")
    lines.append("")
    module_rows = module_stack_rows(events)
    nonempty_modules = [row for row in module_rows if row.get("module_hierarchy")]
    lines.append(f"- ATen module-hierarchy rows with non-empty module metadata: `{len(nonempty_modules)}`")
    lines.append(f"- total module/stack summary rows: `{len(module_rows)}`")
    lines.append("")
    if nonempty_modules:
        lines.append("| module hierarchy | ATen op count | stack tail |")
        lines.append("|---|---:|---|")
        for row in nonempty_modules[:20]:
            lines.append(
                "| {module} | {count} | {stack} |".format(
                    module=str(row["module_hierarchy"]).replace("|", "\\|"),
                    count=row["aten_op_count"],
                    stack=str(row["stack_tail"]).replace("|", "\\|"),
                )
            )
    else:
        lines.append("No non-empty module hierarchy was observed for ATen events in this eager run.")
        lines.append("This matches the PyTorch profiler documentation caveat that `with_modules`")
        lines.append("is TorchScript-oriented and is not a reliable eager-module ownership source.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="visipruner-full")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--layer-profile", action="store_true", default=True)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-flops", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.config}_{args.max_new_tokens}tok_stack_modules_{timestamp}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, image_processor, context_len = load_model(config, args.model_path, args.model_base)
    state = ProfilerState()
    patch_model_for_profiler(model, state, layer_profile=args.layer_profile)

    for _ in range(args.warmup_iters):
        run_request(
            state=state,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=args.image_path,
            prompt=args.prompt,
            conv_mode=args.conv_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            pruning_config=config["pruning_config"],
            prefix="warmup",
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=True,
        with_modules=True,
        with_flops=args.with_flops,
        acc_events=True,
    ) as prof:
        with record_scope("profile.request"):
            request = run_request(
                state=state,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=args.image_path,
                prompt=args.prompt,
                conv_mode=args.conv_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pruning_config=config["pruning_config"],
                prefix="profile",
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    chrome_path = out_dir / "chrome_trace.json"
    prof.export_chrome_trace(str(chrome_path))
    events = list(prof.events())
    key_averages = list(prof.key_averages())

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "config": args.config,
        "description": config["description"],
        "model_path": args.model_path,
        "model_base": args.model_base,
        "image_path": args.image_path,
        "prompt": args.prompt,
        "conv_mode": args.conv_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "warmup_iters": args.warmup_iters,
        "layer_profile": args.layer_profile,
        "target_layer": args.target_layer,
        "profiler_options": {
            "record_shapes": True,
            "profile_memory": args.profile_memory,
            "with_stack": True,
            "with_modules": True,
            "with_flops": args.with_flops,
            "acc_events": True,
        },
        "context_len": context_len,
        "loaded_model_class": model.__class__.__name__,
        "loaded_base_model_class": model.get_model().__class__.__name__,
        "request": request,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_home": os.environ.get("HF_HOME"),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "artifacts": {
            "chrome_trace": str(chrome_path),
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out_dir / "profiler_events.csv", [event_row(event) for event in events])
    write_csv(out_dir / "profiler_key_averages.csv", [key_average_row(event) for event in key_averages])
    write_csv(out_dir / "record_function_scopes.csv", record_scope_rows(events))
    write_csv(out_dir / "module_stack_summary.csv", module_stack_rows(events))
    write_csv(out_dir / "forward_events.csv", state.forward_events)
    write_csv(out_dir / "layer_events.csv", state.layer_events)
    write_csv(out_dir / "selection_events.csv", state.selection_events)
    write_process_view(
        out_dir / "process_view.md",
        metadata=metadata,
        request=request,
        state=state,
        events=events,
        target_layer=args.target_layer,
    )
    (Path(args.output_dir) / "latest_torch_profile_trace_path.txt").write_text(str(out_dir) + "\n", encoding="utf-8")

    print(json.dumps({
        "tag": tag,
        "out_dir": str(out_dir),
        "events": len(events),
        "key_averages": len(key_averages),
        "output_text": request.get("output_text", ""),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
