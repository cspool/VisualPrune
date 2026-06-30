#!/usr/bin/env python3
"""Run a VisiPrune-centric filtered TorchDispatch profile.

This script intentionally does not wrap ``model.generate()`` in a global
TorchDispatchMode. It first builds a dispatch manifest from an existing
algorithmic trace, then opens TorchDispatchMode only while selected
``(forward_id, layer_id)`` layer forwards are executing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import weakref
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOAD_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = WORKLOAD_DIR.parent
REPO_DIR = ROOT_DIR / "repo"
TRACE_SCRIPT_DIR = WORKLOAD_DIR / "algorithmic_trace/tools"
DEFAULT_TRACE = (
    WORKLOAD_DIR
    / "algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/algorithmic_trace.json"
)
DEFAULT_OUTPUT_DIR = WORKLOAD_DIR / "dispatch/profiles"


def parse_gpu_early(argv: list[str]) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--gpu" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return "0"


os.environ.setdefault("CUDA_VISIBLE_DEVICES", parse_gpu_early(sys.argv))
os.environ.setdefault("HF_HOME", str(ROOT_DIR / "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(TRACE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_SCRIPT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image
from torch.utils._python_dispatch import TorchDispatchMode

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from visipruner_algorithmic_trace import (  # noqa: E402
    CONFIGS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_PATH,
    build_prompt,
    int_shape,
    safe_cache_len,
)


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

DISPATCH_OP_FIELD_ORDER = [
    "args",
    "event_id",
    "kv_len",
    "kwargs",
    "module_class",
    "op_schema",
    "outputs",
    "past_len",
    "phase",
    "q_len",
    "token_state",
    "visipruner_role",
    "input_tensor_ids",
    "output_tensor_ids",
]


@dataclass
class DispatchState:
    forward_id: int = 0
    current_forward_id: int | None = None
    current_phase: str | None = None
    current_event: dict[str, Any] | None = None
    module_stack: list[dict[str, Any]] = field(default_factory=list)
    forward_events: list[dict[str, Any]] = field(default_factory=list)
    layer_events: list[dict[str, Any]] = field(default_factory=list)
    op_events: list[dict[str, Any]] = field(default_factory=list)
    event_op_counters: Counter[str] = field(default_factory=Counter)
    module_op_counters: Counter[str] = field(default_factory=Counter)
    module_call_counter: int = 0
    global_op_index: int = 0
    tensor_id_counter: int = 0
    tensor_refs: dict[int, tuple[Any, str]] = field(default_factory=dict)
    tensor_producers: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def describe_value(value: Any, depth: int = 0) -> Any:
    if torch.is_tensor(value):
        return {
            "type": "Tensor",
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, (list, tuple)):
        return [describe_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): describe_value(item, depth + 1) for key, item in value.items()}
    return type(value).__name__


def safe_int_attr(value: Any, name: str) -> int | None:
    try:
        result = getattr(value, name)()
    except Exception:
        return None
    try:
        return int(result)
    except Exception:
        return None


def safe_stride(value: torch.Tensor) -> list[int] | None:
    try:
        return [int(item) for item in value.stride()]
    except Exception:
        return None


def safe_storage_data_ptr(value: torch.Tensor) -> int | None:
    try:
        storage = value.untyped_storage()
        return int(storage.data_ptr())
    except Exception:
        return None


def tensor_runtime_id(state: DispatchState, value: torch.Tensor) -> str:
    object_id = id(value)
    entry = state.tensor_refs.get(object_id)
    if entry is not None:
        ref, runtime_id = entry
        if ref is None:
            return runtime_id
        try:
            if ref() is value:
                return runtime_id
        except Exception:
            pass

    state.tensor_id_counter += 1
    runtime_id = f"t{state.tensor_id_counter:08d}"
    try:
        ref = weakref.ref(value)
    except TypeError:
        ref = None
    state.tensor_refs[object_id] = (ref, runtime_id)
    return runtime_id


def describe_tensor_ref(state: DispatchState, value: torch.Tensor, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "tensor_id": tensor_runtime_id(state, value),
        "python_object_id": id(value),
        "tensor_impl_id": int(getattr(value, "_cdata", 0) or 0),
        "shape": [int(x) for x in value.shape],
        "stride": safe_stride(value),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "layout": str(value.layout).replace("torch.", ""),
        "requires_grad": bool(value.requires_grad),
        "data_ptr": safe_int_attr(value, "data_ptr"),
        "storage_data_ptr": safe_storage_data_ptr(value),
        "storage_offset": safe_int_attr(value, "storage_offset"),
    }


def collect_tensor_refs(state: DispatchState, value: Any, path: str, depth: int = 0) -> list[dict[str, Any]]:
    if torch.is_tensor(value):
        return [describe_tensor_ref(state, value, path)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return []
    if depth >= 8:
        return []
    if isinstance(value, tuple):
        refs: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            refs.extend(collect_tensor_refs(state, item, f"{path}[{index}]", depth + 1))
        return refs
    if isinstance(value, list):
        refs = []
        for index, item in enumerate(value):
            refs.extend(collect_tensor_refs(state, item, f"{path}[{index}]", depth + 1))
        return refs
    if isinstance(value, dict):
        refs = []
        for key, item in value.items():
            refs.extend(collect_tensor_refs(state, item, f"{path}[{json.dumps(str(key), ensure_ascii=False)}]", depth + 1))
        return refs
    return []


def input_tensor_producers(state: DispatchState, input_tensors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    producers: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for tensor in input_tensors:
        tensor_id = str(tensor["tensor_id"])
        producer = state.tensor_producers.get(tensor_id)
        if producer is None:
            producers.append({
                "input_path": tensor["path"],
                "tensor_id": tensor_id,
                "producer": None,
                "source": "external_or_before_captured_scope",
            })
            continue
        item = {
            "input_path": tensor["path"],
            "tensor_id": tensor_id,
            "producer": producer,
            "source": "observed_in_captured_scope",
        }
        producers.append(item)
        edges.append({
            "tensor_id": tensor_id,
            "producer_event_detail_id": producer["event_detail_id"],
            "producer_event_id": producer["event_id"],
            "producer_event_op_index": producer["event_op_index"],
            "producer_global_op_index": producer["global_op_index"],
            "producer_op_name": producer["op_name"],
            "producer_output_path": producer["output_path"],
            "consumer_input_path": tensor["path"],
        })
    return producers, edges


def input_output_aliases(
    input_tensors: list[dict[str, Any]],
    output_tensors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    for input_tensor in input_tensors:
        for output_tensor in output_tensors:
            alias_types = []
            if input_tensor.get("tensor_id") == output_tensor.get("tensor_id"):
                alias_types.append("same_runtime_tensor_id")
            if (
                input_tensor.get("storage_data_ptr") is not None
                and input_tensor.get("storage_data_ptr") == output_tensor.get("storage_data_ptr")
                and input_tensor.get("device") == output_tensor.get("device")
            ):
                alias_types.append("same_storage_data_ptr")
            if not alias_types:
                continue
            aliases.append({
                "input_path": input_tensor["path"],
                "output_path": output_tensor["path"],
                "input_tensor_id": input_tensor["tensor_id"],
                "output_tensor_id": output_tensor["tensor_id"],
                "alias_types": alias_types,
            })
    return aliases


def op_name(func: Any) -> str:
    name = getattr(func, "__name__", None)
    if name:
        return str(name)
    return str(func)


def op_schema(func: Any) -> str:
    schema = getattr(func, "_schema", None)
    if schema is not None:
        return str(schema)
    return str(func)


def event_id(forward_id: int, layer_id: int) -> str:
    return f"input{forward_id}_layer{layer_id}"


def module_scope(module_relative_path: str, module_path: str) -> str:
    return module_relative_path or module_path


def relative_module_path(module_path: str, layer_module_path: str) -> str:
    if not layer_module_path:
        return module_path
    if module_path == layer_module_path:
        return ""
    prefix = f"{layer_module_path}."
    if module_path.startswith(prefix):
        return module_path[len(prefix):]
    return module_path


def module_forward_source(module: Any) -> dict[str, Any]:
    forward = getattr(module, "forward", None)
    code = getattr(forward, "__code__", None)
    if code is None:
        code = getattr(getattr(forward, "__func__", None), "__code__", None)
    if code is None:
        return {"module_forward_file": "", "module_forward_lineno": ""}
    return {
        "module_forward_file": str(code.co_filename),
        "module_forward_lineno": int(code.co_firstlineno),
    }


def push_module_context(
    state: DispatchState,
    module_path: str,
    module_type: str,
    module_class: str,
    forward_source: dict[str, Any],
) -> None:
    current = state.current_event or {}
    module_relative_path = relative_module_path(
        module_path,
        str(current.get("layer_module_path") or ""),
    )
    parent = state.module_stack[-1] if state.module_stack else {}
    state.module_call_counter += 1
    state.module_stack.append({
        "module_call_id": state.module_call_counter,
        "module_parent_call_id": parent.get("module_call_id", ""),
        "module_path": module_path,
        "module_relative_path": module_relative_path,
        "module_type": module_type,
        "module_class": module_class,
        "module_forward_file": forward_source.get("module_forward_file", ""),
        "module_forward_lineno": forward_source.get("module_forward_lineno", ""),
    })


def make_event_detail_id(
    event_key: str,
    module_relative_path: str,
    module_path: str,
    module_call_id: int,
    module_op_index: int,
) -> str:
    component = module_relative_path or module_path or "layer"
    safe_component = component.replace(".", "/")
    return f"{event_key}/{safe_component}/call{module_call_id:04d}/op{module_op_index:04d}"


def add_manifest_row(
    rows: dict[tuple[int, int], dict[str, Any]],
    layer_by_key: dict[tuple[int, int], dict[str, Any]],
    key: tuple[int, int],
    priority: str,
    role: str,
    reason: str,
    selection_result: str = "",
) -> None:
    if key not in layer_by_key:
        return
    layer = layer_by_key[key]
    row = rows.get(key)
    if row is None:
        row = {
            "event_id": event_id(key[0], key[1]),
            "input_id": key[0],
            "layer_id": key[1],
            "phase": layer.get("phase"),
            "q_len": layer.get("q_len"),
            "past_len": layer.get("past_len"),
            "kv_len": layer.get("kv_len"),
            "priority": priority,
            "visipruner_role": role,
            "selection_result": selection_result,
            "token_state": token_state_for_layer(layer),
            "reason": reason,
            "repeat_group": "",
            "repeat_members": "",
            "keep_default": True,
        }
        rows[key] = row
        return
    if PRIORITY_ORDER[priority] < PRIORITY_ORDER[row["priority"]]:
        row["priority"] = priority
    row["visipruner_role"] = merge_text(row["visipruner_role"], role)
    row["selection_result"] = merge_text(row["selection_result"], selection_result)
    row["reason"] = merge_text(row["reason"], reason)


def merge_text(left: str, right: str) -> str:
    if not right:
        return left
    parts = [part for part in left.split(";") if part]
    if right not in parts:
        parts.append(right)
    return ";".join(parts)


def token_state_for_layer(layer: dict[str, Any]) -> str:
    phase = layer.get("phase")
    q_len = int(layer.get("q_len") or 0)
    layer_id = int(layer.get("layer_idx") or layer.get("layer_id") or 0)
    if phase == "prefill":
        if q_len >= 100:
            return "full_visual"
        if layer_id >= 28:
            return "deep_removed"
        return "middle_pruned"
    if phase == "decode":
        past_len = int(layer.get("past_len") or 0)
        if layer_id <= 18 and past_len >= 100:
            return "full_cache"
        if layer_id <= 27:
            return "middle_pruned_cache"
        return "deep_removed_cache"
    return ""


def build_dispatch_manifest(
    trace_payload: dict[str, Any],
    priorities: set[str],
    include_decode_effect: bool,
    include_shallow: bool,
) -> list[dict[str, Any]]:
    layer_events = trace_payload.get("layer_events") or []
    selection_events = trace_payload.get("selection_events") or []
    layer_by_key = {
        (int(row["forward_id"]), int(row["layer_idx"])): row
        for row in layer_events
        if row.get("forward_id") is not None and row.get("layer_idx") is not None
    }
    rows: dict[tuple[int, int], dict[str, Any]] = {}

    if "P0" in priorities:
        for sel in selection_events:
            if sel.get("phase") != "prefill":
                continue
            key = (int(sel["forward_id"]), int(sel["layer_idx"]))
            result = str(sel.get("result_type") or "")
            deep_exit = sel.get("deep_exit")
            if result == "none":
                role = "middle_probe"
                reason = "value_aware_token_selection called and returned none"
                selection_result = "none"
            elif result == "tensor":
                role = "middle_select"
                count = sel.get("selected_visual_token_count")
                reason = f"value_aware_token_selection selected {count} visual tokens"
                selection_result = "tensor"
            elif result == "bool":
                role = "deep_check"
                reason = f"deep exit check returned {deep_exit}"
                selection_result = str(deep_exit).lower()
            else:
                role = "selection_event"
                reason = f"selection event result_type={result}"
                selection_result = result
            add_manifest_row(rows, layer_by_key, key, "P0", role, reason, selection_result)

    prefill = sorted(
        [row for row in layer_events if row.get("phase") == "prefill"],
        key=lambda row: int(row["layer_idx"]),
    )
    if "P1" in priorities and prefill:
        for prev, cur in zip(prefill, prefill[1:]):
            prev_q = int(prev.get("q_len") or 0)
            cur_q = int(cur.get("q_len") or 0)
            prev_kv = int(prev.get("kv_len") or 0)
            cur_kv = int(cur.get("kv_len") or 0)
            if prev_q != cur_q or prev_kv != cur_kv:
                prev_key = (int(prev["forward_id"]), int(prev["layer_idx"]))
                cur_key = (int(cur["forward_id"]), int(cur["layer_idx"]))
                reason = f"token schedule boundary {prev_q}/{prev_kv} -> {cur_q}/{cur_kv}"
                add_manifest_row(rows, layer_by_key, prev_key, "P1", "boundary_before_prune", reason)
                add_manifest_row(rows, layer_by_key, cur_key, "P1", "boundary_after_prune", reason)

    if include_shallow and "P2" in priorities and prefill:
        pruning_config = trace_payload.get("pruning_config") or {}
        shallow_mid = int(pruning_config.get("shallow_mid_layer", 6))
        shallow_candidates = [0, max(0, shallow_mid - 1), shallow_mid]
        for layer_id in sorted(set(shallow_candidates)):
            add_manifest_row(
                rows,
                layer_by_key,
                (1, layer_id),
                "P2",
                "shallow_or_boundary",
                "representative shallow-mask/control-boundary layer",
            )

    if include_decode_effect and "P3" in priorities:
        decode_forward_ids = sorted({
            int(row["forward_id"])
            for row in layer_events
            if row.get("phase") == "decode" and row.get("forward_id") is not None
        })
        prefill_groups = contiguous_prefill_groups(prefill)
        decode_layers: set[int] = set()
        if prefill_groups:
            decode_layers.add(prefill_groups[0][-1])
        if len(prefill_groups) >= 2:
            decode_layers.add(prefill_groups[1][0])
            decode_layers.add(prefill_groups[1][-1])
        if len(prefill_groups) >= 3:
            decode_layers.add(prefill_groups[2][0])
            decode_layers.add(prefill_groups[2][-1])
        if decode_forward_ids:
            for forward_id in sorted({decode_forward_ids[0], decode_forward_ids[-1]}):
                for layer_id in sorted(decode_layers):
                    add_manifest_row(
                        rows,
                        layer_by_key,
                        (forward_id, layer_id),
                        "P3",
                        "decode_prune_effect",
                        "first/last decode representative for pruned KV-cache regime",
                    )

    annotate_repeats(rows)
    return sorted(rows.values(), key=lambda row: (int(row["input_id"]), int(row["layer_id"])))


def contiguous_prefill_groups(prefill: list[dict[str, Any]]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_key: tuple[int, int] | None = None
    for row in prefill:
        key = (int(row.get("q_len") or 0), int(row.get("kv_len") or 0))
        layer_id = int(row["layer_idx"])
        if current_key is None or key == current_key:
            current.append(layer_id)
        else:
            groups.append(current)
            current = [layer_id]
        current_key = key
    if current:
        groups.append(current)
    return groups


def annotate_repeats(rows: dict[tuple[int, int], dict[str, Any]]) -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows.values():
        group_key = "|".join([
            str(row["phase"]),
            str(row["token_state"]),
            str(row["visipruner_role"]),
            str(row["selection_result"]),
            str(row["q_len"]),
            str(row["kv_len"]),
        ])
        groups[group_key].append(str(row["event_id"]))
        row["repeat_group"] = group_key
    for row in rows.values():
        members = groups.get(str(row["repeat_group"]), [])
        row["repeat_members"] = ",".join(members)


class RecordingDispatchMode(TorchDispatchMode):
    def __init__(self, state: DispatchState, record_elapsed_us: bool = False) -> None:
        super().__init__()
        self.state = state
        self.record_elapsed_us = record_elapsed_us

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        kwargs = kwargs or {}
        current = self.state.current_event
        if current is None:
            return func(*args, **kwargs)

        input_tensors = (
            collect_tensor_refs(self.state, args, "args")
            + collect_tensor_refs(self.state, kwargs, "kwargs")
        )
        start = time.perf_counter() if self.record_elapsed_us else None
        result = func(*args, **kwargs)
        elapsed_us = None
        if start is not None:
            elapsed_us = int((time.perf_counter() - start) * 1_000_000)

        name = op_name(func)
        schema = op_schema(func)
        args_desc = describe_value(args)
        kwargs_desc = describe_value(kwargs)
        outputs_desc = describe_value(result)
        output_tensors = collect_tensor_refs(self.state, result, "outputs")
        producer_records, observed_edges = input_tensor_producers(self.state, input_tensors)
        aliases = input_output_aliases(input_tensors, output_tensors)
        module_context = self.state.module_stack[-1] if self.state.module_stack else {}
        module_path = str(module_context.get("module_path") or "")
        module_relative_path = str(module_context.get("module_relative_path") or "")
        module_type = str(module_context.get("module_type") or "")
        module_class = str(module_context.get("module_class") or "")
        module_forward_file = str(module_context.get("module_forward_file") or "")
        module_forward_lineno = module_context.get("module_forward_lineno", "")
        module_call_id = int(module_context.get("module_call_id") or 0)
        module_parent_call_id = module_context.get("module_parent_call_id", "")
        runtime_module_scope = module_scope(module_relative_path, module_path)

        self.state.global_op_index += 1
        event_key = str(current["event_id"])
        self.state.event_op_counters[event_key] += 1
        module_key = "|".join([event_key, str(module_call_id), runtime_module_scope])
        self.state.module_op_counters[module_key] += 1
        module_op_index = self.state.module_op_counters[module_key]
        event_detail = make_event_detail_id(
            event_key,
            module_relative_path,
            module_path,
            module_call_id,
            module_op_index,
        )
        for edge in observed_edges:
            edge.update({
                "consumer_event_detail_id": event_detail,
                "consumer_event_id": event_key,
                "consumer_event_op_index": self.state.event_op_counters[event_key],
                "consumer_global_op_index": self.state.global_op_index,
                "consumer_op_name": name,
            })
        row = {
            "event_id": event_key,
            "layer_event_id": event_key,
            "event_detail_id": event_detail,
            "input_id": current["input_id"],
            "layer_id": current["layer_id"],
            "layer_module_path": current.get("layer_module_path", ""),
            "phase": current["phase"],
            "priority": current["priority"],
            "visipruner_role": current["visipruner_role"],
            "token_state": current["token_state"],
            "q_len": current["q_len"],
            "past_len": current["past_len"],
            "kv_len": current["kv_len"],
            "global_op_index": self.state.global_op_index,
            "event_op_index": self.state.event_op_counters[event_key],
            "module_op_index": module_op_index,
            "module_path": module_path,
            "module_relative_path": module_relative_path,
            "module_type": module_type,
            "module_class": module_class,
            "module_forward_file": module_forward_file,
            "module_forward_lineno": module_forward_lineno,
            "module_call_id": module_call_id,
            "module_parent_call_id": module_parent_call_id,
            "module_depth": len(self.state.module_stack),
            "module_stack": compact_json([
                {
                    "module_call_id": item.get("module_call_id"),
                    "module_relative_path": item.get("module_relative_path"),
                    "module_path": item.get("module_path"),
                    "module_type": item.get("module_type"),
                    "module_class": item.get("module_class"),
                    "module_forward_file": item.get("module_forward_file"),
                    "module_forward_lineno": item.get("module_forward_lineno"),
                }
                for item in self.state.module_stack
            ]),
            "runtime_module_scope": runtime_module_scope,
            "op_name": name,
            "op_schema": schema,
            "args": compact_json(args_desc),
            "kwargs": compact_json(kwargs_desc),
            "outputs": compact_json(outputs_desc),
            "input_tensors": compact_json(input_tensors),
            "output_tensors": compact_json(output_tensors),
            "input_tensor_ids": compact_json([item["tensor_id"] for item in input_tensors]),
            "output_tensor_ids": compact_json([item["tensor_id"] for item in output_tensors]),
            "input_tensor_producers": compact_json(producer_records),
            "observed_tensor_edges": compact_json(observed_edges),
            "input_output_tensor_aliases": compact_json(aliases),
        }
        if elapsed_us is not None:
            row["python_dispatch_elapsed_us"] = elapsed_us
        self.state.op_events.append(row)
        for output_tensor in output_tensors:
            self.state.tensor_producers[str(output_tensor["tensor_id"])] = {
                "event_detail_id": event_detail,
                "event_id": event_key,
                "event_op_index": self.state.event_op_counters[event_key],
                "global_op_index": self.state.global_op_index,
                "op_name": name,
                "op_schema": schema,
                "output_path": output_tensor["path"],
                "module_path": module_path,
                "module_relative_path": module_relative_path,
                "module_call_id": module_call_id,
                "module_op_index": module_op_index,
            }
        return result


def pop_module_context(state: DispatchState, module_path: str) -> None:
    if not state.module_stack:
        return
    if state.module_stack[-1].get("module_path") == module_path:
        state.module_stack.pop()
        return
    for idx in range(len(state.module_stack) - 1, -1, -1):
        if state.module_stack[idx].get("module_path") == module_path:
            del state.module_stack[idx:]
            return


def register_module_context_hooks(model: Any, state: DispatchState) -> dict[int, str]:
    module_paths: dict[int, str] = {}
    for module_path, module in model.named_modules():
        if not module_path:
            continue
        module_paths[id(module)] = module_path
        module_type = type(module).__name__
        module_class = f"{type(module).__module__}.{type(module).__qualname__}"
        forward_source = module_forward_source(module)

        def make_pre_hook(
            path: str,
            typ: str,
            cls: str,
            source: dict[str, Any],
        ):
            def pre_hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
                if state.current_event is None:
                    return
                push_module_context(
                    state,
                    path,
                    typ,
                    cls,
                    source,
                )

            return pre_hook

        def make_post_hook(path: str):
            def post_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
                if state.current_event is None:
                    return
                pop_module_context(state, path)

            return post_hook

        module.register_forward_pre_hook(
            make_pre_hook(
                module_path,
                module_type,
                module_class,
                forward_source,
            )
        )
        module.register_forward_hook(make_post_hook(module_path))
    return module_paths


def wrap_for_filtered_dispatch(
    model: Any,
    state: DispatchState,
    targets: dict[tuple[int, int], dict[str, Any]],
    record_elapsed_us: bool,
) -> None:
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
        previous_forward_id = state.current_forward_id
        previous_phase = state.current_phase
        state.current_forward_id = forward_id
        state.current_phase = "prefill" if seq_len > 1 else "decode"
        event = {
            "forward_id": forward_id,
            "phase": state.current_phase,
            "input_seq_len": seq_len,
            "input_ids_shape": int_shape(input_ids),
            "inputs_embeds_shape": int_shape(inputs_embeds),
        }
        state.forward_events.append(event)
        try:
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
    module_paths = register_module_context_hooks(model, state)
    for layer_idx, layer in enumerate(base_model.layers):
        original_layer_forward = layer.forward
        if id(layer) not in module_paths:
            raise RuntimeError(f"Layer {layer_idx} was not found in model.named_modules(); refusing to synthesize a module path.")
        layer_module_path = module_paths[id(layer)]

        def make_wrapped_layer(orig: Any, idx: int, layer_path: str, layer_module: Any):
            def wrapped_layer(*args: Any, **kwargs: Any) -> Any:
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                position_ids = kwargs.get("position_ids")
                past_key_value = kwargs.get("past_key_value")
                if hidden_states is None:
                    return orig(*args, **kwargs)

                q_len = int(hidden_states.shape[1])
                batch_size = int(hidden_states.shape[0])
                past_len = safe_cache_len(past_key_value, q_len, idx)
                forward_id = int(state.current_forward_id or -1)
                key = (forward_id, idx)
                before_event = {
                    "forward_id": forward_id,
                    "phase": state.current_phase or ("prefill" if q_len > 1 else "decode"),
                    "layer_idx": idx,
                    "batch_size": batch_size,
                    "layer_module_path": layer_module_path,
                    "q_len": q_len,
                    "past_len": past_len,
                    "kv_len": past_len + q_len,
                    "hidden_shape_in": int_shape(hidden_states),
                    "position_shape": int_shape(position_ids),
                    "dispatch_captured": key in targets,
                }

                target = targets.get(key)
                if target is None:
                    output = orig(*args, **kwargs)
                else:
                    current = dict(target)
                    current.update({
                        "actual_q_len": q_len,
                        "actual_past_len": past_len,
                        "actual_kv_len": past_len + q_len,
                        "layer_module_path": layer_path,
                    })
                    previous_event = state.current_event
                    previous_module_stack = list(state.module_stack)
                    state.current_event = current
                    state.module_stack = []
                    try:
                        push_module_context(
                            state,
                            layer_path,
                            type(layer_module).__name__,
                            f"{type(layer_module).__module__}.{type(layer_module).__qualname__}",
                            module_forward_source(layer_module),
                        )
                        with RecordingDispatchMode(state, record_elapsed_us=record_elapsed_us):
                            output = orig(*args, **kwargs)
                    finally:
                        state.module_stack = previous_module_stack
                        state.current_event = previous_event

                if isinstance(output, tuple) and output:
                    before_event["hidden_shape_out"] = int_shape(output[0])
                state.layer_events.append(before_event)
                return output

            return wrapped_layer

        layer.forward = make_wrapped_layer(original_layer_forward, layer_idx, layer_module_path, layer)


def summarize_ops(op_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for row in op_rows:
        grouped[(
            str(row["event_id"]),
            str(row.get("runtime_module_scope") or ""),
            str(row.get("module_path") or ""),
            str(row.get("module_relative_path") or ""),
            str(row["op_name"]),
            str(row["op_schema"]),
        )] += 1
    return [
        {
            "event_id": key[0],
            "runtime_module_scope": key[1],
            "module_path": key[2],
            "module_relative_path": key[3],
            "op_name": key[4],
            "op_schema": key[5],
            "count": count,
        }
        for key, count in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][3],
                item[0][4],
                item[0][5],
            ),
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(DEFAULT_TRACE), help="Algorithmic Trace algorithmic_trace.json used to build the filtered dispatch manifest.")
    parser.add_argument("--config", choices=sorted(CONFIGS), default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--priorities", default="P0,P1,P2,P3", help="Comma-separated priorities to keep. No option profiles all layers.")
    parser.add_argument("--no-decode-effect", action="store_true", help="Drop P3 decode-effect representatives.")
    parser.add_argument("--no-shallow", action="store_true", help="Drop P2 shallow-mask representatives.")
    parser.add_argument("--manifest-only", action="store_true", help="Write the manifest without loading the model or running forward.")
    parser.add_argument("--record-elapsed-us", action="store_true", help="Record Python dispatch wrapper elapsed time. This is instrumentation overhead, not CUDA kernel latency.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    priorities = {item.strip() for item in args.priorities.split(",") if item.strip()}
    invalid = sorted(priorities - set(PRIORITY_ORDER))
    if invalid:
        raise SystemExit(f"Unknown priorities: {invalid}")

    manifest = build_dispatch_manifest(
        trace_payload,
        priorities=priorities,
        include_decode_effect=not args.no_decode_effect,
        include_shallow=not args.no_shallow,
    )
    total_layer_events = len(trace_payload.get("layer_events") or [])
    if not manifest:
        raise SystemExit("Filtered dispatch manifest is empty; refusing to run.")
    if len(manifest) >= total_layer_events:
        raise SystemExit(
            f"Filtered manifest has {len(manifest)} events, total layer events are {total_layer_events}; refusing full dispatch profile."
        )

    config_name = args.config or trace_payload.get("config") or "visipruner-full"
    if config_name not in CONFIGS:
        raise SystemExit(f"Unknown config from args/trace: {config_name}")
    cfg = CONFIGS[config_name]
    model_path = args.model_path or trace_payload.get("model_path") or DEFAULT_MODEL_PATH
    image_path = args.image_path or trace_payload.get("image_path") or DEFAULT_IMAGE_PATH
    prompt = args.prompt or trace_payload.get("prompt") or "Describe the image briefly."
    conv_mode = args.conv_mode or trace_payload.get("conv_mode") or "llava_v1"
    max_new_tokens = int(args.max_new_tokens or trace_payload.get("max_new_tokens") or 32)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"filtered_dispatch_{config_name}_{max_new_tokens}tok_{timestamp}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dispatch_manifest.csv", manifest)

    if args.manifest_only:
        print(json.dumps({
            "manifest": str(out_dir / "dispatch_manifest.csv"),
            "target_event_count": len(manifest),
            "total_layer_events": total_layer_events,
        }, indent=2))
        return

    target_rows = {
        (int(row["input_id"]), int(row["layer_id"])): row
        for row in manifest
        if str(row.get("keep_default", True)).lower() != "false"
    }

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map="cuda:0",
        use_flash_attn=cfg["use_flash_attn"],
        use_visipruner=cfg["use_visipruner"],
    )
    model.eval()

    state = DispatchState()
    wrap_for_filtered_dispatch(
        model,
        state,
        target_rows,
        record_elapsed_us=args.record_elapsed_us,
    )

    def run_generate_request() -> tuple[Any, Any]:
        prompt_text = build_prompt(prompt, conv_mode)
        request_input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)
        request_attention_mask = torch.ones_like(request_input_ids, dtype=torch.long)
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        request_image_tensor = process_images([image], image_processor, model.config)
        if isinstance(request_image_tensor, list):
            request_image_tensor = request_image_tensor[0]
        request_input_ids = request_input_ids.to(device="cuda", non_blocking=True)
        request_attention_mask = request_attention_mask.to(device="cuda", non_blocking=True)
        request_image_tensor = request_image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        with torch.inference_mode():
            request_output_ids = model.generate(
                request_input_ids,
                attention_mask=request_attention_mask,
                images=request_image_tensor,
                image_sizes=[image_size],
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                pruning_config=cfg["pruning_config"],
                use_cache=True,
            )
        return request_input_ids, request_output_ids

    input_ids, output_ids = run_generate_request()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if output_ids.shape[1] > input_ids.shape[1]:
        output_token_ids = output_ids[0, input_ids.shape[1]:]
    else:
        output_token_ids = output_ids[0]
    output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()

    write_csv(out_dir / "dispatch_ops.csv", state.op_events, field_order=DISPATCH_OP_FIELD_ORDER)
    write_csv(out_dir / "dispatch_op_summary.csv", summarize_ops(state.op_events))
    write_csv(out_dir / "observed_layer_events.csv", state.layer_events)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "visipruner_filtered_torch_dispatch_profile",
        "full_profile_forbidden": True,
        "dispatch_scope": "TorchDispatchMode is entered only inside manifest-selected layer.forward calls.",
        "dispatch_detail_scope": "dispatch_ops.csv emits only the selected compact columns: args,event_id,kv_len,kwargs,module_class,op_schema,outputs,past_len,phase,q_len,token_state,visipruner_role,input_tensor_ids,output_tensor_ids.",
        "tensor_provenance_scope": "dispatch_ops.csv keeps only input_tensor_ids and output_tensor_ids for Tensor provenance. Detailed input/output tensor metadata, producer records, observed tensor edges, and alias records are not emitted in the CSV.",
        "source_trace": str(trace_path),
        "tag": tag,
        "config": config_name,
        "model_path": model_path,
        "image_path": image_path,
        "prompt": prompt,
        "conv_mode": conv_mode,
        "max_new_tokens": max_new_tokens,
        "context_len": context_len,
        "target_event_count": len(target_rows),
        "total_layer_events_in_source_trace": total_layer_events,
        "observed_forward_count": len(state.forward_events),
        "observed_layer_event_count": len(state.layer_events),
        "captured_dispatch_op_count": len(state.op_events),
        "output_token_count": int(output_token_ids.numel()),
        "output_text": output_text,
        "matches_source_output_text": output_text == ((trace_payload.get("request") or {}).get("output_text")),
        "outputs": {
            "manifest": str(out_dir / "dispatch_manifest.csv"),
            "dispatch_ops": str(out_dir / "dispatch_ops.csv"),
            "dispatch_op_summary": str(out_dir / "dispatch_op_summary.csv"),
            "observed_layer_events": str(out_dir / "observed_layer_events.csv"),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
