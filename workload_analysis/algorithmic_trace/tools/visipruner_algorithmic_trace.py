#!/usr/bin/env python3
"""Generate a forward-driven algorithmic workload trace for LLaVA/VisiPrune.

The script runs one real generation request to expose VisiPrune's dynamic
token schedule, then computes hardware-agnostic FLOP estimates from formulas.
It does not use CUDA kernel timing or backend profiler output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
WORKLOAD_DIR = ROOT_DIR / "workload_analysis"
REPO_DIR = ROOT_DIR / "repo"
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_PATH = str(ROOT_DIR / "autoresearch/data/benchmark_images/002901d9d194c4fb.jpg")
DEFAULT_OUTPUT_DIR = str(WORKLOAD_DIR / "algorithmic_trace/traces")

CONFIGS: dict[str, dict[str, Any]] = {
    "dense-eager": {
        "use_flash_attn": False,
        "use_visipruner": False,
        "pruning_config": None,
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
    },
}


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

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


@dataclass
class TraceState:
    forward_id: int = 0
    current_forward_id: int | None = None
    current_phase: str | None = None
    forward_events: list[dict[str, Any]] = field(default_factory=list)
    layer_events: list[dict[str, Any]] = field(default_factory=list)
    selection_events: list[dict[str, Any]] = field(default_factory=list)


def int_shape(value: Any) -> list[int] | None:
    if value is None or not hasattr(value, "shape"):
        return None
    return [int(x) for x in value.shape]


def tensor_count(value: Any) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return int(value.numel())
    return None


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


def wrap_for_trace(model: Any, state: TraceState) -> None:
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
    for layer_idx, layer in enumerate(base_model.layers):
        original_layer_forward = layer.forward

        def make_wrapped_layer(orig: Any, idx: int):
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
                before_event = {
                    "forward_id": state.current_forward_id,
                    "phase": state.current_phase or ("prefill" if q_len > 1 else "decode"),
                    "layer_idx": idx,
                    "batch_size": batch_size,
                    "q_len": q_len,
                    "past_len": past_len,
                    "kv_len": past_len + q_len,
                    "hidden_shape_in": int_shape(hidden_states),
                    "position_shape": int_shape(position_ids),
                    "position_last": int(position_ids[0, -1].item()) if torch.is_tensor(position_ids) and position_ids.numel() else None,
                    "important_visual_tokens_in": tensor_count(kwargs.get("important_vis_tokens")),
                    "exit_indicator_in": int(kwargs.get("exit_indicator", 0) or 0),
                }
                output = orig(*args, **kwargs)
                if isinstance(output, tuple) and output:
                    out_hidden = output[0]
                    before_event["hidden_shape_out"] = int_shape(out_hidden)
                    if len(output) >= 2:
                        before_event["important_visual_tokens_out"] = tensor_count(output[-2])
                    if len(output) >= 1:
                        try:
                            before_event["exit_indicator_out"] = int(output[-1] or 0)
                        except Exception:
                            before_event["exit_indicator_out"] = None
                state.layer_events.append(before_event)
                return output

            return wrapped_layer

        layer.forward = make_wrapped_layer(original_layer_forward, layer_idx)

        attn = getattr(layer, "self_attn", None)
        if hasattr(attn, "value_aware_token_selection"):
            original_select = attn.value_aware_token_selection

            def make_wrapped_select(orig: Any, idx: int):
                def wrapped_select(*args: Any, **kwargs: Any) -> Any:
                    result = orig(*args, **kwargs)
                    event = {
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
                        event["selected_visual_token_indices"] = [int(x) for x in result.detach().cpu().tolist()]
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


def build_prompt(raw_prompt: str, conv_mode: str) -> str:
    prompt = raw_prompt
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def linear_flops(batch_size: int, tokens: int, in_dim: int, out_dim: int) -> int:
    return int(2 * batch_size * tokens * in_dim * out_dim)


def layer_formula_ops(layer_event: dict[str, Any], model_dims: dict[str, int]) -> list[dict[str, Any]]:
    b = int(layer_event["batch_size"])
    q = int(layer_event["q_len"])
    kv = int(layer_event["kv_len"])
    d = model_dims["hidden_size"]
    ffn = model_dims["intermediate_size"]
    heads = model_dims["num_attention_heads"]
    kv_heads = model_dims["num_key_value_heads"]
    head_dim = d // heads
    kv_dim = kv_heads * head_dim
    base = {
        "source": "formula",
        "forward_id": layer_event["forward_id"],
        "phase": layer_event["phase"],
        "layer_idx": layer_event["layer_idx"],
        "batch_size": b,
        "q_len": q,
        "kv_len": kv,
    }
    ops = [
        ("q_proj", linear_flops(b, q, d, d), "2*B*Q*D*D"),
        ("k_proj", linear_flops(b, q, d, kv_dim), "2*B*Q*D*KV_DIM"),
        ("v_proj", linear_flops(b, q, d, kv_dim), "2*B*Q*D*KV_DIM"),
        ("attn_qk_matmul", int(2 * b * heads * q * kv * head_dim), "2*B*H*Q*KV*HEAD_DIM"),
        ("attn_sv_matmul", int(2 * b * heads * q * kv * head_dim), "2*B*H*Q*KV*HEAD_DIM"),
        ("o_proj", linear_flops(b, q, d, d), "2*B*Q*D*D"),
        ("mlp_gate_proj", linear_flops(b, q, d, ffn), "2*B*Q*D*FFN"),
        ("mlp_up_proj", linear_flops(b, q, d, ffn), "2*B*Q*D*FFN"),
        ("mlp_down_proj", linear_flops(b, q, ffn, d), "2*B*Q*FFN*D"),
    ]
    return [{**base, "op_name": name, "flops": flops, "formula": formula} for name, flops, formula in ops]


def selection_formula_ops(selection_event: dict[str, Any], model_dims: dict[str, int], visual_tokens: int) -> dict[str, Any]:
    shape = selection_event.get("value_states_shape") or []
    seq_len = int(shape[2]) if len(shape) >= 3 else visual_tokens
    d = model_dims["hidden_size"]
    candidates = int(selection_event.get("input_important_visual_tokens") or visual_tokens)
    candidates = max(0, min(candidates, seq_len))
    # Contributions, masked subtraction, cosine/l2 reductions. This is an
    # algorithmic estimate for pruning bookkeeping, not a CUDA kernel count.
    flops = int(candidates * d * 8)
    return {
        "source": "formula",
        "forward_id": selection_event.get("forward_id"),
        "phase": selection_event.get("phase"),
        "layer_idx": selection_event.get("layer_idx"),
        "batch_size": 1,
        "q_len": 1,
        "kv_len": seq_len,
        "op_name": "visipruner_value_selection_aux",
        "flops": flops,
        "formula": "approx 8*CANDIDATE_VIS_TOKENS*D",
    }


def vision_formula_ops(model: Any, image_token_count: int) -> list[dict[str, Any]]:
    vt = model.get_vision_tower()
    cfg = vt.config
    hidden = int(cfg.hidden_size)
    heads = int(cfg.num_attention_heads)
    head_dim = hidden // heads
    layers = int(cfg.num_hidden_layers)
    intermediate = int(cfg.intermediate_size)
    image_size = int(cfg.image_size)
    patch_size = int(cfg.patch_size)
    patches = int((image_size // patch_size) ** 2)
    clip_seq = patches + 1
    ops: list[dict[str, Any]] = []
    patch_embed = int(2 * patches * (3 * patch_size * patch_size) * hidden)
    ops.append({
        "source": "formula",
        "phase": "vision",
        "layer_idx": -1,
        "op_name": "clip_patch_embed",
        "batch_size": 1,
        "q_len": patches,
        "kv_len": 0,
        "flops": patch_embed,
        "formula": "2*PATCHES*(3*PATCH*PATCH)*VISION_D",
    })
    for idx in range(layers):
        base = {"source": "formula", "phase": "vision", "layer_idx": idx, "batch_size": 1, "q_len": clip_seq, "kv_len": clip_seq}
        ops.extend([
            {**base, "op_name": "clip_qkv_proj", "flops": int(2 * clip_seq * hidden * hidden * 3), "formula": "2*N*D*D*3"},
            {**base, "op_name": "clip_attn_qk_matmul", "flops": int(2 * heads * clip_seq * clip_seq * head_dim), "formula": "2*H*N*N*HEAD_DIM"},
            {**base, "op_name": "clip_attn_sv_matmul", "flops": int(2 * heads * clip_seq * clip_seq * head_dim), "formula": "2*H*N*N*HEAD_DIM"},
            {**base, "op_name": "clip_o_proj", "flops": int(2 * clip_seq * hidden * hidden), "formula": "2*N*D*D"},
            {**base, "op_name": "clip_mlp_fc1", "flops": int(2 * clip_seq * hidden * intermediate), "formula": "2*N*D*FFN"},
            {**base, "op_name": "clip_mlp_fc2", "flops": int(2 * clip_seq * intermediate * hidden), "formula": "2*N*FFN*D"},
        ])
    mm_hidden = int(getattr(model.config, "mm_hidden_size", hidden))
    llm_hidden = int(model.config.hidden_size)
    projector_type = getattr(model.config, "mm_projector_type", "linear")
    if projector_type == "mlp2x_gelu":
        projector_flops = int(2 * image_token_count * mm_hidden * llm_hidden + 2 * image_token_count * llm_hidden * llm_hidden)
        formula = "2*VIS_TOKENS*MM_D*D + 2*VIS_TOKENS*D*D"
    else:
        projector_flops = int(2 * image_token_count * mm_hidden * llm_hidden)
        formula = "2*VIS_TOKENS*MM_D*D"
    ops.append({
        "source": "formula",
        "phase": "vision_projector",
        "layer_idx": -1,
        "op_name": f"mm_projector_{projector_type}",
        "batch_size": 1,
        "q_len": image_token_count,
        "kv_len": 0,
        "flops": projector_flops,
        "formula": formula,
    })
    return ops


def lm_head_ops(state: TraceState, model_dims: dict[str, int]) -> list[dict[str, Any]]:
    d = model_dims["hidden_size"]
    vocab = model_dims["vocab_size"]
    ops = []
    layer_by_forward: dict[int, list[dict[str, Any]]] = {}
    for event in state.layer_events:
        if event.get("forward_id") is not None:
            layer_by_forward.setdefault(int(event["forward_id"]), []).append(event)
    for fwd in state.forward_events:
        forward_id = int(fwd["forward_id"])
        layers = layer_by_forward.get(forward_id, [])
        if layers:
            last = max(layers, key=lambda x: int(x["layer_idx"]))
            out_shape = last.get("hidden_shape_out")
            actual_q = int(out_shape[1]) if out_shape and len(out_shape) >= 2 else int(fwd["input_seq_len"])
        else:
            actual_q = int(fwd["input_seq_len"])
        phase = fwd["phase"]
        ops.append({
            "source": "formula",
            "forward_id": forward_id,
            "phase": phase,
            "layer_idx": -1,
            "batch_size": 1,
            "q_len": actual_q,
            "kv_len": 0,
            "op_name": "lm_head_actual_model_path",
            "flops": linear_flops(1, actual_q, d, vocab),
            "formula": "2*B*FINAL_Q*D*VOCAB",
        })
        ops.append({
            "source": "formula",
            "forward_id": forward_id,
            "phase": phase,
            "layer_idx": -1,
            "batch_size": 1,
            "q_len": 1,
            "kv_len": 0,
            "op_name": "lm_head_ideal_last_token_only",
            "flops": linear_flops(1, 1, d, vocab),
            "formula": "2*B*1*D*VOCAB",
        })
    return ops


def summarize_ops(ops: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, int] = {}
    by_op: dict[str, int] = {}
    for row in ops:
        flops = int(row["flops"])
        by_phase[row.get("phase") or "unknown"] = by_phase.get(row.get("phase") or "unknown", 0) + flops
        by_op[row["op_name"]] = by_op.get(row["op_name"], 0) + flops
    total_actual = sum(int(row["flops"]) for row in ops if row["op_name"] != "lm_head_ideal_last_token_only")
    total_with_ideal_lm = sum(
        int(row["flops"]) for row in ops if row["op_name"] != "lm_head_actual_model_path"
    )
    return {
        "total_flops_actual_model_path": total_actual,
        "total_flops_with_ideal_lm_head": total_with_ideal_lm,
        "by_phase_flops": dict(sorted(by_phase.items())),
        "by_op_flops": dict(sorted(by_op.items(), key=lambda item: item[1], reverse=True)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="visipruner-full")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default="Describe the image briefly.")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CONFIGS[args.config]
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map="cuda:0",
        use_flash_attn=cfg["use_flash_attn"],
        use_visipruner=cfg["use_visipruner"],
    )
    model.eval()

    state = TraceState()
    wrap_for_trace(model, state)

    prompt_text = build_prompt(args.prompt, args.conv_mode)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    image = Image.open(args.image_path).convert("RGB")
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    input_ids = input_ids.to(device="cuda", non_blocking=True)
    attention_mask = attention_mask.to(device="cuda", non_blocking=True)
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            pruning_config=cfg["pruning_config"],
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if output_ids.shape[1] > input_ids.shape[1]:
        output_token_ids = output_ids[0, input_ids.shape[1]:]
    else:
        output_token_ids = output_ids[0]
    output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()

    model_dims = {
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "vocab_size": int(model.config.vocab_size),
    }
    image_token_count = int(model.get_vision_tower().num_patches)
    formula_ops: list[dict[str, Any]] = []
    formula_ops.extend(vision_formula_ops(model, image_token_count))
    for event in state.layer_events:
        formula_ops.extend(layer_formula_ops(event, model_dims))
    for event in state.selection_events:
        formula_ops.append(selection_formula_ops(event, model_dims, image_token_count))
    formula_ops.extend(lm_head_ops(state, model_dims))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.config}_{args.max_new_tokens}tok_{timestamp}"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "analysis_type": "forward_driven_algorithmic_trace",
        "counting_convention": "FLOPs count multiply and add as 2 operations for matmul/linear terms.",
        "exclusions": [
            "model loading",
            "image file I/O",
            "CPU preprocessing",
            "host-device transfer",
            "CUDA kernel launch overhead",
            "runtime/compiler fusion effects",
            "memory traffic and communication",
        ],
        "config": args.config,
        "model_path": args.model_path,
        "image_path": args.image_path,
        "prompt": args.prompt,
        "conv_mode": args.conv_mode,
        "max_new_tokens": args.max_new_tokens,
        "context_len": context_len,
        "use_visipruner": cfg["use_visipruner"],
        "use_flash_attn": cfg["use_flash_attn"],
        "pruning_config": cfg["pruning_config"],
        "request": {
            "prompt_token_count_before_image_expansion": int(input_ids.shape[1]),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "image_token_count": image_token_count,
            "output_token_count": int(output_token_ids.numel()),
            "output_text": output_text,
        },
        "model_dims": model_dims,
        "forward_events": state.forward_events,
        "layer_events": state.layer_events,
        "selection_events": state.selection_events,
        "summary": summarize_ops(formula_ops),
    }

    (out_dir / "algorithmic_trace.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "layer_trace.csv", state.layer_events)
    write_csv(out_dir / "selection_trace.csv", state.selection_events)
    write_csv(out_dir / "operator_flops.csv", formula_ops)
    (Path(args.output_dir) / "latest_algorithmic_trace_path.txt").write_text(str(out_dir / "algorithmic_trace.json") + "\n", encoding="utf-8")

    print(json.dumps({
        "trace": str(out_dir / "algorithmic_trace.json"),
        "operator_flops_csv": str(out_dir / "operator_flops.csv"),
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
