#!/usr/bin/env python3
"""Verify that trace wrappers preserve generate outputs for one request."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

WORKLOAD_DIR = Path(__file__).resolve().parents[3]
ROOT_DIR = WORKLOAD_DIR.parent
ALGORITHMIC_TRACE_TOOLS_DIR = WORKLOAD_DIR / "algorithmic_trace/tools"

if str(ALGORITHMIC_TRACE_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMIC_TRACE_TOOLS_DIR))

from visipruner_algorithmic_trace import (
    CONFIGS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_PATH,
    TraceState,
    build_prompt,
    wrap_for_trace,
)

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DEFAULT_OUTPUT_DIR = WORKLOAD_DIR / "algorithmic_trace/verification/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="visipruner-full")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default="Describe the image briefly.")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tag", default="wrapper_equivalence_visipruner_full_32tok")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CONFIGS[args.config]

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map="cuda:0",
        use_flash_attn=cfg["use_flash_attn"],
        use_visipruner=cfg["use_visipruner"],
    )
    model.eval()

    prompt_text = build_prompt(args.prompt, args.conv_mode)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device="cuda", non_blocking=True)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    image = Image.open(args.image_path).convert("RGB")
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

    generate_kwargs = dict(
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        pruning_config=cfg["pruning_config"],
        use_cache=True,
    )

    with torch.inference_mode():
        unwrapped_output = model.generate(input_ids, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    state = TraceState()
    wrap_for_trace(model, state)

    with torch.inference_mode():
        wrapped_output = model.generate(input_ids, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    same_shape = list(unwrapped_output.shape) == list(wrapped_output.shape)
    same_ids = same_shape and torch.equal(unwrapped_output, wrapped_output)
    def generated_part(output: torch.Tensor) -> torch.Tensor:
        if output.shape[1] > input_ids.shape[1]:
            return output[0, input_ids.shape[1]:]
        return output[0]

    unwrapped_text = tokenizer.decode(generated_part(unwrapped_output), skip_special_tokens=True).strip()
    wrapped_text = tokenizer.decode(generated_part(wrapped_output), skip_special_tokens=True).strip()

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "model_path": args.model_path,
        "image_path": args.image_path,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "comparison": {
            "same_shape": same_shape,
            "same_output_ids": bool(same_ids),
            "same_decoded_text": unwrapped_text == wrapped_text,
            "unwrapped_output_shape": [int(x) for x in unwrapped_output.shape],
            "wrapped_output_shape": [int(x) for x in wrapped_output.shape],
            "unwrapped_output_text": unwrapped_text,
            "wrapped_output_text": wrapped_text,
        },
        "trace_side_effect_check": {
            "forward_events": len(state.forward_events),
            "layer_events": len(state.layer_events),
            "selection_events": len(state.selection_events),
        },
        "method": "Run generate once without wrappers, then monkey-patch wrappers and run generate again on the same loaded model and same deterministic inputs.",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verification": str(out_path), **payload["comparison"]}, indent=2))

    if not same_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
