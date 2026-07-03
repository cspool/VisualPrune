#!/usr/bin/env python3
"""Compare official eager VisiPruner output with the VP-FA path.

The verifier intentionally compares generated token ids, not just decoded text,
because token equality is the strictest end-to-end correctness signal for greedy
inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_GPU = 1
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_PATH = "/workspace/VisiPrune/repo/images/v1_73.jpg"
DEFAULT_PROMPT = "Describe the image briefly."
DEFAULT_OUTPUT_PATH = (
    "/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/"
    "verify_visipruner_vp_fa_outputs.json"
)


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
os.environ.setdefault("HF_HOME", "/workspace/VisiPrune/models")

REPO_DIR = Path(__file__).resolve().parents[4] / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


PRUNING_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {
        "mode": ["shallow", "middle", "deep"],
        "shallow_mid_layer": 6,
        "layer_threshold": 0.995,
        "tokens_threshold": 0.2,
    },
    "middle-deep": {
        "mode": ["middle", "deep"],
        "shallow_mid_layer": 6,
        "layer_threshold": 0.995,
        "tokens_threshold": 0.2,
    },
}


def build_prompt(raw_prompt: str, conv_mode: str) -> str:
    prompt = raw_prompt
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_one_model(
    *,
    model_path: str,
    model_base: str | None,
    backend: str,
    use_flash_attn: bool,
):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map="cuda:0",
        use_flash_attn=use_flash_attn,
        use_visipruner=True,
        visipruner_decode_backend=backend,
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def run_generate(
    *,
    model_path: str,
    model_base: str | None,
    backend: str,
    use_flash_attn: bool,
    image_path: str,
    prompt: str,
    conv_mode: str,
    max_new_tokens: int,
    pruning_config: dict[str, Any],
) -> dict[str, Any]:
    tokenizer, model, image_processor, context_len = load_one_model(
        model_path=model_path,
        model_base=model_base,
        backend=backend,
        use_flash_attn=use_flash_attn,
    )
    prompt_text = build_prompt(prompt, conv_mode)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)

    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pruning_config=pruning_config,
        )

    if output_ids.shape[1] > input_ids.shape[1]:
        generated_tensor = output_ids[0, input_ids.shape[1]:]
    else:
        generated_tensor = output_ids[0]
    generated_ids = generated_tensor.detach().cpu().tolist()
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    result = {
        "backend": backend,
        "resolved_backend": getattr(model.config, "visipruner_decode_backend", None),
        "model_class": model.__class__.__name__,
        "base_model_class": model.get_model().__class__.__name__,
        "context_len": context_len,
        "input_token_count": int(input_ids.shape[1]),
        "generated_token_ids": generated_ids,
        "generated_text": decoded,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--pruning-config", choices=sorted(PRUNING_CONFIGS), default="full")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    pruning_config = PRUNING_CONFIGS[args.pruning_config]
    official = run_generate(
        model_path=args.model_path,
        model_base=args.model_base,
        backend="eager",
        use_flash_attn=False,
        image_path=args.image_path,
        prompt=args.prompt,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
        pruning_config=pruning_config,
    )
    vp_fa = run_generate(
        model_path=args.model_path,
        model_base=args.model_base,
        backend="vp-fa",
        use_flash_attn=True,
        image_path=args.image_path,
        prompt=args.prompt,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
        pruning_config=pruning_config,
    )
    token_match = official["generated_token_ids"] == vp_fa["generated_token_ids"]
    text_match = official["generated_text"] == vp_fa["generated_text"]
    payload = {
        "pruning_config_name": args.pruning_config,
        "pruning_config": pruning_config,
        "model_path": args.model_path,
        "image_path": args.image_path,
        "prompt": args.prompt,
        "conv_mode": args.conv_mode,
        "max_new_tokens": args.max_new_tokens,
        "token_match": token_match,
        "text_match": text_match,
        "official": official,
        "vp_fa": vp_fa,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not token_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
