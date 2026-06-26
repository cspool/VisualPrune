# SGLang VQA Backend Benchmark

This directory adapts the existing LLaVA/VisPrune VQA JSONL workload to an
SGLang OpenAI-compatible multimodal endpoint.

The client keeps the same answer JSONL shape used by `llava.eval`:

```json
{"question_id": "...", "prompt": "...", "text": "...", "answer_id": "...", "model_id": "...", "metadata": {...}}
```

## Quick smoke run

```bash
cd /workspace/VisPrune
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/quick_textvqa.env
```

This uses a small TextVQA slice, `triton torch_native`, and
`--disable-cuda-graph` so it is fast enough for connectivity checks.

The original compatibility entrypoint is still available:

```bash
repo/scripts/v1_5/sglang_eval/run_textvqa_backends.sh
```

Base defaults:

- model: local `liuhaotian/llava-v1.5-7b` snapshot under `models/hub`
- dataset: TextVQA JSONL from `repo/playground/data/eval/textvqa`

If the model snapshot lacks `preprocessor_config.json`, the driver creates a
symlink-only model view at `models/sglang/llava-v1.5-7b` and links the cached
CLIP image processor config into it. The original HuggingFace snapshot is not
modified.

Run a larger comparison:

```bash
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/textvqa_dense_16.env
```

For a full selected JSONL, copy or edit a config and set `LIMIT=0`, or call
`run_eval_backends.sh` directly with the desired environment variables.

## Reusable configs

Known-good configurations are stored in `configs/*.env` and can be launched
through:

```bash
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/mmvet_dense_32_cudagraph.env
```

Available configs:

- `quick_textvqa.env`: fast TextVQA smoke check with CUDA graph disabled.
- `textvqa_dense_16.env`: reproducible dense-backend TextVQA comparison.
- `mmvet_dense_32.env`: generation-heavier MMVet comparison with CUDA graph
  disabled for faster iteration and lower startup overhead.
- `mmvet_dense_32_cudagraph.env`: recommended performance config for dense
  `triton` and `flashinfer`; leaves SGLang CUDA graph capture enabled.
- `tilelang_dsa_probe.env`: explicit TileLang-backed DSA support probe.

## Eval workloads

Use `run_eval_backends.sh` with `TASK` to select one of the local
LLaVA/VisPrune eval JSONL workloads:

```bash
TASK=textvqa repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
TASK=mme     repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
TASK=gqa     repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
TASK=pope    repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
TASK=mmvet   repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
TASK=vqav2   repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
```

The TextVQA compatibility entrypoint is equivalent to:

```bash
TASK=textvqa repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
```

For a custom LLaVA-style JSONL:

```bash
TASK=custom \
QUESTION_FILE=/path/to/questions.jsonl \
IMAGE_FOLDER=/path/to/images \
OUTPUT_DIR=/path/to/output \
repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
```

All built-in tasks use JSONL records with `image` and `text` fields. If direct
image path joining is insufficient, `IMAGE_SEARCH_DEPTH` enables bounded lookup
under the image folder. This is set automatically for POPE because this checkout
stores COCO images one level below the original script path.

## Backend names

- `triton`: `--attention-backend triton --sampling-backend pytorch`
- `torch_native`: `--attention-backend torch_native --sampling-backend pytorch`
- `flashinfer`: `--attention-backend flashinfer --sampling-backend flashinfer`
- `dsa-tilelang`: `--attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang`
- `nsa-tilelang`: `--attention-backend nsa --nsa-prefill-backend tilelang --nsa-decode-backend tilelang`

TileLang is exposed through SGLang's DSA/NSA backend options. Dense LLaVA
models may reject DSA/NSA at server startup; the driver records that as
`<backend>.status.json` and continues with the remaining backends.

## Notes on VisPrune

The original VisPrune eval scripts pass `--pruning-config` into the HuggingFace
LLaVA model's custom `generate` path. Stock SGLang runs its own inference model
executor, so this benchmark records the pruning config in metadata but does not
apply VisPrune token pruning inside SGLang. Applying it in-server requires
porting the custom attention/pruning logic from `repo/llava/model/language_model`
to SGLang's LLaVA model and attention backend path.

## Alternate dataset

MME uses the same JSONL shape:

```bash
TASK=mme LIMIT=32 repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
```

## Verified result

The current checkout was verified with the performance-oriented MMVet config:

```bash
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/mmvet_dense_32_cudagraph.env
```

Summary:

```text
backend     status  completed  mean_latency_s  p95_latency_s  completion_tokens_per_s
flashinfer  ok      32         1.1860          1.6058         61.6850
triton      ok      32         1.1846          1.6086         61.7613
```

The CUDA-graph-disabled MMVet config also completed successfully and is useful
for faster iteration:

```text
backend       status  completed  mean_latency_s  p95_latency_s  completion_tokens_per_s
flashinfer    ok      32         1.2929          1.7632         56.6210
torch_native  ok      32         1.8077          2.4958         40.4332
triton        ok      32         1.4054          1.9805         52.0821
```

TileLang is installed and importable, but the `dsa-tilelang` probe is expected
to be classified as unsupported for this LLaVA model in the current SGLang tree:
`AssertionError: DSA backend only supports DeepSeek DSA`.
