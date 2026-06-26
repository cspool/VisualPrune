# SGLang Eval Results

Date: 2026-06-20

## Scope

This integration connects the local LLaVA/VisPrune eval JSONL workloads under
`repo/scripts/v1_5/eval` and `repo/scripts/v1_5/visiPruner_eval` to an
SGLang OpenAI-compatible multimodal server.

Implemented artifacts:

- `bench_vqa_sglang.py`: sends LLaVA-style `image` + `text` JSONL records to
  `/v1/chat/completions`, writes LLaVA-compatible answer JSONL and metrics JSON.
- `run_eval_backends.sh`: launches SGLang for one or more backends, runs a
  selected eval workload, and aggregates backend metrics/status.
- `run_config.sh`: loads a checked-in `configs/*.env` file and runs the backend
  benchmark with reproducible settings.
- `run_textvqa_backends.sh`: compatibility wrapper for `TASK=textvqa`.
- `configs/*.env`: known-good smoke, backend comparison, performance, and
  TileLang probe configurations.
- `models/sglang/llava-v1.5-7b`: symlink model view with the missing CLIP
  `preprocessor_config.json` linked in for SGLang `AutoProcessor` loading.

Built-in `TASK` values:

- `textvqa`
- `mme`
- `gqa`
- `pope`
- `mmvet`
- `vqav2`
- `custom` with explicit `QUESTION_FILE`, `IMAGE_FOLDER`, and `OUTPUT_DIR`

The local files for all built-in JSONL tasks were dry-run validated:

```text
textvqa  003a8ae2ef43b901  .../textvqa/train_images/003a8ae2ef43b901.jpg
mme      code_reasoning/0020.png  .../MME_Benchmark_release_version/code_reasoning/0020.png
gqa      201307251  .../gqa/data/images/n161313.jpg
pope     1  .../pope/val2014/val2014/COCO_val2014_000000310196.jpg
mmvet    0  .../mmvet/mm-vet/images/v1_0.png
vqav2    262144005  .../vqav2/test2015/COCO_test2015_000000262144.jpg
```

## Environment

Verified with:

```text
torch 2.11.0+cu130
cuda_available True
cuda_device_count 2
triton 3.6.0
tilelang 0.1.8
flashinfer 0.6.12
```

CUDA defaults used by the driver:

```text
CUDA_HOME=/usr/local/cuda-13.2
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
HF_HOME=/workspace/VisPrune/models
```

If a fresh shell cannot see CUDA, activate the user-provided environment:

```bash
conda activate cu132
```

## Recommended Configs

Run configs through:

```bash
cd /workspace/VisPrune
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/mmvet_dense_32_cudagraph.env
```

The checked-in configs are:

```text
quick_textvqa.env             fast connectivity check, CUDA graph disabled
textvqa_dense_16.env          dense TextVQA comparison, CUDA graph disabled
mmvet_dense_32.env            generation-heavier comparison, CUDA graph disabled
mmvet_dense_32_cudagraph.env  recommended dense performance config
tilelang_dsa_probe.env        explicit TileLang DSA support probe
```

Use `mmvet_dense_32_cudagraph.env` for reportable performance numbers. Use
`quick_textvqa.env` or `mmvet_dense_32.env` while iterating because disabling
CUDA graph shortens startup and avoids capture overhead.

## TextVQA Backend Run

Command:

```bash
cd /workspace/VisPrune
TASK=textvqa \
BACKENDS="triton torch_native flashinfer dsa-tilelang" \
LIMIT=16 WARMUP=1 CONCURRENCY=2 MAX_NEW_TOKENS=16 \
SERVER_EXTRA_ARGS=--disable-cuda-graph \
OUTPUT_DIR=/workspace/VisPrune/repo/playground/data/eval/textvqa/answers/sglang_textvqa_16_backends \
repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
```

Summary file:

```text
/workspace/VisPrune/repo/playground/data/eval/textvqa/answers/sglang_textvqa_16_backends/summary.csv
```

Results:

```text
backend       status       completed  failed  mean_latency_s  request_throughput_rps
flashinfer    ok           16         0       0.5210          3.2854
torch_native  ok           16         0       0.5880          2.9780
triton        ok           16         0       0.5132          3.2852
dsa-tilelang  unsupported  0          -       -               -
```

Per-backend answer JSONL, metrics JSON, status JSON, server logs, and client
logs are in:

```text
/workspace/VisPrune/repo/playground/data/eval/textvqa/answers/sglang_textvqa_16_backends
```

## MMVet Dense Run, CUDA Graph Disabled

Command:

```bash
cd /workspace/VisPrune
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/mmvet_dense_32.env
```

Summary file:

```text
/workspace/VisPrune/repo/playground/data/eval/mm-vet/answers/sglang_mmvet_32_backends/summary.csv
```

Results:

```text
backend       status  completed  failed  mean_latency_s  p95_latency_s  completion_tokens_per_s  wall_time_s
flashinfer    ok      32         0       1.2929          1.7632         56.6210                  24.0900
torch_native  ok      32         0       1.8077          2.4958         40.4332                  33.7346
triton        ok      32         0       1.4054          1.9805         52.0821                  26.1894
```

This is the best iteration config when comparing dense backends because all
three backends complete and startup is shorter than CUDA graph capture.

## MMVet Dense Run, CUDA Graph Enabled

Command:

```bash
cd /workspace/VisPrune
repo/scripts/v1_5/sglang_eval/run_config.sh \
  repo/scripts/v1_5/sglang_eval/configs/mmvet_dense_32_cudagraph.env
```

Summary file:

```text
/workspace/VisPrune/repo/playground/data/eval/mm-vet/answers/sglang_mmvet_32_cudagraph_backends/summary.csv
```

Results:

```text
backend     status  completed  failed  mean_latency_s  p95_latency_s  completion_tokens_per_s  wall_time_s
flashinfer  ok      32         0       1.1860          1.6058         61.6850                  22.1124
triton      ok      32         0       1.1846          1.6086         61.7613                  22.0850
```

This is the recommended performance result. Server logs confirm decode batches
used CUDA graph (`cuda graph: True`). On this LLaVA/MMVet slice, `triton` and
`flashinfer` are effectively tied with CUDA graph enabled.

## POPE Backend Run

Command:

```bash
cd /workspace/VisPrune
TASK=pope \
BACKENDS="triton torch_native" \
LIMIT=4 WARMUP=0 CONCURRENCY=2 MAX_NEW_TOKENS=8 \
SERVER_EXTRA_ARGS=--disable-cuda-graph \
OUTPUT_DIR=/workspace/VisPrune/repo/playground/data/eval/pope/answers/sglang_pope_4_backends \
repo/scripts/v1_5/sglang_eval/run_eval_backends.sh
```

Summary file:

```text
/workspace/VisPrune/repo/playground/data/eval/pope/answers/sglang_pope_4_backends/summary.csv
```

Results:

```text
backend       status  completed  failed  mean_latency_s  request_throughput_rps
torch_native  ok      4          0       0.5303          3.3369
triton        ok      4          0       0.4472          3.8719
```

The POPE run verifies the generic task mapping and bounded image lookup because
this checkout stores POPE images under `val2014/val2014` while the original eval
script points to `val2014`.

## TileLang Status

TileLang is installed and importable, and the driver can configure SGLang's DSA
TileLang path with:

```text
--attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
```

For this LLaVA/VisPrune workload, SGLang rejects DSA during backend
initialization:

```text
AssertionError: DSA backend only supports DeepSeek DSA
```

The source assertion is in:

```text
/workspace/VisPrune/sglang/python/sglang/srt/layers/attention/dsa_backend.py
```

The failed backend is therefore classified as `unsupported` rather than as an
installation or CUDA failure.

## VisPrune Pruning Status

The original VisPrune scripts pass:

```json
{"mode":["shallow","middle","deep"],"shallow_mid_layer":6,"layer_threshold":0.995,"tokens_threshold":0.2}
```

to the HuggingFace LLaVA model's custom `generate` path. Stock SGLang uses its
own model executor and attention backends, so the benchmark records this config
in output metadata but does not apply VisPrune token pruning inside SGLang.

Making pruning active in SGLang requires porting the custom logic from:

```text
/workspace/VisPrune/repo/llava/model/language_model/custom_modeling_llama.py
/workspace/VisPrune/repo/llava/model/language_model/llava_llama.py
```

into SGLang's LLaVA model and attention execution path.
