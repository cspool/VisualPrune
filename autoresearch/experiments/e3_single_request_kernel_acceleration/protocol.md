# E3: Single-Request VisiPrune Kernel Acceleration

Date: 2026-07-01

## Objective

Design and evaluate kernels that can turn VisiPrune's token/KV-cache pruning
into latency gains for one request at a time.

This experiment is explicitly scoped to:

- batch size 1;
- one image + one prompt;
- exact greedy decode unless a section is explicitly marked optional;
- no cross-request batching;
- comparison against the existing E2 single-request baselines.

## Background

`workload_analysis/human_draft.md` characterizes the VisiPrune backbone
workload:

- prefill layer 0-18 keeps the full expanded sequence, about 624 tokens in the
  current traced request;
- layer 18 selects a small set of visual tokens;
- layer 19-27 runs on a compacted sequence, about 58 tokens;
- layer 28-31 removes the remaining selected visual tokens, about 48 tokens;
- decode reuses layer-dependent KV cache lengths:
  - layer 0-18: full cache, `624 + t`;
  - layer 19-27: middle-pruned cache, `58 + t`;
  - layer 28-31: deep-removed cache, `48 + t`.

E1/E2 results show the current runtime does not convert this schedule into
single-request speedup:

- E2 dense-FA2 clock: request `1140.97 ms`, prefill `68.95 ms`, decode
  `1003.61 ms`.
- E2 eager VisiPrune clock: request `1456.85 ms`, prefill `98.52 ms`, decode
  `1265.12 ms`, value-aware selection `8.97 ms`.
- E2 Nsight: VisiPrune reduces total CUDA kernel duration slightly versus
  dense-FA2, but request/decode NVTX wall time increases.
- Kernel-family analysis: decode GEMV is the dominant kernel bucket and does
  not decrease under eager VisiPrune.

Therefore, a single-request kernel plan must attack decode GEMV, launch gaps,
and ragged KV attention overhead. Optimizing selection alone is insufficient.

## Hypotheses

### H1: Ragged single-query attention removes VisiPrune decode overhead

For `q_len=1`, a VisiPrune-aware attention kernel that consumes compact/paged
KV cache directly should reduce copy/gather/cat/repeat overhead and exploit the
shorter layer-specific KV lengths.

Prediction:

- `copy_gather_cat` and attention-side small kernels drop materially inside
  `forward_decode` ranges.
- Request latency improves over the current eager/native VisiPrune path.
- Improvement over dense-FA2 is limited unless decode projection cost is also
  reduced.

### H2: Fused single-token decode blocks are required for meaningful speedup

Because exact greedy decode processes one token at a time, the dominant work is
batch-1 projection/MLP GEMV. A fused decode block should reduce launch overhead
and memory traffic around GEMV operations.

Candidate decomposition:

```text
kernel A: rmsnorm + qkv projection + rope + kv append
kernel B: ragged single-query attention + o projection handoff
kernel C: rmsnorm + gate/up/down MLP path + silu/mul + residual
```

Prediction:

- Kernel launch count and NVTX wall/kernel gap decrease.
- Decode wall time improves more than attention-only optimization.
- Without changing GEMV utilization, the achievable speedup remains below the
  theoretical token-pruning ratio.

### H3: Prefill selection and compaction fusion improves the non-dominant part

Layer 7-18 selection/probe logic should avoid materializing full attention
weights and avoid scattered PyTorch tensor operations. A fused kernel should
compute the last-token pruning proxy, score visual tokens, write selected
indices, and compact hidden/KV layout once.

Prediction:

- Prefill latency and small-kernel count decrease.
- End-to-end gain is modest for 32-token generation because prefill is less
  than 7% of dense-FA2 request time in E2.

### H4: CUDA Graph or persistent decode execution is necessary for exact greedy

Nsight shows wall time much larger than summed kernel duration. Capturing the
fixed decode schedule, or using a persistent decode executor, should reduce the
Python/HF generate loop and launch overhead.

Prediction:

- NVTX `forward_decode` wall time approaches CUDA kernel-projected time.
- Benefit is orthogonal to ragged attention and fused compaction.

### H5 optional: Single-request speculative verification changes the problem

If exact greedy can be relaxed while still serving one user request, speculative
or tree verification can batch multiple candidate tokens from the same request
into `[T, hidden] @ W` work. This is outside the default E3 scope, but it is the
most plausible single-request route to larger speedups because it changes GEMV
into small-batch GEMM.

## Kernel Designs To Evaluate

### K1: VisiPrune ragged single-query attention

Inputs:

- `q`: `[num_heads, head_dim]`;
- compact or paged K/V cache;
- `past_len_by_layer`;
- layer regime metadata: full, middle-pruned, deep-removed;
- GQA head mapping.

Required behavior:

- online softmax for one query token;
- no materialized attention matrix;
- no separate `repeat_kv`;
- no per-step `cat` for cache assembly;
- direct support for layer-specific compact KV lengths.

Primary metrics:

- attention-side kernel time inside decode;
- `copy_gather_cat` time;
- number of CUDA launches per decode token;
- numerical/logit parity against the current implementation.

### K2: Prefill selection + compaction fused kernel

Inputs:

- Q/K/V or last-query proxy tensors at decision layers;
- visual-token index range;
- thresholds from VisiPrune config.

Required behavior:

- compute last-token attention proxy;
- compute visual contribution score;
- apply threshold/top-k decision;
- write selected visual indices;
- compact hidden states and KV layout for later layers.

Primary metrics:

- prefill latency;
- selection range latency;
- small-kernel count between layer 7 and layer 18;
- output token/text match against eager VisiPrune for deterministic runs.

### K3: Fused single-token decode block

This is the high-impact but highest-risk prototype.

Minimum viable version:

- fuse norm and projection input preparation;
- fuse QKV projection where practical;
- fuse RoPE and KV append;
- call K1 for attention;
- minimize transitions between projection, attention, and residual updates.

Stretch version:

- fuse MLP gate/up/down around activation and residual;
- use persistent weights or cuBLASLt-style grouped calls where custom GEMV is
  not competitive.

Primary metrics:

- decode wall time;
- decode kernel total time;
- launch count per generated token;
- GEMV bucket time and GEMV call count;
- request total latency.

### K4: CUDA Graph / static decode executor

The initial graph target is fixed shape regime for `max_new_tokens=32`, with
dynamic values but stable allocation/layout.

Primary metrics:

- NVTX wall/kernel gap;
- CPU runtime overhead;
- launch count visible in Nsight;
- compatibility with VisiPrune ragged metadata.

## Baselines

Use E2 artifacts and rerun if code has changed:

- `dense-fa`: dense eager no pruning;
- `dense-fa2`: dense FlashAttention2 no pruning;
- `visipruner-full`: native/eager VisiPrune path;
- `visipruner-full-fa2` or `visipruner-full-vp-fa`: optimized VisiPrune path
  where available and correctness-verified.

Do not mix images or prompts when comparing latency. If a later run uses
`repo/images/v1_73.jpg`, rerun all baselines with that same image.

## Measurement Protocol

Clock timing:

- synchronized wall-clock ranges;
- report request, generate, prefill, decode sum, selection, generate-other.

Nsight Systems:

- CUDA, NVTX, cuBLAS, OS runtime traces;
- CUDA profiler API capture around measured request;
- export NVTX sum, NVTX GPU projection, CUDA kernel summary.

Correctness:

- deterministic generation (`temperature=0`);
- compare generated token ids and text against baseline path;
- for kernel-only tests, compare output tensors/logits with tolerance before
  running full `generate`.

Sanity checks:

- same image, prompt, `max_new_tokens`, model, cache, GPU, and warmup;
- record output token count and observed decode forward calls;
- compare kernel-family categories before interpreting speedup.

## Acceptance Criteria

Minimal useful result:

- K1 or K2 reduces its targeted stage latency and preserves output parity.

Meaningful single-request improvement:

- request latency improves by at least 5% over the strongest same-input
  VisiPrune baseline;
- decode wall time improves, not only prefill.

Strong result:

- request latency approaches or beats same-input dense-FA2 while preserving
  VisiPrune output behavior.

Negative result criteria:

- If K1/K2 improve their local kernels but request latency does not move, record
  this as evidence that projection/launch overhead is the controlling floor.
- If K3/K4 reduce wall/kernel gap but still fail to beat dense-FA2, quantify the
  remaining GEMV floor.

## Expected Ceiling

For the E2 dense-FA2 32-token request, prefill is only about `68.95 / 1140.97 =
6.0%` of request time. Even a perfect prefill kernel cannot produce a large
single-request speedup.

Therefore, under exact greedy decode:

- attention and compaction kernels can make VisiPrune cleaner and reduce local
  overhead;
- decode-block fusion and graph execution are required for request-level gains;
- approaching the original token-pruning theoretical ratio is unlikely unless
  the decode projection/MLP GEMV floor is also reduced or the decoding algorithm
  changes.

## Deliverables

- prototype code under `code/`;
- raw clock/Nsight outputs under `output/`;
- an updated `analysis.md` after each prototype;
- optional visualization comparing stage latency, kernel categories, and
  launch counts.
