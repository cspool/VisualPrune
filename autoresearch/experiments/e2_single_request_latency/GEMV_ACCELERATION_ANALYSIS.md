# VisiPrune GEMV Runtime Analysis

Date: 2026-06-20

## Scope

This note breaks down why the current VisiPrune single-request run does not
beat the dense FlashAttention2 baseline, then evaluates runtime directions for
turning VisiPrune's theoretical compute reduction into real GEMV speedup.

No production runtime code is implemented here. The only new code artifact is a
microbenchmark used to estimate GEMV batching and fusion behavior:

```text
code/benchmark_gemv_concurrency.py
output/gemv_concurrency_bench.json
output/gemv_concurrency_bench.csv
output/kernel_family_summary.json
code/analyze_decode_iterations.py
output/decode_iteration_kernel_breakdown.json
output/decode_iteration_kernel_breakdown.csv
```

## Evidence Used

Workload:

```text
model: liuhaotian/llava-v1.5-7b
prompt: Describe the image briefly.
prefill length after image expansion: 624
max_new_tokens: 32
decode forward calls: 31
GPU: NVIDIA GeForce RTX 4090
torch: 2.12.0+cu132
flash_attn: 2.8.3.post1
```

Measured baselines:

```text
dense-fa2:        dense LLaVA with FlashAttention2, no pruning
visipruner-full:  shallow + middle + deep VisiPrune path
```

## End-to-End Result

Synchronized clock timing:

| method | request ms | generate ms | prefill ms | decode sum ms | selection ms |
|---|---:|---:|---:|---:|---:|
| dense-fa2 | 1140.97 | 1100.62 | 68.95 | 1003.61 | 0.00 |
| VisiPrune full | 1456.85 | 1403.48 | 98.52 | 1265.12 | 8.97 |

Nsight timing:

| method | request NVTX ms | decode NVTX ms | CUDA kernel total ms |
|---|---:|---:|---:|
| dense-fa2 | 1546.85 | 1388.11 | 613.92 |
| VisiPrune full | 1758.21 | 1579.53 | 596.87 |

VisiPrune reduces total CUDA kernel duration by only 17.05 ms versus dense-fa2,
but host/NVTX request time increases by 211.36 ms. The kernel reduction is real
but too small and in the wrong part of the request to improve end-to-end
latency.

## What VisiPrune Actually Removes

Source path:

```text
repo/llava/model/language_model/llava_llama.py
  LlavaLlamaModel.set_num_images(): last_image_token_index = 35 + 576 * images

repo/llava/model/language_model/custom_modeling_llama.py
  FA2VisiPrunerLlamaAttention.forward()
  LlamaModel.forward()
```

For this request, the tracker shows:

```text
middle selection layer: 18
selected visual tokens: 10
deep exit checks: layers 19-27
decode forward calls: 31
```

The resulting prefill sequence-length timeline is:

| layer range | sequence length | reason |
|---|---:|---|
| 0-18 | 624 | full prompt + image tokens |
| 19-27 | 58 | 35 prefix + 10 selected image tokens + 13 text tail |
| 28-31 | 48 | deep exit removes remaining selected image tokens |

That gives the following theoretical prefill reductions:

```text
linear token-layer rows:
  dense:      32 * 624 = 19968
  VisiPrune: 19 * 624 + 9 * 58 + 4 * 48 = 12570
  reduction: 37.0%

attention quadratic token pairs:
  dense:      32 * 624^2 = 12460032
  VisiPrune: 19 * 624^2 + 9 * 58^2 + 4 * 48^2 = 7437636
  reduction: 40.3%
```

This looks meaningful for prefill. It does not solve the measured bottleneck:
the request is dominated by 31 autoregressive decode steps, where each step has
only one query token and still runs 32 layers of small projection GEMV.

## Kernel Family Breakdown

Nsight CUDA kernel families:

| family | dense-fa2 ms | dense-fa2 pct | VisiPrune ms | VisiPrune pct |
|---|---:|---:|---:|---:|
| GEMV decode cuBLAS | 460.84 | 75.1% | 467.34 | 78.3% |
| tensorcore GEMM | 59.55 | 9.7% | 44.41 | 7.4% |
| elementwise/norm/activation | 39.22 | 6.4% | 46.31 | 7.8% |
| copy/gather/cat | 32.48 | 5.3% | 27.91 | 4.7% |
| FlashAttention | 15.82 | 2.6% | 0.00 | 0.0% |
| selection/reduce/scan | 5.44 | 0.9% | 5.80 | 1.0% |
| softmax | 0.32 | 0.1% | 3.34 | 0.6% |

Key observations:

- VisiPrune reduces tensorcore GEMM time by 15.14 ms, consistent with shorter
  prefill sequences after pruning.
- The dominant GEMV bucket does not decrease; it slightly increases from
  460.84 ms to 467.34 ms.
- Single-request decode remains a long chain of small GEMV kernels. This is the
  part that must be transformed if VisiPrune is expected to beat dense-FA.

## Decode Iteration Breakdown

`analyze_decode_iterations.py` attributes CUDA kernels overlapping each
`:visprune.forward_decode` NVTX range from the VisiPrune Nsight SQLite trace.

Across the 31 decode iterations:

| metric | mean ms | min ms | max ms | stdev ms |
|---|---:|---:|---:|---:|
| forward_decode NVTX range | 50.96 | 49.44 | 56.94 | 1.40 |
| CUDA kernel total inside range | 17.23 | 17.10 | 17.62 | 0.10 |

Mean kernel composition per decode iteration:

| family | mean ms / iter | pct of kernel time |
|---|---:|---:|
| GEMV decode cuBLAS | 14.86 | 86.2% |
| elementwise/norm/activation | 1.28 | 7.5% |
| copy/gather/cat | 0.82 | 4.8% |
| selection/reduce/scan | 0.15 | 0.9% |
| softmax | 0.07 | 0.4% |

This rules out a one-off bad decode token. The bottleneck is stable across
iterations: each generated token repeats almost the same chain of batch-1 GEMV
kernels.

## GEMV Concurrency Microbenchmark

The benchmark uses LLaVA-7B projection shapes on RTX 4090. It compares issuing
`B` independent batch-1 GEMV operations with one batched GEMM containing `B`
rows.

Single-projection row batching:

| shape | B=1 TFLOPS | B=8 TFLOPS | B=16 TFLOPS | B=32 TFLOPS | B=64 TFLOPS |
|---|---:|---:|---:|---:|---:|
| 4096x4096 attention projection | 1.7 | 12.9 | 25.9 | 41.0 | 83.1 |
| 4096x11008 MLP up/gate | 0.9 | 7.3 | 14.4 | 28.6 | 51.3 |
| 11008x4096 MLP down | 0.9 | 7.3 | 14.2 | 24.5 | 49.5 |

Measured speedup versus serial batch-1 GEMV:

| shape | B=8 | B=16 | B=32 | B=64 |
|---|---:|---:|---:|---:|
| 4096x4096 attention projection | 9.35x | 18.66x | 29.46x | 60.45x |
| 4096x11008 MLP up/gate | 8.04x | 15.93x | 31.41x | 56.25x |
| 11008x4096 MLP down | 8.08x | 15.82x | 27.19x | 55.04x |

Projection concat alone is not enough:

| fusion | B=1 | B=8 | B=32 | B=64 |
|---|---:|---:|---:|---:|
| QKV concat, 3x4096x4096 | 1.19x | 1.01x | 1.10x | 1.12x |
| MLP gate/up concat, 2x4096x11008 | 1.01x | 1.00x | 0.99x | 0.96x |

The practical message is direct: larger row batches are the real speed lever.
Simple projection concat may reduce launches, but it does not approach the
speedup needed to make VisiPrune beat dense-FA for single-request decode.

## Runtime Design Options

### 1. Cross-Request Decode Batching

This is the strongest path.

Batch together different requests at the same decode layer and projection, so
each projection sees shape `[num_active_requests, hidden] @ W` rather than many
independent `[1, hidden] @ W` GEMVs.

Required runtime changes:

- Keep FlashAttention2 for attention.
- Add a decode scheduler that groups active requests by layer and projection.
- Use row-batched GEMM for q/k/v/o and MLP gate/up/down projections.
- Scatter results back into each request's state after each layer.
- Carry per-request KV-cache length metadata because VisiPrune creates ragged
  layer-wise KV lengths.
- Bucket requests with similar active sequence length where needed for FA2 or
  use varlen attention interfaces.

Expected behavior:

- For one isolated request, this gives no benefit.
- For concurrency >= 8, the microbenchmark shows projection throughput can move
  from roughly 1 TFLOP/s to 7-13 TFLOP/s.
- For concurrency >= 16-32, projection GEMV can become much less dominant.
- VisiPrune can then help because it reduces the per-request attention/KV work
  while the projection work is amortized by batching.

This is the most plausible route to beat dense-FA in a serving setting.

### 2. Cross-Token Batching

Prefill already uses cross-token batching. VisiPrune's prefill token reduction
does work here, but prefill is not the end-to-end bottleneck for this request.

Exact greedy decode cannot batch future tokens from the same request because
token `t+1` depends on token `t`.

Usable variants:

- speculative decoding: draft multiple candidate tokens, verify them in a
  batched target-model pass;
- tree decoding: verify multiple branches together;
- multi-query workloads where multiple independent continuations share the same
  image/prompt prefix.

VisiPrune can combine with these, but this is a different runtime mode than the
current exact single-request greedy decode.

### 3. Cross-Layer Concurrency

Naive cross-layer concurrency is not viable because layer `l+1` depends on the
output of layer `l`.

What is viable:

- pipeline different requests so request A at layer `l` and request B at layer
  `l` can be batched when their wavefronts align;
- fuse or overlap independent projections inside one layer.

Projection fusion alone is low priority. The microbenchmark shows QKV concat is
only about 1.0-1.2x, and gate/up concat is approximately neutral on this GPU.
It can clean up launches, but it cannot recover the 27% end-to-end VisiPrune
regression by itself.

### 4. VisiPrune-Specific Compaction

Current code physically compacts hidden states during prefill using gather/cat
when `important_vision_tokens` is available. This is correct but not enough.

A runtime-oriented design should:

- make the post-layer-18 compaction a planned state transition rather than
  repeated ad hoc tensor cat/gather;
- store per-layer active token maps and compacted KV-cache metadata;
- keep dense contiguous tensors for the active sequence whenever possible;
- avoid creating many small auxiliary kernels in the decode hot path;
- expose active lengths to the scheduler so row batching and varlen FA2 can be
  selected without Python-side branching per request.

### 5. "Fused Dense-FA" Target

The target should not be "replace dense-FA attention." Dense-FA/FA2 is already
good at attention. The target should be:

```text
FA2 attention + grouped/batched projection GEMM + VisiPrune compact KV lengths
```

For a serving runtime, the dense-FA baseline should become the fallback when
concurrency is low or VisiPrune selects too few savings. VisiPrune should be
enabled only when the scheduler can preserve high row-batch occupancy for the
projection GEMMs.

## Practical Recommendation

For this codebase and workload, the priority order is:

1. Measure with per-layer/per-projection NVTX to confirm exactly which
   projection groups dominate each decode layer.
2. Evaluate multi-request decode batching at concurrency 4/8/16/32.
3. Keep FA2 attention and VisiPrune compaction, but move projection execution
   to grouped row-batched GEMM.
4. Add QKV/gate-up fusion only after row batching; by itself it is not enough.
5. Consider speculative or tree decoding only if single-request latency is the
   primary target and batching across requests is unavailable.

The current single-request exact decode path should not be expected to beat
dense-FA. It lacks the row concurrency required to make GEMV efficient, and the
measured kernel profile shows that VisiPrune's saved prefill compute lands in a
small part of the total request.
