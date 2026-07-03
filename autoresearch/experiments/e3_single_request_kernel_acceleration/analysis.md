# E3 Initial Analysis

Date: 2026-07-01

## Current Interpretation

The current VisiPrune workload has a real algorithmic reduction, but the
reduction is not aligned with the dominant single-request runtime cost.

From `human_draft.md`, VisiPrune changes the layer schedule from dense
`624-token` processing to:

```text
prefill layer 0-18: 624 tokens
prefill layer 19-27: 58 tokens
prefill layer 28-31: 48 tokens
decode layer 0-18: 624 + t KV cache
decode layer 19-27: 58 + t KV cache
decode layer 28-31: 48 + t KV cache
```

This is meaningful for attention and cache traffic. It does not remove the
per-token 32-layer projection and MLP path.

E2 shows the problem clearly:

```text
dense-fa2 request:      1140.97 ms
dense-fa2 prefill:        68.95 ms
dense-fa2 decode:       1003.61 ms

eager VisiPrune request: 1456.85 ms
eager VisiPrune prefill:   98.52 ms
eager VisiPrune decode:  1265.12 ms
selection:                 8.97 ms
```

Selection is visible but not the bottleneck. Decode is the bottleneck.

Nsight further shows that VisiPrune can reduce total CUDA kernel duration
slightly while still increasing request wall time. This points to small-kernel
launches, host/runtime overhead, and decode GEMV inefficiency rather than
selection cost alone.

## Implication For Kernel Design

The first-order kernel target for single-request acceleration is not a faster
top-k or cosine-similarity kernel. The target is the decode path.

The useful kernel stack should be:

1. ragged single-query attention for VisiPrune compact KV;
2. fused prefill selection and compaction;
3. fused or graph-captured single-token decode blocks;
4. optional speculative verification if exact greedy is relaxed.

K1/K2 are necessary to make VisiPrune's pruning schedule efficient. K3/K4 are
necessary to move single-request end-to-end latency.

## Main Risk

Exact greedy single-request decoding has a hard utilization problem: every
generated token executes large matrix projections with batch dimension 1. A
kernel that only reduces attention work may improve the correct local metric
while barely moving request latency.

This is not a failed experiment if measured carefully. It would quantify the
projection floor and justify either:

- decode-block fusion / CUDA Graph work; or
- a controlled relaxation to speculative/tree verification for single-user
  requests.

## Next Concrete Step

The immediate prerequisite is to finish and verify the Triton-only VP-FA full
prefill path now that the patched FlashAttention build direction has been
cancelled. This removes the old `flash_attn_2_cuda.vp_fwd` dependency from E2
reruns and gives E3 a stable prefill baseline.

After that, the next implementation step should be a measurement-only prototype
before writing custom CUDA:

1. add per-layer/per-projection NVTX ranges around decode QKV/O/MLP if overhead
   is tolerable;
2. split decode kernels by layer regime: full, middle-pruned, deep-removed;
3. confirm how much time is attention/KV-related versus projection-related
   inside each regime.

Only after that should K1 be implemented, because K1's expected request-level
gain depends on the attention/KV share inside decode.
