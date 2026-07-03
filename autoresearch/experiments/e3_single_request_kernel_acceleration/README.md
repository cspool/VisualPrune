# E3 Single-Request VisiPrune Kernel Acceleration

E3 records the kernel-level plan for accelerating VisiPrune under a single
request, exact greedy decode setting.

2026-07-01 status: the patched FlashAttention VP-FA source/build direction is
cancelled. The current prerequisite for E3 is a Triton-only full VP-FA prefill
path, so E3 can evaluate remaining single-request bottlenecks without depending
on `flash_attn_2_cuda.vp_fwd`.

The motivating observation is from `workload_analysis/human_draft.md` and the
E1/E2 profiling artifacts: VisiPrune really shortens the post-selection token
and KV-cache schedule, but the measured single-request latency is dominated by
decode-time batch-1 GEMV and launch/runtime overhead. Therefore the experiment
does not target cross-request batching. It asks which single-request kernels or
runtime primitives can expose the pruning benefit.

Start with:

- `protocol.md`: locked experiment plan and hypotheses.
- `analysis.md`: current reasoning, limits, and expected ceiling.
- `code/`: placeholder for prototypes and benchmark scripts.
- `output/`: placeholder for generated measurements.
