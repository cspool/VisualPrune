# VP-FA Kernel Checkpoint

Date: 2026-07-01 UTC

## Status

The source-level patched FlashAttention VP-FA implementation and its build plan
are cancelled for the current experiment track.

The active VP-FA full experiment direction is now:

```text
Use Triton kernels for full VisiPrune prefill inference.
Do not require or call a patched flash_attn_2_cuda.vp_fwd extension.
```

This replaces the previous plan to rebuild and verify
`third_party/flash-attention-v2.8.3.post1-vpfa`.

## Cancelled Work

The following previous source-build direction is no longer active:

- modifying FlashAttention C++/CUDA sources under
  `third_party/flash-attention-v2.8.3.post1-vpfa`;
- exporting a Python extension entrypoint named `flash_attn_2_cuda.vp_fwd`;
- rebuilding the patched extension with `setup.py build_ext --inplace`;
- validating native `vp_fwd` no-op equivalence and pre-mask behavior as the
  primary VP-FA path.

The source tree may remain in the worktree as an archived scratch copy, but it
is not part of the current experiment execution path.

## Active Implementation Direction

The active code path is:

- `repo/llava/model/language_model/vp_flash_attention.py`
  - full prefill main attention uses Triton;
  - shallow layers use the existing VisiPrune shallow in-kernel edit path;
  - middle/deep layers use the same Triton causal prefill kernel without
    shallow edits;
  - last-query pruning weights and last-query output are produced by a Triton
    proxy kernel where supported.
- `repo/llava/model/language_model/custom_modeling_llama_decode_optimized.py`
  - routes VP-FA prefill through `vp_flash_attn_prefill()`.
- `repo/llava/model/language_model/decode_optimization.py`
  - keeps `vp-fa` as the experiment-facing backend name, now meaning copied
    optimized modeling with Triton VP-FA prefill rather than a patched FA2
    extension.

## Verification Needed

Before treating the Triton-only VP-FA full path as a performance result, verify:

1. import and syntax checks for the modified modules;
2. Triton prefill numerical agreement against the eager reference for:
   - dense causal prefill without shallow edits;
   - shallow layer 0;
   - shallow layers 1-5;
   - last-query pruning weights and output;
3. model-level generation smoke test for `visipruner-full-vp-fa`;
4. clock and Nsight reruns against same-input dense baselines.

## Verification Completed On 2026-07-01

- `python -m py_compile` passed for the modified VP-FA/modeling/backend modules
  and e2 helper scripts.
- Small CUDA/Triton numerical smoke passed:
  - dense causal prefill max abs diff: `0.00146484375`;
  - shallow layer 0, 7B max abs diff: `0.001953125`;
  - shallow layer 1, 7B max abs diff: `0.0009765625`;
  - last-query weights max abs diff: `0.00018310546875`;
  - last-query output max abs diff: `0.00048828125`.
- Model-level 2-token greedy smoke passed for full VisiPruner:
  `autoresearch/experiments/e2_single_request_latency/output/verify_visipruner_vp_fa_full_triton_smoke_2tok.json`
  has `token_match=true` and `text_match=true` against the eager reference.
- E2 2-token clock smoke produced:
  `autoresearch/experiments/e2_single_request_latency/output/clock_visipruner_full_vpfa_triton_smoke_2tok.json`,
  whose resolved backend reason is `Triton VP-FA prefill/decode`.

Remaining before claiming final performance: rerun 32-token clock and Nsight
against same-input dense baselines after the Triton-only path is warmed and
correctness-checked.

## Important Semantic Note

The full prefill experiment no longer uses source-level FA2 tile skipping. It is
a Triton implementation intended to make the full VisiPrune prefill path
self-contained and easier to iterate on. This means any future claims should be
phrased as "Triton VP-FA prefill" unless a separate native FA2 extension is
reintroduced and verified.
