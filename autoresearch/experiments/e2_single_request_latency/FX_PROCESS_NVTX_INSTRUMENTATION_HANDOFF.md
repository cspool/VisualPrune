# FX Process NVTX Instrumentation Handoff

## Source FX Artifacts

FX process data source:

```text
workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/fx_layer_events.csv
workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/*/fx_process_reconstruction.json
workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/*/fx_process_reconstruction.md
```

This handoff targets the VisiPrune eager full variant:

```text
config: visipruner-full
attention implementation: eager
model attention class: llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention
decoder layer class: llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer
```

Representative FX events used for process instrumentation:

| phase | forward ids | representative layers | q/kv pattern |
|---|---|---|---|
| prefill | 1 | 0, 5-18 | q=624, kv=624 |
| prefill | 1 | 19-27 | q=58, kv=58 |
| prefill | 1 | 28 | q=48, kv=48 |
| decode | 2 | 18 | q=1, kv=625 |
| decode | 2 | 19, 27 | q=1, kv=59 |
| decode | 2 | 28, 31 | q=1, kv=49 |
| decode | 32 | 18 | q=1, kv=655 |
| decode | 32 | 19, 27 | q=1, kv=89 |
| decode | 32 | 28, 31 | q=1, kv=79 |

The non-metadata FX process stages are:

```text
input_rmsnorm
qkv_projection
rope
attention_scores
attention_output
visual_process
output_projection
post_attention_rmsnorm
mlp
```

`inputs` and `layer_output` are metadata/pass-through stages and are not instrumented as GPU-launching process ranges.

## Execution Reproducibility Contract

Strict process attribution requires the FX trace and Nsight trace to describe the same execution conditions.

| field | required value |
|---|---|
| model checkpoint or weight source | `liuhaotian/llava-v1.5-7b` from local HF cache |
| input image/text/prompt source | image `/workspace/VisiPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg`; prompt `Describe the image briefly.` |
| generation arguments | `max_new_tokens=32`, `temperature=0.0`, `conv_mode=llava_v1`, `use_cache=True` |
| backend variant | `visipruner-full` |
| dtype and device | fp16 model tensors on CUDA device selected by `CUDA_VISIBLE_DEVICES`, default GPU env `GPU=1` |
| attention implementation | eager VisiPruner attention, not FA2/VP-FA |
| pruning configuration | `mode=["shallow","middle","deep"]`, `shallow_mid_layer=6`, `layer_threshold=0.995`, `tokens_threshold=0.2` |
| cache state policy | one warmup request outside measured request; measured request starts after `torch.cuda.empty_cache()` and synchronize |
| warmup protocol | `WARMUP_ITERS=1` unless explicitly overridden |
| random seed or determinism controls | no explicit seed in current runner; generation is deterministic because `temperature=0.0` |
| code revision or patch state | this handoff requires `profile_visprune_single_request.py` plus `custom_modeling_llama.py` hook changes in this worktree |
| environment notes that can change dispatch/backend behavior | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=/workspace/VisiPrune/models`; do not add FA2/VP-FA env overrides for this eager run |

If any of these conditions differ, downstream attribution must mark the result `reproducibility_mismatch` rather than strict FX-process attribution.

## Instrumented Code Changes

| file | symbol | change | reason |
|---|---|---|---|
| `repo/llava/model/language_model/custom_modeling_llama.py` | `_visipruner_fx_process_range` | Added a default-disabled context helper that returns `nullcontext()` unless E2 profiler installs `_visprune_fx_process_range`. | Let the E2 profiler place process NVTX ranges at true launch-owning code locations without changing normal inference behavior. |
| `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | Wrapped qkv projection, RoPE, QK/mask/softmax/cache update, attention-weighted V, visual selection, and output projection fragments. | Map FX attention-side stages to CUDA-launching eager attention code. |
| `repo/llava/model/language_model/custom_modeling_llama.py` | `LlamaDecoderLayer.forward` | Wrapped input RMSNorm, attention residual fragment, post-attention RMSNorm, MLP fragment, and final residual fragment. | Map layer-side FX stages and cross-function residual fragments. |
| `autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py` | `LatencyRecorder.nvtx_range` | Added an NVTX-only context that does not synchronize, time, or write clock range stats. | Ensure process instrumentation only introduces NVTX markers. |
| `autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py` | `FX_PROCESS_REPRESENTATIVE_EVENTS` | Added exact representative event set for VisiPrune eager full sampled FX layers. | Restrict process NVTX to representative layers instead of all layers. |
| `autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py` | `patch_model_for_ranges` | Added `fx_process_profile` flag, active `forward_id`, active layer event context, and NVTX-only range factory installed on decoder layers/attention modules. | Connect runtime layer occurrences to FX representative events and produce deterministic process NVTX names without changing timing behavior. |
| `autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py` | `write_outputs` | Added `forward_id` to layer event CSV rows. | Preserve mapping from runtime layer occurrence to FX `input{forward_id}_layer{layer}` representative event. |
| `autoresearch/experiments/e2_single_request_latency/code/run_clock_layer_profile_single_request.sh` | script args | Passes `--fx-process-profile "${FX_PROCESS_PROFILE:-off}"`. | Keep generic runner default off but allow SAME_INPUT eager full to enable process ranges. |
| `autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh` | script args | Passes `--fx-process-profile "${FX_PROCESS_PROFILE:-off}"`. | Make Nsight traces capable of collecting process-level NVTX ranges. |
| `autoresearch/experiments/e2_single_request_latency/code/run_same_input_three_way_layer_retest.sh` | `run_one` | Reads `FX_PROCESS_PROFILE` from the environment unless an explicit fifth argument overrides it. | Make `FX_PROCESS_PROFILE=on` the only runtime switch that unlocks process NVTX insertion. |

## Process Range Inventory

Range name template:

```text
visprune.fx_process.layer{layer:02d}.{phase}.fwd{forward_id:02d}.event{event_id:04d}.{process_id}.{slug}[.part{part:02d}]
```

`event_id` is the runtime layer event id written by `profile_visprune_single_request.py`; `forward_id` follows FX trace naming, where prefill is `input1`, first decode is `input2`, and the representative late decode is `input32`.

| variant_scope | phase | layer_or_layer_pattern | process_id | process_title | fragment_id | aggregation_key | fx_nodes | fx_op_families | expected_kernel_families | torch_code_path | instrumented_file | instrumented_symbol | nvtx_range_name | range_parent | range_guard_or_flag | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| visipruner-full | prefill/decode | representative layers listed above | input_rmsnorm | Input RMSNorm | whole | input_rmsnorm | `_to_copy`, pow, mean, add, rsqrt, mul | norm, elementwise, dtype copy | elementwise_norm_activation | `LlamaDecoderLayer.forward -> self.input_layernorm` | `repo/llava/model/language_model/custom_modeling_llama.py` | `LlamaDecoderLayer.forward` | `...input_rmsnorm.input_rmsnorm` | `visprune.layerXX.phase` | `FX_PROCESS_PROFILE=on`, representative event match | instrumented | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| visipruner-full | prefill/decode | representative layers listed above | qkv_projection | Q/K/V projection and head reshape | whole | qkv_projection | three linear projections, view, transpose | GEMM/GEMV, view/transpose | gemm_tensorcore, gemv_decode_cublas, elementwise/layout | `VisiPrunerLlamaAttention.forward -> q_proj/k_proj/v_proj + head reshape` | `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | `...qkv_projection.qkv_projection` | `visprune.layerXX.phase.attn` | same | instrumented | Prefill should mainly be GEMM; decode q_len=1 may be GEMV/cublas. |
| visipruner-full | prefill/decode | representative layers listed above | rope | RoPE position embedding | whole | rope | rotary table lookup, mul, slice, neg, cat, add | index, elementwise, layout | elementwise_norm_activation, copy_gather_cat | `VisiPrunerLlamaAttention.forward -> rotary_emb + apply_rotary_pos_emb` | `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | `...rope.rope` | `visprune.layerXX.phase.attn` | same | instrumented | Includes `kv_seq_len` bookkeeping; GPU work is RoPE-related tensor ops. |
| visipruner-full | prefill/decode | representative layers listed above | attention_scores | QK scores, mask, softmax | whole | attention_scores | cache update, repeat_kv, QK matmul, mask add, softmax, shallow mask edits | BMM/GEMM, softmax, memory/index, elementwise | gemm_tensorcore, gemv_decode_cublas, softmax, copy_gather_cat, elementwise_norm_activation | `VisiPrunerLlamaAttention.forward -> past_key_value.update/repeat_kv/matmul/mask/softmax/shallow edits` | `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | `...attention_scores.qk_mask_softmax` | `visprune.layerXX.phase.attn` | same | instrumented | Decode cache K/V concatenation appears in this FX stage; value cache work is included here because it is launched by the same cache update block. |
| visipruner-full | prefill/decode | representative layers listed above | attention_output | Attention-weighted V and hidden reshape | whole | attention_output | dropout, attn-weighted V matmul, transpose, contiguous, reshape | BMM/GEMM, layout/copy | gemm_tensorcore, gemv_decode_cublas, copy_gather_cat | `VisiPrunerLlamaAttention.forward -> dropout + attn_weights @ value_states + output reshape` | `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | `...attention_output.weighted_value` | `visprune.layerXX.phase.attn` | same | instrumented | Includes value-side attention matmul and layout materialization. |
| visipruner-full | prefill | layers 7-27 | visual_process | Visual-related value-aware process | whole | visual_process | select, permute, contribution multiply, norm/cosine, threshold, nonzero | reduction, memory/index, elementwise | selection_reduce_scan, copy_gather_cat, elementwise_norm_activation | `VisiPrunerLlamaAttention.forward -> value_aware_token_selection(...)` | `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | `...visual_process.value_aware_selection` or `...visual_process.value_aware_verification` | `visprune.layerXX.prefill.attn` | same | instrumented | This stage is only expected for sampled prefill layers whose FX reconstruction contains `visual_process`; decode sampled layers do not have this process. |
| visipruner-full | prefill/decode | representative layers listed above | output_projection | Attention output projection and residual | part01 | output_projection | o_proj linear | GEMM/GEMV | gemm_tensorcore, gemv_decode_cublas | `VisiPrunerLlamaAttention.forward -> self.o_proj` | `repo/llava/model/language_model/custom_modeling_llama.py` | `VisiPrunerLlamaAttention.forward` | `...output_projection.o_proj.part01` | `visprune.layerXX.phase.attn` | same | instrumented | First fragment of cross-function FX process. |
| visipruner-full | prefill/decode | representative layers listed above | output_projection | Attention output projection and residual | part02 | output_projection | residual add | elementwise | elementwise_norm_activation | `LlamaDecoderLayer.forward -> residual + hidden_states` | `repo/llava/model/language_model/custom_modeling_llama.py` | `LlamaDecoderLayer.forward` | `...output_projection.attention_residual.part02` | `visprune.layerXX.phase` | same | instrumented | Second fragment; downstream must aggregate with part01 by `aggregation_key=output_projection`. |
| visipruner-full | prefill/decode | representative layers listed above | post_attention_rmsnorm | Post-attention RMSNorm | whole | post_attention_rmsnorm | `_to_copy`, pow, mean, add, rsqrt, mul | norm, elementwise, dtype copy | elementwise_norm_activation | `LlamaDecoderLayer.forward -> self.post_attention_layernorm` | `repo/llava/model/language_model/custom_modeling_llama.py` | `LlamaDecoderLayer.forward` | `...post_attention_rmsnorm.post_attention_rmsnorm` | `visprune.layerXX.phase` | same | instrumented | CPU range should launch post-attention norm kernels or fused norm kernels. |
| visipruner-full | prefill/decode | representative layers listed above | mlp | MLP and final residual | part01 | mlp | gate/up/down projections, SiLU, multiply | GEMM/GEMV, activation, elementwise | gemm_tensorcore, gemv_decode_cublas, elementwise_norm_activation | `LlamaDecoderLayer.forward -> self.mlp` | `repo/llava/model/language_model/custom_modeling_llama.py` | `LlamaDecoderLayer.forward` | `...mlp.mlp.part01` | `visprune.layerXX.phase` | same | instrumented | First fragment of MLP FX process. |
| visipruner-full | prefill/decode | representative layers listed above | mlp | MLP and final residual | part02 | mlp | residual add | elementwise | elementwise_norm_activation | `LlamaDecoderLayer.forward -> residual + hidden_states` | `repo/llava/model/language_model/custom_modeling_llama.py` | `LlamaDecoderLayer.forward` | `...mlp.final_residual.part02` | `visprune.layerXX.phase` | same | instrumented | Second fragment; downstream must aggregate with part01 by `aggregation_key=mlp`. |

## Range Naming Contract

Process ranges are emitted when the runtime process NVTX switch is enabled and the active layer occurrence has representative FX process data:

```text
FX_PROCESS_PROFILE == "on"
(forward_id, layer_idx, phase) is in FX_PROCESS_REPRESENTATIVE_EVENTS
```

`FX_PROCESS_PROFILE=on` is the only switch that unlocks source-code NVTX insertion. The `FX_PROCESS_REPRESENTATIVE_EVENTS` check is not an extra variant guard; it prevents process labels from being emitted for layer occurrences whose FX process mapping is unavailable. This handoff's strict attribution contract still targets the eager `visipruner-full` run listed above. If the switch is used with another backend or input/weight state, downstream reports must mark the result as non-strict or `reproducibility_mismatch` unless matching FX artifacts are regenerated.

Process ranges are NVTX-only. They are expected in Nsight NVTX events and NVTX-kernel projection tables, but they are not written to the clock `ranges.csv` and do not add `torch.cuda.synchronize()` calls.

Forward id contract:

```text
prefill forward -> fwd01
first decode forward -> fwd02
late representative decode forward -> fwd32
```

Examples expected in Nsight NVTX events:

```text
visprune.fx_process.layer18.prefill.fwd01.event0018.attention_scores.qk_mask_softmax
visprune.fx_process.layer19.prefill.fwd01.event0019.visual_process.value_aware_verification
visprune.fx_process.layer19.decode.fwd02.event0051.qkv_projection.qkv_projection
visprune.fx_process.layer31.decode.fwd32.event1023.mlp.final_residual.part02
```

The exact `eventNNNN` value depends on runtime output length and layer-event ordering. Join process ranges to layer ranges by timestamp nesting and to FX representative events by `(forward_id, layer, phase)`.

## Expected Trace Outputs

Run from:

```bash
cd /workspace/VisiPrune
CONFIG=visipruner-full \
TAG=nsys_sameinput_visipruner_full_eager_32tok \
FX_PROCESS_PROFILE=on \
autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
```

Or run the three-way SAME_INPUT workflow:

```bash
cd /workspace/VisiPrune
FX_PROCESS_PROFILE=on \
autoresearch/experiments/e2_single_request_latency/code/run_same_input_three_way_layer_retest.sh
```

Expected eager full trace outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok_stats_nvtx_kern_sum.csv
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok_stats_cuda_gpu_kern_sum.csv
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok_layer_events.csv
```

Expected clock outputs with process ranges:

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/clock_sameinput_visipruner_full_eager_32tok.json
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/clock_sameinput_visipruner_full_eager_32tok_ranges.csv
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/clock_sameinput_visipruner_full_eager_32tok_layer_events.csv
```

Downstream process attribution should create:

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv
autoresearch/experiments/e2_single_request_latency/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md
autoresearch/experiments/e2_single_request_latency/SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md
```

## Validation Performed

The following checks were run and passed after the code changes:

```bash
python -m py_compile \
  autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py \
  repo/llava/model/language_model/custom_modeling_llama.py
bash -n \
  autoresearch/experiments/e2_single_request_latency/code/run_clock_layer_profile_single_request.sh \
  autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh \
  autoresearch/experiments/e2_single_request_latency/code/run_same_input_three_way_layer_retest.sh
rg -n 'visprune.fx_process|_visprune_fx_process_range|FX_PROCESS_PROFILE' \
  autoresearch/experiments/e2_single_request_latency/code \
  repo/llava/model/language_model/custom_modeling_llama.py
test -f autoresearch/experiments/e2_single_request_latency/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
git diff --check -- \
  autoresearch/experiments/e2_single_request_latency/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md \
  autoresearch/experiments/e2_single_request_latency/README.md \
  autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py \
  autoresearch/experiments/e2_single_request_latency/code/run_clock_layer_profile_single_request.sh \
  autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh \
  autoresearch/experiments/e2_single_request_latency/code/run_same_input_three_way_layer_retest.sh \
  repo/llava/model/language_model/custom_modeling_llama.py
```

Representative event coverage was also checked against
`workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/fx_layer_events.csv`:

```text
constant_count 35
expected_count 35
missing []
extra []
```

Strict process Nsight profiling was run after this instrumentation handoff with:

```bash
CONFIG=visipruner-full \
TAG=nsys_sameinput_visipruner_full_eager_32tok \
FX_PROCESS_PROFILE=on \
MAX_NEW_TOKENS=32 \
WARMUP_ITERS=1 \
GPU=1 \
bash autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
```

The rerun produced 371 `visprune.fx_process.*` NVTX ranges, 301 aggregated process rows, and 35 parent layer ranges in the representative traced scope.

## Open Risks

- The model source hook is default-disabled unless `FX_PROCESS_PROFILE=on`. When enabled, the hook only enters NVTX-only context managers through the E2 recorder; it must not add synchronization, clock timing, tensor computation, pruning decisions, or backend changes. Re-run syntax checks before profiling.
- `attention_scores` intentionally includes decode cache update and `repeat_kv` because the FX decode stage includes cache K/V materialization before QK/softmax. Downstream reports should state this process boundary.
- `output_projection` and `mlp` are cross-function FX processes. Downstream must aggregate their fragments by `aggregation_key`; single-fragment rows are incomplete process timings.
- `visual_process` is only expected for representative prefill layers whose FX reconstruction includes value-aware selection or verification. Missing visual process ranges on decode events are expected.
- `eventNNNN` depends on runtime layer-event ordering. Use `(forward_id, layer, phase)` plus timestamp nesting rather than event id alone when joining to FX artifacts.
