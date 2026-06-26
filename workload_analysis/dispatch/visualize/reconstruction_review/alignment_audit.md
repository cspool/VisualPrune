# Dispatch Reconstruction Review

This review compares each layer's generated small-tensor ONNX stages against the real TorchDispatch op trace.

## Summary

- layers reviewed: `35`
- matching reconstructions: `35`
- needs revision: `0`
- split aligned: `35`
- process aligned: `35`

## Key Findings

- The base transformer-block stages are present for every layer: RMSNorm, Q/K/V projection, RoPE, attention, attention output, and MLP.
- The previous generated flow is not layer-specific enough for every dispatch trace.
- Decode layers need an explicit K/V cache concatenation stage because dispatch attention has `q_len=1` and `kv_len>1`.
- Middle/deep VisiPrune probe layers need cosine-similarity/check stages when dispatch contains `aten::cosine_similarity`.
- Some layers have no visual attention adjustment, while the generated ONNX still includes `05_visual_adjust.onnx`.
- `input1_layer5` has a clear-only visual adjustment, whereas the generated visual-adjust stage also folds tail visual mass.

## Per-Layer Review

### input1_layer0

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`shallow_or_boundary`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp, full_flow`

### input1_layer10

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer11

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer12

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer13

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer14

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer15

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer16

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer17

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer18

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_select;boundary_before_prune`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer19

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check;boundary_after_prune`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer20

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer21

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer22

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer23

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer24

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer25

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer26

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer27

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`deep_check;boundary_before_prune`, token_state=`middle_pruned`, q_len=`58`, kv_len=`58`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer28

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`boundary_after_prune`, token_state=`deep_removed`, q_len=`48`, kv_len=`48`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp, full_flow`

### input1_layer5

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`shallow_or_boundary`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp, full_flow`

### input1_layer6

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`shallow_or_boundary`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp, full_flow`

### input1_layer7

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer8

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer9

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`prefill`, role=`middle_probe`, token_state=`full_visual`, q_len=`624`, kv_len=`624`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input2_layer18

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`full_cache`, q_len=`1`, kv_len=`625`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer19

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`middle_pruned_cache`, q_len=`1`, kv_len=`59`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer27

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`middle_pruned_cache`, q_len=`1`, kv_len=`59`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer28

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`deep_removed_cache`, q_len=`1`, kv_len=`49`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer31

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`deep_removed_cache`, q_len=`1`, kv_len=`49`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer18

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`full_cache`, q_len=`1`, kv_len=`655`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer19

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`middle_pruned_cache`, q_len=`1`, kv_len=`89`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer27

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`middle_pruned_cache`, q_len=`1`, kv_len=`89`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer28

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`deep_removed_cache`, q_len=`1`, kv_len=`79`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer31

- status: `match`
- split_alignment: `aligned`
- process_alignment: `aligned`
- dispatch: phase=`decode`, role=`decode_prune_effect`, token_state=`deep_removed_cache`, q_len=`1`, kv_len=`79`
- dimensions: hidden=`4096`, heads=`32`, head_dim=`128`, ffn=`11008`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- generated stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`
