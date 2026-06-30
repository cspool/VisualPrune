# Layer Reconstruction Audit

This audit re-checks each layer against its own dispatch rows, generated process files, small-shape ONNX manifests, and runtime module split.

## Summary

- layers audited: `35`
- pass: `35`
- fail: `0`

## Issue Counts

- none

## Per-Layer Results

### input1_layer0

- status: `pass`
- rows/modules: `91` / `14`
- dispatch: phase=`prefill`, role=`shallow_or_boundary`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp, full_flow`

### input1_layer10

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer11

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer12

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer13

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer14

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer15

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer16

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer17

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer18

- status: `pass`
- rows/modules: `104` / `14`
- dispatch: phase=`prefill`, role=`middle_select;boundary_before_prune`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer19

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check;boundary_after_prune`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer20

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer21

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer22

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer23

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer24

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer25

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer26

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer27

- status: `pass`
- rows/modules: `100` / `14`
- dispatch: phase=`prefill`, role=`deep_check;boundary_before_prune`, q_len=`58`, kv_len=`58`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer28

- status: `pass`
- rows/modules: `83` / `14`
- dispatch: phase=`prefill`, role=`boundary_after_prune`, q_len=`48`, kv_len=`48`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp, full_flow`

### input1_layer5

- status: `pass`
- rows/modules: `83` / `14`
- dispatch: phase=`prefill`, role=`shallow_or_boundary`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visual_adjust, attention_output, mlp, full_flow`

### input1_layer6

- status: `pass`
- rows/modules: `75` / `14`
- dispatch: phase=`prefill`, role=`shallow_or_boundary`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, attention_output, mlp, full_flow`

### input1_layer7

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer8

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input1_layer9

- status: `pass`
- rows/modules: `97` / `14`
- dispatch: phase=`prefill`, role=`middle_probe`, q_len=`624`, kv_len=`624`
- expected stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, attention, visipruner_similarity_check, attention_output, mlp, full_flow`

### input2_layer18

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`625`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer19

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`59`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer27

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`59`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer28

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`49`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input2_layer31

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`49`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer18

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`655`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer19

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`89`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer27

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`89`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer28

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`79`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`

### input32_layer31

- status: `pass`
- rows/modules: `76` / `14`
- dispatch: phase=`decode`, role=`decode_prune_effect`, q_len=`1`, kv_len=`79`
- expected stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp`
- onnx stages: `input_rmsnorm, qkv_projection, rope, kv_cache_concat, attention, attention_output, mlp, full_flow`
