# input2_layer27 Runtime Module Split

This split is derived directly from sampled `module_*` columns in this layer's dispatch rows.

| First op | Last op | Ops | Op indices | Module | Type | Tensor ID inputs | Tensor ID outputs | Forward source | Top ATen ops |
|---:|---:|---:|---|---|---|---|---|---|---|
| 1 | 8 | 8 | `1,2,3,4,5,6,7,8` | `model.layers.27.input_layernorm` | `LlamaRMSNorm` | `t00002605, t00002279` | `t00002613` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:139` | `mul.Tensor` x2, `to.dtype` x2, `add.Tensor` x1, `mean.dim` x1, `pow.Tensor_Scalar` x1, `rsqrt.default` x1 |
| 9 | 9 | 1 | `9` | `model.layers.27.self_attn.q_proj` | `Linear` | `t00002613, t00002281` | `t00002614` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 10 | 10 | 1 | `10` | `model.layers.27.self_attn.k_proj` | `Linear` | `t00002613, t00002283` | `t00002615` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 11 | 11 | 1 | `11` | `model.layers.27.self_attn.v_proj` | `Linear` | `t00002613, t00002285` | `t00002616` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 12 | 60 | 41 | `12,13,14,15,16,17,18,19,20,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60` | `model.layers.27.self_attn` | `VisiPrunerLlamaAttention` | `t00002614, t00002615, t00002616, t00002481, t00002627, t00002628, t00002318, t00002292` | `t00002625, t00002658` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:614` | `transpose.int` x5, `add.Tensor` x4, `cat.default` x4, `mul.Tensor` x4, `slice.Tensor` x4, `view.default` x3 |
| 21 | 28 | 8 | `21,22,23,24,25,26,27,28` | `model.layers.27.self_attn.rotary_emb` | `LlamaRotaryEmbedding` | `t00002625, t00002297, t00002299` | `` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:175` | `item.default` x2, `slice.Tensor` x2, `to.dtype` x2, `gt.Scalar` x1, `is_nonzero.default` x1 |
| 61 | 61 | 1 | `61` | `model.layers.27.self_attn.o_proj` | `Linear` | `t00002658, t00002353` | `t00002660` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 62 | 76 | 2 | `62,76` | `model.layers.27` | `LlamaDecoderLayer` | `t00002605, t00002660, t00002674` | `t00002675` | `/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py:881` | `add.Tensor` x2 |
| 63 | 70 | 8 | `63,64,65,66,67,68,69,70` | `model.layers.27.post_attention_layernorm` | `LlamaRMSNorm` | `t00002661, t00002363` | `t00002669` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:139` | `mul.Tensor` x2, `to.dtype` x2, `add.Tensor` x1, `mean.dim` x1, `pow.Tensor_Scalar` x1, `rsqrt.default` x1 |
| 71 | 71 | 1 | `71` | `model.layers.27.mlp.gate_proj` | `Linear` | `t00002669, t00002365` | `t00002670` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 72 | 72 | 1 | `72` | `model.layers.27.mlp.act_fn` | `SiLU` | `t00002670` | `t00002671` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py:471` | `silu.default` x1 |
| 73 | 73 | 1 | `73` | `model.layers.27.mlp.up_proj` | `Linear` | `t00002669, t00002368` | `t00002672` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 74 | 74 | 1 | `74` | `model.layers.27.mlp` | `LlamaMLP` | `t00002671, t00002672` | `t00002673` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:277` | `mul.Tensor` x1 |
| 75 | 75 | 1 | `75` | `model.layers.27.mlp.down_proj` | `Linear` | `t00002673, t00002371` | `t00002674` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
