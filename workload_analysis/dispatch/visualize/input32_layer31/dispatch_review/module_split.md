# input32_layer31 Runtime Module Split

This split is derived directly from sampled `module_*` columns in this layer's dispatch rows.

| First op | Last op | Ops | Op indices | Module | Type | Tensor ID inputs | Tensor ID outputs | Forward source | Top ATen ops |
|---:|---:|---:|---|---|---|---|---|---|---|
| 1 | 8 | 8 | `1,2,3,4,5,6,7,8` | `model.layers.31.input_layernorm` | `LlamaRMSNorm` | `t00003121, t00002754` | `t00003129` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:139` | `mul.Tensor` x2, `to.dtype` x2, `add.Tensor` x1, `mean.dim` x1, `pow.Tensor_Scalar` x1, `rsqrt.default` x1 |
| 9 | 9 | 1 | `9` | `model.layers.31.self_attn.q_proj` | `Linear` | `t00003129, t00002756` | `t00003130` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 10 | 10 | 1 | `10` | `model.layers.31.self_attn.k_proj` | `Linear` | `t00003129, t00002758` | `t00003131` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 11 | 11 | 1 | `11` | `model.layers.31.self_attn.v_proj` | `Linear` | `t00003129, t00002760` | `t00003132` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 12 | 60 | 41 | `12,13,14,15,16,17,18,19,20,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60` | `model.layers.31.self_attn` | `VisiPrunerLlamaAttention` | `t00003130, t00003131, t00003132, t00002848, t00003143, t00003144, t00003163, t00003165` | `t00003141, t00003176` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:614` | `transpose.int` x5, `add.Tensor` x4, `cat.default` x4, `mul.Tensor` x4, `slice.Tensor` x4, `view.default` x3 |
| 21 | 28 | 8 | `21,22,23,24,25,26,27,28` | `model.layers.31.self_attn.rotary_emb` | `LlamaRotaryEmbedding` | `t00003141, t00002772, t00002774` | `` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:175` | `item.default` x2, `slice.Tensor` x2, `to.dtype` x2, `gt.Scalar` x1, `is_nonzero.default` x1 |
| 61 | 61 | 1 | `61` | `model.layers.31.self_attn.o_proj` | `Linear` | `t00003176, t00002809` | `t00003178` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 62 | 76 | 2 | `62,76` | `model.layers.31` | `LlamaDecoderLayer` | `t00003121, t00003178, t00003192` | `t00003193` | `/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py:881` | `add.Tensor` x2 |
| 63 | 70 | 8 | `63,64,65,66,67,68,69,70` | `model.layers.31.post_attention_layernorm` | `LlamaRMSNorm` | `t00003179, t00002819` | `t00003187` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:139` | `mul.Tensor` x2, `to.dtype` x2, `add.Tensor` x1, `mean.dim` x1, `pow.Tensor_Scalar` x1, `rsqrt.default` x1 |
| 71 | 71 | 1 | `71` | `model.layers.31.mlp.gate_proj` | `Linear` | `t00003187, t00002821` | `t00003188` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 72 | 72 | 1 | `72` | `model.layers.31.mlp.act_fn` | `SiLU` | `t00003188` | `t00003189` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py:471` | `silu.default` x1 |
| 73 | 73 | 1 | `73` | `model.layers.31.mlp.up_proj` | `Linear` | `t00003187, t00002824` | `t00003190` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 74 | 74 | 1 | `74` | `model.layers.31.mlp` | `LlamaMLP` | `t00003189, t00003190` | `t00003191` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:277` | `mul.Tensor` x1 |
| 75 | 75 | 1 | `75` | `model.layers.31.mlp.down_proj` | `Linear` | `t00003191, t00002827` | `t00003192` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
