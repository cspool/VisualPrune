# input1_layer28 Runtime Module Split

This split is derived directly from sampled `module_*` columns in this layer's dispatch rows.

| First op | Last op | Ops | Op indices | Module | Type | Tensor ID inputs | Tensor ID outputs | Forward source | Top ATen ops |
|---:|---:|---:|---|---|---|---|---|---|---|
| 1 | 8 | 8 | `1,2,3,4,5,6,7,8` | `model.layers.28.input_layernorm` | `LlamaRMSNorm` | `t00002374, t00002382` | `t00002383` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:139` | `mul.Tensor` x2, `to.dtype` x2, `add.Tensor` x1, `mean.dim` x1, `pow.Tensor_Scalar` x1, `rsqrt.default` x1 |
| 9 | 9 | 1 | `9` | `model.layers.28.self_attn.q_proj` | `Linear` | `t00002383, t00002384` | `t00002385` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 10 | 10 | 1 | `10` | `model.layers.28.self_attn.k_proj` | `Linear` | `t00002383, t00002386` | `t00002387` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 11 | 11 | 1 | `11` | `model.layers.28.self_attn.v_proj` | `Linear` | `t00002383, t00002388` | `t00002389` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 12 | 67 | 48 | `12,13,14,15,16,17,18,19,20,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67` | `model.layers.28.self_attn` | `VisiPrunerLlamaAttention` | `t00002385, t00002387, t00002389, t00002396, t00002402, t00002404, t00002426, t00000057` | `t00002399, t00002433` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:614` | `add.Tensor` x6, `mul.Tensor` x5, `transpose.int` x5, `select.int` x4, `slice.Tensor` x4, `view.default` x3 |
| 21 | 28 | 8 | `21,22,23,24,25,26,27,28` | `model.layers.28.self_attn.rotary_emb` | `LlamaRotaryEmbedding` | `t00002399, t00002401, t00002403` | `` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:175` | `item.default` x2, `slice.Tensor` x2, `to.dtype` x2, `gt.Scalar` x1, `is_nonzero.default` x1 |
| 68 | 68 | 1 | `68` | `model.layers.28.self_attn.o_proj` | `Linear` | `t00002433, t00002442` | `t00002443` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 69 | 83 | 2 | `69,83` | `model.layers.28` | `LlamaDecoderLayer` | `t00002374, t00002443, t00002461` | `t00002462` | `/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py:881` | `add.Tensor` x2 |
| 70 | 77 | 8 | `70,71,72,73,74,75,76,77` | `model.layers.28.post_attention_layernorm` | `LlamaRMSNorm` | `t00002444, t00002452` | `t00002453` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:139` | `mul.Tensor` x2, `to.dtype` x2, `add.Tensor` x1, `mean.dim` x1, `pow.Tensor_Scalar` x1, `rsqrt.default` x1 |
| 78 | 78 | 1 | `78` | `model.layers.28.mlp.gate_proj` | `Linear` | `t00002453, t00002454` | `t00002455` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 79 | 79 | 1 | `79` | `model.layers.28.mlp.act_fn` | `SiLU` | `t00002455` | `t00002456` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py:471` | `silu.default` x1 |
| 80 | 80 | 1 | `80` | `model.layers.28.mlp.up_proj` | `Linear` | `t00002453, t00002457` | `t00002458` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
| 81 | 81 | 1 | `81` | `model.layers.28.mlp` | `LlamaMLP` | `t00002456, t00002458` | `t00002459` | `/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py:277` | `mul.Tensor` x1 |
| 82 | 82 | 1 | `82` | `model.layers.28.mlp.down_proj` | `Linear` | `t00002459, t00002460` | `t00002461` | `/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py:130` | `linear.default` x1 |
