# VisiPruner Triton VP-FA: 当前研究计划

Date: 2026-07-01

## Context

E1/E2 已经说明了核心问题：VisiPruner 的算法级 token/KV 缩短是真实的，
但原始 eager attention 路径让它无法和 dense-FA2 做公平的单请求延迟对比。

此前的方案是把 VisiPruner 接到 FlashAttention2：

- dense attention 走 `flash_attn_func()`;
- pruning 决策额外计算最后一个 query 的 `Q_last @ K^T`;
- 后续尝试 source-level patched FlashAttention，暴露
  `flash_attn_2_cuda.vp_fwd`。

这个 source-level patched FlashAttention build 方向现在取消。当前 VP-FA
full 实验不再依赖编译 patched FA2 extension。

## Current Direction

当前目标是让 `visipruner-full-vp-fa` 的 prefill 全部走 Triton 实现：

```text
shallow layers: Triton causal prefill + VisiPruner shallow post-softmax edits
middle/deep layers: Triton dense causal prefill
selection proxy: Triton last-query attention weights/output where shape allows
decode: 保持现有 optimized/eager/FA2 混合路径，后续由 E3 单独处理
```

这意味着当前 VP-FA 结论应表述为 **Triton VP-FA prefill**，不能表述为
native/patched FlashAttention VP-FA。

## Implementation Scope

### Active files

| 文件 | 当前职责 |
|---|---|
| `repo/llava/model/language_model/vp_flash_attention.py` | Triton VP-FA prefill helper；不导入、不调用 `flash_attn_2_cuda.vp_fwd` |
| `repo/llava/model/language_model/custom_modeling_llama_decode_optimized.py` | `FA2VisiPrunerLlamaAttention` prefill 调用 `vp_flash_attn_prefill()` |
| `repo/llava/model/language_model/decode_optimization.py` | 将 `vp-fa` 后端名解析到 copied optimized modeling；语义更新为 Triton prefill |
| `VP_FA_KERNEL_CHECKPOINT.md` | 记录 native FA2 patch/build 取消和当前 Triton-only 状态 |

### Cancelled files/directions

`third_party/flash-attention-v2.8.3.post1-vpfa` 只作为历史 scratch tree 保留。
当前实验不得要求：

- `setup.py build_ext --inplace`;
- patched `flash_attn_2_cuda`；
- `flash_attn_2_cuda.vp_fwd`;
- native no-op/premask equivalence 作为主路径验收。

## Technical Plan

### Step A: Triton full prefill

把 prefill 主 attention 统一到 Triton kernel：

1. 输入布局保持 LLaMA attention 内部格式：
   `query=[B,Hq,Q,D]`, `key/value=[B,Hkv,K,D]`。
2. kernel 内做 causal online softmax。
3. `apply_shallow=True` 时执行 VisiPruner shallow 权重编辑：
   - layer 0, 7B: 视觉 token attention mass 折叠到 token 35；
   - layer 0, 13B: 按官方路径清零 vision-vision / text-half-vision；
   - layer 1-5: 清零 text-to-vision cross attention。
4. `apply_shallow=False` 时同一个 kernel 作为 dense causal Triton prefill。

### Step B: Triton last-query pruning proxy

middle/deep selection 只需要最后一个 query 的 attention weights 和用于
`value_aware_token_selection()` 的最后一个 query output。当前 proxy kernel 应：

1. 计算 `Q_last @ K^T / sqrt(D)`；
2. softmax 后写出 `[B,Hq,1,K]` pruning weights；
3. 同时计算最后一个 query 的 `weights @ V`，避免回退到 PyTorch matmul；
4. 对不支持的长序列形状回退到 torch reference，而不是 silently 出错。

### Step C: verification gates

在把结果写成实验结论前，必须验证：

1. Python import / syntax；
2. dense causal Triton prefill 对齐 eager reference；
3. shallow layer 0、shallow layer 1-5 对齐 eager reference；
4. Triton last-query weights/output 对齐 eager reference；
5. `visipruner-full-vp-fa` 模型级 smoke generation；
6. clock + Nsight 重新跑同输入 dense baseline 和 VP-FA full。

## Experiment Mapping

### E2: single-request latency

E2 保留现有 dense-FA2、dense-FA、eager VisiPruner 和旧 VP-FA 测量结果。
新的 full VP-FA 结果必须在当前 Triton-only prefill 代码上重跑，旧结果不能
作为当前实现的最终性能结论。

### E3: single-request kernel acceleration

E3 继续处理 prefill 之外的主要瓶颈：单请求 decode GEMV、launch/runtime gap、
ragged KV attention 和 CUDA Graph/static executor。Triton prefill 是 E3 的
前置清理项，不是单请求理论加速比的完整答案。

## Expected Outcome

短期目标不是宣称已经接近理论 speedup，而是建立一个可验证、可迭代的 VP-FA
full prefill 路径：

- 不需要 patched FA2 build；
- prefill attention 不再回到 eager `torch.matmul(Q,K^T)` 主路径；
- middle/deep selection proxy 不再物化完整 attention matrix；
- 结果报告明确剩余瓶颈主要在 decode。
