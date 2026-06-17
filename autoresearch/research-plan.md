# VisiPruner + FA2: 公平加速研究计划

## Context

**问题**: 当前 VisiPruner 被迫使用 eager attention（`torch.matmul`/`F.softmax`），因为 token pruning 需要访问 attention weights。但 eager attention 比 Flash Attention 2 (FA2) 慢 ~19×（nsys 实测：FA2 79ms vs eager 1490ms）。这导致 VisiPruner 与 Dense-FA2 的对比完全不公平——实际测得的 0.052× 加速比主要来自 attention backend 差异，而非 pruning 本身。

**技术突破点**: `value_aware_token_selection()` 只需要 **最后一个 query token** 的 attention weights (`attn_weights[:, :, -1, :]`)，这是一个 O(L×d) 的向量计算，不需要 O(L²×d) 的完整 attention 矩阵。因此可以：
- 用 FA2 计算完整 attention（快）
- 单独计算 `Q_last @ K^T`（便宜，O(L×d)）用于 pruning 决策
- **永远不需要物化完整的 attention 矩阵**

**目标**: 让 VisiPruner 的 dense attention 部分使用 FA2，仅 pruning 决策使用轻量级 eager 计算，实现公平的 FA2 vs FA2-Pruning 对比。

## Key Technical Design

### 1. FA2-Accelerated Attention with Lightweight Pruning Proxy

**新的 attention forward() 流程**:
```
1. QKV projection (same as before)
2. RoPE (same)
3. KV cache update (same)
4. [NEW] If prefill + middle/deep layer: compute Q_last @ K^T (O(L×d), cheap)
5. [NEW] Run FA2 for full attention: flash_attn_func(Q, K, V) → attn_output
6. [NEW] If prefill + middle/deep layer: use Q_last @ K^T to compute token importance
7. [NEW] If prefill + shallow layer: NO pruning (FA2 doesn't benefit from mask-based pruning)
8. Output projection (same)
```

### 2. Shallow Pruning Removal

Shallow pruning (layers 0-5) modifies post-softmax attention weights:
- Layer 0: Collapse vision-vision attention
- Layers 1-5: Mask text-vision cross attention

FA2 无法做这种 per-cell weight manipulation，且这种 masking 不会减少 FA2 的计算量（FA2 始终做 dense attention）。因此：**在 FA2 版本中完全移除 shallow pruning**。这对输出质量的影响需要在实验中验证，但从计算角度看是必要的简化。

### 3. Middle/Deep Pruning 保留并加速

Middle/Deep pruning 物理上从序列中移除 token：
- Middle (layer 6+): 用 `value_aware_token_selection()` 选择重要 vision tokens
- Deep (layer 6+): 验证所有 vision tokens 是否可被恢复，全部移除

这些操作在 attention 之外（token 级别），与 FA2 完全兼容。且因为序列缩短，后续层的 FA2 会更快。

### 4. 实现文件

| 文件 | 修改内容 |
|------|---------|
| `repo/llava/model/language_model/custom_modeling_llama.py` | 新增 `FA2VisiPrunerLlamaAttention` 类；修改 `LlamaDecoderLayer` 支持动态选择 attention 类 |
| `autoresearch/experiments/e1_baseline/code/bench_fa2_visipruner.py` | 新的 standalone inference + profiling script |

## Implementation Plan

### Step A: 创建 `FA2VisiPrunerLlamaAttention` 类

在 `custom_modeling_llama.py` 中新增一个 attention 类，继承自 `VisiPrunerLlamaAttention`，重写 `forward()`：

```python
class FA2VisiPrunerLlamaAttention(VisiPrunerLlamaAttention):
    """
    FA2-accelerated VisiPruner attention.
    - Uses flash_attn_func() for dense attention computation
    - Computes Q_last @ K^T (cheap) for token importance when needed
    - Skips shallow pruning entirely (incompatible with FA2's fused kernel)
    """
    def forward(self, hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache,
                important_vis_tokens=None, exit_indicator=0, **kwargs):
        # 1. QKV projection (same as parent)
        # 2. RoPE (same)
        # 3. KV cache (same)
        # 4. Q_last @ K^T for pruning proxy (ONLY when prefill + middle/deep layer)
        # 5. FA2: attn_output = flash_attn_func(q, k, v, causal=True)
        # 6. Token importance from q_last @ k^T (if applicable)
        # 7. Output projection
        # 8. Return (attn_output, important_vis_tokens, exit_indicator) or attn_output
```

核心逻辑：
- **Line 669 (eager QK^T)**: 替换为 `flash_attn_func(q, k, v, causal=True, softmax_scale=1/sqrt(head_dim))`
- **Line 685 (softmax)**: 由 FA2 内部处理
- **Line 687-699 (shallow pruning)**: 删除
- **Line 703 (AV matmul)**: 由 FA2 内部处理
- **Line 715-719 (value_aware_token_selection)**: 保留，但输入改为轻量级计算的 `q_last_weights`
- **Pruning proxy 计算**: 仅当 `q_len > 1 and self.layer_idx > shallow_mid_layer` 时计算 `q_last @ K^T / sqrt(d)` 和 softmax

### Step B: 修改 `LlamaDecoderLayer` 支持动态 attention 选择

当前 `LlamaDecoderLayer.__init__` 硬编码了 `VisiPrunerLlamaAttention`（line 1042）。修改为根据 config 选择：

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        if config._attn_implementation == "flash_attention_2" and config.use_visipruner:
            self.self_attn = FA2VisiPrunerLlamaAttention(config=config, layer_idx=layer_idx)
        elif config.use_visipruner:
            self.self_attn = VisiPrunerLlamaAttention(config=config, layer_idx=layer_idx)
        elif config._attn_implementation == "flash_attention_2":
            self.self_attn = LlamaFlashAttention2(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
```

这需要 `LlavaConfig` 暴露 `use_visipruner` 属性。

### Step C: 编写 standalone inference + profiling script

基于 `model_vqa_loader.py` 的可运行 pattern，编写 `bench_fa2_visipruner.py`：

```python
# Configs:
#   dense-fa2           Dense + FA2 (fastest baseline)
#   visipruner-eager    Original VisiPruner + eager attention
#   visipruner-fa2      NEW: VisiPruner + FA2 attention + lightweight pruning proxy
```

支持 nsys/ncu profiling（与现有 bench 相同接口）。

### Step D: 更新 Research Plan

删除 e2-e7，重新设计实验：

| Experiment | 目的 | Configs |
|-----------|------|---------|
| **E1** (保留) | Baseline gap: FA2 vs Eager vs VisiPruner-Eager | dense-fa2, dense-eager, visipruner-full, visipruner-shallow-only |
| **E2** (新) | FA2-VisiPruner: FA2 attention + pruning 是否比 FA2-Dense 快 | dense-fa2, visipruner-fa2-full, visipruner-fa2-shallow |
| **E3** (新) | 质量对比: FA2-VisiPruner vs Eager-VisiPruner 输出质量是否一致 | 同上 + visipruner-eager |
| **E4** (新) | 可扩展性: batch size / sequence length scaling | 同上，变参数扫描 |

## Verification

1. **Smoke test**: 每个新 config 加载模型并生成 token（`--mode smoke`）
2. **nsys profiling**: 获取 GPU kernel 时间线，验证 FA2 kernel 占比
3. **输出一致性**: 用相同输入对比 FA2-VisiPruner 和 Eager-VisiPruner 的输出
4. **Expected result**: visipruner-fa2 T_total 应接近 dense-fa2（而非 dense-eager 的 1490ms）
