# VisiPrune Attn 阶段处理代码与矩阵展开示例

本文档从当前实现中提取 VisiPrune 对 attention 的分阶段处理逻辑，并用最后二维矩阵视角明确说明：

- 哪些 token / attention block 参与当前层计算。
- 哪些 attention block 在结果上被置零或重定向。
- 哪些位置执行 `sum` 聚合。
- 哪些 token 在裁剪后从后续层计算中真正跳过。

主要源码：

- `repo/llava/model/language_model/custom_modeling_llama.py`
- `repo/llava/model/builder.py`
- `autoresearch/experiments/e1_baseline/code/bench_e1_baseline.py`
- `autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py`

## 0. 图例和坐标

Attention 权重的坐标定义：

```python
attn_weights[:, :, query_position, key_position]
```

本文用以下 token 区间表示一条 multimodal 序列：

```text
P = prefix/text-before-image tokens = [0, 35)
V = visual tokens                 = [35, vis_end_index)
T = text-after-image tokens       = [vis_end_index, L)
```

对单图 7B 常见路径：

```text
vis_end_index = 35 + 576 * num_images
```

常用张量形状：

```text
hidden_states: [B, L, C]
Q:             [B, H, Lq, D]
K, V:          [B, H, Lkv, D]      # repeat_kv 后
A:             [B, H, Lq, Lkv]     # attention weights
O_heads:       [B, H, Lq, D]
O:             [B, Lq, C]
```

矩阵块图中：

```text
行 = query token
列 = key/value token
keep = 保持原 attention 权重
zero = 结果置零
sum  = 将多个视觉 key 的 attention mass 聚合到 key=35
drop = token 从后续层序列中删除，不再参与 Q/K/V/Attn/MLP
```

重要区分：

- `zero` 是 softmax 后修改 `attn_weights`，当前实现仍然已经计算过对应 `QK^T`。
- `drop` 是修改 `hidden_states` 序列长度，后续层真的不再计算被删除 token。

下文所有字符画默认固定一个 batch/head，只看张量最后二维：

```text
Q[b,h,:,:]      -> [query_token, head_dim]
K[b,h,:,:].T    -> [head_dim, key_token]
A[b,h,:,:]      -> [query_token, key_token]
V[b,h,:,:]      -> [key_token, head_dim]
hidden_states   -> [token, hidden_dim]
```

因此 `Q @ K^T` 的矩阵块按 token 分区就是：

```text
              K^T columns = key tokens
              P keys      V keys      T keys
Q rows P    [ P->P ]    [ P->V ]    [ P->T ]
Q rows V    [ V->P ]    [ V->V ]    [ V->T ]
Q rows T    [ T->P ]    [ T->V ]    [ T->T ]
```

## 1. Attention 类选择

模型加载时，`use_flash_attn` 决定 HuggingFace attention implementation：

```python
if use_flash_attn:
    kwargs['attn_implementation'] = 'flash_attention_2'
else:
    kwargs['attn_implementation'] = 'eager'
```

当 `use_visipruner=True` 时，builder 会把标志写入 config：

```python
cfg.use_visipruner = True
cfg._attn_implementation = kwargs.get('attn_implementation', 'eager')
kwargs['config'] = cfg
```

`LlamaDecoderLayer` 再按 config 选择 attention 类：

```python
if use_visipruner and attn_impl == "flash_attention_2":
    self.self_attn = FA2VisiPrunerLlamaAttention(config=config, layer_idx=layer_idx)
elif use_visipruner:
    self.self_attn = VisiPrunerLlamaAttention(config=config, layer_idx=layer_idx)
elif attn_impl == "flash_attention_2":
    self.self_attn = LlamaFlashAttention2(config=config, layer_idx=layer_idx)
else:
    self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
```

E1 配置中的 `visipruner-shallow-only` 明确不使用 FA2：

```python
"visipruner-shallow-only": {
    "use_flash_attn": False,
    "pruning_config": {
        "mode": ["shallow"],
        "shallow_mid_layer": 6,
    },
}
```

命名注意：

- E1 `bench_e1_baseline.py` 中，`visipruner-full` 配置是 `mode=["middle", "deep"]`。
- E2 `profile_visprune_single_request.py` 中，`visipruner-full` 配置是 `mode=["shallow", "middle", "deep"]`。
- 本文按源码能力拆解 shallow/middle/deep 三个阶段；具体实验是否启用某阶段，以对应脚本的 `pruning_config["mode"]` 为准。

## 2. Eager VisiPruner 的公共 Attn 主路径

原版 `VisiPrunerLlamaAttention.forward()` 先完整 materialize attention matrix：

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)

attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

# shallow 在这里修改 attn_weights

attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
attn_output = torch.matmul(attn_weights, value_states)
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
```

参与和跳过：

| 项 | 当前层是否参与计算 | 说明 |
|---|---|---|
| 当前序列内的 P/V/T tokens | 参与 | 都会生成 Q/K/V，并进入 `Q @ K^T` 和 `A @ V` |
| shallow 置零的 block | 仍参与前半段计算 | 先算完整 `attn_weights`，再在 softmax 后改权重 |
| middle/deep 已经 drop 的视觉 token | 不参与 | 如果之前已经裁剪，后续层 `hidden_states` 已经不含这些 token |

最后二维字符画，完整 eager attention 的两个矩阵乘法如下：

```text
1) Score = Q @ K^T

Q [Lq x D]                    K^T [D x Lkv]                  Score/A [Lq x Lkv]
┌───────────────┐             ┌──────────────────────┐       ┌──────────────────────┐
│ P query rows  │             │ P keys | V keys | T  │       │ P->P | P->V | P->T  │
├───────────────┤   @         │        |        |    │   =   ├──────┼──────┼───────┤
│ V query rows  │             │   all D feature rows │       │ V->P | V->V | V->T  │
├───────────────┤             │        |        |    │       ├──────┼──────┼───────┤
│ T query rows  │             │ P keys | V keys | T  │       │ T->P | T->V | T->T  │
└───────────────┘             └──────────────────────┘       └──────────────────────┘

这里 9 个 block 都参与计算。shallow 后续只改 A 的部分 block，不会阻止 Score 被算出。

2) O_heads = A @ V

A [Lq x Lkv]                  V [Lkv x D]                    O_heads [Lq x D]
┌──────────────────────┐      ┌──────────────┐               ┌───────────────┐
│ P->P | P->V | P->T  │      │ P value rows │               │ P output rows │
├──────┼──────┼───────┤  @   ├──────────────┤           =   ├───────────────┤
│ V->P | V->V | V->T  │      │ V value rows │               │ V output rows │
├──────┼──────┼───────┤      ├──────────────┤               ├───────────────┤
│ T->P | T->V | T->T  │      │ T value rows │               │ T output rows │
└──────────────────────┘      └──────────────┘               └───────────────┘
```

## 3. Shallow Pruning

触发条件：

```python
if (
    'shallow' in getattr(self, "pruning_mode", [])
    and hasattr(self, 'shallow_mid_layer')
    and self.layer_idx < self.shallow_mid_layer
    and q_len > 611
    and self.num_images > 0
):
    ...
```

也就是说 shallow 只发生在 prefill 长序列、浅层、包含图像的情况下。

边界条件注意：

```text
shallow:     layer_idx < shallow_mid_layer
middle/deep: layer_idx > shallow_mid_layer
```

如果 `shallow_mid_layer = 6`，则 shallow 覆盖 layer 0..5，middle/deep 从 layer 7 开始；`layer_idx == 6` 不触发这三种 pruning 分支。

### 3.1 7B layer 0

代码：

```python
sum_vision_attn_weights = attn_weights[:, :, self.vis_end_index:, 35:self.vis_end_index].sum(dim=-1)
attn_weights[:, :, 35:, 35:self.vis_end_index] = 0
attn_weights[:, :, self.vis_end_index:, 35] = sum_vision_attn_weights
```

分解为三个张量操作：

```text
S_T = A[:, :, T, V].sum(dim=-1)      # [B, H, |T|]
A[:, :, V+T, V] = 0                 # visual query 和 suffix text query 对 visual keys 置零
A[:, :, T, key=35] = S_T            # suffix text query 的视觉 attention mass 聚合到第一个视觉 key
```

矩阵块示意，行是 query，列是 key/value：

```text
7B layer 0, shallow mutation after softmax

                 key: P             key: V                         key: T
query: P         keep               keep                           keep
query: V         keep               zero                           keep
query: T         keep               sum -> key 35, others zero     keep
```

按最后二维 `A [Lq x Lkv]` 展开，修改前后是：

```text
Before shallow layer 0:

A rows \ cols      P keys             V keys                         T keys
┌─────────────┬─────────────────┬──────────────────────────────┬─────────────┐
│ P queries   │ keep            │ keep                         │ keep        │
├─────────────┼─────────────────┼──────────────────────────────┼─────────────┤
│ V queries   │ keep            │ computed                    │ keep        │
├─────────────┼─────────────────┼──────────────────────────────┼─────────────┤
│ T queries   │ keep            │ v35 v36 ... v_end-1         │ keep        │
└─────────────┴─────────────────┴──────────────────────────────┴─────────────┘

Operation:

S_T = sum over the V-key columns for each T-query row

T-query row before:  [ P cols ][ v35  v36  ...  v_end-1 ][ T cols ]
                           \____ sum over this block ____/

T-query row after:   [ P cols ][ S_T   0   ...     0     ][ T cols ]
                              key=35 gets the sum

V-query rows after:  [ P cols ][  0    0   ...     0     ][ T cols ]

After shallow layer 0:

A rows \ cols      P keys             V keys                         T keys
┌─────────────┬─────────────────┬──────────────────────────────┬─────────────┐
│ P queries   │ keep            │ keep                         │ keep        │
├─────────────┼─────────────────┼──────────────────────────────┼─────────────┤
│ V queries   │ keep            │ zero                         │ keep        │
├─────────────┼─────────────────┼──────────────────────────────┼─────────────┤
│ T queries   │ keep            │ sum at key 35, rest zero     │ keep        │
└─────────────┴─────────────────┴──────────────────────────────┴─────────────┘
```

`sum` 的含义：

```text
对每个 batch/head/文本 query:

原始视觉注意力总量
  A[..., query=t, key=35] + A[..., query=t, key=36] + ... + A[..., query=t, key=vis_end_index-1]

被写回:
  A[..., query=t, key=35]

其他视觉 key:
  A[..., query=t, key=36:vis_end_index] = 0
```

计算结论：

| Block | 结果如何处理 | 当前层是否省计算 |
|---|---|---|
| P -> P/V/T | keep | 不省，正常参与 |
| V -> P/T | keep | 不省，正常参与 |
| V -> V | zero | 不省，先算后置零 |
| T -> V | sum 到 key 35 | 不省，先算后聚合 |
| T -> P/T | keep | 不省，正常参与 |

注意：`query: V -> key: V` 被置零后没有重新归一化；`query: T -> key: V` 的 mass 被聚合到 key 35，因此 suffix text 仍可通过第一个视觉 token 读取聚合后的视觉信息。

### 3.2 7B layer 1 到 shallow_mid_layer-1

代码：

```python
attn_weights[:, :, self.vis_end_index:, 35:self.vis_end_index] = 0
```

矩阵块示意：

```text
7B layer 1..shallow_mid_layer-1, shallow mutation after softmax

                 key: P             key: V             key: T
query: P         keep               keep               keep
query: V         keep               keep               keep
query: T         keep               zero               keep
```

最后二维 `A [Lq x Lkv]` 字符画：

```text
Before:

A rows \ cols      P keys             V keys             T keys
┌─────────────┬─────────────────┬─────────────────┬─────────────────┐
│ P queries   │ keep            │ keep            │ keep            │
├─────────────┼─────────────────┼─────────────────┼─────────────────┤
│ V queries   │ keep            │ keep            │ keep            │
├─────────────┼─────────────────┼─────────────────┼─────────────────┤
│ T queries   │ keep            │ computed        │ keep            │
└─────────────┴─────────────────┴─────────────────┴─────────────────┘

After:

A rows \ cols      P keys             V keys             T keys
┌─────────────┬─────────────────┬─────────────────┬─────────────────┐
│ P queries   │ keep            │ keep            │ keep            │
├─────────────┼─────────────────┼─────────────────┼─────────────────┤
│ V queries   │ keep            │ keep            │ keep            │
├─────────────┼─────────────────┼─────────────────┼─────────────────┤
│ T queries   │ keep            │ zero            │ keep            │
└─────────────┴─────────────────┴─────────────────┴─────────────────┘

注意：T->V 的 Score 和 softmax 已经算过；这里只是把 A 的这块写 0。
```

计算结论：

| Block | 结果如何处理 | 当前层是否省计算 |
|---|---|---|
| T -> V | zero | 不省，先算完整 attention，再置零 |
| V -> V | keep | 视觉 token 作为 query 时仍可 attend 到视觉 token |
| P/V/T 其他 block | keep | 正常参与 |

因此，`shallow-only` 不是“只计算文本对文本和视觉 token 的注意力”。它仍然完整计算当前层 dense attention，只是把部分 text-to-visual block 在结果上置零或重定向。

### 3.3 13B layer 0 分支

代码中还有 13B 专用逻辑：

```python
attn_weights[:, :, 35:self.vis_end_index, 35:self.vis_end_index] = 0
attn_weights[:, :, self.vis_end_index:, 35:self.vis_half_index] = 0
```

矩阵块示意：

```text
13B layer 0, shallow mutation after softmax

                 key: P             key: V_first_half     key: V_second_half    key: T
query: P         keep               keep                  keep                 keep
query: V         keep               zero for all V keys   zero for all V keys  keep
query: T         keep               zero                  keep                 keep
```

最后二维 `A [Lq x Lkv]` 字符画：

```text
A rows \ cols      P keys       V first half       V second half      T keys
┌─────────────┬─────────────┬─────────────────┬─────────────────┬─────────────┐
│ P queries   │ keep        │ keep            │ keep            │ keep        │
├─────────────┼─────────────┼─────────────────┼─────────────────┼─────────────┤
│ V queries   │ keep        │ zero            │ zero            │ keep        │
├─────────────┼─────────────┼─────────────────┼─────────────────┼─────────────┤
│ T queries   │ keep        │ zero            │ keep            │ keep        │
└─────────────┴─────────────┴─────────────────┴─────────────────┴─────────────┘
```

这里没有 `sum` 聚合。

### 3.4 shallow-only 配置的实际含义

`visipruner-shallow-only` 只设置：

```python
mode = ["shallow"]
```

因此：

- 只执行本节的 post-softmax `attn_weights` 修改。
- 不调用 middle 的重要视觉 token 选择。
- 不调用 deep 的 exit check。
- 不裁剪 `hidden_states` 长度。
- 不使用 FA2。

## 4. Middle Pruning

Middle pruning 的作用是根据最后一个 query token 的输出贡献，选择重要视觉 token，并在后续层物理裁剪掉不重要视觉 token。

### 4.1 在 attention 层内选择 important visual tokens

触发代码：

```python
if self.num_images > 0 and q_len > 1 and self.layer_idx > getattr(self, "shallow_mid_layer", 80):
    if "middle" in self.pruning_mode and important_vis_tokens is None and q_len == position_ids[0, -1] + 1:
        important_vis_tokens = self.value_aware_token_selection(value_states, attn_output, attn_weights)
```

`value_aware_token_selection()` 核心代码：

```python
attn_output_last = attn_output[:, -1, :]
attn_weights_last = attn_weights[:, :, -1, :]

contributions = attn_weights_last[:, :, :, None] * value_states
contributions = contributions.permute(0, 2, 1, 3).contiguous()
contributions = contributions.view(batch_size, seq_len, -1)

vision_contributions = contributions[:, 35:self.vis_end_index, :]
masked_attn_output_last = attn_output_last[:, None, :] - vision_contributions
cos_sim = F.cosine_similarity(masked_attn_output_last, attn_output_last[:, None, :], dim=-1).squeeze(0)

if torch.any(cos_sim < self.layer_threshold).item():
    l_2 = torch.norm(masked_attn_output_last - attn_output_last[:, None, :], p=2, dim=-1).squeeze(0)
    important_vis_tokens = torch.where(l_2 > self.tokens_threshold)[0] + 35
```

逐条张量指令的矩阵视角展开如下。为了便于展示，固定 `B=1`，假设 `H=2`、每个 head 的 `D=2`，所以 `C=H*D=4`。token 分区用小例子表示：

```text
P = [p0, p1]
V = [v0, v1, v2]          # 实际代码中是 [35, vis_end_index)
T = [t0, t1]
last = t1
L = 7
```

1. `attn_output_last = attn_output[:, -1, :]`

从 `attn_output [L x C]` 取最后一个 query token 的输出行：

```text
attn_output [L x C]
┌────┬──────────────────────────────┐
│ p0 │ O_p0                         │
│ p1 │ O_p1                         │
│ v0 │ O_v0                         │
│ v1 │ O_v1                         │
│ v2 │ O_v2                         │
│ t0 │ O_t0                         │
│ t1 │ O_t1 = [o0, o1, o2, o3]      │  <-- selected
└────┴──────────────────────────────┘

attn_output_last = O_t1 = [o0, o1, o2, o3]     # [C]
```

2. `attn_weights_last = attn_weights[:, :, -1, :]`

从每个 head 的 `A [L x L]` 中取最后一个 query row。只取最后一行，其他 query row 已经算出但不参与 selection：

```text
A [L x L] full matrix has already been computed:

              key columns
              P keys         V keys                 T keys
query P     [ ... ]        [ ... ]                [ ... ]
query V     [ ... ]        [ ... ]                [ ... ]
query T     [ ... ]        [ ... ]                [ ... ]
last query  [ a_P... ]     [ a_V35 ... a_Vend ]   [ a_T... ]   <-- only this row is used

head 0:
A_last^0 [1 x L] = [a0_p0, a0_p1 | a0_v0, a0_v1, a0_v2 | a0_t0, a0_t1]

head 1:
A_last^1 [1 x L] = [a1_p0, a1_p1 | a1_v0, a1_v1, a1_v2 | a1_t0, a1_t1]

attn_weights_last shape = [H, L]
```

3. `contributions = attn_weights_last[:, :, :, None] * value_states`

这是按 token 行缩放 `V`，不是矩阵乘法。固定 head 0 展开：

```text
value_states^0 [L x D]
┌────┬────────────────────┐
│ p0 │ [vp0_0, vp0_1]     │
│ p1 │ [vp1_0, vp1_1]     │
│ v0 │ [vv0_0, vv0_1]     │
│ v1 │ [vv1_0, vv1_1]     │
│ v2 │ [vv2_0, vv2_1]     │
│ t0 │ [vt0_0, vt0_1]     │
│ t1 │ [vt1_0, vt1_1]     │
└────┴────────────────────┘

A_last^0 [L]
[a0_p0, a0_p1, a0_v0, a0_v1, a0_v2, a0_t0, a0_t1]

contributions^0 [L x D]
┌────┬──────────────────────────────────────┐
│ p0 │ a0_p0 * [vp0_0, vp0_1]               │
│ p1 │ a0_p1 * [vp1_0, vp1_1]               │
│ v0 │ a0_v0 * [vv0_0, vv0_1]               │
│ v1 │ a0_v1 * [vv1_0, vv1_1]               │
│ v2 │ a0_v2 * [vv2_0, vv2_1]               │
│ t0 │ a0_t0 * [vt0_0, vt0_1]               │
│ t1 │ a0_t1 * [vt1_0, vt1_1]               │
└────┴──────────────────────────────────────┘
```

两个 head 都执行同样的行缩放，得到 `contributions [H x L x D]`。

4. `contributions = contributions.permute(0, 2, 1, 3).contiguous()`

去掉 batch 维后，形状从 head-major 变成 token-major：

```text
Before permute: [H x L x D]

head 0 matrix: rows p0,p1,v0,v1,v2,t0,t1, cols d0,d1
head 1 matrix: rows p0,p1,v0,v1,v2,t0,t1, cols d0,d1

After permute: [L x H x D]

token p0 row owns:
    head0 contribution for p0: [c0_p0_d0, c0_p0_d1]
    head1 contribution for p0: [c1_p0_d0, c1_p0_d1]

token v1 row owns:
    head0 contribution for v1: [c0_v1_d0, c0_v1_d1]
    head1 contribution for v1: [c1_v1_d0, c1_v1_d1]
```

5. `contributions = contributions.view(batch_size, seq_len, -1)`

把每个 token 的所有 head 维度拼成 hidden/channel 维：

```text
After view: contributions [L x C], where C = H * D = 4

┌────┬──────────────────────────────────────────────────────┐
│ p0 │ [c0_p0_d0, c0_p0_d1, c1_p0_d0, c1_p0_d1]             │
│ p1 │ [c0_p1_d0, c0_p1_d1, c1_p1_d0, c1_p1_d1]             │
│ v0 │ [c0_v0_d0, c0_v0_d1, c1_v0_d0, c1_v0_d1]             │
│ v1 │ [c0_v1_d0, c0_v1_d1, c1_v1_d0, c1_v1_d1]             │
│ v2 │ [c0_v2_d0, c0_v2_d1, c1_v2_d0, c1_v2_d1]             │
│ t0 │ [c0_t0_d0, c0_t0_d1, c1_t0_d0, c1_t0_d1]             │
│ t1 │ [c0_t1_d0, c0_t1_d1, c1_t1_d0, c1_t1_d1]             │
└────┴──────────────────────────────────────────────────────┘
```

6. `vision_contributions = contributions[:, 35:self.vis_end_index, :]`

只切出视觉 token 行作为候选：

```text
contributions [L x C]
┌────┬──────────────┐
│ p0 │ C_p0         │  ignored by middle selection
│ p1 │ C_p1         │  ignored by middle selection
│ v0 │ C_v0         │  candidate
│ v1 │ C_v1         │  candidate
│ v2 │ C_v2         │  candidate
│ t0 │ C_t0         │  ignored by middle selection
│ t1 │ C_t1         │  ignored by middle selection
└────┴──────────────┘

vision_contributions [|V| x C]
┌────┬──────────────┐
│ v0 │ C_v0         │
│ v1 │ C_v1         │
│ v2 │ C_v2         │
└────┴──────────────┘
```

7. `masked_attn_output_last = attn_output_last[:, None, :] - vision_contributions`

把同一个 `O_t1` 广播到每个视觉 token 行，然后逐行减掉该视觉 token 的 contribution：

```text
attn_output_last = O_t1 = [o0, o1, o2, o3]

masked_attn_output_last [|V| x C]
┌────┬──────────────────────────────────────┐
│ v0 │ O_t1 - C_v0                          │
│ v1 │ O_t1 - C_v1                          │
│ v2 │ O_t1 - C_v2                          │
└────┴──────────────────────────────────────┘
```

含义：假设“删除某个视觉 token 的贡献”，最后 query 的输出会变成什么。

8. `cos_sim = F.cosine_similarity(masked_attn_output_last, attn_output_last[:, None, :], dim=-1).squeeze(0)`

每一行都和原始 `O_t1` 做 cosine：

```text
cos_sim [|V|]
┌────┬──────────────────────────────────────────────┐
│ v0 │ cos(O_t1 - C_v0, O_t1)                       │
│ v1 │ cos(O_t1 - C_v1, O_t1)                       │
│ v2 │ cos(O_t1 - C_v2, O_t1)                       │
└────┴──────────────────────────────────────────────┘
```

9. `if torch.any(cos_sim < self.layer_threshold).item():`

只要任意视觉 token 的“移除后输出”与原输出相似度低于层阈值，就认为这一层需要选择重要视觉 token：

```text
cos_sim = [0.998, 0.970, 0.996]
layer_threshold = 0.995

cos_sim < threshold = [False, True, False]
any(...) = True
```

10. `l_2 = torch.norm(masked_attn_output_last - attn_output_last[:, None, :], p=2, dim=-1)`

由于 `masked_attn_output_last = O_t1 - C_vi`，所以：

```text
masked_attn_output_last - O_t1 = -C_vi
l_2[i] = ||C_vi||_2
```

矩阵视角：

```text
l_2 [|V|]
┌────┬────────────────────┐
│ v0 │ ||C_v0||_2          │
│ v1 │ ||C_v1||_2          │
│ v2 │ ||C_v2||_2          │
└────┴────────────────────┘
```

11. `important_vis_tokens = torch.where(l_2 > self.tokens_threshold)[0] + 35`

把局部视觉行号转成原序列里的绝对 token index：

```text
l_2 = [0.05, 0.42, 0.18]
tokens_threshold = 0.20

l_2 > threshold = [False, True, False]
torch.where(...)[0] = [1]             # visual-local index
+ 35 => [36]                          # absolute token index, i.e. v1
```

输出：

```text
important_vis_tokens = selected absolute visual-token indices
```

参与和跳过：

| 项 | 当前 selection 层 | 后续层 |
|---|---|---|
| P/V/T tokens 的 Q/K/V 和 full attention | 参与 | 取决于是否被裁剪 |
| `A[:, :, -1, :]` 最后一行 | 参与选择 | 不适用 |
| 非最后 query 行的 attention | 已经算出，但不参与 token selection | 不适用 |
| P/T token contributions | 当前实现中先算 `contributions`，但候选选择只切 `V` | 保留 |
| 不重要视觉 token | 当前层已参与 | 从下一次 compaction 后开始 drop |

### 4.2 在 LlamaModel 主循环中裁剪序列

attention 层返回 `important_vision_tokens` 后，主循环在后续层进入前裁剪：

```python
if hidden_states.shape[1] > 1 and "middle" in getattr(self, "pruning_mode", []) and layer_idx > getattr(self, "shallow_mid_layer", 80) and hidden_states.shape[1] == inputs_embeds.shape[1]:
    if important_vision_tokens != None:
        important_vision_hidden_states = hidden_states[:, important_vision_tokens, :]
        hidden_states = torch.cat(
            (
                hidden_states[:, :35, :],
                important_vision_hidden_states,
                hidden_states[:, self.last_image_token_index:, :],
            ),
            dim=1,
        )
        position_ids = torch.cat(
            (
                position_ids[:, :35],
                position_ids[:, important_vision_tokens],
                position_ids[:, self.last_image_token_index:],
            ),
            dim=1,
        )
        pruned_vis_len = hidden_states.shape[1]
        attention_mask = complete_attn_mask[:, :, :pruned_vis_len, :pruned_vis_len]
        self.last_image_token_index = 35 + important_vision_tokens.shape[0]
```

裁剪示意：

```text
Before middle compaction:

H = [ P ][ V_drop + V_keep mixed in original visual positions ][ T ]

important_vision_tokens = selected absolute indices inside V

After middle compaction:

H' = [ P ][ V_keep ][ T ]
          ^
          only selected visual token hidden states are copied

position_ids' = [ P_pos ][ selected V_pos ][ T_pos ]
mask'         = complete_attn_mask[:, :, :L', :L']
```

最后二维 `hidden_states [L x C]` 的行选择字符画：

```text
Before:

hidden_states [L x C]
┌──────────────────────────────┐
│ P rows                       │  keep
├──────────────────────────────┤
│ V0                           │  drop if not selected
│ V1                           │  keep if in important_vision_tokens
│ V2                           │  drop if not selected
│ ...                          │
│ V575                         │  keep/drop by selection
├──────────────────────────────┤
│ T rows                       │  keep
└──────────────────────────────┘

Gather/cat:

hidden_states' = cat(P rows, selected V rows, T rows)

After:

hidden_states' [L' x C]
┌──────────────────────────────┐
│ P rows                       │  participate in later Q/K/V/Attn/MLP
├──────────────────────────────┤
│ V_keep rows                  │  participate in later Q/K/V/Attn/MLP
├──────────────────────────────┤
│ T rows                       │  participate in later Q/K/V/Attn/MLP
└──────────────────────────────┘

V_drop rows are absent from the matrix, so later layers cannot compute them.
```

计算结论：

| Token group | selection 层 | compaction 后的后续层 |
|---|---|---|
| P | 参与 | 参与 |
| V_keep | 参与 | 参与，位置被压到 `[35, 35 + k)` |
| V_drop | 参与 | drop，不再生成 Q/K/V，不再进入 attention 和 MLP |
| T | 参与 | 参与，接在 V_keep 后 |

这是 VisiPrune 中真正减少后续计算量的阶段。

## 5. Deep Pruning

Deep pruning 在 middle 已经选出 `important_vis_tokens` 后继续检查：剩下的视觉 token 是否还必要。如果连续通过 exit check，主循环会把剩余视觉 token 全部移除。

### 5.1 在 attention 层内做 exit check

触发代码：

```python
elif (
    "deep" in self.pruning_mode
    and important_vis_tokens is not None
    and q_len == position_ids[0, -1] + 1 - 576 * self.num_images + important_vis_tokens.shape[0]
):
    exit_indicator += self.value_aware_token_selection(
        value_states, attn_output, attn_weights, important_vis_tokens
    )
```

`important_vis_tokens is not None` 时，`value_aware_token_selection()` 走 else 分支：

```python
offset_tokens_index = torch.arange(35, 35 + important_vis_tokens.shape[0], device=important_vis_tokens.device)
vision_contributions = contributions[:, offset_tokens_index, :]
masked_attn_output_last = attn_output_last[:, None, :] - vision_contributions
cos_sim = F.cosine_similarity(masked_attn_output_last, attn_output_last[:, None, :], dim=-1).squeeze(0)

if torch.any(cos_sim < 0.999).item():
    return False
else:
    return True
```

Deep check 复用 middle selection 中的 `attn_output_last`、`attn_weights_last`、`contributions` 计算；差异是候选行不再是原始视觉区间，而是 compact 后连续排列的 `V_keep` 行，并且返回值是 bool。

最后二维矩阵视角，deep check 在 compact sequence 上重复使用最后 query row：

```text
Current compact sequence:

hidden_states' [L' x C] = [ P rows ][ V_keep rows ][ T rows ]

A' [L' x L'] full matrix has already been computed for this compact sequence:

              key columns
              P keys          V_keep keys             T keys
query P     [ ... ]         [ ... ]                 [ ... ]
query Vkeep [ ... ]         [ ... ]                 [ ... ]
query T     [ ... ]         [ ... ]                 [ ... ]
last query  [ a_P... ]      [ a_Vkeep... ]          [ a_T... ]  <-- used

contributions' [L' x C]
┌──────────────────────────────┐
│ P contribution rows          │  computed, not checked
├──────────────────────────────┤
│ V_keep contribution rows     │  checked one by one
├──────────────────────────────┤
│ T contribution rows          │  computed, not checked
└──────────────────────────────┘

For every kept visual row j:
O_last_without_j = O_last - contributions'[V_keep_j, :]
if all removals keep cosine >= 0.999:
    return True
else:
    return False
```

逐条指令差异：

1. `offset_tokens_index = torch.arange(35, 35 + important_vis_tokens.shape[0], device=...)`

middle compaction 后，保留下来的视觉 token 被压到连续区间：

```text
compact hidden_states' [L' x C]
┌──────────────┐
│ P rows       │
├──────────────┤
│ V_keep_0     │  row 35
│ V_keep_1     │  row 36
│ ...          │
│ V_keep_k-1   │  row 35+k-1
├──────────────┤
│ T rows       │
└──────────────┘

offset_tokens_index = [35, 36, ..., 35+k-1]
```

2. `vision_contributions = contributions[:, offset_tokens_index, :]`

从 `contributions' [L' x C]` 里只取当前还保留的视觉行：

```text
contributions' [L' x C]
┌──────────────┬──────────────┐
│ P rows       │ ignored      │
│ V_keep_0     │ checked      │
│ V_keep_1     │ checked      │
│ ...          │ checked      │
│ T rows       │ ignored      │
└──────────────┴──────────────┘

vision_contributions [k x C] = rows V_keep_0..V_keep_k-1
```

3. `masked_attn_output_last = attn_output_last[:, None, :] - vision_contributions`

```text
masked_attn_output_last [k x C]
┌──────────┬────────────────────────┐
│ V_keep_0 │ O_last - C_Vkeep_0     │
│ V_keep_1 │ O_last - C_Vkeep_1     │
│ ...      │ ...                    │
└──────────┴────────────────────────┘
```

4. `cos_sim = cosine_similarity(...)`

```text
cos_sim [k]
┌──────────┬────────────────────────────────────┐
│ V_keep_0 │ cos(O_last - C_Vkeep_0, O_last)    │
│ V_keep_1 │ cos(O_last - C_Vkeep_1, O_last)    │
│ ...      │ ...                                │
└──────────┴────────────────────────────────────┘
```

5. 返回逻辑：

```text
if any cos_sim < 0.999:
    return False     # 仍然有视觉 token 对最后输出有明显影响，不能 deep exit
else:
    return True      # 所有 V_keep 移除后都几乎不影响 O_last
```

参与和跳过：

| 项 | deep check 当前层 | 说明 |
|---|---|---|
| P/V_keep/T | 参与 | 当前 compact sequence 内的 token 都进入 attention |
| 已经 middle drop 的 V_drop | 不参与 | 已从 `hidden_states` 删除 |
| `A_last` | 参与 | 只用最后 query 的 attention row |
| V_keep contributions | 参与 exit check | 逐个测试移除该视觉 token 对 `O_last` 的影响 |
| P/T contributions | 当前实现中先算 `contributions`，但 exit check 只切 V_keep | 不作为候选 |

### 5.2 在主循环中移除剩余视觉 token

主循环逻辑：

```python
elif hidden_states.shape[1] > 1 and "deep" in getattr(self, "pruning_mode", []) and layer_idx > getattr(self, "shallow_mid_layer", 80) and important_vision_tokens is not None and hidden_states.shape[1] == inputs_embeds.shape[1] - 576 * self.num_images + important_vision_tokens.shape[0]:
    if exit_indicator == 2:
        hidden_states = torch.cat(
            (
                hidden_states[:, :35, :],
                hidden_states[:, self.last_image_token_index:, :],
            ),
            dim=1,
        )
        position_ids = torch.cat(
            (
                position_ids[:, :35],
                position_ids[:, self.last_image_token_index:],
            ),
            dim=1,
        )
        pruned_vis_len = hidden_states.shape[1]
        attention_mask = complete_attn_mask[:, :, :pruned_vis_len, :pruned_vis_len]
```

裁剪示意：

```text
Before deep compaction:

H' = [ P ][ V_keep ][ T ]

When exit_indicator == 2:

H'' = [ P ][ T ]
          ^
          all remaining visual tokens are removed
```

最后二维 `hidden_states [L' x C]` 的 deep compaction：

```text
Before deep compaction:

hidden_states' [L' x C]
┌──────────────────────────────┐
│ P rows                       │ keep
├──────────────────────────────┤
│ V_keep rows                  │ drop when exit_indicator == 2
├──────────────────────────────┤
│ T rows                       │ keep
└──────────────────────────────┘

After:

hidden_states'' [L'' x C]
┌──────────────────────────────┐
│ P rows                       │ participate in later layers
├──────────────────────────────┤
│ T rows                       │ participate in later layers
└──────────────────────────────┘

All visual rows are absent after this point.
```

计算结论：

| Token group | deep compaction 前 | deep compaction 后 |
|---|---|---|
| P | 参与 | 参与 |
| V_keep | 参与 | drop，不再参与后续层 |
| T | 参与 | 参与 |
| V_drop | 已不参与 | 仍不参与 |

## 6. Decode 阶段的 mask 调整

当生成阶段 `hidden_states.shape[1] == 1`，如果 middle/deep 已经改变 KV cache 长度，主循环会裁剪 attention mask 的 KV 维度：

```python
if hidden_states.shape[1] == 1 and ("middle" in getattr(self, "pruning_mode", []) or "deep" in getattr(self, "pruning_mode", [])) and layer_idx > getattr(self, "shallow_mid_layer", 80):
    attention_mask = complete_attn_mask
    pruned_kv_seq_len = past_key_values.get_usable_length(seq_length, layer_idx=layer_idx) + seq_length
    pruned_attention_mask = attention_mask[:, :, :, -pruned_kv_seq_len:]
    attention_mask = pruned_attention_mask
```

示意：

```text
Decode query length = 1

Q_decode: [B,H,1,D]
K_cache:  [B,H,L_pruned_cache,D]
mask:     [B,1,1,L_pruned_cache]

Only the current layer's usable pruned KV cache length is exposed to attention.
```

最后二维 attention 矩阵视角：

```text
Q_decode [1 x D]            K_cache^T [D x L_pruned_cache]          A_decode [1 x L_pruned_cache]
┌──────────────┐            ┌──────────────────────────────┐        ┌──────────────────────────────┐
│ current token│     @      │ P cache | V_keep? | T cache │   =    │ P attn | V_keep? | T attn   │
└──────────────┘            └──────────────────────────────┘        └──────────────────────────────┘

If middle dropped V_drop:
    V_drop columns are absent from K_cache^T and A_decode.

If deep removed all V_keep:
    visual columns are absent entirely:

K_cache^T columns:  [ P cache | T cache ]
A_decode columns:  [ P attn  | T attn  ]
```

参与和跳过：

| 项 | Decode 当前 token |
|---|---|
| 当前新 token query | 参与 |
| pruned KV cache 中保留的 P/V_keep/T | 参与 |
| prefill 中 drop 的视觉 token KV | 不在 pruned KV cache 中，不参与 |

## 7. FA2 VisiPruner 适配路径

`FA2VisiPrunerLlamaAttention` 是为了兼容 FlashAttention2 的改造路径。关键区别：

- FA2 不 materialize 完整 `A = softmax(QK^T)`。
- shallow pruning 需要 post-softmax 改 `attn_weights`，因此在 FA2 路径中跳过。
- middle/deep 只需要最后 query 的 attention weights，所以单独计算 `Q_last @ K^T` 作为 pruning proxy。

源码注释已经明确：

```python
Uses flash_attn_func() for dense attention computation.
Computes Q_last @ K^T as a lightweight pruning proxy.
Shallow pruning is SKIPPED.
Middle/deep pruning is PRESERVED.
```

FA2 pruning proxy 代码：

```python
need_pruning_proxy = (
    self.num_images > 0
    and q_len > 1
    and self.layer_idx > shallow_mid
    and ("middle" in pruning_mode or "deep" in pruning_mode)
)

if need_pruning_proxy:
    key_states_for_pruning = repeat_kv(key_states, self.num_key_value_groups)
    q_last = query_states[:, :, -1:, :]
    pruning_weights = torch.matmul(
        q_last, key_states_for_pruning.transpose(2, 3)
    ) / math.sqrt(self.head_dim)
    if attention_mask is not None and attention_mask.dim() == 4:
        pruning_weights = pruning_weights + attention_mask[:, :, -1:, :kv_seq_len]
    pruning_weights = nn.functional.softmax(pruning_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
```

FA2 attention 代码：

```python
attn_output = flash_attn_func(
    query_states_fa2,
    key_states_fa2,
    value_states_fa2,
    dropout_p=self.attention_dropout if self.training else 0.0,
    softmax_scale=1.0 / math.sqrt(self.head_dim),
    causal=is_causal,
    window_size=(-1, -1),
)
```

最后二维矩阵视角：

```text
FA2 dense attention:

Conceptually computes O = softmax(Q @ K^T) @ V,
but the full A [L x L] matrix below is NOT materialized:

              key columns
              P keys      V keys      T keys
query P     [ fused ]   [ fused ]   [ fused ]
query V     [ fused ]   [ fused ]   [ fused ]
query T     [ fused ]   [ fused ]   [ fused ]

"fused" means the block participates inside flash_attn_func, but there is no
post-softmax A block for Python code to edit. Therefore shallow cannot write:
    A[T, V] = 0
    A[V, V] = 0
    A[T, key=35] = sum(...)

FA2 pruning proxy for middle/deep:

Q_last [1 x D]          K^T [D x L]                         pruning_weights [1 x L]
┌─────────────┐         ┌──────────────────────────────┐     ┌──────────────────────────────┐
│ last query  │   @     │ P keys | V keys | T keys    │  =  │ P weights | V weights | T    │
└─────────────┘         └──────────────────────────────┘     └──────────────────────────────┘

Only this 1 x L row is materialized for value_aware_token_selection.
```

传入 `value_aware_token_selection()` 时，`pruning_weights [B,H,1,L]` 直接扮演 eager 路径里的 `attn_weights[:, :, -1:, :]`：

```text
eager path:  full A [L x L] exists, then use A[last, :]
FA2 path:    full A [L x L] absent, only compute A_proxy[last, :]

A_proxy[last, :] = softmax(Q_last @ K^T + last_query_mask)
```

FA2 路径参与和跳过：

| 项 | FA2 当前层 |
|---|---|
| Dense attention output | 参与，通过 `flash_attn_func(Q,K,V)` 计算 |
| 完整 attention matrix `A [B,H,L,L]` | 跳过，不 materialize |
| shallow post-softmax mask | 跳过，FA2 路径不做 |
| middle/deep selection | 参与，通过 `pruning_weights [B,H,1,L]` 近似原版最后 query row |
| 后续层 token drop | 保留，与 eager middle/deep compaction 相同 |

## 8. 阶段总表

| 阶段 | 触发位置 | 当前层 Attn 是否 dense 计算 | `attn_weights` 是否 materialize | 是否 post-softmax 改 A | 是否 sum | 是否让后续层跳过 token |
|---|---|---:|---:|---:|---:|---:|
| Dense eager | `LlamaAttention`/无 pruning | 是 | 是 | 否 | 否 | 否 |
| Shallow 7B layer 0 | `VisiPrunerLlamaAttention`, `layer_idx==0` | 是 | 是 | 是 | 是，`A[:, :, T, V].sum(dim=-1)` | 否 |
| Shallow 7B layer 1..5 | `0 < layer_idx < shallow_mid_layer` | 是 | 是 | 是 | 否 | 否 |
| Middle selection | `layer_idx > shallow_mid_layer`, prefill | 是 | 是 | 否 | 否 | 下一层开始 drop V_drop |
| Middle compaction | `LlamaModel.forward` layer loop | 不适用 | 不适用 | 不适用 | 不适用 | 是，保留 P/V_keep/T |
| Deep check | middle 后 compact sequence | 是，对 P/V_keep/T | 是 | 否 | 否 | 暂不 drop，先累计 exit_indicator |
| Deep compaction | `exit_indicator == 2` | 不适用 | 不适用 | 不适用 | 不适用 | 是，移除全部 V_keep |
| FA2 VisiPruner | `use_visipruner + flash_attention_2` | 是，通过 FA2 | 否，只有 proxy row | shallow 跳过 | proxy 不 sum | middle/deep 后续 drop 保留 |

## 9. 对常见问题的直接结论

### shallow-only 使用 FA2 吗？

不使用。E1 `visipruner-shallow-only` 设置 `use_flash_attn=False`，走 eager attention。

### shallow-only 是否跳过视觉 token 的计算？

不跳过。它先完整计算当前层 dense attention，再修改 softmax 后的 `attn_weights`。它不会缩短序列，因此不会减少后续层 Q/K/V、attention 或 MLP 的 token 数。

### shallow-only 是否只保留文本对文本和视觉 token 的注意力？

不准确。7B layer 0 会对 `T -> V` 做 sum 聚合到 key 35，并把 `V -> V` 置零；layer 1..5 只把 `T -> V` 置零，`V -> V` 仍保留。

### 真正跳过计算发生在哪里？

发生在 middle/deep compaction 之后：

```text
middle: [P][V][T] -> [P][V_keep][T]
deep:   [P][V_keep][T] -> [P][T]
```

只有从 `hidden_states` 中删除的视觉 token，才会在后续层真正不参与 Q/K/V、attention 和 MLP 计算。
