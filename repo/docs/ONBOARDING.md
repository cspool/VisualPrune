# VisiPruner 项目入门指南

> 基于知识图谱 + 源码深度分析，聚焦**如何训练**与**如何加速推理**。

---

## 1. 项目概览

| 项 | 值 |
|---|---|
| 论文 | *VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs* |
| 会议 | EMNLP 2025 Main |
| 基础框架 | LLaVA (基于 Vicuna/LLaMA + CLIP) |
| 核心贡献 | **无需训练的视觉 token 剪枝框架**，最多减少 99% 视觉注意力计算、53.9% FLOPs |
| 技术栈 | PyTorch, HuggingFace Transformers, DeepSpeed, FastAPI, Gradio |
| 代码规模 | ~50 Python 源文件 |

### 核心发现（论文三阶段理论）

1. **Shallow layers（浅层 0–5）**：识别任务意图，视觉 token 作为被动 "attention sink"
2. **Middle layers（中层 6–~20）**：跨模态融合在此**突然发生**，仅少数关键视觉 token 驱动
3. **Deep layers（深层）**：抛弃视觉 token，仅做语言精炼

VisiPruner 基于此发现，**对不同层采用不同剪枝策略**。

---

## 2. 架构分层

知识图谱识别了 7 个架构层：

```
┌─────────────────────────────────────────────┐
│  服务层 (llava/serve/)                       │
│  FastAPI Controller/Worker, Gradio, CLI     │
├─────────────────────────────────────────────┤
│  评估层 (llava/eval/)                        │
│  VQA/MMBench/GQA/TextVQA/MME/POPE/SEED...  │
├─────────────────────────────────────────────┤
│  训练层 (llava/train/)                       │
│  train.py, LLaVATrainer, Flash-Attn patch   │
├─────────────────────────────────────────────┤
│  模型层 (llava/model/)  ← 核心               │
│  LLaVA架构 + VisiPruner剪枝 + CLIP/SigLIP   │
│  视觉编码器 + 多模态投影器 + LLaMA骨干       │
├─────────────────────────────────────────────┤
│  核心工具层 (llava/)                         │
│  conversation, mm_utils, constants          │
├─────────────────────────────────────────────┤
│  数据脚本层 (scripts/)                       │
│  数据集转换, 权重提取/合并                    │
├─────────────────────────────────────────────┤
│  演示与分析层 (visualization/, playground/)   │
│  Jupyter Notebook 可视化, 注意力/L1范数分析  │
└─────────────────────────────────────────────┘
```

---

## 3. 如何训练

### 3.1 两阶段训练流程

LLaVA/VisiPruner 沿用标准 LLaVA 两阶段训练：

#### 阶段一：预训练（视觉-语言对齐）

```bash
# scripts/v1_5/pretrain.sh
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \        # ZeRO-2 (优化器状态分片)
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \                          # 纯描述文本，非对话格式
    --tune_mm_mlp_adapter True \               # ★ 仅训练 mm_projector
    --mm_projector_type mlp2x_gelu \           # 2层MLP投影器
    --vision_tower openai/clip-vit-large-patch14-336 \
    --learning_rate 1e-3 \                     # 较高学习率
    --bf16 True \
    --num_train_epochs 1
```

**关键点**：此阶段冻结视觉编码器和语言模型，**仅训练多模态投影器（mm_projector）**，将 CLIP 视觉特征映射到 LLM 的语义空间。

#### 阶段二：指令微调

```bash
# scripts/v1_5/finetune.sh
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \        # ZeRO-3 (参数+优化器+梯度分片)
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --version v1 \                             # 对话格式
    --learning_rate 2e-5 \                     # 较低学习率
    --group_by_modality_length True \          # 按模态长度分组采样
    --bf16 True
```

**关键点**：加载阶段一的 `mm_projector.bin`，全参数微调（或 LoRA/QLoRA）。

### 3.2 训练入口文件

| 文件 | 用途 |
|---|---|
| `llava/train/train_mem.py` | **实际入口**，使用 Flash Attention 2 加速 |
| `llava/train/train.py` | 训练主逻辑：数据预处理、Dataset、`train()` 函数 |
| `llava/train/train_xformers.py` | 使用 xFormers 替代 Flash Attention 的入口 |
| `llava/train/llava_trainer.py` | 自定义 `LLaVATrainer`，继承 HF Trainer，实现长度分组采样 |
| `llava/train/llama_flash_attn_monkey_patch.py` | 猴子补丁：替换标准注意力为 Flash Attention v2 |

### 3.3 训练数据流 (`train.py` 核心逻辑)

```
原始对话数据 (JSON)
  ↓ preprocess()  — 按格式(v1/v0/llama2/mpt/plain)预处理
  ↓ tokenizer_image_token() — 将 <image> token 替换为 IMAGE_TOKEN_INDEX
  ↓ LazySupervisedDataset — 懒加载，按需读取图像
  ↓ DataCollatorForSupervisedDataset — 批处理 + padding
  ↓ LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal()
      — 将图像通过 vision_tower → mm_projector 转为 embeddings
      — 替换 input_ids 中 IMAGE_TOKEN_INDEX 位置的 embeddings
  ↓ LlavaLlamaForCausalLM.forward() → LlamaForCausalLM.forward()
  ↓ CrossEntropyLoss (仅计算回答部分的 loss)
```

### 3.4 其他训练变体

| 脚本 | 用途 |
|---|---|
| `scripts/finetune_lora.sh` | LoRA (`rank=128, alpha=256`) |
| `scripts/finetune_qlora.sh` | QLoRA (4-bit + LoRA) |
| `scripts/finetune_full_schedule.sh` | 3 epoch 完整训练 |
| `scripts/finetune_sqa.sh` | ScienceQA 专用微调 |

### 3.5 DeepSpeed 配置

| 配置文件 | ZeRO Stage | 用途 |
|---|---|---|
| `scripts/zero2.json` | Stage 2 | 预训练（优化器状态分片） |
| `scripts/zero3.json` | Stage 3 | 全参数微调（参数+优化器+梯度分片） |
| `scripts/zero3_offload.json` | Stage 3 + CPU Offload | 显存极度受限时 |

### 3.6 关键训练超参数

| 参数 | 预训练 | 微调 |
|---|---|---|
| Learning rate | 1e-3 | 2e-5 |
| Batch size / GPU | 32 | 16 |
| Epochs | 1 | 1 |
| Weight decay | 0 | 0 |
| Warmup ratio | 0.03 | 0.03 |
| LR scheduler | cosine | cosine |
| Model max length | 2048 | 2048 |
| Gradient checkpointing | True | True |

---

## 4. 如何加速推理（VisiPruner 剪枝）

### 4.1 核心机制：`VisiPrunerLlamaAttention`

剪枝的核心代码在 `llava/model/language_model/custom_modeling_llama.py` 的 `VisiPrunerLlamaAttention` 类（第 496–735 行）。它替换了标准 `LlamaAttention`，在注意力计算中注入三层剪枝策略。

**注意**：`LlamaDecoderLayer`（第 1036 行）**硬编码**使用 `VisiPrunerLlamaAttention`（第 1042 行），这意味着**训练和推理都经过剪枝注意力模块**，但只有当 `pruning_config` 被设置时剪枝逻辑才生效。

### 4.2 三阶段剪枝策略

```python
pruning_config = {
    "mode": ["shallow", "middle", "deep"],  # 启用哪些阶段
    "shallow_mid_layer": 6,                 # shallow/middle 分界层
    "layer_threshold": 0.995,               # 中层: 余弦相似度阈值
    "tokens_threshold": 0.2,                # 中层: L2 距离阈值 (token保留比例)
}
```

#### Shallow layer 剪枝（`layer_idx < shallow_mid_layer`，第 687–699 行）

**原理**：浅层视觉 token 是 "passive attention sink"，可大幅削减。

- **7B 模型 `layer_idx==0`**：将所有视觉 token 对全部视觉 token 的注意力求和，重定向到第 35 号位置（第一个视觉 token），其余视觉-视觉注意力置零
- **13B 模型 `layer_idx==0`**：视觉-视觉注意力置零，后半部分视觉-文本注意力也置零
- **其他浅层**（`0 < layer_idx < shallow_mid_layer`）：将文本→视觉的交叉注意力置零

**效果**：浅层中文本 token 不再关注视觉 token，视觉 token 仅互相传递信息或充当 sink。

#### Middle layer 剪枝（`layer_idx > shallow_mid_layer`，第 716–717 行）

**原理**：跨模态融合在此突然发生，仅少数关键 visual token 驱动。使用 **value-aware token selection**。

**`value_aware_token_selection` 算法**（第 541–568 行）：

```
1. 取最后一个 token 的 attention 输出 attn_output_last
2. 计算每个视觉 token 对最后 token 的贡献 contribution[i] = attn_weight[i] * value[i]
3. 计算 masked_output = attn_output_last - contribution[i] （移除第 i 个视觉 token 的贡献）
4. 如果 cos_sim(masked_output, 原始output) < layer_threshold (0.995)
   → 该视觉 token 是关键的，保留
5. 进一步过滤：L2_norm(差异) > tokens_threshold (0.2)
```

在 prefill 阶段（`q_len > 1` 且 `q_len == position_ids[-1]+1`），此函数返回 `important_vis_tokens`（关键视觉 token 索引）。

在 `LlamaModel.forward()` 中（第 1389–1396 行），下一层将**丢弃非关键视觉 token**，只保留：
- 前 35 个 token（system prompt）
- 关键视觉 token
- 文本 token

#### Deep layer 剪枝（第 718–719 行）

**原理**：深层已不需要视觉信息，可以在确认视觉信息足够后，**完全丢弃所有视觉 token**。

使用类似的 value-aware 检查（第 560–568 行），但阈值更严格（`cos_sim < 0.999`），每层检查一次。当连续 2 次检查通过（`exit_indicator == 2`），在 `LlamaModel.forward()` 中（第 1398–1403 行）将所有剩余视觉 token 丢弃。

### 4.3 剪枝配置传递链路

```
CLI (cli_pruning.py)
  ↓ pruning_config dict
model.generate(pruning_config=pruning_config)
  ↓ llava_llama.py:LlavaLlamaForCausalLM.generate() [第126行]
self.set_pruning_config(pruning_config)
  ↓ 验证配置 validate_pruning_config()
  ↓ 传播到所有层
LlamaModel.set_pruning_config()
  ↓ 逐层设置
VisiPrunerLlamaAttention.set_pruning_config()
  设置 self.pruning_mode, self.shallow_mid_layer,
       self.layer_threshold, self.tokens_threshold
```

### 4.4 剪枝效果

根据论文数据（LLaVA-v1.5 7B）：

| 指标 | 效果 |
|---|---|
| 视觉注意力计算减少 | 最高 99% |
| 总 FLOPs 减少 | 最高 53.9% |
| 性能保持 | 显著优于 FastV 等已有方法 |
| 通用性 | 可泛化到多种 MLLM 架构 |

### 4.5 如何使用剪枝推理

#### 方式一：命令行

```bash
# 单 GPU 评估（以 GQA 为例）
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/visiPruner_eval/gqa.sh
```

评估脚本支持 `--pruning-config` 参数：

```bash
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --pruning-config '{"mode":["shallow","middle","deep"],"shallow_mid_layer":6,"layer_threshold":0.995,"tokens_threshold":0.2}' \
    ...
```

#### 方式二：Python 代码

```python
from llava.serve.cli_pruning import load_image, main
# 或直接使用 model.generate() 传入 pruning_config

pruning_config = {
    "mode": ["middle", "deep"],       # 仅启用 middle + deep
    "shallow_mid_layer": 6,
    "layer_threshold": 0.995,
    "tokens_threshold": 0.2,
}

with torch.inference_mode():
    output = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        pruning_config=pruning_config,  # ★ 关键参数
        use_cache=True,
        ...
    )
```

### 4.6 剪枝参数调优

| 参数 | 含义 | 调大效果 | 调小效果 |
|---|---|---|---|
| `shallow_mid_layer` | shallow/middle 分界层 | 扩大 shallow 范围，更激进 | 缩小 shallow，更保守 |
| `layer_threshold` | 中层 token 保留的余弦相似度阈值 | 更宽松（接近 1.0），保留更多 token | 更严格，保留更少 token |
| `tokens_threshold` | 中层 token 保留的 L2 距离阈值 | 更严格，保留更少 token | 更宽松，保留更多 token |
| `mode` | 启用哪些阶段 | 启用更多阶段 = 更激进剪枝 | 启用更少 = 更保守 |

### 4.7 其他加速技术（训练 + 推理通用）

| 技术 | 文件 | 说明 |
|---|---|---|
| **Flash Attention 2** | `llava/train/llama_flash_attn_monkey_patch.py` | 训练/推理时替换标准注意力，减少显存和加速 |
| **DeepSpeed ZeRO** | `scripts/zero2.json`, `zero3.json`, `zero3_offload.json` | 分布式训练显存优化 |
| **LoRA / QLoRA** | `scripts/finetune_lora.sh`, `finetune_qlora.sh` | 参数高效微调，减少可训练参数 |
| **Gradient Checkpointing** | 训练脚本中 `--gradient_checkpointing True` | 用计算换显存 |
| **BF16** | 训练脚本中 `--bf16 True` | 混合精度训练 |

---

## 5. 文件地图

### 模型层（核心）

| 文件 | 复杂度 | 作用 |
|---|---|---|
| `llava/model/language_model/custom_modeling_llama.py` | **极高** | ★ VisiPruner 的核心：`VisiPrunerLlamaAttention`（剪枝注意力）、`LlamaModel`（含剪枝 forward）、`LlamaDecoderLayer`、`LlamaForCausalLM`（含 `set_pruning_config`） |
| `llava/model/llava_arch.py` | 高 | LLaVA 多模态基类：`LlavaMetaModel`（视觉编码器+投影器初始化）、`LlavaMetaForCausalLM`（多模态输入预处理、图像 embedding 注入） |
| `llava/model/language_model/llava_llama.py` | 高 | LLaVA-Llama 适配器：`LlavaLlamaForCausalLM.generate()` — **pruning_config 入口** |
| `llava/model/builder.py` | 高 | 模型加载：`load_pretrained_model()`，支持 LoRA、量化、多架构 |
| `llava/model/multimodal_encoder/clip_encoder.py` | 高 | CLIP 视觉编码器（标准 + 多尺度 S2） |
| `llava/model/multimodal_projector/builder.py` | 中 | 视觉投影器：IdentityMap、SimpleResBlock、MLP |

### 训练层

| 文件 | 作用 |
|---|---|
| `llava/train/train.py` | 训练主逻辑：数据预处理、Dataset/Collator 类、`train()` 函数 |
| `llava/train/train_mem.py` | 训练入口（使用 Flash Attention） |
| `llava/train/llava_trainer.py` | 自定义 HF Trainer：长度分组采样、多模态适配器优化器管理 |
| `llava/train/llama_flash_attn_monkey_patch.py` | Flash Attention 猴子补丁 |

### 推理/服务

| 文件 | 作用 |
|---|---|
| `llava/serve/cli_pruning.py` | ★ VisiPruner 剪枝推理 CLI 示例 |
| `llava/serve/cli.py` | 标准 CLI 推理 |
| `llava/serve/model_worker.py` | 分布式推理 Worker |
| `llava/serve/controller.py` | 分布式推理 Controller（负载均衡） |

### 训练脚本

| 脚本 | 用途 |
|---|---|
| `scripts/v1_5/pretrain.sh` | 预训练：仅训练 mm_projector |
| `scripts/v1_5/finetune.sh` | 全参数微调 |
| `scripts/v1_5/finetune_lora.sh` | LoRA 微调 |
| `scripts/finetune_qlora.sh` | QLoRA 微调 |
| `scripts/finetune_full_schedule.sh` | 完整 3-epoch 微调 |

### VisiPruner 评估脚本

| 脚本 | 基准 |
|---|---|
| `scripts/v1_5/visiPruner_eval/gqa.sh` | GQA |
| `scripts/v1_5/visiPruner_eval/mme.sh` | MME |
| `scripts/v1_5/visiPruner_eval/textvqa.sh` | TextVQA |

### 可视化分析

| Notebook | 内容 |
|---|---|
| `visualization/project_vo_to_semantic_space.ipynb` | Logits Lens：观察各层 value-output 在语义空间的投影 |
| `visualization/project_hidden_states_to_semantic_space.ipynb` | Hidden States 语义投影 |
| `visualization/attention_visualization.ipynb` | 注意力分布可视化 |
| `visualization/L1_norms_of_value_matrix.ipynb` | Value 矩阵 L1 范数分析（attention sink 证据） |

---

## 6. 关键设计决策与概念

### 6.1 `set_num_images` 的作用

在 `VisiPrunerLlamaAttention.set_num_images()`（第 570 行）中：
- `self.vis_end_index = 35 + 576 * num_images` — 计算视觉 token 结束位置
- `35` 是 system prompt 的固定 token 数
- `576 = 24×24` 是 CLIP-ViT-L/336 的 patch 数

这个索引用于确定哪些 token 是视觉 token，在剪枝时精确区隔文本/视觉区域。

### 6.2 `CustomGenerationMixin`

`llava/model/language_model/custom_generation.py` 覆盖了 HuggingFace `GenerationMixin._validate_model_kwargs()`，**禁用了未使用参数的检查**，使得 `pruning_config` 等自定义参数可以传递而不报错。

### 6.3 Attention 类注册表

```python
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,            # 标准注意力
    "visi_pruner": VisiPrunerLlamaAttention,  # ★ VisiPruner 剪枝注意力
    "flash_attention_2": LlamaFlashAttention2,  # Flash Attention 2
    "sdpa": LlamaSdpaAttention,         # PyTorch SDPA
}
```

**注意**：当前 `LlamaDecoderLayer.__init__()` 硬编码使用 `VisiPrunerLlamaAttention`，不依赖 config 的 `_attn_implementation` 选择。

### 6.4 数据格式约定

- 视觉 token 使用 `IMAGE_TOKEN_INDEX`（一个特殊整数）在 `input_ids` 中占位
- `prepare_inputs_labels_for_multimodal()` 将图像通过 vision_tower→mm_projector 转为 embeddings，替换占位符
- 支持多种对话格式：`llava_v1`, `llava_v0`, `llava_llama_2`, `mistral_instruct`, `chatml_direct`, `mpt`

---

## 7. 复杂度热点

| 文件 | 复杂度 | 原因 |
|---|---|---|
| `custom_modeling_llama.py` | **极高** | 1760 行，包含完整 LLaMA 实现 + VisiPruner 注意力 + Flash Attention 2 + SDPA + 剪枝 token 选择逻辑 + 动态 attention mask 调整 |
| `llava_arch.py` | 高 | 多模态输入预处理 + 多尺度图像 + anyres + spatial merge |
| `llava_llama.py` | 高 | 多模态 forward/generate 重写 |
| `conversation.py` | 高 | 多轮对话管理，多种格式 |
| `train.py` | 高 | 完整训练流程 + 多种对话格式预处理 |

**建议**：理解项目时从 `cli_pruning.py`（120 行，清晰展示剪枝调用）→ `llava_llama.py`（generate 入口）→ `custom_modeling_llama.py` 的 `VisiPrunerLlamaAttention.forward()` 入手。

---

## 8. 快速开始命令

```bash
# 安装
conda create -n llava_visiPruner python=3.10 -y
conda activate llava_visiPruner
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# 剪枝推理（GQA 评估）
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/visiPruner_eval/gqa.sh

# 预训练
bash scripts/v1_5/pretrain.sh

# 微调
bash scripts/v1_5/finetune.sh
```

---

*基于知识图谱 `knowledge-graph.json`（2026-06-16 分析，commit `97fed23`）和源码深度阅读生成。*
