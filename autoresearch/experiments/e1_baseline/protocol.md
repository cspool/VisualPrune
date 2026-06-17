# E1: Baseline Gap — theoretical 2.17× vs actual wall-clock speedup

## 实验目的

Gap 多大？L1 占比多少？即 VisiPruner 的理论加速比（2.17×）与实际端到端 GPU 加速比之间的差距是多少，以及 L1 cache 命中率在此差距中扮演的角色。

## 核心假设

1. 实际加速比远小于理论 2.17×，主要因为 VisiPruner 引入了大量 memory-bound 小算子，导致 L1 cache miss 增加。
2. Dense-FA2 是实际速度基线（最快后端），VisiPruner 可能仅与 Dense-FA2 相当甚至更慢。
3. L1 数据搬运占比（L1/T_total）低于 50%，L2/L3 延迟成为 dominant factor。

## 预测

- VisiPruner-Full vs Dense-FA2：实际加速比 < 1.0×（即 VisiPruner 更慢或持平）。
- L1 时间占比 < 50%，L2+L3 合计 > 50%。
- VisiPruner-Shallow-Only 加速比略高于 Full，但仍不及 Dense-FA2。

## 实验设计

### Configs

| Config ID | 描述 | 后端 | Pruning |
|-----------|------|------|---------|
| dense-fa2 | Dense baseline, Flash Attention 2 | FA2 | 无 |
| dense-eager | Dense baseline, Eager mode | Eager | 无 |
| visipruner-full | VisiPruner 完整流水线 | Eager | 全部层 |
| visipruner-shallow-only | 仅 shallow layer pruning | Eager | 仅 shallow 层 |

### 公共参数

- 模型：LLaVA-ViSU-7B
- 输入：单图 + 固定长度 prompt（256 tokens）
- Precision：bf16
- 每个 config 独立运行（单次推理即可，nsys 采样覆盖全量 kernel）

### 测量方法（统一两层）

**禁止使用**：Python `time.time()` / `CUDATimer` wall-clock 计时、CUPTI / `torch.profiler`。

所有性能数据来源：nsys（时间维度）+ ncu（硬件计数器维度）。

### 主要 Metrics

| Metric | 含义 | 来源 |
|--------|------|------|
| T_kernel_sum | GPU kernel duration 之和（ms，不含 CPU/launch gap） | nsys `cuda_gpu_kern_sum` |
| T_e2e_nsys | profiled inference 端到端时间（ms） | nsys SQLite 中 NVTX `e1_inference` range |
| Top-10 kernel list | 耗时前 10 的 kernel 名 + 各自 GPU 时间 + 占比 | nsys stats export (SQLite / CSV) |
| kernel_sum_speedup | T_kernel_sum_dense / T_kernel_sum_visipruner | 计算值 |
| nsys_e2e_speedup | T_e2e_nsys_dense / T_e2e_nsys_visipruner | 计算值 |
| L1/T_kernel_sum | L1 cache 时间占比（热点 kernel） | ncu metrics |
| L2/T_kernel_sum | L2 cache 时间占比（热点 kernel） | ncu metrics |
| L3/T_kernel_sum | L3 cache 时间占比（热点 kernel） | ncu metrics |
| bottleneck_class | 每个热点 kernel 的瓶颈类型 | ncu `--analysis-metrics` |
| U_bw / U_compute / O_SM | 内存带宽利用率 / 计算利用率 / SM 占用率 | ncu metrics |

---

## Step 1: nsys 全量采样 — 定位热点 kernel

**目标**：获取每个 config 的完整 GPU 时间线，识别耗时最长的 kernel（类别和单个 kernel）。

对每个 config 运行 nsys，**不做任何 kernel 过滤**，收集全量 GPU kernel 时间分布：

```bash
for config in dense-fa2 dense-eager visipruner-full visipruner-shallow-only; do
    nsys profile \
        --trace cuda,nvtx,cublas \
        --stats=true \
        --force-overwrite=true \
        --output e1_nsys_${config} \
        python profiling/bench_e1_baseline.py \
            --config ${config} \
            --mode nsys-profile \
            --max-new-tokens 128
done
```

**nsys 输出分析**：

从 nsys SQLite 数据库导出 kernel 统计表：

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
    --output e1_nsys_${config}_kernels.csv \
    e1_nsys_${config}.nsys-rep
```

提取每个 config 的：
1. **Top-10 单 kernel**：按 GPU time 降序排列，记录 kernel 名、耗时、占比、调用次数
2. **Kernel 类别聚合**：按类别（GEMM / softmax / elementwise / reduction / attention / pruning-specific）聚合 GPU time
3. **T_kernel_sum**：所有 kernel duration 的总和
4. **T_e2e_nsys**：NVTX `e1_inference` range 的端到端持续时间

**产出（nsys 阶段）**：
- `e1_nsys_top10_kernels.csv` — 每个 config 的 top-10 kernel 表
- `e1_nsys_kernel_categories.csv` — 每个 config 按类别聚合的 GPU time
- `e1_nsys_timeline_summary.csv` — T_total 和加速比

**关键决策点**：基于 nsys 结果，确定每个 config 需要 ncu 深入分析的 3-5 个 kernel。选择标准：
- 单 kernel GPU time 占比 > 5%
- 或 kernel 类别在 VisiPruner config 中占比显著高于 Dense config

---

## Step 2: ncu 深度分析 — 对热点 kernel 逐一剖析

**目标**：对 Step 1 识别出的热点 kernel，获取硬件计数器级别的瓶颈分类和 cache 行为。

**前提**：Step 1 的 top-10 kernel 表已完成，热点 kernel 名称已知。

对每个 config 的热点 kernel 分别运行 ncu（使用 `--kernel-name` 精确匹配，避免全量 replay）：

```bash
# 示例：对 dense-fa2 的热点 kernel
for kernel_pattern in "flash_attention" "matmul" "gemm"; do
    ncu --set full \
        --kernel-name regex:"${kernel_pattern}" \
        --launch-count 50 \
        --launch-skip 5 \
        --import-source yes \
        --output e1_ncu_dense_fa2_${kernel_pattern} \
        python profiling/bench_e1_baseline.py \
            --config dense-fa2 \
            --mode ncu-profile \
            --max-new-tokens 32
done

# 示例：对 visipruner-full 的热点 kernel（包括 pruning 相关）
for kernel_pattern in "cosine_similarity|norm|index|where|permute|topk" \
                      "matmul" \
                      "scaled_dot_product"; do
    ncu --set full \
        --kernel-name regex:"${kernel_pattern}" \
        --launch-count 50 \
        --launch-skip 5 \
        --import-source yes \
        --output e1_ncu_visipruner_${kernel_pattern} \
        python profiling/bench_e1_baseline.py \
            --config visipruner-full \
            --mode ncu-profile \
            --max-new-tokens 32
done
```

**ncu 指标选择**（`--set full` 已包含以下核心指标）：

| Metric | 含义 | 用于判断 |
|--------|------|---------|
| `gpu__time_duration.sum` | 该 kernel 的总 GPU 时间 | 与 nsys 交叉验证 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算吞吐占峰值比例 | compute utilization |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽占峰值比例 | memory utilization |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_read.sum` | L1 读取 sectors | L1 数据搬运量 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | L2 读取 sectors | L2 数据搬运量 |
| `dram__bytes_read.sum` + `dram__bytes_write.sum` | DRAM 总流量 | 绝对 DRAM 压力 |
| `smsp__warps_active.avg.pct_of_peak_sustained_elapsed` | 活跃 warp 比例 | SM 占用率 |
| `launch__time_total` | kernel launch 开销 | L2 运行时开销 |

**瓶颈分类**（基于 ncu 指标）：

```
Compute-bound:  U_compute > 60% AND U_bw < 60%
Memory-bound:   U_bw > 60%      AND U_compute < 60%
Latency-bound:  O_SM < 60%      AND U_bw < 30% AND U_compute < 30%
```

---

## Step 3: 汇总分析

将 nsys 和 ncu 数据合并，产出最终报告。

### 交叉验证

- nsys 的 T_total 与 ncu 各 kernel `gpu__time_duration` 之和对比，确认覆盖完整
- nsys 的 top-10 kernel 排名与 ncu 的硬件指标交叉解释

### 预期指标参考

| Config | T_total (ms) | L1% | L2% | L3% | Dominant Bottleneck |
|--------|-------------|-----|-----|-----|---------------------|
| dense-fa2 | 最低 | - | - | - | Compute-bound (GEMM) |
| dense-eager | ~1.1-1.3× | - | - | - | Memory-bound (attention) |
| visipruner-full | > dense-fa2 | < 50% | ~30% | ~20% | Memory-bound (pruning ops) |
| visipruner-shallow-only | 略低于 full | < 50% | ~30% | ~20% | Memory-bound |

## 产出物

1. **`e1_nsys_top10_kernels.csv`** — 每个 config 的 top-10 kernel（nsys），含 GPU time + 占比 + 调用次数
2. **`e1_nsys_kernel_categories.png`** — 各 config 按 kernel 类别（GEMM, softmax, reduce, elementwise, pruning-specific）的 stacked bar chart
3. **`e1_ncu_per_kernel_bottleneck.csv`** — 每个热点 kernel 的 ncu 瓶颈分类 + U_bw / U_compute / O_SM
4. **`e1_roofline.png`** — 各 config 的热点 kernel 在 roofline 模型上的位置
5. **`e1_dram_bytes.png`** — 各 config 的 DRAM bytes 柱状图（热点 kernel 合计）
6. **`e1_l1_l2_l3_stacked.png`** — 各 config 的 cache level 时间占比 stacked bar（热点 kernel 按时间加权）
7. **`e1_gap_summary.json`** — 汇总 T_total、actual_speedup、L1/L2/L3%、dominant bottleneck
