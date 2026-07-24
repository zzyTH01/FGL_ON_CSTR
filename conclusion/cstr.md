# CSTR 实验文档

## 1. 概述

CSTR（Continuous Stirred Tank Reactor，连续搅拌釜式反应器）模块是将 FGL（Future-Guided Learning）方法应用于化学反应过程时序预测的实验扩展。该模块不属于原论文（Nature Communications 2025）的内容，为后续独立扩展工作。

**核心任务**：利用 FGL 方法预测 CSTR 中 H₂/O₂ 燃烧反应的 H₂O 质量分数时间序列。

---

## 2. 文件结构

```
cstr/
├── generate.py                  # 数据生成脚本（Cantera 模拟）
├── analyze_data.py              # 数据分析脚本（统计 + 振荡特征）
├── plot_data.py                 # 可视化脚本（5 张图）
├── cstr.md                      # 本文档
├── data.pkl                     # 温度序列数据 (3001, 2)
├── data_h2o.pkl                 # H₂O 质量分数序列 (3001, 2) ← 默认使用
│
├── plots_original/              # 原始数据可视化（6 张）
├── plots_forced/                # 外部驱动对比可视化（4 张）
│
├── exp/
│   ├── fgl_cstr.py              # FGL 实验主脚本（分类模式，离散化 + KL 蒸馏）
│   ├── fgl_cstr_regression.py   # FGL 实验（回归模式，连续值 + MSE 蒸馏）
│   ├── fgl_cstr_seq2seq.py      # FGL 实验（Seq2Seq 模式，周期预测 + 序列 KL 蒸馏）
│   └── fgl_cstr_lstm.py         # FGL 实验（LSTM 替换 RNN）
│
└── *.png                        # 可视化输出（5 张图）
```

### 依赖关系

```
cstr/exp/fgl_cstr.py
    │
    ├── from utils.utils import RNN, create_time_series_dataset, KL
    │       ↑
    │       └── mackey_glass/utils/utils.py   （论文原始共用代码）
    │
    └── cstr/data_h2o.pkl                     （本模块生成的数据）
```

---

## 3. 数据生成 (`generate.py`)

### 物理模型

使用 **Cantera** 化学动力学库，模拟化学计量比 H₂/O₂ 混合气体在 CSTR 中的反应过程。反应由 H₂O 作为第三体链终止剂引发的周期性振荡驱动：

$$\mathrm{H + O_2 + M \rightleftharpoons HO_2 + M}$$

### 模拟参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 初始温度 | 770 K | |
| 压力 | 60 Torr (7999 Pa) | |
| 进气组成 | H₂:2, O₂:1 | 化学计量比 |
| 反应器体积 | 10 cm³ | |
| 质量流量 | 12 sccm | |
| 传热系数 | 0.02 | 壁面散热 |
| 模拟时间 | 300 s | |
| 时间步长 | 0.1 s | |
| 总数据点 | 3001 | |

### 输出文件

| 文件 | 内容 | 形状 | 特征 |
|------|------|------|------|
| `data.pkl` | 温度 (K) | (3001, 2) | 周期性尖峰，770~2000 K |
| `data_h2o.pkl` | H₂O 质量分数 | (3001, 2) | 平滑连续振荡，0~0.96 |

数据格式：`(N, 2)` float64 tensor，两列为相同序列值（与 Mackey-Glass 数据格式一致）。

默认使用 `data_h2o.pkl`（连续振荡更适合 FGL 框架）。

### 生成命令

```bash
uv run python cstr/generate.py
```

---

### 3.1 数据集深度分析

#### 3.1.1 振荡机制：交替弱/强点火

CSTR 表现出**倍周期分岔**（period-doubling bifurcation）。H₂O 作为高效的第三体链终止剂：一旦反应产生足够 H₂O，链式反应被终止；等 H₂O 从出口排空后，混合物再次点燃。系统不需要外部周期驱动，完全由反应动力学和物质输运的耦合产生自激振荡。

点火事件每 ~7.15s 发生一次，但**强弱交替**：

```
时间 →
  弱点火    强点火    弱点火    强点火    弱点火
  T≈810K   T≈1400K   T≈810K   T≈1400K   T≈810K
  H2O≈0.96  H2O≈0.95  H2O≈0.96  H2O≈0.95  H2O≈0.96
    |         |         |         |         |
    ├─── 7.15s ──┤─── 7.15s ──┤         |
    ├──────────── 14.3s 完整周期 ────────┤
```

- **弱点火**：温度仅升至 ~810K，持续 0.1s（单步）即回落
- **强点火**：温度飙升至 1400-2000K，持续 0.1s（单步）；峰值递增（1370→1394→1419→1445→1472K...），因为每次强点火后残余 H₂O 略多，下一次需要更高温度才能点燃

物理原因：前一次强点火残留的 H₂O 抑制了下一次反应的强度，形成"强-弱-强-弱"的交替模式。这与论文中的 Mackey-Glass 混沌系统有本质区别：CSTR 是完全确定的周期振荡。

#### 3.1.2 Temperature vs H₂O 对比

| 特性 | Temperature `data.pkl` | H₂O `data_h2o.pkl` |
|------|----------------------|---------------------|
| 数据形态 | **几乎是一条水平线** + 21个单点尖峰 | 平滑锯齿波，连续振荡 |
| 范围 | 770 ~ 2018 K | 0.000 ~ 0.964 |
| 均值 ± 标准差 | 777.0 ± 76.8 K | 0.180 ± 0.223 |
| >基线值占比 | **仅 1.4%** 的点 >800K | 91.5% 的点 >0.01 |
| 异常点数 | 21 / 3001 (0.7%) | 0 |
| 尖峰宽度 | **0.1s（单步单点）** | N/A（无尖峰结构） |
| 尖峰之间的信息 | 无（全是 770K 恒温基线） | 完整的指数衰减曲线 |
| 周期 | 14.3s (143步) — 仅强点火 | 7.15s (72步) — 子振荡 |
| 互相关 | — | H₂O 领先 Temperature 7.1s |

**关键结论**：温度数据 98.6% 的点是恒定基线（770K），21 个尖峰每个仅持续单步。对于 lookback=8（0.8s）的历史窗口，模型绝大多数时候看到的是一段水平线。温度预测等价于**极端稀疏事件检测**——3001 个样本中仅 21 个正例。这也是为什么当前代码默认使用 H₂O 而非温度。

#### 3.1.3 H₂O 子振荡微观结构（100-130s 窗口）

```
Time    Temp    H2O        解释
100.0   770.0   0.013      接近基线
101.0   770.1   0.007      接近耗尽
102.0   772.8   0.429  ←   突然跃升（反应部分重燃）
103.0   770.0   0.228      指数衰减中 ↓
104.0   770.0   0.117      ↓
105.0   770.0   0.059      ↓
106.0   770.0   0.030      ↓
107.0   770.0   0.015      ↓
108.0   770.0   0.007      再次接近耗尽
109.0   772.8   0.473  ←   又一次跃升（峰值递增）
```

H₂O 在一个时间步内从 ~0.007 跳变到 ~0.4-0.7（反应瞬间产生大量水），然后在 ~7 步（0.7s）内指数衰减回 ~0.01，形成锯齿状子振荡。同时子振荡的峰值在缓慢增长（0.43→0.47→0.53→0.60→0.68），处于大周期的累积阶段，最终达到 ~0.96 后触发下一次点火。

#### 3.1.4 滑动窗口覆盖分析

| 参数 | 值 | 占子周期 (72步) | 占完整周期 (143步) |
|------|-----|:---:|:---:|
| lookback=8 | 0.8s | 11% | 5.6% |
| H=5 预测跨度 | 0.5s | 7% | 3.5% |
| 总时间跨度 | 1.3s | 18% | 9.1% |

lookback=8 仅覆盖完整周期的 ~5.6%，模型本质上"看不清"当前处于哪个振荡阶段。H=5 的目标仅 0.5s 之后，此时 H₂O 仍处在同一条指数衰减曲线上，预测任务过于简单。

#### 3.1.5 离散化分析（50 bins, H₂O）

- Bin 边界：均匀分布 `[0.0000, 0.9644]`，每 bin 宽 ~0.020
- H₂O 的单步跳变（如 0.007→0.429）跨越约 22 个 bin，但相邻步之间通常仅跨越 1-2 个 bin
- 7/47 个 bin 的样本数 <10，4 个 bin 的样本数 <0.1%
- Bin 分布严重不平衡：bin 1（近零区）有 670 个样本，而高值区 bin 稀疏

#### 3.1.6 对 FGL 方法的含义

| 问题 | 影响 |
|------|------|
| lookback=8 仅覆盖 5.6% 完整周期 | 模型看不到完整振荡结构，无法判断当前相位 |
| H=5 仅预测 0.5s 后 | 目标仍在同一衰减曲线上，任务太简单，Teacher 无信息优势 |
| H₂O 跃升是单步突变 | 离散化 50 bin 在跳变处产生极大的 bin 跨度（22 bins/步） |
| 周期完全确定（非混沌） | 没有混沌性，Teacher 的 1 步预测对周期性信号无额外信息价值 |
| 数据仅 3001 点 | 约 42 个子周期 / 21 个完整周期，训练/验证/测试各覆盖约 25/8/8 个周期 |
| 温度数据 98.6% 是基线 | 温度序列几乎无可用信息，稀疏事件检测极具挑战 |

---

## 4. FGL 实验脚本 (`exp/fgl_cstr.py`)

### 4.1 整体架构

```
CSTR 数据 (data_h2o.pkl)
        │
        ├─ create_time_series_dataset(horizon=1, offset=H-1) ──→ Teacher 训练
        │                                                         │
        ├─ create_time_series_dataset(horizon=H, offset=0)  ──→ Baseline 训练
        │                                                         │
        │                                              KL 蒸馏 (冻结 Teacher)
        │                                                         │
        └─ create_time_series_dataset(horizon=H, offset=0)  ──→ Student 训练 (FGL)
                                                                  │
                                                  ┌───────────────┴───────────────┐
                                                  ↓                               ↓
                                           普通评估 (MSE)              Page-Hinkley 评估 (MSE)
```

### 4.2 模型架构

复用 Mackey-Glass 的 `RNN` 模型：

```
RNN(input_size, hidden_size=128, output_size=num_bins, num_layers=2)
  ├── nn.RNN(input_size, 128, 2 layers, batch_first=True, dropout=0.2)
  ├── nn.Linear(128, 128)
  ├── ReLU
  └── nn.Linear(128, num_bins)
```

输入形状：`(batch, 1, lookback_window)`；输出形状：`(batch, num_bins)`

### 4.3 三阶段训练流程

#### 阶段 1：Teacher（教师）训练
- **任务**：1 步预测（forecasting_horizon=1）
- **时间对齐**：offset = H-1，使 Teacher 的输入窗口相对于 Student 前移 H-1 步
- **损失函数**：CrossEntropyLoss
- **早停机制**：patience=5

#### 阶段 2：Baseline（基线）训练
- **任务**：H 步预测（forecasting_horizon=H）
- **不使用** Teacher 引导
- **对照目的**：作为 FGL 方法的基准线

#### 阶段 3：Student（学生）训练（FGL 核心）
- **任务**：H 步预测 + 知识蒸馏
- **损失函数**：

$$\mathcal{L} = \alpha \cdot \text{CE}(\text{student\_output}, \text{ground\_truth}) + (1-\alpha) \cdot T^2 \cdot \text{KL}\left(\text{softmax}\frac{\text{teacher\_logits}}{T} \;\|\; \log\text{-softmax}\frac{\text{student\_logits}}{T}\right)$$

- **Teacher 冻结**：`torch.no_grad()` 计算 Teacher logits
- **数据流**：`zip(student_train, teacher_train)` 对齐训练

### 4.4 评估方法

#### 普通评估
```python
pred_bin = model(x).argmax(dim=1).float()   # 预测 bin 索引
mse = MSE(pred_bin, y_bin)                   # 与真实 bin 索引的 MSE
```

#### Page-Hinkley 漂移检测评估 (`--use_ph`)
- 逐样本计算预测误差
- 累积 Page-Hinkley 统计量
- 当 PH > λ_threshold 时，用最近窗口数据重新微调模型（retrain_epochs=3）
- 重置 PH 统计量和窗口

### 4.5 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--horizon` | 5 | Student 预测步长 H |
| `--alpha` | 0.5 | CE loss 权重 (0=纯蒸馏, 1=纯CE) |
| `--num_bins` | 50 | 离散化 bin 数量 |
| `--epochs` | 30 | 训练轮数 |
| `--temperature` | 4 | 蒸馏温度 T |
| `--lookback_window` | 8 | 历史窗口长度 |
| `--batch_size` | 64 | 批大小 |
| `--val_size` | 0.2 | 验证集比例 |
| `--test_size` | 0.2 | 测试集比例 |
| `--patience` | 5 | 早停耐心值 |
| `--sweep` | False | 运行 horizon 扫描 |
| `--sweep_range` | "2,31" | 扫描范围 |
| `--use_ph` | False | 启用 Page-Hinkley 评估 |

### 4.6 运行命令

```bash
# 单次运行
python cstr/exp/fgl_cstr.py --horizon 5 --alpha 0.5 --epochs 30

# Horizon 扫描
python cstr/exp/fgl_cstr.py --sweep --alpha 0.5 --epochs 30

# 带 Page-Hinkley 漂移检测
python cstr/exp/fgl_cstr.py --sweep --use_ph --alpha 0.0
```

---

## 5. Bug 发现与修复

### 5.1 Bug：Teacher 与 Student 离散化 Bin 不一致

**发现时间**：代码审查过程中

**根因**：`create_time_series_dataset` 根据各自的 `y_train` 计算 bin 边界。由于 Teacher（H=1）比 Student（H≥2）产生更多滑动窗口，两者的 `y_train` 长度不同，导致 min/max 略有差异，从而产生不同的 bin 边界。

```
修复前实测（H=5）：
  Teacher y_train: 1795 values, range=[0.000011, 0.961763]
  Student y_train: 1793 values, range=[0.006688, 0.961763]
  Bin edges equal: False  ← BUG!
```

**后果**：Teacher 输出的 "bin 38" 和 Student 预测的 "bin 38" 代表不同的数值范围，KL 散度在不可比较的分布之间计算，蒸馏信号无效。

**影响评估**：
- CSTR（3001 点）：影响严重，bin edges 差异明显
- Mackey-Glass（10000 点）：影响轻微，min/max 通常一致

### 5.2 修复方案

#### 修改 1：`mackey_glass/utils/utils.py`

`create_time_series_dataset` 新增参数 `bin_edges: np.ndarray = None`：

```python
# 修复前：始终从 y_train 计算
bin_edges = np.linspace(y_train.min(), y_train.max(), num_bins - 1)

# 修复后：支持外部传入
if bin_edges is None:
    bin_edges = np.linspace(y_train.min(), y_train.max(), num_bins - 1)
```

#### 修改 2：`cstr/exp/fgl_cstr.py`

在 `run_fgl()` 中，于调用 `create_time_series_dataset` 之前，从 Teacher 的全量 y_windows 计算统一的 bin edges：

```python
# 从 Teacher 的全量 y_windows（未分 train/val/test）计算共享 bin edges
x_raw = np.array([float(pt[0]) for pt in data])
y_raw = np.array([float(pt[1]) for pt in data])
all_y_windows = []
for i in range(len(x_raw) - lookback_window - 1 + 1):
    all_y_windows.append(y_raw[i + lookback_window + 1 - 1])
all_y = np.array(all_y_windows)
shared_bin_edges = np.linspace(all_y.min(), all_y.max(), num_bins - 1)

# 同时传给 Teacher 和 Student
create_time_series_dataset(..., bin_edges=shared_bin_edges)
```

#### 修复验证

```
修复后：
  Bin edges: [0.0000, 0.9644] → 50 bins  （全部 H 统一）
  Teacher × Student target 匹配：1792/1792 (100%)  ← 完美对齐
```

---

---
## 6. 原始实验结果

修复 bin edge bug 后，α×T 网格搜索（36 组）和 horizon 扫描（H=3→9）均在 0% 附近。FGL 对 CSTR 无效果。

---

## 7. 优化尝试

为排除实现层面的问题，依次尝试了六个方向。所有实验均在 H₂O 数据上运行（epochs=30, α=0.5 除非特别说明）。

### 7.1 原始基线

**措施**：使用默认参数（H₂O, L=8, RNN 分类），跑 α×T 网格搜索（36 组）和 horizon 扫描（H=3→9）。

**结果**：36 组 α×T 组合全部为负，最优 α=0.9, T=8 → Δ=-0.4%。Horizon 扫描 avg Δ=-0.9%，仅 H=7 出现了唯一一次正向（+0.4%）。

**原因**：FGL 在 CSTR 上完全无效，首先排除了"超参数不合适"的可能。

### 7.2 温度数据

**措施**：温度序列有周期性点火尖峰（770→2000K），类似论文中 EEG 的癫痫事件。改用 `data.pkl` 配合 L=3。

**结果**：Teacher、Baseline、Student 三者 MSE 完全相等（14.42），Δ=0.0%。模型坍塌为常数预测器（始终输出 bin 0 = 770K）。

**原因**：温度数据 98.6% 为 770K 基线，仅有 21 个单步尖峰（0.7%）。极度类别不平衡导致模型学到"始终预测无事件"即可获得极低 loss。

### 7.3 缩短 lookback

**措施**：原始 L=8 覆盖子周期 11%。缩短至 L=3 和 L=4，试图制造"信息不对称"——Teacher 的近未来信号或许对信息匮乏的 Student 更有价值。

**结果**：

| 配置 | Avg Δ | 对比 L=8 |
|------|:---:|:---:|
| L=8 (原始) | -0.9% | — |
| L=3 | -2.2% | 恶化 2.4× |
| L=4 | -4.0% | 恶化 4.4× |

**原因**：lookback 越短，Teacher 自身的预测质量也同步下降（L=3 时 Teacher MSE 从 31.3 升至 34.8）。Teacher 和 Student 同样匮乏——不存在独属于 Teacher 的优势。

### 7.4 回归模式

**措施**：跳过离散化，直接预测连续值。新建 `fgl_cstr_regression.py`。模型输出 1 个标量，蒸馏损失改为 α·MSE(pred, truth) + (1-α)·MSE(pred, teacher_pred)。

**结果**：H=3→9 全部为负，avg Δ=-18.1%（所有方向中最差）。α 敏感性：越低越差（α=0.0 → -12.9%，α=0.9 → -8.5%）。

**原因**：Teacher 的 1 步预测值 ≠ Student 的 H 步目标值。例如 H₂O 当前 0.3 且衰减：Teacher 预测 0.28（t+1 正确），Ground Truth 是 0.10（t+5 正确）。MSE 蒸馏强迫 Student 折中到 ~0.19——两者皆错。这解释了原论文使用 KL 散度（分布→分布，传递不确定性形状）而非 MSE（标量→标量，传递错误的数值目标）。

### 7.5 LSTM 替换 RNN

**措施**：新建 `fgl_cstr_lstm.py`，将 RNN 替换为 2 层 LSTM（hidden_size=128），其余架构完全不变。

**结果**：

| | Teacher (H=5) | Baseline (H=5) | Avg Δ |
|------|:---:|:---:|:---:|
| RNN | 31.3 | 107.9 | -0.9% |
| LSTM | 30.4 | 108.1 | -2.0% |

**原因**：LSTM Teacher 略有改善（30.4 vs 31.3），但 Baseline 完全持平（108.1 vs 107.9）。门控机制无法从 8 步确定性周期输入中提取额外信息。**瓶颈不在模型容量——在数据本身。**

### 7.6 Seq2Seq 周期预测

**措施**：新建 `fgl_cstr_seq2seq.py`。Teacher 预测前 K 步（短序列，容易），Student 预测全 72 步（完整子周期，困难）。两者使用相同输入窗口，Teacher 优势来自 K << 72 的任务难度差。KL 蒸馏仅作用于前 K 步。

**结果**：K=3→19 全部为负，avg Δ=-19.8%。即使 K=3 时 Teacher 远优于 Baseline（MSE 54.6 vs 91.4），Student 仍比 Baseline 差 20.7%。α=0.0 时 Student 直接坍塌（Δ=-447%）。

**原因**：Teacher 的前 K 步预测虽准确，但信息源和 Student 完全相同（同一个 lookback 窗口）。对确定性周期信号，从 8 步历史推断前 K 步的模式，Student 自己也能做到——不需要 Teacher 来教。

### 7.7 小结

| # | 方向 | 最佳 Δ | 排除的假设 |
|---|------|:---:|------|
| 1 | 原始基线 | +0.4% | 超参数不合适 |
| 2 | 温度数据 | 0.0% | 数据类型问题 |
| 3 | 缩短 lookback | -0.1% | 信息不对称不够 |
| 4 | 回归模式 | -2.3% | 离散化损失信息 |
| 5 | LSTM | +0.3% | 模型容量不足 |
| 6 | Seq2Seq | -6.3% | 任务难度差不够 |

**六个方向逐一排除了实现层面的所有可能解释。问题不在代码、模型或超参数——在 CSTR 数据本身的动力学结构。**

---

## 8. 参数扫描与外部驱动

为排查"CSTR 的周期性是否是根因"，进行了两轮尝试。

**参数扫描**（`cstr/param_sweep.py`）：在 U、K、流量三个维度上扫描 21 组参数。全部 periodicity > 0.93，无一组产生非周期行为。H₂/O₂ 燃烧的负反馈开关天然产生鲁棒的周期 1 振荡。

**外部正弦驱动**（`cstr/generate_forced.py`）：对进气流量施加正弦扰动（A=0.5, f=0.05Hz），periodicity 从 0.952 降至 0.489。FGL Δ 仍为 ~0%。外部驱动产生的是准周期（两个频率的加性叠加），相邻轨迹平行移动，Teacher 仍无独占信息。

**关键洞察**：破坏周期性 ≠ 创造混沌。外部正弦驱动把 periodicity 从 0.952 压到 0.489，FGL Δ 仍为 ~0%——这说明 FGL 需要的不是"非周期性"这一表象，而是 Teacher 相对 Student **真正的近未来信息优势**。至于把这种优势的来源归因为"反馈环复杂度差异"，是后续 τ 扫描（§9）提出的**假说**，此处不预设。

---

## 9. Mackey-Glass τ 扫描——动力学归因

> **口径声明**：本节及后续（§9–§12）沿用早期版本的"反馈环因果链"叙事（τ → 反馈环复杂度 → 信息不对称 → FGL Δ）。按现口径，其中把"反馈环数量 / 倍周期分岔"当作**上游因果自变量**的部分是**探索性假说而非定论**——三系统 Δ 排序与之相关，但缺乏因果干预实验，且"可预测性差异""信息不对称程度"等替代解释同样自洽（见 [`final_conclusions.md`](final_conclusions.md) §3）。本节中**稳健可重复**的部分是：τ 扫描实测数据、Lyapunov/periodicity 指标、几何检验排除"纯 L+H−1 共振"、以及"信息不对称是直接机制"。以下叙事保留以记录探索路径，其中的因果断言请理解为假说。

### 9.1 τ 是核心自由参数

Mackey-Glass 方程中，时延参数 τ 是控制动力学行为的唯一自由参数：

$$\frac{dx}{dt} = \beta \cdot \frac{x(t-\tau)}{1 + [x(t-\tau)]^n} - \gamma \cdot x(t)$$

τ 的含义是"过去的影响力需要多久才到达当下"。τ 每增大一点，系统的记忆就延长一点，反馈就复杂一点。τ 是自变量；Lyapunov 指数、periodicity、Baseline MSE ——这些都是因变量。

```
τ (时延) ──→ 反馈环复杂度 ──→ 可预测性 ──→ Teacher-Student 信息差 ──→ FGL Δ
              │                  │
              ├─ Lyapunov        ├─ Periodicity
              ├─ 分岔类型        ├─ 自相关衰减速度
              └─ 吸引子维度       └─ Baseline MSE
```

### 9.2 τ 的三个区域

| τ 范围 | 动力学 | 反馈环数量 | 信息结构 | FGL 预测 |
|:---:|------|:---:|------|:---:|
| < 5 | 稳定不动点 | 0 | 无振荡 | — |
| 10~12 | **极限环（周期 1）** | 1 | 短程 = 长程，Teacher 无信息优势 | ≈ 0% |
| ~13 | **倍周期分岔（周期 2）** | 2 | 短程 ≠ 长程，交替模式让 Student 困惑但 Teacher 轻松 | 巨大 |
| ≥ 17 | **混沌** | ∞ | 指数发散，Teacher 受影响小 | 正向 |

τ=13 是"甜品区"——两个竞争性的时间尺度（产生项自身的周期 + 延迟反馈）产生倍周期分岔。Teacher 只需 1 步判断"上升还是下降"（局部梯度，极易），Student 需从 8 步历史识别"这是周期 2 的哪个相位"（交替模式在窗口内多次翻转，困难）。信息可提取难度的不对称性达到峰值。

τ≥17 时，混沌发散同时惩罚 Teacher 和 Student，只是程度不同。收益稳定正向但不如 τ=13 大。

### 9.3 实验结果

新建 `mackey_glass/exp/tau_sweep.py`。扫描 τ ∈ {10, 13, 17, 23, 30}。

| τ | 分岔类型 | Lyapunov | Periodicity | Baseline MSE | Teacher MSE | **FGL Δ** |
|:---:|------|:---:|:---:|:---:|:---:|:---:|
| 10 | 周期 1 | +0.0009 | 0.996 | 0.37 | 0.30 | **+5.2%** |
| 13 | 周期 2 | +0.0009 | 0.993 | 3.96 | 0.55 | **+79.1%** |
| 17 | 混沌起始 | +0.0056 | 0.817 | 13.13 | 1.96 | **+11.4%** |
| 23 | 中等混沌 | +0.0099 | 0.427 | 12.77 | 1.43 | **+7.9%** |
| 30 | 高维混沌 | +0.0086 | 0.459 | 11.13 | 1.33 | **+8.8%** |

**五个 τ 值全部 FGL 正向。** 关键观察：

1. **τ=10（周期 1）→ FGL 几乎无效（+5.2%）**——Baseline 轻松学会简单振荡，Teacher 无信息优势
2. **τ=13（周期 2，非混沌！）→ FGL 爆发（+79.1%）**——最大收益不在混沌区，而在倍周期分岔
3. **τ≥17（混沌）→ FGL 持续正向（+8~11%）**——混沌发散创造了持久的信息不对称

### 9.4 动力学归因（假说性）

```
"周期性不行"是表象。
τ 决定 Teacher/Student 间信息提取难度的对称性（直接机制，稳健）；把其上游归因为"反馈环复杂度"则是假说。
τ 决定了 Teacher 和 Student 之间是否存在信息可提取难度的不对称性。

τ=10：   1 个反馈环 → 周期 1 → 无不对称 → FGL ≈ 0%
τ=13：   2 个反馈环 → 周期 2 → 不对称极大 → FGL +79%
τ≥17：   ∞ 个反馈环 → 混沌   → 不对称稳定 → FGL +8~11%
```

**"混沌"不是 FGL 生效的必要条件**——MG τ=13 的峰值出现在 Lyapunov≈0 的倍周期分岔区而非混沌区，这是稳健的事实观察。至于把该现象进一步归因为"反馈环 ≥ 2"，则是**待检验的假说**而非已证实的必要条件（见 [`final_conclusions.md`](final_conclusions.md) §3）。

### 9.5 CSTR 失败的一种解释（假说性）

CSTR 的 H₂/O₂ 燃烧动力学只有**一个负反馈环**：

```
反应 → 产 H₂O → H₂O 抑制反应 → H₂O 排空 → 反应重启
         ↑____________________________↓
                  唯一的反馈环
```

这个单环结构等价于 Mackey-Glass τ<10 或 τ≈10 的极限环区。调 U/K/流量改变的是振荡的振幅和频率，改变不了反馈环的数量。一个环就是一个环——它永远产生周期 1。

外部正弦驱动（方向六）本质上是**加性**的——两个频率叠加但不耦合。倍周期分岔需要的是**乘性**的非线性耦合——两个时间尺度在反馈环内部相互作用。这解释了为什么外部驱动降低了 periodicity 但仍无法使 FGL 获益。

**CSTR 不存在 τ 这个自由参数**——这是它和 Mackey-Glass 之间确实存在的结构性差异（事实）。至于把"CSTR 上 FGL 增益有限"完全归因于"单反馈环 / 无可调反馈延迟"，则属于**上游假说**：更直接、稳健的解释是 CSTR 周期 1 动力学下 Teacher 相对 Student 几乎不携带独占近未来信息（§2.4 机制），且存在 Baseline 地板效应。"反馈环"只是对这一现象的一种可能刻画。

---

## 10. Lorenz-63 独立验证

### 10.1 动机

Mackey-Glass τ 扫描确立了因果链：τ → 反馈环复杂度 → FGL Δ。但 MG 和 CSTR 是截然不同的系统（数学 DDE vs 化学 ODE），可能存在混淆变量。需要一个**第三种系统**做独立验证。

选择 Lorenz-63——最经典的混沌 ODE 系统，由 Rayleigh 数 ρ 控制分岔行为：

$$\begin{aligned} \frac{dx}{dt} &= \sigma(y - x) \\ \frac{dy}{dt} &= x(\rho - z) - y \\ \frac{dz}{dt} &= xy - \beta z \end{aligned}$$

| ρ | 动力学 | Lyapunov | MG 等价 | 预测 FGL Δ |
|:---:|------|:---:|------|:---:|
| 100 | 极限环（周期 1） | ≈ 0 | τ≈10, CSTR | ≈ 0% 或负 |
| 28 | 经典混沌 | ~1.1 | τ≈17 | > 0 |
| 60 | 强混沌 | ~1.6 | τ≈30 | > 0 且更稳定 |

新建 `lorenz/generate_lorenz.py`，用 `scipy.integrate.solve_ivp` 生成数据，取 x(t) 作为观测序列，直接复用 `fgl_cstr.py` 的 FGL 流水线。

### 10.2 结果

| ρ | Lyapunov | Periodicity | H=3 | H=5 | H=7 | H=9 | **Avg Δ** | 判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 100 | ≈ 0 | 0.996 | -67% | -114% | -48% | -17% | **-61.4%** | ❌ 周期 1 |
| 28 | ~1.1 | 0.098 | -12% | -111% | -2% | **+37%** | -22.0% | 混合 |
| 60 | ~1.6 | 0.077 | **+18%** | **+37%** | **+15%** | **+19%** | **+22.0%** | ✓ 全正向 |

### 10.3 分析

**ρ=100（周期 1）**：FGL 大幅为负（-61.4%）。比 CSTR 的 ~0% 更差——因为 Lorenz ρ=100 的极限环比 CSTR 更稳定、Teacher 的预测分布更尖锐（过度自信），KL 散度反而把 Student 拉向错误方向。

**ρ=28（经典混沌）**：混合结果。H=9 时 +36.6%，但 H=5 时 -111%。混沌强度不足以在所有 horizon 上创造稳定的信息不对称。

**ρ=60（强混沌）**：**四个 horizon 全部正向**，avg +22.0%，best +36.5%。强混沌创造了持续可靠的信息不对称——Teacher 离目标近（受发散影响小），Student 离目标远（受发散影响大）。

### 10.4 三条预测全部验证

| 预测 | 预期 | 实际 | 验证 |
|------|:---:|:---:|:---:|
| 周期 1 → FGL≈0% 或负 | Δ ≈ 0% | Δ = -61.4% | ✓ |
| 混沌 → FGL > 0 | Δ > 0 | H=9 时 +36.6% | ✓ (部分) |
| 强混沌 → FGL > 0 且稳定 | Δ > 0, 各 H 一致 | Δ avg +22.0%, 4/4 正 | ✓ |

---

## 11. 几何共振假说的证伪检验

### 11.1 竞争性假说

MG τ=13 的 FGL 峰值（+79.1%）被解释为倍周期分岔（反馈环=2）。但存在一个竞争性假说：这个"甜品区"的位置可能不是由 MG 方程自身的分岔结构决定的，而是由超参数几何关系 `L+H-1` 的数值巧合造成的。

推导依据：Teacher 使用 offset=H-1，两者共享同一预测目标 `y(t+L+H-1)`。该点的瞬时变化率由 MG 方程中的延迟项 `x(t+L+H-1-τ)` 驱动。当 `τ ≈ L+H-1` 时，该延迟值精确落入 Teacher 窗口而不落入 Student 窗口——形成纯几何的信息不对称，与 MG 的分岔结构无关。

当前默认配置 L=8, H=5, L+H-1=12，与实测甜品区 τ=13 仅差 1，高度可疑。

### 11.2 两个理论的可证伪预测

| 理论 | 预测 |
|------|------|
| ① 动力学分岔 | 甜品区始终在 τ≈13 附近，不随 L、H 变化 |
| ② 几何共振 | 甜品区位置 ≈ L+H-1，会随配置变化而"搬家" |

### 11.3 实验设计

新建 `mackey_glass/exp/tau_sweep_geometry.py`。4 个 (L,H) 配置，覆盖不同的 L+H-1 值：

| Config | L | H | L+H−1 |
|--------|---|:---:|:---:|
| config_A (原始) | 8 | 5 | 12 |
| config_B (长L) | 12 | 5 | 16 |
| config_C (长H) | 8 | 9 | 16 |
| config_D (短L+H) | 5 | 5 | 9 |

每个配置扫描 τ 值（几何临界点 ±5 + 原始参考点），每个 (L,H,τ) 跑 3 个随机种子。总计 162 次 FGL 训练。

### 11.4 结果

| Config | L+H−1 | **τ_peak** | **Δ_peak** | offset |
|--------|:---:|:---:|:---:|:---:|
| config_A | 12 | **13** | +36.7% | +1 |
| config_B | 16 | **15** | +49.3% | −1 |
| config_C | 16 | **14** | +47.4% | −2 |
| config_D | 9 | **12** | +42.5% | +3 |

### 11.5 判定

**结果支持理论①（动力学分岔），否定了几何共振假说。**

四个配置的 τ_peak 全部集中在 12~15 范围内（range=3），不随 L+H-1 的大幅变化（9→16）而显著移动。若几何共振假说成立，config_B 和 config_C（均为 L+H-1=16）的峰值应在 τ≈16~17，实际峰值在 14~15；config_D（L+H-1=9）峰值应在 τ≈9，实际峰值在 12，离原始 τ=13 仅差 1。

所有四个配置在 τ≈13 附近均观测到强 FGL 正向（+36~49%），确认甜品区是 MG 方程自身动力学结构的产物，与超参数 L、H 的选择无关——**几何检验稳健地排除了"纯 L+H−1 共振"这一竞争假说**。需注意：这次检验证明的是"峰值位置由动力学而非几何决定"；至于把该动力学进一步归因为"反馈环复杂度"，仍属上游假说（见 [`final_conclusions.md`](final_conclusions.md) §3），并未被本次检验单独证实。

---

## 12. 最终结论

### 12.1 全部实验汇总（三个系统，17 个实验）

| # | 系统 | 反馈环 | Lyapunov | FGL Δ | 判定 |
|---|------|:---:|:---:|:---:|------|
| 1~7 | CSTR（七个方向） | 1 | 0 | ≤ +0.4% | ❌ 全部无效 |
| 8 | CSTR 外部驱动 | 1+加性 | 0 | +0.4% | ❌ 准周期无帮助 |
| 9 | CSTR 双反应器 | 1+线性 | 0 | 负 | ❌ 线性耦合不够 |
| — | **MG τ=10** | 1 | ≈0 | +5.2% | 周期 1 基准 |
| — | **MG τ=13** | **2 (非线性)** | ≈0 | **+79.1%** | ★ 甜品区 |
| — | MG τ=17 | ∞ | +0.006 | +11.4% | ✓ 混沌正向 |
| — | MG τ=23 | ∞ | +0.010 | +7.9% | ✓ 混沌正向 |
| — | MG τ=30 | ∞ | +0.009 | +8.8% | ✓ 混沌正向 |
| — | **Lorenz ρ=100** | 1 | ≈0 | **-61.4%** | ❌ 周期 1, FGL 有害 |
| — | Lorenz ρ=28 | ∞ | ~1.1 | 混合 (+36.6%) | 弱混沌不稳定 |
| — | **Lorenz ρ=60** | ∞ | ~1.6 | **+22.0%** | ✓ 强混沌全正向 |

### 12.2 核心结论

**一种有解释力的假说**：FGL 的有效性与系统中非线性反馈环的数量相关（下图与三条归纳均为该假说的展开，**非已证实的因果规律**；直接、稳健的机制仍是 §2.4 的信息不对称）。

```
                        FGL 效果
                          ↑
           +79% ┤         ● MG τ=13 (周期2, 甜品区)
                │
           +22% ┤         ● Lorenz ρ=60 (强混沌)
           +11% ┤    ● MG τ=17 (混沌起始)
            +5% ┤ ● MG τ=10 (周期1)
                │
             0% ┤────● CSTR ────→ 反馈环数
                │    (周期1)
                │
                │         1 个环      2 个环      ∞ 个环
                │        (周期1)   (倍周期分岔)   (混沌)
```

- **1 个非线性反馈环（周期 1）**：Teacher 和 Student 面对同等简单的任务。FGL ≈ 0% 或负。CSTR、MG τ=10、Lorenz ρ=100 全部落在此区。
- **2 个非线性反馈环（倍周期分岔）**：短程≠长程。信息可提取难度不对称最大化。FGL 爆发式正向（MG τ=13, +79%）。
- **∞ 个非线性反馈环（混沌）**：指数发散创造持续不对称。FGL 稳定正向（MG τ≥17, Lorenz ρ=60），混沌越强越稳定。

**"混沌 vs 周期"是表象；反馈环数量是**一种可能的**上游自变量（假说）。** 稳健的表述是：CSTR 上 FGL 增益有限，源于其周期 1 动力学下 Teacher 几乎不携带独占近未来信息、且 Baseline 命中可预测性地板——"单反馈环 / 无可调反馈延迟"是对这一现象的一种可能的结构性刻画，尚需在更多系统上做因果干预检验，不应读为定论。

### 12.3 方法论意义

三个独立系统（CSTR 化学机理、Mackey-Glass DDE、Lorenz-63 ODE）的 FGL Δ 排序与"反馈环数量"一致，这是一种**有解释力的相关性**，但尚未构成已证实的因果规律。稳健、可用于预测的部分是**信息不对称机制**（以及 L/H 决定性、阈值效应、地板效应）：评估一个新数据集上 Teacher 相对 Student 的近未来信息优势，比计数反馈环更直接、也更可靠。

### 12.4 后续方向

1. 在更多经典系统上验证（Rössler、Chen 系统、Kuramoto-Sivashinsky），并对"反馈环数量"假说做严格的因果干预检验
2. MG τ=13 的 +79.1% 和 Lorenz ρ=60 的 +22.0% 之间是否存在一个统一的标度律（如 FGL Δ ∝ |Teacher MSE - Baseline MSE| / Baseline MSE）
3. 寻找真正有倍周期分岔的化学动力学机理（冷焰燃烧、Belousov-Zhabotinsky 反应），在化学系统中复现 MG τ=13 的甜品区

---

## 13. 共用代码修改注意事项

`mackey_glass/utils/utils.py` 的 `create_time_series_dataset` 新增了 `bin_edges` 参数（向后兼容）。该修改同样适用于 Mackey-Glass 实验（`base_exp.py`、`drift_exp.py`、`analysis.py`），虽然 MG 数据量大且 bin 差异极小，但建议同步修改以保持一致性。

---

## 14. 环境与运行

### 包管理

本项目使用 **`uv`** 进行包管理（见仓库根目录 `pyproject.toml` 和 `uv.lock`）。所有 Python 命令需通过 `uv run` 执行：

```bash
# 生成原始 CSTR 数据
uv run python cstr/generate.py

# 生成外部驱动 CSTR 数据
uv run python cstr/generate_forced.py --amplitude 0.5 --freq 0.05 --t_end 600

# 运行 FGL 实验（分类模式）
uv run python cstr/exp/fgl_cstr.py --horizon 5 --alpha 0.5 --epochs 30

# 运行 FGL 实验（回归模式）
uv run python cstr/exp/fgl_cstr_regression.py --horizon 5 --alpha 0.5 --epochs 30

# 运行 FGL 实验（LSTM 模式）
uv run python cstr/exp/fgl_cstr_lstm.py --horizon 5 --alpha 0.5 --epochs 30

# 运行 FGL 实验（Seq2Seq 周期预测）
uv run python cstr/exp/fgl_cstr_seq2seq.py --horizon 72 --teacher_steps 5 --alpha 0.5 --epochs 50

# 参数扫描（寻找非周期态）
uv run python cstr/param_sweep.py

# 数据分析
uv run python cstr/analyze_data.py

# 生成可视化
uv run python cstr/plot_data.py
uv run python cstr/plot_forced_comparison.py

# Mackey-Glass τ 扫描（混沌 vs FGL 对照实验）
cd /tmp && source .venv/bin/activate && python mackey_glass/exp/tau_sweep.py --epochs 50
```

### 依赖

```
Python 3.11, PyTorch 2.1.1
cantera >= 3.2.0    # CSTR 模拟
matplotlib           # 可视化
numpy, scipy, tqdm, scikit-learn
```
