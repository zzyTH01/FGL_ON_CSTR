# Future-Guided Learning (FGL) 代码审查、Bug 溯源解析与重跑验收全记录（2026-10-16）

> **文档定位**：本报告完整记录了对 `FutureGuidedLearning` 仓库的两轮代码审查（初审 9 项缺陷 + 二次验收发现的 5 处残留隐患）、数学与物理层面的 Bug 机理推导、代码修复方案、回归测试建设（`120 passed` + `1 slow passed`），以及修复前后 GPU 实验结果的定量对比。

---

## 目录

1. [背景：为什么对 2026-10-14 实验发起全面审查](#一-背景为什么对-2026-10-14-实验发起全面审查)
2. [致命级 Bug 深度解析（数学机理与日志铁证）](#二-致命级-bug-深度解析数学机理与日志铁证)
   - [2.1 张量广播导致早停机制反转（假阳性 `+96.9%` 与假阴性 `-8281%` 的共同元凶）](#21-张量广播导致早停机制反转假阳性-969-与假阴性--8281-的共同元凶)
   - [2.2 测试集选模泄露（Test-set Selection Leakage）](#22-测试集选模泄露test-set-selection-leakage)
   - [2.3 复杂 CSTR 生成器的控制律失效、CVODES 体积静默忽略与反应器熄火](#23-复杂-cstr-生成器的控制律失效cvodes-体积静默忽略与反应器熄火)
   - [2.4 Teacher Sweep 中训练集充当验证集早停](#24-teacher-sweep-中训练集充当验证集早停)
3. [核心共享库（`fgl_common/`）的方法学缺陷与修复](#三-核心共享库fgl_common的方法学缺陷与修复)
   - [3.1 `RNN` / `LSTMModel` 单时间步退化问题与 `TemporalRNN` 增补](#31-rnn--lstmmodel-单时间步退化问题与-temporalrnn-增补)
   - [3.2 `continuous_fgl.py` 缺失物理单位反标准化与 Teacher 端广播](#32-continuous_fglpy-缺失物理单位反标准化与-teacher-端广播)
   - [3.3 `data.py` 先切分后 `offset` 导致师生时间窗错位及全量分箱泄露](#33-datapy-先切分后-offset-导致师生时间窗错位及全量分箱泄露)
   - [3.4 `transfer.py` 算力预算失衡与跨源分箱语义漂移](#34-transferpy-算力预算失衡与跨源分箱语义漂移)
   - [3.5 可视化诊断脚本中的测试集选模](#35-可视化诊断脚本中的测试集选模)
4. [两阶段修复与二次验收过程复盘](#四-两阶段修复与二次验收过程复盘)
5. [修复前后核心实验结果对比与科学结论反转](#五-修复前后核心实验结果对比与科学结论反转)
6. [回归测试防线建设清单](#六-回归测试防线建设清单)

---

## 一、 背景：为什么对 2026-10-14 实验发起全面审查

在 `conclusion/1014.md` 中，实验报告声称：
- 强 Teacher FGL 在 8 个数据族、10 个代表数据集上实现“**100% 全部正向蒸馏增益（`+64.9% ~ +96.9%`）**”；
- 在 `data_h2o.pkl` 上，同配置 PatchTST 的 `baseline_mse` 为 `3.70e-2`，而蒸馏后的 `student_mse` 降至 `1.15e-3`（改善 `+96.9%`）。

然而，交叉核对历史基线（`cstr_external_baselines.csv` 与 `cstr_strong_teacher_distill.csv`）即可发现严重反常：
- 同一 `data_h2o.pkl (L=20, H=12)` 任务上，外部与内部正常训练的 PatchTST baseline 误差均为 **`1.24e-3 ~ 1.98e-3`**；
- 为什么 `cstr_foundation_teacher.csv` 中的同架构 `baseline_mse` 会突然劣化 **30 倍** 变成 `3.70e-2`（接近目标序列本身的方差 `~4.9e-2`），而 `student_mse`（`1.15e-3`）却处于正常的 `1e-3` 量级？
- 同样，在 `cstr_student_sweep.csv` 中，为什么 `teacher_mse` 全表恒定为 `0.049204988`（恰好等于测试集方差），且 `patchtst` 学生误差全部卡死在 `4.70e-2`？

带着这一反常线索审查代码与运行日志，定位出了横跨训练脚本、数据生成器与核心库的 9 项缺陷。

---

## 二、 致命级 Bug 深度解析（数学机理与日志铁证）

### 2.1 张量广播导致早停机制反转（假阳性 `+96.9%` 与假阴性 `-8281%` 的共同元凶）

#### （1）代码根因
`fgl_common/baselines.py` 中的 `PatchTST.forward()` 输出形状为二维张量 `(batch, 1)`，而训练脚本构造的目标张量 `y` 形状为一维张量 `(batch,)`。
当调用 `nn.MSELoss()(pred, y)` 且未对 `pred` 执行 `.squeeze(-1)` 时，PyTorch 会触发 `(N, 1)` 与 `(N,)` 的二维广播，生成 `(N, N)` 矩阵，并在控制台抛出警告（见 `foundation_teacher_v2_20261014_134425.log` 与 `student_sweep_20261016_223139_valfull.log`）：
```text
UserWarning: Using a target size (torch.Size([594])) that is different to the input size (torch.Size([594, 1])). This will likely lead to incorrect results due to broadcasting.
```

#### （2）数学机理推导：为什么广播会让模型退化为 Epoch-1 常数预测？
设样本数为 $N$，模型预测为 $\hat{y} \in \mathbb{R}^{N \times 1}$，真实标签为 $y \in \mathbb{R}^{1 \times N}$。广播后的损失函数为：
$$\mathcal{L}_{\text{broadcast}}(\hat{y}, y) = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} (\hat{y}_i - y_j)^2 = \underbrace{\text{Var}(y)}_{\text{常数}} + \text{Var}(\hat{y}) + (\bar{\hat{y}} - \bar{y})^2$$

这一恒等式带来了灾难性的后果：
1. **如果在训练损失中发生广播**（如 `run_student_sweep.py` 中训练 Teacher 和 `patchtst` 学生）：
   对 $\mathcal{L}_{\text{broadcast}}$ 求梯度并最小化，最优解是**令 $\text{Var}(\hat{y}) = 0$ 且 $\bar{\hat{y}} = \bar{y}$**！即梯度下降主动抹平所有时序波动，强迫网络对任何输入都输出全局均值常数 $\bar{y}$。因此测试集物理 MSE 精确等于测试集方差 $\text{Var}(y_{\text{test}}) \approx 0.0492$！
2. **如果仅在验证损失中发生广播**（如 `run_foundation_teacher.py` 的 `_train` 函数，训练步写了 `.squeeze(-1)` 但验证步第 138 行漏写）：
   在标准化空间中，$\text{Var}(y_{\text{val}}) \approx 1, \bar{y}_{\text{val}} \approx 0$。
   - **Epoch 1 刚初始化时**：模型尚未学习，输出接近常数 $0$（$\text{Var}(\hat{y}_{\text{val}}) \approx 0$），此时 $\mathcal{L}_{\text{broadcast}} \approx 1.0 + 0 = 1.0$。
   - **随着训练正常收敛**：模型输出 $\hat{y}_{\text{val}}$ 精准拟合真实波形 $y_{\text{val}}$，其方差 $\text{Var}(\hat{y}_{\text{val}}) \to 1.0$，此时广播验证损失反而单调上升至 $\mathcal{L}_{\text{broadcast}} \approx 1.0 + 1.0 = 2.0$！
   - `EarlyStopper` 监测到验证损失从 Epoch 1 起“单调恶化”，在第 16 轮触发早停，并将 **Epoch 1 几乎未学习的平坦模型**恢复为 `best_state`！

#### （3）对不同实验臂的不对称冲击
- **在 `run_foundation_teacher.py` 中（制造假阳性）**：
  `scratch_teacher`、`pretrain_foundation`、`finetune_teacher` 和 **`baseline`** 全部调用 `_train()`（验证步漏写 `.squeeze(-1)`），因此全部退化为 Epoch-1 欠拟合模型（`baseline_mse = 3.70e-2`）；唯独 `student` 调用的是 `_distill()`（验证步写了对的 `.squeeze(-1)`），在 $\alpha=0.9$（90% 真值监督）下正常收敛至 `1.15e-3`。用正常训练的 `student` 对比被 Epoch-1 锁死的 `baseline`，制造了 **`+64.9% ~ +96.9%` 跨 10 个数据集全部正向的虚假繁荣**。
- **在 `run_student_sweep.py` 中（制造假阴性）**：
  初审修复了 `_train` 和 `_eval`，但二次验收发现 `_train_with_distill` 第 234 行 `out = model(xs.to(device))` 仍漏掉 `_forward_1d`。当学生为 `patchtst`（输出 `(64, 1)`）时，训练步广播迫使 `patchtst` 学生输出常数（`student_mse = 4.70e-2`），而 `baseline` 已正常收敛（`9.31e-4`），从而产生 **`-3422% ~ -8281%` 的虚假负迁移**，掩盖了 `patchtst/medium` 实际上是最佳学生架构的事实！

---

### 2.2 测试集选模泄露（Test-set Selection Leakage）

审查发现多处代码使用测试集指标进行超参数、架构或迭代轮次选择：
1. `cstr/run_foundation_teacher.py`：用测试集 MSE（`t_ft_mse < t_scratch_mse`）挑选下游蒸馏使用的 Teacher。
2. `cstr/run_teacher_sweep.py`：Stage 0 用测试集 `teacher_mse_mean` 选择最佳 `(L_t, d_model, nhead, lr)`。
3. `cstr/run_student_sweep.py`：`_best_from_csv()` 用测试集 `student_mse` 选择进入 `L_s` 扫描的学生架构；且 `--phase all --fresh` 时第二次调用 `_save_rows` 因 `args.fresh` 未重置而覆盖清空了 `arch` 阶段写入的 CSV。
4. `cstr/plot_pretrain_finetune_eiter.py` 与 `cstr/plot_method_comparison.py`：虽注释写 `best-by-val`，但 `snap` 回调只存了 `mse_test` 并按 `mse_test` 最小挑选迭代轮次。

**修复方案**：上述全部脚本统一改为在验证集上计算并记录 `val_mse`，严格依据验证集 MSE 冻结模型、超参与架构选择，测试集仅在最终评估时读取一次。

---

### 2.3 复杂 CSTR 生成器的控制律失效、CVODES 体积静默忽略与反应器熄火

对 `cstr/generate_complex.py` 的两轮审查发现了 4 个物理仿真与代码逻辑错误：

1. **延迟信号未接入 EMA 滤波器（初审发现）**：
   原代码写为 `dh_smooth = filter_beta * h2o + (1 - filter_beta) * dh_smooth`，滤波的是当前瞬时浓度 `h2o` 而非延迟缓冲区首元素 `delayed_val = buf[0]`；且 `simulate_multi_delay` 的多个延迟通道共用同一个标量 `dh_smooth`。
2. **延迟时间步单位放大 10 倍（初审发现）**：
   `generate_delayed_stable.py` 中 `tau_delay=50` 表示 50 个采样步（5.0 秒），而 `generate_complex.py` 写成了 `int(tau / dt)`（500 个采样步 = 50.0 秒）。
3. **缓冲区移位与读取顺序颠倒（二次验收发现）**：
   初审修改后，`simulate_forced_delay`、`simulate_param_mod`、`simulate_noisy_control` 写成了先 `delay_buf[:-1] = delay_buf[1:]` 覆盖丢弃 `delay_buf[0]`、再使用上一轮循环遗留的 `delayed_val`、最后才赋值 `delayed_val = delay_buf[0]`。二次验收已修正为统一先读 `delayed_val = delay_buf[0]` 再移位。
4. **`simulate_param_mod` 中 Cantera 体积赋值静默失效与反应器熄火（二次验收发现，解释了 `-3553.8%` 异常值）**：
   - **现象**：第一轮重跑 `conclusion/1016.md` 时，`param_mod` 数据集出现 `teacher_mse = 2.9e-10`、`student improvement = -3553.8%` 的极端异常。
   - **根因查明**：检查生成的 `.pkl` 数组发现，`V8.0e-06-1.4e-05_T200.pkl` 与 `V6.0e-06-1.6e-05_T300.pkl` 两个文件**逐字节完全相同（`max_abs_diff = 0.0`）**，且从 `step = 2000` 到 `6000`（涵盖全部验证集 `3600..4800` 和测试集 `4800..6000`）序列恒为死值 `0.018582`（标准差 `5.9e-14`）！
     这是因为：(a) 在 Cantera 中，`ReactorNet` 初始化后在循环内直接修改 `cstr.volume = ...` 而不驱动移动壁面会被 CVODES 积分器完全忽略（且 `t=0` 时两配置的 `vol_mid` 均为 `11e-6`）；(b) 正反馈幅值设为 `0.9` 且未加 `feedback_limit=0.2` 限幅，导致 `step ≈ 1000` 时进气流量剧烈波动将火焰吹熄。
   - **修复方案**：通过 Cantera 物理壁面移动速度 `wall.velocity = (vol_amp / wall_area) * (2 * pi / mod_period) * cos(2 * pi * t / mod_period)` 在 CVODES 内部连续积分体积变化，并加入 `feedback_limit=0.2` 防熄火限幅。修复后 3 个 `param_mod` 数据集轨迹截然不同（`max_diff = 0.965`），全程保持强非周期振荡（`std_val ≈ 0.22, std_te ≈ 0.22, periodicity ≈ 0.23–0.25`）。

---

### 2.4 Teacher Sweep 中训练集充当验证集早停

- **定位**：`cstr/run_teacher_sweep.py:236`
- **根因**：Stage 1（`_train_baseline_and_student`）在训练 Teacher 时未构造 `tx_val`，直接传入了 `_train_regressor(teacher, tx_tr, y_tr, tx_tr, y_tr, ...)`，导致 Teacher 早停完全基于训练损失。
- **修复**：构造 `tx_val, _ = _to_tensor(TX[n_tr:n_tr+n_val], ...)` 并传入 `_train_regressor`。

---

## 三、 核心共享库（`fgl_common/`）的方法学缺陷与修复

### 3.1 `RNN` / `LSTMModel` 单时间步退化问题与 `TemporalRNN` 增补
- **根因**：历史 `RNN` / `LSTMModel` 定义为 `nn.RNN(input_size=L, hidden_size=128, batch_first=True)`，输入张量形状为 `(batch, 1, L)`。在 `batch_first=True` 下这代表 `seq_len = 1, input_size = L`——初始隐状态 $h_0 = 0$ 时 $W_{hh} h_0 \equiv 0$，循环层在数学上完全退化为单步前馈线性层（MLP），没有任何跨时间步递归。
- **修复**：保留 `RNN` / `LSTMModel` 以维持历史实验可复现性并补充注释说明；在 `fgl_common/models.py` 中新增真正的时序递归模型 `TemporalRNN`（`input_size=1`，在 `forward` 中执行 `x.transpose(1, 2)` 沿 $L$ 个时间步递归）。

### 3.2 `continuous_fgl.py` 缺失物理单位反标准化与 Teacher 端广播
- **根因**：`_cont_mse` 计算的是标准化后的 z-score MSE，漏乘了 `y_std ** 2`（比物理 MSE 偏大 $\approx 20$ 倍）；同时 `_train_cont` 中模型输出与 Teacher 输出未调用 `_forward_1d`（初审修复了学生端，二次验收补齐了第 100 行 `tout = _forward_1d(teacher, tx)`）。

### 3.3 `data.py` 先切分后 `offset` 导致师生时间窗错位及全量分箱泄露
- **根因**：
  1. `create_time_series_dataset` 先按各自窗口总数切分 `60/20/20` 再在每个 split 内切片 `tup[offset:]`，导致 Teacher 的训练/验证/测试集比 Student 少 $H-1$ 个样本，且验证集和测试集的起始时间步相对 Student 发生 $0.6(H-1)$ 与 $0.8(H-1)$ 的错位。
  2. `compute_shared_bin_edges` 与 `run_seq2seq` 在整条序列（含验证/测试集）上取 `min/max` 划分分箱边界。
- **修复**：`create_time_series_dataset` 默认启用 `align_before_split=True`（切分前先对总窗口执行 `X = X[offset:], y = y[offset:]`），同步移除 `training.py` 中的 `j - (H - 1)` 补偿；`compute_shared_bin_edges` 与 `_shared_seq_bin_edges` 改为严格仅使用训练段（前 60%）计算 `min/max`。

### 3.4 `transfer.py` 算力预算失衡与跨源分箱语义漂移
- **根因**：跨 $\tau$ 迁移实验中，`pretrained` 教师累计训练 36 轮 + 微调 3 轮，而对照 `scratch` 教师在 `lr=1e-4` 下仅训练 3 轮（严重欠拟合，MSE 高达 `213.43`）；且各数据集独立计算 `bin_edges` 导致同一下标的 bin 对应不同物理浓度区间。
- **修复**：新增 `compute_global_bin_edges` 从所有源数据集训练段联合计算共享分箱边界，并在 `evaluate_teacher_transfer` 中增加收敛级对照 `scratch_converged`（30 epochs + `EarlyStopper`）。

### 3.5 可视化诊断脚本中的测试集选模
- **修复**：`cstr/plot_pretrain_finetune_eiter.py` 与 `cstr/plot_method_comparison.py` 均已在快照元组中同时记录 `(round_idx, model, float(mse_val), float(mse_test))`，并按验证集 MSE（`t[2]`）最小选出代表模型。

---

## 四、 两阶段修复与二次验收过程复盘

| 阶段 | 执行动作 | 发现与解决的关键问题 |
|---|---|---|
| **阶段 1：初审与首轮 GPU 重跑** | 按初审报告修改代码并在远程 `gpu` 重跑 4 组实验 | 解决了 `run_foundation_teacher.py` 验证广播与 `run_teacher_sweep.py` 验证集泄露，推翻了 `1014.md` 的 `+96.9%` 假阳性 |
| **阶段 2：严格验收审查** | 逐行核对 diff、运行日志与重生成 `.pkl` 数据分布 | 揪出 3 处关键残留：(1) `run_student_sweep.py:234` 漏掉 `_forward_1d` 导致 `patchtst` 学生仍广播崩坏（`-8281%`）且 `--fresh` 覆盖导致 CSV 验证列为空；(2) `generate_complex.py` 中 `delayed_val` 顺序颠倒且 `param_mod` 体积未进 CVODES + 反应器熄火（导致 `-3553.8%` 异常）；(3) `continuous_fgl.py:100` 教师端漏掉 `_forward_1d` |
| **阶段 3：二次修复与 GPU 补跑定稿** | 修复全部残留 Bug、重生成 15 个复杂数据集、在 `gpu` 上补跑 `student_sweep`、`param_mod`、10-seed 锚点与跨族 `L_t=60` 对照 | `patchtst/medium` 学生恢复真实性能（3/3 正向，`+39.4% ~ +43.5%`）并展现清晰的 $L_s$ 信息差衰减律；`param_mod` 恢复为 `+12.4%` 正增益；Foundation Teacher 实现 **30/30（100%）** 教师胜率（`+54.5%`） |

---

## 五、 修复前后核心实验结果对比与科学结论反转

| 实验模块 | 修复前（`1014.md`，含广播/泄露 Bug） | 初审重跑（含残留 Bug） | 最终完全修复后（`1016.md` 定稿） | 核心科学结论修正 |
|---|---|---|---|---|
| **Teacher Sweep (`L_t=20→60`)** | 按测试集选出 `L_t=60`（`9.56e-5`） | 按验证集选出 `L_t=60, d=64, h=8`（val `8.85e-5`, test `9.56e-5`） | 同左（稳健成立） | **信息超集条件（$L_t \ge L_s+H-1$）真实稳健**，$L_t=60$ 比 $L_t=31$ 误差再降 10 倍 |
| **Strong Teacher + `patchtst/small`** | 误与崩坏 baseline 对比声称 `+85%` | `α=0.9` 均值 `+16.71%`（误写为 2/3 正向） | `α=0.9` 均值 **`+16.71%`**（实为 **1/3 seed 正向**） | 小容量学生（`d=32`）存在表征瓶颈，单靠弱蒸馏不能保证逐 seed 稳定正向 |
| **Student Sweep（最优学生架构）** | Teacher 广播崩坏在 `0.0492`，`patchtst` 学生崩坏 | `patchtst` 学生仍因第 234 行广播崩坏在 `4.70e-2`（`-4913%`），误选 `transformer/small`（`-2.8%`） | 验证集选出 **`patchtst/medium` (`d=64, h=8`)**，`α=0.5/0.7/0.9` **全部 3/3 正向**（`+17.1% / +43.5% / +39.4%`）；$L_s=20\to 40$ 增益从 **`+39.4% \to 0.0%`** 单调归零；10-seed 锚点验证 $r(B, B-S)=+0.833 (p=0.0028)$ | **师生容量匹配**是连续 PatchTST-FGL 生效的关键前提；连续模型同样遵循**信息差衰减律**与**地板效应** |
| **Foundation Teacher（教师质量）** | 声称微调比从头训练好 `+22.2%`（22/30 胜），delayed 族为盲区 | 验证选模下 28/30 胜（`+50.13%`），唯独 `param_mod` 异常 | 修复 `param_mod` 熄火后，**30/30（100%）全面胜出**，验证/测试 MSE 平均降低 **`+54.51% / +54.58%`** | 逐源 train-only 标准化后，跨工况预训练在**包括强混沌 delayed 在内的所有数据族上 100% 提升 Teacher** |
| **Foundation Teacher（学生蒸馏）** | 声称 10 个数据集 `+64.9% ~ +96.9%` 全部正向（实为 baseline 广播崩坏 30 倍所致） | 受 `param_mod` 熄火影响均值 `-118.1%`（中位数 `-0.6%`） | 在 `L_t=31 + patchtst/small` 下配对均值 **`+0.58%`**（6/10 数据集为正，`param_mod` `+12.4%`）；跨族 `L_t=60` 对比证明当 `T < B` 时均值 `+1.95%`（13/20 正向），而 `forced/dual/param_mod` 上长窗口需预训练防过拟合 | 消除了虚假 `+96.9%` 泡沫，揭示了“**长窗口过拟合 vs 预训练正则化**”以及“**教师增益向学生传导的容量/视野条件**” |

---

## 六、 回归测试防线建设清单

所有修复均已固化至 `tests/regression/test_debug1016.py`（配合原有测试共 **`120 passed` + `1 slow passed`**）：

1. `test_offset_is_applied_before_splitting`：验证 `create_time_series_dataset` 在切分前应用 `offset=H-1`，师生测试集长度与目标完全相等。
2. `test_shared_bin_edges_use_training_targets_only` & `test_seq2seq_shared_bin_edges_are_train_only`：注入测试段极端异常值 `10,000.0`，验证分箱边界不受验证/测试段污染。
3. `test_continuous_mse_is_physical_and_prevents_broadcast` & `test_continuous_iterative_fgl_prevents_teacher_broadcast`：使用返回 `(batch, 1)` 的 `OneD` 模型验证物理单位反标准化与端到端零广播警告。
4. `test_student_sweep_squeezes_only_extra_prediction_dimension` & `test_student_sweep_selects_architecture_by_validation`：验证 `_train` 与 `_train_with_distill` 均调用 `_forward_1d`、`_save_rows` 重置 `args.fresh = False`、且按 `student_val_mse` 选模。
5. `test_tau_changes_delayed_dynamics_after_buffer_warmup` & `test_param_mod_volume_parameters_change_trajectory`：验证 `generate_complex.py` 中延迟参数 $\tau$ 与体积调制参数 `(volume_start, volume_end, mod_period)` 真实改变仿真轨迹。
6. `test_transfer_uses_shared_edges_and_converged_control`：验证跨 $\tau$ 教师迁移包含共享分箱与 `scratch_converged` 收敛对照组。
