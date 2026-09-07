# FGL 下一阶段实验设计与当前状态

本文档记录 2026-09 下一阶段的三个研究分支、当前实现状态与冒烟结果。

## 1. CSTR τ 家族跨例基础教师迁移

**研究问题**：单例训练的模型跨例泛化是否显著退化？能否通过多例预训练得到一个基础教师，并在目标 τ 例上微调后提升教师质量？

**实现**：
- 模块：`fgl_common/transfer.py`
- 入口：`uv run python cstr/run.py -e transfer_generalization`
- 输出：`cstr/results/transfer/transfer_teacher_lopo.csv`
- 方法：leave-one-example-out。对每个目标 τ，用其余 τ 数据集训练基础教师；比较：
  - `zero_shot`：冻结基础教师；
  - `finetuned`：目标例教师训练集微调；
  - `scratch`：目标例从零训练。

**冒烟结果**：`--quick` 已生成 12 行结果（4 个目标例 × 3 条件）。该结果只验证流程，不用于研究结论。

**下一步**：
1. 跑全量 13 目标例 × 3 seeds；
2. 汇总三条件均值与 bootstrap CI；
3. 若 `finetuned` 优于 `scratch`，再进行学生蒸馏对比。

## 2. 教师优化到学生优化的量化传递

**研究问题**：教师 MSE 每降低多少，能够按什么比例转化为学生 MSE 降低？

**实现**：
- 模块：`fgl_common/teacher_transfer.py`、`fgl_common/transfer_analysis.py`
- 入口：`uv run python cstr/run.py -e teacher_student_relation`
- 输出：`cstr/results/teacher_transfer/teacher_optimization_transfer.csv`
- 教师条件：`baseline`、`more_epochs`、`lower_lr`、`ensemble`、`regularized`；
- 学生侧超参数冻结，教师与学生逐 seed 配对；
- 核心字段：
  - `teacher_gain = baseline_teacher_mse - treated_teacher_mse`
  - `student_gain = baseline_teacher_student_mse - treated_teacher_student_mse`
  - `transfer_ratio = student_gain / teacher_gain`
  - `normalized_transfer = relative_student_gain / relative_teacher_gain`

**冒烟结果**：`--quick` 已生成 5 行结果（5 个条件 × 1 seed）。该结果只验证流程，不用于研究结论。

**下一步**：
1. 全量运行标准 CSTR `L=20,H=15` × 5 seeds；
2. 用 `summarize_transfer` 汇总各条件；
3. 拟合 `student_gain ~ teacher_gain`，并对 seed 做 bootstrap CI；
4. 再扩展到 τ 家族有效/失效格点，检查传递率是否随信息不对称 regime 改变。

## 3. 复杂 CSTR 多产物数据

**研究问题**：包含更多产物的复杂 CSTR 是否会改变 FGL 有效区？

**实现**：
- 生成器：`cstr/generate_complex.py`
- 多通道批处理：`fgl_common/multichannel.py`
- 生成命令：
  ```bash
  uv run python cstr/generate_complex.py --tau 50 --t-end 300 --dt 0.1 --k 6
  ```
- 输出：
  - NPZ：`cstr/data/complex/data_complex_cstr_multichannel_tau{tau}.npz`
  - 单通道 PKL：`cstr/data/complex/data_complex_cstr_{channel}_tau{tau}.pkl`
  - FGL 结果：`cstr/results/complex/complex_cstr_fgl.csv`
- 入口：`uv run python cstr/run.py -e complex_multichannel`

**冒烟结果**：当前没有已生成的复杂通道数据，`--quick` 写入 0 行，符合“无数据不训练”的保护逻辑。

**下一步**：
1. 在 Cantera 环境生成 τ=50/70/90/100 的多物种数据；
2. 检查 `cstr/data/complex/INDEX.md` 与通道元数据；
3. 每通道独立跑 FGL，先比较温度、H2O 和高方差物种；
4. 汇总通道动力学指标与 FGL 增益。

## 验证记录

- 当前测试基线：`uv run pytest tests -q` → 58 passed。
- `uv run python cstr/run.py --list` 已显示三个新实验，默认均关闭。
- 三个新实验的 `--quick` 路径均已运行；迁移与传递实验生成非空 CSV，复杂多通道在无数据时输出 0 行。

## 4. 2026-09-07 实验测试结果

以下为当前第一轮全量/代表性实验。除复杂 CSTR 外，其余均为完整计划中的首轮样本量。

### 4.1 方向 1：跨例基础教师迁移

**设置**：13 个延迟 CSTR τ 数据集，leave-one-example-out，3 seeds，共 39 次目标评估 × 3 条件。教师预训练 3 epochs，目标微调 3 epochs，scratch 3 epochs。

| τ | zero_shot MSE | finetuned MSE | scratch MSE | finetuned − zero_shot |
|---:|---:|---:|---:|---:|
| 30 | 41.48 | 26.07 | 153.78 | −15.41 |
| 35 | 24.71 | 20.72 | 249.96 | −3.99 |
| 40 | 17.67 | 17.35 | 174.43 | −0.33 |
| 45 | 20.06 | 13.91 | 162.22 | −6.15 |
| 50 | 18.20 | 14.64 | 153.78 | −3.56 |
| 55 | 20.20 | 17.22 | 167.05 | −2.98 |
| 60 | 20.97 | 17.08 | 220.40 | −3.90 |
| 65 | 21.42 | 17.47 | 215.50 | −3.96 |
| 70 | 24.48 | 21.52 | 233.42 | −2.95 |
| 80 | 28.44 | 28.85 | 294.85 | +0.40 |
| 100 | 32.06 | 29.05 | 223.83 | −3.01 |
| 120 | 33.39 | 27.67 | 227.02 | −5.72 |
| 150 | 32.43 | 23.48 | 298.32 | −8.95 |

**总体均值**：

- zero_shot：25.81 ± 7.23
- finetuned：21.16 ± 5.39
- scratch：213.43 ± 49.43

**统计检验**：

- finetuned vs zero_shot，13 个 τ 配对：Wilcoxon p=0.00073；paired t p=0.00127。
- scratch vs finetuned：Wilcoxon p=0.00024。
- 13 个 τ 中 12 个 finetuned 优于 zero_shot；唯一例外为 τ=80，且差距很小。

**初步结论**：

1. 单例/冻结跨例模型泛化确实很差：zero_shot 平均比目标例微调差约 4.65 MSE（约 22% 相对差距）。
2. 多例预训练教师并非“直接可用”，但目标例少量微调非常有效，说明存在可迁移的公共表征/动态结构。
3. 从零教师在相同 3-epoch 预算下完全失败，说明预训练初始化带来的优化与表征收益显著。
4. 当前结论限于教师任务 H=1、短微调预算；尚不能直接外推到 H=15 学生任务。

### 4.2 方向 2：教师优化到学生优化的传递关系

**设置**：标准 CSTR `data_h2o.pkl`，`L=20,H=15`，5 seeds，每个 seed 下教师条件与学生严格配对。学生超参冻结。

**条件汇总（跨 5 seeds 聚合）**：

| 条件 | 教师 MSE 变化 | 学生 MSE 变化 | 聚合传递率 |
|---|---:|---:|---:|
| baseline | 0 | 0 | — |
| more_epochs | 教师 MSE 降低 0.445 | 学生 MSE 改善 2.324 | 5.22 |
| lower_lr | 教师 MSE 恶化 126.99 | 学生 MSE 恶化 56.96 | 0.45 |
| ensemble | 教师 MSE 降低 0.118 | 学生 MSE 恶化 7.344 | −62.21 |
| regularized | 教师 MSE 降低 0.016 | 学生 MSE 恶化 0.001 | −0.07 |

**逐条件解释**：

- `lower_lr` 是唯一教师显著恶化的条件；学生也相应恶化，配对传递率约 0.45。这给出一个负向但很有价值的信号：教师变差会强烈传导到学生。
- `more_epochs` 教师只小幅变好（0.445），但学生平均改善 2.324。逐 seed 比率波动很大，不能解释为稳定“1 单位教师改善换 5 单位学生改善”。
- `ensemble` 教师 MSE 几乎不变，但学生 MSE 明显变差。说明教师单点 MSE 不是蒸馏收益的充分指标，教师输出分布/输入-输出对齐结构同样重要。
- `regularized` 教师变化过小，学生差异接近零，不能用于传递率推断。

**相关性**：

- `student_gain ~ teacher_gain`：Pearson r=0.880，p=3.2e-7（n=20 个非 baseline 条件-seed 配对）。
- Spearman r=0.361，p=0.117，说明线性相关主要由 `lower_lr` 造成的极端教师恶化驱动。
- 教师变差的 6 个配对中，学生平均恶化 47.47 MSE，聚合比率约 0.50。
- 教师变好的 13 个配对中，学生平均反而恶化 1.93 MSE，且比率极不稳定。

**初步结论**：

1. 教师 MSE 与学生 MSE 存在强总体线性关系，主要来自“教师变差 → 学生变差”。
2. 目前证据不支持“教师每改善 1 单位，学生稳定改善某一固定比例”。教师小幅改善时，学生响应高度非线性和条件依赖。
3. 教师 MSE 不是充分指标：ensemble 的教师 MSE 基本不变，但学生明显变差；需要纳入教师输出熵、teacher-student 分布 KL、逐样本误差结构等指标。
4. 对用户原始问题的当前回答：教师恶化的传递率目前约 0.45–0.50；教师改善的传递率尚不稳定，甚至可能为负。

### 4.3 方向 3：复杂 CSTR 多产物 τ=50

**设置**：使用多物种 H2O2 CSTR τ=50 数据，`L=20,H=15`，3 seeds，每通道独立 FGL。自动保留通道：temperature、H2O、H2、O2、OH、O。

| 通道 | baseline MSE 均值 | student MSE 均值 | FGL Δ 均值 | 状态 |
|---|---:|---:|---:|---|
| H2O | 140.13 | 112.85 | +19.56% | 正向 |
| H2 | 148.49 | 143.32 | +3.36% | 微弱正向，1/3 seed 负 |
| O2 | 149.20 | 144.65 | +2.92% | 微弱正向，1/3 seed 负 |
| OH | 4.351 | 4.351 | 0% | baseline 已到离散地板 |
| O | 2.682 | 2.682 | 0% | baseline 已到离散地板 |
| temperature | 8.219 | 8.219 | 0% | baseline 已到离散地板 |

**初步结论**：

1. 复杂多物种数据已经可以完整生成并进入现有 FGL 流程。
2. 同一系统内不同产物通道的 FGL 收益差异很大：H2O 有明显正向增益，H2/O2 增益微弱且不稳定，OH/O/temperature 处于 baseline 地板，无蒸馏空间。
3. 这与既有“地板效应”一致，但新增了更细的化学通道视角：FGL 是否有效不仅取决于系统级混沌/周期，还取决于目标变量的可用预测结构和 baseline 是否已经饱和。
4. τ=50 只是 4 个计划 τ 中的第 1 个；现在还不能给出“复杂产物改变 FGL 有效区”的完整结论。

## 5. 当前总体判断与下一步

### 当前总体判断

- **基础教师方向最有希望的信号**：多例预训练 + 少量目标微调显著优于冻结迁移和同预算从零训练。
- **师生传递关系的核心发现**：教师变差的传导很清晰，教师小幅改善的传导不稳定；应从“MSE 单指标传递率”转向“教师质量多维表征 + 分 regime 传递函数”。
- **复杂 CSTR 方向的机制线索**：FGL 增益具有通道依赖性；高动态、未饱和通道（H2O）可能获得较大增益，饱和通道没有蒸馏空间。

### 下一步优先级

1. **学生蒸馏迁移验证**：用方向 1 中胜出的 `finetuned` 基础教师蒸馏 H=15 学生，与 scratch 教师学生、无蒸馏 baseline 对比。
2. **扩展方向 2 的教师质量指标**：增加教师熵、teacher-student KL、逐样本误差相关性和 bootstrap CI；把“教师小幅改善”与“教师大幅恶化”分开建模。
3. **补齐复杂 CSTR τ 扫描**：生成 τ=70/90/100，先确认每个通道是否存在非饱和 baseline；只对非饱和通道跑多 seed FGL。
4. **避免比率误读**：后续报告以配对增益和 CI 为主，传递率只在教师变化幅度超过噪声阈值时计算。
