# FGL 下一阶段研究设计：跨例泛化、师生优化关系、复杂 CSTR

## 1. 研究目标与已确认决策

本阶段在现有 FGL 框架上新增三个研究方向：

1. **方向 1：模型泛化与基础教师**
   - 研究问题：单例模型跨例迁移有多差？能否得到一个预训练一次的“基础教师”，在新 CSTR τ 例上通过微调或蒸馏提升学生？
   - 已确认范围：**CSTR τ 家族内部迁移**，不作为第一版跨 CSTR / MG / Lorenz 的通用模型。
   - 已确认策略：**多例预训练 + 目标例微调**；必须保留冻结教师和从零训练教师对照。
   - 第一版先评估教师迁移；教师选定后再做学生蒸馏对比。
2. **方向 2：教师优化量到学生优化量的可量化传递关系**
   - 核心问题：教师 MSE 或教师表征质量的下降，有多少比例反映在学生 MSE 上？
   - 主指标定义：
     - 教师改进量：`teacher_gain = teacher_baseline_mse - teacher_treated_mse`；
     - 学生改进量：`student_gain = student_baseline_mse - student_treated_mse`；
     - 传递率：`transfer_ratio = student_gain / teacher_gain`；
     - 归一化传递率：`normalized_transfer = (student_gain / student_baseline_mse) / (teacher_gain / teacher_baseline_mse)`。
   - 需要报告相关性和回归，但必须区分相关性与因果；至少包含原始 MSE、相对 MSE 和教师输出分布距离三类解释变量。
3. **方向 3：复杂 CSTR 多产物数据**
   - 已确认建模方式：**多数据集分别建模**。每个目标通道单独运行现有 FGL 流程，第一版不实现多输出模型。
   - 必须生成并保存多个产物通道；先以 H2O 为主目标保留兼容性，再扩展温度和其他主要物种。

## 2. 数据与实验对象

### 2.1 CSTR τ 家族

- 使用 `cstr/data/data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl` 的 13 个延迟反馈数据集。
- 每个数据集是单通道 PyTorch tensor `(N, 1)`，现有 FGL 流程可直接使用。
- 实验划分采用 leave-one-example-out：
  - 对目标例 `tau_target`，源例为其余 τ 例；
  - 预训练集按数据集混合，不打乱时间顺序；
  - 目标例使用标准时间划分 `train/val/test = 0.6/0.2/0.2`，预训练阶段不接触目标例。
- 预训练与目标微调都必须使用目标例的共享 bin edges；预训练阶段为避免目标信息泄漏，只能使用各源例自身 bin edges，微调阶段再切换到目标 bin edges。
- 最低规模：13 个目标例 × 3 个随机种子。若预训练成本过高，缓存每个 target 对应的预训练教师权重。

### 2.2 师生优化关系实验

- 先使用现有标准 CSTR 与 13 个 τ 例；不新造数据。
- 以固定 `(L,H)` 锚点网格为基础，建议第一阶段用：
  - 标准 CSTR：`L=20,H=15`；
  - τ 家族：每例 `L=20,H=15`，后续再扩展到当前已知的有效/失效格点。
- 教师处理条件至少包括：
  - `baseline`：常规教师；
  - `more_epochs`：教师训练轮数增加；
  - `lower_lr`：教师学习率降低；
  - `ensemble`：3 教师预测分布平均；
  - `regularized`：权重衰减或 dropout 变化；
  - 可选 `noisy`：输入噪声扰动教师。
- 学生统一使用同一套蒸馏超参数和训练预算，避免把学生超参变化误认为教师效应。
- 每个条件至少 5 个随机种子；教师和学生必须逐 seed 配对。

### 2.3 复杂 CSTR 多产物数据

- 扩展现有 Cantera 生成器，输出一个多通道 `.npz` 或向后兼容的多个 `.pkl`：
  - 第一版推荐保存一个 `data_multichannel_tau{tau}.npz`，包含 `time`、`temperature`、`h2o`、主要中间/产物物种数组；
  - 同时为每个通道生成兼容现有 FGL 的 `(N,1)` tensor `.pkl`；
  - 命名规则：`data_complex_cstr_{channel}_tau{tau}.pkl`。
- 第一版扫 τ ∈ `{50, 70, 90, 100}`，固定延迟反馈参数 `sign=+1, A=0.9, beta=0.03`。
- 通道选择由生成器按方差/动态范围自动保留 top-k，且必须包含温度和 H2O；最低保留 6 个通道。
- 每个 `.npz` 必须附带元数据 JSON：机制文件、物种列表、采样参数、边界条件、tau、生成时间、单位说明。

## 3. 方法设计

### 3.1 跨例基础教师

对每个目标 τ：

1. 读取其余 12 个源例；
2. 用各源例自身的离散化边界构造 H=1 教师任务；
3. 依次或混合训练教师；第一版采用“数据集轮换混合训练”，每个 batch 只来自一个数据集；
4. 得到 `pretrained_teacher`；
5. 建立三个教师条件：
   - `zero_shot`：冻结预训练教师，直接在目标例测试；
   - `finetuned`：在目标例教师训练集上微调少量 epoch；
   - `scratch`：目标例从零训练，训练预算与 finetuned 的总预算匹配；
6. 使用相同测试段计算教师 MSE；
7. 选出更好的教师策略后，再进入学生蒸馏：
   - `distill_from_pretrained_finetuned_teacher`
   - `distill_from_scratch_teacher`
   - `baseline_no_distillation`
   - 学生只使用目标例训练集，教师输入仍按现有 `offset=H-1` 对齐。

### 3.2 教师优化传递关系

对每个数据集、`(L,H)` 和 seed：

1. 训练 `student_baseline_mse`：无蒸馏的 H 步 baseline；
2. 训练 `teacher_baseline_mse`：常规 H=1 教师；
3. 对每个教师处理条件训练得到 `teacher_treated_mse`；
4. 用未处理教师和处理后教师分别蒸馏出两个学生，得到 `student_baseline_teacher_mse` 与 `student_treated_mse`；
5. 保存逐样本 MSE、预测 bin 分布、教师熵和 KL 曲线；
6. 计算传递率；
7. 对每个数据集和整体池分别拟合：
   - 一级线性回归：`student_gain ~ teacher_gain`；
   - 相对模型：`student_relative_gain ~ teacher_relative_gain`；
   - 多变量模型：加入教师熵、教师-学生分布 KL、任务难度 `H/(L+H-1)`；
   - 使用 seed 配对 bootstrap 报告 95% CI。

### 3.3 多通道 FGL

- 对每个通道独立执行 `compute_shared_bin_edges` + `create_time_series_dataset` + `run_fgl_experiment`；
- 不修改模型输入格式；
- 输出统一表结构：`dataset, channel, L, H, seed, baseline_mse, teacher_mse, student_mse, fgl_delta`；
- 每通道按动力学指标（periodicity、Lyapunov、方差、动态范围）分层汇总，判断复杂产物通道是否改变 FGL 有效区。

## 4. 代码与文件结构

新增共享模块，不把实验逻辑写入 `fgl_common/training.py`：

```text
fgl_common/
  transfer.py          # 数据集注册、混合加载、预训练/微调/冻结教师、迁移评估
  transfer_analysis.py # leave-one-out 结果汇总、传递率、统计检验
  teacher_transfer.py  # 教师处理条件、教师/学生配对实验与传递率
  multichannel.py      # 多通道数据读取、元数据校验、单通道 FGL 批处理
cstr/
  generate_complex.py  # 扩展 Cantera 多物种生成器
  run.py               # 注册 transfer_generalization / teacher_student_relation / complex_multichannel
results/
  transfer/            # 跨例迁移结果
  teacher_transfer/    # 师生优化关系结果
tests/
  fgl_common/test_transfer.py
  fgl_common/test_teacher_transfer.py
  fgl_common/test_multichannel.py
  cstr/test_generate_complex.py
```

结果文件命名遵循现有 `{domain}_{experiment}[_{variant}]` 规则：

- `cstr/results/transfer/transfer_teacher_lopo.csv`
- `cstr/results/transfer/transfer_student_distill.csv`
- `cstr/results/teacher_transfer/teacher_optimization_transfer.csv`
- `cstr/results/transfer/transfer_summary.md`
- `cstr/results/teacher_transfer/teacher_transfer_summary.md`
- `cstr/results/complex_cstr_index.md`

## 5. 验收标准

### 功能验收

- `uv run pytest tests/fgl_common/test_transfer.py tests/fgl_common/test_teacher_transfer.py tests/fgl_common/test_multichannel.py tests/cstr/test_generate_complex.py` 全部通过。
- 现有测试不回归。
- `python cstr/run.py --list` 能显示三个新实验开关，默认均为 `enabled=False`。
- 每个新实验支持 `-e` 单独调用，并有 `--quick` 或小参数路径可用于冒烟测试。
- 生成器在无 Cantera 环境下应给出明确错误，而不是静默生成假数据。

### 研究验收

- 方向 1 至少输出 13 个目标例 × 3 seeds 的教师迁移表，比较 `zero_shot / finetuned / scratch`。
- 方向 2 输出逐 seed 配对的教师增益、学生增益、传递率、相关系数和 bootstrap CI。
- 方向 3 至少输出 4 个 τ × 6 个通道 × 3 seeds 的 FGL 表。
- 每个方向在 `results/` 下有可读 summary，说明样本量、显著性、主要异常和下一步建议。

## 6. 风险与控制

| 风险 | 控制 |
|---|---|
| 源例与目标例量纲差异导致预训练失败 | 使用 per-example 标准化 + 目标微调；保留 scratch 对照 |
| 不同源例 bin edges 不兼容 | 预训练用源例各自 edges，微调切目标 edges；记录每例 edges |
| 预训练泄漏目标信息 | leave-one-example-out 硬隔离；目标例测试集不进入任何训练/早停 |
| 教师变化被学生超参吸收 | 学生超参冻结；教师处理和学生蒸馏逐 seed 配对 |
| MSE 变化幅度过小导致传递率不稳定 | 仅在 `teacher_gain > noise_threshold` 时计算；全表保留原始量并用 CI 而不是单点比率下结论 |
| 多物种生成运行成本高 | 第一版仅 4 个 τ；生成前检查已有文件；缓存原始模拟与元数据 |
| Cantera 版本差异 | 元数据记录版本；生成器启动时检查 `cantera >= 3.2.0` |

## 7. 实施顺序

1. 先实现方向 2，因为不依赖新数据且直接服务“教师优化能否转成学生优化”的核心问题。
2. 再实现方向 1 的教师迁移实验。
3. 最后实现复杂 CSTR 生成器和多通道 FGL。
4. 每方向先跑 quick 冒烟，再跑全量。
5. 三个方向结果分别进入独立 summary；如机制一致，再写跨方向综合结论。
