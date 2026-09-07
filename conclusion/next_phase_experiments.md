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
