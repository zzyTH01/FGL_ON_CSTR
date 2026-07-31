# 迭代自适应蒸馏(E-iter)纳入 FGL 框架 —— 设计文档

**日期:** 2026-07-31
**状态:** 已批准(用户:核心 API + 统一入口都要)
**关联:** `conclusion/iterative_distillation_summary.md`、`fgl_common/training.py::run_iterative_distillation`、`fgl_common/distillation.py::compute_weights`

---

## 1. 背景与动机

连续自适应蒸馏(E-iter,迭代暖启动 + teacher–student MSE-gap 加权)已在 CSTR 验证有效,其权重分布有两个语义明确的变体:

- **E(硬)**:零地板放大——学生已会的样本蒸馏权重 = 0,最难点饱和到 W_MAX=4,把全部蒸馏预算集中在弱点上。
- **E-soft(稍软化)**:sigmoid 软地板——权重 = `wf + (W_MAX−wf)·σ((gap−c)/s)`,负 gap 落到软地板 wf(默认 0.2),老师信号不断流。MG 上修复了 E 的"干涸崩溃",但稀释 CSTR 的浓度优势(w_floor 是浓度/稳定旋钮,无免费午餐)。

**现状缺口:** 核心代码已存在于 `fgl_common`,但"E-iter 方法"不是一等框架方法:

1. `run_iterative_distillation` 的臂名硬编码为 `"E_single"/"E_iter"`——即使传 `variant="E-soft"`,输出臂名仍叫 E_*,无法区分两种权重分布。
2. 一次调用只能跑一种权重分布,无法同时对比硬/软。
3. `cstr/run.py` / `mackey_glass/run.py` 的 `iterative_distill` 实验已注册但 `enabled=False`,且 **CLI 未暴露 `--variant`/`--w_floor`**,实验体硬编码 4 臂名、不写 CSV。

**目标:** 让 E-iter 成为框架正式训练方法,一次调用同时保留硬(E)与稍软化(E-soft)两种权重分布;统一入口可切换、可落盘、CSTR 默认启用。

---

## 2. 需求决策(用户拍板)

| # | 决策 | 选择 |
|---|---|---|
| 1 | 集成形态 | **核心 API + 统一入口都要**(不是只接线 run.py,也不是并入 run_fgl_experiment 管线) |
| 2 | 臂结构 | **变体列表 × 单/迭代臂**:一次调用传 `weight_distributions=("E","E-soft")`,每变体产出 single+iter 两臂,共享 A_single 对照 |
| 3 | A_iter 对照 | **保留**(均匀权重迭代剥离"多训几轮"效应——地板战役证明其归因价值) |
| 4 | 默认开关 | **CSTR 开、MG 关**(CSTR 验证有效;MG 负结果不烧算力) |

---

## 3. 核心 API 设计(`fgl_common/training.py`)

### 3.1 签名(重构 `run_iterative_distillation`)

```python
def run_iterative_distillation(data, L=20, H=15, alpha=0.5, temperature=4, num_bins=50,
                               epochs=30, round_epochs=15, batch_size=64, patience=5,
                               K=5, eps=0.01, N_stall=2, seed=42, variant=None,
                               weight_distributions=None, w_floor=None, w_floors=None,
                               val_size=0.2, test_size=0.2, lr=1e-4, verbose=True,
                               e_iter_snapshot_fn=None):
```

### 3.2 参数语义

- **`weight_distributions`**(新主参数):变体名元组/列表,如 `("E", "E-soft")`。每个变体产出 `{v}_single` 与 `{v}_iter` 两臂。两者(与 `variant`)都未传时缺省为 `("E",)`——与现状行为一致。
- **`variant`**(向后兼容别名):`variant="E-soft"` ≡ `weight_distributions=("E-soft",)`。与 `weight_distributions` 同时传 → `ValueError`。
- **`w_floor`**(全局,默认 `None` = 0.2,与旧签名语义一致):仅作用于 E-soft 的 sigmoid 软地板。
- **`w_floors`**(可选 per-variant 覆盖):`{"E-soft": 0.1}`,优先级高于 `w_floor`。
- **变体名规范化**:连字符 → 下划线(`"E-soft"` → 臂名 `E_soft_single`/`E_soft_iter`)。修掉"传 E-soft 臂名仍叫 E_*"的 bug。
- **`"A"` 传入 `weight_distributions` → `ValueError`**:A 是对照臂,固定包含,不允许重复指定。

### 3.3 返回结构

```python
{
  "A_single": {...},  "A_iter": {...},            # 共享对照(round-0 / 均匀权重迭代)
  "E_single": {...},  "E_iter": {...},            # 硬权重分布
  "E_soft_single": {...}, "E_soft_iter": {...},   # 稍软化权重分布
}
```

每臂值与现状一致:`{teacher_mse, baseline_mse, student_mse, fgl_delta, init_delta, rounds_used, total_epochs, mse_curve_val, mse_curve_test}`。

### 3.4 内部实现

- `_iterate_student` / `_compute_arm_weights` / `_should_stop` **零改动**——变体名已是透传参数,`compute_weights` 已支持 E / E-soft + `w_floor`。
- 仅重构 `run_iterative_distillation` 的臂组装部分:`_arm(variant, max_rounds)` 不变,`arms` 字典从"固定 4 臂"改为"A_single + A_iter + 遍历 weight_distributions 生成 {v}_single/{v}_iter"。
- `w_floors` 解析:每变体取 `w_floors.get(v, w_floor)`(E 变体忽略)。

### 3.5 向后兼容

- `variant="E"`(旧默认)产出 4 臂与现在**逐字段一致**——由既有测试保证。
- 旧调用方零强制改动:`cstr/run_floor_sweep.py`(variant="E")、`cstr/run_iterative_delayed.py`、`cstr/plot_iterative_rounds.py`、`cstr/sweep_iterative.py`(--variant 已有)继续可用。

---

## 4. 统一入口设计(`cstr/run.py` + `mackey_glass/run.py`)

### 4.1 `cstr/run.py`

- `EXPERIMENTS["iterative_distill"]` → `enabled=True`,note 更新为"迭代自适应蒸馏(E 硬 / E-soft 稍软化,双权重分布)"。
- CLI 新增:
  - `--distill_variants`(默认 `"E,E-soft"`):逗号分隔变体列表。
  - `--w_floor`(默认 0.2):E-soft 软地板。
- 实验体 `run_iterative_distill_exp`:
  - 传 `weight_distributions=tuple(args.distill_variants.split(","))`、`w_floor=args.w_floor`。
  - **动态取臂**:不再硬编码 `("A_single","E_single","A_iter","E_iter")`,遍历 `arms` 结果 dict 的键。
  - **CSV 落盘** `cstr/results/iterative_distill.csv`,每行 = seed × arm:`{seed, arm, student_mse, baseline_mse, teacher_mse, fgl_delta, init_delta, rounds_used, total_epochs}`(与 `adaptive_lh_sweep.csv` 等落盘惯例一致)。
  - 终端 summary 保持,但臂列表动态。

### 4.2 `mackey_glass/run.py`

- `EXPERIMENTS["iterative_distill"]` 保持 `enabled=False`(MG 负结果)。
- 同样加 `--distill_variants`/`--w_floor` CLI(手动 `-e iterative_distill` 可用),实验体同样动态取臂 + CSV 落盘 `mackey_glass/results/iterative_distill.csv`。

---

## 5. 测试计划(`tests/fgl_common/test_iterative_distillation.py`)

**既有测试全部保持通过**(向后兼容的回归网):`test_run_iterative_four_arms_structure`、`test_all_arms_share_round0`、`test_E_single_uses_one_round` 等。

**新增用例:**

| 用例 | 断言 |
|---|---|
| `test_weight_distributions_list_produces_all_arms` | 传 `("E","E-soft")` → 臂集合恰为 `{A_single, A_iter, E_single, E_iter, E_soft_single, E_soft_iter}` |
| `test_variant_alias_equivalence` | `variant="E-soft"` 与 `weight_distributions=("E-soft",)` 产出的臂结果一致 |
| `test_variant_and_weight_distributions_conflict` | 两者同传 → `ValueError` |
| `test_A_in_weight_distributions_rejected` | 传 `("A",)` → `ValueError` |
| `test_arm_name_normalization` | 变体 `"E-soft"` 的臂名为 `E_soft_single`/`E_soft_iter` |
| `test_w_floors_per_variant_override` | `w_floors={"E-soft": 0.1}` 时 E-soft 臂权重地板为 0.1(经 `_compute_arm_weights` 直接测) |
| `test_default_weight_distributions_is_E` | 不传 variant → 4 臂结构(回归) |

---

## 6. 文档更新

- `fgl_common/training.py` 的 `run_iterative_distillation` docstring:双权重分布语义、臂命名、参数说明。
- `fgl_common/__init__.py` 模块 docstring 微调(如需要)。
- `CLAUDE.md`:cstr/run.py 行 `(enabled: baseline, lh_sweep)` → 加 `iterative_distill`;fgl_common 段 `run_iterative_distillation` 描述更新为"双权重分布(E 硬 / E-soft)";mg/run.py 行注明 iterative_distill 保持 off。

---

## 7. 错误处理与边界

| 情况 | 行为 |
|---|---|
| `variant` + `weight_distributions` 同传 | `ValueError`,提示用哪个 |
| `weight_distributions` 含 `"A"` | `ValueError`,A 是对照臂固定包含 |
| `weight_distributions` 空元组 | `ValueError`(无臂可产) |
| `w_floors` 含未使用变体 | 忽略(不报错) |
| 变体名不在 compute_weights 支持集 | 由 `compute_weights` 既有 `ValueError` 兜底 |

---

## 8. 范围外(YAGNI)

- 不并入 `run_fgl_experiment` 管线(用户未选"融入管线"形态)。
- 不做"动态地板"(迭代蒸馏总结 §5.5 提到的下一步 idea)——不在本任务。
- 不迁移旧实验脚本的调用点(兼容即可,`sweep_iterative.py` 等保持独立)。
- 不新增域(lorenz 不注册 iterative_distill)。

---

## 9. 验收标准

1. `uv run pytest tests/fgl_common/test_iterative_distillation.py` 全绿(旧 + 新用例)。
2. `uv run python cstr/run.py --list` 显示 `iterative_distill: enabled=True`;`uv run python cstr/run.py -e iterative_distill --distill_variants E,E-soft --seeds 1` 跑通并产出 `cstr/results/iterative_distill.csv`(含 6 臂)。
3. `uv run python mackey_glass/run.py --list` 显示 `iterative_distill: enabled=False` 且 CLI 含 `--distill_variants`。
4. 旧调用 `variant="E"` 的 4 臂结果与重构前一致(由测试 3.5 覆盖)。
