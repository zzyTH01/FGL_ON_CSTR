# 迭代自适应蒸馏(Iterative Adaptive Distillation)— 设计文档

> 日期：2026-07-28
> 状态：设计已与用户确认，待写实现计划
> 范围：CSTR 域（锚点 L=20,H=15 + 5×5 L×H 网格）
> 依赖：`fgl_common/training.py::run_adaptive_weight`（单趟 E 已验证有效，网格级配对 t=−8.16，df=49，p≪0.001）

---

## 0. 驱动假设（用户核心理念）

**H1**："既然单趟实验已证明老师能教会学生（变体 E 显著优于均匀对照 A），那么多蒸馏几轮（把上一轮 student 再用更新后的权重蒸馏一次）效果会更好。"

这是一个**待检验的假设，不是已知结论**。实验必须能区分三种可能结局：

| 结局 | 现象 | 机理 | 判据 |
|------|------|------|------|
| **H1a 单调变好** | MSE 随轮数持续下降 | 每轮修不同弱点，渐进精修 | E-iter 曲线单调降且 > E-single |
| **H1b 饱和** | 前几轮降，之后平台 | 教师"独占信息"有限，榨干见底（信息不对称天花板） | 曲线前几轮降后趋于常数 |
| **H1c 退化** | 先降后升 | 暖启让学生过拟合 teacher 的 t+1 分布，偏离 t+H 目标（地板效应放大） | 曲线出现极小值后回升 |

**外加混淆（必须排除）**："多教"还是"多训"？——由 A-iter 归因对照分离：同样多轮暖启、同样 epoch 预算，但权重始终均匀。`E-iter > A-iter` 才是"迭代权重精修"的净贡献。

> 无论落到哪种结局，都是有价值的结论（含负结果）。

---

## 1. 机制：暖启动渐进自蒸馏（变体 E）

固定 teacher（1-step oracle，冻结）与 baseline（H-step，冻结）。只让学生迭代：

```
Round 0:  student_0 ← from scratch, uniform KL (标准 FGL)
          eval(val) → MSE_0

Round t (t = 1..K):
  1. 暖启动:     student_t ← deepcopy(student_{t-1}.state_dict())
  2. 重估权重:   w_t = compute_weights('E',
                   gap = max(0, se_student_{t-1} − se_teacher))
                 零地板放大到 [0, W_MAX]
  3. 继续训:     student_t 跑 round_epochs，
                 loss = α·CE(output, target) + (1−α)·T²·weighted_KL(w_t)
  4. eval(val) → MSE_t
  5. 停止判定（见 §3）

返回: best-by-val 的 student（非最后一轮）
```

**关键点**：每轮权重 `w_t` 由**上一轮 student** 重新估计——学生修好旧弱点、新弱点浮现，蒸馏预算随之转移。这是"循环"的本质，也是与单趟 E（权重只估一次）的根本区别。

---

## 2. 四臂：2×2 因子设计（归因分解）

四臂**共享 round-0**（teacher / baseline / student_0 按 seed 完美对齐，只训一次），然后分支。每臂开始前 `deepcopy(student_0.state_dict())`，保证四臂从**同一个 round-0 权重**出发。

|  | **single**（round-0 后停） | **iter**（round-0 后再暖启 ≤K 轮） |
|--|--|--|
| **uniform (A)** | A-single = student_0 本身 | A-iter（权重始终均匀，每轮"伪重估"得全 1）|
| **adaptive E** | E-single（暖启 1 轮，w 来自 student_0） | **E-iter**（每轮重估 w） |

**归因对比**：
- `E-single vs A-single`：一发 E 权重 + 暖启的增益。
- `A-iter vs A-single`：**纯训练量**效应（同样均匀，只是多训 K 轮）。
- **`E-iter vs A-iter`**：匹配预算下 E 的**迭代净增益** ← 核心论断。
- `E-iter vs E-single`：超过一发的迭代收益。

**健全性检查（必做）**：E-single（暖启 1 轮）应近似复现现有 fresh-E 的 Δinit≈+33%（L=20,H=15）。若明显偏离，先排查再继续——保证新框架与已验证结果接轨。

> 说明：现有 `run_adaptive_weight` 的 E-single 是 **fresh 重训**最终 student；本设计的 E-single 改为**暖启 1 轮**，是为了让四臂构成干净的 2×2（仅"轮数"与"权重"两维度变化）。两者数值应接近，但口径不同；对比以本设计的四臂为准。

---

## 3. 停止规则

每轮后在 **val MSE** 上判定（不碰 test，防泄漏），返回历史 val 最优 student：

- **收敛**：`(MSE_{t-1} − MSE_t) / MSE_{t-1} < ε` 连续 N 次 → 停
- **退化**：`MSE_t > MSE_{t-1}` → 立即停（暖启漂移保护）
- **兜底**：`t ≥ K` → 停

默认：ε=1%, N=2, K=5。

---

## 4. 逐轮 MSE 曲线（直接回答 H1）

每个 iter 臂记录并返回逐轮轨迹 `MSE_0, MSE_1, …, MSE_{used}`（val 与 test 各一条；test 仅记录、不参与停止决策）。sweep 脚本绘制：

- E-iter 与 A-iter 的逐轮 MSE 曲线（按 cell 平均，或锚点上按 seed 平均）
- 曲线形状 = H1 的直接视觉答案（单调降 / 饱和 / 先降后升）

这让"退化即停 + keep-best"不只是保险，而是 H1c 的正面记录。

---

## 5. 预算与控制公平性

- round-0：`epochs=30` + EarlyStopper（patience）。
- round 1..K：每轮 `round_epochs=15` + 独立 EarlyStopper（val）。
- **E-iter 与 A-iter 用完全相同的轮次表**，唯一差别是权重是否更新 → 训练总量按构造对齐。
- EarlyStopper 会让两臂实际 epoch 略有差异（val 决定）；**每臂报告 `total_epochs_consumed`** 透明化，对比时一并看。

---

## 6. 实现位置

| 文件 | 改动 |
|------|------|
| `fgl_common/training.py` | 新增 `run_iterative_distillation(...)`：训一次 teacher/baseline/student_0，分支跑四臂；每臂返回 `{arm, rounds_used, total_epochs, teacher_mse, baseline_mse, student_mse, fgl_delta, init_delta, mse_curve_val:[...], mse_curve_test:[...]}`。复用 `compute_per_sample_mse`、`compute_weights`、`KL_weighted`、`EarlyStopper`、`compute_shared_bin_edges`、`create_time_series_dataset`。 |
| `fgl_common/distillation.py` | 无改动（`compute_weights` 已支持 E）。 |
| `fgl_common/__init__.py` | 导出 `run_iterative_distillation`。 |
| `cstr/run.py` | 加实验开关 `iterative_distill`（默认 `enabled=False`），薄包装调用 + n-seed 汇总打印。 |
| `cstr/sweep_iterative.py`（新） | 镜像 `sweep_adaptive.py`：L×H 网格 × seeds × 4 臂；输出 `iterative_lh_sweep.csv` + 两张热力图（E-iter vs E-single、E-iter vs A-iter）+ 逐轮 MSE 曲线图。 |

### 函数签名（草案）

```python
def run_iterative_distillation(
    data, L=20, H=15, alpha=0.5, temperature=4, num_bins=50,
    epochs=30, round_epochs=15, batch_size=64, patience=5,
    K=5, eps=0.01, N_stall=2,            # 停止规则
    W_MAX=4.0,                            # 变体 E 放大上限
    seed=42, variant='E', verbose=True,
):
    """返回 {arm: {arm, rounds_used, total_epochs, teacher_mse,
                  baseline_mse, student_mse, fgl_delta, init_delta,
                  mse_curve_val, mse_curve_test}} for arm in
       {A_single, E_single, A_iter, E_iter}."""
```

teacher/student 对齐沿用现有 `offset=H−1` + `j−(H−1)` 重映射（已修复，见 `run_adaptive_weight`）。

---

## 7. 评估口径与成功判据（分两阶段）

### Phase 0 —— 试点（先拿信号，便宜一个量级）

目的：在投入全网格前，用少量典型点验证概念是否成立。三点按"单趟 E 相对 A 的降幅"分层选取，覆盖中/大/天花板三档，能看出**迭代收益是否随可改进空间缩放**：

| 典型点 | L | H | A(均匀) | E(加权) | E vs A | 选点理由 |
|--------|---|---|---------|---------|--------|----------|
| **P1 锚点·中增益** | 20 | 15 | 118.7 | 94.1 | −20.7% | 已验证、口径对照、参考基准 |
| **P2 高增益·大空间** | 8 | 30 | 129.8 | 64.4 | −50.3% | 均匀学生最弱、迭代空间最大；短 L+大 H=强信息不对称 |
| **P3 地板·天花板检验** | 72 | 15 | 14.7 | 14.2 | −3.5% | A 已逼近地板，迭代**应当**无增益；验证停止规则不退化 |

- 配置：3 典型点 × 3 seeds × 4 臂；round-0 `epochs=20`、`round_epochs=10`、`K=3`。
- 记录每臂逐轮 MSE 曲线（§4）。
- **放行判据**（全部满足才进 Phase 1）：
  - P1/P2：E-iter ≤ E-single（迭代有收益）**且** E-iter ≤ A-iter（净效应为正，排除"只是多训"）。
  - P3：E-iter ≈ A-iter（±噪声）且**不退化**（停止规则有效，未漂移）。
- **不放行**：E-iter 在 P1/P2 退化、或 E-iter ≈ A-iter 但 A-iter ≈ A-single（纯训练量无加权收益）→ 复盘（查曲线形状、W_MAX、round_epochs）再定，不盲目上全网格。

### Phase 1 —— 全验证（Phase 0 放行后）

- **锚点 L=20, H=15，n=10 seeds**（配对）：
  - 主判据：`E-iter vs A-iter` 配对 t / Wilcoxon，p<0.05 且有效应量 → **迭代权重精修有净价值**。
  - 次判据：`E-iter vs E-single`（迭代 > 一发？）、`A-iter vs A-single`（训练量成分）。
- **5×5 L×H 网格 × 2 seeds**：E-iter<A-iter 胜率 + 平均相对降幅 + 网格级配对 t（同 `sweep_adaptive` 口径）。

**结论分级**：
- ✅ **概念成立**：E-iter 显著 > A-iter（锚点 p<0.05）。
- ✅ **强结果**：同时 E-iter > E-single（迭代超一发），且网格胜率 >80%。
- ⚠️ **负结果（仍报告）**：E-iter ≈ A-iter 但都 > E-single → 增益主要是"多训"，迭代加权无净贡献。
- 曲线形状判定 H1a/H1b/H1c（§4）。

---

## 8. 风险与防护

| 风险 | 防护 |
|------|------|
| 暖启漂向 teacher 的 t+1 目标 → t+H MSE 爆 | 退化即停 + keep-best-by-val + α·CE 锚定真 t+H 目标 |
| 多 epoch 过拟合 | 每轮 EarlyStopper(val) + keep-best-by-val |
| 权重振荡不收敛 | 收敛停滞检测（N_stall）+ K 兜底 |
| test 泄漏 | 停止判定全在 val，test 仅记录、报告 best-by-val 的 test |

---

## 9. 默认参数表

| 参数 | Phase 0 试点 | Phase 1 全验证 | 说明 |
|------|------|------|------|
| 典型点 / 网格 | P1/P2/P3 三点 | 5×5 全网格 + 锚点 | |
| L, H（锚点） | 20, 15 | 20, 15 | 与现有 E 验证同口径 |
| α, T | 0.5, 4 | 0.5, 4 | 与现有 E 一致 |
| num_bins | 50 | 50 | |
| epochs (round-0) | 20 | 30 | + EarlyStopper |
| round_epochs | 10 | 15 | round 1..K 每轮 |
| K | 3 | 5 | 最大轮数兜底 |
| eps, N_stall | 1%, 2 | 1%, 2 | 收敛判定 |
| W_MAX | 4.0 | 4.0 | 变体 E 放大上限（未调） |
| n_seeds | 3 | 10（锚点）/ 2（网格） | |

---

## 10. 不在本 spec 内（未来工作）

- 移植到 MG / Lorenz（迭代版 vs 单趟 E 的跨域普适性）。
- 温和变体 C 的迭代版（对照"显著化是否仍只来自放大版 E"）。
- W_MAX / 地板 / round_epochs 的调参。
- 与 α×T 调优的叠加。
