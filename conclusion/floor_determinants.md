# 延迟 CSTR 地板成因结论

**日期:** 2026-07-30
**数据:** `cstr/results/floor_sweep.csv`(45 行:τ=100 深挖 13 cells×3 seeds + τ50/τ150 锚)+ `cstr/results/lyapunov_tau.csv`
**深挖数据集:** tau100(per=0.486,λ=0.0079/样本)
**问题:** 是什么决定了能达到的最低 student MSE?如何达到它?

---

## TL;DR

1. **H1 证伪**:CSTR 上**没有** MG 那种 `L+H-1≥τ` 的尖锐相变;baseline 随 L+H-1 是**缓降**(H=15 列 172→143→134),无阈值。
2. **H2 支持**:连续蒸馏地板 `≈ 0.76× baseline`(CV=0.21)——蒸馏是**固定比例降幅**,地板的决定因素不变。
3. **H3 修正后成立**:floor 不是简单 ∝ teacher_mse(裸拟合 R²=0.12),但在固定 H 下 floor~teacher R²=0.5–0.73;多元 `log(floor) = 3.83 + 0.016·teacher + 0.035·H − 3.49·(1/L)`,**R²=0.72**。
4. **H4 证伪(分层后更精细)**:连续蒸馏**压破**了匹配算力的 baseline 地板(p=0.0002),但**只在难 horizon**——H=15 时 E_iter 比 baseC 低 20.7(18/21 赢),H=30 低 10.2;H=5(易)反而**高 3.0**(1/9 赢)。
5. **如何达到最低 MSE**:H≥15 时用连续蒸馏;H 小时蒸馏无益甚至有害,直接训 baseline 到收敛即可。

---

## 四假设判决(含修正)

### H1 — `L+H-1≥τ` 相变:**证伪**

固定 H=15 扫 L(L+H-1 ∈ {34,54,64,84,99,114,134}),baseline_mse:172→138→143→138→139→134→131。最大连续降幅仅 0.99×(基本持平)。**无 τ=100 处的相变**,只有随 L 的缓降。

> **与既有结论的对账:** 这**证伪了** `chaotic_cstr_fgl_exploration.md` §4.2 的假设("L+H-1≥τ 阈值在 CSTR 上的 1:1 映射")。MG 的 `L+H-1≥τ` 巧合是**几何特异性**的(倍周期分岔、teacher offset 精确对齐),**不普适到混沌延迟 CSTR**。在混沌数据上,增大 L 是"看到更多历史 → 误差缓降",没有阈值开关。与 `final_conclusions.md` §2.2 的适用边界提示("仅固定 H 扫 L 严格成立")一致,但进一步表明即便满足该条件,混沌 CSTR 上也无相变。

### H2 — `c=E_iter/baseline` 常数:**支持**

跨 13 cells,c 均值 **0.76**(CV=0.21)。即连续蒸馏稳定地把地板降到 baseline 的 ~76%,比例在 (L,H) 上近似恒定。

> **与既有结论的对账:** 把 `chaotic_cstr_fgl_exploration.md` §4.1 的"aperiodic ~0.80–0.87× baseline"(单点 L20H15)推广到整个 L×H 网格:比例 ~0.76,且**不随 L/H 系统漂移**。含义:蒸馏不改变"什么决定地板",只给一个固定折价。

### H3 — floor∝teacher_mse(Lyapunov 标定):**修正后部分成立**

- **裸拟合**(跨所有 H/L):floor vs teacher_mse R²=**0.12**(弱)——因为 floor 主要由 H 驱动,teacher_mse 变化小。
- **固定 H 下**:floor~teacher 的 R²:H=5→0.47,H=15→0.73,H=30→0.63。**控制住 H,teacher 质量是 floor 的强预测器**。
- **多元模型**:`log(floor) = 3.83 + 0.016·teacher + 0.035·H − 3.49·(1/L)`,**R²=0.72**。
  - H 的系数 0.035/样本 ≈ **4.4× Lyapunov λ(0.0079)**。floor 随 H 的增长比纯轨迹发散更快(离散化 + 模型误差复合),所以 H3 的**严格形式** `floor~teacher·exp(λH)`(斜率=λ)**证伪**,但**定性结构**(floor 随 H 指数增长、固定 H 下正比 teacher)**成立**。
  - teacher 系数 +、1/L 系数 −:teacher 越好 / L 越大 → floor 越低,与机制自洽。

### H4 — 蒸馏 vs 匹配算力 baseline 地板:**证伪(分层后精细化为 regime-dependent)**

整体配对 t:E_iter 比 baseline_converged 低 **12.8**(p=0.0002)→ 蒸馏**确实压破**了"训 baseline 到收敛"的地板。**但分层后真相是 regime-dependent:**

| H | E_iter − baseC 均值 | 蒸馏更低占比 | 判读 |
|---|---|---|---|
| 5(易) | **+3.0** | 1/9 | 蒸馏**无益甚至有害** |
| 15(难) | **−20.7** | 18/21 | 蒸馏显著压破地板 |
| 30(最难) | **−10.2** | 6/9 | 蒸馏压破(方差大) |

> **机制对账(`final_conclusions.md` §2.4 信息不对称):** H 小时,学生历史窗口已覆盖 teacher 能提供的"近未来"→ 蒸馏冗余、目标错位反而拉偏(KL 把学生拉向 teacher 的 t+1,而学生要 t+H,二者在易任务上几乎重合 → 噪声)。H 大时,teacher 的 offset=H−1 真正提供了学生历史推不出的近未来 → 蒸馏净增益。**这正是"何时 FGL 有效"在 floor 层面的具象化。**

> **与既有结论的对账:** 不冲突。`iterative_distillation_summary.md` 的"E_iter≈A_iter(同地板)"比较的是**两种蒸馏之间**;这里 H4 比的是**蒸馏 vs 无 teacher baseline**——结论是 teacher 信号本身(而非"多训")压破了地板。两者互补:蒸馏内部 E≈A(权重不决定),但蒸馏整体 < 收敛 baseline(teacher 信息决定)。

---

## 如何达到最低 MSE(可操作结论)

1. **先选 (L, H)**:floor 由多元公式 `log(floor) ≈ 3.83 + 0.016·teacher + 0.035·H − 3.49/L` 主导。压低地板 → **降 H、升 L**。
2. **再决定是否蒸馏**:
   - **H 大(难,本数据 H≥15)** → 用连续蒸馏(E_iter),能再压 ~13–21% 到 baseline 之下。
   - **H 小(易,H=5)** → 别蒸馏,直接训 baseline 到收敛(epochs↑、patience↑);蒸馏反而 +3%。
3. **蒸馏超参数不决定地板**(α/T/加权):与 `iterative_distillation_summary.md` 一致。决定地板的是 **(L, H) 与数据可预测性(teacher_mse、λ)**,以及"是否蒸馏"这个二分。

---

## Lyapunov 顺带结果(补项目缺口)

全部 13 个延迟数据集 λ ∈ [0.0018, 0.0108]/样本,**均为正** → **它们是真混沌,不是噪声**。这消除了 `chaotic_cstr_fgl_exploration.md` §1.4 的 caveat("低周期性≠严格混沌,未算 Lyapunov")。λ 与 periodicity 不单调(τ=30 λ=0.0108 最大但 per=0.79;τ=50 λ=0.0018 最小但 per=0.56)→ "周期性"与"混沌度"是两个独立维度,不应混用。

---

## 局限 / 下一步

- **n=3 seeds**,H4 的 H=30、H3 的 H=5 R² 偏低,需补到 5 seeds(关键 cell 已留接口)。
- **val≈test 过拟合**:周期性残余 → keep-best-by-val 近似在 test 选优;H4 的配对 t 偏乐观。独立 holdout 待做。
- **τ vs periodicity 混杂**:横向锚点 τ50/150 同时变两量,横向结论仅"趋势"。**方案 C(去混杂:固定 per 变 τ)** 是把 H1 变因果检验的下一步。
- **MSE 单位**为 bin-index²(50 bins),仅相对分析。
- 图:`cstr/results/floor_h{1,2,3,4}_*.png`。
