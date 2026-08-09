# 迭代蒸馏的"收敛速度"维度:E_iter vs A_iter 谁更快到地板?

**日期:** 2026-08-09
**研究问题:** 选择自适应权重(E_iter)做连续蒸馏时,student 到达 MSE 地板的速度,是否比只做均匀权重连续蒸馏(A_iter)更快?在 CSTR(标准 + 混沌延迟)与 MG 上分别如何?MG 上 E 效果不好是否与收敛速度有关?
**一句话回答:** **标准 CSTR 上 E 稳健地更快(中位 1 轮 vs 2 轮,首轮进步是 A 的 ~1.7×,p=0.016/3e-5),但地板持平;混沌 CSTR 上 E 的地板优势是 regime-dependent(中 L/H15 cell 真赢、最难 cell 灾难性崩);MG 上 E 的速度优势消失——且 MG 的 E 效果差确实与"收敛速度"有关,但机制是底层任务收敛太快、把 E 的加机制饿死。**

> 配套代码:`cstr/archive/analyze_convergence_speed.py`(速度主指标 + 配对统计 + 逐 cell 分化,跨域)、`cstr/archive/verify_chaotic_iter_floor.py`(混沌 CSTR K=10 饱和核实驱动)。
> 新数据:`cstr/results/chaotic_floor_verify.csv`(4 cell × 5 seed × 2 arm + 逐轮曲线)。
> 上游结论:`iterative_distillation_summary.md`、`floor_study_final_report.md`、`iterative_pilot_phase0*.md`。

---

## 1. 背景与动机:被弱化的"速度"维度

迭代(暖启动)自适应蒸馏的四臂 2×2 设计(`fgl_common.run_iterative_distillation`,共享 round-0):

|  | single(round-0 后停) | iter(round-0 后暖启 ≤K 轮) |
|--|--|--|
| **均匀 (A)** | A-single = 标准 FGL | **A_iter = 连续蒸馏,权重恒均匀**(归因对照)|
| **自适应 E** | E-single(暖启 1 轮)| **E_iter = 连续蒸馏,每轮重估 E 权重**(新方法)|

`E_iter vs A_iter` = 迭代权重精修的净效应(排除"只是多训")。本文对照即这两条臂。

**已有结论(`iterative_distillation_summary.md`,2026-07-28)的口径**:把"地板高度(per-seed-min test MSE)"当主指标,跑到饱和(K=6/15)后发现 E_iter 与 A_iter 地板都到 ~30,于是用配对 t **p=0.94** 判定"E_iter ≈ A_iter"。并附带一句"E 更可靠 + 更快到地板,非更低地板"。

**问题**:那句"更快"从没被当成主指标量化过。p=0.94 是**地板高度口径,不是速度口径**——它把速度优势埋掉了。本轮把"到达地板的速度"立为主指标,跨所有 CSTR(标准 + 混沌)与 MG 数据补上这一刀。

---

## 2. 方法

**速度主指标**(逐轮曲线 `mse_curve_test`,index 0 = round-0,t = 第 t 轮后):
- `rounds_to_floor`:到达"共享地板 10% 内"的轮数(共享地板 = 两条曲线的全局 min;越少越快)。
- `round1_gain`:round-0→round-1 相对进步 `(c0−c1)/c0`(越大越快)。
- `auc_norm`:归一化曲线下面积 `mean(c/c0)`(越低越快)。
- `floor`:各臂 min(test MSE)——仅作口径参照。

**统计**:配对均值差 E−A + 95% CI(t 分布)+ Wilcoxon signed-rank + 胜/平/负计数。退化率 = 曲线含任何上行(重估权重反噬)的 seed 比例。

**数据源**:
- **标准 CSTR(有曲线)**:`iterative_phase1_anchor.csv`(L20H15 n=10,K=6)、`iterative_phase1_grid.csv`(5×5×2,K=6)、`iterative_sweep.csv`(3 点 ×3,K=3)、`iterative_saturation_sweep.csv`(2 点 ×3,K=15)。
- **混沌 CSTR**:① `iterative_delayed_summary.csv`(4 个 τ 数据集,仅 `rounds_used` 汇总,无曲线);② `floor_sweep.csv`(τ100,13 cell ×3 seed,仅最终地板);③ **本轮新增 `chaotic_floor_verify.csv`(4 cell ×5 seed,K=10 饱和,有曲线)**。
- **MG(有曲线)**:`iterative_mg_sweep.csv`(3 点 ×3,K=5)+ 4 个 E-soft w_floor 变体。

---

## 3. 标准 CSTR:E 稳健更快,地板持平(补上速度侧,与旧结论不冲突)

| 数据集(配对差 E−A) | rounds_to_floor Δ [95%CI] | p | round1_gain Δ | floor Δ | rounds 中位 A→E | 速度胜 E/A |
|---|---|---|---|---|---|---|
| **anchor L20H15 (n=10)** | **−1.43 [−1.92, −0.93]** | **0.016** | +0.27(正=E 进步大),p=0.002 | −0.34(NS)| 2.5→1.0 | **7 / 0** |
| **grid (50 配对)** | **−0.78 [−1.15, −0.41]** | **3e-5** | +0.13,p=3e-5 | −3.9(NS)| 1→1 | **16 / 0** |
| sweep (K=3) | 0(K 太小)| — | — | −21.4(K=3 快照)| — | — |
| saturation (K=15) | median **7→4** | — | ≈ | −17.3(NS)| 7→4 | 2 / 0 |

- **速度**:anchor 中位 1 轮到地板 vs A 的 2 轮;**首轮进步 E=+67.8% vs A=+40.9%(~1.7×)**;grid 上 E 在 16/50 配置更快、**A 在 0 个更快**;即便 K=15 饱和,E 仍比 A 早 3 轮。
- **地板**:anchor 持平(同 ~30,Δ NS);grid/saturation 上 E 略低但不稳健。
- 即旧结论"E≈A"在**地板高度**上成立,但**速度**上 E 明显更快——"用一半轮数到同一地板"。

⚠️ E 曲线退化率 anchor 9/10、grid 33/50——重估反噬频繁,但中 L 处早期红利盖过它。

---

## 4. 混沌 CSTR:K=5 有混淆,K=10 饱和核实(本轮新实验,核心)

### 4.1 发现混淆

`iterative_delayed_summary.csv` 里 τ100 的 **A_iter `rounds_used=5.0` 正好顶到 K=5 cap** → A 到第 5 轮还在下降、被截断、未收敛;而 E_iter `rounds_used=3.33` 提前停。`floor_sweep`(K=5)读到"E<A 9/13 cell,Δ=−9.5 [−17.6,−1.4],p=0.053"含"A 没追上"成分。**必须饱和后重判。**

### 4.2 K=10 饱和核实(`chaotic_floor_verify.csv`,τ100,4 关键 cell × 5 seed)

| cell | A 地板(mean±sd) | E 地板(mean±sd) | E−A [95%CI] | A rnd | E rnd | E 赢 | E 崩¹ | 判定 |
|---|---|---|---|---|---|---|---|---|
| **L85H15** | 119.7±2.2 | **87.0±21.2** | **−32.7 [−58.6, −6.8]** | 5 | 4 | **4/5** | 0/5 | ✅ **E 真赢(CI 排除 0)** |
| L50H15 | 118.1±1.3 | 100.5±33.7 | −17.6 [−59.3, +24.0] | 7 | 4 | 3/5 | 0/5 | ⚠️ E 赢但不稳(sd 33.7)|
| L20H15(锚点)| 145.3±1.2 | 145.6±10.2 | +0.3 [−13.6, +14.1] | 6 | 4 | 3/5 | 0/5 | ≈ 持平 |
| **L100H30** | 89.5±6.2 | 156.4±15.3 | **+66.8 [+49.0, +84.6]** | 10² | 2 | 0/5 | **5/5** | ❌ **E 灾难性崩** |

¹ E 崩 = E > A×1.3(早停在坏盆底,启发式阈值)。
² A 在 L100H30 用满 10 轮仍未饱和(顶 cap),但已 2× 优于 E。

**关键结论**:
- **L85H15 上 E 的地板优势是真实的,不是 K=5 假象**:K=10 饱和后仍 −27%、CI 排除 0、4/5 seed。
- **"E≈A on delayed CSTR"是锚点特例**:旧结论只看 L20H15(确实 ≈),摊到网格上是 **regime-dependent**——中 L/H15 cell E 真赢,最难 cell(L100H30)E 灾难性失败。
- **L100H30 的 E 崩机制**(典型曲线):E `201→170→176`,第 2 轮就上行 → 触发 degradation-stop → 锁死在 ~155;A 用满 10 轮磨到 89。混沌 CSTR 上 **E 曲线退化率 19/20**(pooled)。

### 4.3 混沌 CSTR 汇总(4 个 τ 数据集,`rounds_used` 粗速度)

| dataset | periodicity | A 地板 | E 地板 | A rnd | E rnd | 判读 |
|---|---|---|---|---|---|---|
| base_h2o(period-1) | 0.94 | 30.1 | 39.2 | 4.67 | 3.00 | E 更少轮但地板更高(早停)|
| tau50 | 0.56 | 84.6 | 82.9 | 3.33 | 2.33 | E 更快 + 同/更低地板 |
| tau100 | 0.49 | 145.8 | 144.6 | 5.00 | 3.33 | E 更快 + 同地板 |
| tau150 | 0.47 | 135.8 | 129.6 | 4.33 | 4.33 | 速度持平 |

混沌化(τ↑、periodicity↓)整体上 E 用更少轮数收敛,base_h2o 上 E 早停导致地板反高。

---

## 5. MG:速度优势消失;E 效果差确实与收敛速度有关——但机制是"底层收敛太快饿死 E"

`iterative_mg_sweep.csv`(3 点 ×3,K=5):

| 指标(E−A) | Δ [95%CI] | p | 读数 |
|---|---|---|---|
| rounds_to_floor | −0.67 [−4.46, +3.13] | 0.50 | NS;中位 A=4 **E=5**(E 反而略慢)|
| round1_gain | +0.054 [−0.042, +0.15] | 0.16 | NS(E 首轮进步优势消失)|
| floor | +0.496 [−0.65, +1.64] | 0.64 | NS;A 略低 5/9 |
| 退化率 | — | — | A=3/9,**E=5/9** |

**MG 上 E 既不更快(中位还慢 1 轮),地板也不更低。** E-soft(wf 0.05–0.20)也救不回速度(全 NS)。

### MG 机制:CSTR vs MG 对照(回答"E 效果差是否与收敛速度有关")

| | round-1 进步 A | round-1 进步 E | **E/A 进步比** | E 曲线退化率 |
|---|---|---|---|---|
| 标准 CSTR anchor | +40.9% | +67.8% | **1.70** | 9/10 |
| MG sweep | +32.2% | +37.6% | **1.27** | 5/9 |

**答:有关,但不是"E 收敛慢",而是"底层任务收敛太快,把 E 的加机制搞坏"。** MG 是易任务(baseline ~12、地板 ~0.5),student 收敛极快 → teacher−student MSE gap 在第 2–3 轮塌掉 → E 的 gap 驱动权重在大部本上归零(蒸馏池"干涸")→ 重估出噪声权重 → E 退化(5/9)早停 → 最终地板还略输。E/A 首轮进步比从 CSTR 的 1.70 缩到 MG 的 1.27,即 E 的"浓度"加成在 MG 上基本用不上。

---

## 6. 统一图景:E 的自适应重估是 bimodal

把三域的 E 失败模式并起来,机制是同一个:

| regime | gap 状态 | E 重估结果 | E 表现 |
|---|---|---|---|
| 中 L / H15(CSTR L85、混沌 CSTR L50/85)| informative | 浓缩到难点、放大教师信号 | ✅ 大赢(−27% 地板)|
| 锚点 / 易任务(MG M1、CSTR L20H15)| 干涸或 ≈0 | 权重归零、E≈CE-only | ≈ 持平 / 略输 |
| 最难 cell(CSTR L100H30)| 太噪 | 重估出坏权重、首 轮上行 | ❌ 退化早停、锁死坏盆底 |

**E 的自适应重估要么大赢、要么大输,取决于 gap 是否 informative;A 的均匀权重是稳健主力。** 这也解释了为何 E 的方差总是远大于 A(L50H15:E sd 33.7 vs A 1.7)——bimodal 分布。

→ **推论 / 下一步**:动态地板(早期硬地板保浓度、后期软地板防干涸)现在多一个动机——救 L100H30 这类 E 崩(最难 cell 的"太噪"gap)。与 `iterative_distillation_summary.md` §8 里 E-soft 扫描留下的"动态地板"开放问题对接。

---

## 7. 局限

1. **混沌核实 n=5/cell** 偏小;L85 的 CI 虽排除 0 但下沿 −6.8 不宽,加 seeds 更稳。
2. **val≈test 过拟合**:周期/混沌信号下 keep-best-by-val ≈ 挑 test 最优,曲线绝对值偏乐观;但对两臂对称,**相对速度/地板结论稳健**。
3. **"E 崩"阈值**(E>A×1.3)启发式;改阈值 L50H15 崩点数会变,但 L100H30 5/5、L85H15 0/5 的两极结论不变。
4. floor_sweep(K=5)与 chaotic_floor_verify(K=10)的 cell 仅 4 个重叠(L20H15/L50H15/L85H15/L100H30);其余 9 cell 的饱和行为未实测,方向判断基于 H/L regime 外推。
5. 标准 CSTR saturation 仅 n=6,K=15 速度结论 CI 宽。

---

## 8. 对旧结论的修正

| 旧口径(2026-07 报告)| 本轮修正 |
|---|---|
| "E_iter ≈ A_iter(p=0.94)" | 仅**地板高度**口径成立;**速度**上标准 CSTR E 稳健更快(p=0.016/3e-5)|
| "E≈A on delayed CSTR"(基于 L20H15 锚)| **锚点特例**;网格上 regime-dependent——中 L/H15 cell E 真赢(L85 −27% CI 排除 0),最难 cell E 灾难性崩(L100H30)|
| "E 优势是可靠+更快,非更低地板" | 标准 CSTR 上**速度侧证实并量化**;混沌 CSTR 上**连地板都赢**(中 L/H15),但代价是 bimodal 不稳 |
| "MG 负结果 = 不跨域" | 补充机制量化:E/A 首轮进步比 1.70→1.27,底层收敛太快饿死 E 的 gap 加权 |

---

## 9. 复现

```bash
# 跨域收敛速度分析(含逐 cell 分化、混沌汇总、MG 机制对照)
uv run python cstr/archive/analyze_convergence_speed.py

# 混沌 CSTR K=10 饱和核实(生成 chaotic_floor_verify.csv,约 47 min @ MPS)
uv run python cstr/archive/verify_chaotic_iter_floor.py --K 10 --seeds 5
```

数据:
- 本轮新增:`cstr/results/chaotic_floor_verify.csv`、`cstr/results/logs/chaotic_floor_verify.log`
- 复用:`cstr/results/iterative_phase1_{anchor,grid}.csv`、`iterative_{sweep,saturation_sweep}.csv`、`iterative_delayed_summary.csv`、`floor_sweep.csv`、`mackey_glass/results/iterative_mg_sweep*.csv`

## 10. 关联

- 上游 / 修订:`conclusion/iterative_distillation_summary.md`(§3、§7 口径按本文 §8 修正)、`conclusion/iterative_pilot_phase0{,_mg}.md`、`conclusion/iterative_phase1_cstr.md`、`conclusion/floor_study_final_report.md`
- 记忆:`iterative-distillation-status`(2026-08-09 更新)、`cstr-adaptive-weight-E-works`、`cstr-floor-determinants`
- 设计 / 计划:`docs/superpowers/{specs,plans}/2026-07-28-iterative-adaptive-distillation*.md`
