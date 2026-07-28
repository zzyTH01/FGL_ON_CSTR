# 迭代(暖启动)自适应蒸馏 —— 实验总总结(2026-07-28)

> 本文汇总"迭代自适应蒸馏"这条研究线的全部实验。详细分报告见文末索引。设计文档:`docs/superpowers/specs/2026-07-28-iterative-adaptive-distillation(-mg)-design.md`。

## 1. 方法一句话

把单趟自适应蒸馏(变体 E:teacher−student MSE 差距驱动的逐样本加权,零地板放大到 [0,4])扩展为**暖启动迭代**:每轮用上一轮 student 重估权重 → 暖启动 → 再蒸馏,直到 val MSE 收敛/退化/K 轮兜底,keep-best-by-val。

**2×2 四臂因子设计**(共享 round-0,归因分解):

|  | single(round-0 后停) | iter(round-0 后暖启 ≤K 轮) |
|--|--|--|
| **uniform (A)** | A-single = 标准 FGL | A_iter = 纯多训(权重恒均匀,归因对照) |
| **adaptive E** | E-single(暖启 1 轮) | **E-iter**(每轮重估 E 权重,新方法) |

`E_iter vs A_iter` = 迭代权重精修的**净效应**(排除"只是多训")。实现:`fgl_common.run_iterative_distillation`。

## 2. 实验清单与结论一览

| 实验 | 配置 | 核心结果 | 结论 |
|------|------|----------|------|
| CSTR Phase 0 | 3 典型点 × 3 seeds,K=3 | E_iter vs A_iter −31~35%(K=3 时点) | 概念成立,放行 |
| CSTR 饱和 | 同 2 点,K=15 | E_iter 收敛 ~30(降 82%),A_iter 同地板 | **修正口径**:E 优势是可靠+更快,非更低地板 |
| **CSTR Phase 1** | 锚点 n=10 + 5×5×2,K=6 | 锚点 p=0.94 无差异;网格 52% 胜/中位数 0%/p=0.005 名义 | **可行但窄**:锚点不显著,网格效应不稳健 |
| MG Phase 0 | τ=13,3 跨阈值点 × 3 seeds,K=5 | E_iter vs A_iter:M1 −24%、M2 +13%、M3 ≈ | **负结果**:不跨域泛化 |
| 可视化 | CSTR L20H15 / MG L4H10,逐轮 | 红点(预测)逐轮贴向灰线(真实) | 收敛形态直观可视 |

## 3. CSTR 线 —— 从乐观到收紧

**Phase 0**(K=3)看似不错:E_iter 在有空间的点比 A_iter 低 31-35%。但**饱和分析(K=15)**揭穿:让 A_iter 也跑够,E_iter 到 ~30,A_iter 收敛后也到 ~30;E_iter 的真正优点是**三 seed 全收敛**(A_iter 三 seed 两个卡在 57)。即"可靠+更快到地板,非更低地板"。

**Phase 1**(n=10 + 网格,K=6 饱和)正式收紧:
- 锚点 L20H15(n=10):E_iter vs A_iter **配对 t p=0.94、Wilcoxon p=0.70**——**无显著差异**。Phase 0 的 +31% 是早期速度快照。
- 网格 5×5×2:E<A **26/50(52%)**,均值降 +6.6% 但**中位数 0.0%**,配对 t p=0.005——**名义显著但由离群 + A_iter 卡死驱动,不稳健**。
- 唯一站得住的窄事实:**中等难度 cell(L8-35、H30-60)上 E_iter 比 A_iter 更不容易卡死**(A_iter 偶尔停在 56-95,E_iter 收敛到 44-67)。

详见 `conclusion/iterative_pilot_phase0.md`(Phase 0 + 饱和)、`conclusion/iterative_phase1_cstr.md`(Phase 1)。

## 4. MG 线 —— 负结果

MG τ=13(倍周期分岔,FGL 本就强 +59~78%)上 E_iter 的净效应为负/零:M1(阈值下)−24% 且 1/3 seed 不收敛;M2(阈值)略好 +13%;M3(过阈值)≈。**与 CSTR 相反**:CSTR 上 E_iter 是"更可靠"的那个,MG 上 E_iter 反而**更不可靠**。**方法不跨域泛化。** 详见 `conclusion/iterative_pilot_phase0_mg.md`。

## 5. 机制洞察(解释两域分歧)

> E 的零地板让"可蒸馏样本池"随学生进步而收缩。学生越强,gap 越多归零,E 蒸馏的样本越少 —— E 会**自己关水龙头**。

- **难任务(CSTR,baseline ~120)**:弱点持续存在 → E 每轮都找得到 gap 瞄准 → 浓缩奏效(但饱和后仍同地板)。
- **易任务(MG,baseline 低)**:学生很快收敛 → gap 迅速消失 → E 的蒸馏池**早早干涸** → 后期 E≈CE-only,丢掉 A 持续供给的老师信号 → E吃亏。
- 即用户点出的"**A 客观上给学生更多信息**"在收敛后期成立:A 全样本恒均匀,E 跳过已解决样本。
- 推论:放宽零地板(gap=0→权重 0.2,温和变体 C)可能在易任务上恢复 E 的有效性。**未验证。**

## 5.5 E-soft / w_floor 扫描:浓度 vs 稳定的直接权衡(无免费午餐)

基于"硬零地板让 E 干涸"的诊断(§5),实现 **E-soft 变体**(sigmoid 软地板,`w_floor` 可调),在两域验证"把零地板放宽"能否救回 E。

**MG w_floor 扫描**(3 点 × 3 seeds,E_iter per-seed-min 中位数,方括号为 per-seed 范围):

| 点 | A_iter | E(wf=0) | wf=0.05 | wf=0.1 | wf=0.15 | wf=0.2 |
|---|---:|---:|---:|---:|---:|---:|
| M1 L4H7 | 0.74 | 0.92 ⟂[.57,**5.06**] | 0.77 [.58,.93] | 0.75 | 0.75 | 0.75 |
| M2 L4H10 | 0.60 | **0.52** | 0.60 | 0.67 | 0.65 | 0.66 |
| M3 L13H7 | 0.28 | 0.29 | 0.30 | 0.30 | 0.30 | 0.30 |

- **M1 崩溃**:任何非零地板(wf≥0.05)都修好——5.06 → <0.93,三 seed 全收敛。
- **M2 浓度**:任何非零地板都稀释——E(wf=0)0.52 最好,软地板全输。
- **直接冲突,无甜点**:wf=0 崩 M1;wf>0 输 M2。wf=0.05 是"最不坏"的软地板(修好 M1、只略输 M2),但仍不如 E 在 M2。

**CSTR E-soft(wf=0.2)**(3 点 × 3 seeds,epochs=30/K=6):

| 点 | A_iter | E-soft | E-soft vs A |
|---|---:|---:|---:|
| L20H15 锚点 | 30.0 | 29.9 | +0.4%(中性) |
| L8H30 中难度 | 30.4 | 34.6 | **−13.8%(更差)** |
| L72H15 地板 | 14.1 | 14.1 | ≈ |

锚点/地板中性;**中难度 cell 反而更差**——软地板稀释了 E 在这里的可靠性优势(E 的唯一 CSTR 价值)。E-soft 也没修好 CSTR 的 per-seed stall([29.9, **56.7**, 29.9]),因为那是训练动态卡死、非"干涸"。

**综合结论**:`w_floor` 是**浓度/稳定的旋钮,无免费午餐**。MG 的"干涸崩溃"(易任务)与 CSTR/M2 的"浓度优势"(有空间)是**零地板机制的两面**,静态地板无法兼得 → 暗示真正的解法是**动态地板**(早期硬地板保浓度、后期软地板防干涸),这是一个具体的、可验证的下一步 idea。数据见 `mackey_glass/results/iterative_mg_sweep_E-soft_wf*.csv`、`cstr/results/iterative_sweep_E-soft_wf0.2.csv`。

## 6. 过拟合警示(贯穿所有实验)

CSTR/MG 都属周期性较强信号 → **val≈test**(相邻窗口分布近乎相同)→ keep-best-by-val 实际等同在 test 上挑最优 → 偶发**低于地板的离群**(CSTR L8H30 seed=1.4、锚点 seed9=14.5、MG L72H5=0.16)。**对策:per-seed 中位数**(对离群鲁棒);**独立 holdout** 留待更严格验证(本批未做,是已知缺口)。

## 7. 总体结论

迭代(暖启动)自适应蒸馏:**工程上可行、机制上清晰,但科学收益窄且不稳健。**

1. **锚点不显著**:CSTR L20H15 n=10 上 E_iter ≈ A_iter(p=0.94)。
2. **网格名义显著但不稳健**:p=0.005 由离群 + A_iter 卡死驱动,中位数 0%。
3. **唯一窄价值**:中等难度 cell 上 E_iter 比 A_iter 更不容易卡死(可靠性)。
4. **不跨域**:MG τ=13 上净效应为负/零。
5. **机制**:E 的零地板让蒸馏池随收敛收缩,易任务/饱和后吃亏。

**不应作为 CSTR/MG 的胜利结论宣传。** 它是一个有清晰机制解释、但收益窄于预期的探索方向。

## 8. 开放问题 / 下一步

1. **独立 holdout 复核**:剔除 val≈test 过拟合后,网格 p=0.005 是否仍成立?
2. **温和地板版**(gap=0→0.2,W_MAX=2):能否把可靠性优势扩展到锚点/地板 cell、并恢复 MG?
3. **Lorenz-63**:强混沌、baseline 不触地板,是 E 机制可能表现不同的第三域(未测)。
4. **诊断 gap 分布**:对比 CSTR/MG 的 teacher−student MSE gap 稳定性,直接验证"MG 的 gap 逐轮漂移更剧烈"假说。

## 9. 索引:分报告与代码

| 内容 | 文件 |
|------|------|
| CSTR Phase 0 + 饱和分析 | `conclusion/iterative_pilot_phase0.md` |
| CSTR Phase 1 全验证 | `conclusion/iterative_phase1_cstr.md` |
| MG Phase 0(负结果) | `conclusion/iterative_pilot_phase0_mg.md` |
| 方法实现 | `fgl_common/training.py::run_iterative_distillation`(+ `_iterate_student` / `_compute_arm_weights` / `_should_stop`) |
| CSTR sweep / 分析 / 可视化 | `cstr/sweep_iterative.py`、`cstr/analyze_iterative.py`、`cstr/plot_iterative_rounds.py` |
| MG sweep / 可视化 | `mackey_glass/sweep_iterative.py`、`mackey_glass/plot_iterative_rounds.py` |
| 测试 | `tests/fgl_common/test_iterative_distillation.py`(15 用例) |
