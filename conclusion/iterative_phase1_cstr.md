# 迭代自适应蒸馏 — CSTR Phase 1 全验证结果(2026-07-28)

> 对应 spec `docs/superpowers/specs/2026-07-28-iterative-adaptive-distillation-design.md` §7 Phase 1。
> 数据:`cstr/results/iterative_phase1_anchor.csv`(锚点 n=10)+ `iterative_phase1_grid.csv`(5×5 × 2 seeds)。
> 分析脚本:`cstr/analyze_iterative.py`。前置 Phase 0 / 饱和分析:`conclusion/iterative_pilot_phase0.md`。

## 设置

四臂 2×2(共享 round-0):A-single / E-single / A-iter(归因对照)/ E-iter。
- 锚点 L=20,H=15,n=10 seeds;网格 5×5(L∈{8,20,35,50,72},H∈{5,15,30,45,60})× 2 seeds。
- epochs=30,round_epochs=15,K=6(按饱和分析取到饱和),α=0.5,T=4,batch=64。
- **口径:每 seed 最小 test MSE**(per-seed-min),CSTR 周期信号下防 val≈test keep-best 过拟合。

## 锚点 L20H15(n=10)—— 迭代 E 在饱和后无优势

|  | per-seed-min test MSE | 中位数 | E<A |
|---|---|---:|---:|
| E_iter | [31.3, 30.2, 29.8, 29.9, 31.4, 36.2, 30.3, 60.1, 31.0, **14.5**] | 30.67 | 5/10 |
| A_iter | [30.0, 30.2, 30.0, 30.2, 30.4, 57.1, 29.7, 30.0, 29.9, 30.6] | 30.10 | — |

- **E_iter vs A_iter:配对 t p=0.94,Wilcoxon p=0.70 —— 无显著差异。**
- E_iter vs E_single p=0.29(迭代未明显胜单趟);A_iter vs A_single p=9e-10(多训练显著有效)。
- ⚠️ E_iter seed9=14.5(低于 ~30 地板)= val≈test keep-best 过拟合,中位数不受影响但提示需独立 holdout。

**解读**:Phase 0 在 K=3 看到的"E_iter 比 A_iter 好 31%"是**收敛速度的早期快照**;到饱和(K=6)、正规 n=10,**两臂到达同一个地板 ~30**。饱和分析的预言在锚点上被确认。

## 网格 5×5 × 2 seeds —— 名义显著但不稳健

**网格级 E_iter vs A_iter(n=50 cell×seed)**:
- E<A 在 **26/50(52%)**;均值降 **+6.6%**,**中位数 0.0%**;配对 t p=0.005。
- 中位数 0% + 52% 胜率 ⇒ 效应**主要由离群驱动,不稳健**。

E_iter 真正明显赢的 cell(均为 A_iter 卡死、E_iter 收敛):

| cell | E_iter | A_iter | rel | 模式 |
|---|---:|---:|---:|---|
| L8H60 | 66.6 | 94.6 | +30% | A 卡死,E 收敛 |
| L20H30 | 43.6 | 60.6 | +29% | A 卡死,E 收敛 |
| L35H30 | 57.4 | 81.9 | +30% | A 卡死,E 收敛 |
| L20H45 | 42.2 | 56.6 | +25% | A 卡死,E 收敛 |
| L8H45 | 44.5 | 56.0 | +17% | A 卡死,E 收敛 |

地板 cell(L≥50 / 大 L+H):两臂同 ~14,E≈A(±1%)。L35H15、L50H30 等 E 略输(−3~6%)。

## 诚实结论

迭代自适应蒸馏在 CSTR Phase 1:**可行,但效果比 Phase 0 暗示的窄。**

1. **锚点 E_iter ≈ A_iter(饱和后同地板)**:在最常研究的 L20H15 上,迭代 E 没有净优势。Phase 0 的正信号 = K=3 早期速度快照 + 过拟合离群。
2. **E_iter 的真实价值 = 中等难度 cell 上"更可靠地收敛"**:A_iter 偶尔卡死(MSE 56~95),E_iter 能收敛到 44~67。这是**可靠性,不是更低地板**——与饱和分析口径一致。
3. 网格名义 p=0.005 但中位数 0% / 52% 胜率,**主要由 A_iter 的卡死 + 少数过拟合离群(seed9=14.5、L72H5=0.16 等)驱动,非稳健**。
4. 过拟合离群反复出现(val≈test + keep-best),**Phase 1 未加独立 holdout 是已知缺口**。

## 与跨域结论的关系

MG τ=13 上 E_iter 净效应为负(见 `iterative_pilot_phase0_mg.md`)。CSTR Phase 1 进一步收紧:E_iter 即便在 CSTR 上,**也只有"可靠性"这点窄价值,没有"更低地板"或"锚点级"优势**。两域合起来指向同一机制:**E 的零地板让蒸馏池随收敛收缩,易任务/饱和后 E≈A 甚至吃亏;只在 A_iter 自身不稳(卡死)的中等难度 cell 上,E 的稳定收敛才体现为净增益。**

## 建议

- **不建议**把"迭代自适应蒸馏"作为 CSTR 上的胜利结论宣传——锚点不显著、网格效应不稳健。
- 可写进结论的**窄事实**:在中等难度 cell(L8-35、H30-60 区段)上,E_iter 比 A_iter 更不容易卡死,收敛更稳。
- 真正值得做的:独立 holdout 复核网格 p=0.005 在剔除过拟合离群后是否仍成立;以及温和地板版(gap=0→0.2)能否把"可靠性"扩展到锚点/地板 cell。

## 局限

- 网格仅 2 seeds(锚点 n=10);网格 cell 的 p 值多数不显著(n=2 无统计力)。
- 无独立 holdout;离群(MSE<10)未剔除,可能抬高网格均值显著性。
- K=6 已饱和;更高 K 不会改变"同地板"结论。
