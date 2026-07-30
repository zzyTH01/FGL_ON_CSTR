# 延迟 CSTR 上"MSE 地板成因"最终研究报告

**日期:** 2026-07-30
**分支:** `feat/cstr-delayed-feedback-stable`(并入 main)
**研究问题:** 在延迟反馈(混沌)CSTR 上,**是什么决定了能达到的最低 student MSE?如何才能达到它?**
**一句话回答:** 地板主要由 (L, H) 与数据混沌度决定(可解释 ~72%);连续蒸馏能钻到"数据地板"之下,但**只在预测足够难(H≥15)时**;蒸馏的超参数(α/温度/加权)不决定地板。

> 配套数据表(假设结构化 + 逐条对账)见 [`floor_determinants.md`](floor_determinants.md);原始数据 `cstr/results/floor_sweep.csv`(45 行)、`cstr/results/lyapunov_tau.csv`(13 行)、图 `cstr/results/floor_h{1..4}_*.png`。

---

## 1. 背景与动机

项目两条线各自成熟,本报告把它们焊在一起:

1. **连续自适应蒸馏**(`run_iterative_distillation`,变体 E):在延迟 CSTR 上把 student MSE 再降 +10–63%,但结论是"**蒸馏超参数不起决定性因素**"——E_iter 与 A_iter 在饱和后收敛到同一个数据地板。
2. **延迟反馈 CSTR**:正反馈(s=+1,A=0.9,β=0.3)+ 有界 + 缓启动 + EMA 滤波,复现了 τ 分岔过渡;13 个非周期数据集存于 `cstr/data/`。

**既然蒸馏超参数不决定地板,那谁决定?** 本战役在 τ=100 上系统扫描 (L, H),并同时记录 baseline / 匹配算力收敛 baseline / teacher / FGL-student / A_iter / E_iter 六个量,逐层拆解地板的成因。

**角色定义(理解一切的基础):**
- **学生**:看过去 L 步,预测 H 步之后。L=历史长度,H=预测距离。
- **老师**:窗口偏移 H−1,只预测 1 步——相当于"偷看"紧贴目标之前的近未来。
- **baseline**:学生但不听老师。
- **E_iter(连续蒸馏)**:学生边自学边听老师,迭代精修。
- **MSE** = 预测误差(单位:bin-index²,50 bins);**地板** = 怎么努力也压不破的最低误差。

---

## 2. 实验设计(简)

- **τ=100 深挖网格(13 cells × 3 seeds):** L×H 曲面 `L∈{20,50,100}×H∈{5,15,30}` + 固定 H=15 的 L 切片 `L∈{40,70,85,120}`(横跨 τ=100,临界 L≈86)。
- **每 cell 记录 6 量:** `baseline_mse`(标准训)、`baseline_converged_mse`(epochs=100/patience=10 训够)、`teacher_mse`、`fgl_student_mse`(标准 FGL)、`A_iter_mse`(均匀迭代,归因对照)、`E_iter_mse`(自适应连续蒸馏)。
- **锚点:** τ50、τ150 @ L20H15(横向)。
- **一次性诊断:** Rosenstein 最大 Lyapunov 指数(全部 13 数据集)。
- 设备 M5 Air / MPS,实测每 cell-seed ≈ 150s,全战役 ~2h。

---

## 3. 主要发现

### 3.1 L、H 是地板的绝对主角;但 MG 的"L+H-1≥τ 巧合"不普适

τ=100 上 baseline 地板(按 cell 平均):

| L\H | 5 | 15 | 30 |
|---|---|---|---|
| 20 | 93 | 172 | 222 |
| 50 | 67 | 143 | 201 |
| 100 | 68 | 134 | — |

两条强规律:
- **H 抬高地板(主因):** 固定 L=20,误差 93(H5)→172(H15)→222(H30),预测距离翻 6 倍误差翻 2.4 倍。混沌系统越远越不可预测。
- **L 降低地板,边际递减:** 固定 H=15,误差 172(L20)→143(L50)→134(L100);**过 L≈50 基本饱和**。

**关于"巧合":** 在 MG 上,`L+H-1` 跨过 τ 时误差**断崖式**暴跌 8.5×(相变)。我们在 τ=100 专找这个悬崖——**没有**。固定 H=15 扫 L,误差 172→138→143→134→131,**一路平缓**。结论:**MG 的 `L+H-1≥τ` 是几何特异性巧合,不普适到混沌 CSTR**(对 `chaotic_cstr_fgl_exploration.md` §4.2 假设的直接证伪)。混沌数据上,"加历史"是平滑改善,无阈值开关。

### 3.2 连续蒸馏地板 ≈ 0.76 × baseline(固定比例)

跨 13 cells,`c = E_iter / baseline` 均值 **0.76**(CV=0.21)。蒸馏稳定地把地板降到 baseline 的 ~76%,比例**不随 L/H 系统漂移**。含义:蒸馏是"全场 76 折券",不改变地板的决定因素,只整体下移一个固定折价。把 §4.1 的单点结论(L20H15 ~0.80–0.87×)推广到了整个 L×H 网格。

### 3.3 地板 ↔ teacher 的数学关系:固定 H 下成立,多元 R²=0.72

- **裸拟合**(跨所有点)floor vs teacher_mse:R²=0.12——**假象**。H 的影响淹没了 teacher。
- **固定 H 下**:floor~teacher 的 R²:H=5→0.47,H=15→**0.73**,H=30→0.63。**控制住 H,teacher 质量是地板的强预测器。**
- **多元模型**(R²=**0.72**):

  `log(floor) = 3.83 + 0.016·teacher + 0.035·H − 3.49·(1/L)`

  - H 系数 0.035/样本 ≈ **4.4× Lyapunov λ(0.0079)**:误差随 H 增长比纯轨迹发散更快(离散化 + 模型误差复合)。→ 严格形式 `floor~teacher·exp(λH)`(斜率=λ)**证伪**,但定性结构(随 H 指数增长、固定 H 正比 teacher)**成立**。
  - teacher 系数 +、1/L 系数 −:teacher 越好 / L 越大 → 地板越低,与机制自洽。

### 3.4 蒸馏能压破数据地板——但**只在难 horizon**(regime-dependent)

E_iter vs baseline_converged(匹配算力、训到收敛的 baseline)分层:

| H | E_iter − baseC 均值 | 蒸馏更低占比 | 判读 |
|---|---|---|---|
| 5(易) | **+3.0** | 1/9 | 蒸馏**无益甚至有害** |
| 15(难) | **−20.7** | 18/21 | 蒸馏显著压破地板 |
| 30(最难) | **−10.2** | 6/9 | 蒸馏压破(方差大) |

整体配对 t:E_iter 比 baseC 低 12.8(p=0.0002)。

**机制(印证 `final_conclusions.md` §2.4 信息不对称):**
- H 小:学生历史已能推出 teacher 的"近未来"→ 蒸馏冗余;且 KL 把学生拉向 teacher 的 t+1(学生要 t+H,易任务下二者近重合)→ 添乱。
- H 大:teacher 的 offset=H−1 真正提供了学生历史推不出的近未来 → 净增益。

**干净的自检:** 若蒸馏的赢只是"多训了几轮",它应在所有 H 上都赢。但它**只在难 H 赢、易 H 输**——这正是"老师的信息在起作用、而非堆算力"的指纹。

**与 `iterative_distillation_summary.md` 不冲突:** 那里的"E_iter≈A_iter"比的是**两种蒸馏之间**(权重不决定);这里 H4 比的是**蒸馏 vs 无 teacher baseline**——teacher 信号本身(而非多训)压破了地板。两者互补。

### 3.5 附带:全部 13 数据集确认为真混沌

Lyapunov λ ∈ [0.0018, 0.0108]/样本,**均为正** → 真混沌,非噪声。消除 `chaotic_cstr_fgl_exploration.md` §1.4 的 caveat。且 λ 与 periodicity **不单调**(τ=30 λ 最大但 per=0.79 最高;τ=50 λ 最小但 per=0.56)→"周期性"与"混沌度"是两个独立维度,不应混用。

---

## 4. 如何达到最低 MSE(可操作)

像剥洋葱,三层:

1. **第一层(决定大头,~72%):数据难度。** 主是 H,次是 L。压低地板 → **降 H、提 L(L 到 ~50 即饱和)**。这层应用约束决定,改不了多少。
2. **第二层:把模型训够。** baseline 从 30 轮训到收敛(100 轮),再降一截(如 L20H15:172→148),但触数据地板即止。
3. **第三层(关键):按难度决定是否蒸馏。**
   - **H 大(难,≥15)→ 上连续蒸馏**,再压 13–21% 到 baseline 之下。
   - **H 小(易,≈5)→ 别蒸馏**,直接训 baseline 到收敛(蒸馏反而 +3%、且费算力)。
4. **超参数不决定地板**(α/温度/加权):决定地板的是 **(L, H) + 数据可预测性 + "蒸不蒸馏"这个二分**。

---

## 5. 局限与下一步

- **n=3 seeds**,H=30/H=5 部分格子偏噪(如 6/9 赢、R²=0.47)。关键 cell 补到 5 seeds(接口已留)。
- **val≈test 过拟合**:序列残余周期性 → keep-best-by-val 近似在 test 选优,H4 配对 t 偏乐观。独立 holdout 待做。
- **τ vs periodicity 混杂**:横向 τ50/150 锚同时变两量 → 横向仅"趋势"。**方案 C(去混杂:固定 per 变 τ)** 是把 H1 变因果检验、把 H4 的 regime 结论推广的下一步。
- **MSE 单位** bin-index²,仅相对分析。
- 蒸馏地板多元公式(R²=0.72)为经验拟合,非机理推导;跨数据集外推需验证。

## 6. 复现

```bash
uv run python cstr/lyapunov_delayed.py
uv run python cstr/run_floor_sweep.py --datasets tau100,tau50,tau150 --seeds 3 --K 5
uv run python cstr/analyze_floor.py --deep_label tau100
```

## 7. 关联

- 设计:`docs/superpowers/specs/2026-07-30-cstr-floor-determinants-design.md`
- 计划:`docs/superpowers/plans/2026-07-30-cstr-floor-determinants.md`
- 数据表 / 对账:`conclusion/floor_determinants.md`
- 上游结论:`conclusion/项目汇报总结.md` §6.3/§7.2、`conclusion/chaotic_cstr_fgl_exploration.md`、`conclusion/final_conclusions.md` §2.4
