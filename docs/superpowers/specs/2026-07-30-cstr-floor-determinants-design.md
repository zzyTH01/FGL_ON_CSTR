# 延迟 CSTR 上"MSE 地板成因"战役设计

**日期:** 2026-07-30
**分支:** `feat/cstr-delayed-feedback-stable`
**前置工作:** 连续自适应蒸馏(已证明蒸馏超参数 α/T/加权不决定地板)+ 延迟反馈 CSTR 数据集(已复现 τ 分岔过渡)
**核心问题:** 是什么决定了能达到的最低 student MSE?如何达到它?

---

## 0. 动机

两条研究线已各自成熟:
1. **连续自适应蒸馏**(`run_iterative_distillation`,E 变体):在延迟 CSTR 上把 student MSE 再降 +10–63%,但结论是"蒸馏超参数不起决定性因素"——E_iter 与 A_iter 在饱和后收敛到**同一个数据地板**(~0.80–0.87× baseline)。
2. **延迟反馈 CSTR**:正反馈(s=+1,A=0.9,β=0.03)+ 有界 + 缓启动 + EMA 滤波,复现了 τ 分岔过渡;13 个非周期数据集存于 `cstr/data/`。

把两者结合:既然蒸馏超参数不决定地板,那**什么决定?** 本战役在延迟 CSTR 上系统检验 (L, H, teacher) 对"能达到的最低 MSE"的决定作用,并给出"如何达到最低 MSE"的可操作结论。

## 1. 科学问题 → 四个可证伪假设

| # | 假设 | 数学形式 | 证伪条件 |
|---|---|---|---|
| **H1** | baseline 地板随 H 升、随 L 降;固定 H 扫 L 时在 `L+H-1 ≈ τ_data` 处有相变(MG 的 `L+H-1>τ` 巧合在延迟 CSTR 成立) | H=15 列:baseline_mse 在 L+H-1≈100(τ_data=100)处骤降;临界 L≈86 | 无骤变,或骤变位置与 τ_data 无关 |
| **H2** | 连续蒸馏地板 = c × baseline 地板,c 在 (L,H) 上**近似常数**(蒸馏只给固定比例降幅,不改变地板的决定因素) | `E_iter_mse / baseline_mse ≈ const`,跨 13 cells 的 c 变异系数小 | c 随 L/H 系统性漂移 |
| **H3** | 地板 ∝ teacher_mse(1 步可预测性);H 步地板 = teacher_mse × exp(λ·H)(混沌 Lyapunov 标定) | `log(floor) vs H` 近线性且斜率≈λ;`floor vs teacher_mse` 强相关(高 R²) | 既无线性也非 exp 关系 |

> H3 的"floor"**同时**在两个定义上检验:`baseline_converged_mse`(任务难度地板)与 `E_iter_mse`(连续蒸馏地板);二者若都跟踪 teacher_mse 则结论更强。
| **H4** | 连续蒸馏地板 ≈ 匹配算力的 baseline 地板 → **最好的 MSE 就是数据地板,蒸馏压不破** | `E_iter_mse ≈ baseline_converged_mse`(配对接近,差值在噪声内) | E_iter 显著低于收敛 baseline |

**H4 直接回答"如何达到最好 MSE":**
- 若 H4 成立 → "把 baseline 训到收敛即可达到最低 MSE,蒸馏不额外加分";
- 若 H4 不成立 → "连续蒸馏确实压破数据地板,是达到最低 MSE 的手段"。

## 2. 实验设计

### 2.1 τ=100 深挖网格(13 cells,3 seeds)

- **L×H 曲面:** `L ∈ {20, 50, 100} × H ∈ {5, 15, 30}` = 9 cells
- **L+H-1≥τ 切片(固定 H=15):** 补 `L ∈ {40, 70, 85, 120}` → H=15 列成 `L ∈ {20, 40, 50, 70, 85, 100, 120}` 共 7 点,横跨 τ_data=100(理论临界 L≈86)
- 关键 cell(H1 相变附近 L=85/100、H4 判决点)后补到 5 seeds

### 2.2 每个 cell 记录的量(一张主 CSV)

字段:`dataset, tau, periodicity, L, H, LplusH_minus_1, seed, baseline_mse, baseline_converged_mse, teacher_mse, fgl_student_mse, A_iter_mse, E_iter_mse`

来源:
- `baseline_mse` / `fgl_student_mse` / `teacher_mse` ← `run_fgl_experiment`(返回字典已含 `teacher`)
- `A_iter_mse` / `E_iter_mse` ← `run_iterative_distillation`(返回字典已含 `teacher_mse`)
- `baseline_converged_mse` ← **新增** `run_baseline_converged`(只训 baseline 模型,epochs=100、patience=10,匹配算力找真地板)

### 2.3 锚点(横向)

τ50、τ150 @ L20H15,3 seeds,记录同样的全部量。用于看 floor/teacher 随混沌度的**趋势**(不作因果,见 §5)。

### 2.4 一次性诊断

τ=100 的最大 Lyapunov 指数 λ(Rosenstein 估计);顺手给全部 13 个延迟数据集的 λ,补上"低周期性≠严格混沌,未算 Lyapunov"的缺口。

## 3. 代码改动(最小,向后兼容)

| 文件 | 改动 |
|---|---|
| `fgl_common/training.py` | 加 `run_baseline_converged(data, L, H, epochs=100, patience=10, num_bins=50, ...)`——单 baseline 模型,返回 `{"baseline_mse": ...}`。不改现有签名。 |
| **新** `cstr/run_floor_sweep.py` | 主驱动:遍历 (dataset × L × H × seed),调 `run_fgl_experiment` + `run_iterative_distillation` + `run_baseline_converged`,写 `cstr/results/floor_sweep.csv`。CLI:`--datasets tau100/tau50/tau150/base`、`--anchors`、`--seeds`、`--K`。 |
| **新** `cstr/lyapunov_delayed.py` | Rosenstein 最大 Lyapunov 指数(纯 numpy/scipy,无新依赖)+ 对全部 `data_delayed_stable_h2o_tau*_*.pkl` 批量估计,写 `cstr/results/lyapunov_tau.csv`。 |
| **新** `cstr/analyze_floor.py` | 出四假设的全部图表 + 拟合(R²、斜率、配对 t、变异系数),写 `conclusion/floor_determinants.md`。 |
| `cstr/run.py` | `EXPERIMENTS` 加 `floor_sweep`(默认 `enabled=False`,显式开)。 |

复用:`fgl_common/sweep.py` 的 `run_lh_sweep` 不直接用(它只跑单 run_fn;本战役需 FGL+迭代+收敛 baseline 三路同 cell),所以 `run_floor_sweep.py` 自带循环,但 CSV/绘图风格与之保持一致。

## 4. 分析与交付物

### 4.1 图表(对应四假设)
- **H1:** `baseline_mse` 的 L×H 热力图;H=15 列 `baseline_mse vs L+H-1` 曲线(标注 τ_data=100,看骤降)
- **H2:** `c = E_iter_mse / baseline_mse` 随 (L,H) 的分布(箱线/散点),报变异系数
- **H3:** `floor vs teacher_mse` 散点 + 线性/幂律拟合(R²);`log(floor) vs H` 拟合斜率 vs λ
- **H4:** `E_iter_mse vs baseline_converged_mse` 配对散点(对角线参考)+ 配对 t

### 4.2 文字结论
`conclusion/floor_determinants.md`:四假设逐条判决(成立/证伪/部分)+ "如何达到最低 MSE"的可操作结论 + 与 `项目汇报总结.md` §6.3/§7.2 的衔接。

### 4.3 数据
`cstr/results/floor_sweep.csv`(主矩阵)、`cstr/results/lyapunov_tau.csv`、各图 `.png`。

## 5. 风险 / 混杂 / 缓解

| 风险 | 缓解 |
|---|---|
| **τ vs periodicity 混杂**(§4.2 caveat) | τ=100 单点深挖时 τ 固定,混杂**不发作**;横向锚点 τ50/150 同时变 τ 和 periodicity → 横向结论仅作"趋势",不作因果。真因果分离(固定 periodicity 变 τ 的去混杂数据)留作方案 C 后续。 |
| **val≈test 过拟合**(周期信号老问题,迭代蒸馏 §6 已记) | 报 per-seed 中位数 + 3 seeds ±std;关键 cell 补 5 seeds;H4 判决用配对 t 而非单点。 |
| **MSE 单位** | 全是 bin-index²(50 bins 离散化),与项目口径一致;只做相对/回归分析,不跨口径比较绝对值。 |
| **L 过大窗内样本不足 / RNN 输入过长** | N=6000,L≤120 → 窗口数 ≥5864,充足;L=120 的 RNN 输入仍小(2 层 RNN,hidden 128),实测 ~1.5× L20 耗时,可接受。 |

## 6. 时间预算

实测(M5 Air,MPS):FGL run 21.4s,迭代 run(K=5)83.9s(@L20H15)。大 L ~1.5×。
每 cell/seed ≈ FGL(≈25s)+ 迭代(≈100s)+ 收敛 baseline(≈20s) ≈ 145s。
13 cells × 3 seeds ≈ 1.6h;锚点 ~15min;Lyapunov/分析可忽略。**总计 ≈ 2 小时。**

## 7. 成功标准

- 四假设各有明确判决(非"趋势暧昧")。
- 给出"如何达到最低 MSE"的可操作结论(H4 的二选一)。
- 主 CSV 可复现、字段完整(含 teacher_mse),为方案 C(去混杂)留好接口。

## 8. 复现入口(实施后)

```bash
uv run python cstr/run_floor_sweep.py --datasets tau100 --seeds 3 --K 5
uv run python cstr/run_floor_sweep.py --anchors          # tau50/tau150 @ L20H15
uv run python cstr/lyapunov_delayed.py                   # 全 13 数据集 λ
uv run python cstr/analyze_floor.py                      # 出图 + 结论
```
