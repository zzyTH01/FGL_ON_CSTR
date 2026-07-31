# 混沌化 CSTR 上的 FGL 探索总结

**日期:** 2026-07-30
**分支:** `feat/cstr-delayed-feedback-stable`(9 commits `246fc3e..2a12a7c`,未合并)
**核心问题:** 能否把强周期性的 CSTR 混沌化?如果能,FGL 类方法在混沌 CSTR 上的增益能否回升,打破周期-1 的"地板效应"?

---

## 0. 背景与动机

项目主线结论:CSTR 单反馈环、周期-1,FGL 增益有限("地板效应");MG(τ=13,混沌)和 Lorenz(ρ=60)上 FGL 有效。一个自然问题:**"地板"是不是单纯的周期性伪影?** 若把 CSTR 做成混沌,FGL 增益是否回升?

## 1. Phase A —— 把 CSTR 混沌化

### 1.1 基础 CSTR 调参做不到(经验 + 理论双重证据)

`generate.py` 是刚性弛豫振荡器,周期性 ≈ 0.95。扫 U/K/flow 三参数:

| 扫描参数 | 周期性范围 |
|---|---|
| U(散热 0.01–0.05) | 0.935–0.976 |
| K(阀门) | 0.936–0.964 |
| flow(6–18 sccm) | 0.936–0.959 |

最低 0.9348。**单一物理参数调不动**——结构原因:固定入口单 CSTR 是有限维 ODE 的稳健极限环。

### 1.2 既有延迟反馈脚本崩溃

`generate_delayed_feedback.py`(mdot 延迟反馈,DDE,理论唯一保证混沌的路线)对所有 τ、所有 A 都崩(CVODES 在点火尖峰发散)。根因:**正反馈 × 无界 × 硬接通**。dual_cstr / forced 也都锁相到 periodicity=1.0。

### 1.3 修复:有界 + 缓启动 + EMA 低通滤波

新建 `cstr/generate_delayed_stable.py`(TDD,14 单元测试),保留 mdot 反馈拓扑,只改控制律:
- `mdot = mdot₀·(1 + s·A_eff·tanh((H₂O[t−τ]−c)/w))` —— tanh 把 mdot 夹在 [mdot₀(1−A), mdot₀(1+A)],根除发散;
- `A_eff` 在 ~50s 内从 0 缓升,消除第 τ 步硬接通;
- **EMA 低通滤波 sensed signal**(β)——关键修复:TDD 的 smoke test 抓到 t=22s 仍崩,根因是 H₂O 尖峰化让 mdot 与尖峰同步剧烈摆动,把点火锐化到 CVODES 容忍度外。降 A 无效(A=0.1 更早崩),**滤波才是解**。

### 1.4 结果:周期↔非周期 τ 过渡曲线被复现 ✅

参数搜索 (sign, A, β) 发现:**负反馈/弱反馈 → 锁相**(周期性 0.96–0.99,无过渡);**正反馈 sign=+1, A=0.9, β=0.03 → 出现清晰过渡**(全稳定,0 崩):

| τ(steps) | 5 | 30 | 40–65 | 70–80 | 100–150 |
|---|---|---|---|---|---|
| periodicity | 0.91 | 0.79 | 0.54–0.56 | 0.79–0.81 | 0.47–0.49 |

periodic → aperiodic(τ40–65)→ 准周期恢复窗(τ70–80)→ 再 aperiodic(τ100–150),典型 DDE 分岔的交替带。**证实了"延迟参数在中间区间出现周期↔非周期过渡"的假设。** 13 个非周期数据集存入 `cstr/data/`。

> Caveat:低周期性 ≠ 严格混沌(未算 Lyapunov;dom_period=2.0s 是指标地板效应)。

## 2. Phase B —— 一次性 FGL(基线对照)

`cstr/run_fgl_delayed.py`:`run_fgl_experiment`(分类/bins,L20/H15,与主线同口径),3 seeds。

| 数据集 | per | baseline | student | Δ% |
|---|---|---|---|---|
| base(period-1) | 0.94 | 127.5 | 112.5 | +11.6%±7.0 |
| τ=30 | 0.79 | 247.5 | 245.3 | +0.9%±1.4 |
| τ=50 | 0.56 | 104.0 | 104.7 | **−0.7%**±1.0 |
| τ=80 | 0.79 | 160.7 | 149.5 | +7.0%±2.6 |
| τ=100 | 0.49 | 172.5 | 160.5 | +6.9%±1.0 |
| τ=150 | 0.47 | 159.3 | 150.9 | +5.2%±3.8 |

**结果混杂,非干净正相关**:5/6 有正增益(含几个 aperiodic),但最 aperiodic 的 τ=50 反而 ~0/略负;无"越 aperiodic → 增益越大"的单调趋势。n=3 功率不足。**强化了项目主线:CSTR 对 FGL 普遍偏难,与周期性无关。**

## 3. Phase C —— 自适应连续蒸馏(正向突破)

`cstr/run_iterative_delayed.py`:`run_iterative_distillation`(E 变体,4 arm,K=5,3 seeds)。init_delta = 相对 round-0(一次性 FGL)student 的降幅。

| 数据集 | per | A_single(一次性FGL) | A_iter(均匀) | E_iter(自适应) | E_iter Δinit |
|---|---|---|---|---|---|
| base | 0.94 | 108.95 | **30.10** | 39.19 | +63.1% |
| τ=50 | 0.56 | 104.10 | 84.57 | **82.87** | +20.4% |
| τ=100 | 0.49 | 160.45 | 145.78 | **144.59** | +9.9% |
| τ=150 | 0.47 | 149.39 | 135.84 | **129.63** | +13.2% |

**两个关键发现:**
1. **E_iter 在所有数据集都把 student MSE 再降 +10%–63%**(对"能否继续降低"的肯定回答)。
2. **算力匹配下,E_iter(自适应)在 3 个 aperiodic 数据集上都赢 A_iter(均匀),且用更少轮数;但在周期 base 上 A_iter 反而赢。** → **自适应-E 的价值专显于混沌数据。** 这是混沌 CSTR 上 FGL 族方法目前最强的正向信号。

> Caveat:n=3 不足,τ50/100 上 E_iter-vs-A_iter 差距小可能噪声;vs A_single 的大降幅部分是"多训几轮"(但 E_iter-vs-A_iter 是算力匹配的,结论干净)。

## 4. Phase D —— 下界(floor)诊断与 L/H 假设

### 4.1 "降幅不均"的真相:不同数据集有不同 floor,E_iter 触及各自的 floor

```
dataset      baseline  A_single  E_iter  A_iter  E_iter/baseline  A_iter/baseline
base          121.6    109.0     39.2    30.1     0.32×            0.25×
τ=50          103.8    104.1     82.9    84.6     0.80×            0.81×
τ=100         172.5    160.4    144.6   145.8     0.84×            0.85×
τ=150         156.8    149.4    129.6   135.8     0.83×            0.87×
```

- **aperiodic 上自适应与均匀都收敛到同一个 ~0.80–0.87× baseline** → 是**数据地板**,不是方法瓶颈。混沌的 H-step 未来本身就难预测。
- **base 降到 0.32× 因为 floor 低 + round-0 留下巨大 headroom**(且 base 上主要是"多训",A_iter < E_iter)。

### 4.2 floor 与 H、L 强相关(核心洞察)

- **H 抬高混沌的 floor**(主因):混沌预测误差随 horizon 指数增长(Lyapunov)。周期数据几乎与 H 无关。这正解释了"混沌 floor 高、周期 floor 低"主要是 H=15 在两类数据上的效应。
- **L 降低 floor,直到覆盖系统记忆/延迟**:项目的 `L+H-1 ≥ τ` 阈值(MG)正是此意。
- **对我们数据的推论**:L=20, H=15 → **L+H-1=34**,但混沌数据集反馈延迟 τ=50/100/150,**34 < τ** → 学生窗口看不到一个完整反馈周期 → 无法重构延迟动力学 → floor 居高。L 对这些数据集的延迟**过短**。周期 base 不受此限(不论窗口都可预测)。

> Caveat:`L+H-1≥τ` 规则来自 MG,属探索性;在 CSTR 上的 1:1 映射是**假设**,需测试。周期性与延迟效应在我们数据中是混杂的。

## 5. 结论

1. **CSTR 可以被混沌化**(正反馈 + 有界 + 缓启动 + EMA 滤波),并复现了 Mackey-Glass 式的 τ 分岔过渡曲线。
2. **一次性 FGL 在混沌 CSTR 上仍然混杂**——aperiodicity 不明显提升 FGL 增益,"地板"不是单纯周期性伪影。
3. **自适应连续蒸馏能可靠降低 student MSE**(所有数据集 +10%–63%),且**自适应权重(E)的优势专显于混沌数据**(matched-compute 下赢均匀迭代)。这是目前混沌 CSTR 上 FGL 族最强的正向信号。
4. **下界是 (L,H) 量,不是数据集固有属性**:H 抬高混沌 floor,L 降到覆盖延迟为止;当前 L=20 对 τ≥50 的混沌数据过短,这很可能是混沌 floor 居高、一次性 FGL 增益被抑制的部分原因。

## 6. 开放问题 / 下一步

| # | 实验 | 目的 |
|---|---|---|
| 1 | **τ=100 上 L×H 扫描**(L∈{15,30,50,80}, H∈{5,15,30}, 3 seeds) | 直接验证 floor 随 H 升、随 L+H-1≳τ 降;把 §4.2 假设变成曲线 |
| 2 | n≥5 + 配对显著性检验 | 把"E_iter > A_iter on chaos"从趋势变结论 |
| 3 | 每个 数据集的 `teacher_mse` | 验证"混沌上教师更弱 → 蒸馏传递少"机制 |
| 4 | 更长 epoch 的 baseline(matched compute) | 找真 floor,隔离"多训"贡献 |
| 5 | τ=50 反常(一次性 FGL 零增益) | 单独查(教师质量?regime-specific transfer?) |
| 6 | 回归模式(`regression=True`) | 与分类模式对照 |

推荐 **#1 + #2** 组合:既测 L/H 主导性(项目核心论点在混沌 CSTR 上的延伸),又把自适应优势坐实。

## 7. 产物索引

**脚本(分支上):**
- `cstr/generate_delayed_stable.py` —— 稳定延迟反馈生成 + τ-sweep + CLI
- `cstr/archive/plot_delayed_stable.py` —— 数据集小多重图
- `cstr/run_fgl_delayed.py` —— 一次性 FGL 驱动
- `cstr/run_iterative_delayed.py` —— 自适应连续蒸馏驱动
- `tests/cstr/test_delayed_stable.py` —— 14 单元测试(全仓 31/31 通过)

**报告/数据(分支上,`cstr/results/` + `cstr/data/`;过程报告已归档 `conclusion/archive/`):**
- `conclusion/archive/delayed_tau_sweep_report.md` + `cstr/results/plots/delayed_tau_sweep_s1_A0.9_b0.03.png` —— 过渡曲线
- `conclusion/archive/fgl_delayed_report.md` + `cstr/results/fgl_delayed_summary.csv` —— 一次性 FGL
- `conclusion/archive/iterative_delayed_report.md` + `cstr/results/iterative_delayed_summary.csv` —— 自适应连续蒸馏
- `data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl` —— 13 个非周期数据集
- `delayed_stable_h2o_panels.png` —— 数据集可视化

**设计文档:** `docs/superpowers/specs/2026-07-29-cstr-delayed-feedback-stable-design.md`、`docs/superpowers/plans/2026-07-29-cstr-delayed-feedback-stable.md`

**复现入口:**
```bash
uv run python cstr/generate_delayed_stable.py --sweep --sign 1 --amplitude 0.9 --filter_beta 0.03 --fine_around 50
uv run python cstr/run_fgl_delayed.py --seeds 3
uv run python cstr/run_iterative_delayed.py --seeds 3 --K 5
uv run pytest tests/cstr/test_delayed_stable.py -v
```

**关联记忆:** `cstr-delayed-feedback-chaos-transition`、`fgl-on-delayed-cstr-mixed`、`iterative-distillation-status`、`cstr-adaptive-weight-E-works`。
