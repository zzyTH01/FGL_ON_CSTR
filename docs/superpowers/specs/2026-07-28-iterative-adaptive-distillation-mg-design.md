# 迭代自适应蒸馏 — MG 数据集设计文档(Phase 0 试点)

> 日期：2026-07-28
> 状态：设计已与用户确认，待写实现计划
> 范围：Mackey-Glass τ=13,Phase 0 试点(3 典型点 × 3 seeds)
> 前置：CSTR 版方法已实现并验证(`docs/superpowers/specs/2026-07-28-iterative-adaptive-distillation-design.md` + `fgl_common.run_iterative_distillation`)。本 spec **只定义 MG 上的实验设计**,方法本体零改动。

---

## 0. 目标与驱动问题

把已在 CSTR 上实现的**迭代(暖启动)自适应蒸馏**方法,原样移植到 Mackey-Glass τ=13,做 Phase 0 试点。

**驱动问题**(CSTR 结论的外推):
1. CSTR(周期 1)上,迭代 E 权重"可靠 + 更快地把 student 推到离散化地板,而非更低地板"。在 MG τ=13(倍周期分岔、FGL 本就强 +59~78%)上,**迭代能否在已经很强的 FGL 之上再加分**?
2. MG 的成败由 **L+H−1 vs τ 阈值**主导。迭代蒸馏的增益**是否也随阈值位置缩放**(阈值下有效、过阈值失效甚至有害)?
3. 过阈值处 FGL 强负(−54%,与 CSTR 地板处的近中性不同):**迭代会放大伤害,还是 keep-best/停止规则能护住**?

---

## 1. 完全复用(零改动)

- `fgl_common.run_iterative_distillation` / `_iterate_student` / `_compute_arm_weights` / `_should_stop`。
- 四臂 2×2 因子设计(共享 round-0):A-single / E-single / A-iter / E-iter。
- 停止规则(val 上 stall/退化/cap,keep-best-by-val)、逐轮 `mse_curve_val`/`mse_curve_test` 记录。
- 理由:MG 数据格式与 CSTR 完全一致——`generate_mg_data` 返回 `(N, 2)` 张量,两列均为序列值(自回归),`run_iterative_distillation(data, ...)` 直接消费。

---

## 2. MG 特有设计

### 2.1 数据

`generate_mg_data(tau=13, n_points=10000)`(`mackey_glass/run.py` 已有,jitcdde 生成)。每次 sweep 生成一次、所有 cell 共享(同 CSTR 的数据共享口径)。

### 2.2 三典型点 —— 跨 L+H−1 vs τ=13 阈值

MG 的成败轴是阈值(CSTR 是地板),故按阈值位置分档取点(数据来自既有 MG τ=13 L×H 扫描):

| 点 | L | H | L+H−1 | 位置 | 已知 FGL Δ | base MSE | 角色 |
|---|---|---|---:|---|---:|---:|---|
| **M1** | 4 | 7 | 10 | <τ | +41% | 18.3 | 阈值下,FGL 有效,有空间 |
| **M2** | 4 | 10 | 13 | =τ | +59% | 20.2 | 阈值上(峰值),最大空间 |
| **M3** | 13 | 7 | 19 | >τ | **−54%** | 0.56 | 过阈值,FGL 有害(天花板/有害检验) |

> **M3 与 CSTR P3 的关键差异**:CSTR 地板处 FGL≈0(中性),MG 过阈值处 FGL **强负**(−54%)。M3 检验迭代会否放大伤害、keep-best/停止规则能否护住。

### 2.3 过拟合防护(应用 CSTR 教训)

CSTR 饱和分析发现:周期信号下 val≈test,keep-best-by-val 等同挑 test 最优 → 单 seed 异常(1.4)。MG τ=13 亦周期性较强,同一风险。

**试点口径**:**全曲线 + per-seed 中位数**,不靠 keep-best 单点。
- sweep 已把每轮 `mse_curve_test` 写进 CSV(每 cell×seed×arm)。
- 分析时取**每 seed 的最小 test MSE**,再取跨 seed **中位数**(对异常 seed 鲁棒)。
- 报告完整逐轮曲线,标注异常 seed(若有 1.4 式离群)。
- 独立 holdout 留到 Phase 1(更严格的发表级防护)。

### 2.4 超参(对齐 MG 既有实验)

MG 既有实验用 epochs=50 / batch=128(见 `mackey_glass/run.py` 默认),故:
- epochs=50,round_epochs=20,K=5,n_seeds=3,α=0.5,T=4,bins=50,batch_size=128。

---

## 3. 实现

| 文件 | 改动 |
|------|------|
| `mackey_glass/sweep_iterative.py`(新) | 镜像 `cstr/sweep_iterative.py`,仅把 `_load()` 换成 `generate_mg_data(tau=13)`;默认 `--cells "4,7;4,10;13,7"`;CSV + 逐轮曲线图同 CSTR。 |
| `mackey_glass/run.py` | 加实验开关 `iterative_distill`(默认 `enabled=False`)+ CLI `--round_epochs`/`--K`(MG run.py 现在没有);薄包装调用 `run_iterative_distillation` + n-seed 汇总。 |
| `fgl_common` | **不动**。 |

复用 CSTR 的 `sweep_iterative.py` 结构(sweep 循环、CSV、`_plot_curves` 原样照搬);两脚本仅数据来源与默认 cells 不同。> 注:两域 sweep 脚本约 90% 代码相同,后续可提取共享 `run_iterative_sweep` 到 fgl_common;本试点不做(避免动已验证代码),留作未来清理。

---

## 4. 评估口径与放行门

### 4.1 四臂对比(同 CSTR)

每点报 A-single / E-single / A-iter / E_iter 的 student MSE(per-seed 中位数),及三个归因对比:
- E-iter vs A-iter(净迭代效应,排除"多训")
- E-iter vs E-single(迭代 > 一发?)
- A-iter vs A-single(训练量成分)

### 4.2 放行门(按阈值位置分段)

- **M1/M2(有空间)**:E-iter ≤ E-single **且** E-iter ≤ A-iter(净迭代效应为正);逐轮曲线单调降。
- **M3(过阈值,FGL 有害)**:E-iter **不显著恶化**(keep-best/停止规则护住,不放大 −54% 的伤害);逐轮曲线不暴增。
- **过拟合护栏**:全曲线 + per-seed 中位数下,无 1.4 式离群主导结论;若某 seed 离群,按中位数口径报并标注。
- **放行 → MG Phase 1**(锚点 n=10 + 5×5 网格 + 独立 holdout)。

### 4.3 结论分级

- ✅ **概念成立(跨域泛化)**:M1/M2 上迭代有净正效应,M3 不恶化。
- ⚠️ **阈值主导**:迭代只在阈值下有效、过阈值失效(与 FGL 本身同律)——仍是有价值的跨域结论。
- ⚠️ **负结果**:M1/M2 上 E-iter ≈ A-iter(迭代无净贡献,只是多训)——与方法在 CSTR 饱和后的发现一致,如实记录。

---

## 5. 默认参数表

| 参数 | 值 | 说明 |
|------|------|------|
| τ | 13 | FGL 甜品区(倍周期分岔) |
| cells | (4,7)(4,10)(13,7) | 跨阈值三点 |
| α, T | 0.5, 4 | 与 CSTR / MG 既有实验一致 |
| bins | 50 | |
| epochs / round_epochs / K | 50 / 20 / 5 | 对齐 MG epochs=50 |
| batch_size | 128 | MG 既有默认 |
| n_seeds | 3 | 试点 |

---

## 6. 不在本 spec 内(未来工作)

- MG Phase 1(锚点 n=10 显著性 + 5×5 网格 + 独立 holdout)。
- τ 扫描(10/13/17/23/30):迭代增益随混沌强度的变化。
- Lorenz-63 移植(强混沌,第三域)。
- 提取共享 `run_iterative_sweep` 到 fgl_common(消除 CSTR/MG 两 sweep 脚本的重复)。
