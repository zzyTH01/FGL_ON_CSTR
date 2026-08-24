# Mackey-Glass 结果索引

本文件是 `mackey_glass/results/` 下全部结果文件的溯源与读法索引:每个文件对应哪个实验、由哪个程序产生、实验条件是什么、怎么看。MG 域研究 FGL 的**阈值效应**——当联合信息窗口 `L+H−1 ≥ τ`(τ=13)时教师近未来信息不再独占,FGL 由正转负。结论合集见 `conclusion/mackey_glass_tests.md`,几何检验详见 `conclusion/geometry_test_full_report.md` / `geometry_test2_full_report.md`,迭代蒸馏 MG 负结果见 `conclusion/iterative_pilot_phase0_mg.md`。

**目录布局速览**:根 = 12 个 csv/json 结果表;`plots/` = 17 张 PNG;`logs/` = 5 个运行日志。数据集:`../data/data.pkl`(τ=13,10000 点)。

---

## 速查总表

| 文件 | 一句话说明 | 分组 |
|---|---|---|
| `mg_lh_sweep.csv` | τ=13 主线 L×H 网格(25 配置×3 seeds) | 一 |
| `tau_sweep_results.json` | τ 扫描:FGL Δ 与 Lyapunov/周期性逐 τ 记录 | 一 |
| `threshold_quick_results.csv` | 阈值效应快速验证(L=9→10 符号反转) | 二 |
| `l_threshold_results.csv` | 固定 H 扫 L —— 干净因果检验 | 二 |
| `h_threshold_results.csv` | 固定 L 扫 H —— 对称性检验(U 型) | 二 |
| `geometry_test_results.csv` | 几何检验第一轮:(L,H) 大幅平移看 τ_peak | 三 |
| `geometry_test2_results.csv` | 几何检验第二轮:非对角几何配置 | 三 |
| `iterative_mg_sweep.csv` | 迭代蒸馏 Phase 0 试点(E 硬零地板) | 四 |
| `iterative_mg_sweep_E-soft.csv` | E-soft 软地板(wf 默认 0.2) | 四 |
| `iterative_mg_sweep_E-soft_wf{0.05,0.1,0.15}.csv` ×3 | E-soft 地板参数扫描 | 四 |
| `iterative_distill.csv` | run.py `iterative_distill` 框架输出(MG 保持 off) | 四 |
| `plots/mg_lh_sweep.png` | 主线三联热力图 | 一 |
| `plots/tau_sweep.png` | τ 扫描曲线图 | 一 |
| `plots/mg_tau_overview.png` | 各 τ 序列概览小倍数图 | 五 |
| `plots/mg_phase_portraits.png` | 各 τ 相空间肖像 | 五 |
| `plots/mg_summary.png` | 数据集汇总面板 | 五 |
| `plots/mg_autocorr_overlay.png` | 自相关函数叠加对比 | 五 |
| `plots/cstr_vs_mg.png` | CSTR vs MG 动力学对照 | 五 |
| `plots/threshold_figures*` 见 `h_threshold_figures.png` 等 | 阈值检验配图 | 二 |
| `plots/fig1_absolute_tau.png` | 绝对 τ 坐标下的阈值位置 | 三 |
| `plots/fig2_relative_offset.png` | 相对偏移坐标下的阈值位置 | 三 |
| `plots/geometry_test2_figure.png` | 几何检验二配图 | 三 |
| `plots/h_threshold_figures.png` | H 扫描 U 型 + 分段统计 | 二 |
| `plots/iterative_mg_curves*.png` ×5 | 迭代蒸馏逐轮 val/test MSE 曲线 | 四 |
| `plots/iterative_mg_round_predictions.png` | 逐轮预测 vs 真实可视化 | 四 |

---

## 一、主线扫描

FGL 在 MG 上的总体网格与逐 τ 表现。核心结论:最佳增益 **+78.5%(L=9,H=5)**,正向配置约六成。

#### mg_lh_sweep.csv
- **实验归属**: MG τ=13 L×H 主线扫描
- **产生程序**: `mackey_glass/run.py -e lh_sweep`(经 `fgl_common/sweep.py`;早期版本为 `archive/mg_lh_sweep.py`)
- **关联数据**: `../data/data.pkl`
- **实验条件**: L=H∈{4,7,10,13,16},seeds {0,1,2},α=0.5,T=4,epochs=30,bins=50
- **基本类型**: 每行一个 (L,H,seed):baseline/teacher/student MSE + fgl_delta
- **读表指南**: 按 (L,H) 聚合看 fgl_delta 均值;正值区集中在 L+H−1<13 的左上角;报告版见同目录历史 `*_report.md`(已随整理移至 conclusion/archive 的兄弟文档)

#### plots/mg_lh_sweep.png
- **基本类型**: 三联热力图(FGL Δ% / 绝对改进 / baseline MSE)
- **读图指南**: 绿=蒸馏有益;沿 L 增大方向 Δ% 由正翻负的位置即阈值带

#### tau_sweep_results.json + plots/tau_sweep.png
- **实验归属**: τ 扫描(FGL 效果 vs 系统记忆尺度)
- **产生程序**: `mackey_glass/run.py -e tau_sweep`(默认 off;早期为 `archive/tau_sweep.py`)
- **实验条件**: 固定 L=8,H=5,逐 τ(10–50),每 τ 1 seed,附 Lyapunov 与周期性得分
- **读表指南**: json 键含逐 τ 指标;找 FGL Δ% 由正转负的 τ 临界位置并与 L+H−1=13 比较

## 二、阈值检验(L/H 单向扫描)

验证 `L+H−1 ≥ τ` 公式。**固定 H 扫 L 是干净因果检验**(H 不动则教师 offset 不变);固定 L 扫 H 因 offset=H−1 与任务难度耦合,呈非单调 U 型。

#### threshold_quick_results.csv
- **产生程序**: `archive/threshold_quick.py`
- **实验条件**: L=9 vs 10(H=5):baseline MSE 5.90→0.69(8.5× 骤降),FGL Δ +78.5%→−15.7%
- **读表指南**: 这是最常被引用的"符号反转"原始证据行

#### l_threshold_results.csv
- **产生程序**: `mackey_glass/run.py -e l_threshold`(out_prefix="l_threshold")
- **读表指南**: 看 FGL Δ% 过零点是否落在 L=τ−H+1=9 处

#### h_threshold_results.csv + plots/h_threshold_figures.png
- **产生程序**: `mackey_glass/run.py -e h_threshold`
- **读表指南**: U 型曲线;Welch t 检验分段(H<9 vs H≥9,p≈0.005);"任务难度"与"信息窗口"两效应竞争的解释见 mackey_glass_tests.md

## 三、几何检验(排除"纯共振"假说)

把 (L,H) 大幅平移/变形后 τ_peak 是否仍在 [12,15]。结论:**在**,甜品区绑定倍周期动力学,而非某个几何巧合。

#### geometry_test_results.csv + plots/fig1_absolute_tau.png / fig2_relative_offset.png
- **产生程序**: `archive/tau_sweep_geometry.py`(第一轮)
- **读表指南**: fig1 绝对坐标、fig2 相对偏移坐标,看 τ_peak 稳定性

#### geometry_test2_results.csv + plots/geometry_test2_figure.png
- **产生程序**: `mackey_glass/run.py -e geometry`(第二轮,非对角几何)
- **读表指南**: 同上;两轮一致才支撑"绑定动力学"结论

## 四、迭代蒸馏(MG 负结果系列)

E 变体在 MG 上"干涸":易任务上学生几乎全对 → gap 权重趋零 → 蒸馏信号断流崩溃;E-soft 用 sigmoid 软地板修复干涸但增益有限。**MG 的 iterative_distill 保持 off 即源于此**。

#### iterative_mg_sweep.csv(+E-soft 及 wf0.05/0.1/0.15 系列)+ plots/iterative_mg_curves*.png
- **产生程序**: `archive/sweep_iterative.py`(Phase 0 试点);logs/ 下同名日志
- **实验条件**: 试点 3 个 (L,H) 点×3 seeds,K=3 轮;wf ∈ {0.05,0.1,0.15,0.2}
- **读表指南**: E 列 test MSE 反升=干涸指纹;E-soft 看 wf 越大越稳但稀释;round_predictions 图直观展示逐轮退化

#### iterative_distill.csv
- **产生程序**: `mackey_glass/run.py -e iterative_distill`(CLI/CSV 已就绪,默认不启用)
- **读表指南**: 六臂结构(A_single/A_iter/E*/E_soft_*_single/iter),与 CSTR 同口径便于跨域对照

## 五、数据与动力学背景图

#### plots/mg_tau_overview.png / mg_phase_portraits.png / mg_summary.png / mg_autocorr_overlay.png / cstr_vs_mg.png
- **产生程序**: `archive/plot_mg_data.py`(+`analysis.py`)
- **用途**: 展示不同 τ 下序列形态、相空间、自相关时间尺度及 CSTR/MG 对照,为"τ=系统记忆尺度"提供直觉依据
