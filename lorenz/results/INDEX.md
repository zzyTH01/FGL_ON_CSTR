# Lorenz-63 结果索引

本文件是 `lorenz/results/` 下全部结果文件的溯源与读法索引。Lorenz 域是三系统中最稳健的:**ρ=60 强混沌下 FGL 最佳增益 +62.2%(L=8,H=5),96% 配置(24/25)正向,免调参**。专题报告见 `conclusion/archive/data_5_lorenz_lh_sweep.md`(整理时自本目录移入),总结论见 `conclusion/final_conclusions.md` 与 `conclusion/研究进展报告.md`。

**目录布局速览**:根 = 1 个 csv + 1 个自动报告 md;`plots/` = 1 张热力图;`logs/` 为空(lh_sweep 未落日志,后续运行可写入)。

---

## 速查总表

| 文件 | 一句话说明 |
|---|---|
| `lorenz_lh_sweep.csv` | ρ=60 L×H 主线网格原始结果(25 配置×3 seeds) |
| `lorenz_lh_sweep_report.md` | sweep 框架每次重跑**自动生成**的汇总报告(会被覆盖刷新) |
| `plots/lorenz_lh_sweep.png` | 三联热力图(FGL Δ% / 绝对改进 / baseline MSE) |

---

#### lorenz_lh_sweep.csv
- **实验归属**: Lorenz ρ=60 L×H 主线扫描
- **产生程序**: `lorenz/run.py -e lh_sweep`(经 `fgl_common/sweep.py`,name="lorenz_lh_sweep")
- **关联数据**: `../data/lorenz_rho60.pkl`
- **实验条件**: L=H∈{4,6,8,10,12},seeds {0,1,2},α=0.5,T=4,bins=50;强混沌(Lyapunov ~1.6)
- **基本类型**: 每行一个 (L,H,seed):baseline/teacher/student MSE + abs_improvement + fgl_delta
- **读表指南**: 按 (L,H) 聚合 fgl_delta 均值——与 CSTR/MG 不同,Lorenz 几乎全域正向且短 H(H=5)最甜;baseline MSE 随 H 单调升(混沌可预报性极限 exp(λH)),无 CSTR 式地板

#### lorenz_lh_sweep_report.md
- **说明**: 由 `run_lh_sweep` 自动写出(含 Δ% 热力图 Markdown 表),**每次重跑覆盖**;手工结论以 `conclusion/archive/data_5_lorenz_lh_sweep.md` 为准

#### plots/lorenz_lh_sweep.png
- **产生程序**: 同 csv(`sweep.py` 输出,整理时移入 plots/,今后新图直写 plots/)
- **读图指南**: 左图绿区=FGL 有效配置;右图 baseline MSE 给出任务难度参照(随 H 变红)
