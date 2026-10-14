# CSTR 数据集索引

本目录为 CSTR 研究的数据集总索引。所有文件均为 `(N, 2) float64` PyTorch tensor(经 `pickle.dump` 保存),约定第 0 列为输入、第 1 列为目标;除标注外均为 H₂O 质量分数双列格式。

## 一、核心周期数据

| 文件 | 规模 | 物理含义 | 生成来源 |
|---|---:|---|---|
| `data.pkl` | (3001, 2) | 反应器温度 | `cstr/generate.py`;原 H₂/O₂ 燃烧周期数据 |
| `data_h2o.pkl` | (3001, 2) | H₂O 质量分数 | `cstr/generate.py`;周期-1 锯齿目标,主线 FGL 默认数据 |

标准 CSTR 仿真条件:初温 770 K、压力 8 kPa、体积 10 μm³、进气流量 12 sccm、`t_end=300s`、`dt=0.1s`。主线 L×H 扫描、锚点优化、迭代蒸馏和外部 baseline 均优先使用 `data_h2o.pkl`。

## 二、强迫与双釜对照

| 文件家族 | 数量 | 说明 |
|---|---:|---|
| `data_forced_h2o_A{0.3,0.5,0.7}_f0.05.pkl` | 3 | H₂O 目标 + 外部正弦强迫 |
| `data_forced_temp_A{0.3,0.5,0.7}_f0.05.pkl` | 3 | 温度目标 + 外部正弦强迫 |
| `data_dual_h2o_Vb{2.0e-05,5.0e-05}_rc{0.1,0.2,0.3,0.5}.pkl` | 5 | 双 CSTR 耦合,不同体积损失/回流参数 |

生成程序:`cstr/generate_forced.py`、`cstr/generate_dual_cstr.py`。用于动力形式/多釜对照,不与主线周期数据混表。

## 三、稳定延迟反馈混沌族(13 个)

命名:`data_delayed_stable_h2o_tau{τ}_s1_A0.9_b0.03.pkl`,τ∈{30,35,40,45,50,55,60,65,70,80,100,120,150}。

- 生成程序:`cstr/generate_delayed_stable.py`(有界控制 + onset + EMA)
- 用途:τ 扫描、Lyapunov 诊断、延迟 FGL、地板战役
- 关键证据:`cstr/results/lyapunov_tau.csv` 显示全部 13 个数据集 Rosenstein 最大 Lyapunov 指数为正
- 注意:τ=50/100/150 是 `conclusion/1014.md` 中 Foundation Teacher 的三个 delayed 代表集

## 四、复杂数据集家族(generate_complex.py,2026-10-14)

生成程序：`cstr/generate_complex.py`（`--sweep` 批量产出）。
研究用途：拓展复杂度维度（多延迟/强迫+延迟/参数慢调/噪声控制），测试 FGL 的通用性。

| 文件 | 模式 | 关键参数 |
|---|---|---|
| `data_complex_multidelay_tau30_70.pkl` | multi_delay | τ={30,70}, A=0.1 |
| `data_complex_multidelay_tau50_120.pkl` | multi_delay | τ={50,120}, A=0.1 |
| `data_complex_multidelay_tau30_70_120.pkl` | multi_delay | τ={30,70,120}, A=0.05 |
| `data_complex_forceddelay_tau50_A0.05_f0.03.pkl` | forced_delay | τ=50, A=0.05, f=0.03 |
| `data_complex_forceddelay_tau50_A0.1_f0.05.pkl` | forced_delay | τ=50, A=0.1, f=0.05 |
| `data_complex_forceddelay_tau80_A0.08_f0.02.pkl` | forced_delay | τ=80, A=0.08, f=0.02 |
| `data_complex_forceddelay_tau100_A0.05_f0.07.pkl` | forced_delay | τ=100, A=0.05, f=0.07 |
| `data_complex_parammod_tau50_V8e-06-1e-05_T200.pkl` | param_mod | τ=50, V 8–14 μm³, T=200s |
| `data_complex_parammod_tau80_V8e-06-1e-05_T150.pkl` | param_mod | τ=80, V 8–14 μm³, T=150s |
| `data_complex_parammod_tau50_V6e-06-2e-05_T300.pkl` | param_mod | τ=50, V 6–16 μm³, T=300s |
| `data_complex_noisy_tau50_s0.01.pkl` | noisy | τ=50, σ=0.01 |
| `data_complex_noisy_tau50_s0.02.pkl` | noisy | τ=50, σ=0.02 |
| `data_complex_noisy_tau100_s0.02.pkl` | noisy | τ=100, σ=0.02 |

所有文件:(6000, 2) float64,H₂O 双列,与 `data_h2o.pkl` 同格式;`data_complex_parammod_tau50.pkl` 为脚本早期/备用输出,正式 Foundation Teacher 使用参数化命名版本。

## 复用规则

1. 查找历史结果前,先在本文件确认数据集归属,再查 `cstr/results/INDEX.md`。
2. 历史 bin-index 结果不能直接与连续物理值 MSE 比较。
3. 复杂数据族与主数据族的量纲/动态不同,跨数据集结论必须逐数据集标注。
4. 重新生成数据前先核对生成脚本参数和 git 历史;不要盲跑覆盖。
