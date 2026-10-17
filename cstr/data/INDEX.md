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

## 四、复杂数据集家族(generate_complex.py,2026-10-16 修正重跑)

生成程序：`cstr/generate_complex.py`（`--sweep` 批量产出）。
控制律与物理调制修正：
1. 所有模式统一先读取 `delayed_val = delay_buf[0]` 再执行缓冲区移位更新；`τ` 单位统一为采样步（`dt=0.1s`）；多通道维护独立 EMA（`filter_beta=0.03`）。
2. `simulate_param_mod` 改为通过 Cantera 移动壁面速度 `wall.velocity` 在 CVODES 内部连续驱动体积正弦调制，并加入 `feedback_limit=0.2` 防熄火限幅（解决原直接赋值 `cstr.volume` 未生效及强反馈熄火平坦化问题）。
3. 批量生成采用 multi_delay `A=0.4/0.25`、forced_delay `A=0.9, limit=0.85`、param_mod `A=0.9, limit=0.2`、noisy `A=0.5, limit=0.2`。

| 文件 | 家族 | 参数与备注 |
|---|---|---|
| `data_complex_multidelay_tau30_70.pkl` | multi_delay | τ={30,70}, A=0.4;周期性 0.854,dom lag 198 |
| `data_complex_multidelay_tau50_120.pkl` | multi_delay | τ={50,120}, A=0.4;周期性 0.821,dom lag 208 |
| `data_complex_multidelay_tau30_70_120.pkl` | multi_delay | τ={30,70,120}, A=0.25;周期性 0.870,dom lag 202 |
| `data_complex_forceddelay_tau50_A0.05_f0.03.pkl` | forced_delay | τ=50, A=0.05, f=0.03;周期性 0.388,dom lag 140 |
| `data_complex_forceddelay_tau50_A0.1_f0.05.pkl` | forced_delay | τ=50, A=0.1, f=0.05;周期性 0.576,dom lag 399 |
| `data_complex_forceddelay_tau80_A0.08_f0.02.pkl` | forced_delay | τ=80, A=0.08, f=0.02;周期性 0.416,dom lag 20 |
| `data_complex_forceddelay_tau100_A0.05_f0.07.pkl` | forced_delay | τ=100, A=0.05, f=0.07;周期性 0.449,dom lag 20 |
| `data_complex_parammod_tau50_V8.0e-06-1.4e-05_T200.pkl` | param_mod | τ=50, V 8–14 μm³, T=200s;周期性 0.231,dom lag 222 |
| `data_complex_parammod_tau80_V8.0e-06-1.4e-05_T150.pkl` | param_mod | τ=80, V 8–14 μm³, T=150s;周期性 0.249,dom lag 222 |
| `data_complex_parammod_tau50_V6.0e-06-1.6e-05_T300.pkl` | param_mod | τ=50, V 6–16 μm³, T=300s;周期性 0.228,dom lag 116 |
| `data_complex_noisy_tau50_s0.01.pkl` | noisy | τ=50, σ=0.01, A=0.5;周期性 0.933,dom lag 86 |
| `data_complex_noisy_tau50_s0.02.pkl` | noisy | τ=50, σ=0.02, A=0.5;周期性 0.933,dom lag 86 |
| `data_complex_noisy_tau50_s0.05.pkl` | noisy | τ=50, σ=0.05, A=0.5;周期性 0.933,dom lag 86 |
| `data_complex_noisy_tau100_s0.02.pkl` | noisy | τ=100, σ=0.02, A=0.5;周期性 0.933,dom lag 86 |
| `data_complex_noisy_tau100_s0.08.pkl` | noisy | τ=100, σ=0.08, A=0.5;周期性 0.933,dom lag 86 |

- 上述 `.pkl` 均为 `(6000, 2)`、H₂O 双列；2026-10-14 旧复杂数据与旧 `V{:.0e}` 文件名已不能作为当前结论依据。
- `data_complex_parammod_tau50.pkl` 仅是历史早期/备用输出，不在 Foundation Teacher 正式 `FAMILIES` 中。
- noisy sweep 现包含 15 个正式文件中的 5 个 noisy 数据（旧报告写 3 个）；Foundation Teacher 默认代表集仍只用 `tau50_s0.02`。

## 复用规则

1. 查找历史结果前,先在本文件确认数据集归属,再查 `cstr/results/INDEX.md`。
2. 历史 bin-index 结果不能直接与连续物理值 MSE 比较。
3. 复杂数据族与主数据族的量纲/动态不同,跨数据集结论必须逐数据集标注。
4. 重新生成数据前先核对生成脚本参数和 git 历史;不要盲跑覆盖。
