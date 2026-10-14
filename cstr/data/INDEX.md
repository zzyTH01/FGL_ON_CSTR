

## 五、复杂数据集家族（generate_complex.py，2026-10-14）

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

所有文件： (6000, 2) float64，H₂O 双列，与 data_h2o 同格式。
