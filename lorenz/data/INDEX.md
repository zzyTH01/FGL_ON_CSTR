# Lorenz-63 数据集索引

本目录存放 Lorenz-63 域的全部 `.pkl` 数据集。Lorenz-63 是 3D 混沌 ODE(ẋ=σ(y−x), ẏ=x(ρ−z)−y, ż=xy−βz;σ=10, β=8/3),本项目以 **ρ=60(强混沌)** 为 FGL 基准。数据由 `lorenz/run.py -e generate` 生成(solve_ivp,t_end=500,dt=0.05,seed=42,去掉前 20% 暂态后截取),PyTorch tensor、`(6000, 2)`、float64,两列相同。文件名 `lorenz_rho{ρ}.pkl`,ρ 为整数时不带小数点(`%g` 格式)。

**速查总表**

| 文件 | ρ 区间 | 一句话用途 |
|---|---|---|
| `lorenz_rho20.pkl` | 周期/低混沌边界 | ρ 扫描筛选样本 |
| `lorenz_rho24.pkl` | 过渡区 | 同上 |
| `lorenz_rho25.pkl` | 过渡区(经典分岔链附近) | 同上 |
| `lorenz_rho28.pkl` | 经典混沌 | 同上(教科书参数) |
| `lorenz_rho32.pkl` | 混沌 | 同上 |
| `lorenz_rho40.pkl` | 强混沌前段 | 同上 |
| `lorenz_rho60.pkl` ⭐ | 强混沌 | **FGL L×H 扫描基准**(Lorenz 最稳健结论的数据源) |
| `lorenz_rho80.pkl` | 强混沌 | 扫描样本 |
| `lorenz_rho100.pkl` | 强混沌 | 扫描样本 |

---

## 说明

- **生成方式**: `uv run python lorenz/run.py -e generate --sweep` 按 ρ 列表 [20,22,24,25,28,32,40,60,80,99.5,100,100.5] 批量生成,**仅周期性得分 <0.85 或含倍周期判定的 ρ 落盘**(故现存 9 个文件,22/99.5/100.5 未入选或失败)。单次生成:`-e generate --rho <值>`。
- **ρ=60 主力**: `lorenz/results/plots/lorenz_lh_sweep.png` 与 `../results/lorenz_lh_sweep.csv`(25 配置×3 seeds,FGL Δ 最佳 +62.2%、96% 配置正向)全部基于 rho60。
- 其余 ρ 用于"哪个 ρ 适合做 FGL 基准"的筛选研究(周期性得分 + 最大 Lyapunov 估计,见 generate 运行日志与 `conclusion/archive/data_5_lorenz_lh_sweep.md`)。
- 命名约定:ρ 整数无 `.0` 后缀(如 `rho100.pkl`;历史遗留的 `rho20.0.pkl` 已删除,内容与 rho20 差异 ~1e-6 数值噪声)。
