# Mackey-Glass 数据集索引

本目录存放 Mackey-Glass 域的全部 `.pkl` 数据集。MG 是延迟微分方程 x'(t) = βx(t)/(1+x(t)^n) + x(t−τ) 产生的经典混沌/倍周期系统,本项目基准为 **τ=13(倍周期分岔区)**。数据为 PyTorch tensor(`pickle.dump` 保存),形状 `(N, 2)`、float64,两列相同(第 0 列作输入、第 1 列作目标)。

**速查总表**

| 文件 | 点数 | 一句话用途 |
|---|---|---|
| `data.pkl` | 10000 | τ=13 倍周期基准快照,MG 域全部实验的唯一直接数据源 |

---

#### data.pkl ⭐ 唯一数据集
- **生成程序**: 预生成快照;再生成走 `mackey_glass/utils/generate.py`(基于 `mackey_glass/utils/utils.py` 的 `MackeyGlass` 类,jitcdde 数值积分)
- **参数含义**: 基准系统参数 **τ=13**(见 CLAUDE.md / README);dt=1.0
- **规格**: (10000, 2) float64(实测)
- **用途**: MG 域所有实验的数据源——base/drift(`run.py -e base,drift`)、L×H 扫描(`mg_lh_sweep.csv`)、τ/L/H 阈值测试、几何检验、迭代蒸馏试点,全部由它滑窗构造。

## ⚠️ 已知不一致(如实记录)

`utils/generate.py` 当前脚本内默认参数为 **τ=17**,与现存快照的基准 τ=13 不一致。现存 `data.pkl` 是早期预生成快照(τ=13 口径,与论文结论绑定);若需重新生成,应把脚本参数改回 τ=13 或显式确认意图,否则会产出不同动力学的数据集。历史 archive 脚本(`tau_sweep.py` 等)则按各自扫描自行传 τ。
