# CSTR 数据集索引

本目录存放 CSTR(H₂/O₂ 连续搅拌反应器,Cantera 化学动力学模拟)域的全部 `.pkl` 数据集。均为 PyTorch tensor(`pickle.dump` 保存),形状 `(N, 2)`、float64:两列内容相同或相近(第 0 列作输入序列、第 1 列作目标值)。由 `cstr/generate*.py` 各生成器产生(生成器平铺在 `cstr/` 根,输出已统一直写本目录)。

**速查总表**

| 文件 | 家族 | 点数 | 一句话用途 |
|---|---|---|---|
| `data.pkl` | 基础 | 3001 | 温度通道(点火尖峰),早期 spike 任务用,FGL 主线未用 |
| `data_h2o.pkl` | 基础 | 3001 | H₂O 质量分数,**FGL 主线默认数据集**(周期-1 极限环) |
| `data_forced_temp_A{0.3,0.5,0.7}_f0.05.pkl` ×3 | 强迫 | 6000 | 外部正弦强迫的温度通道(混沌化探索) |
| `data_forced_h2o_A{0.3,0.5,0.7}_f0.05.pkl` ×3 | 强迫 | 6000 | 同上,H₂O 通道 |
| `data_dual_h2o_Vb*_rc*.pkl` ×5 | 双反应器 | 6000 | 带回收延迟的双 CSTR(混沌化探索) |
| `data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl` ×13 | 延迟稳定化 | 6000 | 正反馈延迟混沌化 τ 扫描系列(地板战役/Lyapunov/延迟 FGL 主对象) |

---

## 一、基础家族(周期-1 极限环)

生成程序:`cstr/generate.py`(H₂/O₂ 燃烧,t_end=300 s,dt=0.1 s → 3001 点)。
研究用途:**全项目主线**(L×H 扫描、锚点精扫、自适应蒸馏、迭代蒸馏、地板战役的"原始周期-1"对照)。

#### data.pkl
- **规格**: (3001, 2) float64
- **用途**: 温度通道,含周期性强点火尖峰;适合 spike 检测类任务,FGL 研究中仅作背景参考。

#### data_h2o.pkl ⭐ 主力数据集
- **规格**: (3001, 2) float64
- **用途**: H₂O 质量分数,平滑连续锯齿振荡(子周期 ≈72 步)。`cstr/run.py` 的 `DEFAULT_DATASET="h2o"` 即指它;所有主线 CSV(cstr_lh_sweep 等)均基于此数据集。

## 二、强迫家族(外部正弦强迫)

生成程序:`cstr/generate_forced.py`(t_end=600 s,dt=0.1 s)。
研究用途:通过外部强迫诱导非周期性(混沌化尝试之一);结果见 `conclusion/cstr_aperiodic_generation_report.md` 与 `conclusion/results/plots/cstr_forced_*.png`(图在 `../results/plots/`)。

#### data_forced_temp_A{A}_f0.05.pkl / data_forced_h2o_A{A}_f0.05.pkl(A ∈ 0.3/0.5/0.7)
- **参数含义**: `A`=强迫幅值,`f`=强迫频率(Hz,固定 0.05)
- **规格**: (6000, 2) float64
- **用途**: 比较不同强迫强度下的动力学(对比图 `../results/plots/cstr_forced_comparison.png`);FGL 未在此族上展开。

## 三、双反应器家族(回收延迟)

生成程序:`cstr/generate_dual_cstr.py`(`--volume_B`/`--recycle`,t_end=600 s,dt=0.1 s;`--sweep` 批量)。
研究用途:以反应器间回收流引入延迟自由度(混沌化尝试之二)。

#### data_dual_h2o_Vb{Vb}_rc{rc}.pkl(Vb ∈ 2.0e-05/5.0e-05,rc ∈ 0.1–0.5)
- **参数含义**: `Vb`=B 反应器体积(m³),`rc`=回流比
- **规格**: (6000, 2) float64
- **用途**: 探索参数空间中的准周期/非周期窗口;FGL 主实验未采用。

## 四、延迟稳定化家族(τ 扫描系列)⭐ 地板战役主对象

生成程序:`cstr/generate_delayed_stable.py`(有界控制 + 缓启动 onset + EMA 低通滤波 b;`--sweep` 批量产出 τ 系列;有单元测试 `tests/cstr/test_delayed_stable.py`)。
研究用途:**延迟反馈混沌化**——正反馈(s=+1, A=0.9, β=0.03)下复现 τ 分岔过渡曲线(τ 增大:周期→准周期→非周期)。13 个数据集全部经 Rosenstein 法验证最大 Lyapunov 指数为正(真混沌),见 `../results/lyapunov_tau.csv`。消费方:`run_floor_sweep.py`(地板战役)、`run_fgl_delayed.py` / `run_iterative_delayed.py`(延迟 FGL)、`lyapunov_delayed.py`。

#### data_delayed_stable_h2o_tau{T}_s1_A0.9_b0.03.pkl(T ∈ 30/35/40/45/50/55/60/65/70/80/100/120/150)
- **参数含义**: `tau`=反馈延迟步数;`s1`=正反馈符号;`A0.9`=反馈增益;`b0.03`=EMA 滤波系数
- **规格**: (6000, 2) float64
- **用途**: τ=100 是地板战役深挖网格(`../results/floor_sweep.csv`,45 行×6 量)的数据集;其余 τ 用于分岔曲线与 Lyapunov 谱;小 τ(30–70)偏周期、大 τ(≥80)非周期。

---

## 备注

- 历史生成器 `generate_delayed_feedback.py`(无界版,输出名 `data_delayed_h2o_*.pkl`)已被 delayed_stable 取代,本目录**无**该族现存文件。
- 各家族点数差异(3001 vs 6000)来自 t_end 设置不同,不影响滑动窗口实验口径。
- MSE 类结果均基于 50-bin 离散化的 bin-index 口径(见 `fgl_common/data.py`)。
