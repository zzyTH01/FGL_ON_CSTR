# CSTR 结果索引

本文件是 `cstr/results/` 下全部结果文件(CSV 结果表 / plots 图 / logs 日志)的溯源与读法索引:每个文件对应哪个实验、由哪个程序产生、实验条件是什么、怎么看。研究背景与结论详见 `conclusion/`(重点:`floor_study_final_report.md`、`floor_determinants.md`、`adaptive_continuous_distillation.md`、`iterative_distillation_summary.md`、`chaotic_cstr_fgl_exploration.md`、`best_sample_cstr.md`)。

**目录布局速览**:`results/` 根 = 21 个 `*.csv` 结果表;`results/plots/` = 37 个 `*.png`;`results/logs/` = 7 个日志。数据集在 `cstr/data/`(周期-1:`data.pkl` 温度 / `data_h2o.pkl` H₂O 质量分数,3001 点 dt=0.1s;延迟反馈混沌化:`data_delayed_stable_h2o_tau{...}_s1_A0.9_b0.03.pkl` 系列)。MSE 单位均为 bin-index²(50 bins 离散化),越小越好。

---

## 速查总表

| 文件 | 一句话说明 | 分组 |
|---|---|---|
| `cstr_lh_sweep.csv` | 主线 L×H 网格扫描原始结果(75 行) | 一 |
| `cstr_lh_sweep.png` | 主线三联热力图(Δ% / 绝对改进 / baseline) | 一 |
| `cstr_data_overview.png` | 数据四面板概览(温度/H₂O 全程+前 60s) | 二 |
| `cstr_full_series.png` | 温度+H₂O 全程 300s 曲线 | 二 |
| `cstr_zoom_60s.png` | 前 60s 点火细节放大 | 二 |
| `cstr_h2o_sawtooth.png` | H₂O 子振荡锯齿细节(t=100–130s) | 二 |
| `cstr_temperature_analysis.png` | 温度分布(对数)+ 21 次强点火叠加 | 二 |
| `cstr_dashboard.png` | 数据仪表盘(4 面板) | 二 |
| `cstr_forced_comparison.png` | 原始 vs 外部正弦强制数据对比 | 二 |
| `cstr_forced_zoom.png` | 强制对比前 120s 放大 | 二 |
| `cstr_autocorr_comparison.png` | 各变体自相关对比(含 MG 参照) | 二 |
| `cstr_forced_summary.png` | 强制实验汇总面板 | 二 |
| `param_sweep_results.txt` | U/K/flow 参数扫描 periodicity 文本输出 | 二 |
| `delayed_tau_sweep_s-1_A0.3_b0.1.csv` | 负反馈弱反馈 τ 扫描(全锁相,负对照) | 二 |
| `delayed_tau_sweep_s-1_A0.3_b0.1.png` | 同上曲线图 | 二 |
| `delayed_tau_sweep_s1_A0.9_b0.03.csv` | 正反馈强反馈 τ 扫描(周期↔非周期过渡) | 二 |
| `delayed_tau_sweep_s1_A0.9_b0.03.png` | 同上曲线图 | 二 |
| `delayed_stable_h2o_panels.png` | 各 τ 延迟数据集 H₂O 序列小倍数图 | 二 |
| `lyapunov_tau.csv` | 13 个延迟数据集最大 Lyapunov 指数 | 二/六 |
| `anchor_refine_results.csv` | 锚点 L×H 精细网格(125 次) | 三 |
| `anchor_lstm_results.csv` | LSTM 架构对照(5 seeds) | 三 |
| `anchor_regression_results.csv` | 回归模式 α 扫描(25 次) | 三 |
| `anchor_alpha_T_results.csv` | α×T 网格(108 次) | 三 |
| `adaptive_weight_results.csv` | 蒸馏权重变体 A–D × 5 seeds | 三 |
| `cstr_predictions_L20_H12.png` | L20H12 三模型预测 vs 真实 | 三 |
| `cstr_scatter_L20_H12.png` | L20H12 预测-真实散点 | 三 |
| `cstr_predictions_L20_H60.png` | L20H60 预测(baseline 坍缩异常) | 三 |
| `cstr_scatter_L20_H60.png` | L20H60 散点 | 三 |
| `cstr_per_sample_mse.png` | 逐样本平方误差(三模型) | 三 |
| `cstr_true_vs_error.png` | 真值 vs 误差散点 | 三 |
| `cstr_adaptive_diagnostic.png` | 变体 C 弱点分布诊断 | 三 |
| `cstr_weight_diagnostic.png` | 蒸馏权重诊断图 | 三 |
| `cstr_adaptive_best_predictions.png` | 最佳样本(变体 C seed3)预测 | 三 |
| `cstr_adaptive_best_scatter.png` | 最佳样本散点 | 三 |
| `adaptive_anchor_L20H15_n5.csv` | 变体 A/C/E 锚点 n=5 补档重跑 | 四 |
| `adaptive_lh_sweep.csv` | 变体 E vs A 的 L×H 网格(100 行) | 四 |
| `adaptive_lh_E_vs_A.png` | E 相对 A 降幅热力图 | 四 |
| `iterative_sweep.csv` | 迭代蒸馏 Phase 0 试点(K=3,3 点×3 seeds) | 五 |
| `iterative_curves.png` | Phase 0 逐轮 val MSE 曲线(3 面板) | 五 |
| `iterative_sweep_E-soft_wf0.2.csv` | E-soft 软地板版(3 点×3 seeds) | 五 |
| `iterative_curves_E-soft_wf0.2.png` | E-soft 逐轮曲线 | 五 |
| `iterative_saturation_sweep.csv` | K=15 饱和探索(2 点×3 seeds) | 五 |
| `iterative_saturation_curves.png` | 饱和逐轮曲线 | 五 |
| `iterative_phase1_anchor.csv` | Phase 1 锚点 L20H15 n=10 K=6 | 五 |
| `iterative_phase1_anchor_curves.png` | 锚点逐轮曲线 | 五 |
| `iterative_phase1_grid.csv` | Phase 1 网格 5×5×2 seeds K=6 | 五 |
| `iterative_phase1_grid_curves.png` | 网格逐轮曲线(25 面板) | 五 |
| `iterative_distill.csv` | run.py `iterative_distill` 双权重分布输出 | 五 |
| `iterative_round_predictions.png` | E_iter 逐轮预测 vs 真实可视化 | 五 |
| `fgl_delayed_summary.csv` | 一次性 FGL on 6 个延迟数据集汇总 | 六 |
| `iterative_delayed_summary.csv` | E 迭代蒸馏 on 4 个延迟数据集汇总 | 六 |
| `chaotic_floor_verify.csv` | 混沌 CSTR K=10 饱和核实(4 cell×5 seed) | 六 |
| `floor_sweep.csv` | 地板战役 τ=100 深挖网格(45 行×6 量) | 七 |
| `floor_h1_baseline_heatmap.png` | H1:baseline 地板热力图 | 七 |
| `floor_h1_transition.png` | H1:地板 vs L+H-1(H=15 列) | 七 |
| `floor_h2_c_ratio.png` | H2:c=E_iter/baseline 箱线图 | 七 |
| `floor_h3_floor_vs_teacher.png` | H3:地板 vs teacher_mse 散点 | 七 |
| `floor_h3_logfloor_vs_H.png` | H3:log(地板) vs H | 七 |
| `floor_h4_paired.png` | H4:E_iter vs 收敛 baseline 配对散点 | 七 |
| `logs/chaotic_floor_verify.log` | 混沌核实运行日志 | 八 |
| `logs/iterative_cstr_esoft_log.txt` | E-soft 扫描日志 | 八 |
| `logs/iterative_phase0_log.txt` | Phase 0 试点日志 | 八 |
| `logs/iterative_phase1_anchor_log.txt` | Phase 1 锚点日志 | 八 |
| `logs/iterative_phase1_grid_log.txt` | Phase 1 网格日志 | 八 |
| `logs/iterative_saturation_log.txt` | K=15 饱和日志 | 八 |
| `logs/param_sweep_results.txt` | 参数扫描文本输出 | 八 |

---

## 一、主线 L×H 扫描(标准 FGL)

研究问题:CSTR(周期-1 极限环)上 FGL 增益随 lookback L 与 horizon H 如何变化——项目主线变量。核心结论:CSTR 普遍偏难,增益有限且有"地板效应";H=60 出现 baseline 反超 teacher 的异常。报告见 `conclusion/final_conclusions.md`、`conclusion/best_sample_cstr.md`。

#### cstr_lh_sweep.csv
- **实验归属**:主线 L×H 网格扫描(对应原 `cstr/exp/fgl_cstr_lh_sweep.py`)
- **产生程序**:`cstr/run.py`(`uv run python cstr/run.py -e lh_sweep`),核心逻辑在 `fgl_common/sweep.py::run_lh_sweep`
- **实验条件**:`data_h2o.pkl`;L∈{8,20,35,50,72} × H∈{5,15,30,45,60} × seeds{0,1,2};α=0.5,T=4,bins=50,epochs=30,2 层 RNN 分类模式
- **基本类型**:75 行(5L×5H×3 seeds),列:`L,H,seed,baseline_mse,teacher_mse,student_mse,abs_improvement,fgl_delta`
- **读表指南**:`fgl_delta`(%)>0 表示 FGL student 优于 baseline;`abs_improvement`=baseline−student 绝对量。CSTR 周期约 72 步,注意 L+H 与子周期尺度的关系

#### cstr_lh_sweep.png
- **实验归属 / 产生程序 / 关联数据 / 实验条件**:同上
- **基本类型**:三联热力图(`fgl_common/sweep.py::_plot_heatmaps`),x=H、y=L
- **读图指南**:左"FGL Δ%"(RdYlGn,绿=正增益)、中"Abs Improvement"、右"Baseline MSE"(YlOrRd,越亮=任务越难);每格标注数值。绿格多=该区域 FGL 有效;右侧亮格=难任务区

---

## 二、数据集画像与混沌化生成

研究问题:CSTR 数据本身长什么样、周期性有多强、能否通过调参或延迟反馈把它混沌化。核心结论:单 CSTR 刚性极限环调参调不动(periodicity≥0.93);延迟反馈(正反馈 s=+1, A=0.9, β=0.03)复现了周期↔非周期 τ 过渡,产出 13 个非周期数据集。

### 2.1 基础数据画像(`cstr/archive/plot_data.py`,读 `data.pkl`+`data_h2o.pkl`)

#### cstr_data_overview.png
- **实验归属**:早期数据概览(见 `conclusion/项目汇报总结.md` §6.1 引用)
- **产生程序**:未能溯源(建议结合 git log 考古;重构前快照 7d5ad18 已存在,产生脚本未入库)
- **关联数据**:`data.pkl`(温度)+ `data_h2o.pkl`(H₂O)
- **基本类型**:4 面板:温度全程 300s / 温度前 60s / H₂O 全程 / H₂O 前 60s
- **读图指南**:温度呈点火尖峰(峰值逐周期漂移升高),H₂O 呈平滑锯齿振荡,周期约 7.15s(72 步)

#### cstr_full_series.png / cstr_zoom_60s.png / cstr_h2o_sawtooth.png / cstr_temperature_analysis.png / cstr_dashboard.png
- **实验归属**:基础数据特性画像(同上)
- **产生程序**:`cstr/archive/plot_data.py`(`uv run python cstr/archive/plot_data.py`;原图输出于 cstr/ 根,commit e2ce36b 归并入 results/plots/)
- **关联数据**:`data.pkl` + `data_h2o.pkl`
- **实验条件**:300s 仿真,dt=0.1s,3001 点
- **基本类型 / 读图指南**:
  - `cstr_full_series.png`:温度+H₂O 全程双图;看点火尖峰间隔与 H₂O 锯齿周期
  - `cstr_zoom_60s.png`:前 60s 放大;展示"小点火→强点火"交替结构
  - `cstr_h2o_sawtooth.png`:t=100–130s H₂O 子振荡,锯齿波形+包络增长
  - `cstr_temperature_analysis.png`:温度直方图(对数 y 轴,展示 770K 主体的稀疏尖峰)+ 全部 21 次强点火曲线叠加
  - `cstr_dashboard.png`:4 面板仪表盘(温度 300s / H₂O 300s / 前 60s / 两变量直方图),一图看全数据

### 2.2 外部强制对照(`cstr/archive/plot_forced_comparison.py`)

#### cstr_forced_comparison.png / cstr_forced_zoom.png / cstr_autocorr_comparison.png / cstr_forced_summary.png
- **实验归属**:外部周期强制(准周期化尝试,负结果——锁相)
- **产生程序**:`cstr/archive/plot_forced_comparison.py`;数据由 `cstr/generate_forced.py` 生成(入口正弦流量扰动 f=0.05Hz、幅值 ±30%)
- **关联数据**:`data_h2o.pkl` vs 强制数据集
- **基本类型 / 读图指南**:
  - `cstr_forced_comparison.png`:原始 vs 强制的全程对比(各带 Full/Zoom/Autocorr 三列)
  - `cstr_forced_zoom.png`:前 120s 放大
  - `cstr_autocorr_comparison.png`:各 CSTR 变体 + Mackey-Glass 的自相关衰减对比;周期信号自相关峰不衰减、混沌信号快速衰减,以此判别周期性
  - `cstr_forced_summary.png`:汇总面板(标注 periodicity 分数)。结论:强制后锁相 periodicity≈1.0,混沌化失败

### 2.3 物理参数扫描与延迟反馈混沌化

#### param_sweep_results.txt(亦列于日志节)
- **实验归属**:U/K/flow 三参数扫描,寻找非周期工况(失败)
- **产生程序**:`cstr/archive/param_sweep.py`(Cantera)
- **基本类型**:纯文本,每行一组 `U, K, flow, periodicity, spikes, T范围, n`
- **读表指南**:看 periodicity 列——全部 ≥0.935,证明单一物理参数调不出混沌

#### delayed_tau_sweep_s-1_A0.3_b0.1.csv / .png
- **实验归属**:延迟反馈 τ 扫描——负反馈弱反馈构型(负对照)
- **产生程序**:`cstr/generate_delayed_stable.py --sweep --sign -1 --amplitude 0.3 --filter_beta 0.1`
- **实验条件**:mdot 反馈 `mdot₀·(1+s·A·tanh((H₂O[t−τ]−c)/w))`+缓启动+EMA 滤波;t_end=600s,dt=0.1
- **基本类型**:每行一个 τ,列 `tau, periodicity, dom_period, dom_period_s, T_min, T_max, status`;12 个 τ,其中 1 个 fail(CanteraError)
- **读表指南 / 读图指南**:PNG 为双 y 轴曲线(periodicity 左轴、主周期右轴,虚线 0.85 为非周期阈值)。本构型全部锁相(periodicity≈0.99)→ 证明"负反馈/弱反馈不出过渡"

#### delayed_tau_sweep_s1_A0.9_b0.03.csv / .png
- **实验归属**:延迟反馈 τ 扫描——正反馈强反馈构型(**成功**的混沌化路线)
- **产生程序**:`cstr/generate_delayed_stable.py --sweep --sign 1 --amplitude 0.9 --filter_beta 0.03`
- **基本类型**:同上,17 个 τ 全 ok
- **读表指南 / 读图指南**:periodicity 随 τ 呈"周期(0.91)→非周期(τ40–65,≈0.55)→准周期窗(τ70–80,≈0.8)→再非周期(τ100–150,≈0.48)"的 DDE 分岔交替带。`status=ok` 且 periodicity<0.85 的 τ 会另存训练数据集到 `cstr/data/`。后续延迟实验全部基于此构型的数据

#### delayed_stable_h2o_panels.png
- **实验归属**:延迟数据集目检
- **产生程序**:`cstr/archive/plot_delayed_stable.py`
- **关联数据**:`cstr/data/data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl`
- **基本类型**:小倍数图——每个已存 τ 一块 H₂O 时序,按 τ 排序并标注 periodicity
- **读图指南**:对比各 τ 波形从规则锯齿(低 τ)到不规则(τ100–150)的过渡;配合 2.3 的 CSV 看

#### lyapunov_tau.csv
- **实验归属**:延迟数据集最大 Lyapunov 指数估计(补"低周期性≠严格混沌"的证据缺口;地板战役 H3 的 λ 来源)
- **产生程序**:`cstr/lyapunov_delayed.py`(`uv run python cstr/run.py -e lyapunov`),Rosenstein 方法
- **实验条件**:glob 匹配全部 `tau*_s1_A0.9_b0.03.pkl`,burn_frac=0.2
- **基本类型**:13 行,列 `file, tau, lyap, N`
- **读表指南**:`lyap`>0 即混沌;τ=100 的 λ≈0.0079/样本被 `floor_determinants.md` H3 引为标定

---

## 三、锚点优化与蒸馏权重变体(单趟,2026-07 战役)

研究问题:围绕锚点 L=20,H=15 系统优化——换模型(LSTM)、换模式(回归)、调超参(α×T)、精细扫 L×H、给 KL 加样本权重(A–D 变体)。核心结论:最优配置是 L20H12(+25.7%),模型/模式/超参都不是瓶颈;权重变体 C 温和有效,后被放大版 E 取代(→ 第四组)。报告见 `conclusion/best_sample_cstr.md`。

#### anchor_refine_results.csv
- **实验归属**:Task 1 锚点邻域精细 L×H 网格
- **产生程序**:`cstr/archive/anchor_optimization.py --task 1`(原 `cstr/exp/`,经子进程调用 `fgl_cstr.py`)
- **实验条件**:`data_h2o.pkl`;L∈{15,18,20,22,25}×H∈{10,12,15,18,20}×5 seeds=125 次;epochs=30,bins=50,T=4,batch=64
- **基本类型**:126 行(含表头),列 `L,H,seed,baseline_mse,teacher_mse,student_mse,abs_improvement,fgl_delta`
- **读表指南**:按 (L,H) 聚合看 `fgl_delta` 均值;最优点 L20H12(+25.7%±11.2%)

#### anchor_lstm_results.csv
- **实验归属**:Task 2 LSTM 架构对照
- **产生程序**:`cstr/archive/anchor_optimization.py --task 2`(调用 `fgl_cstr_lstm.py`)
- **实验条件**:L20H15,5 seeds,其余同上
- **基本类型**:5 行,列含 `architecture`(全 LSTM)
- **读表指南**:与 RNN 基线对比 fgl_delta;结论 LSTM≈RNN → 模型容量非瓶颈(失败探索)

#### anchor_regression_results.csv
- **实验归属**:Task 3 连续值回归模式 α 扫描
- **产生程序**:`cstr/archive/anchor_optimization.py --task 3`(调用 `fgl_cstr_regression.py`)
- **实验条件**:L20H15;α∈{0.0,0.3,0.5,0.7,0.9}×5 seeds=25 次;注意此表 MSE 为连续值口径(量级 ~0.01)
- **基本类型**:25 行,列含 `mode,alpha`
- **读表指南**:回归模式整体不敌分类离散化 → 失败探索;勿与其他 CSV 的 MSE 直接比大小

#### anchor_alpha_T_results.csv
- **实验归属**:Task 4 α×T 蒸馏超参网格
- **产生程序**:`cstr/archive/anchor_optimization.py --task 4`
- **实验条件**:L20H15;α∈{0,0.1,0.3,0.5,0.7,0.9}×T∈{2,4,6,8,10,12}×3 seeds=108 次
- **基本类型**:108 行,列 `L,H,alpha,temperature,seed,...,fgl_delta`
- **读表指南**:聚合后看 α×T 平面;结论:超参不敏感、不决定地板

#### adaptive_weight_results.csv
- **实验归属**:Experiment 7 自适应蒸馏权重变体(A 均匀 / B 按基线误差 / C teacher−student 差距 / D 放大版)
- **产生程序**:`cstr/archive/adaptive_weight_exp.py`
- **实验条件**:`data_h2o.pkl`,L20H15,5 seeds,α=0.5,T=4
- **基本类型**:20 行(A–D 各 5),列 `variant,L,H,seed,baseline_mse,teacher_mse,student_mse,abs_improvement,fgl_delta`
- **读表指南**:按 variant 聚合比 fgl_delta;C 有效但落在噪声边缘,催生后续变体 E(见第四组)

### 预测可视化系列(单次训练快照,非扫描)

#### cstr_predictions_L20_H12.png / cstr_scatter_L20_H12.png
- **实验归属**:最优配置 L20H12 的 teacher/baseline/student 预测展示
- **产生程序**:`cstr/archive/plot_predictions.py`(现场训练后绘图)
- **关联数据**:`data_h2o.pkl`;L=20,H=12,α=0.5,T=4
- **基本类型**:折线图(真实 vs 三模型预测)/ 散点图(预测-真实,y=x 参考线)
- **读图指南**:散点越贴 y=x 越好;student 点应比 baseline 更贴,尤其在尖峰处

#### cstr_predictions_L20_H60.png / cstr_scatter_L20_H60.png
- **实验归属**:H=60 异常诊断——baseline "坍缩"(MSE≈15,corr≈0.94,反超 teacher),破坏 FGL 前提
- **产生程序**:`cstr/archive/plot_predictions_L20_H60.py`
- **基本类型 / 读图指南**:同上;此图用于展示"baseline 好到 teacher 没东西可教"的反常工况,是地板效应的重要证据

#### cstr_per_sample_mse.png / cstr_true_vs_error.png
- **实验归属**:逐样本误差解剖(L20H12,seed=0)
- **产生程序**:`cstr/archive/plot_per_sample_mse.py`
- **基本类型**:逐样本平方误差三模型对比折线 / 真值 vs 误差散点
- **读图指南**:误差高度集中于点火尖峰样本 → 为"加权蒸馏"提供动机(与第三组 CSV 呼应)

#### cstr_adaptive_diagnostic.png / cstr_weight_diagnostic.png
- **实验归属**:变体 C 权重机制诊断(L20H15,seed=0)
- **产生程序**:`cstr/archive/plot_adaptive_diagnostic.py`
- **基本类型**:弱点分布图(student−teacher 误差差距沿时间轴)+ 权重分配图
- **读图指南**:看弱点是否集中(结论:top-10% 点占 99.6% gap)→ 支持把蒸馏预算集中到弱点的 E 变体设计

#### cstr_adaptive_best_predictions.png / cstr_adaptive_best_scatter.png
- **实验归属**:最佳单样本展示(变体 C,L20H15,seed=3)
- **产生程序**:`cstr/archive/plot_adaptive_best.py`
- **基本类型 / 读图指南**:同预测/散点系列;用于报告中的"最佳案例"插图

---

## 四、自适应蒸馏变体 E(teacher−student 差距放大加权)

研究问题:把 KL 蒸馏预算按 teacher−student 误差差距(零地板放大)加权,是否稳定优于均匀加权(A)。核心结论:是——锚点 E 5/5 seeds 胜 A(Δinit +33.2%,配对 p≈0.018),网格 25/25 cell 均值为正、平均降 30.4%;机制是弱点高度集中。报告见 `conclusion/adaptive_continuous_distillation.md` §3。

#### adaptive_anchor_L20H15_n5.csv
- **实验归属**:变体 A/C/E 锚点重跑补档(commit d2a2070,精确复现原主张)
- **产生程序**:临时重跑脚本(未入库),内部调用与 `cstr/run.py -e adaptive_weight` 同一接口 `fgl_common.run_adaptive_weight`
- **实验条件**:`data_h2o.pkl`,L20H15,A/C/E × 5 seeds,α=0.5,T=4,bins=50,epochs=30
- **基本类型**:10 行(A/E 各 5),列 `variant,L,H,seed,teacher_mse,baseline_mse,student_mse_init,student_mse,abs_improvement,fgl_delta,init_delta`
- **读表指南**:`init_delta`(相对初始 student 的降幅)为主判据;E 应全正且大于 A/C

#### adaptive_lh_sweep.csv
- **实验归属**:变体 E vs A 的 L×H 配对网格
- **产生程序**:`cstr/sweep_adaptive.py`(`uv run python cstr/run.py -e adaptive_grid` 或直接运行)
- **实验条件**:`data_h2o.pkl`;L∈{8,20,35,50,72}×H∈{5,15,30,45,60}×{A,E}×2 seeds=100 行;α=0.5,T=4,epochs=30
- **基本类型**:列 `L,H,seed,variant,baseline_mse,student_mse_init,student_mse,fgl_delta,init_delta`
- **读表指南**:同 seed 下 E−A 为配对比较;按 (L,H) 聚合看 E 是否低于 A。L=72 整行 A 已近地板(MSE≈14),E 增益仅 +1~9%

#### adaptive_lh_E_vs_A.png
- **实验归属 / 产生程序 / 关联数据 / 实验条件**:同上
- **基本类型**:热力图(`sweep_adaptive.py::_heatmap`),RdYlGn,格内标注 `降幅% + E<A seed 数`
- **读图指南**:x=H、y=L;绿色=E 更好,数字 `n/N` 为该格 E 胜出的 seed 数。中部(L8–35、H15–45)最绿

---

## 五、迭代(连续)自适应蒸馏

研究问题:把单趟蒸馏改成"训练→按 E 权重重蒸馏"的多轮迭代(4 臂:A_single=标准 FGL 参照 / E_single=1 轮 E / A_iter=迭代均匀对照 / E_iter=迭代 E),并测试软地板 E-soft 与饱和轮数 K。核心结论:E_iter 的价值是**更快、更可靠地到达数据地板**(K=3 时 +31~42%,饱和后与 A_iter 同地板 ~30);E-soft 无免费午餐;Phase 1 大样本后锚点无显著差异。报告见 `conclusion/adaptive_continuous_distillation.md`、`iterative_distillation_summary.md`、`iterative_phase1_cstr.md`、`iterative_pilot_phase0.md`。

#### iterative_sweep.csv
- **实验归属**:Phase 0 试点(3 典型点)
- **产生程序**:`cstr/sweep_iterative.py --cells "20,15;8,30;72,15" --seeds 3 --epochs 20 --round_epochs 10 --K 3`(日志 `logs/iterative_phase0_log.txt`)
- **实验条件**:`data_h2o.pkl`;cells {L20H15, L8H30, L72H15}×3 seeds×4 臂;variant E
- **基本类型**:36 行,列 `L,H,seed,arm,baseline_mse,student_mse,fgl_delta,init_delta,rounds_used,mse_curve_val,mse_curve_test`(曲线分号分隔)
- **读表指南**:比较 E_iter vs A_iter 的 `student_mse`(K=3 时点:E_iter 领先 +32~40%);`mse_curve_test` 可还原逐轮轨迹

#### iterative_curves.png
- **实验归属 / 产生程序 / 实验条件**:同上(Phase 0 K=3)
- **基本类型**:3 面板逐轮 val MSE 折线(4 臂,x=round,0=round-0 student)
- **读图指南**:E_iter(红)下降最快即"迭代 E 更快";注意 L72H15 三臂重合(易任务无差别)。⚠️ 本文件历史上被多次运行覆盖后由 git 还原,现内容为 Phase 0 版本(见文末异常备注)

#### iterative_sweep_E-soft_wf0.2.csv / iterative_curves_E-soft_wf0.2.png
- **实验归属**:E-soft 软地板版(w_floor=0.2)同点对照
- **产生程序**:`cstr/sweep_iterative.py --variant E-soft --w_floor 0.2`(同 3 cells × 3 seeds,K=3;日志 `logs/iterative_cstr_esoft_log.txt`)
- **基本类型**:同 `iterative_sweep.csv`
- **读表指南 / 读图指南**:E_iter vs A_iter 变为 −29%(软地板抵消了 E 的浓度优势)→ 支撑"w_floor 是浓度/稳定旋钮、无免费午餐"结论

#### iterative_saturation_sweep.csv / iterative_saturation_curves.png
- **实验归属**:K=15 饱和探索(L20H15、L8H30 两点 × 3 seeds)
- **产生程序**:`cstr/sweep_iterative.py` 同接口、K=15(commit 5c690ce;当时脚本尚无 tag 机制,输出后手工改名为 `iterative_saturation_*`;日志 `logs/iterative_saturation_log.txt`)
- **基本类型**:24 行,列同上
- **读表指南 / 读图指南**:看每 seed `mse_curve_test` 最小值——E_iter 稳定到 ~30 地板(round 5–6 饱和,最大降 82%);A_iter 常停滞(卡 57)。注意 L8H30 有 1.42 的单 seed 过拟合离群(val≈test + keep-best),非可复现值

#### iterative_phase1_anchor.csv / iterative_phase1_anchor_curves.png
- **实验归属**:Phase 1 锚点全验证(L20H15,n=10 seeds,K=6 饱和)
- **产生程序**:Phase 1 临时脚本(未入库;commit e385ce8 直接落盘结果;分析用 `cstr/archive/analyze_iterative.py`;日志 `logs/iterative_phase1_anchor_log.txt`)
- **实验条件**:epochs=30,round_epochs=15,K=6,α=0.5,T=4,batch=64
- **基本类型**:40 行(10 seeds×4 臂),列同上
- **读表指南**:per-seed 取 `mse_curve_test` 最小值后配对比较 E_iter vs A_iter → p=0.94 无显著差异,两臂同到 ~30 地板;seed9=14.5 为过拟合离群

#### iterative_phase1_grid.csv / iterative_phase1_grid_curves.png
- **实验归属**:Phase 1 网格(5×5×2 seeds,K=6)
- **产生程序**:同上(日志 `logs/iterative_phase1_grid_log.txt`)
- **基本类型**:200 行(25 cells×2 seeds×4 臂),列同上
- **读表指南 / 读图指南**:网格级 E_iter vs A_iter 名义显著(p=0.005)但中位数 0%、胜率 52% → 不稳健;E_iter 真实价值=中等难度 cell(L8–35、H30–60)上 A_iter 卡死时 E_iter 能收敛(+17~30%)。PNG 为 25 面板逐轮曲线

#### iterative_distill.csv
- **实验归属**:`run.py` 常驻开关实验 `iterative_distill`(双权重分布 E 硬 / E-soft)
- **产生程序**:`cstr/run.py`(`uv run python cstr/run.py -e iterative_distill --distill_variants E,E-soft`)
- **实验条件**:默认 L20H15、3 seeds、K=5、round_epochs=15、w_floor=0.2;当前存档为 E/E-soft 双变体输出
- **基本类型**:每行一个 (seed, arm),列 `seed,arm,student_mse,baseline_mse,teacher_mse,fgl_delta,init_delta,rounds_used,total_epochs`
- **读表指南**:arm 含 `{E,E-soft}×{single,iter}`;比较 E_iter vs A_iter 的 student_mse 与 init_delta。注意此文件会被每次运行覆盖,代表最新一次开关实验而非历史存档

#### iterative_round_predictions.png
- **实验归属**:E_iter 逐轮预测可视化(L20H15)
- **产生程序**:`cstr/archive/plot_iterative_rounds.py`(`uv run python cstr/archive/plot_iterative_rounds.py`)
- **关联数据**:`data_h2o.pkl`(现场跑 `run_iterative_distillation`)
- **基本类型**:多面板折线:每轮(含 round-0)test 集真实值(灰线)vs 预测(红点)
- **读图指南**:红点逐轮贴合灰线的程度=迭代蒸馏起效的直观证据(commit 9c132c8)

---

## 六、延迟反馈混沌化 CSTR 上的 FGL 与迭代蒸馏

研究问题:数据从周期(base)变非周期(τ 扫描)后,FGL 增益是否回升?E 迭代蒸馏在混沌数据上是否更强?核心结论:一次性 FGL 增益混杂、与周期性无单调关系;但 E_iter 在 3 个非周期数据集上全胜 A_iter——自适应 E 的价值专显于混沌数据;饱和(K=10)核实后 E≈A 同地板。报告见 `conclusion/chaotic_cstr_fgl_exploration.md`、`adaptive_continuous_distillation.md` §5、`iterative_convergence_speed.md`。

#### fgl_delayed_summary.csv
- **实验归属**:Phase B 一次性 FGL 基线对照
- **产生程序**:`cstr/run_fgl_delayed.py`(`uv run python cstr/run.py -e delayed_fgl`)
- **实验条件**:6 个数据集(base + τ30/50/80/100/150)×3 seeds;L20H15,α=0.5,T=4,与主线同口径
- **基本类型**:每数据集一行,列 `dataset,periodicity,N,baseline_mse,baseline_sd,student_mse,student_sd,improvement,improvement_sd`
- **读表指南**:看 `improvement`(%) 及其 sd;结论为"混杂、无单调趋势"(5/6 正增益但最非周期的 τ50≈0)

#### iterative_delayed_summary.csv
- **实验归属**:E 迭代蒸馏 on 延迟数据集
- **产生程序**:`cstr/run_iterative_delayed.py`(`uv run python cstr/run.py -e delayed_iter`)
- **实验条件**:4 个数据集(base/τ50/τ100/τ150)×3 seeds×4 臂;L20H15,K=5,round_epochs=15
- **基本类型**:16 行(dataset×arm),列 `dataset,periodicity,arm,student_mse,student_sd,init_delta,init_delta_sd,rounds_used,baseline_mse`
- **读表指南**:aperiodic 数据集上 E_iter 的 `init_delta` 应显著>0 且 `rounds_used` 更少;周期 base 上 A_iter 反而赢 → E 专利于混沌数据

#### chaotic_floor_verify.csv
- **实验归属**:混沌 CSTR 上 E_iter vs A_iter 地板核实(K=10 饱和重跑,commit 38601ea)
- **产生程序**:`cstr/archive/verify_chaotic_iter_floor.py`(`--dataset tau100 --K 10 --seeds 5`;日志 `logs/chaotic_floor_verify.log`)
- **实验条件**:tau100;cells {(20,15),(50,15),(85,15),(100,30)}×5 seeds×2 臂(A_iter/E_iter);epochs=30,round_epochs=15
- **基本类型**:40 行,列 `dataset,L,H,seed,arm,baseline_mse,student_mse,rounds_used,total_epochs,mse_curve_test`
- **读表指南**:per-seed 取曲线最小值配对比较;结论:K=10 饱和后 E≈A(L20H15 同 ~145 地板),K=5 时 A 未饱和造成的"表面优势"消失;仅 L50H15 上 E 仍占优但 seed 间波动大

---

## 七、地板成因战役(H1–H4)

研究问题:在混沌延迟 CSTR(τ=100)上,是什么决定了能达到的最低 student MSE?核心结论:H1 证伪(无 L+H−1≥τ 尖锐相变,只有缓降);H2 支持(E_iter≈0.76×baseline,固定折价);H3 修正后成立(多元 log(floor) 模型 R²=0.72);H4 分层成立(H≥15 时连续蒸馏压破匹配算力 baseline 地板,H=5 无益)。报告见 `conclusion/floor_study_final_report.md`(主)与 `floor_determinants.md`(自动生成的对账版)。

#### floor_sweep.csv
- **实验归属**:地板战役主数据表
- **产生程序**:`cstr/run_floor_sweep.py`(`uv run python cstr/run.py -e floor_sweep` 或 `uv run python cstr/run_floor_sweep.py --datasets tau100 --seeds 3 --K 5`)
- **实验条件**:tau100 深挖 13 cells(L∈{20,50,100}×H∈{5,15,30} + H=15 的 L∈{40,70,85,120})×3 seeds,加 τ50/τ150 锚点(L20H15)×3 seeds=45 行;α=0.5,T=4,bins=50,epochs=30,K=5,conv_epochs=100(匹配算力 baseline)
- **基本类型**:列 `dataset,tau,periodicity,L,H,LplusH_minus_1,seed,baseline_mse,baseline_converged_mse,teacher_mse,fgl_student_mse,A_iter_mse,E_iter_mse`——每行同时记录 6 个关键量
- **读表指南**:`baseline_converged_mse`=数据地板近似;`E_iter_mse`/`baseline_mse`≈0.76(H2);`fgl_student_mse` 为一次性 FGL;按 H 分层看 H4(H=15/30 蒸馏赢,H=5 输)

#### floor_h1_baseline_heatmap.png
- **实验归属 / 产生程序**:H1 假设图(`cstr/analyze_floor.py`,`uv run python cstr/analyze_floor.py`)
- **关联数据**:`floor_sweep.csv`(tau100 子集)
- **基本类型**:baseline_mse 热力图(YlOrRd),x=H、y=L
- **读图指南**:越亮=地板越高;观察地板主要由 H 驱动、随 L 缓降

#### floor_h1_transition.png
- **实验归属 / 产生程序 / 关联数据**:同上
- **基本类型**:折线:H=15 列的 baseline_mse vs L+H−1,红色虚线标 τ_data=100
- **读图指南**:若 MG 式相变存在应在 τ=100 处陡降——实测只有缓降(172→131),H1 证伪

#### floor_h2_c_ratio.png
- **实验归属 / 产生程序 / 关联数据**:同上
- **基本类型**:箱线图:c=E_iter_mse/baseline_mse(逐 cell-seed)
- **读图指南**:箱体紧凑且均值≈0.76 → 蒸馏是"固定比例折价";CV<0.25 即支持 H2

#### floor_h3_floor_vs_teacher.png / floor_h3_logfloor_vs_H.png
- **实验归属 / 产生程序 / 关联数据**:同上(`lyapunov_tau.csv` 提供 λ)
- **基本类型**:散点+线性拟合(floor vs teacher_mse,标注 R²)/ 折线(log(floor) vs H)
- **读图指南**:裸拟合 R²=0.12 弱(地板主要由 H 驱动);固定 H 后 R²=0.47–0.73 → teacher 质量在控制 H 后是强预测器;log(floor)-H 斜率与 λ·H 量级对照

#### floor_h4_paired.png
- **实验归属 / 产生程序 / 关联数据**:同上
- **基本类型**:配对散点:x=baseline_converged_mse,y=E_iter_mse,红色虚线 y=x,标题标注配对 t 检验 p 值与均值差
- **读图指南**:点在 y=x 下方=E_iter 赢(压破地板);整体 p=0.0002 但需按 H 分层看——H≥15 蒸馏赢、H=5 反而高 3.0

---

## 八、日志(logs/)

| 文件 | 对应实验 | 内容一句话 |
|---|---|---|
| `chaotic_floor_verify.log` | `cstr/archive/verify_chaotic_iter_floor.py`(tau100,K=10,4 cells×5 seeds) | 20 个 cell-seed 的 A/E 逐条结果与耗时,末尾汇总结论 |
| `iterative_cstr_esoft_log.txt` | `cstr/sweep_iterative.py --variant E-soft --w_floor 0.2`(K=3) | E-soft 三点扫描进度 + 末尾相对降幅汇总表(E_iter vs A_iter −29%) |
| `iterative_phase0_log.txt` | `cstr/sweep_iterative.py` Phase 0 试点(variant E,K=3) | 三典型点 9 次运行进度 + 汇总表(E_iter +32~40%) |
| `iterative_phase1_anchor_log.txt` | Phase 1 锚点脚本(L20H15,n=10,K=6) | 10 seeds 进度 + 汇总(E vs A +1.0%,不显著) |
| `iterative_phase1_grid_log.txt` | Phase 1 网格脚本(5×5×2,K=6) | 50 cell-seed 逐条进度(含各 cell E/A 对比)+ 汇总表 |
| `iterative_saturation_log.txt` | K=15 饱和运行(2 点×3 seeds) | 进度 + 汇总(E_iter vs A_iter +37~42%,K=3 时点口径) |
| `param_sweep_results.txt` | `cstr/archive/param_sweep.py` | U/K/flow 参数扫描逐行 periodicity 输出(全部 ≥0.935,调参无法混沌化) |

---

## 溯源备注与已知异常

1. **`iterative_curves.png` 多次被覆盖**:Phase 0、饱和、锚点、网格四次运行的日志都记录了写该文件(tag 机制加入前的固定输出名);现内容经 git 还原为 Phase 0(K=3,3 面板)版本,后三次运行的曲线分别存于 `iterative_saturation_curves.png`、`iterative_phase1_anchor_curves.png`、`iterative_phase1_grid_curves.png`。
2. **`iterative_sweep.csv` 与 `iterative_saturation_sweep.csv` 同源不同批**:两者 round-0/A_single/E_single 行相同(同 seeds),iter 臂分别为 K=3 与 K=15;饱和运行当时直接写到 `iterative_sweep.csv` 后手工分离,commit 5c690ce 的说明("原 iterative_sweep.csv 仍为 Phase 0 数据")与最终落盘一致。
3. **未入库的产生脚本**:`iterative_phase1_{anchor,grid}` 的 CSV/PNG/日志与 `adaptive_anchor_L20H15_n5.csv` 由临时脚本产生(结果经 commit e385ce8 / d2a2070 入库,脚本本身未提交),实验条件以 `conclusion/iterative_phase1_cstr.md` 与 commit message 为准。
4. **`cstr_data_overview.png`** 未能溯源(重构前快照已存在,产生脚本未入库);内容为数据四面板概览。
5. **`adaptive_weight_results.csv` 含 A–D 四变体**,而脚本 docstring 只写 A/B/C——D 为后续补充;E 变体见第四组(archive 版脚本未含 E)。
6. **`anchor_*.csv` 的产生脚本现位于 `cstr/archive/`**,原始路径为 `cstr/exp/`(其 `RESULTS_DIR` 指向当时目录,结果文件在 commit 6a24236 重构时归并入 `cstr/results/`)。
7. **`delayed_tau_sweep_s-1_A0.3_b0.1.csv` 含 1 个 fail 行**(CanteraError),为负反馈构型的负对照数据,非数据损坏。
