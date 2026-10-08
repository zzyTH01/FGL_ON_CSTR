# 当前成果总结（2026-10-08）

## 1. 一句话结论

本次完成了 **连续回归版 PatchTST-FGL** 的实现与 GPU 试点，并修正了外部连续 baseline 的物理 MSE 反标准化错误。结果显示：

> PatchTST backbone 本身仍显著优于旧 RNN-FGL；把它接入当前连续 MSE 蒸馏框架后，新 PatchTST-FGL 相对旧 RNN-FGL 大幅降低误差，但在 `α=0.5` 下未能超过同骨干 PatchTST baseline，出现明显负迁移。

因此，当前不能声称“FGL 提升了 PatchTST”；更准确的表述是：

> 更强的 backbone 能把旧 FGL 流程的误差压低约一个数量级，但当前 teacher 信号不足以继续改进强 PatchTST baseline。

---

## 2. 已完成工作

### 2.1 PatchTST-FGL 框架实现

新增连续回归版 teacher→baseline→student 流程：

- 代码：
  - `fgl_common/patchtst_fgl.py`
  - `cstr/run_patchtst_fgl_driver.py`
- 统一入口：
  - `uv run python cstr/run.py -e patchtst_fgl`
- 任务口径：
  - 输入：真实历史窗口 `x[t:t+L]`
  - 目标：direct horizon `y[t+L+H-1]`
  - teacher 输入：真实窗口向后平移 `H-1` 步
  - teacher / baseline / student 使用同一 PatchTST backbone
  - 全程连续输入、连续输出，报告物理值 MSE
- 蒸馏损失：
  - `loss = α · MSE(student, y_true) + (1-α) · MSE(student, teacher)`
- 标准化：
  - 仅使用 train split 估计 mean/std
  - 评估时反标准化回物理单位

实验默认关闭，符合仓库中“非主线实验默认 off”的约定。

---

## 3. GPU 试点设置

- 数据：`cstr/data/data_h2o.pkl`
- 设备：`gpu` 服务器，RTX 4060 Laptop GPU
- 配置：
  - `L=20`
  - `H∈{12,15}`
  - seeds：`0..4`
  - `α=0.5`
  - `d_model=32`
  - `nhead=4`
  - `dim_feedforward=64`
  - `lr=5e-4`
  - max epochs：50
  - patience：10
- 主指标：测试集物理值 MSE

---

## 4. PatchTST-FGL 内部结果

### 4.1 汇总

| 配置 | Teacher MSE | PatchTST baseline | PatchTST-FGL student | Student / Baseline | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| L20 / H12 | 9.834e-3 ± 4.182e-4 | 2.132e-3 ± 2.301e-4 | 2.681e-3 ± 1.538e-4 | **1.258** | **+25.8% 更差** |
| L20 / H15 | 9.838e-3 ± 3.631e-4 | 1.521e-3 ± 3.270e-4 | 2.335e-3 ± 1.936e-4 | **1.536** | **+53.6% 更差** |

### 4.2 统计检验

配对 t 检验：baseline vs student。

- H12：`t=-4.18`，`p=0.0140`
- H15：`t=-5.43`，`p=0.0056`

两个配置下 PatchTST-FGL student 都显著差于同骨干 PatchTST baseline。

### 4.3 解释

当前 teacher 的物理 MSE 约为 `9.8e-3`，比 PatchTST baseline 高约一个数量级。此时使用 `α=0.5` 的较强 MSE 蒸馏，会迫使学生靠近一个更差的 teacher 输出，产生负迁移。

---

## 5. 修正外部 baseline 指标 bug

### 5.1 问题

检查外部连续 baseline 时发现，深度模型返回的 `val_mse_norm` / `test_mse_norm` 已经是 normalized MSE，但旧代码又把它当作 normalized error 传给 `_physical_mse`，导致 `y_std²` 被二次平方。

修正前逻辑等价于：

```text
physical MSE ≈ normalized MSE² × y_std²
```

正确逻辑应为：

```text
physical MSE = normalized MSE × y_std²
```

因此，旧报告中的外部 PatchTST 数值被显著低估。

### 5.2 修正结果

修正后的 5-seed PatchTST baseline 为：

| 配置 | 旧 PatchTST 数值 | 修正后 PatchTST |
|---|---:|---:|
| L20 / H12 | 2.006e-4 | **1.983e-3 ± 2.125e-4** |
| L20 / H15 | 1.560e-4 | **1.413e-3 ± 3.032e-4** |

旧值不应继续引用。

---

## 6. 修正后的横向比较

### 6.1 PatchTST vs RNN baseline

RNN baseline 来自原 bin-index 流程的训练集质心物理 MSE 重映射，其数值不受外部 baseline rescaling bug 影响。

| 配置 | RNN baseline | 修正后 PatchTST | PatchTST / RNN | PatchTST 相对降低 |
|---|---:|---:|---:|---:|
| L20 / H12 | 5.160e-2 | 1.983e-3 | **1 / 26.0** | **96.2%** |
| L20 / H15 | 4.968e-2 | 1.413e-3 | **1 / 35.2** | **97.2%** |

### 6.2 PatchTST vs 旧最佳 FGL 方法

旧 RNN-FGL 连续重映射实验中，均值最好的臂是 `E_soft_iter`。

| 配置 | 旧最佳 RNN-FGL | 修正后 PatchTST | PatchTST / best-FGL | PatchTST 相对降低 |
|---|---:|---:|---:|---:|
| L20 / H12 | 1.167e-2 | 1.983e-3 | **1 / 5.89** | **83.0%** |
| L20 / H15 | 1.174e-2 | 1.413e-3 | **1 / 8.31** | **88.0%** |

配对 t 检验：

- H12：`p=1.35e-7`
- H15：`p=4.67e-7`

因此，即使修正外部 baseline 指标后，PatchTST 本身仍然显著优于旧 RNN-FGL。

### 6.3 新 PatchTST-FGL vs 旧最佳 RNN-FGL

| 配置 | 旧最佳 RNN-FGL | 新 PatchTST-FGL | 新方法相对降低 | 新方法 / PatchTST baseline |
|---|---:|---:|---:|---:|
| L20 / H12 | 1.167e-2 | 2.681e-3 | **77.0%** | 1.352 |
| L20 / H15 | 1.174e-2 | 2.335e-3 | **80.1%** | 1.653 |

这说明更换为 PatchTST backbone 后，FGL 系列内部误差确实大幅下降；但仍未超过同骨干 baseline。

---

## 7. 当前研究表述建议

不应写成：

> PatchTST-FGL 优于 PatchTST。

也不应继续引用旧外部 baseline 数值：

> PatchTST MSE 为 `2.0e-4` / `1.6e-4`。

更稳妥的表述是：

> 在 CSTR H₂O 数据的 direct-horizon 连续回归任务中，PatchTST 本身显著优于旧 RNN baseline 和旧最佳 RNN-FGL。将其接入当前连续 MSE 蒸馏框架后，PatchTST-FGL 相对旧 RNN-FGL 误差降低约 77%–80%，但在 `α=0.5` 下仍差于同骨干 PatchTST baseline，说明当前 teacher 信号对强 backbone 的额外增益为负。

---

## 8. 产物

### 代码

- `fgl_common/patchtst_fgl.py`
- `cstr/run_patchtst_fgl_driver.py`
- `cstr/run.py`
- `fgl_common/baselines.py`

### 结果

- `cstr/results/cstr_patchtst_fgl.csv`
- `cstr/results/cstr_patchtst_fgl_summary.csv`
- `cstr/results/cstr_external_baselines.csv`
- `cstr/results/plots/cstr_patchtst_fgl_summary.png`
- `cstr/results/logs/cstr_patchtst_fgl_gpu.log`

### 相关报告

- `conclusion/patchtst_fgl_report.md`
- `conclusion/cstr_external_baseline_remap_report.md`
- `cstr/results/INDEX.md`

---

## 9. 验证状态

- 本地非 slow 测试：

```text
86 passed, 1 deselected
```

- GPU smoke test：

```text
[fgl] device = cuda
```

- GPU 正式实验已完成。

---

## 10. 下一步建议

1. **弱蒸馏扫描**
   - 测试 `α∈{0.9,0.95,0.99}`
   - 判断负迁移是否随蒸馏权重减弱而消失
2. **响应分布蒸馏**
   - 用 Gaussian NLL / Gaussian KL 替代直接 MSE 拉向 teacher
   - 避免 student 被迫复制高误差 teacher 点预测
3. **选择性蒸馏**
   - 只在 teacher 优于 baseline 或 teacher 置信度高的样本上蒸馏
   - 引入样本级 gate 或动态 `α`
4. **提高 teacher 质量**
   - 当前 teacher 比同骨干 baseline 差很多
   - 若希望 FGL 继续有效，需要构造真正更强的 near-future teacher
5. **扩展数据**
   - 将实验扩展到 delayed / aperiodic CSTR、MG、Lorenz
   - 检查负迁移是否为 CSTR H₂O 特有现象

---

## 11. Git 记录

本次相关提交包括：

- `178a40f` Add continuous PatchTST FGL experiment
- `3d35f1e` Fix continuous baseline physical MSE rescaling
- `28a1b97` Stabilize PatchTST FGL arm seeding
- `d68cbe8` Refresh corrected PatchTST baseline results
- `f72a8b5` Flag superseded external baseline MSE report
