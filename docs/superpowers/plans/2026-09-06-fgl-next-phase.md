# FGL 下一阶段研究 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 CSTR τ 家族跨例基础教师迁移、教师优化量到学生优化量的量化传递实验，以及多产物复杂 CSTR 的单通道 FGL 流程。

**Architecture:** 在 `fgl_common` 新增 transfer、teacher_transfer、multichannel 三个独立模块；在 `cstr` 新增多物种生成器；实验入口通过 `cstr/run.py` 注册为默认关闭的实验开关。结果统一写入 `cstr/results/{transfer,teacher_transfer,complex}/` 并附带 summary。

**Tech Stack:** Python 3.11、PyTorch、NumPy、pandas、Cantera >= 3.2.0、pytest。

**Spec:** `docs/superpowers/specs/2026-09-06-fgl-next-phase-design.md`

## Global Constraints

- 使用仓库现有 Python 3.11 / uv 环境。
- 新实验默认关闭：`enabled=False`。
- 不修改现有 FGL 模型输入格式；多通道第一版按通道独立建模。
- 跨例实验必须 leave-one-example-out，目标例不进入源例预训练或早停。
- 师生传递实验学生超参冻结，教师和学生逐 seed 配对。
- 结果路径与命名遵循 `cstr/results/...` 与 `{domain}_{experiment}[_{variant}]` 规则。
- Cantera 生成器必须检查版本 >= 3.2.0；缺失时显式报错。

---

### Task 1: 跨例迁移核心工具

**Files:**
- Create: `fgl_common/transfer.py`
- Test: `tests/fgl_common/test_transfer.py`

**Interfaces:**
- Produces:
  - `list_delayed_cstr_datasets(data_dir: str) -> list[str]`
  - `split_lopo(paths: list[str], target: str) -> tuple[list[str], str]`
  - `build_pretrained_teacher(paths: list[str], H: int = 1, hidden: int = 128, epochs: int = 3, seed: int = 42, device: torch.device | None = None) -> torch.nn.Module`
  - `evaluate_teacher_transfer(path: str, pretrained: torch.nn.Module, H: int = 1, hidden: int = 128, finetune_epochs: int = 3, scratch_epochs: int = 3, num_bins: int = 50, seed: int = 42) -> dict`
  - 返回字典键：`path, condition, teacher_mse`，其中 `condition ∈ {zero_shot, finetuned, scratch}`。

- [ ] **Step 1: Write failing tests**

```python
import torch
from fgl_common.transfer import evaluate_teacher_transfer, split_lopo


def test_split_lopo_excludes_target():
    paths = ["tau10.pkl", "tau20.pkl", "tau30.pkl"]
    source, target = split_lopo(paths, "tau20.pkl")
    assert target == "tau20.pkl"
    assert source == ["tau10.pkl", "tau30.pkl"]


def test_evaluate_teacher_transfer_conditions(tmp_path):
    N, L, bins = 260, 8, 10
    for tau in (10, 20):
        x = torch.linspace(0, 1, N + L).unsqueeze(1) + tau / 100
        torch.save(x, tmp_path / f"tau{tau}.pkl")
    teacher = torch.nn.Linear(L, bins)
    rows = evaluate_teacher_transfer(str(tmp_path / "tau10.pkl"), teacher, L=L, H=1, hidden=8, finetune_epochs=1, scratch_epochs=1, num_bins=bins, seed=0)
    assert {r["condition"] for r in rows} == {"zero_shot", "finetuned", "scratch"}
    assert all(0 <= r["teacher_mse"] < bins ** 2 for r in rows)
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run pytest tests/fgl_common/test_transfer.py -q`

Expected: FAIL, `ModuleNotFoundError: No module named 'fgl_common.transfer'`.

- [ ] **Step 3: Implement minimal transfer module**

```python
from pathlib import Path
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import create_time_series_dataset
from .models import RNN
from .training import compute_shared_bin_edges, evaluate


def list_delayed_cstr_datasets(data_dir):
    return sorted(str(p) for p in Path(data_dir).glob("data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl"))


def split_lopo(paths, target):
    return [p for p in paths if Path(p).resolve() != Path(target).resolve()], target


def _train(model, loader, L, epochs, lr=1e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for _, x, y in loader:
            x = x.float().to(next(model.parameters()).device).view(-1, 1, L)
            opt.zero_grad()
            ce(model(x), y.long().to(next(model.parameters()).device)).backward()
            opt.step()
    model.eval()
    return model


def build_pretrained_teacher(paths, H=1, hidden=128, epochs=3, seed=42, device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = None
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        data = torch.as_tensor(data, dtype=torch.float32)
        L = 20
        edges, _, _ = compute_shared_bin_edges(data.numpy(), L, 50)
        train, val, test, _, _ = create_time_series_dataset(data, L, H, 50, offset=H - 1, batch_size=64, bin_edges=edges)
        if model is None:
            model = RNN(L, hidden, 50, 2).to(device or next(model.parameters()).device if model else torch.device("cpu"))
        _train(model, train, L, epochs)
    return model


def evaluate_teacher_transfer(path, pretrained, L=20, H=1, hidden=128, finetune_epochs=3, scratch_epochs=3, num_bins=50, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = torch.as_tensor(data, dtype=torch.float32)
    edges, _, _ = compute_shared_bin_edges(data.numpy(), L, num_bins)
    train, val, test, _, _ = create_time_series_dataset(data, L, H, num_bins, offset=H - 1, batch_size=64, bin_edges=edges)

    frozen = copy.deepcopy(pretrained).eval()
    zero = evaluate(frozen, test, L)

    finetuned = copy.deepcopy(pretrained)
    _train(finetuned, train, L, finetune_epochs)
    fine = evaluate(finetuned, test, L)

    torch.manual_seed(seed + 1)
    scratch = RNN(L, hidden, num_bins, 2)
    _train(scratch, train, L, scratch_epochs)
    from_scratch = evaluate(scratch, test, L)

    return [
        {"path": path, "condition": "zero_shot", "teacher_mse": zero},
        {"path": path, "condition": "finetuned", "teacher_mse": fine},
        {"path": path, "condition": "scratch", "teacher_mse": from_scratch},
    ]
```

- [ ] **Step 4: Run tests and fix until passing**

Run: `uv run pytest tests/fgl_common/test_transfer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fgl_common/transfer.py tests/fgl_common/test_transfer.py
git commit -m "feat(transfer): add leave-one-example-out teacher transfer"
```

### Task 2: 教师优化传递实验

**Files:**
- Create: `fgl_common/teacher_transfer.py`
- Test: `tests/fgl_common/test_teacher_transfer.py`

**Interfaces:**
- Produces:
  - `TEACHER_CONDITIONS: tuple[str, ...] = ("baseline", "more_epochs", "lower_lr", "ensemble", "regularized")`
  - `train_teachers(data, L=20, H=15, num_bins=50, seeds=(0, 1, 2), epochs=3, batch_size=64) -> dict[str, object]`
  - `run_teacher_student_transfer(data, L=20, H=15, num_bins=50, seed=0, epochs=3, batch_size=64) -> list[dict]`
  - 每行至少包含 `condition, seed, teacher_gain, student_gain, transfer_ratio, teacher_relative_gain, student_relative_gain, normalized_transfer`。

- [ ] **Step 1: Write failing test**

```python
import torch
from fgl_common.teacher_transfer import run_teacher_student_transfer


def test_transfer_rows_and_ratio():
    N, L, H = 260, 8, 3
    x = torch.linspace(0, 1, N + L).unsqueeze(1)
    data = x.numpy()
    rows = run_teacher_student_transfer(data, L=L, H=H, num_bins=10, seed=0, epochs=1, batch_size=32)
    conditions = {r["condition"] for r in rows}
    assert {"baseline", "more_epochs", "lower_lr", "ensemble", "regularized"} <= conditions
    for row in rows:
        if row["teacher_gain"] > 1e-8:
            assert abs(row["transfer_ratio"] - row["student_gain"] / row["teacher_gain"]) < 1e-6
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run pytest tests/fgl_common/test_teacher_transfer.py -q`

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement paired teacher/student experiment**

实现策略：

- 每个条件训练教师；`more_epochs` epoch 翻倍，`lower_lr` 使用 `lr/10`，`regularized` 使用 `weight_decay=1e-3`，`ensemble` 对三个 seed 的 softmax 概率取均值后等价为平均 logits；
- 先训练无蒸馏 H 步 baseline；
- 对每个教师条件使用统一 `run_fgl_experiment` 得到学生 MSE；
- 教师基线 MSE 来自 `baseline` 条件；
- 学生基线 MSE 来自 `baseline` 教师蒸馏前无蒸馏学生；
- 所有比率仅在分母绝对值大于 `1e-8` 时计算，否则为 `None`；
- 保留 `teacher_mse`、`student_mse` 原始值，避免只用比率判断。

- [ ] **Step 4: Run test and verify pass**

Run: `uv run pytest tests/fgl_common/test_teacher_transfer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fgl_common/teacher_transfer.py tests/fgl_common/test_teacher_transfer.py
git commit -m "feat(teacher-transfer): add quantified teacher-to-student transfer"
```

### Task 3: 迁移与传递结果分析

**Files:**
- Create: `fgl_common/transfer_analysis.py`
- Test: `tests/fgl_common/test_transfer_analysis.py`

**Interfaces:**
- Produces:
  - `summarize_transfer(rows: list[dict], group_keys: tuple[str, ...] = ("path", "condition")) -> list[dict]`
  - `compute_transfer_ratios(rows: list[dict]) -> list[dict]`
  - 输出字段包含 `n, mean, std, ci_low, ci_high`。

- [ ] **Step 1: Write failing test**

```python
from fgl_common.transfer_analysis import compute_transfer_ratios, summarize_transfer


def test_summary_and_ratio():
    rows = [
        {"path": "a", "condition": "zero_shot", "teacher_mse": 2.0, "seed": 0},
        {"path": "a", "condition": "zero_shot", "teacher_mse": 4.0, "seed": 1},
    ]
    summary = summarize_transfer(rows)
    assert summary[0]["n"] == 2
    assert summary[0]["mean"] == 3.0
    paired = [{"teacher_gain": 2.0, "student_gain": 1.0}, {"teacher_gain": 4.0, "student_gain": 2.0}]
    ratios = compute_transfer_ratios(paired)
    assert all(abs(r["transfer_ratio"] - 0.5) < 1e-12 for r in ratios)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/fgl_common/test_transfer_analysis.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement summaries**

- 使用均值、标准差和 seed bootstrap 95% percentile CI；
- `compute_transfer_ratios` 汇总聚合增益而非逐样本比率；
- 分母保护：`abs(teacher_gain) <= 1e-8` 时 `transfer_ratio=None`。

- [ ] **Step 4: Run tests and verify pass**

Run: `uv run pytest tests/fgl_common/test_transfer_analysis.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fgl_common/transfer_analysis.py tests/fgl_common/test_transfer_analysis.py
git commit -m "feat(analysis): add transfer and teacher-gain summaries"
```

### Task 4: CSTR 运行入口注册

**Files:**
- Modify: `cstr/run.py`
- Modify: `cstr/results/INDEX.md`
- Test: `tests/cstr/test_run_next_phase_entries.py`

**Interfaces:**
- Consumes: Task 1–3 的模块函数。
- Produces:
  - EXPERIMENTS 新键：`transfer_generalization`, `teacher_student_relation`, `complex_multichannel`；
  - 三项默认 `enabled=False`；
  - 每个 driver 接受 `args.quick` 并输出对应 CSV。

- [ ] **Step 1: Write failing entry test**

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cstr"))
import run


def test_next_phase_entries_registered_and_disabled():
    expected = {"transfer_generalization", "teacher_student_relation", "complex_multichannel"}
    assert expected <= set(run.EXPERIMENTS)
    assert all(run.EXPERIMENTS[k]["enabled"] is False for k in expected)
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run pytest tests/cstr/test_run_next_phase_entries.py -q`

Expected: FAIL.

- [ ] **Step 3: Add experiment drivers and register entries**

- `transfer_generalization`：扫描 13 个 τ，默认 `seeds=3`，输出 `cstr/results/transfer/transfer_teacher_lopo.csv`；
- `teacher_student_relation`：默认标准 CSTR `L=20,H=15`、`seeds=5`，输出 `cstr/results/teacher_transfer/teacher_optimization_transfer.csv`；
- `complex_multichannel`：第一版检查 `cstr/data/complex/` 下已生成通道，逐通道调用现有 FGL；
- 均实现 `--quick` 缩减参数：迁移 3 例 × 1 seed；传递 2 条件 × 1 seed；多通道 1 τ × 1 channel × 1 seed。

- [ ] **Step 4: Run entry and existing smoke tests**

Run: `uv run pytest tests/cstr/test_run_next_phase_entries.py -q`  
Run: `python cstr/run.py --list`

Expected: 测试 PASS；`--list` 显示三个新开关为 off。

- [ ] **Step 5: Commit**

```bash
git add cstr/run.py cstr/results/INDEX.md tests/cstr/test_run_next_phase_entries.py
git commit -m "feat(cstr): register next-phase experiment entries"
```

### Task 5: 复杂 CSTR 多物种生成器

**Files:**
- Create: `cstr/generate_complex.py`
- Test: `tests/cstr/test_generate_complex.py`

**Interfaces:**
- Produces:
  - `select_channels(species_names, species_arrays, temperature, k=6) -> list[str]`
  - `save_multichannel(path, time, temperature, species_names, species_arrays, metadata) -> None`
  - `export_single_channel_pkls(npz_path, outdir) -> list[str]`
  - CLI: `uv run python cstr/generate_complex.py --tau 50 --t-end 300 --dt 0.1 --k 6`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest
import torch
from cstr.generate_complex import export_single_channel_pkls, select_channels, save_multichannel


def test_select_channels_includes_temperature_and_h2o():
    names = ["O2", "H2", "H2O", "OH"]
    arrays = [np.zeros(10), np.ones(10) * 0.1, np.linspace(0, 2, 10), np.ones(10) * 0.01]
    selected = select_channels(names, arrays, np.linspace(300, 400, 10), k=3)
    assert "H2O" in selected and "temperature" in selected


def test_save_and_export(tmp_path):
    n = 20
    metadata = {"tau": 50, "mechanism": "h2o2.yaml"}
    save_multichannel(tmp_path / "complex.npz", np.arange(n), np.linspace(300, 400, n), ["H2O"], [np.linspace(0, 1, n)], metadata)
    files = export_single_channel_pkls(tmp_path / "complex.npz", tmp_path)
    assert len(files) == 2
    obj = torch.load(files[-1], weights_only=False)
    assert obj.shape == (n, 1)


def test_generate_complex_import_error(monkeypatch):
    import cstr.generate_complex as g
    monkeypatch.setattr(g, "ct", None)
    with pytest.raises(RuntimeError, match="Cantera"):
        g.simulate_complex_cstr(tau=50, t_end=1.0, dt=0.1)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/cstr/test_generate_complex.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement generator**

- 复用 `cstr/generate_delayed_stable.py` 的稳定控制律和 H2O2 机制；
- `simulate_complex_cstr` 返回完整 `species_names` 和所有物种时间序列；
- NPZ 保存 `time, temperature, species_names, <species arrays>, metadata_json`；
- `export_single_channel_pkls` 为 temperature 和每个保存的物种导出现有 FGL 兼容的 `(N,1)` tensor；
- 目录默认 `cstr/data/complex/`，生成器不覆盖已有文件除非传入 `--overwrite`。

- [ ] **Step 4: Run tests and verify pass**

Run: `uv run pytest tests/cstr/test_generate_complex.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cstr/generate_complex.py tests/cstr/test_generate_complex.py
git commit -m "feat(cstr): add complex multichannel generator"
```

### Task 6: 多通道 FGL 批处理

**Files:**
- Create: `fgl_common/multichannel.py`
- Test: `tests/fgl_common/test_multichannel.py`

**Interfaces:**
- Consumes: `run_fgl_experiment`。
- Produces:
  - `list_complex_channels(data_dir: str, tau: int | None = None) -> list[dict]`
  - `run_channel_grid(channel_paths: list[str], cells: list[tuple[int, int]], seeds: list[int], out_csv: str, quick: bool = False) -> list[dict]`
  - 每行字段：`path, channel, tau, L, H, seed, baseline_mse, teacher_mse, student_mse, fgl_delta`。

- [ ] **Step 1: Write failing test**

```python
from fgl_common.multichannel import run_channel_grid


def test_channel_grid(tmp_path, monkeypatch):
    import torch
    from fgl_common import multichannel
    N = 240
    path = tmp_path / "data_complex_cstr_H2O_tau50.pkl"
    torch.save(torch.linspace(0, 1, N).unsqueeze(1), path)
    monkeypatch.setattr(multichannel, "run_fgl_experiment", lambda *a, **k: {"lookback": k.get("lookback_window"), "horizon": k.get("forecasting_horizon"), "teacher": 1.0, "baseline": 2.0, "student": 1.5, "improvement": 25.0})
    rows = run_channel_grid([str(path)], [(4, 2)], [0], str(tmp_path / "out.csv"), quick=True)
    assert rows[0]["channel"] == "H2O"
    assert rows[0]["fgl_delta"] == 25.0
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run pytest tests/fgl_common/test_multichannel.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement channel runner**

- 从文件名解析 `channel` 和 `tau`；
- 调用 `run_fgl_experiment(..., regression=False)`；
- 写 pandas CSV；
- `quick=True` 缩短 epochs 并允许 monkeypatch 测试。

- [ ] **Step 4: Run test and verify pass**

Run: `uv run pytest tests/fgl_common/test_multichannel.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fgl_common/multichannel.py tests/fgl_common/test_multichannel.py
git commit -m "feat(multichannel): run FGL independently per complex CSTR channel"
```

### Task 7: 全量验证与研究文档

**Files:**
- Modify: `cstr/results/INDEX.md`
- Create: `conclusion/next_phase_experiments.md`

**Interfaces:**
- Consumes: Task 1–6 生成器与运行入口。

- [ ] **Step 1: Run all relevant tests**

Run: `uv run pytest tests -q`

Expected: PASS with no regressions.

- [ ] **Step 2: Run quick experiments**

Run:  
`python cstr/run.py -e transfer_generalization --quick`  
`python cstr/run.py -e teacher_student_relation --quick`  
`python cstr/run.py -e complex_multichannel --quick`

Expected: 每个实验生成 CSV；无静默空结果。

- [ ] **Step 3: Generate summaries**

- 从迁移表比较三条件均值与 95% CI；
- 从传递表报告教师增益→学生增益相关性和聚合传递率；
- 从多通道表比较温度、H2O 与其他物种的 FGL 增益；
- 写入 `conclusion/next_phase_experiments.md`。

- [ ] **Step 4: Update INDEX files**

- 记录每个 CSV 的实验归属、产生程序、条件、类型和读表指南。

- [ ] **Step 5: Final verification and commit**

Run: `uv run pytest tests -q`  
Run: `git status --short`

Expected: 测试 PASS；工作区仅包含本阶段文件。  
Commit:

```bash
git add fgl_common cstr tests conclusion docs
git commit -m "feat: add next-phase FGL research infrastructure"
```
