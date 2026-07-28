# 迭代自适应蒸馏(Iterative Adaptive Distillation)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把单趟自适应蒸馏(变体 E)扩展为暖启动迭代——每轮用上一轮 student 重估 teacher−student MSE 差距权重再蒸馏一次——并用 2×2 四臂因子设计(共享 round-0)验证"迭代是否优于单趟、且净收益不来自'多训练'"。先跑 Phase 0 试点(3 典型点)拿信号。

**Architecture:** 在 `fgl_common/training.py` 新增 `run_iterative_distillation(...)`:训练一次共享的 teacher / baseline / round-0 student_0,然后分支跑四臂(A-single / E-single / A-iter / E-iter)。每臂由私有 `_iterate_student(...)` 暖启动循环实现,停止规则抽成纯函数 `_should_stop(...)`。权重计算抽成 `_compute_arm_weights(...)`(A=均匀、E=gap 零地板放大)。复用现有 `compute_weights` / `KL_weighted` / `compute_per_sample_mse` / `EarlyStopper` / `evaluate`。CSTR 侧加 `run.py` 开关 + `sweep_iterative.py`。

**Tech Stack:** Python 3.11, PyTorch 2.1.1, pytest(本仓库首次引入), uv。

## Global Constraints

- 测试强制 CPU:conftest 设 `FGL_DEVICE=cpu`(必须在 import fgl_common 前)。
- 复用现有辅助函数,不重写:`compute_weights`、`KL_weighted`、`compute_per_sample_mse`、`EarlyStopper`、`evaluate`、`compute_shared_bin_edges`、`create_time_series_dataset`、`RNN`。
- teacher 用 `offset=H−1`,teacher 逐样本误差按 `int(j−(H−1))` 重映射到 student 索引空间(已验证的对齐修复)。
- 停止判定在 **val MSE** 上;返回 **keep-best-by-val** 的 student;`student_mse` 报告该最优 student 的 **test** MSE(防泄漏)。
- 变体 E 的 `W_MAX=4.0` 沿用 `compute_weights` 内置值(调参属未来工作,本计划不暴露该参数)。
- 每臂返回 `mse_curve_val` / `mse_curve_test`(逐轮),test 仅记录、不参与停止。
- 提交规范:中文 commit message + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` 尾注。

## File Structure

| 文件 | 责任 | 改动 |
|------|------|------|
| `tests/conftest.py` | 强制测试在 CPU 运行 | 新建 |
| `tests/fgl_common/test_iterative_distillation.py` | 纯函数 + 结构 + 共享不变量测试 | 新建 |
| `fgl_common/training.py` | `_should_stop` / `_compute_arm_weights` / `_iterate_student` / `run_iterative_distillation` | 新增(追加到文件末尾) |
| `fgl_common/__init__.py` | 导出 `run_iterative_distillation` | 改 |
| `cstr/run.py` | `run_iterative_distill_exp` + EXPERIMENTS 开关 + CLI 参数 | 改 |
| `cstr/sweep_iterative.py` | L×H / 显式 cells × seeds × 4 臂 扫描 + CSV + 热力图 + 逐轮曲线 | 新建 |

---

## Task 1: pytest 基建 + 纯函数 `_should_stop`

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fgl_common/test_iterative_distillation.py`
- Modify: `fgl_common/training.py`(在 `run_adaptive_inference` 之前插入 `_should_stop`,约 line 602 处)
- Modify: `pyproject.toml`(加 pytest dev 依赖 + `[tool.pytest.ini_options]`)

**Interfaces:**
- Produces: `_should_stop(mse_history: list[float], eps: float, N_stall: int, max_rounds: int) -> tuple[bool, str]`,返回 `(stop, reason)`,`reason ∈ {"cap","degradation","stall","continue"}`。

- [ ] **Step 1: 加 pytest 依赖与配置**

Run:
```bash
uv add --dev pytest
```

在 `pyproject.toml` 末尾追加(若已有 `[tool.pytest.ini_options]` 则合并):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: 建 conftest 强制 CPU**

`tests/conftest.py`:
```python
import os

# 必须在 fgl_common 被 import 之前设置,使 training.device 解析为 CPU。
os.environ.setdefault("FGL_DEVICE", "cpu")
```

- [ ] **Step 3: 写失败测试**

`tests/fgl_common/test_iterative_distillation.py`:
```python
from fgl_common.training import _should_stop


def test_stop_cap():
    # 4 条记录 => t=3 >= max_rounds=3
    assert _should_stop([1.0, 0.9, 0.8, 0.7], eps=0.01, N_stall=2, max_rounds=3) == (True, "cap")


def test_stop_degradation():
    # 0.8 -> 0.75 上升
    assert _should_stop([1.0, 0.9, 0.8, 0.75], eps=0.01, N_stall=2, max_rounds=5) == (True, "degradation")


def test_stop_stall():
    # 0.50->0.496 (0.8%), 0.496->0.492 (0.8%),均 < 1%
    assert _should_stop([0.50, 0.496, 0.492], eps=0.01, N_stall=2, max_rounds=5) == (True, "stall")


def test_stop_continue_big_improvement():
    assert _should_stop([1.0, 0.5], eps=0.01, N_stall=2, max_rounds=5) == (False, "continue")


def test_stop_round0_continues():
    assert _should_stop([1.0], eps=0.01, N_stall=2, max_rounds=5) == (False, "continue")


def test_stop_stall_reset_by_big_improvement():
    # 0.50->0.30 (大), 0.30->0.296 (停滞) => 只 1 次停滞,不够 N_stall=2
    assert _should_stop([0.50, 0.30, 0.296], eps=0.01, N_stall=2, max_rounds=5) == (False, "continue")
```

- [ ] **Step 4: 跑测试确认失败**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -v`
Expected: FAIL — `ImportError: cannot import name '_should_stop'`

- [ ] **Step 5: 实现 `_should_stop`**

在 `fgl_common/training.py` 的 `run_adaptive_weight` 与 `run_adaptive_inference` 之间(约 line 602,`# == Inference-time adaptive blending ==` 注释之前)插入:
```python
def _should_stop(mse_history, eps, N_stall, max_rounds):
    """迭代蒸馏停止规则(纯函数)。

    Args:
        mse_history: 逐轮 val MSE 列表;index 0 = round-0(初始 student),
            index t = 第 t 轮后。len == 已完成轮数 + 1。
        eps: 相对改进低于此值视为"停滞"。
        N_stall: 连续停滞达此次数则停。
        max_rounds: 轮数上限(K);t >= max_rounds 即停。
    Returns:
        (stop, reason),reason ∈ {"cap","degradation","stall","continue"}。
    """
    t = len(mse_history) - 1
    if t >= max_rounds:
        return True, "cap"
    if t == 0:
        return False, "continue"
    cur, prev = mse_history[t], mse_history[t - 1]
    if cur > prev:
        return True, "degradation"
    stall = 0
    for s in range(t, 0, -1):
        p, c = mse_history[s - 1], mse_history[s]
        if p <= 0 or c > p:
            break
        if (p - c) / p < eps:
            stall += 1
        else:
            break
    if stall >= N_stall:
        return True, "stall"
    return False, "continue"
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: 提交**

```bash
git add tests/conftest.py tests/fgl_common/test_iterative_distillation.py fgl_common/training.py pyproject.toml uv.lock
git commit -m "feat: 迭代蒸馏停止规则 _should_stop + pytest 基建

纯函数:cap/degradation/stall/continue 四态。引入 pytest(仓库首个测试套件),
FGL_DEVICE=cpu 隔离数值非确定性。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: 权重计算 `_compute_arm_weights`

**Files:**
- Modify: `fgl_common/training.py`(紧接 `_should_stop` 之后)
- Modify: `tests/fgl_common/test_iterative_distillation.py`(追加测试)

**Interfaces:**
- Consumes: `compute_per_sample_mse(model, loader, L) -> dict`, `compute_weights(variant, student_errors, teacher_errors, student_train_indices) -> (weights, raw, normalized)`。
- Produces: `_compute_arm_weights(variant, student, teacher, student_train_full, teacher_train_full, student_train_indices, L, H) -> dict[idx, float]`。

- [ ] **Step 1: 写失败测试**

追加到 `tests/fgl_common/test_iterative_distillation.py`:
```python
import pytest
import torch
from fgl_common import RNN
from fgl_common.training import _compute_arm_weights, device


def _tiny_series(n=400, seed=0):
    """平滑周期序列 [(x, y), ...],x=y=series(自回归)。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    series = np.sin(t * 0.3) * 50.0 + 100.0 + 2.0 * rng.randn(n)
    return [(float(series[i]), float(series[i])) for i in range(n)]


@pytest.fixture(scope="module")
def tiny_loaders():
    from fgl_common import compute_shared_bin_edges, create_time_series_dataset
    L, H, num_bins = 20, 15, 50
    data = _tiny_series()
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    student_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=0, batch_size=1, bin_edges=bin_edges)
    teacher_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=H - 1, batch_size=1, bin_edges=bin_edges)
    indices = [idx[0].item() for idx, _, _ in student_train_full]
    return dict(L=L, H=H, num_bins=num_bins,
                sf=student_train_full, tf=teacher_train_full, indices=indices)


def test_weights_A_uniform(tiny_loaders):
    L, H, nb = tiny_loaders["L"], tiny_loaders["H"], tiny_loaders["num_bins"]
    student = RNN(L, 16, nb, 1).to(device)
    teacher = RNN(L, 16, nb, 1).to(device)
    w = _compute_arm_weights("A", student, teacher,
                             tiny_loaders["sf"], tiny_loaders["tf"],
                             tiny_loaders["indices"], L, H)
    assert set(w) == set(tiny_loaders["indices"])
    assert all(v == 1.0 for v in w.values())


def test_weights_E_bounded(tiny_loaders):
    L, H, nb = tiny_loaders["L"], tiny_loaders["H"], tiny_loaders["num_bins"]
    student = RNN(L, 16, nb, 1).to(device)
    teacher = RNN(L, 16, nb, 1).to(device)
    w = _compute_arm_weights("E", student, teacher,
                             tiny_loaders["sf"], tiny_loaders["tf"],
                             tiny_loaders["indices"], L, H)
    assert set(w) == set(tiny_loaders["indices"])
    assert all(0.0 <= v <= 4.0 for v in w.values())  # compute_weights E 分支恒落于 [0, W_MAX=4]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py::test_weights_A_uniform -v`
Expected: FAIL — `cannot import name '_compute_arm_weights'`

- [ ] **Step 3: 实现 `_compute_arm_weights`**

紧接 `_should_stop` 之后插入:
```python
def _compute_arm_weights(variant, student, teacher, student_train_full,
                         teacher_train_full, student_train_indices, L, H):
    """单臂逐样本蒸馏权重(对齐目标)。

    variant='A' -> 恒为 1.0(对照臂:从不更新权重)。
    variant='E' -> max(0, se_student − se_teacher) 差距,零地板放大到 [0, W_MAX=4]
                   (由 compute_weights 处理)。
    teacher loader 因 offset=H−1,其原始 idx j 对齐到 student idx j−(H−1),此处重映射。
    """
    if variant == "A":
        return {idx: 1.0 for idx in student_train_indices}
    se = compute_per_sample_mse(student, student_train_full, L)
    te_raw = compute_per_sample_mse(teacher, teacher_train_full, L)
    te = {int(j - (H - 1)): e for j, e in te_raw.items()}
    weights, _, _ = compute_weights("E", se, te, student_train_indices)
    return weights
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: 提交**

```bash
git add fgl_common/training.py tests/fgl_common/test_iterative_distillation.py
git commit -m "feat: _compute_arm_weights(A 均匀 / E 差距零地板放大)

抽离单臂权重计算,A→恒 1.0、E→compute_weights('E')。teacher 索引按 j-(H-1) 对齐。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: 单臂暖启动循环 `_iterate_student`

**Files:**
- Modify: `fgl_common/training.py`(紧接 `_compute_arm_weights` 之后)
- Modify: `tests/fgl_common/test_iterative_distillation.py`(追加测试)

**Interfaces:**
- Consumes: `_should_stop`, `_compute_arm_weights`, `KL_weighted(student_logits, teacher_logits, temperature, alpha, sample_weights)`, `EarlyStopper`, `evaluate(model, loader, L)`, `compute_shared_bin_edges`, `create_time_series_dataset`, `RNN`, `device`。
- Produces: `_iterate_student(student_0, teacher, variant, max_rounds, student_train, teacher_train, student_val, student_test, student_train_full, teacher_train_full, student_train_indices, L, H, alpha, temperature, round_epochs, patience, eps, N_stall, lr) -> dict` with keys `{rounds_used, total_epochs, mse_curve_val, mse_curve_test, student}`。

- [ ] **Step 1: 写失败测试**

追加:
```python
from fgl_common import compute_shared_bin_edges, create_time_series_dataset
from fgl_common.training import _iterate_student, evaluate


@pytest.fixture(scope="module")
def tiny_setup(tiny_loaders):
    """训练一个共享 teacher + round-0 student_0 供 _iterate_student 测试。"""
    L, H, nb = tiny_loaders["L"], tiny_loaders["H"], tiny_loaders["num_bins"]
    data = _tiny_series()
    bin_edges, _, _ = compute_shared_bin_edges(data, L, nb)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=nb,
        val_size=0.2, test_size=0.2, offset=0, batch_size=8, bin_edges=bin_edges)
    teacher_train, teacher_val, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=nb,
        val_size=0.2, test_size=0.2, offset=H - 1, batch_size=8, bin_edges=bin_edges)
    torch.manual_seed(0)
    teacher = RNN(L, 16, nb, 1).to(device)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for _ in range(3):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            opt.zero_grad(); ce(teacher(x), y.long().to(device)).backward(); opt.step()
    teacher.eval()
    # round-0 student(uniform KL,from scratch)
    torch.manual_seed(1)
    student_0 = RNN(L, 16, nb, 1).to(device)
    from fgl_common import KL
    opt = torch.optim.Adam(student_0.parameters(), lr=1e-3)
    for _ in range(3):
        student_0.train()
        for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
            xs = xs.float().to(device).view(-1, 1, L)
            out = student_0(xs)
            xt = xt.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                tlog = teacher(xt)
            loss = 0.5 * ce(out, ys.long().to(device)) + KL(out, tlog, 4.0, 0.5)
            opt.zero_grad(); loss.backward(); opt.step()
    student_0.eval()
    return dict(teacher=teacher, student_0=student_0,
                student_train=student_train, teacher_train=teacher_train,
                student_val=student_val, student_test=student_test,
                sf=tiny_loaders["sf"], tf=tiny_loaders["tf"],
                indices=tiny_loaders["indices"], L=L, H=H, nb=nb)


def test_iterate_student_structure(tiny_setup):
    s = tiny_setup
    res = _iterate_student(
        s["student_0"], s["teacher"], "A", max_rounds=2,
        student_train=s["student_train"], teacher_train=s["teacher_train"],
        student_val=s["student_val"], student_test=s["student_test"],
        student_train_full=s["sf"], teacher_train_full=s["tf"],
        student_train_indices=s["indices"],
        L=s["L"], H=s["H"], alpha=0.5, temperature=4.0,
        round_epochs=2, patience=5, eps=0.01, N_stall=2, lr=1e-3)
    assert {"rounds_used", "total_epochs", "mse_curve_val", "mse_curve_test", "student"} <= set(res)
    assert res["rounds_used"] <= 2
    assert len(res["mse_curve_val"]) == res["rounds_used"] + 1
    assert len(res["mse_curve_test"]) == res["rounds_used"] + 1
    assert res["total_epochs"] >= res["rounds_used"]


def test_iterate_student_keeps_best_by_val(tiny_setup):
    s = tiny_setup
    res = _iterate_student(
        s["student_0"], s["teacher"], "E", max_rounds=3,
        student_train=s["student_train"], teacher_train=s["teacher_train"],
        student_val=s["student_val"], student_test=s["student_test"],
        student_train_full=s["sf"], teacher_train_full=s["tf"],
        student_train_indices=s["indices"],
        L=s["L"], H=s["H"], alpha=0.5, temperature=4.0,
        round_epochs=2, patience=5, eps=0.01, N_stall=2, lr=1e-3)
    # 返回的 student 应对应 val 最优那一轮
    best_val = min(res["mse_curve_val"])
    assert abs(evaluate(res["student"], s["student_val"], s["L"]) - best_val) < 1e-4


def test_iterate_student_maxrounds0_is_round0(tiny_setup):
    s = tiny_setup
    res = _iterate_student(
        s["student_0"], s["teacher"], "A", max_rounds=0,
        student_train=s["student_train"], teacher_train=s["teacher_train"],
        student_val=s["student_val"], student_test=s["student_test"],
        student_train_full=s["sf"], teacher_train_full=s["tf"],
        student_train_indices=s["indices"],
        L=s["L"], H=s["H"], alpha=0.5, temperature=4.0,
        round_epochs=2, patience=5, eps=0.01, N_stall=2, lr=1e-3)
    assert res["rounds_used"] == 0
    assert len(res["mse_curve_val"]) == 1
    # 与输入 student_0 的 val MSE 一致(未训练)
    assert abs(res["mse_curve_val"][0] - evaluate(s["student_0"], s["student_val"], s["L"])) < 1e-4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py::test_iterate_student_structure -v`
Expected: FAIL — `cannot import name '_iterate_student'`

- [ ] **Step 3: 实现 `_iterate_student`**

紧接 `_compute_arm_weights` 之后插入:
```python
def _iterate_student(student_0, teacher, variant, max_rounds,
                     student_train, teacher_train, student_val, student_test,
                     student_train_full, teacher_train_full, student_train_indices,
                     L, H, alpha, temperature, round_epochs, patience,
                     eps, N_stall, lr):
    """单臂暖启动迭代蒸馏。

    student_0: 已训练的 round-0 student(共享,本函数不修改其入参对象)。
    variant: 'A'(每轮恒均匀)或 'E'(每轮按当前 student 重估 gap 权重)。
    max_rounds: 该臂最大轮数(E-single=1;iter 臂=K)。
    返回 dict: {rounds_used, total_epochs, mse_curve_val, mse_curve_test, student}。
    student 为 keep-best-by-val 的模型实例。
    """
    ce = torch.nn.CrossEntropyLoss()
    student = RNN(student_0-param-shape)  # 占位,见下方真实实现
    student.load_state_dict({k: v.clone() for k, v in student_0.state_dict().items()})
    student.eval()

    mse_curve_val = [evaluate(student, student_val, L)]
    mse_curve_test = [evaluate(student, student_test, L)]
    best_val = mse_curve_val[0]
    best_state = {k: v.clone() for k, v in student.state_dict().items()}
    total_epochs = 0
    rounds_used = 0

    for r in range(1, max_rounds + 1):
        weights = _compute_arm_weights(variant, student, teacher,
                                       student_train_full, teacher_train_full,
                                       student_train_indices, L, H)
        opt = torch.optim.Adam(student.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(round_epochs):
            student.train()
            for (idx_s, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
                x_s = x_s.float().to(device).view(-1, 1, L)
                targets = y_s.long().to(device)
                outputs = student(x_s)
                x_t = x_t.float().to(device).view(-1, 1, L)
                with torch.no_grad():
                    logits = teacher(x_t)
                bw = torch.tensor([weights.get(i.item(), 1.0) for i in idx_s],
                                  dtype=torch.float32, device=device)
                loss = alpha * ce(outputs, targets) + KL_weighted(outputs, logits, temperature, alpha, bw)
                opt.zero_grad(); loss.backward(); opt.step()
            student.eval()
            with torch.no_grad():
                vl = sum(ce(student(x.float().to(device).view(-1, 1, L)),
                            y.long().to(device)).item()
                         for _, x, y in student_val) / len(student_val)
            total_epochs += 1
            if stop.step(vl, student):
                break
        stop.restore(student)
        student.eval()

        mv = evaluate(student, student_val, L)
        mse_curve_val.append(mv)
        mse_curve_test.append(evaluate(student, student_test, L))
        if mv < best_val:
            best_val = mv
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
        rounds_used = r

        stop_flag, _ = _should_stop(mse_curve_val, eps, N_stall, max_rounds)
        if stop_flag:
            break

    student.load_state_dict(best_state)
    student.eval()
    return {"rounds_used": rounds_used, "total_epochs": total_epochs,
            "mse_curve_val": mse_curve_val, "mse_curve_test": mse_curve_test,
            "student": student}
```

> ⚠️ 实现者注意:上面 `student = RNN(student_0-param-shape)` 是**伪代码占位**,真实实现必须用与 `student_0` 相同的构造新建一个 RNN 再 `load_state_dict`。由于本函数没有 hidden/layers/output 参数,改为**直接克隆入参模型**:把开头两行替换为:
> ```python
> import copy
> student = copy.deepcopy(student_0)
> ```
> (`copy.deepcopy` 对 `nn.Module` 在 CPU 上安全且保留构造信息,避免重复传构造参数。)

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 提交**

```bash
git add fgl_common/training.py tests/fgl_common/test_iterative_distillation.py
git commit -m "feat: _iterate_student 单臂暖启动循环 + keep-best-by-val

每轮:deepcopy(student_0) 暖启动 → 重估权重 → round_epochs 训练 → val 评估。
停止规则 _should_stop 驱动,返回历史 val 最优 student。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: 编排器 `run_iterative_distillation` + 导出

**Files:**
- Modify: `fgl_common/training.py`(紧接 `_iterate_student` 之后)
- Modify: `fgl_common/__init__.py`(导出)
- Modify: `tests/fgl_common/test_iterative_distillation.py`(追加集成测试)

**Interfaces:**
- Consumes: `_iterate_student`, `compute_shared_bin_edges`, `create_time_series_dataset`, `RNN`, `EarlyStopper`, `KL`, `evaluate`, `device`, `torch`, `optim`, `nn`, `np`。
- Produces: `run_iterative_distillation(data, L=20, H=15, alpha=0.5, temperature=4, num_bins=50, epochs=30, round_epochs=15, batch_size=64, patience=5, K=5, eps=0.01, N_stall=2, seed=42, variant='E', val_size=0.2, test_size=0.2, lr=1e-4, verbose=True) -> dict[arm, dict]`,arm ∈ `{"A_single","E_single","A_iter","E_iter"}`;每值含 `{teacher_mse, baseline_mse, student_mse, fgl_delta, init_delta, rounds_used, total_epochs, mse_curve_val, mse_curve_test}`。

- [ ] **Step 1: 写失败测试**

追加:
```python
from fgl_common import run_iterative_distillation


def _tiny_data():
    return _tiny_series(n=400)


def test_run_iterative_four_arms_structure():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    assert set(res) == {"A_single", "E_single", "A_iter", "E_iter"}
    expected = {"teacher_mse", "baseline_mse", "student_mse", "fgl_delta", "init_delta",
                "rounds_used", "total_epochs", "mse_curve_val", "mse_curve_test"}
    for arm, r in res.items():
        assert expected <= set(r), f"{arm} missing keys"


def test_all_arms_share_round0():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    a0 = res["A_single"]["mse_curve_test"][0]
    for arm in ("E_single", "A_iter", "E_iter"):
        assert abs(res[arm]["mse_curve_test"][0] - a0) < 1e-6, f"{arm} round-0 differs"


def test_A_single_is_round0_only():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    assert res["A_single"]["rounds_used"] == 0
    assert len(res["A_single"]["mse_curve_val"]) == 1


def test_E_single_uses_one_round():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    assert res["E_single"]["rounds_used"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/fgl_common/test_run_iterative_four_arms_structure -v`
Expected: FAIL — `ImportError: cannot import name 'run_iterative_distillation'`

- [ ] **Step 3: 实现 `run_iterative_distillation`**

紧接 `_iterate_student` 之后插入:
```python
def run_iterative_distillation(data, L=20, H=15, alpha=0.5, temperature=4, num_bins=50,
                               epochs=30, round_epochs=15, batch_size=64, patience=5,
                               K=5, eps=0.01, N_stall=2, seed=42, variant="E",
                               val_size=0.2, test_size=0.2, lr=1e-4, verbose=True):
    """迭代(暖启动)自适应蒸馏,2×2 四臂因子设计(共享 round-0)。

    训一次 teacher / baseline / student_0(round-0,uniform KL),然后分支:
      A_single = student_0 本身(不迭代)。
      E_single = 暖启 1 轮,E 权重(来自 student_0)。
      A_iter   = 暖启 ≤K 轮,权重恒均匀(归因对照)。
      E_iter   = 暖启 ≤K 轮,每轮重估 E 权重(新方法)。
    停止规则在 val MSE,返回每臂 keep-best-by-val 的 student,报告其 test MSE。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden, output, layers = 128, num_bins, 2
    ce = torch.nn.CrossEntropyLoss()

    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=H - 1, batch_size=batch_size, bin_edges=bin_edges)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=0, batch_size=batch_size, bin_edges=bin_edges)
    student_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=0, batch_size=1, bin_edges=bin_edges)
    teacher_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=H - 1, batch_size=1, bin_edges=bin_edges)
    student_train_indices = [idx[0].item() for idx, _, _ in student_train_full]

    if verbose:
        print(f"  [iter] L={L} H={H} α={alpha} T={temperature} K={K} seed={seed}")

    def _train_simple(model, loader, vloader):
        opt = optim.Adam(model.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(epochs):
            model.train()
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, L)
                opt.zero_grad(); ce(model(x), y.long().to(device)).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = sum(ce(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                         for _, x, y in vloader) / len(vloader)
            if stop.step(vl, model):
                break
        stop.restore(model); model.eval()
        return model

    teacher = _train_simple(RNN(L, hidden, output, layers).to(device), teacher_train, teacher_val)
    baseline = _train_simple(RNN(L, hidden, output, layers).to(device), student_train, student_val)

    # round-0 student: from scratch, uniform KL(标准 FGL)
    student_0 = RNN(L, hidden, output, layers).to(device)
    opt = optim.Adam(student_0.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    for _ in range(epochs):
        student_0.train()
        for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
            xs = xs.float().to(device).view(-1, 1, L)
            out = student_0(xs)
            xt = xt.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                tlog = teacher(xt)
            loss = alpha * ce(out, ys.long().to(device)) + KL(out, tlog, temperature, alpha)
            opt.zero_grad(); loss.backward(); opt.step()
        student_0.eval()
        with torch.no_grad():
            vl = sum(ce(student_0(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop.step(vl, student_0):
            break
    stop.restore(student_0); student_0.eval()

    teacher_mse = evaluate(teacher, teacher_test, L)
    baseline_mse = evaluate(baseline, student_test, L)

    common = dict(teacher=teacher, student_train=student_train, teacher_train=teacher_train,
                  student_val=student_val, student_test=student_test,
                  student_train_full=student_train_full, teacher_train_full=teacher_train_full,
                  student_train_indices=student_train_indices, L=L, H=H, alpha=alpha,
                  temperature=temperature, round_epochs=round_epochs, patience=patience,
                  eps=eps, N_stall=N_stall, lr=lr)

    def _arm(variant, max_rounds):
        if max_rounds == 0:
            return {"rounds_used": 0, "total_epochs": 0,
                    "mse_curve_val": [evaluate(student_0, student_val, L)],
                    "mse_curve_test": [evaluate(student_0, student_test, L)],
                    "student": student_0}
        return _iterate_student(student_0, teacher, variant, max_rounds, **common)

    arms = {
        "A_single": _arm("A", 0),
        "E_single": _arm("E", 1),
        "A_iter":   _arm("A", K),
        "E_iter":   _arm("E", K),
    }

    results = {}
    for name, a in arms.items():
        s_mse = evaluate(a["student"], student_test, L)
        init_mse = arms["A_single"]["mse_curve_test"][0]  # round-0 test MSE
        fgl_delta = (baseline_mse - s_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
        init_delta = (init_mse - s_mse) / init_mse * 100 if init_mse > 0 else 0
        results[name] = {"teacher_mse": teacher_mse, "baseline_mse": baseline_mse,
                         "student_mse": s_mse, "fgl_delta": fgl_delta, "init_delta": init_delta,
                         "rounds_used": a["rounds_used"], "total_epochs": a["total_epochs"],
                         "mse_curve_val": a["mse_curve_val"], "mse_curve_test": a["mse_curve_test"]}
        if verbose:
            print(f"    {name:9s}: student_mse={s_mse:.1f}  Δbase={fgl_delta:+.1f}%  "
                  f"Δinit={init_delta:+.1f}%  rounds={a['rounds_used']}  ep={a['total_epochs']}")
    return results
```

- [ ] **Step 4: 导出**

`fgl_common/__init__.py`:
- 在 `from .training import (...)` 块内追加 `run_iterative_distillation,`(按字母序放在 `run_fgl_experiment` 之前)。
- 在 `__all__` 的 training 段追加 `"run_iterative_distillation",`。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -v`
Expected: PASS (15 passed)

- [ ] **Step 6: 提交**

```bash
git add fgl_common/training.py fgl_common/__init__.py tests/fgl_common/test_iterative_distillation.py
git commit -m "feat: run_iterative_distillation 2×2 四臂编排 + 导出

共享 teacher/baseline/student_0,分支 A-single/E-single/A-iter/E-iter。
keep-best-by-val,报告 test MSE + 逐轮曲线。15 测试全过。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: 接入 `cstr/run.py`

**Files:**
- Modify: `cstr/run.py`(新增包装函数 + EXPERIMENTS 条目 + CLI 参数)

**Interfaces:**
- Consumes: `run_iterative_distillation`(从 `fgl_common`)。

- [ ] **Step 1: 加 CLI 参数**

在 `_add_common_args(p)` 内追加(`--seeds` 之后):
```python
    p.add_argument("--round_epochs", type=int, default=15, help="[iterative_distill] 每轮 epoch")
    p.add_argument("--K", type=int, default=5, help="[iterative_distill] 最大迭代轮数")
```

- [ ] **Step 2: 加包装函数**

在 `run_adaptive_weight_exp` 之后插入:
```python
def run_iterative_distill_exp(args):
    """迭代自适应蒸馏 4 臂对比(A-single / E-single / A-iter / E-iter)。"""
    data = _load_data(args.dataset)
    n_seeds = args.seeds if args.seeds else 3
    rows = []
    for s in range(n_seeds):
        arms = run_iterative_distillation(
            data, L=args.L, H=args.H, alpha=args.alpha, temperature=args.temperature,
            num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
            batch_size=args.batch_size, K=args.K, patience=args.patience, seed=s, verbose=False)
        for arm, r in arms.items():
            rows.append({"seed": s, "arm": arm, "student_mse": r["student_mse"],
                         "baseline_mse": r["baseline_mse"], "fgl_delta": r["fgl_delta"],
                         "init_delta": r["init_delta"], "rounds_used": r["rounds_used"]})

    print(f"\n{'=' * 60}\nSUMMARY: iterative_distill (L={args.L} H={args.H})\n{'=' * 60}")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("student_mse", "fgl_delta", "init_delta"):
            agg[r["arm"]][k].append(r[k])

    def _sd(a):
        a = np.array(a); return a.std(ddof=1) if len(a) > 1 else 0.0

    for arm in ("A_single", "E_single", "A_iter", "E_iter"):
        if arm not in agg:
            continue
        sm = np.array(agg[arm]["student_mse"])
        fd = np.array(agg[arm]["fgl_delta"])
        idt = np.array(agg[arm]["init_delta"])
        print(f"  {arm:9s}: student_mse={sm.mean():.1f}±{_sd(sm):.1f}  "
              f"Δbase={fd.mean():+.1f}%±{_sd(fd):.1f}  "
              f"Δinit={idt.mean():+.1f}%±{_sd(idt):.1f}  (n={len(sm)})")
```

- [ ] **Step 3: 注册实验开关**

在 `EXPERIMENTS` 字典内(`adaptive_weight` 之后)追加:
```python
    "iterative_distill": dict(fn=run_iterative_distill_exp, enabled=False,
                              note="迭代自适应蒸馏 4 臂(A-single/E-single/A-iter/E-iter);Phase 0 先跑典型点"),
```

并在文件顶部 `from fgl_common import (...)` 内追加 `run_iterative_distillation,`。

- [ ] **Step 4: 冒烟验证(1 seed,极短)**

Run:
```bash
uv run python cstr/run.py -e iterative_distill --L 20 --H 15 --epochs 3 --round_epochs 2 --K 2 --seeds 1
```
Expected: 打印 4 臂 SUMMARY,无报错。

- [ ] **Step 5: 提交**

```bash
git add cstr/run.py
git commit -m "feat: cstr/run.py 接入 iterative_distill 开关 + CLI

4 臂 n-seed 汇总打印(均值±std)。默认 enabled=False。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: `cstr/sweep_iterative.py` + Phase 0 试点

**Files:**
- Create: `cstr/sweep_iterative.py`

**Interfaces:**
- Consumes: `run_iterative_distillation`, `numpy`, `csv`, `matplotlib`。

- [ ] **Step 1: 写 sweep 脚本**

`cstr/sweep_iterative.py`:
```python
#!/usr/bin/env python
"""L×H / 显式 cells:迭代蒸馏 4 臂对比 + 逐轮 MSE 曲线。

用法::

    # Phase 0 试点:3 典型点
    uv run python cstr/sweep_iterative.py --cells "20,15;8,30;72,15" --seeds 3 --epochs 20 --round_epochs 10 --K 3
    # Phase 1 全网格
    uv run python cstr/sweep_iterative.py --grid --seeds 2 --epochs 30 --round_epochs 15 --K 5

输出 cstr/results/iterative_sweep.csv + 热力图(E-iter vs E-single、E-iter vs A-iter)+ 逐轮曲线 png。
"""
import argparse
import csv
import os
import pickle
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from fgl_common import run_iterative_distillation

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
ARMS = ("A_single", "E_single", "A_iter", "E_iter")


def _load(name="data_h2o.pkl"):
    for d in ("data", "."):
        p = os.path.join(_CSTR_DIR, d, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(name)


def _parse_cells(args):
    if args.grid:
        Ls = [8, 20, 35, 50, 72]
        Hs = [5, 15, 30, 45, 60]
        return [(L, H) for L in Ls for H in Hs]
    pairs = []
    for tok in args.cells.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        L, H = tok.split(",")
        pairs.append((int(L), int(H)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="iterative distillation 4-arm sweep")
    ap.add_argument("--cells", default="20,15;8,30;72,15", help="semicolon-separated L,H pairs")
    ap.add_argument("--grid", action="store_true", help="5x5 full grid (overrides --cells)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--round_epochs", type=int, default=10)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    cells = _parse_cells(args)
    seeds = list(range(args.seeds))
    data = _load()
    outdir = os.path.join(_CSTR_DIR, "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "iterative_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["L", "H", "seed", "arm", "baseline_mse", "student_mse", "fgl_delta", "init_delta",
             "rounds_used", "mse_curve_val", "mse_curve_test"])

    # (L,H) -> arm -> [student_mse per seed];  以及逐轮曲线
    cell_results = {}
    total = len(cells) * len(seeds)
    done = 0
    for (L, H) in cells:
        per_arm = {a: [] for a in ARMS}
        curves_val = {a: [] for a in ARMS}
        for s in seeds:
            res = run_iterative_distillation(
                data, L=L, H=H, alpha=args.alpha, temperature=args.temperature,
                num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
                K=args.K, seed=s, verbose=False)
            for arm, r in res.items():
                per_arm[arm].append(r["student_mse"])
                curves_val[arm].append(r["mse_curve_val"])
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [L, H, s, arm, r["baseline_mse"], r["student_mse"], r["fgl_delta"],
                         r["init_delta"], r["rounds_used"],
                         ";".join(f"{v:.3f}" for v in r["mse_curve_val"]),
                         ";".join(f"{v:.3f}" for v in r["mse_curve_test"])])
            done += 1
        cell_results[(L, H)] = (per_arm, curves_val)
        # 打印该 cell 的关键对比
        e_iter = np.mean(per_arm["E_iter"]); e_single = np.mean(per_arm["E_single"])
        a_iter = np.mean(per_arm["A_iter"])
        print(f"[{done}/{total}] L={L:<3} H={H:<3}: E_iter={e_iter:6.1f}  "
              f"E_single={e_single:6.1f}  A_iter={a_iter:6.1f}  "
              f"(E_iter vs A_iter {(a_iter-e_iter)/a_iter*100 if a_iter>0 else float('nan'):+.1f}%)",
              flush=True)

    _report(cell_results, seeds, outdir)


def _report(cell_results, seeds, outdir):
    print(f"\n{'=' * 70}\nE-iter 相对对照的 student MSE 下降 (%)  [+ = E-iter 更好]\n{'=' * 70}")
    print(f"{'L,H':>10} | {'vs A-single':>12} | {'vs E-single':>12} | {'vs A-iter':>10}")
    print("-" * 60)
    for (L, H), (per_arm, _) in cell_results.items():
        a_s = np.mean(per_arm["A_single"]); e_s = np.mean(per_arm["E_single"])
        a_i = np.mean(per_arm["A_iter"]); e_i = np.mean(per_arm["E_iter"])

        def rel(base):
            return (base - e_i) / base * 100 if base > 0 else float("nan")
        print(f"({L:>2},{H:<3})    | {rel(a_s):>+11.1f}% | {rel(e_s):>+11.1f}% | {rel(a_i):>+9.1f}%")
    _plot_curves(cell_results, outdir)
    print(f"\nCSV: {os.path.join(outdir, 'iterative_sweep.csv')}")


def _plot_curves(cell_results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_cells = len(cell_results)
    fig, axes = plt.subplots(1, n_cells, figsize=(5 * n_cells, 4), squeeze=False)
    for ax, ((L, H), (_, curves_val)) in zip(axes[0], cell_results.items()):
        for arm in ARMS:
            curves = curves_val[arm]
            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            arr = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            mean = np.nanmean(arr, axis=0)
            ax.plot(range(max_len), mean, marker="o", label=arm)
        ax.set_title(f"L={L}, H={H}")
        ax.set_xlabel("round (0 = round-0 student)")
        ax.set_ylabel("val MSE")
        ax.legend(fontsize=8)
    fig.suptitle("Per-round val MSE by arm (lower = better; E-iter shape answers H1)")
    fig.tight_layout()
    png = os.path.join(outdir, "iterative_curves.png")
    fig.savefig(png, dpi=120)
    print(f"逐轮曲线: {png}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟验证脚本(1 cell,1 seed,极短)**

Run:
```bash
uv run python cstr/sweep_iterative.py --cells "20,15" --seeds 1 --epochs 3 --round_epochs 2 --K 2
```
Expected: 打印 1 cell 的 E_iter/E_single/A_iter + 生成 `cstr/results/iterative_sweep.csv` 与 `iterative_curves.png`,无报错。

- [ ] **Step 3: 跑 Phase 0 试点(真实参数)**

Run:
```bash
uv run python cstr/sweep_iterative.py --cells "20,15;8,30;72,15" --seeds 3 --epochs 20 --round_epochs 10 --K 3
```
记录输出(E_iter vs A_iter 在 P1/P2/P3 的相对降幅 + 逐轮曲线形状)。

- [ ] **Step 4: 判定 Phase 0 放行门**

按 spec §7 Phase 0 放行判据核对输出:
- P1(L20H15)/ P2(L8H30):E-iter ≤ E-single **且** E-iter ≤ A-iter?
- P3(L72H15):E-iter ≈ A-iter(±噪声)**且**曲线不退化?
- 逐轮曲线:E-iter 是否单调降(H1a)/ 饱和(H1b)/ 先降后升(H1c)?

把结论写进 `conclusion/iterative_pilot_phase0.md`(新建):三 cell 的 E_iter/E_single/A_iter/A_single MSE 表 + 曲线形状判定 + 是否放行 Phase 1。

- [ ] **Step 5: 提交**

```bash
git add cstr/sweep_iterative.py conclusion/iterative_pilot_phase0.md cstr/results/iterative_sweep.csv cstr/results/iterative_curves.png
git commit -m "feat: cstr/sweep_iterative.py + Phase 0 试点(3 典型点)

4 臂 L×H/explicit-cells 扫描 + 逐轮 MSE 曲线图。Phase 0:P1(L20H15)/P2(L8H30)/
P3(L72H15)×3 seeds,K=3。记录曲线形状与放行判定。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review(写计划后自查)

**1. Spec 覆盖**
- §1 机制(暖启动每轮重估权重)→ Task 3 `_iterate_student` + Task 4 编排 ✓
- §2 四臂 2×2 共享 round-0 → Task 4 `_arm()` + `test_all_arms_share_round0` ✓
- §3 停止规则(stall/degradation/cap,keep-best)→ Task 1 `_should_stop` + Task 3 keep-best ✓
- §4 逐轮 MSE 曲线 → Task 4 `mse_curve_val/test` + Task 6 `_plot_curves` ✓
- §5 预算公平性(E-iter/A-iter 同轮次表)→ Task 4 两臂共用 `common`(同 round_epochs/K)✓
- §6 实现位置(training/__init__/run.py/sweep)→ Tasks 1–6 ✓
- §7 Phase 0 试点 + 放行判据 → Task 6 Step 3–4 ✓
- §0 H1 三结局由曲线形状判定 → Task 6 Step 4 ✓
- §8 风险防护(退化即停/keep-best/val 停止)→ Task 1+3 ✓

**2. 占位扫描**:`_iterate_student` 实现里有一处显式标注的伪代码占位(`RNN(student_0-param-shape)`),已紧跟给出真实替换(`copy.deepcopy(student_0)`)。无其他 TBD/TODO。✓

**3. 类型/命名一致性**:`_should_stop(mse_history, eps, N_stall, max_rounds)` 在 Task 1 定义、Task 3 调用,签名一致;`_compute_arm_weights` 参数顺序 Task 2 定义与 Task 3 调用一致;`run_iterative_distillation` 返回键在 Task 4 定义、Task 5/6 消费一致;`arms` 四个名字跨 Task 4/5/6 一致。✓

**未覆盖(故意,属 Phase 1 / 未来)**:Phase 1 全网格 n=10 + 5×5、MG/Lorenz 移植、W_MAX 调参(spec §10)。这些在 Phase 0 放行后另立计划。
