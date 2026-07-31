# 迭代自适应蒸馏(E-iter)纳入 FGL 框架 —— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 `run_iterative_distillation` 使其一次调用同时支持硬(E)与稍软化(E-soft)两种权重分布并产出规范臂名,同时把 `iterative_distill` 接入 cstr/mg 两域 run.py 统一入口(CSTR 默认启用、MG 保持关闭),支持 CLI 切换变体 + CSV 落盘。

**Architecture:** 核心改动集中在 `fgl_common/training.py` 的 `run_iterative_distillation`(796-914 行):把固定 4 臂组装改为"A 对照臂 + 遍历 `weight_distributions` 生成 {v}_single/{v}_iter",变体解析抽成两个纯函数以便单测。`_iterate_student`/`_compute_arm_weights`/`_should_stop`/`compute_weights` 零改动(变体名与 `w_floor` 已是透传参数)。两个域的 run.py 实验体从硬编码 4 臂名改为动态取臂并写 CSV。

**Tech Stack:** Python 3.11 / PyTorch 2.1.1 / pytest;运行方式 `uv run pytest`、`uv run python cstr/run.py`。设备 MPS(M5 Air),测试用极小模型 + 2-3 epochs 保证快。

## Global Constraints

- 向后兼容:`variant="E"`(旧默认)产出的 4 臂与重构前逐字段一致;既有 17 个测试用例必须全部保持通过。
- `variant` 与 `weight_distributions` 同时传 → `ValueError`;都不传 → 缺省 `("E",)`。
- `weight_distributions` 含 `"A"` 或空元组 → `ValueError`。
- 变体名规范化:连字符 → 下划线(`"E-soft"` → 臂名 `E_soft_single`/`E_soft_iter`)。
- `w_floor` 默认 `None`(≡ 0.2,保持旧签名);`w_floors`(dict 变体名→float)优先级更高;E 变体忽略。
- CSTR `iterative_distill` 默认 `enabled=True`;MG 保持 `enabled=False`。
- `e_iter_snapshot_fn` 只挂在**第一个**变体的 iter 臂上(多变体时其他变体无快照)。

---

### Task 1: 核心重构 `run_iterative_distillation`(双权重分布 + 规范臂名)

**Files:**
- Modify: `fgl_common/training.py`(660-714 行附近加 helper;796-914 行重构主函数)
- Test: `tests/fgl_common/test_iterative_distillation.py`

**Interfaces:**
- Consumes: `compute_weights`(已有,E/E-soft 支持)、`_iterate_student`、`_compute_arm_weights`(已有,`w_floor` 透传)。
- Produces:
  - `_resolve_variants(variant, weight_distributions) -> tuple[str, ...]` — 校验 + 缺省 + 冲突报错
  - `_resolve_w_floor(variant, w_floor, w_floors) -> float | None` — per-variant 地板覆盖
  - 重构后的 `run_iterative_distillation(..., variant=None, weight_distributions=None, w_floor=None, w_floors=None, ...) -> dict[str, dict]`

- [ ] **Step 1: 写失败测试(helper + 主函数结构)**

在 `tests/fgl_common/test_iterative_distillation.py` 末尾追加:

```python
# ==================== Task 1: 双权重分布重构 ====================
import pytest as _pytest
from fgl_common.training import _resolve_variants, _resolve_w_floor


def test_resolve_variants_default_is_E():
    assert _resolve_variants(None, None) == ("E",)


def test_resolve_variants_alias_equivalence():
    assert _resolve_variants("E-soft", None) == ("E-soft",)
    assert _resolve_variants(None, ("E", "E-soft")) == ("E", "E-soft")


def test_resolve_variants_conflict_raises():
    with _pytest.raises(ValueError):
        _resolve_variants("E", ("E", "E-soft"))


def test_resolve_variants_rejects_A_and_empty():
    with _pytest.raises(ValueError):
        _resolve_variants(None, ("A",))
    with _pytest.raises(ValueError):
        _resolve_variants(None, ())


def test_resolve_w_floor_default_and_override():
    assert _resolve_w_floor("E-soft", None, None) is None          # → compute_weights 内部 0.2
    assert _resolve_w_floor("E-soft", None, {"E-soft": 0.1}) == 0.1
    assert _resolve_w_floor("E-soft", 0.5, None) == 0.5
    assert _resolve_w_floor("E-soft", 0.5, {"E-soft": 0.1}) == 0.1  # per-variant 优先
    assert _resolve_w_floor("E", 0.5, {"E-soft": 0.1}) == 0.5       # 无关变体的覆盖被忽略


def test_weight_distributions_list_produces_all_arms():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False,
        weight_distributions=("E", "E-soft"))
    assert set(res) == {"A_single", "A_iter", "E_single", "E_iter",
                        "E_soft_single", "E_soft_iter"}


def test_variant_alias_equivalence():
    res_alias = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False, variant="E-soft")
    res_list = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False, weight_distributions=("E-soft",))
    assert set(res_alias) == set(res_list)
    assert set(res_alias["E_soft_iter"]) == set(res_list["E_soft_iter"])


def test_A_in_weight_distributions_rejected():
    with _pytest.raises(ValueError):
        run_iterative_distillation(
            _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
            batch_size=8, K=2, seed=0, verbose=False, weight_distributions=("A",))


def test_w_floors_override_reaches_arm_weights(tiny_loaders):
    """w_floors 的 per-variant 地板经 _compute_arm_weights 到达权重。"""
    s = tiny_loaders
    student = RNN(s["L"], 16, s["num_bins"], 1).to(device)
    teacher = RNN(s["L"], 16, s["num_bins"], 1).to(device)
    w = _compute_arm_weights("E-soft", student, teacher, s["sf"], s["tf"],
                             s["indices"], s["L"], s["H"], w_floor=0.1)
    assert all(v >= 0.1 - 1e-6 for v in w.values())
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -x -k "resolve or weight_distributions or variant_alias or A_in_weight or w_floors_override"`
Expected: FAIL — `ImportError: cannot import name '_resolve_variants'`(helper 未定义)。

- [ ] **Step 3: 实现 helper 函数**

在 `fgl_common/training.py` 中 `_should_stop` 之前(约 665 行)插入:

```python
def _resolve_variants(variant, weight_distributions):
    """把 variant / weight_distributions 归一化为变体元组(校验+缺省)。

    - 两者同传 → ValueError(歧义)。
    - 都不传 → ('E',)(保持旧默认)。
    - 含 'A' 或空 → ValueError(A 是对照臂,固定包含,不许重复指定)。
    """
    if variant is not None and weight_distributions is not None:
        raise ValueError("variant 与 weight_distributions 不能同时指定;"
                         "请只传 weight_distributions")
    if weight_distributions is not None:
        vd = tuple(weight_distributions)
    elif variant is not None:
        vd = (variant,)
    else:
        vd = ("E",)
    if not vd:
        raise ValueError("weight_distributions 不能为空")
    if "A" in vd:
        raise ValueError("'A' 是对照臂(固定包含),不要传入 weight_distributions")
    return vd


def _resolve_w_floor(variant, w_floor, w_floors):
    """per-variant 软地板:w_floors[variant] 优先于全局 w_floor。"""
    if w_floors is not None and variant in w_floors:
        return float(w_floors[variant])
    return None if w_floor is None else float(w_floor)
```

- [ ] **Step 4: 重构 `run_iterative_distillation` 的签名与臂组装**

签名改为(其余参数不变):

```python
def run_iterative_distillation(data, L=20, H=15, alpha=0.5, temperature=4, num_bins=50,
                               epochs=30, round_epochs=15, batch_size=64, patience=5,
                               K=5, eps=0.01, N_stall=2, seed=42, variant=None,
                               weight_distributions=None, w_floor=None, w_floors=None,
                               val_size=0.2, test_size=0.2, lr=1e-4, verbose=True,
                               e_iter_snapshot_fn=None):
    """迭代(暖启动)自适应蒸馏,双权重分布变体 × 单/迭代臂(共享 round-0)。

    训一次 teacher / baseline / student_0(round-0,uniform KL),然后分支:
      A_single = student_0 本身(不迭代)。
      A_iter   = 暖启 ≤K 轮,权重恒均匀(归因对照)。
      对每个 v ∈ weight_distributions('E' 硬零地板 / 'E-soft' 稍软化 sigmoid):
        {v}_single = 暖启 1 轮,v 权重。
        {v}_iter   = 暖启 ≤K 轮,每轮重估 v 权重。
    变体名规范化:连字符 → 下划线('E-soft' → 'E_soft_*')。
    w_floor 仅作用于 E-soft;w_floors 可 per-variant 覆盖。
    停止规则在 val MSE,返回每臂 keep-best-by-val 的 student,报告其 test MSE。
    """
```

臂组装部分(替换现有 `arms = {...}` 块,约 894-899 行)改为:

```python
    variants = _resolve_variants(variant, weight_distributions)

    def _snapshots():
        # 多变体时快照只挂第一个变体的 iter 臂(与 e_iter_snapshot_fn 语义一致)
        for i, v in enumerate(variants):
            if i == 0:
                yield v, e_iter_snapshot_fn
            else:
                yield v, None

    def _arm(variant, max_rounds, snapshot_fn=None, w_floor_override=None):
        if max_rounds == 0:
            return {"rounds_used": 0, "total_epochs": 0,
                    "mse_curve_val": [evaluate(student_0, student_val, L)],
                    "mse_curve_test": [evaluate(student_0, student_test, L)],
                    "student": student_0}
        c = dict(common)                     # common 已含 w_floor=None 键
        if w_floor_override is not None:
            c["w_floor"] = w_floor_override
        return _iterate_student(student_0, teacher, variant, max_rounds,
                                snapshot_fn=snapshot_fn, **c)

    arms = {"A_single": _arm("A", 0), "A_iter": _arm("A", K)}
    for v, snap in _snapshots():
        vnorm = v.replace("-", "_")
        vw = _resolve_w_floor(v, w_floor, w_floors)
        arms[f"{vnorm}_single"] = _arm(v, 1, snapshot_fn=None, w_floor_override=vw)
        arms[f"{vnorm}_iter"] = _arm(v, K, snapshot_fn=snap, w_floor_override=vw)
```

注意 `_arm` 新签名 `_arm(variant, max_rounds, snapshot_fn=None, w_floor_override=None)`:内部 `c = dict(common)` 后覆盖 `w_floor` 再展开给 `_iterate_student`(common 含 `w_floor=None` 键、不含 `snapshot_fn` 键,无冲突)。single 与 iter 两臂都用 per-variant 地板(E-soft 的 sigmoid 地板对单轮臂同样生效);`vw` 为 None(E 变体)时 `c` 与 `common` 等价,`w_floor=None` 语义不变。

- [ ] **Step 5: 运行全部测试确认通过**

Run: `uv run pytest tests/fgl_common/test_iterative_distillation.py -v`
Expected: 全部 PASS(旧 17 用例回归 + 新 9 用例)。

- [ ] **Step 6: 提交**

```bash
git add fgl_common/training.py tests/fgl_common/test_iterative_distillation.py
git commit -m "feat(fgl_common): run_iterative_distillation 双权重分布(E/E-soft)+规范臂名

- weight_distributions 变体列表:每变体产出 {v}_single/{v}_iter,共享 A_single/A_iter 对照
- variant 保留为向后兼容别名;冲突/含 A/空列表 → ValueError
- w_floors per-variant 覆盖 E-soft 软地板;臂名连字符→下划线修掉旧 bug
- 新增 9 用例:解析 helper + 6 臂结构 + 别名等价 + w_floors 传递"
```

---

### Task 2: CSTR 统一入口(默认启用 + CLI 切换 + CSV 落盘)

**Files:**
- Modify: `cstr/run.py`(140-175 实验体、229 EXPERIMENTS、252-257 CLI)

**Interfaces:**
- Consumes: Task 1 的 `run_iterative_distillation(weight_distributions=..., w_floor=...)`。
- Produces: CLI `--distill_variants`(默认 `"E,E-soft"`)、`--w_floor`(默认 0.2);CSV `cstr/results/iterative_distill.csv`。

- [ ] **Step 1: 改 EXPERIMENTS 注册为默认启用**

```python
    "iterative_distill": dict(fn=run_iterative_distill_exp, enabled=True,
                              note="迭代自适应蒸馏(E 硬 / E-soft 稍软化,双权重分布);CSTR 已验证有效"),
```

- [ ] **Step 2: 加 CLI 参数**

在 `cstr/run.py` 的 `--K` 行后追加:

```python
    p.add_argument("--distill_variants", type=str, default="E,E-soft",
                   help="[iterative_distill] 权重分布变体,逗号分隔(如 E,E-soft)")
    p.add_argument("--w_floor", type=float, default=0.2,
                   help="[iterative_distill] E-soft 软地板(默认 0.2)")
```

- [ ] **Step 3: 重写实验体(动态臂 + CSV)**

替换 `run_iterative_distill_exp` 整个函数(约 140-175 行)为:

```python
def run_iterative_distill_exp(args):
    """迭代自适应蒸馏:变体列表 × 单/迭代臂,双权重分布(E 硬 / E-soft 稍软化)。"""
    import csv
    data = _load_data(args.dataset)
    n_seeds = args.seeds if args.seeds else 3
    variants = tuple(v.strip() for v in args.distill_variants.split(","))
    rows = []
    for s in range(n_seeds):
        arms = run_iterative_distillation(
            data, L=args.L, H=args.H, alpha=args.alpha, temperature=args.temperature,
            num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
            batch_size=args.batch_size, K=args.K, patience=args.patience, seed=s,
            weight_distributions=variants, w_floor=args.w_floor, verbose=False)
        for arm, r in arms.items():
            rows.append({"seed": s, "arm": arm, "student_mse": r["student_mse"],
                         "baseline_mse": r["baseline_mse"], "teacher_mse": r["teacher_mse"],
                         "fgl_delta": r["fgl_delta"], "init_delta": r["init_delta"],
                         "rounds_used": r["rounds_used"], "total_epochs": r["total_epochs"]})

    os.makedirs(os.path.join(_CSTR_DIR, "results"), exist_ok=True)
    out = os.path.join(_CSTR_DIR, "results", "iterative_distill.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  → {out} ({len(rows)} rows)")

    print(f"\n{'=' * 60}\nSUMMARY: iterative_distill (L={args.L} H={args.H})\n{'=' * 60}")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("student_mse", "fgl_delta", "init_delta"):
            agg[r["arm"]][k].append(r[k])

    def _sd(a):
        a = np.array(a); return a.std(ddof=1) if len(a) > 1 else 0.0

    for arm in sorted(agg):
        sm = np.array(agg[arm]["student_mse"])
        fd = np.array(agg[arm]["fgl_delta"])
        idt = np.array(agg[arm]["init_delta"])
        print(f"  {arm:14s}: student_mse={sm.mean():.1f}±{_sd(sm):.1f}  "
              f"Δbase={fd.mean():+.1f}%±{_sd(fd):.1f}  "
              f"Δinit={idt.mean():+.1f}%±{_sd(idt):.1f}  (n={len(sm)})")
```

先确认 `cstr/run.py` 无 `RESULTS_DIR` 常量——用 `_CSTR_DIR`(顶部已定义)。

- [ ] **Step 4: 验证**

Run:
```bash
uv run python cstr/run.py --list          # iterative_distill 应显示 ✓ ON
uv run python cstr/run.py -e iterative_distill --seeds 1 --epochs 5 --round_epochs 3 --K 2
```
Expected: 终端 6 行 summary(E_soft 臂在内);`cstr/results/iterative_distill.csv` 生成,含 6 臂 × 1 seed = 6 行。

- [ ] **Step 5: 提交**

```bash
git add cstr/run.py cstr/results/iterative_distill.csv
git commit -m "feat(cstr): iterative_distill 默认启用,CLI 切换 E/E-soft + CSV 落盘"
```

---

### Task 3: MG 统一入口(保持关闭,CLI + CSV 就绪)

**Files:**
- Modify: `mackey_glass/run.py`(345-390 实验体、390 EXPERIMENTS note、418-419 CLI)

**Interfaces:**
- Consumes: Task 1 的 `run_iterative_distillation`。
- Produces: CLI `--distill_variants`/`--w_floor`;CSV `mackey_glass/results/iterative_distill.csv`(手动 `-e` 时)。

- [ ] **Step 1: 加 CLI 参数**

在 `mackey_glass/run.py` 的 `--K` 行(419)后追加:

```python
    parser.add_argument("--distill_variants", type=str, default="E,E-soft",
                        help="[iterative_distill] 权重分布变体,逗号分隔(如 E,E-soft)")
    parser.add_argument("--w_floor", type=float, default=0.2,
                        help="[iterative_distill] E-soft 软地板(默认 0.2)")
```

- [ ] **Step 2: 更新 EXPERIMENTS note**(保持 `enabled=False`)

```python
    "iterative_distill": dict(fn=run_iterative_distill_exp, enabled=False,
                              note="迭代自适应蒸馏(E/E-soft 双权重分布);MG 负结果,默认关"),
```

- [ ] **Step 3: 重写实验体(动态臂 + CSV)**

替换 `run_iterative_distill_exp` 整个函数(约 345-390 行,当前硬编码 4 臂且不落盘)为:

```python
def run_iterative_distill_exp(args):
    """迭代自适应蒸馏:变体列表 × 单/迭代臂,双权重分布(E 硬 / E-soft 稍软化)。"""
    import csv
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}")
    n_seeds = args.seeds if args.seeds else 3
    variants = tuple(v.strip() for v in args.distill_variants.split(","))
    rows = []
    for s in range(n_seeds):
        arms = run_iterative_distillation(
            data, L=args.L, H=args.H, alpha=args.alpha, temperature=args.temperature,
            num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
            batch_size=args.batch_size, K=args.K, patience=args.patience, seed=s,
            weight_distributions=variants, w_floor=args.w_floor, verbose=False)
        for arm, r in arms.items():
            rows.append({"seed": s, "arm": arm, "student_mse": r["student_mse"],
                         "baseline_mse": r["baseline_mse"], "teacher_mse": r["teacher_mse"],
                         "fgl_delta": r["fgl_delta"], "init_delta": r["init_delta"],
                         "rounds_used": r["rounds_used"], "total_epochs": r["total_epochs"]})

    out = os.path.join(_RESULTS_DIR, "iterative_distill.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  → {out} ({len(rows)} rows)")

    print(f"\n{'=' * 60}\nSUMMARY: iterative_distill MG (L={args.L} H={args.H} τ={args.tau})\n{'=' * 60}")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("student_mse", "fgl_delta", "init_delta"):
            agg[r["arm"]][k].append(r[k])

    def _sd(a):
        a = np.array(a); return a.std(ddof=1) if len(a) > 1 else 0.0

    for arm in sorted(agg):
        sm = np.array(agg[arm]["student_mse"])
        fd = np.array(agg[arm]["fgl_delta"])
        idt = np.array(agg[arm]["init_delta"])
        print(f"  {arm:14s}: student_mse={sm.mean():.3f}±{_sd(sm):.3f}  "
              f"Δbase={fd.mean():+.1f}%±{_sd(fd):.1f}  "
              f"Δinit={idt.mean():+.1f}%±{_sd(idt):.1f}  (n={len(sm)})")
```

注意:`defaultdict` 与原 `generate_mg_data`/`_RESULTS_DIR`/`os`/`np` 均已在文件顶部 import(原函数已用 `defaultdict` 与 `np.array`);`csv` 局部导入即可。

- [ ] **Step 4: 验证**

Run:
```bash
uv run python mackey_glass/run.py --list   # iterative_distill 应显示 off,CLI 有 --distill_variants
uv run python mackey_glass/run.py -e iterative_distill --seeds 1 --epochs 3 --round_epochs 2 --K 1
```
Expected: summary 含 E_soft 臂;`mackey_glass/results/iterative_distill.csv` 生成。

- [ ] **Step 5: 提交**

```bash
git add mackey_glass/run.py mackey_glass/results/iterative_distill.csv
git commit -m "feat(mg): iterative_distill CLI 双权重分布 + CSV 落盘(保持默认关)"
```

---

### Task 4: 文档更新

**Files:**
- Modify: `fgl_common/__init__.py`(docstring 微调)
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新 `fgl_common/__init__.py` docstring**

training 行改为:

```
    training.py       —— device / EarlyStopper / evaluate* / run_fgl_experiment /
                         run_iterative_distillation(双权重分布 E/E-soft)/ run_adaptive_weight /
                         run_adaptive_inference / run_seq2seq / run_baseline_converged
```

- [ ] **Step 2: 更新 CLAUDE.md**

- `fgl_common` 段 `training.py` 行的 `run_iterative_distillation` 描述后追加:`run_iterative_distillation`(双权重分布:E 硬零地板 / E-soft 稍软化,变体列表×单/迭代臂)。
- `cstr/run.py` 行:`(enabled: baseline, lh_sweep)` → `(enabled: baseline, lh_sweep, iterative_distill)`。
- `mackey_glass/run.py` 行的实验列表注明 `iterative_distill` 保持 off。

- [ ] **Step 3: 提交**

```bash
git add fgl_common/__init__.py CLAUDE.md
git commit -m "docs: 双权重分布迭代蒸馏文档(CLAUDE.md + __init__)"
```

---

### Task 5: 全量验收

- [ ] **Step 1: 全测试**

Run: `uv run pytest tests/ -v`
Expected: 全绿(含既有 17 + 新 9 用例)。

- [ ] **Step 2: 端到端冒烟(CSTR 默认路径)**

Run: `uv run python cstr/run.py --list`
Expected: `iterative_distill [✓ ON]`。

- [ ] **Step 3: 端到端冒烟(MG 关闭)**

Run: `uv run python mackey_glass/run.py --list`
Expected: `iterative_distill [  off]`。

- [ ] **Step 4: 提交(若有遗留)**

```bash
git add -A && git commit -m "chore: 验收修复" || echo "无遗留改动"
```
