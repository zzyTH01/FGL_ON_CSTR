# 迭代自适应蒸馏 — MG 数据集实现计划(Phase 0 试点)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已实现的迭代自适应蒸馏原样移植到 Mackey-Glass τ=13,跑 Phase 0 试点(3 跨阈值点 × 3 seeds),报告放行门判定。

**Architecture:** 方法本体 `run_iterative_distillation` 零改动(已由 CSTR 的 15 个 pytest 覆盖)。新增 `mackey_glass/sweep_iterative.py`(镜像 `cstr/sweep_iterative.py`,数据源换成 jitcdde 生成的 MG 序列)+ `mackey_glass/run.py` 实验开关。试点用"全曲线 + per-seed 中位数"口径防 val≈test 过拟合。

**Tech Stack:** Python 3.11, PyTorch 2.1.1, jitcdde(MG 数据生成), uv。

## Global Constraints

- τ=13;cells = (4,7)(4,10)(13,7);α=0.5,T=4,bins=50;epochs=50,round_epochs=20,K=5,n_seeds=3,batch=128。
- **`fgl_common` 零改动**(方法已由 `tests/fgl_common/test_iterative_distillation.py` 的 15 个测试覆盖)。
- 过拟合防护:报告**全曲线 + per-seed 中位数**,不报 keep-best 单点(CSTR 教训)。
- 运行强制 CPU:`FGL_DEVICE=cpu uv run python ...`。
- 复用 `cstr/sweep_iterative.py` 的结构(sweep 循环、CSV 写入、`_plot_curves` 原样照搬)。
- commit 中文 message + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` 尾注。
- 测试约定:本计划为应用/接线层(脚本+开关),沿用仓库惯例用 **CLI 冒烟**验证,不加新 pytest(方法层已覆盖)。

## File Structure

| 文件 | 责任 | 改动 |
|------|------|------|
| `mackey_glass/sweep_iterative.py` | MG 4 臂 sweep(cells × seeds)+ 逐轮曲线 CSV + 图 | 新建(镜像 CSTR) |
| `mackey_glass/run.py` | `iterative_distill` 实验开关 + CLI(--round_epochs/--K) | 改 |
| `conclusion/iterative_pilot_phase0_mg.md` | MG Phase 0 结果 + 放行门判定 | 新建 |

---

## Task 1: `mackey_glass/sweep_iterative.py` + CLI 冒烟

**Files:**
- Create: `mackey_glass/sweep_iterative.py`

**Interfaces:**
- Consumes: `fgl_common.run_iterative_distillation(data, L, H, num_bins, epochs, round_epochs, K, seed, alpha, temperature, verbose) -> dict[arm, dict]`;`mackey_glass/utils/utils.py::MackeyGlass`(jitcdde 数据生成)。
- Produces: `mackey_glass/sweep_iterative.py`(`main()` CLI:`--cells/--grid/--seeds/--epochs/--round_epochs/--K/--tau/--n_points`)→ `mackey_glass/results/iterative_mg_sweep.csv` + `iterative_mg_curves.png`。

- [ ] **Step 1: 写 sweep 脚本**

`mackey_glass/sweep_iterative.py`:
```python
#!/usr/bin/env python
"""MG τ=13:迭代蒸馏 4 臂对比 + 逐轮 MSE 曲线(镜像 cstr/sweep_iterative.py,换 MG 数据源)。

用法::
    # Phase 0 试点:3 跨阈值点
    FGL_DEVICE=cpu uv run python mackey_glass/sweep_iterative.py --cells "4,7;4,10;13,7" --seeds 3 --epochs 50 --round_epochs 20 --K 5
输出 mackey_glass/results/iterative_mg_sweep.csv + iterative_mg_curves.png。
"""
import argparse
import csv
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MG_DIR)  # utils.utils 在此

import numpy as np
import torch
from fgl_common import run_iterative_distillation
from utils.utils import MackeyGlass

_RESULTS_DIR = os.path.join(_MG_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
ARMS = ("A_single", "E_single", "A_iter", "E_iter")


def generate_mg_data(tau=13.0, n_points=10000, seed=42):
    """生成 MG 序列,返回 (N,2) 张量(两列均为序列值,自回归)。与 run.py 同口径。"""
    mg = MackeyGlass(tau=tau, constant_past=0.9, nmg=10, beta=0.2, gamma=0.1,
                     dt=1.0, splits=(float(n_points), 0.0), seed_id=seed)
    vals = [mg[idx][1].squeeze().item() for idx in range(len(mg))]
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    return torch.cat((col, col.clone()), dim=1), mg.lyap_exp


def _parse_cells(args):
    if args.grid:
        Ls = [4, 7, 10, 13, 16]
        Hs = [4, 7, 10, 13, 16]
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
    ap = argparse.ArgumentParser(description="MG iterative distillation 4-arm sweep")
    ap.add_argument("--cells", default="4,7;4,10;13,7", help="semicolon-separated L,H pairs")
    ap.add_argument("--grid", action="store_true", help="5x5 grid (overrides --cells)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--round_epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--tau", type=float, default=13.0)
    ap.add_argument("--n_points", type=int, default=10000)
    args = ap.parse_args()

    cells = _parse_cells(args)
    seeds = list(range(args.seeds))
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}, {args.n_points} pts", flush=True)
    csv_path = os.path.join(_RESULTS_DIR, "iterative_mg_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["L", "H", "seed", "arm", "baseline_mse", "student_mse", "fgl_delta", "init_delta",
             "rounds_used", "mse_curve_val", "mse_curve_test"])

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
                batch_size=args.batch_size, K=args.K, seed=s, verbose=False)
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
        e_iter = np.mean(per_arm["E_iter"]); e_single = np.mean(per_arm["E_single"])
        a_iter = np.mean(per_arm["A_iter"])
        rel = (a_iter - e_iter) / a_iter * 100 if a_iter > 0 else float("nan")
        print(f"[{done}/{total}] L={L:<3} H={H:<3}: E_iter={e_iter:6.3f}  "
              f"E_single={e_single:6.3f}  A_iter={a_iter:6.3f}  "
              f"(E_iter vs A_iter {rel:+.1f}%)", flush=True)

    _report(cell_results)


def _report(cell_results):
    print(f"\n{'=' * 70}\nE-iter 相对对照的 student MSE 下降 (%)  [+ = E-iter 更好]\n{'=' * 70}")
    print(f"{'L,H':>10} | {'vs A-single':>12} | {'vs E-single':>12} | {'vs A-iter':>10}")
    print("-" * 60)
    for (L, H), (per_arm, _) in cell_results.items():
        a_s = np.mean(per_arm["A_single"]); e_s = np.mean(per_arm["E_single"])
        a_i = np.mean(per_arm["A_iter"]); e_i = np.mean(per_arm["E_iter"])

        def rel(base):
            return (base - e_i) / base * 100 if base > 0 else float("nan")
        print(f"({L:>2},{H:<3})    | {rel(a_s):>+11.1f}% | {rel(e_s):>+11.1f}% | {rel(a_i):>+9.1f}%")
    _plot_curves(cell_results)
    print(f"\nCSV: {os.path.join(_RESULTS_DIR, 'iterative_mg_sweep.csv')}")


def _plot_curves(cell_results):
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
            ax.plot(range(max_len), np.nanmean(arr, axis=0), marker="o", label=arm)
        ax.set_title(f"L={L}, H={H}")
        ax.set_xlabel("round (0 = round-0 student)")
        ax.set_ylabel("val MSE")
        ax.legend(fontsize=8)
    fig.suptitle("MG τ=13: Per-round val MSE by arm (E-iter shape answers H1)")
    fig.tight_layout()
    png = os.path.join(_RESULTS_DIR, "iterative_mg_curves.png")
    fig.savefig(png, dpi=120)
    print(f"逐轮曲线: {png}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: CLI 冒烟(1 cell,1 seed,极短参数)**

Run:
```bash
FGL_DEVICE=cpu uv run python mackey_glass/sweep_iterative.py --cells "4,7" --seeds 1 --epochs 3 --round_epochs 2 --K 2 --n_points 800
```
Expected: 打印 MG τ/Lyapunov 行 + `[1/1] L=4 H=7: E_iter=... E_single=... A_iter=...` + 生成 `mackey_glass/results/iterative_mg_sweep.csv` 与 `iterative_mg_curves.png`,无报错。(`--n_points 800` 缩短 jitcdde 生成时间仅供冒烟。)

- [ ] **Step 3: 提交**

```bash
git add mackey_glass/sweep_iterative.py
git commit -m "feat: mackey_glass/sweep_iterative.py(MG τ=13 迭代蒸馏 4 臂 sweep)

镜像 cstr/sweep_iterative.py,数据源换成 generate_mg_data(jitcdde)。
默认 cells 跨阈值三点 (4,7)(4,10)(13,7)。CLI 冒烟通过。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: 接入 `mackey_glass/run.py`

**Files:**
- Modify: `mackey_glass/run.py`(import + 包装函数 + CLI 参数 + EXPERIMENTS 条目)

**Interfaces:**
- Consumes: `fgl_common.run_iterative_distillation`。

- [ ] **Step 1: 加 import**

`mackey_glass/run.py` 顶部 `from fgl_common import RNN, run_fgl_experiment, run_lh_sweep` 改为:
```python
from fgl_common import RNN, run_fgl_experiment, run_lh_sweep, run_iterative_distillation  # noqa: E402
```

- [ ] **Step 2: 加包装函数**

在 `run_geometry` 之后、`# ==================== Experiment switches ====================` 之前插入:
```python
def run_iterative_distill_exp(args):
    """迭代自适应蒸馏 4 臂对比(A-single / E-single / A-iter / E-iter)在 MG τ=13 上。"""
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}")
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

    print(f"\n{'=' * 60}\nSUMMARY: iterative_distill MG (L={args.L} H={args.H} τ={args.tau})\n{'=' * 60}")
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
        print(f"  {arm:9s}: student_mse={sm.mean():.3f}±{_sd(sm):.3f}  "
              f"Δbase={fd.mean():+.1f}%±{_sd(fd):.1f}  "
              f"Δinit={idt.mean():+.1f}%±{_sd(idt):.1f}  (n={len(sm)})")
```

- [ ] **Step 3: 注册开关 + CLI 参数**

`EXPERIMENTS` 字典内(`geometry` 之后)追加:
```python
    "iterative_distill": dict(fn=run_iterative_distill_exp, enabled=False,
                              note="迭代自适应蒸馏 4 臂(MG τ=13);Phase 0 先跑典型点"),
```

在 `main()` 的 argparse 块(`--H_values` 之后)追加:
```python
    parser.add_argument("--round_epochs", type=int, default=20, help="[iterative_distill] 每轮 epoch")
    parser.add_argument("--K", type=int, default=5, help="[iterative_distill] 最大迭代轮数")
```

- [ ] **Step 4: 冒烟验证(1 seed,极短)**

Run:
```bash
FGL_DEVICE=cpu uv run python mackey_glass/run.py -e iterative_distill --L 4 --H 7 --epochs 3 --round_epochs 2 --K 2 --seeds 1 --n_points 800
```
Expected: 打印 MG τ/Lyapunov + 4 臂 SUMMARY(A_single/E_single/A_iter/E_iter 各一行 student_mse),无报错。

- [ ] **Step 5: 提交**

```bash
git add mackey_glass/run.py
git commit -m "feat: mackey_glass/run.py 接入 iterative_distill 开关 + CLI

4 臂 n-seed 汇总打印。新增 --round_epochs/--K 参数。默认 enabled=False。
冒烟:4 臂端到端跑通(MG τ=13)。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Phase 0 试点 + 结论 + 放行门判定

**Files:**
- Create: `conclusion/iterative_pilot_phase0_mg.md`
- Produce: `mackey_glass/results/iterative_mg_sweep.csv` + `iterative_mg_curves.png`(由试点运行生成)

- [ ] **Step 1: 跑 Phase 0 试点(真实参数)**

Run:
```bash
FGL_DEVICE=cpu uv run python mackey_glass/sweep_iterative.py --cells "4,7;4,10;13,7" --seeds 3 --epochs 50 --round_epochs 20 --K 5
```
记录输出(M1/M2/M3 的 E_iter/E_single/A_iter/A_single MSE + E_iter vs A_iter 相对降幅 + 逐轮曲线形状)。预计运行数分钟~十几分钟。

- [ ] **Step 2: 全曲线 + per-seed 中位数分析(防过拟合口径)**

用以下脚本分析 CSV(取每 seed 最小 test MSE,再取跨 seed 中位数;并打印逐轮曲线均值,标注离群 seed):
```bash
FGL_DEVICE=cpu uv run python -c "
import csv, numpy as np
from collections import defaultdict
rows=list(csv.DictReader(open('mackey_glass/results/iterative_mg_sweep.csv')))
mins=defaultdict(lambda: defaultdict(list)); curves=defaultdict(list)
for r in rows:
    key=(r['L'],r['H'],r['arm'])
    c=[float(x) for x in r['mse_curve_test'].split(';')]
    mins[key]['min'].append(min(c))
    curves[(r['L'],r['H'],r['arm'])].append(c)
print('=== per-seed min test MSE (median over seeds) ===')
for (L,H,arm) in sorted(mins):
    arr=np.array(mins[(L,H,arm)]['min'])
    print(f'  L={L} H={H} {arm:9s}: per-seed={arr.round(3).tolist()}  median={np.median(arr):.3f}')
print()
print('=== E_iter 逐轮 test MSE 曲线(mean over seeds)===')
for (L,H,arm),cs in sorted(curves.items()):
    if arm!='E_iter': continue
    ml=max(len(c) for c in cs); arr=np.full((len(cs),ml),np.nan)
    for i,c in enumerate(cs): arr[i,:len(c)]=c
    print(f'  L={L} H={H}: ' + ' -> '.join(f'{v:.3f}' for v in np.nanmean(arr,axis=0)))
"
```
判读:有无 1.4 式离群 seed 主导均值?若有,按中位数口径报。

- [ ] **Step 3: 写结论 `conclusion/iterative_pilot_phase0_mg.md`**

按 spec §4.2 放行门核对并记录:
- **M1(L4H7,<τ)/ M2(L4H10,=τ)**:E-iter ≤ E-single 且 E-iter ≤ A-iter?(净迭代效应)
- **M3(L13H7,>τ,FGL 强负)**:E-iter 是否不显著恶化(keep-best/停止护住,不放大 −54% 伤害)?
- 逐轮曲线:E-iter 在 M1/M2 单调降?M3 是否暴增?
- 过拟合:per-seed 中位数下有无离群主导?
- **放行门判定**:放行 MG Phase 1 / 阈值主导 / 负结果,三选一并说明。

文档结构:设置 → 绝对 MSE 表(per-seed 中位数)→ 归因对比(E_iter vs A_iter / vs E_single)→ 逐轮曲线形状 → H1/阈值判定 → 放行门 → 局限。

- [ ] **Step 4: 提交**

```bash
git add conclusion/iterative_pilot_phase0_mg.md mackey_glass/results/iterative_mg_sweep.csv mackey_glass/results/iterative_mg_curves.png
git commit -m "feat: MG τ=13 迭代蒸馏 Phase 0 试点 + 放行门判定

3 跨阈值点 × 3 seeds,K=5,epochs=50/round_epochs=20。
per-seed 中位数口径(防 val≈test 过拟合)。放行门结论见文档。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review(写计划后自查)

**1. Spec 覆盖**
- §1 方法零改动 → Global Constraints + 不改 fgl_common ✓
- §2.1 数据(generate_mg_data)→ Task 1 内联实现 ✓
- §2.2 三典型点跨阈值 → Task 1 默认 cells + Task 3 试点 ✓
- §2.3 全曲线 + per-seed 中位数 → Task 3 Step 2 分析脚本 ✓
- §2.4 超参(epochs=50/round_epochs=20/K=5/seeds=3/batch=128)→ Task 3 Step 1 + Task 1 默认 ✓
- §3 实现(sweep + run.py,fgl_common 不动)→ Task 1 + Task 2 ✓
- §4 放行门(M1/M2 净效应、M3 不恶化、无离群主导)→ Task 3 Step 3 ✓

**2. 占位扫描**:无 TBD/TODO;每个 Step 含完整代码或具体命令 + 期望输出。✓

**3. 类型/命名一致性**:`run_iterative_distillation` 签名(Global Constraints)与 Task 1/Task 2 调用一致;四臂名 `A_single/E_single/A_iter/E_iter` 跨任务一致;`generate_mg_data` 在 Task 1(sweep)与 Task 2(run.py,已存在)同名同参。✓

**注**:Task 2 复用 `mackey_glass/run.py` 中**已存在**的 `generate_mg_data`、`defaultdict`、`np`(已 import),无需新增 import。
