#!/usr/bin/env python
"""收敛速度分析:E_iter(自适应权重) vs A_iter(均匀权重)连续蒸馏,谁更快到地板。

与 ``analyze_iterative.py`` 互补:那个看**地板高度**(per-seed-min test MSE),
本脚本把**到达地板的速度**立为主指标——回答"自适应权重是否让 student 更快收敛"。

臂定义(来自 ``fgl_common.run_iterative_distillation``,共享 round-0、设置全同、
只差权重方案,故 A vs E 是干净对照):
  A_iter = 暖启动迭代蒸馏 + 权重恒均匀   = "只连续训练(不带自适应权重)"
  E_iter = 暖启动迭代蒸馏 + 每轮重估 E 权重 = "连续训练 + 自适应权重"
  (E_soft_iter = 软地板变体,MG 用)

速度主指标(逐轮曲线 mse_curve_test,index 0=round-0,t=第 t 轮后):
  rounds_to_floor  到达"共享地板 10% 内"的轮数(共享地板=min over 两臂曲线;越少越快)
  round1_gain      round-0→round-1 相对进步 (c0-c1)/c0(越大越快)
  auc_norm         归一化曲线下面积 mean(c/c0)(越低越快)
  floor            各臂 min(test MSE)——仅作口径参照,不是速度
统计: 配对均值差 E−A + 95% CI(t 分布) + Wilcoxon + 胜/平/负计数。
MG 机制: 退化(曲线上行=重估权重反噬)频率、E/A round-1 进步比。

数据源(默认全跑,跨域):
  标准 CSTR(有曲线):  cstr/results/iterative_phase1_anchor.csv (n=10,K=6)
                      cstr/results/iterative_phase1_grid.csv   (5x5x2,K=6)
                      cstr/results/iterative_sweep.csv          (3pt x3,K=3)
                      cstr/results/iterative_saturation_sweep.csv (2pt x3,K=15)
  混沌 CSTR(有曲线):  cstr/results/chaotic_floor_verify.csv     (高K饱和核实,可选)
  混沌 CSTR(仅汇总):  cstr/results/iterative_delayed_summary.csv (rounds_used)
                      cstr/results/floor_sweep.csv               (仅最终地板)
  MG(有曲线):         mackey_glass/results/iterative_mg_sweep.csv + E-soft wf*

用法::
    uv run python cstr/archive/analyze_convergence_speed.py            # 全跨域默认
    uv run python cstr/archive/analyze_convergence_speed.py a.csv b.csv  # 指定文件
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]  # 仓库根

# (标签, 相对路径, 配置分组列)
DEFAULT_CURVE_FILES = [
    ("标准CSTR anchor L20H15 (n=10,K=6)", "cstr/results/iterative_phase1_anchor.csv", ["L", "H", "seed"]),
    ("标准CSTR grid (5x5x2,K=6)",          "cstr/results/iterative_phase1_grid.csv",   ["L", "H", "seed"]),
    ("标准CSTR sweep (3pt x3,K=3)",        "cstr/results/iterative_sweep.csv",         ["L", "H", "seed"]),
    ("标准CSTR saturation (2pt x3,K=15)",  "cstr/results/iterative_saturation_sweep.csv", ["L", "H", "seed"]),
    ("混沌CSTR 高K核实 (有曲线)",          "cstr/results/chaotic_floor_verify.csv",   ["L", "H", "seed"]),
    ("MG sweep (3pt x3,K=5)",              "mackey_glass/results/iterative_mg_sweep.csv", ["L", "H", "seed"]),
    ("MG E-soft wf=0.05",                  "mackey_glass/results/iterative_mg_sweep_E-soft_wf0.05.csv", ["L", "H", "seed"]),
    ("MG E-soft wf=0.10",                  "mackey_glass/results/iterative_mg_sweep_E-soft_wf0.1.csv",  ["L", "H", "seed"]),
    ("MG E-soft wf=0.15",                  "mackey_glass/results/iterative_mg_sweep_E-soft_wf0.15.csv", ["L", "H", "seed"]),
    ("MG E-soft wf=0.20",                  "mackey_glass/results/iterative_mg_sweep_E-soft.csv",        ["L", "H", "seed"]),
]

ARMS = ("A_iter", "E_iter", "E_soft_iter")


# ---------------- loaders ----------------
def _curve(s):
    return [float(x) for x in s.split(";")] if s else []


def load_curve_csv(path, config_cols):
    """读迭代蒸馏 CSV,只留 A_iter/E_iter(含 E_soft_iter)。返回 list[dict]。"""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["arm"] not in ARMS:
                continue
            rows.append({
                "arm": r["arm"],
                "curve_test": _curve(r.get("mse_curve_test", "")),
                "rounds_used": int(r.get("rounds_used") or 0),
                "student_mse": float(r.get("student_mse", "nan") or "nan"),
                "config": tuple(int(r[c]) for c in config_cols),
            })
    return rows


# ---------------- speed metrics ----------------
def rounds_to_thresh(curve, thresh):
    """首个 <= thresh 的轮次(含 round 0);达不到返回 None。"""
    for t, v in enumerate(curve):
        if v <= thresh:
            return t
    return None


def round1_gain(curve):
    """round-0→round-1 相对进步(越大越好)。"""
    if len(curve) < 2 or curve[0] == 0:
        return None
    return (curve[0] - curve[1]) / curve[0]


def auc_norm(curve):
    if not curve or curve[0] == 0:
        return None
    return sum(v / curve[0] for v in curve) / len(curve)


def has_degradation(curve):
    """曲线是否含任何上行(重估权重反噬)。"""
    return any(curve[i + 1] > curve[i] for i in range(len(curve) - 1))


# ---------------- stats ----------------
def paired_ci(diffs):
    """配对差 -> (mean, lo, hi, n) 95% CI。"""
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = sum(diffs) / n
    if n == 1:
        return m, float("nan"), float("nan"), 1
    sd = stats.tstd(diffs)
    half = stats.t.ppf(0.975, n - 1) * sd / (n ** 0.5)
    return m, m - half, m + half, n


def wilcoxon_p(diffs):
    nz = [d for d in diffs if abs(d) > 1e-12]
    if len(nz) < 2:
        return float("nan")
    try:
        return float(stats.wilcoxon(nz).pvalue)
    except Exception:
        return float("nan")


def fmt_p(p):
    if p != p:
        return "p=n/a"
    return f"p={p:.4f}" if p >= 1e-4 else f"p={p:.2e}"


def median(xs):
    return float(np.median(xs)) if xs else float("nan")


def mean(xs):
    return stats.tmean(xs) if xs else float("nan")


# ---------------- per-dataset paired analysis ----------------
def analyze_paired(name, rows):
    by_cfg = defaultdict(dict)
    for r in rows:
        if r["arm"] in ("A_iter", "E_iter"):
            by_cfg[r["config"]][r["arm"]] = r
    pairs = {k: v for k, v in by_cfg.items() if "A_iter" in v and "E_iter" in v}
    if not pairs:
        return
    n = len(pairs)
    d_rt, d_r1, d_auc, d_fl = [], [], [], []
    rt_a, rt_e, r1_a, r1_e = [], [], [], []
    deg_a = deg_e = reach_a = reach_e = win_sa = win_se = win_fa = win_fe = 0
    for cfg, d in sorted(pairs.items()):
        ac, ec = d["A_iter"]["curve_test"], d["E_iter"]["curve_test"]
        floor = min(min(ac), min(ec))
        ta, te = rounds_to_thresh(ac, 1.10 * floor), rounds_to_thresh(ec, 1.10 * floor)
        if ta is not None:
            rt_a.append(ta); reach_a += 1
        if te is not None:
            rt_e.append(te); reach_e += 1
        if ta is not None and te is not None:
            if te < ta: win_se += 1
            elif ta < te: win_sa += 1
            d_rt.append(te - ta)
        ga, ge = round1_gain(ac), round1_gain(ec)
        if ga is not None and ge is not None:
            d_r1.append(ge - ga); r1_a.append(ga); r1_e.append(ge)
        aa, ae = auc_norm(ac), auc_norm(ec)
        if aa is not None and ae is not None:
            d_auc.append(ae - aa)
        fa, fe = min(ac), min(ec)
        d_fl.append(fe - fa)
        if fe < fa - 1e-6: win_fe += 1
        elif fa < fe - 1e-6: win_fa += 1
        if has_degradation(ac): deg_a += 1
        if has_degradation(ec): deg_e += 1

    print(f"\n{'=' * 84}\n[{name}]  N={n} 配对")
    print(f"  配对差 E−A(负=E 更快/更低;round1_gain 正=E 进步更大):")
    for label, diffs, sign_hint in [
        ("rounds_to_floor", d_rt, "负=E更快"), ("round1_gain", d_r1, "正=E进步大"),
        ("auc_norm", d_auc, "负=E更快"), ("floor", d_fl, "负=E更低")]:
        m, lo, hi, k = paired_ci(diffs)
        tag = sign_hint if ((sign_hint.startswith("负") and m < 0) or (sign_hint.startswith("正") and m > 0)) else ""
        print(f"    {label:<16} Δ={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  {fmt_p(wilcoxon_p(diffs)):<12} {tag}")
    print(f"  绝对量: rounds_to_floor 中位 A={median(rt_a):.1f} E={median(rt_e):.1f} | "
          f"round1_gain 均值 A={mean(r1_a)*100:+.1f}% E={mean(r1_e)*100:+.1f}%")
    print(f"  计数:   速度胜 A={win_sa}/E={win_se} | 地板更低 A={win_fa}/E={win_fe} | "
          f"到达共享地板 A={reach_a}/E={reach_e}")
    print(f"  退化(曲线上行): A_iter={deg_a}/{n}  E_iter={deg_e}/{n}")


# ---------------- per-cell breakdown (for divergent cells) ----------------
def analyze_per_cell(name, rows):
    """逐 cell 打印 A vs E 地板/速度——cell 行为分化时(divergent)必看,pooling 会抵消。
    典型场景:混沌 CSTR 高K核实(L85H15 E 赢、L100H30 E 崩,pooling 会 wash out)。"""
    by_cfg = defaultdict(dict)
    for r in rows:
        if r["arm"] in ("A_iter", "E_iter"):
            by_cfg[r["config"]][r["arm"]] = r
    # 按 (L,H) 聚合 seed
    by_cell = defaultdict(lambda: {"A": [], "E": [], "Ar": [], "Er": []})
    for cfg, d in by_cfg.items():
        if "A_iter" not in d or "E_iter" not in d:
            continue
        L, H = cfg[0], cfg[1]
        ac, ec = d["A_iter"]["curve_test"], d["E_iter"]["curve_test"]
        by_cell[(L, H)]["A"].append(min(ac))
        by_cell[(L, H)]["E"].append(min(ec))
        by_cell[(L, H)]["Ar"].append(d["A_iter"]["rounds_used"])
        by_cell[(L, H)]["Er"].append(d["E_iter"]["rounds_used"])
    if len(by_cell) < 2:  # 单 cell 没必要逐 cell 展开
        return
    print(f"\n  ·[{name}] 逐 cell 地板(K 饱和,E−A 负=E 更低):")
    print(f"    {'cell':<10}{'A(mean±sd)':<16}{'E(mean±sd)':<16}{'E−A [95%CI]':<26}"
          f"{'Arnd':<6}{'Ernd':<6}{'Ewin':<7}{'Ecoll':<6}")
    for (L, H), c in sorted(by_cell.items()):
        A, E = np.array(c["A"]), np.array(c["E"])
        diff = E - A
        m, lo, hi, k = paired_ci(list(diff))
        ci = f"[{lo:+.1f},{hi:+.1f}]" if lo == lo else "[n/a]"
        Am = f"{A.mean():.1f}±{A.std(ddof=1):.1f}" if len(A) > 1 else f"{A.mean():.1f}"
        Em = f"{E.mean():.1f}±{E.std(ddof=1):.1f}" if len(E) > 1 else f"{E.mean():.1f}"
        ewin = int((E < A - 1e-6).sum())
        collapse = int((E > A * 1.3).sum())  # E 早停在坏盆底
        mark = "✓E赢" if (m < 0 and lo < 0) else ("✗A赢" if (m > 0 and hi > 0) else "≈")
        print(f"    L{L}H{H:<4}{Am:<16}{Em:<16}{m:+5.1f} {ci:<20}"
              f"{median(c['Ar']):<6.0f}{median(c['Er']):<6.0f}{ewin}/{k:<5}{collapse}/{k}  {mark}")


# ---------------- chaotic CSTR summary-only ----------------
def analyze_chaotic_summary():
    print(f"\n{'#' * 84}\n# 混沌 CSTR(延迟反馈 τ 扫描)—— 汇总数据(无逐轮曲线)\n{'#' * 84}")
    path = ROOT / "cstr/results/iterative_delayed_summary.csv"
    if not path.exists():
        print("  [skip] iterative_delayed_summary.csv 不存在"); return
    print(f"\n[rounds_used + 最终地板]  {path.name}")
    print(f"  {'dataset':<14}{'period':>8}{'A_floor':>10}{'E_floor':>10}{'A_rnd':>8}{'E_rnd':>8}  判读")
    by_ds = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["arm"] in ("A_iter", "E_iter"):
                by_ds[r["dataset"]][r["arm"]] = r
    for ds, d in by_ds.items():
        a, e = d["A_iter"], d["E_iter"]
        af, ef = float(a["student_mse"]), float(e["student_mse"])
        ar, er = float(a["rounds_used"]), float(e["rounds_used"])
        if er < ar - 0.3 and ef <= af * 1.02: v = "E更快+同/更低地板"
        elif er < ar - 0.3 and ef > af * 1.02: v = "E更少轮但地板高(早停?)"
        elif abs(er - ar) <= 0.3: v = "速度持平"
        else: v = "A更快"
        print(f"  {ds:<14}{float(a['periodicity']):>8.2f}{af:>10.1f}{ef:>10.1f}{ar:>8.2f}{er:>8.2f}  {v}")

    # floor_sweep: τ100 多 cell 最终地板(无速度)
    fpath = ROOT / "cstr/results/floor_sweep.csv"
    if not fpath.exists():
        return
    print(f"\n[τ100 最终地板 — floor_sweep, 无速度]  E_iter vs A_iter")
    by_cell = defaultdict(lambda: {"a": [], "e": []})
    with open(fpath) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "tau100":
                by_cell[(r["L"], r["H"])]["a"].append(float(r["A_iter_mse"]))
                by_cell[(r["L"], r["H"])]["e"].append(float(r["E_iter_mse"]))
    diffs, ew, aw = [], 0, 0
    print(f"  {'cell(L,H)':<12}{'A_iter':>10}{'E_iter':>10}{'E/A':>8}  判读")
    for cell, v in sorted(by_cell.items()):
        am, em = mean(v["a"]), mean(v["e"])
        ratio = em / am if am else float("nan")
        judge = "E更低" if em < am - 1e-6 else ("A更低" if am < em - 1e-6 else "≈")
        if em < am - 1e-6: ew += 1
        elif am < em - 1e-6: aw += 1
        diffs.extend(ei - ai for ei, ai in zip(v["e"], v["a"]))
        print(f"  {str(cell):<12}{am:>10.1f}{em:>10.1f}{ratio:>8.2f}  {judge}")
    m, lo, hi, _ = paired_ci(diffs)
    print(f"  汇总: E−A 地板差 Δ={m:+.2f} 95%CI[{lo:+.2f},{hi:+.2f}] {fmt_p(wilcoxon_p(diffs))} | "
          f"E更低={ew} A更低={aw} (共 {len(by_cell)} cell)")


# ---------------- MG mechanism: CSTR vs MG contrast ----------------
def mg_mechanism():
    print(f"\n{'#' * 84}\n# MG 机制:收敛速度是否拖垮 E?(CSTR anchor vs MG sweep 对照)\n{'#' * 84}")
    cstr = load_curve_csv(ROOT / "cstr/results/iterative_phase1_anchor.csv", ["L", "H", "seed"])
    mg = load_curve_csv(ROOT / "mackey_glass/results/iterative_mg_sweep.csv", ["L", "H", "seed"])
    for label, rows in [("标准CSTR anchor", cstr), ("MG sweep", mg)]:
        by_cfg = defaultdict(dict)
        for r in rows:
            if r["arm"] in ("A_iter", "E_iter"):
                by_cfg[r["config"]][r["arm"]] = r
        ga, ge, deg_e, n = [], [], 0, 0
        for cfg, d in by_cfg.items():
            if "A_iter" not in d or "E_iter" not in d:
                continue
            n += 1
            ra, re = round1_gain(d["A_iter"]["curve_test"]), round1_gain(d["E_iter"]["curve_test"])
            if ra is not None: ga.append(ra)
            if re is not None: ge.append(re)
            if has_degradation(d["E_iter"]["curve_test"]): deg_e += 1
        ratios = [re / ra for ra, re in zip(ga, ge) if ra and ra > 1e-9]
        print(f"  [{label}] N={n} | round-1 进步 A={mean(ga)*100:+.1f}% E={mean(ge)*100:+.1f}% "
              f"| E/A 进步比={mean(ratios):.2f} | E 曲线退化率={deg_e}/{n}")


def main():
    files = DEFAULT_CURVE_FILES
    if len(sys.argv) > 1:
        files = [(Path(a).name, a, ["L", "H", "seed"]) for a in sys.argv[1:]]
    for name, rel, cols in files:
        path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        if not path.exists():
            print(f"[skip] {name}: {path} 不存在")
            continue
        rows = load_curve_csv(path, cols)
        analyze_paired(name, rows)
        analyze_per_cell(name, rows)
    analyze_chaotic_summary()
    mg_mechanism()


if __name__ == "__main__":
    main()
