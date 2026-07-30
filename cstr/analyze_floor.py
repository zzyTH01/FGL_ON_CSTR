#!/usr/bin/env python
"""读 floor_sweep.csv + lyapunov_tau.csv,出 H1-H4 的判决、图、与 conclusion/floor_determinants.md。

H1: baseline 地板 vs (L,H),且固定 H 扫 L 时 L+H-1≈τ_data 处相变?
H2: c = E_iter/baseline 在 (L,H) 上近似常数?
H3: floor ∝ teacher_mse 且 log(floor) vs H 斜率≈λ?
H4: E_iter ≈ baseline_converged?(蒸馏是否压破数据地板)

用法::

    uv run python cstr/analyze_floor.py
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def _agg(rows, group_keys, value_key):
    """mean over seeds for each group."""
    buckets = defaultdict(list)
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        buckets[k].append(float(r[value_key]))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def analyze(csv_path="cstr/results/floor_sweep.csv",
            lyap_path="cstr/results/lyapunov_tau.csv",
            outdir="cstr/results", conclusion_dir="conclusion",
            deep_label="tau100"):
    rows = _load_csv(csv_path)
    for r in rows:
        r["L"] = int(r["L"]); r["H"] = int(r["H"])
        r["LplusH_minus_1"] = int(r["LplusH_minus_1"])
    deep = [r for r in rows if r["dataset"] == deep_label]

    # ---- H1: baseline floor vs (L,H) + transition on H=15 column ----
    base_grid = _agg(deep, ["L", "H"], "baseline_mse")
    Ls = sorted({r["L"] for r in deep})
    Hs = sorted({r["H"] for r in deep})
    h1_heat = np.full((len(Ls), len(Hs)), np.nan)
    for i, L in enumerate(Ls):
        for j, H in enumerate(Hs):
            h1_heat[i, j] = base_grid.get((L, H), np.nan)
    # transition along H=15 (or nearest available H)
    h15 = sorted([(r["LplusH_minus_1"], float(np.mean(
        [float(x["baseline_mse"]) for x in deep if x["L"] == L and x["H"] == 15])))
        for L in {r["L"] for r in deep if r["H"] == 15}])
    h1_transition_drop = 0.0
    h1_transition_at = None
    if len(h15) >= 2:
        xs = np.array([a for a, _ in h15]); ys = np.array([b for _, b in h15])
        ratios = ys[:-1] / np.maximum(ys[1:], 1e-9)
        idx = int(np.argmax(ratios))
        h1_transition_drop = float(ratios[idx])
        h1_transition_at = float(xs[idx + 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(h1_heat, aspect="auto", origin="lower", cmap="YlOrRd",
                   extent=[Hs[0] - 0.5, Hs[-1] + 0.5, Ls[0] - 0.5, Ls[-1] + 0.5])
    ax.set_xticks(Hs); ax.set_yticks(Ls); ax.set_xlabel("H"); ax.set_ylabel("L")
    ax.set_title(f"H1: baseline_mse floor ({deep_label})")
    plt.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "floor_h1_baseline_heatmap.png"), dpi=150)
    plt.close(fig)

    if h15:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([a for a, _ in h15], [b for _, b in h15], "o-")
        ax.axvline(100, color="r", ls="--", label="τ_data=100")
        ax.set_xlabel("L+H-1"); ax.set_ylabel("baseline_mse")
        ax.set_title("H1: baseline floor vs L+H-1 (H=15)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h1_transition.png"), dpi=150)
        plt.close(fig)

    # ---- H2: c = E_iter / baseline, constant across (L,H)? ----
    cs = []
    for r in deep:
        b = float(r["baseline_mse"])
        if b > 1e-9:
            cs.append(float(r["E_iter_mse"]) / b)
    cs = np.array(cs)
    h2_c_mean = float(np.mean(cs)) if len(cs) else float("nan")
    h2_c_cv = float(np.std(cs, ddof=1) / np.abs(h2_c_mean)) if len(cs) > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(cs); ax.set_ylabel("c = E_iter / baseline")
    ax.set_title(f"H2: c constancy (mean={h2_c_mean:.2f}, CV={h2_c_cv:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h2_c_ratio.png"), dpi=150)
    plt.close(fig)

    # ---- H3: floor vs teacher_mse + log(floor) vs H ----
    teacher = _to_float(deep, "teacher_mse")
    floor_conv = _to_float(deep, "baseline_converged_mse")
    lr = sstats.linregress(teacher, floor_conv)
    h3_r2 = float(lr.rvalue ** 2)
    h3_slope = float(lr.slope)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(teacher, floor_conv)
    xs = np.linspace(teacher.min(), teacher.max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, "r-", label=f"R²={h3_r2:.2f}")
    ax.set_xlabel("teacher_mse"); ax.set_ylabel("floor (baseline_converged)")
    ax.set_title("H3: floor vs teacher_mse"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h3_floor_vs_teacher.png"), dpi=150)
    plt.close(fig)

    logfloor_vs_H = _agg(deep, ["H"], "baseline_converged_mse")
    Hs_h3 = sorted(int(k[0]) for k in logfloor_vs_H)
    floor_H = np.array([logfloor_vs_H[(h,)] for h in Hs_h3])
    mask = floor_H > 0
    h3_logfloor_slope = float("nan")
    if mask.sum() >= 2:
        h3_logfloor_slope = float(sstats.linregress(np.array(Hs_h3)[mask], np.log(floor_H[mask])).slope)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Hs_h3, np.log(np.maximum(floor_H, 1e-9)), "o-")
    ax.set_xlabel("H"); ax.set_ylabel("log(floor)"); ax.set_title("H3: log(floor) vs H")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h3_logfloor_vs_H.png"), dpi=150)
    plt.close(fig)

    h3_lambda = float("nan")
    if os.path.exists(lyap_path):
        for lr_row in _load_csv(lyap_path):
            if str(lr_row.get("tau")) == "100" or lr_row.get("file", "").find("tau100") >= 0:
                h3_lambda = float(lr_row["lyap"])
                break

    # ---- H4: E_iter vs baseline_converged (paired) ----
    e_iter = _to_float(deep, "E_iter_mse")
    base_c = _to_float(deep, "baseline_converged_mse")
    if len(e_iter) >= 2:
        t = sstats.ttest_rel(e_iter, base_c)
        h4_paired_p = float(t.pvalue)
    else:
        h4_paired_p = float("nan")
    h4_mean_diff = float(np.mean(e_iter - base_c)) if len(e_iter) else float("nan")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(base_c, e_iter)
    lo = min(np.nanmin(base_c), np.nanmin(e_iter))
    hi = max(np.nanmax(base_c), np.nanmax(e_iter))
    ax.plot([lo, hi], [lo, hi], "r--", label="y=x")
    ax.set_xlabel("baseline_converged_mse"); ax.set_ylabel("E_iter_mse")
    ax.set_title(f"H4: paired p={h4_paired_p:.3g}, mean(E_iter-baseC)={h4_mean_diff:+.1f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "floor_h4_paired.png"), dpi=150)
    plt.close(fig)

    # ---- verdicts ----
    h1_verdict = ("支持" if h1_transition_drop > 1.5 else "弱/无明显相变") + \
                 f"(最大连续降幅 {h1_transition_drop:.2f}× @ L+H-1≈{h1_transition_at})"
    h2_verdict = ("支持常数 c" if (np.isfinite(h2_c_cv) and h2_c_cv < 0.25) else "c 有漂移") + \
                 f"(c={h2_c_mean:.2f}±CV={h2_c_cv:.2f})"
    h3_verdict = ("支持 floor∝teacher" if h3_r2 > 0.7 else "弱关系") + \
                 f"(R²={h3_r2:.2f}, slope={h3_slope:.2f}, log(floor)/H={h3_logfloor_slope:.4f}, λ={h3_lambda:.4f})"
    if not np.isfinite(h4_paired_p):
        h4_verdict = "样本不足"
    elif h4_paired_p > 0.05:
        h4_verdict = f"支持:蒸馏≈数据地板(p={h4_paired_p:.3g}, 无法压破)"
    else:
        win = "压破" if h4_mean_diff < 0 else "反而更高"
        h4_verdict = f"证伪:蒸馏{win}了地板(p={h4_paired_p:.3g}, Δ={h4_mean_diff:+.1f})"

    os.makedirs(conclusion_dir, exist_ok=True)
    md_path = os.path.join(conclusion_dir, "floor_determinants.md")
    with open(md_path, "w") as f:
        f.write(f"# 延迟 CSTR 地板成因结论\n\n")
        f.write(f"**数据:** `{csv_path}` + `{lyap_path}`\n")
        f.write(f"**深挖数据集:** {deep_label}\n\n")
        f.write("## 四假设判决\n\n")
        f.write(f"- **H1**(L+H-1≥τ 相变): {h1_verdict}\n")
        f.write(f"- **H2**(c=E_iter/baseline 常数): {h2_verdict}\n")
        f.write(f"- **H3**(floor∝teacher_mse, Lyapunov 标定): {h3_verdict}\n")
        f.write(f"- **H4**(蒸馏 vs 匹配算力 baseline 地板): {h4_verdict}\n\n")
        f.write("## 如何达到最低 MSE\n\n")
        if np.isfinite(h4_paired_p) and h4_paired_p > 0.05:
            f.write("H4 成立 ⇒ **最好的 MSE 就是数据地板**:把 baseline 训到收敛即可达到,"
                    "连续蒸馏不额外加分。压低地板的关键在 (L, H) 与数据本身的可预测性。\n")
        else:
            f.write("H4 证伪 ⇒ 连续蒸馏确实改变了地板;达到最低 MSE 需用连续蒸馏(配合 H2/H3 的 (L,H,teacher) 规律选点)。\n")
    print(f"wrote {md_path}")

    return {"h1_transition_drop": h1_transition_drop,
            "h1_transition_at": h1_transition_at,
            "h2_c_mean": h2_c_mean, "h2_c_cv": h2_c_cv,
            "h3_floor_vs_teacher_r2": h3_r2, "h3_floor_vs_teacher_slope": h3_slope,
            "h3_logfloor_vs_H_slope": h3_logfloor_slope, "h3_lambda": h3_lambda,
            "h4_paired_p": h4_paired_p, "h4_mean_diff": h4_mean_diff}


def main():
    p = argparse.ArgumentParser(description="Analyze floor_sweep.csv → H1-H4 verdicts")
    p.add_argument("--csv", default="cstr/results/floor_sweep.csv")
    p.add_argument("--lyap", default="cstr/results/lyapunov_tau.csv")
    p.add_argument("--outdir", default="cstr/results")
    p.add_argument("--conclusion_dir", default="conclusion")
    p.add_argument("--deep_label", default="tau100")
    args = p.parse_args()
    analyze(args.csv, args.lyap, args.outdir, args.conclusion_dir, args.deep_label)


if __name__ == "__main__":
    main()
