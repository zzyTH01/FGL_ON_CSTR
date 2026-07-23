#!/usr/bin/env python
"""
CSTR L×H combined sweep.

CSTR H2O sub-cycle period ≈ 72 steps. Sweep L and H from defaults (8,5)
up to ~70% of period (~50 steps), with coarser grid to keep ≤30 configs.

5 L values × 5 H values = 25 configs × 3 seeds = 75 runs.

Usage:
  uv run python cstr/exp/fgl_cstr_lh_sweep.py
"""

import argparse, pickle, sys, os, csv, time
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.utils import RNN, create_time_series_dataset, KL

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")

DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data_h2o.pkl")

L_VALUES = [8, 20, 35, 50, 72]   # up to full sub-cycle
H_VALUES = [5, 15, 30, 45, 60]    # up to ~83% of period
SEEDS = [0, 1, 2]
EPOCHS = 30
ALPHA = 0.5
TEMPERATURE = 4
NUM_BINS = 50
PATIENCE = 5
BATCH_SIZE = 64


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.p, self.d, self.b, self.c, self.bs = patience, min_delta, float('inf'), 0, None
    def step(self, l, m):
        if l + self.d < self.b: self.b = l; self.c = 0; self.bs = {k: v.cpu() for k, v in m.state_dict().items()}; return False
        self.c += 1; return self.c >= self.p
    def restore(self, m):
        if self.bs: m.load_state_dict(self.bs)


def run_fgl(L, H, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    hs, nl, lr, os_, vs, ts = 128, 2, 1e-4, NUM_BINS, 0.2, 0.2

    xr = np.array([float(pt[0]) for pt in data]); yr = np.array([float(pt[1]) for pt in data])
    ay = [yr[i + L] for i in range(len(xr) - L - 1 + 1)]
    se = np.linspace(np.array(ay).min(), np.array(ay).max(), os_ - 1)

    tt, tv, tte, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=1,
        num_bins=os_, val_size=vs, test_size=ts, offset=H-1, batch_size=BATCH_SIZE, bin_edges=se)
    st, sv, ste, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=os_, val_size=vs, test_size=ts, offset=0, batch_size=BATCH_SIZE, bin_edges=se)

    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    def train_m(model, loader, vloader, opt):
        es = EarlyStopper(PATIENCE)
        for _ in range(EPOCHS):
            model.train()
            for _, x, y in loader: x = x.float().to(device).view(-1,1,L); y = y.long().to(device); opt.zero_grad(); ce(model(x), y).backward(); opt.step()
            model.eval()
            with torch.no_grad(): vl = sum(ce(model(x.float().to(device).view(-1,1,L)), y.long().to(device)).item() for _,x,y in vloader)/len(vloader)
            if es.step(vl, model): break
        es.restore(model)

    teacher = RNN(L, hs, os_, nl).to(device)
    train_m(teacher, tt, tv, optim.Adam(teacher.parameters(), lr=lr))

    baseline = RNN(L, hs, os_, nl).to(device)
    train_m(baseline, st, sv, optim.Adam(baseline.parameters(), lr=lr))

    student = RNN(L, hs, os_, nl).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr); es_s = EarlyStopper(PATIENCE)
    for _ in range(EPOCHS):
        student.train()
        for (_,xs,ys),(_,xt,_) in zip(st,tt):
            xs=xs.float().to(device).view(-1,1,L); ys=ys.long().to(device); out=student(xs)
            xt=xt.float().to(device).view(-1,1,L)
            with torch.no_grad(): lt=teacher(xt)
            loss=ALPHA*ce(out,ys)+KL(out,lt,TEMPERATURE,ALPHA); opt_s.zero_grad(); loss.backward(); opt_s.step()
        student.eval()
        with torch.no_grad(): vl=sum(ce(student(x.float().to(device).view(-1,1,L)), y.long().to(device)).item() for _,x,y in sv)/len(sv)
        if es_s.step(vl, student): break
    es_s.restore(student)

    def ev(m,ld):
        m.eval(); t=0.0
        with torch.no_grad():
            for _,x,y in ld: x=x.float().to(device).view(-1,1,L); t+=mse(m(x).argmax(1).float(), y.float().to(device).squeeze(-1)).item()
        return t/len(ld)
    bm=ev(baseline,ste); sm=ev(student,ste); tm=ev(teacher,tte)
    return bm,sm,tm


def main():
    with open(DATA_PATH, "rb") as f: data = pickle.load(f)
    n_pts = data.shape[0]; period = 72
    print(f"CSTR H₂O: {n_pts} pts, sub-cycle ≈ {period} steps")
    print(f"L: {L_VALUES}  |  H: {H_VALUES}")
    total = len(L_VALUES)*len(H_VALUES)*len(SEEDS)
    print(f"Configs: {len(L_VALUES)*len(H_VALUES)}, Total runs: {total}, Est: ~{total*0.7:.0f} min\n")

    outdir = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "cstr_lh_sweep.csv")

    rows = []; t0 = time.time()
    for L in L_VALUES:
        for H in H_VALUES:
            n_win = n_pts - L - H + 1
            label = f"L={L:2d} H={H:2d} (L+H-1={L+H-1:3d}, {n_win} windows)"
            print(f"\n{'='*50}\n  {label}\n{'='*50}")
            bms, sms, tms = [], [], []
            for s in tqdm(SEEDS, desc="  Seeds"):
                bm, sm, tm = run_fgl(L, H, data, s)
                bms.append(bm); sms.append(sm); tms.append(tm)
                d = (bm-sm)/bm*100 if bm>0 else 0
                rows.append({"L":L,"H":H,"seed":s,"baseline_mse":bm,"teacher_mse":tm,"student_mse":sm,"abs_improvement":bm-sm,"fgl_delta":d})
            print(f"  Base={np.mean(bms):.2f}±{np.std(bms):.2f}  "
                  f"Stu={np.mean(sms):.2f}±{np.std(sms):.2f}  "
                  f"Δ={np.mean([(b-s)/b*100 if b>0 else 0 for b,s in zip(bms,sms)]):+.1f}%  "
                  f"({(time.time()-t0)/60:.1f} min)")

    # Save CSV
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["L","H","seed","baseline_mse","teacher_mse","student_mse","abs_improvement","fgl_delta"]); w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Aggregate
    agg = defaultdict(list)
    for r in rows: agg[(r["L"],r["H"])].append(r)
    L_s, H_s = sorted(set(r["L"] for r in rows)), sorted(set(r["H"] for r in rows))

    # Heatmap data
    grid_d = np.zeros((len(L_s), len(H_s)))
    grid_b = np.zeros((len(L_s), len(H_s)))
    grid_a = np.zeros((len(L_s), len(H_s)))
    for i, L in enumerate(L_s):
        for j, H in enumerate(H_s):
            rs = agg[(L,H)]
            bm_m = np.mean([r["baseline_mse"] for r in rs])
            grid_d[i,j] = np.mean([r["fgl_delta"] for r in rs])
            grid_b[i,j] = bm_m
            grid_a[i,j] = np.mean([r["abs_improvement"] for r in rs])

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    for ax, grid, title, cmap, label in [
        (axes[0], grid_d, "FGL Δ%", "RdYlGn", "Δ%"),
        (axes[1], grid_a, "Abs Improvement (Base−Stu)", "RdYlGn", "Abs Imp"),
        (axes[2], grid_b, "Baseline MSE (task difficulty)", "YlOrRd", "Base MSE"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                       extent=[H_s[0]-0.5, H_s[-1]+0.5, L_s[0]-0.5, L_s[-1]+0.5])
        ax.set_xticks(H_s); ax.set_yticks(L_s)
        ax.set_xlabel("H (forecast horizon)"); ax.set_ylabel("L (lookback)")
        ax.set_title(title, fontweight="bold")
        for i in range(len(L_s)):
            for j in range(len(H_s)):
                v = grid[i,j]
                ax.text(H_s[j], L_s[i], f"{v:.0f}" if abs(v)<100 else f"{v:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > (grid.max()+grid.min())/3 else "black")
        plt.colorbar(im, ax=ax, label=label)

    plt.suptitle(f"CSTR L×H Sweep (sub-cycle ≈ 72 steps)", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "cstr_lh_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {outdir}/cstr_lh_sweep.png")

    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS: CSTR L×H Sweep")
    print("=" * 70)
    lh_label = "L\\H"
    print(f"  {lh_label:>5s}", end="")
    for H in H_s: print(f"  {H:>8d}", end="")
    print("\n  " + "-" * (5+9*len(H_s)))
    for L in L_s:
        print(f"  {L:5d}", end="")
        for H in H_s:
            rs = agg[(L,H)]
            dm = np.mean([r["fgl_delta"] for r in rs])
            print(f"  {dm:+7.1f}%", end="")
        print()
    print()

    # Best/worst
    all_aggs = []
    for (L,H), rs in agg.items():
        dm = np.mean([r["fgl_delta"] for r in rs]); am = np.mean([r["abs_improvement"] for r in rs])
        all_aggs.append({"L":L,"H":H,"delta":dm,"abs":am,"base":np.mean([r["baseline_mse"] for r in rs])})

    all_aggs.sort(key=lambda x: x["delta"], reverse=True)
    print("Top 5 FGL Δ:")
    for a in all_aggs[:5]: print(f"  L={a['L']:2d} H={a['H']:2d}: Δ={a['delta']:+.1f}%  Base={a['base']:.2f}")
    print("Bottom 5 FGL Δ:")
    for a in all_aggs[-5:]: print(f"  L={a['L']:2d} H={a['H']:2d}: Δ={a['delta']:+.1f}%  Base={a['base']:.2f}")

    # Report
    md_path = os.path.join(outdir, "cstr_lh_sweep_report.md")
    with open(md_path, "w") as f:
        f.write("# CSTR L×H Combined Sweep\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Data:** data_h2o.pkl ({n_pts} pts, sub-cycle ≈ {period} steps)\n")
        f.write(f"**L values:** {L_VALUES}\n")
        f.write(f"**H values:** {H_VALUES}\n")
        f.write(f"**Seeds:** {SEEDS}  |  **Epochs:** {EPOCHS}  |  **α:** {ALPHA}\n\n")

        f.write("## FGL Δ% Heatmap\n\n")
        f.write("| L\\H | " + " | ".join(f"{h}" for h in H_s) + " |\n")
        f.write("|" + "---|" * (len(H_s)+1) + "\n")
        for L in L_s:
            f.write(f"| {L} | " + " | ".join(f"{np.mean([r['fgl_delta'] for r in agg[(L,H)]]):+.1f}%" for H in H_s) + " |\n")

        f.write("\n## Abs Improvement Heatmap\n\n")
        f.write("| L\\H | " + " | ".join(f"{h}" for h in H_s) + " |\n")
        f.write("|" + "---|" * (len(H_s)+1) + "\n")
        for L in L_s:
            f.write(f"| {L} | " + " | ".join(f"{np.mean([r['abs_improvement'] for r in agg[(L,H)]]):+.1f}" for H in H_s) + " |\n")

        f.write("\n## Baseline MSE Heatmap\n\n")
        f.write("| L\\H | " + " | ".join(f"{h}" for h in H_s) + " |\n")
        f.write("|" + "---|" * (len(H_s)+1) + "\n")
        for L in L_s:
            f.write(f"| {L} | " + " | ".join(f"{np.mean([r['baseline_mse'] for r in agg[(L,H)]]):.1f}" for H in H_s) + " |\n")

        f.write(f"\n## Data: `cstr_lh_sweep.csv` ({len(rows)} rows)\n")

    print(f"Report saved: {md_path}")


if __name__ == "__main__":
    main()
