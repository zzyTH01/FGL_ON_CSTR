#!/usr/bin/env python
"""Diagnostic plot for adaptive weight experiment — variant C at L=20,H=15,seed=0."""
import pickle, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)
from utils.utils import RNN, create_time_series_dataset

L, H, seed, alpha, T = 20, 15, 0, 0.5, 4
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data_h2o.pkl")
device = torch.device("mps")
torch.manual_seed(seed)
np.random.seed(seed)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
x_raw = np.array([float(pt[0]) for pt in data])
y_raw = np.array([float(pt[1]) for pt in data])
all_y = []
for i in range(len(x_raw) - L - 1 + 1):
    all_y.append(y_raw[i + L + 1 - 1])
edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), 49)
celoss = nn.CrossEntropyLoss()

# Datasets
tt, tv, ttest, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=1,
    num_bins=50, val_size=0.2, test_size=0.2, offset=H-1, batch_size=64, bin_edges=edges)
st, sv, stest, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=H,
    num_bins=50, val_size=0.2, test_size=0.2, offset=0, batch_size=64, bin_edges=edges)
# Full loaders for weight computation
st_full, _, _, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=H,
    num_bins=50, val_size=0.2, test_size=0.2, offset=0, batch_size=1, bin_edges=edges)
tt_full, _, _, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=1,
    num_bins=50, val_size=0.2, test_size=0.2, offset=H-1, batch_size=1, bin_edges=edges)

# Train teacher
teacher = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(teacher.parameters(), lr=1e-4)
for _ in range(30):
    for _, x, y in tt: opt.zero_grad(); celoss(teacher(x.float().to(device).view(-1,1,L)), y.long().to(device)).backward(); opt.step()
teacher.eval()

# Train baseline
baseline = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(baseline.parameters(), lr=1e-4)
for _ in range(30):
    for _, x, y in st: opt.zero_grad(); celoss(baseline(x.float().to(device).view(-1,1,L)), y.long().to(device)).backward(); opt.step()
baseline.eval()

# Compute per-sample weights (variant C)
baseline_errors = {}
with torch.no_grad():
    for indices, x, y in st_full:
        x = x.float().to(device).view(-1, 1, L)
        yt = y.long().to(device)
        err = celoss(baseline(x), yt).item()
        baseline_errors[indices[0].item()] = err

teacher_errors = {}
with torch.no_grad():
    for indices, x, y in tt_full:
        x = x.float().to(device).view(-1, 1, L)
        yt = y.long().to(device)
        err = celoss(teacher(x), yt).item()
        teacher_errors[indices[0].item()] = err

train_indices = []
for indices, _, _ in st_full:
    train_indices.append(indices[0].item())

raw_w = np.array([max(0.0, baseline_errors.get(i,0) - teacher_errors.get(i,0)) for i in train_indices])
p5, p95 = np.percentile(raw_w, 5), np.percentile(raw_w, 95)
norm_w = 0.2 + 1.8 * (np.clip(raw_w, p5, p95) - p5) / (p95 - p5)
weights_dict = {i: float(norm_w[j]) for j, i in enumerate(train_indices)}

# Train student with weighted KL
def KL_weighted(s_logits, t_logits, T, alpha, w):
    lp = F.log_softmax(s_logits / T, dim=1)
    pt = F.softmax(t_logits / T, dim=1)
    kl = F.kl_div(lp, pt, reduction='none').sum(dim=1)
    return (1-alpha) * (T**2) * (w * kl).mean()

student_w = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(student_w.parameters(), lr=1e-4)
for _ in range(30):
    student_w.train()
    for (idx_s, xs, ys), (_, xt, _) in zip(st, tt):
        xs = xs.float().to(device).view(-1,1,L)
        ys_l = ys.long().to(device)
        xtt = xt.float().to(device).view(-1,1,L)
        out = student_w(xs)
        with torch.no_grad(): tl = teacher(xtt)
        bw = torch.tensor([weights_dict.get(i.item(), 1.0) for i in idx_s], dtype=torch.float32, device=device)
        loss = alpha * celoss(out, ys_l) + KL_weighted(out, tl, T, alpha, bw)
        opt.zero_grad(); loss.backward(); opt.step()
student_w.eval()

# Train standard student for comparison (variant A)
student_a = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(student_a.parameters(), lr=1e-4)
for _ in range(30):
    student_a.train()
    for (_, xs, ys), (_, xt, _) in zip(st, tt):
        xs = xs.float().to(device).view(-1,1,L)
        ys_l = ys.long().to(device)
        xtt = xt.float().to(device).view(-1,1,L)
        out = student_a(xs)
        with torch.no_grad(): tl = teacher(xtt)
        # Standard unweighted KL
        from utils.utils import KL as KL_orig
        loss = alpha * celoss(out, ys_l) + KL_orig(out, tl, T, alpha)
        opt.zero_grad(); loss.backward(); opt.step()
student_a.eval()

# Per-sample errors on test set
mse = nn.MSELoss()
errors = {"Teacher": [], "Baseline": [], "Student_A": [], "Student_C": []}
with torch.no_grad():
    for _, x, y in ttest:
        x = x.float().to(device).view(-1,1,L); yt = y.long().to(device).squeeze(-1)
        p = teacher(x).argmax(dim=1).float()
        for i in range(len(p)): errors["Teacher"].append((p[i].item()-yt[i].item())**2)
    for _, x, y in stest:
        x = x.float().to(device).view(-1,1,L); yt = y.long().to(device).squeeze(-1)
        bp = baseline(x).argmax(dim=1).float()
        sp_a = student_a(x).argmax(dim=1).float()
        sp_c = student_w(x).argmax(dim=1).float()
        for i in range(len(bp)):
            errors["Baseline"].append((bp[i].item()-yt[i].item())**2)
            errors["Student_A"].append((sp_a[i].item()-yt[i].item())**2)
            errors["Student_C"].append((sp_c[i].item()-yt[i].item())**2)

# ================================================================
# Plot 1: Per-sample MSE comparison
# ================================================================
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)

colors = {"Teacher": "#2196F3", "Baseline": "#FF9800", "Student_A": "#4CAF50", "Student_C": "#E91E63"}
titles = {
    "Teacher": "Teacher (1-step ahead)", "Baseline": "Baseline (no FGL)",
    "Student_A": "Student — Variant A (unweighted KL)", "Student_C": "Student — Variant C (difference-weighted KL)"
}

for ax, (name, errs) in zip(axes, errors.items()):
    idx = np.arange(len(errs))
    c = colors[name]
    ax.scatter(idx, errs, s=6, alpha=0.5, color=c, edgecolors="none")
    running = np.convolve(errs, np.ones(20)/20, mode="same")
    ax.plot(idx, running, color="black", linewidth=1.5, alpha=0.8, label="Running mean (w=20)")
    ax.axhline(y=np.mean(errs), color=c, linestyle="--", linewidth=2, label=f"Mean={np.mean(errs):.1f}")
    ax.set_ylabel("Sq. Error (bins²)")
    ax.set_title(titles[name])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-50, max(np.max(errs)*1.1, 100))

axes[-1].set_xlabel("Test Sample Index")
fig.suptitle(f"CSTR H₂O — Adaptive Weight Diagnostic  |  L={L} H={H} α={alpha} T={T} seed={seed}",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out1 = os.path.join(os.path.dirname(SCRIPT_DIR), "cstr_adaptive_diagnostic.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()

# ================================================================
# Plot 2: Weight distribution + Student A vs C error difference
# ================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

# Weight histogram
axes2[0].hist(raw_w, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes2[0].axvline(x=np.mean(raw_w), color="red", linestyle="--", linewidth=2, label=f"Mean={np.mean(raw_w):.3f}")
axes2[0].axvline(x=np.median(raw_w), color="orange", linestyle="--", linewidth=2, label=f"Median={np.median(raw_w):.3f}")
axes2[0].set_xlabel("Raw weight = max(0, CE_baseline − CE_teacher)")
axes2[0].set_ylabel("Count")
axes2[0].set_title("Weight Distribution (Training Set)")
axes2[0].legend(fontsize=9)

# Student error comparison: C vs A per sample
s_a = np.array(errors["Student_A"])
s_c = np.array(errors["Student_C"])
diff = s_a - s_c  # positive = C better than A

# Running mean of difference
running_diff = np.convolve(diff, np.ones(30)/30, mode="same")
axes2[1].bar(np.arange(len(diff)), diff, color=np.where(diff>0, "green", "red"), alpha=0.3, width=1)
axes2[1].plot(np.arange(len(diff)), running_diff, color="black", linewidth=2, label="Running mean (w=30)")
axes2[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
axes2[1].axhline(y=np.mean(diff), color="blue", linestyle="--", linewidth=1.5, label=f"Mean diff={np.mean(diff):+.1f}")
axes2[1].set_xlabel("Test Sample Index")
axes2[1].set_ylabel("MSE(A) − MSE(C)  (positive = C better)")
axes2[1].set_title("Student Error Difference: Variant A − Variant C")
axes2[1].legend(fontsize=9)

fig2.suptitle(f"Adaptive Weight Analysis — Variant C  |  L={L} H={H} seed={seed}", fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(os.path.dirname(SCRIPT_DIR), "cstr_weight_diagnostic.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# Stats
print(f"\nMSE Summary:")
for name in ["Teacher", "Baseline", "Student_A", "Student_C"]:
    e = np.array(errors[name])
    print(f"  {name:>12}: mean={np.mean(e):.1f} median={np.median(e):.1f} "
          f"p95={np.percentile(e,95):.1f} max={np.max(e):.0f}")

delta_a = (np.mean(errors["Baseline"])-np.mean(errors["Student_A"]))/np.mean(errors["Baseline"])*100
delta_c = (np.mean(errors["Baseline"])-np.mean(errors["Student_C"]))/np.mean(errors["Baseline"])*100
print(f"\n  Student_A Δ={delta_a:+.1f}%")
print(f"  Student_C Δ={delta_c:+.1f}%")
print(f"  Weight stats: raw mean={raw_w.mean():.3f} median={raw_w.median():.3f} >0 frac={(raw_w>0.01).mean():.1%}")
