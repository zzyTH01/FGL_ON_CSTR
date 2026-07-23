#!/usr/bin/env python
"""Best adaptive distillation (variant C, L=20,H=15,seed=3) — prediction vs actual."""
import pickle, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)
from utils.utils import RNN, create_time_series_dataset

L, H, seed, alpha, T = 20, 15, 3, 0.5, 4
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data_h2o.pkl")
device = torch.device("mps")
torch.manual_seed(seed)
np.random.seed(seed)
celoss = nn.CrossEntropyLoss()

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
x_raw = np.array([float(pt[0]) for pt in data])
y_raw = np.array([float(pt[1]) for pt in data])
all_y = []
for i in range(len(x_raw) - L - 1 + 1):
    all_y.append(y_raw[i + L + 1 - 1])
edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), 49)
bin_centers = (edges[:-1] + edges[1:]) / 2
bin_centers = np.concatenate([[edges[0]-(edges[1]-edges[0])/2], bin_centers, [edges[-1]+(edges[1]-edges[0])/2]])

def bin_to_h2o(b):
    return bin_centers[np.clip(int(b), 0, len(bin_centers)-1)]

# Datasets
tt, tv, ttest, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=1,
    num_bins=50, val_size=0.2, test_size=0.2, offset=H-1, batch_size=64, bin_edges=edges)
st, sv, stest, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=H,
    num_bins=50, val_size=0.2, test_size=0.2, offset=0, batch_size=64, bin_edges=edges)
st_full, _, _, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=H,
    num_bins=50, val_size=0.2, test_size=0.2, offset=0, batch_size=1, bin_edges=edges)
tt_full, _, _, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=1,
    num_bins=50, val_size=0.2, test_size=0.2, offset=H-1, batch_size=1, bin_edges=edges)

print("Training Teacher...")
teacher = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(teacher.parameters(), lr=1e-4)
for _ in range(30):
    for _, x, y in tt: opt.zero_grad(); celoss(teacher(x.float().to(device).view(-1,1,L)), y.long().to(device)).backward(); opt.step()
teacher.eval()

print("Training Baseline...")
baseline = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(baseline.parameters(), lr=1e-4)
for _ in range(30):
    for _, x, y in st: opt.zero_grad(); celoss(baseline(x.float().to(device).view(-1,1,L)), y.long().to(device)).backward(); opt.step()
baseline.eval()

print("Computing weights (variant C)...")
b_errs, t_errs = {}, {}
with torch.no_grad():
    for indices, x, y in st_full:
        x = x.float().to(device).view(-1,1,L)
        b_errs[indices[0].item()] = celoss(baseline(x), y.long().to(device)).item()
    for indices, x, y in tt_full:
        x = x.float().to(device).view(-1,1,L)
        t_errs[indices[0].item()] = celoss(teacher(x), y.long().to(device)).item()

train_idx = [indices[0].item() for indices, _, _ in st_full]
raw_w = np.array([max(0.0, b_errs.get(i,0)-t_errs.get(i,0)) for i in train_idx])
p5, p95 = np.percentile(raw_w, 5), np.percentile(raw_w, 95)
norm_w = 0.2 + 1.8 * (np.clip(raw_w, p5, p95) - p5) / (p95 - p5)
w_dict = {i: float(norm_w[j]) for j, i in enumerate(train_idx)}

print("Training Student (variant C, difference-weighted KL)...")
student = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(student.parameters(), lr=1e-4)
for _ in range(30):
    student.train()
    for (idx_s, xs, ys), (_, xt, _) in zip(st, tt):
        xs = xs.float().to(device).view(-1,1,L); ys_l = ys.long().to(device)
        xtt = xt.float().to(device).view(-1,1,L)
        out = student(xs)
        with torch.no_grad(): tl = teacher(xtt)
        bw = torch.tensor([w_dict.get(i.item(),1.0) for i in idx_s], dtype=torch.float32, device=device)
        lp = F.log_softmax(out/T, dim=1); pt = F.softmax(tl/T, dim=1)
        kl = F.kl_div(lp, pt, reduction='none').sum(dim=1)
        loss = alpha*celoss(out,ys_l) + (1-alpha)*(T**2)*(bw*kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
student.eval()

# ---- Predictions ----
print("Generating predictions...")
t_preds, t_targets, b_preds, s_preds, s_targets = [], [], [], [], []
with torch.no_grad():
    for _, x, y in ttest:
        x = x.float().to(device).view(-1,1,L); yt = y.long().to(device).squeeze(-1)
        p = teacher(x).argmax(dim=1)
        for i in range(len(p)): t_preds.append(p[i].item()); t_targets.append(yt[i].item())
    for _, x, y in stest:
        x = x.float().to(device).view(-1,1,L); yt = y.long().to(device).squeeze(-1)
        bp = baseline(x).argmax(dim=1); sp = student(x).argmax(dim=1)
        for i in range(len(bp)):
            b_preds.append(bp[i].item()); s_preds.append(sp[i].item()); s_targets.append(yt[i].item())

t_preds=np.array(t_preds); t_targets=np.array(t_targets)
b_preds=np.array(b_preds); s_preds=np.array(s_preds); s_targets=np.array(s_targets)

N_PLOT = min(200, len(s_targets))
start = len(s_targets) - N_PLOT

# Convert to H2O
t_p_h2o = np.array([bin_to_h2o(p) for p in t_preds[start:]])
t_t_h2o = np.array([bin_to_h2o(t) for t in t_targets[start:]])
b_p_h2o = np.array([bin_to_h2o(p) for p in b_preds[start:]])
s_p_h2o = np.array([bin_to_h2o(p) for p in s_preds[start:]])
s_t_h2o = np.array([bin_to_h2o(t) for t in s_targets[start:]])

# ================================================================
# Plot
# ================================================================
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(2, 3, figsize=(21, 11))

# Row 1: Bin space
for ax, preds, targets, title, color in [
    (axes[0,0], t_preds[start:], t_targets[start:], "Teacher (1-step)", "#2196F3"),
    (axes[0,1], b_preds[start:], s_targets[start:], f"Baseline (H={H}, no FGL)", "#FF9800"),
    (axes[0,2], s_preds[start:], s_targets[start:], f"Student C — Adaptive Weighted KL\n(Variant C, Δ=+24.7%)", "#E91E63"),
]:
    ax.plot(targets, 'o', ms=3, alpha=0.5, color='gray', label='Actual')
    ax.plot(preds, 's', ms=3, alpha=0.7, color=color, label='Predicted')
    ax.set_title(title); ax.set_xlabel("Test sample"); ax.set_ylabel("Bin (0–49)")
    ax.legend(fontsize=8); ax.set_ylim(0, 50)

# Row 2: H2O space
for ax, preds, targets, title, color in [
    (axes[1,0], t_p_h2o, t_t_h2o, "Teacher (1-step)", "#2196F3"),
    (axes[1,1], b_p_h2o, s_t_h2o, f"Baseline (H={H}, no FGL)", "#FF9800"),
    (axes[1,2], s_p_h2o, s_t_h2o, f"Student C — Adaptive Weighted KL\n(Variant C, Δ=+24.7%)", "#E91E63"),
]:
    ax.plot(targets, 'o', ms=3, alpha=0.5, color='gray', label='Actual')
    ax.plot(preds, 's', ms=3, alpha=0.7, color=color, label='Predicted')
    ax.set_title(title); ax.set_xlabel("Test sample"); ax.set_ylabel("H₂O mass fraction")
    ax.legend(fontsize=8)

fig.suptitle(f"CSTR H₂O — Adaptive Distillation (Variant C, L={L} H={H} α={alpha} seed={seed})  |  Δ=+24.7%",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out1 = os.path.join(os.path.dirname(SCRIPT_DIR), "cstr_adaptive_best_predictions.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()

# ================================================================
# Scatter: Predicted vs Actual (Baseline vs Student C)
# ================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

for ax, preds, targets, title, color in [
    (axes2[0], b_p_h2o, s_t_h2o, f"Baseline (no FGL)\nMSE(bin)={np.mean((b_preds-s_targets)**2):.1f}", "#FF9800"),
    (axes2[1], s_p_h2o, s_t_h2o, f"Student C — Adaptive Weighted KL\nMSE(bin)={np.mean((s_preds-s_targets)**2):.1f}", "#E91E63"),
]:
    ax.scatter(targets, preds, alpha=0.4, s=12, color=color, edgecolors='none')
    ax.plot([0,1], [0,1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_title(title); ax.set_xlabel("Actual H₂O"); ax.set_ylabel("Predicted H₂O")

fig2.suptitle(f"Predicted vs Actual — Adaptive Distillation (Variant C, L={L} H={H} seed={seed})",
              fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(os.path.dirname(SCRIPT_DIR), "cstr_adaptive_best_scatter.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# Stats
print(f"\nResults:")
print(f"  Teacher  MSE(bin)={np.mean((t_preds-t_targets)**2):.1f}")
print(f"  Baseline MSE(bin)={np.mean((b_preds-s_targets)**2):.1f}")
print(f"  Student  MSE(bin)={np.mean((s_preds-s_targets)**2):.1f}")
delta = (np.mean((b_preds-s_targets)**2)-np.mean((s_preds-s_targets)**2))/np.mean((b_preds-s_targets)**2)*100
print(f"  Δ = {delta:+.1f}%")
