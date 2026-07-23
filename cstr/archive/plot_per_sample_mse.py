#!/usr/bin/env python
"""Plot per-sample squared error for Teacher, Baseline, and Student at L=20, H=12."""
import pickle, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mackey_glass"))
from utils.utils import RNN, create_time_series_dataset, KL

L, H, seed, alpha, T = 20, 12, 0, 0.5, 4
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_h2o.pkl")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

x_raw = np.array([float(pt[0]) for pt in data])
y_raw = np.array([float(pt[1]) for pt in data])
all_y = []
for i in range(len(x_raw) - L - 1 + 1):
    all_y.append(y_raw[i + L + 1 - 1])
edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), 49)

device = torch.device("mps")
torch.manual_seed(seed)
celoss = nn.CrossEntropyLoss()

tt, tv, ttest, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=1,
    num_bins=50, val_size=0.2, test_size=0.2, offset=H-1, batch_size=64, bin_edges=edges)
st, sv, stest, _, _ = create_time_series_dataset(data=data, lookback_window=L, forecasting_horizon=H,
    num_bins=50, val_size=0.2, test_size=0.2, offset=0, batch_size=64, bin_edges=edges)

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

# Train student
student = RNN(L, 128, 50, 2).to(device)
opt = optim.Adam(student.parameters(), lr=1e-4)
for _ in range(30):
    for (_, xs, ys), (_, xt, _) in zip(st, tt):
        xs, ys = xs.float().to(device).view(-1,1,L), ys.long().to(device)
        xtt = xt.float().to(device).view(-1,1,L)
        out = student(xs)
        with torch.no_grad(): tl = teacher(xtt)
        loss = alpha * celoss(out, ys) + KL(out, tl, T, alpha)
        opt.zero_grad(); loss.backward(); opt.step()
student.eval()

# ================================================================
# Per-sample evaluation using DataLoaders (consistent with training)
# ================================================================
t_errors, b_errors, s_errors, true_vals = [], [], [], []
t_preds, b_preds, s_preds = [], [], []

with torch.no_grad():
    # Teacher samples (on teacher test set)
    for _, x, y in ttest:
        x = x.float().to(device).view(-1, 1, L)
        yt = y.long().to(device).squeeze(-1)
        pred = teacher(x).argmax(dim=1)
        for i in range(len(pred)):
            t_errors.append((pred[i].item() - yt[i].item())**2)

    # Baseline & Student samples (on student test set)
    for _, x, y in stest:
        x = x.float().to(device).view(-1, 1, L)
        yt = y.long().to(device).squeeze(-1)
        bp = baseline(x).argmax(dim=1)
        sp = student(x).argmax(dim=1)
        for i in range(len(bp)):
            b_errors.append((bp[i].item() - yt[i].item())**2)
            s_errors.append((sp[i].item() - yt[i].item())**2)
            true_vals.append(yt[i].item())

t_errors = np.array(t_errors)
b_errors = np.array(b_errors)
s_errors = np.array(s_errors)
true_vals = np.array(true_vals)

# Note: teacher_test has different samples than student_test (teacher offset shifts window indices).
# For aligned per-sample comparison, we use student_test for baseline & student,
# and teacher_test separately for teacher. The x-axis indices differ between them.

# ================================================================
# Plot
# ================================================================
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

colors = {"Teacher": "#2196F3", "Baseline": "#FF9800", "Student": "#4CAF50"}
for ax, errors, name, idx in [
    (axes[0], t_errors, "Teacher", np.arange(len(t_errors))),
    (axes[1], b_errors, "Baseline", np.arange(len(b_errors))),
    (axes[2], s_errors, "Student", np.arange(len(s_errors))),
]:
    color = colors[name]
    # Scatter: individual squared errors
    ax.scatter(idx, errors, s=8, alpha=0.6, color=color, edgecolors="none", label=f"{name} (per sample)")
    # Running mean (window=20)
    running = np.convolve(errors, np.ones(20)/20, mode="same")
    ax.plot(idx, running, color="black", linewidth=1.5, alpha=0.8, label="Running mean (window=20)")
    # Overall mean
    ax.axhline(y=np.mean(errors), color=color, linestyle="--", linewidth=2, alpha=0.9,
               label=f"Mean = {np.mean(errors):.1f}")
    ax.set_ylabel("Squared Error (bins²)")
    ax.set_title(f"{name} — Per-Sample MSE (L={L}, H={H})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-50, max(np.max(errors)*1.1, 100))

axes[-1].set_xlabel("Test Sample Index")

fig.suptitle(f"CSTR H₂O — Per-Sample Prediction Error  |  L={L}  H={H}  α={alpha}  T={T}  seed={seed}",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cstr_per_sample_mse.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()

# ================================================================
# Second plot: true H2O values + error overlay
# ================================================================
fig2, ax2 = plt.subplots(figsize=(20, 5))

# True H2O values (bin -> original)
bin_centers = (edges[:-1] + edges[1:]) / 2
bin_centers = np.concatenate([[edges[0] - (edges[1]-edges[0])/2], bin_centers, [edges[-1] + (edges[1]-edges[0])/2]])
true_h2o = bin_centers[np.clip(true_vals, 0, len(bin_centers)-1)]

ax2.plot(np.arange(len(true_h2o)), true_h2o, color="gray", linewidth=1.5, alpha=0.7, label="True H₂O")
ax2.set_ylabel("H₂O Mass Fraction", color="gray")
ax2.set_xlabel("Test Sample Index")

# Overlay student error as color-coded scatter
ax2b = ax2.twinx()
err_colors = np.where(s_errors > 100, "red", np.where(s_errors > 25, "orange", "green"))
ax2b.scatter(np.arange(len(s_errors)), s_errors, s=15, c=err_colors, alpha=0.5, edgecolors="none")
ax2b.set_ylabel("Student Squared Error (bins²)", color="red")
ax2b.tick_params(axis="y", labelcolor="red")

ax2.set_title(f"CSTR H₂O True Values + Student Error Overlay  |  L={L}  H={H}")
ax2.legend(loc="upper left", fontsize=9)

plt.tight_layout()
out_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cstr_true_vs_error.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.close()

# Stats
print(f"\nPer-sample MSE stats:")
print(f"  Teacher:  mean={np.mean(t_errors):.1f}, median={np.median(t_errors):.1f}, max={np.max(t_errors):.0f}")
print(f"  Baseline: mean={np.mean(b_errors):.1f}, median={np.median(b_errors):.1f}, max={np.max(b_errors):.0f}")
print(f"  Student:  mean={np.mean(s_errors):.1f}, median={np.median(s_errors):.1f}, max={np.max(s_errors):.0f}")
print(f"  Student Δ over Baseline: {(np.mean(b_errors)-np.mean(s_errors))/np.mean(b_errors)*100:+.1f}%")
