#!/usr/bin/env python
"""
Plot CSTR predictions vs actual for Teacher, Baseline, and Student (FGL) models.
Uses best config: L=20, H=12.
"""
import pickle, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import RNN, create_time_series_dataset, KL

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")

DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data_h2o.pkl")

# Config
L = 20
H = 12
SEED = 0
EPOCHS = 30
ALPHA = 0.5
TEMPERATURE = 4
NUM_BINS = 50
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LR = 1e-4
PATIENCE = 5
VAL_SIZE = 0.2
TEST_SIZE = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load data
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

x_raw = np.array([float(pt[0]) for pt in data])
y_raw = np.array([float(pt[1]) for pt in data])

# Shared bin edges
all_y_windows = []
for i in range(len(x_raw) - L - 1 + 1):
    all_y_windows.append(y_raw[i + L + 1 - 1])
all_y = np.array(all_y_windows)
shared_bin_edges = np.linspace(all_y.min(), all_y.max(), NUM_BINS - 1)

# Create datasets
teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
    data=data, lookback_window=L, forecasting_horizon=1,
    num_bins=NUM_BINS, val_size=VAL_SIZE, test_size=TEST_SIZE,
    offset=H-1, batch_size=BATCH_SIZE, bin_edges=shared_bin_edges,
)
student_train, student_val, student_test, _, orig_test = create_time_series_dataset(
    data=data, lookback_window=L, forecasting_horizon=H,
    num_bins=NUM_BINS, val_size=VAL_SIZE, test_size=TEST_SIZE,
    offset=0, batch_size=BATCH_SIZE, bin_edges=shared_bin_edges,
)

celoss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.counter, self.best_state = float("inf"), 0, None
    def step(self, loss, model):
        if loss + self.min_delta < self.best_loss:
            self.best_loss, self.counter = loss, 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience
    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

def train_model(model, loader, val_loader, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=PATIENCE)
    for epoch in range(epochs):
        model.train()
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            opt.zero_grad()
            celoss(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(celoss(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in val_loader) / len(val_loader)
        if stopper.step(vl, model):
            break
    stopper.restore(model)
    return model

# --- Train Teacher ---
print("Training Teacher...")
teacher = RNN(L, HIDDEN_SIZE, NUM_BINS, NUM_LAYERS).to(device)
teacher = train_model(teacher, teacher_train, teacher_val, EPOCHS, LR)

# --- Train Baseline ---
print("Training Baseline...")
baseline = RNN(L, HIDDEN_SIZE, NUM_BINS, NUM_LAYERS).to(device)
baseline = train_model(baseline, student_train, student_val, EPOCHS, LR)

# --- Train Student (FGL) ---
print("Training Student (FGL)...")
student = RNN(L, HIDDEN_SIZE, NUM_BINS, NUM_LAYERS).to(device)
opt_s = optim.Adam(student.parameters(), lr=LR)
stopper_s = EarlyStopper(patience=PATIENCE)
for epoch in range(EPOCHS):
    student.train()
    for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
        x_s = x_s.float().to(device).view(-1, 1, L)
        targets = y_s.long().to(device)
        outputs = student(x_s)
        x_t = x_t.float().to(device).view(-1, 1, L)
        with torch.no_grad():
            logits = teacher(x_t)
        loss = ALPHA * celoss(outputs, targets) + KL(outputs, logits, TEMPERATURE, ALPHA)
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
    student.eval()
    with torch.no_grad():
        vl = sum(celoss(student(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                 for _, x, y in student_val) / len(student_val)
    if stopper_s.step(vl, student):
        break
stopper_s.restore(student)

# --- Generate predictions on test set ---
def get_predictions(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device).squeeze(-1)
            pred_bin = model(x).argmax(dim=1)
            all_preds.extend(pred_bin.cpu().numpy())
            all_targets.extend(y_int.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)

t_preds, t_targets = get_predictions(teacher, teacher_test)
b_preds, b_targets = get_predictions(baseline, student_test)
s_preds, s_targets = get_predictions(student, student_test)

# Convert bin indices back to original H2O values (bin center)
bin_centers = (shared_bin_edges[:-1] + shared_bin_edges[1:]) / 2
bin_centers = np.concatenate([[shared_bin_edges[0] - (shared_bin_edges[1]-shared_bin_edges[0])/2],
                               bin_centers,
                               [shared_bin_edges[-1] + (shared_bin_edges[1]-shared_bin_edges[0])/2]])

def bin_to_value(bin_idx):
    """Map bin index back to approximate H2O value."""
    idx = np.clip(bin_idx, 0, len(bin_centers)-1)
    return bin_centers[idx]

t_preds_val = bin_to_value(t_preds)
t_targets_val = bin_to_value(t_targets)
b_preds_val = bin_to_value(b_preds)
s_preds_val = bin_to_value(s_preds)
b_targets_val = bin_to_value(b_targets)
s_targets_val = bin_to_value(s_targets)

# Compute metrics
t_mse = np.mean((t_preds - t_targets)**2)
b_mse = np.mean((b_preds - b_targets)**2)
s_mse = np.mean((s_preds - s_targets)**2)
fgl_delta = (b_mse - s_mse) / b_mse * 100

print(f"\nResults:")
print(f"  Teacher  MSE (bin): {t_mse:.1f}")
print(f"  Baseline MSE (bin): {b_mse:.1f}")
print(f"  Student  MSE (bin): {s_mse:.1f}  (Δ={fgl_delta:+.1f}%)")

# ============================================================
# Plotting
# ============================================================
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11})

# Use a consistent subset of test points for visualization
N_PLOT = min(200, len(b_targets))
# Pick the last N_PLOT points from test set for visualization
plot_start = len(b_targets) - N_PLOT

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# --- Row 1: Bin-index space ---
for ax, preds, targets, title, color in [
    (axes[0, 0], t_preds[plot_start:], t_targets[plot_start:], "Teacher (1-step ahead)", "#2196F3"),
    (axes[0, 1], b_preds[plot_start:], b_targets[plot_start:], f"Baseline ({H}-step ahead, no FGL)", "#FF9800"),
    (axes[0, 2], s_preds[plot_start:], s_targets[plot_start:], f"Student FGL ({H}-step ahead, Δ={fgl_delta:+.1f}%)", "#4CAF50"),
]:
    ax.plot(targets, 'o', markersize=3, alpha=0.5, label='Actual', color='gray')
    ax.plot(preds, 's', markersize=3, alpha=0.7, label='Predicted', color=color)
    ax.set_title(title)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Bin index (0–49)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 50)

# --- Row 2: Original H₂O value space ---
for ax, preds, targets, title, color in [
    (axes[1, 0], t_preds_val[plot_start:], t_targets_val[plot_start:], "Teacher (1-step ahead)", "#2196F3"),
    (axes[1, 1], b_preds_val[plot_start:], b_targets_val[plot_start:], f"Baseline ({H}-step ahead, no FGL)", "#FF9800"),
    (axes[1, 2], s_preds_val[plot_start:], s_targets_val[plot_start:], f"Student FGL ({H}-step ahead, Δ={fgl_delta:+.1f}%)", "#4CAF50"),
]:
    ax.plot(targets, 'o', markersize=3, alpha=0.5, label='Actual', color='gray')
    ax.plot(preds, 's', markersize=3, alpha=0.7, label='Predicted', color=color)
    ax.set_title(title)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("H₂O mass fraction")
    ax.legend(loc='upper right', fontsize=9)

fig.suptitle(f"CSTR H₂O Prediction — L={L}, H={H}, α={ALPHA}, T={TEMPERATURE} (seed={SEED})",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(SCRIPT_DIR), "cstr_predictions_L20_H12.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.close()

# ============================================================
# Additional: scatter plot (predicted vs actual)
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, preds, targets, title, color in [
    (axes2[0], b_preds_val[plot_start:], b_targets_val[plot_start:], f"Baseline (no FGL)\nMSE(bin)={b_mse:.1f}", "#FF9800"),
    (axes2[1], s_preds_val[plot_start:], s_targets_val[plot_start:], f"Student FGL\nMSE(bin)={s_mse:.1f}, Δ={fgl_delta:+.1f}%", "#4CAF50"),
    (axes2[2], t_preds_val[plot_start:], t_targets_val[plot_start:], "Teacher (1-step)\nReference", "#2196F3"),
]:
    ax.scatter(targets, preds, alpha=0.5, s=15, color=color, edgecolors='none')
    lims = [0, 1.0]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("Actual H₂O")
    ax.set_ylabel("Predicted H₂O")

fig2.suptitle(f"CSTR H₂O: Predicted vs Actual — L={L}, H={H}", fontsize=13, fontweight='bold')
plt.tight_layout()

out_path2 = os.path.join(os.path.dirname(SCRIPT_DIR), "cstr_scatter_L20_H12.png")
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Scatter plot saved to: {out_path2}")
