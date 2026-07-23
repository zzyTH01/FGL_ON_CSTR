#!/usr/bin/env python
"""
Compare original vs forced CSTR datasets.

Usage:
  uv run python cstr/plot_forced_comparison.py
"""

import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots_forced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ---- Load datasets ----
datasets = {}
for label, path in [
    ("Original (unforced)", "data_h2o.pkl"),
    ("Forced A=0.3",       "data_forced_h2o_A0.3_f0.05.pkl"),
    ("Forced A=0.5",       "data_forced_h2o_A0.5_f0.05.pkl"),
]:
    full = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(full):
        with open(full, "rb") as f:
            data = pickle.load(f)
        datasets[label] = data[:, 0].numpy()
    else:
        print(f"WARNING: {full} not found, skipping")

print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")


def autocorr(series):
    s = series - series.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    return ac


# =====================================================================
# Figure 1: Three-dataset overview (H2O only, 2×3 layout)
# =====================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

colors = ["#2ecc71", "#3498db", "#e74c3c"]

for row, (label, h2o) in enumerate(datasets.items()):
    time = np.arange(len(h2o)) * 0.1
    color = colors[row]

    # Column 1: Full H2O series
    ax = axes[row, 0]
    ax.plot(time, h2o, color=color, linewidth=0.3)
    ax.set_ylabel("H₂O Mass Fraction")
    if row == 0:
        ax.set_title("Full Series")
    ax.grid(True, alpha=0.15)

    # Column 2: Zoom first 120s
    ax = axes[row, 1]
    mask = time <= 120
    ax.plot(time[mask], h2o[mask], color=color, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    if row == 0:
        ax.set_title("First 120s (Zoom)")
    ax.grid(True, alpha=0.15)

    # Column 3: Autocorrelation
    ax = axes[row, 2]
    ac = autocorr(h2o)
    lags = np.arange(len(ac)) * 0.1
    ax.plot(lags[:800], ac[:800], color=color, linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (s)")
    if row == 0:
        ax.set_title("Autocorrelation")
    ax.grid(True, alpha=0.15)

    # Periodicity score
    score = float(ac[20:400].max()) if len(ac) > 20 else 1.0
    ax.annotate(f"Periodicity: {score:.3f}", xy=(0.95, 0.90),
                xycoords="axes fraction", ha="right", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Add dataset label on the left
    axes[row, 0].annotate(label, xy=(-0.12, 0.5), xycoords="axes fraction",
                          ha="left", va="center", fontsize=12, fontweight="bold",
                          rotation=90, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cstr_forced_comparison.png"))
print("Saved: cstr_forced_comparison.png")
plt.close()


# =====================================================================
# Figure 2: Side-by-side H2O first 120s detail
# =====================================================================
fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 4 * len(datasets)),
                          sharex=True)

for row, (label, h2o) in enumerate(datasets.items()):
    time = np.arange(len(h2o)) * 0.1
    color = colors[row]
    ax = axes[row]
    mask = time <= 120
    ax.plot(time[mask], h2o[mask], color=color, linewidth=0.8)
    ax.set_ylabel("H₂O")
    ax.grid(True, alpha=0.15)
    ax.set_title(label, fontweight="bold", color=color)
    ax.set_ylim(-0.02, 1.02)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cstr_forced_zoom.png"))
print("Saved: cstr_forced_zoom.png")
plt.close()


# =====================================================================
# Figure 3: Autocorrelation comparison (overlay)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 5))

for label, h2o in datasets.items():
    ac = autocorr(h2o)
    lags = np.arange(len(ac)) * 0.1
    score = float(ac[20:400].max()) if len(ac) > 20 else 1.0
    ax.plot(lags[:600], ac[:600], linewidth=1.2,
            label=f"{label}  (periodicity={score:.3f})")

# Add Mackey-Glass reference for comparison
mg_path = os.path.join(os.path.dirname(SCRIPT_DIR), "mackey_glass", "data.pkl")
if os.path.exists(mg_path):
    with open(mg_path, "rb") as f:
        mg_data = pickle.load(f)
    mg = mg_data[:, 0].numpy()
    ac_mg = autocorr(mg[:3000])
    lags_mg = np.arange(len(ac_mg)) * 1.0
    score_mg = float(ac_mg[20:200].max()) if len(ac_mg) > 20 else 1.0
    ax.plot(lags_mg[:600], ac_mg[:600], "k--", linewidth=1.5, alpha=0.7,
            label=f"Mackey-Glass (τ=17)  (periodicity={score_mg:.3f})")

ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation Comparison: CSTR variants vs Mackey-Glass", fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, 60)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cstr_autocorr_comparison.png"))
print("Saved: cstr_autocorr_comparison.png")
plt.close()


# =====================================================================
# Figure 4: Summary statistics
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

lines = [
    "CSTR Dataset Comparison — Original vs Forced",
    "=" * 55,
    "",
]
for label, h2o in datasets.items():
    time = np.arange(len(h2o)) * 0.1
    ac = autocorr(h2o)
    score = float(ac[20:400].max()) if len(ac) > 20 else 1.0

    # Count temperature spikes if temp data available
    n = len(h2o)
    lines.extend([
        f"── {label} ──",
        f"  Points:         {n}",
        f"  Time span:      {time[-1]:.0f}s",
        f"  H₂O range:      [{h2o.min():.4f}, {h2o.max():.4f}]",
        f"  H₂O mean ± std: {h2o.mean():.4f} ± {h2o.std():.4f}",
        f"  Periodicity:    {score:.4f}",
        f"  vs unforced:    {(score - 0.9523):+.4f}",
        "",
    ])

# Add FGL results summary
lines.extend([
    "── FGL Results (H=5, α=0.5) ──",
    "  Original:       Δ = -1.2%",
    "  Forced A=0.3:   Δ = +0.2%  (periodicity 0.689)",
    "  Forced A=0.5:   Δ = +0.2%  (periodicity 0.489)",
    "",
    "Conclusion: External forcing reduces periodicity but",
    "FGL still shows zero improvement. Quasi-periodic ≠ chaotic.",
    "FGL requires exponential trajectory divergence (chaos).",
])

ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cstr_forced_summary.png"))
print("Saved: cstr_forced_summary.png")
plt.close()

print("\nAll plots generated.")
