#!/usr/bin/env python
"""
Plot Mackey-Glass datasets for different τ values,
in the same visual style as the CSTR plots for direct comparison.

Usage:
  cd /tmp && source .venv/bin/activate && python mackey_glass/exp/plot_mg_data.py
"""

import sys
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MG_DIR)

from utils.utils import MackeyGlass

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "mg_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAU_VALUES = [10, 13, 17, 23, 30]
N_POINTS = 3000  # match CSTR length for fair visual comparison


def generate_mg(tau, n_points=N_POINTS):
    mg = MackeyGlass(
        tau=tau, constant_past=0.9,
        nmg=10, beta=0.2, gamma=0.1,
        dt=1.0, splits=(float(n_points), 0.), seed_id=42,
    )
    vals = []
    for idx in range(len(mg)):
        _, target = mg[idx]
        vals.append(target.squeeze().item())
    return np.array(vals), mg.lyap_exp


def autocorr(series):
    s = series - series.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    return ac


# ---- Pre-generate all data ----
print("Generating Mackey-Glass data...")
datasets = {}
for tau in TAU_VALUES:
    data, lyap = generate_mg(tau)
    ac = autocorr(data)
    score = float(ac[20:200].max()) if len(ac) > 200 else 1.0
    ac_zero = next((i for i in range(1, len(ac)) if ac[i] < 0), len(ac))
    datasets[tau] = {
        "data": data, "lyap": lyap, "ac": ac,
        "periodicity": score, "ac_zero": ac_zero,
    }
    print(f"  τ={tau}: lyap={lyap:+.6f}  periodicity={score:.4f}  AC zero at lag {ac_zero}")


# =====================================================================
# Figure 1: 5×3 overview (matching CSTR's forced_comparison layout)
# =====================================================================
fig, axes = plt.subplots(len(TAU_VALUES), 3, figsize=(18, 16))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(TAU_VALUES)))

for row, tau in enumerate(TAU_VALUES):
    d = datasets[tau]
    data = d["data"]
    time = np.arange(len(data)) * 1.0  # dt=1 for MG
    color = colors[row]

    # Col 1: Full series
    ax = axes[row, 0]
    ax.plot(time, data, color=color, linewidth=0.3)
    ax.set_ylabel("x(t)")
    if row == 0:
        ax.set_title(f"Full Series ({N_POINTS} steps)")
    ax.grid(True, alpha=0.15)

    # Col 2: Zoom first 200 steps
    ax = axes[row, 1]
    mask = time <= 200
    ax.plot(time[mask], data[mask], color=color, linewidth=0.8)
    if row == 0:
        ax.set_title("First 200 steps (Zoom)")
    ax.grid(True, alpha=0.15)

    # Col 3: Autocorrelation
    ax = axes[row, 2]
    ac = d["ac"]
    lags = np.arange(len(ac))
    ax.plot(lags[:200], ac[:200], color=color, linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(x=d["ac_zero"], color="red", linewidth=0.5, linestyle=":", alpha=0.5)
    if row == 0:
        ax.set_title("Autocorrelation")
    ax.grid(True, alpha=0.15)
    ax.annotate(f"P={d['periodicity']:.3f}  λ={d['lyap']:+.5f}",
                xy=(0.95, 0.90), xycoords="axes fraction",
                ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Row label
    label = f"τ={tau}"
    if tau == 10:
        label += " (limit cycle)"
    elif tau == 13:
        label += " (period-2)"
    elif tau == 17:
        label += " (chaos onset)"
    elif tau == 23:
        label += " (moderate chaos)"
    elif tau == 30:
        label += " (high-D chaos)"
    axes[row, 0].annotate(label, xy=(-0.13, 0.5), xycoords="axes fraction",
                          ha="left", va="center", fontsize=11, fontweight="bold",
                          rotation=90, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mg_tau_overview.png"))
print("Saved: mg_tau_overview.png")
plt.close()


# =====================================================================
# Figure 2: Autocorrelation overlay (matching CSTR's autocorr_comparison)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 5))

for tau in TAU_VALUES:
    d = datasets[tau]
    ac = d["ac"]
    lags = np.arange(len(ac))
    ax.plot(lags[:400], ac[:400], linewidth=1.2,
            label=f"τ={tau}  (λ={d['lyap']:+.5f}, P={d['periodicity']:.3f})")

ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xlabel("Lag (steps)", fontsize=12)
ax.set_ylabel("Autocorrelation", fontsize=12)
ax.set_title("Mackey-Glass Autocorrelation by τ", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, 200)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mg_autocorr_overlay.png"))
print("Saved: mg_autocorr_overlay.png")
plt.close()


# =====================================================================
# Figure 3: CSTR vs MG side-by-side comparison
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Load CSTR H2O data
cstr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))),
                         "cstr", "data_h2o.pkl")
if os.path.exists(cstr_path):
    with open(cstr_path, "rb") as f:
        cstr_data = pickle.load(f)
    cstr_h2o = cstr_data[:, 0].numpy()
    cstr_time = np.arange(len(cstr_h2o)) * 0.1

    ax = axes[0]
    mask = cstr_time <= 60
    ax.plot(cstr_time[mask], cstr_h2o[mask], color="#3498db", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H₂O Mass Fraction")
    ax.set_title("CSTR (period-1, P=0.952)", fontweight="bold", color="#3498db")
    ax.grid(True, alpha=0.15)
    ax.set_ylim(-0.02, 1.02)

ax = axes[1]
mg_data = datasets[17]["data"]
mg_time = np.arange(len(mg_data)) * 1.0
mask = mg_time <= 200
ax.plot(mg_time[mask], mg_data[mask], color="#e74c3c", linewidth=0.8)
ax.set_xlabel("Time (steps, dt=1)")
ax.set_ylabel("x(t)")
ax.set_title("Mackey-Glass τ=17 (chaos, P=0.817, λ>0)", fontweight="bold", color="#e74c3c")
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cstr_vs_mg.png"))
print("Saved: cstr_vs_mg.png")
plt.close()


# =====================================================================
# Figure 4: Phase portraits (x(t) vs x(t-τ)) for each τ
# =====================================================================
fig, axes = plt.subplots(1, len(TAU_VALUES), figsize=(20, 4))

for i, tau in enumerate(TAU_VALUES):
    data = datasets[tau]["data"]
    ax = axes[i]
    # Skip initial transient
    start = 500
    delay = int(tau)
    ax.plot(data[start:start+2000], data[start+delay:start+2000+delay],
            color=colors[i], linewidth=0.3, alpha=0.8)
    ax.set_xlabel("x(t)")
    if i == 0:
        ax.set_ylabel("x(t-τ)")
    ax.set_title(f"τ={tau}\nλ={datasets[tau]['lyap']:+.5f}", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

plt.suptitle("Mackey-Glass Phase Portraits: x(t) vs x(t-τ)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mg_phase_portraits.png"))
print("Saved: mg_phase_portraits.png")
plt.close()


# =====================================================================
# Figure 5: Summary statistics panel
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 9))
ax.axis("off")

lines = [
    "Mackey-Glass Dataset Overview — τ Parameter Sweep",
    "=" * 55,
    "",
    "Equation:  dx/dt = β·x(t-τ) / (1 + x(t-τ)^n) - γ·x(t)",
    "           β=0.2, γ=0.1, n=10, N=3000",
    "",
    f"{'τ':>6s}  {'Lyapunov':>10s}  {'Periodicity':>12s}  {'AC ZeroLag':>10s}  "
    f"{'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'State':>20s}",
    f"{'─'*6}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}",
]

for tau in TAU_VALUES:
    d = datasets[tau]
    data = d["data"]
    if tau == 10:
        state = "Limit cycle (period-1)"
    elif tau == 13:
        state = "Period-doubling (period-2)"
    elif tau == 17:
        state = "Chaos onset"
    elif tau == 23:
        state = "Moderate chaos"
    elif tau == 30:
        state = "High-dimensional chaos"
    lines.append(
        f"  {tau:4.0f}  {d['lyap']:+10.6f}  {d['periodicity']:12.4f}  "
        f"{d['ac_zero']:10d}  {data.mean():8.4f}  {data.std():8.4f}  "
        f"{data.min():8.4f}  {data.max():8.4f}  {state:>20s}"
    )

lines.extend([
    "",
    "── Key Differences vs CSTR ──",
    "",
    "CSTR (H₂O):    Period-1 limit cycle, periodicity=0.952",
    "                Autocorrelation peaks at 71, 143 steps (r≈0.95)",
    "                Perfectly predictable → FGL Δ≈0%",
    "",
    "MG τ=10:        Also period-1, periodicity=0.996",
    "                Similar to CSTR — simple oscillation, FGL barely helps",
    "",
    "MG τ=13:        Period-doubling, still Lyapunov≈0",
    "                Alternating pattern confuses baseline but not teacher",
    "                → FGL Δ=+79.1%  (BEST result, NOT in chaos!)",
    "",
    "MG τ=17:        Chaos onset, Lyapunov>0,",
    "                AC decays to zero at lag 13",
    "                → FGL Δ=+11.4%  (chaos mechanism)",
    "",
    "MG τ=30:        High-D chaos,",
    "                AC decays fastest (zero at lag 21 due to longer memory)",
    "                → FGL Δ=+8.8%  (consistent positive)",
])

ax.text(0.03, 0.98, "\n".join(lines), transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mg_summary.png"))
print("Saved: mg_summary.png")
plt.close()

print(f"\nAll plots saved to {OUTPUT_DIR}")
