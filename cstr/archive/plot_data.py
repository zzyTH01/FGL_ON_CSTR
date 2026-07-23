#!/usr/bin/env python
"""
Generate CSTR dataset visualization plots.

Usage:
  uv run python cstr/plot_data.py
"""

import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots_original")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(SCRIPT_DIR, "data.pkl"), "rb") as f:
    temp_data = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, "data_h2o.pkl"), "rb") as f:
    h2o_data = pickle.load(f)

temp = temp_data[:, 0].numpy()
h2o = h2o_data[:, 0].numpy()
time = np.arange(len(temp)) * 0.1  # dt = 0.1s

# ---- Style ----
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save(path):
    full = os.path.join(OUTPUT_DIR, path)
    plt.savefig(full)
    print(f"Saved: {full}")
    plt.close()


# =====================================================================
# Figure 1: Full series overview (Temperature + H2O, side by side)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax1.plot(time, temp, color="#e74c3c", linewidth=0.4)
ax1.set_ylabel("Temperature (K)")
ax1.set_title("CSTR Temperature — Full 300s (H₂/O₂ combustion)")
ax1.grid(True, alpha=0.2)
ax1.set_ylim(700, 2100)

ax2.plot(time, h2o, color="#3498db", linewidth=0.4)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("H₂O Mass Fraction")
ax2.set_title("CSTR H₂O Mass Fraction — Full 300s")
ax2.grid(True, alpha=0.2)
ax2.set_ylim(-0.02, 1.02)

plt.tight_layout()
save("cstr_full_series.png")


# =====================================================================
# Figure 2: Zoomed view — first 60 seconds (Temperature + H2O overlay)
# =====================================================================
fig, ax1 = plt.subplots(figsize=(16, 6))

mask = time <= 60
t_zoom = time[mask]

color_t = "#e74c3c"
color_h = "#3498db"

ax1.plot(t_zoom, temp[mask], color=color_t, linewidth=1.0, label="Temperature (K)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Temperature (K)", color=color_t)
ax1.tick_params(axis="y", labelcolor=color_t)
ax1.set_ylim(700, 2100)

ax2 = ax1.twinx()
ax2.plot(t_zoom, h2o[mask], color=color_h, linewidth=1.0, label="H₂O Mass Fraction")
ax2.set_ylabel("H₂O Mass Fraction", color=color_h)
ax2.tick_params(axis="y", labelcolor=color_h)
ax2.set_ylim(-0.02, 1.02)

# Mark ignition events (T > 1000K)
spike_mask = temp[mask] > 1000
spike_times = t_zoom[spike_mask]
for st in spike_times:
    ax1.axvline(x=st, color="#e74c3c", linestyle="--", alpha=0.4, linewidth=0.8)

# Mark weak ignition events (800 < T <= 1000)
weak_mask = (temp[mask] > 800) & (temp[mask] <= 1000)
weak_times = t_zoom[weak_mask]
for wt in weak_times:
    ax1.axvline(x=wt, color="orange", linestyle=":", alpha=0.4, linewidth=0.8)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color_t, lw=2, label="Temperature"),
    Line2D([0], [0], color=color_h, lw=2, label="H₂O Mass Fraction"),
    Line2D([0], [0], color="#e74c3c", linestyle="--", lw=1, label="Strong ignition (T>1000K)"),
    Line2D([0], [0], color="orange", linestyle=":", lw=1, label="Weak ignition (800<T≤1000K)"),
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.title("CSTR — First 60s: Alternating Weak/Strong Ignitions", fontweight="bold")
plt.tight_layout()
save("cstr_zoom_60s.png")


# =====================================================================
# Figure 3: Single oscillation sub-cycle detail (H2O sawtooth)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 5))

mask = (time >= 100) & (time <= 130)
t_win = time[mask]
h_win = h2o[mask]
t_win_t = temp[mask]

ax.plot(t_win, h_win, "o-", color="#3498db", linewidth=1.5, markersize=4, label="H₂O Mass Fraction")
ax.set_xlabel("Time (s)")
ax.set_ylabel("H₂O Mass Fraction", color="#3498db")
ax.tick_params(axis="y", labelcolor="#3498db")
ax.set_ylim(-0.02, 0.75)
ax.grid(True, alpha=0.2)

# Mark H2O peak points
peaks, _ = signal.find_peaks(h_win, height=0.3, distance=4)
ax.scatter(t_win[peaks], h_win[peaks], color="#e74c3c", s=80, zorder=5,
           label=f"Sub-oscillation peaks ({len(peaks)} in 30s)")
for p in peaks:
    ax.annotate(f"{h_win[p]:.3f}", (t_win[p], h_win[p]),
                textcoords="offset points", xytext=(0, 10), fontsize=9, ha="center")

# Twin axis for temperature
ax2 = ax.twinx()
ax2.plot(t_win, t_win_t, color="#e74c3c", linewidth=1.0, alpha=0.5, label="Temperature")
ax2.set_ylabel("Temperature (K)", color="#e74c3c")
ax2.tick_params(axis="y", labelcolor="#e74c3c")
ax2.set_ylim(760, 780)

ax.set_title("CSTR H₂O Sub-Oscillation Detail (t=100–130s): Sawtooth Pattern with Growing Envelope",
             fontweight="bold")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

plt.tight_layout()
save("cstr_h2o_sawtooth.png")


# =====================================================================
# Figure 4: Temperature spike characterization
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 4a: Temperature histogram
ax = axes[0]
ax.hist(temp, bins=80, color="#e74c3c", edgecolor="white", alpha=0.8)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Count")
ax.set_title("Temperature Distribution (note log scale)")
ax.set_yscale("log")
# Annotate
ax.annotate(f"98.6% of values\nare at baseline\n(770–780 K)",
            xy=(780, 100), fontsize=10, color="#c0392b",
            bbox=dict(boxstyle="round", facecolor="#fadbd8", alpha=0.8))

# 4b: Spike profile (all 21 strong ignition spikes overlaid)
ax = axes[1]
spike_mask = temp > 1000
spike_diff = np.diff(np.concatenate([[0], spike_mask.astype(int), [0]]))
spike_starts = np.where(spike_diff == 1)[0]
spike_ends = np.where(spike_diff == -1)[0]

colors = plt.cm.inferno(np.linspace(0.2, 0.9, len(spike_starts)))
for j, (s, e) in enumerate(zip(spike_starts, spike_ends)):
    # Extract ±5 steps around each spike
    win_start = max(0, s - 3)
    win_end = min(len(temp), e + 3)
    x_rel = (np.arange(win_start, win_end) - s) * 0.1  # time relative to spike start
    ax.plot(x_rel, temp[win_start:win_end], "o-", color=colors[j],
            markersize=3, linewidth=0.8, alpha=0.7)

ax.set_xlabel("Time relative to ignition (s)")
ax.set_ylabel("Temperature (K)")
ax.set_title("All 21 Strong Ignition Spikes Overlaid")
ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.2)

plt.tight_layout()
save("cstr_temperature_analysis.png")


# =====================================================================
# Figure 5: Summary dashboard — 2×3 grid
# =====================================================================
fig = plt.figure(figsize=(18, 12))

# 5a: Full temperature
ax = fig.add_subplot(2, 3, 1)
ax.plot(time, temp, color="#e74c3c", linewidth=0.3)
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature (300s)")
ax.grid(True, alpha=0.15)

# 5b: Full H2O
ax = fig.add_subplot(2, 3, 2)
ax.plot(time, h2o, color="#3498db", linewidth=0.3)
ax.set_ylabel("H₂O Mass Fraction")
ax.set_title("H₂O Mass Fraction (300s)")
ax.grid(True, alpha=0.15)

# 5c: Zoom 60s overlay
ax = fig.add_subplot(2, 3, 3)
mask = time <= 60
ax.plot(time[mask], h2o[mask], color="#3498db", linewidth=1.0, label="H₂O")
ax2 = ax.twinx()
ax2.plot(time[mask], temp[mask], color="#e74c3c", linewidth=0.7, alpha=0.6, label="Temp")
ax.set_ylabel("H₂O", color="#3498db")
ax2.set_ylabel("Temp (K)", color="#e74c3c")
ax.set_title("First 60s: Alternating Ignitions")
ax.grid(True, alpha=0.15)

# 5d: Temperature distribution
ax = fig.add_subplot(2, 3, 4)
ax.hist(temp, bins=100, color="#e74c3c", edgecolor="white", alpha=0.8)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Count (log)")
ax.set_yscale("log")
ax.set_title("Temperature Histogram")

# 5e: H2O distribution
ax = fig.add_subplot(2, 3, 5)
ax.hist(h2o, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
ax.set_xlabel("H₂O Mass Fraction")
ax.set_ylabel("Count")
ax.set_title("H₂O Distribution")
ax.annotate(f"Zero-heavy:\n{h2o[h2o<0.01].shape[0]} pts < 0.01",
            xy=(0.05, 0.75), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#d5f5e3", alpha=0.8))

# 5f: Statistics text panel
ax = fig.add_subplot(2, 3, 6)
ax.axis("off")

spike_count = len(spike_starts)
strong_peaks = spike_count
h2o_peaks, _ = signal.find_peaks(h2o, height=0.5, distance=20)
cycle_steps = 71.5
L, H = 8, 5

stats = (
    "CSTR Dataset Statistics\n"
    "═══════════════════════\n\n"
    f"Data points:           3,001\n"
    f"Time span:             300 s (dt=0.1s)\n\n"
    f"── Temperature ──\n"
    f"Range:                 770 – 2,018 K\n"
    f"Mean ± Std:            777.0 ± 76.8 K\n"
    f"Baseline (<800K):      98.6% of points\n"
    f"Strong ignitions:      {strong_peaks} (T>1000K)\n"
    f"Spike duration:        0.1s (single point)\n\n"
    f"── H₂O ──\n"
    f"Range:                 0.000 – 0.964\n"
    f"Mean ± Std:            0.180 ± 0.223\n"
    f"Values <0.01:          8.5% of points\n"
    f"Values >0.5:           10.5% of points\n\n"
    f"── Oscillation ──\n"
    f"Sub-cycle period:      ~7.15s (72 steps)\n"
    f"Full cycle (weak+strong): ~14.3s (143 steps)\n"
    f"H₂O sub-oscillations:  ~42 peaks\n\n"
    f"── FGL Window (L={L}, H={H}) ──\n"
    f"History window:        0.8s (5.6% of full cycle)\n"
    f"Prediction horizon:    0.5s (3.5% of full cycle)\n"
    f"Total span:            1.3s (9.1% of full cycle)\n"
)
ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

plt.tight_layout()
save("cstr_dashboard.png")

print("\nAll plots generated successfully.")
