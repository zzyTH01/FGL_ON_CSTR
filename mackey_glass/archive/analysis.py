#!/usr/bin/env python
import sys
import os
import json
import pickle
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils.utils import RNN, create_time_series_dataset, KL

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.pkl")


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def run_single_experiment(
    student_horizon,
    alpha,
    num_bins=50,
    val_size=0.2,
    test_size=0.2,
    epochs=50,
    temperature=4,
    lookback_window=8,
    batch_size=128,
    patience=5,
):
    torch.manual_seed(42)
    hidden_size = 128
    output_size = num_bins
    num_layers = 2
    lr = 1e-4

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=1,
        num_bins=num_bins,
        val_size=val_size,
        test_size=test_size,
        offset=student_horizon - 1,
        batch_size=batch_size,
    )
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        val_size=val_size,
        test_size=test_size,
        offset=0,
        batch_size=batch_size,
    )

    mse = torch.nn.MSELoss()
    celoss = torch.nn.CrossEntropyLoss()

    teacher = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y = y.long().to(device)
            opt_t.zero_grad()
            celoss(teacher(x), y).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(teacher(x.float().to(device).view(-1, 1, lookback_window)), y.long().to(device)).item()
                for _, x, y in teacher_val
            ) / len(teacher_val)
        if stop_t.step(val_loss, teacher):
            break
    stop_t.restore(teacher)

    baseline = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y = y.long().to(device)
            opt_b.zero_grad()
            celoss(baseline(x), y).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(baseline(x.float().to(device).view(-1, 1, lookback_window)), y.long().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_b.step(val_loss, baseline):
            break
    stop_b.restore(baseline)

    student = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        student.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, lookback_window)
            targets = y_s.long().to(device)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, lookback_window)
            with torch.no_grad():
                logits = teacher(x_t)
            loss = alpha * celoss(outputs, targets) + KL(outputs, logits, temperature, alpha)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(student(x.float().to(device).view(-1, 1, lookback_window)), y.long().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_s.step(val_loss, student):
            break
    stop_s.restore(student)

    def evaluate(model, loader):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, lookback_window)
                tot += mse(model(x).argmax(dim=1).float(), y.float().to(device).squeeze(-1)).item()
        return tot / len(loader)

    t_mse = evaluate(teacher, teacher_test)
    b_mse = evaluate(baseline, student_test)
    s_mse = evaluate(student, student_test)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0

    print(f"  H={student_horizon:2d} α={alpha:.2f} T={temperature:.1f} | "
          f"Teacher:{t_mse:.4f} Baseline:{b_mse:.4f} Student:{s_mse:.4f} | Δ={improvement:+.1f}%")
    return {"teacher": t_mse, "baseline": b_mse, "student": s_mse, "improvement": improvement}


def sweep_horizons(
    horizons=None,
    alpha=0.5,
    temperature=4,
    num_bins=50,
    epochs=30,
    lookback_window=8,
    batch_size=128,
):
    if horizons is None:
        horizons = list(range(2, 31))
    results = []
    print(f"\n=== Horizon Sweep: α={alpha}, T={temperature} ===")
    for h in horizons:
        r = run_single_experiment(
            student_horizon=h,
            alpha=alpha,
            num_bins=num_bins,
            epochs=epochs,
            temperature=temperature,
            lookback_window=lookback_window,
            batch_size=batch_size,
        )
        r["horizon"] = h
        results.append(r)
    return results


def sweep_alpha(
    horizons=None,
    alphas=None,
    temperature=4,
    num_bins=50,
    epochs=30,
    lookback_window=8,
    batch_size=128,
):
    if horizons is None:
        horizons = [5, 10, 15]
    if alphas is None:
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = []
    print(f"\n=== Alpha Sweep: T={temperature} ===")
    for h in horizons:
        for a in alphas:
            r = run_single_experiment(
                student_horizon=h,
                alpha=a,
                num_bins=num_bins,
                epochs=epochs,
                temperature=temperature,
                lookback_window=lookback_window,
                batch_size=batch_size,
            )
            r["horizon"] = h
            r["alpha"] = a
            results.append(r)
    return results


def sweep_temperature(
    horizons=None,
    alpha=0.5,
    temperatures=None,
    num_bins=50,
    epochs=30,
    lookback_window=8,
    batch_size=128,
):
    if horizons is None:
        horizons = [5, 10, 15]
    if temperatures is None:
        temperatures = [1, 2, 3, 4, 5, 6, 8, 10]
    results = []
    print(f"\n=== Temperature Sweep: α={alpha} ===")
    for h in horizons:
        for t in temperatures:
            r = run_single_experiment(
                student_horizon=h,
                alpha=alpha,
                num_bins=num_bins,
                epochs=epochs,
                temperature=t,
                lookback_window=lookback_window,
                batch_size=batch_size,
            )
            r["horizon"] = h
            r["temperature"] = t
            results.append(r)
    return results


def save_results(results, name):
    path = os.path.join(OUTPUT_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return path


def plot_horizon_curves(results, save_name="horizon_curves"):
    horizons = [r["horizon"] for r in results]
    baseline = [r["baseline"] for r in results]
    student = [r["student"] for r in results]
    improvement = [r["improvement"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(horizons, baseline, "o-", color="#e74c3c", linewidth=2, markersize=6, label="Baseline (no FGL)")
    ax1.plot(horizons, student, "s-", color="#2ecc71", linewidth=2, markersize=6, label="Student (FGL)")
    ax1.set_xlabel("Forecasting Horizon H", fontsize=13)
    ax1.set_ylabel("Test MSE", fontsize=13)
    ax1.set_title("MSE vs Horizon", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in improvement]
    ax2.bar(horizons, improvement, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Forecasting Horizon H", fontsize=13)
    ax2.set_ylabel("FGL Improvement (%)", fontsize=13)
    ax2.set_title("FGL Improvement over Baseline", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_name}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def plot_alpha_curves(results, save_name="alpha_curves"):
    horizons_set = sorted(set(r["horizon"] for r in results))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(horizons_set)))
    for h, c in zip(horizons_set, colors):
        pts = [r for r in results if r["horizon"] == h]
        pts.sort(key=lambda x: x["alpha"])
        alphas = [p["alpha"] for p in pts]
        imp = [p["improvement"] for p in pts]
        ax1.plot(alphas, imp, "o-", color=c, linewidth=2, markersize=6, label=f"H={h}")

    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Alpha (α)", fontsize=13)
    ax1.set_ylabel("FGL Improvement (%)", fontsize=13)
    ax1.set_title("Alpha Sensitivity", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for h, c in zip(horizons_set, colors):
        pts = [r for r in results if r["horizon"] == h]
        pts.sort(key=lambda x: x["alpha"])
        alphas = [p["alpha"] for p in pts]
        stu = [p["student"] for p in pts]
        ax2.plot(alphas, stu, "s-", color=c, linewidth=2, markersize=6, label=f"H={h}")

    ax2.set_xlabel("Alpha (α)", fontsize=13)
    ax2.set_ylabel("Student Test MSE", fontsize=13)
    ax2.set_title("Student MSE vs Alpha", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_name}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def plot_temperature_curves(results, save_name="temperature_curves"):
    horizons_set = sorted(set(r["horizon"] for r in results))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(horizons_set)))
    for h, c in zip(horizons_set, colors):
        pts = [r for r in results if r["horizon"] == h]
        pts.sort(key=lambda x: x["temperature"])
        temps = [p["temperature"] for p in pts]
        imp = [p["improvement"] for p in pts]
        ax1.plot(temps, imp, "o-", color=c, linewidth=2, markersize=6, label=f"H={h}")

    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Temperature (T)", fontsize=13)
    ax1.set_ylabel("FGL Improvement (%)", fontsize=13)
    ax1.set_title("Temperature Sensitivity", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for h, c in zip(horizons_set, colors):
        pts = [r for r in results if r["horizon"] == h]
        pts.sort(key=lambda x: x["temperature"])
        temps = [p["temperature"] for p in pts]
        stu = [p["student"] for p in pts]
        ax2.plot(temps, stu, "s-", color=c, linewidth=2, markersize=6, label=f"H={h}")

    ax2.set_xlabel("Temperature (T)", fontsize=13)
    ax2.set_ylabel("Student Test MSE", fontsize=13)
    ax2.set_title("Student MSE vs Temperature", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_name}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def plot_alpha_temp_heatmap(results, horizon, save_name="alpha_temp_heatmap"):
    pts = [r for r in results if r["horizon"] == horizon]
    if not pts:
        print(f"No data for horizon {horizon}")
        return
    alphas = sorted(set(p["alpha"] for p in pts))
    temps = sorted(set(p["temperature"] for p in pts))
    grid = np.zeros((len(temps), len(alphas)))
    for p in pts:
        i = temps.index(p["temperature"])
        j = alphas.index(p["alpha"])
        grid[i, j] = p["improvement"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=-10, vmax=30)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(temps)))
    ax.set_yticklabels([f"{t:.0f}" for t in temps])
    ax.set_xlabel("Alpha (α)", fontsize=13)
    ax.set_ylabel("Temperature (T)", fontsize=13)
    ax.set_title(f"FGL Improvement (%) — H={horizon}", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Improvement (%)", fontsize=12)
    for i in range(len(temps)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(grid[i, j]) > 15 else "black")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_name}_H{horizon}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def plot_summary_dashboard(horizon_results, alpha_results=None, temp_results=None, save_name="summary_dashboard"):
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    horizons = [r["horizon"] for r in horizon_results]
    baseline = [r["baseline"] for r in horizon_results]
    student = [r["student"] for r in horizon_results]
    improvement = [r["improvement"] for r in horizon_results]

    ax1.plot(horizons, baseline, "o-", color="#e74c3c", linewidth=2, markersize=5, label="Baseline")
    ax1.plot(horizons, student, "s-", color="#2ecc71", linewidth=2, markersize=5, label="Student (FGL)")
    ax1.set_xlabel("Horizon H", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_title("MSE vs Horizon", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in improvement]
    ax2.bar(horizons, improvement, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    avg_imp = np.mean(improvement)
    ax2.axhline(y=avg_imp, color="blue", linewidth=1.5, linestyle="--",
                label=f"Avg: {avg_imp:+.1f}%")
    ax2.set_xlabel("Horizon H", fontsize=12)
    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_title("FGL Improvement", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = fig.add_subplot(2, 2, 3)
    teacher = [r["teacher"] for r in horizon_results]
    ax3.plot(horizons, teacher, "D-", color="#3498db", linewidth=2, markersize=5, label="Teacher (H=1)")
    ax3.set_xlabel("Horizon H", fontsize=12)
    ax3.set_ylabel("MSE", fontsize=12)
    ax3.set_title("Teacher Stability (always predicts 1 step)", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    summary_text = (
        f"Experiment Summary\n"
        f"{'='*35}\n"
        f"Total horizons tested: {len(horizon_results)}\n"
        f"FGL wins: {sum(1 for v in improvement if v > 0)}/{len(improvement)}\n"
        f"Average improvement: {avg_imp:+.1f}%\n"
        f"Max improvement: {max(improvement):+.1f}%\n"
        f"Min improvement: {min(improvement):+.1f}%\n"
        f"Best horizon: H={horizons[improvement.index(max(improvement))]}\n"
        f"Worst horizon: H={horizons[improvement.index(min(improvement))]}\n"
        f"\nBaseline MSE range: {min(baseline):.2f} – {max(baseline):.2f}\n"
        f"Student MSE range:  {min(student):.2f} – {max(student):.2f}\n"
    )
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_name}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mackey-Glass FGL Analysis Suite")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["horizon", "alpha", "temperature", "alphatemp", "all", "plot"],
                        help="Experiment mode")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30, reduce for quick test)")
    parser.add_argument("--horizon_range", type=str, default="2,31",
                        help="Horizon range as 'start,end' (end exclusive)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if JSON results file already exists")
    args = parser.parse_args()

    all_results = {}

    if args.mode in ("horizon", "all"):
        start, end = map(int, args.horizon_range.split(","))
        horizons = list(range(start, end))
        results = sweep_horizons(horizons=horizons, epochs=args.epochs)
        all_results["horizon"] = results
        save_results(results, "horizon_sweep")
        plot_horizon_curves(results)
        plot_summary_dashboard(results, save_name="horizon_dashboard")

    if args.mode in ("alpha", "all"):
        results = sweep_alpha(horizons=[5, 10, 15], epochs=args.epochs)
        all_results["alpha"] = results
        save_results(results, "alpha_sweep")
        plot_alpha_curves(results)

    if args.mode in ("temperature", "all"):
        results = sweep_temperature(horizons=[5, 10, 15], epochs=args.epochs)
        all_results["temperature"] = results
        save_results(results, "temperature_sweep")
        plot_temperature_curves(results)

    if args.mode in ("alphatemp", "all"):
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        temps = [1, 2, 3, 4, 5, 7, 10]
        results = []
        print(f"\n=== Alpha × Temperature Grid Search (H=5,10,15) ===")
        for h in [5, 10, 15]:
            for a in alphas:
                for t in temps:
                    r = run_single_experiment(
                        student_horizon=h, alpha=a, temperature=t,
                        epochs=args.epochs,
                    )
                    r["horizon"] = h
                    r["alpha"] = a
                    r["temperature"] = t
                    results.append(r)
        all_results["alphatemp"] = results
        save_results(results, "alphatemp_sweep")
        for h in [5, 10, 15]:
            plot_alpha_temp_heatmap(results, horizon=h)

    if args.mode == "plot":
        json_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")])
        if not json_files:
            print("No JSON results found in analysis_output/. Run experiments first.")
            return
        print(f"Found result files: {json_files}")
        target = json_files[-1]
        with open(os.path.join(OUTPUT_DIR, target)) as f:
            data = json.load(f)
        mode_key = "horizon"
        if any("alpha" in target for target in json_files[-1:]):
            mode_key = "alpha" if "alpha_sweep" in target else "temperature" if "temp" in target else "alphatemp"
        print(f"Using {target}, detected mode: {mode_key}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
