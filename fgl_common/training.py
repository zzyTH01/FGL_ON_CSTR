"""FGL 共享训练基础设施。

收敛了原本在 ~18 个脚本里各自重复定义的:
  - ``device``                  —— 设备选择(CUDA > MPS > CPU)
  - ``EarlyStopper``            —— 早停(带 best-state 恢复)
  - ``evaluate`` / ``evaluate_with_ph`` / ``evaluate_regression`` / ``evaluate_seq``
  - ``page_hinkley_update``     —— 漂移检测
  - ``compute_shared_bin_edges``—— teacher/student 共享离散化边界

统一的实验入口(均与原 ``cstr/exp/*.py`` 数值等价,仅代码搬迁):
  - ``run_fgl_experiment``       —— 三阶段(teacher→baseline→student)。覆盖 baseline /
                                     lstm / regression / drift,差异收敛为 ``model_fn`` /
                                     ``regression`` / ``use_ph`` 参数。
  - ``run_adaptive_weight``      —— 自适应蒸馏权重 A/B/C/D(teacher−student MSE 差距,独立流程)。
  - ``run_adaptive_inference``   —— 推理时 teacher-student 融合(独立流程)。
  - ``run_seq2seq``              —— 多步序列 FGL(独立流程)。
"""
import os
from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import RNN, LSTMModel, RNNRegression, SeqRNN
from .data import create_time_series_dataset, create_seq_dataset
from .distillation import KL, KL_weighted, seq_KL, compute_weights


# ==================== Device ====================
def _select_device() -> torch.device:
    """CUDA > MPS > CPU 自动检测。

    可用环境变量 ``FGL_DEVICE`` 强制覆盖(cpu / mps / cuda),用于在 MPS 上
    遇到不支持的 op 时回退 CPU 等场景;指定不可用的设备会显式报错而非静默回退。
    """
    env = os.environ.get("FGL_DEVICE", "").strip().lower()
    if env:
        if env not in ("cpu", "mps", "cuda"):
            raise ValueError(f"FGL_DEVICE={env!r} 不合法,应为 cpu / mps / cuda 之一")
        if env == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FGL_DEVICE=cuda 但未检测到可用 CUDA")
        if env == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("FGL_DEVICE=mps 但未检测到可用 MPS")
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _select_device()
print(f"[fgl] device = {device}")  # 换机器运行时一眼看到落到了哪


# ==================== Early Stopping ====================
class EarlyStopper:
    """Patience-based early stopping; restores the best model state on ``restore()``."""

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
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ==================== Page-Hinkley drift detection ====================
def page_hinkley_update(error, state, delta=0.005):
    state["t"] += 1
    m_prev = state["m"]
    state["m"] += (error - state["m"]) / state["t"]
    state["PH"] = max(0.0, state["PH"] + (error - m_prev - delta))
    return state


# ==================== Shared bin edges ====================
def compute_shared_bin_edges(data, lookback_window, num_bins):
    """Bin edges computed from *all* H=1 targets (widest coverage), so teacher and
    student share the same discretization. Matches ``fgl_cstr.py`` /
    ``adaptive_weight_exp.py``. Returns ``(bin_edges, y_min, y_max)``."""
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    L = lookback_window
    all_y = np.array([y_raw[i + L] for i in range(len(x_raw) - L - 1 + 1)])
    return np.linspace(all_y.min(), all_y.max(), num_bins - 1), all_y.min(), all_y.max()


# ==================== Evaluation ====================
def evaluate(model, loader, lookback_window):
    """Classification MSE: predicted bin index vs true bin index."""
    mse_loss = nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y_int = y.long().to(device).squeeze(-1)
            pred = model(x).argmax(dim=1).float()
            total += mse_loss(pred, y_int.float()).item()
    return total / len(loader)


def evaluate_with_ph(model, loader, use_ph=False,
                     delta=0.005, lambda_thr=1.0,
                     window_size=50, retrain_epochs=3,
                     lr=1e-4, lookback_window=8):
    """Classification evaluation; with Page-Hinkley drift retrain when ``use_ph``.
    Collected from ``cstr/exp/fgl_cstr.py``."""
    if not use_ph:
        return evaluate(model, loader, lookback_window)

    mse_loss = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    state = {"m": 0.0, "PH": 0.0, "t": 0}
    window = deque(maxlen=window_size)
    errors = []

    for _, x, y in loader:
        x = x.float().to(device).view(-1, 1, lookback_window)
        y_int = y.long().to(device).squeeze(-1)
        y_float = y_int.float()
        with torch.no_grad():
            pred_class = model(x).argmax(dim=1)
        err = mse_loss(pred_class.float(), y_float).item()
        errors.append(err)
        state = page_hinkley_update(err, state, delta)
        window.append((x.cpu(), y_int.cpu()))

        if state["PH"] > lambda_thr and len(window) == window_size:
            model.train()
            for _ in range(retrain_epochs):
                for wx, wy_int in window:
                    wx = wx.to(device)
                    wy = wy_int.to(device)
                    out = model(wx)
                    loss = celoss(out, wy)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            state = {"m": 0.0, "PH": 0.0, "t": 0}
            window.clear()

    return sum(errors) / len(errors)


def evaluate_regression(model, loader, use_ph=False,
                        delta=0.005, lambda_thr=1.0,
                        window_size=50, retrain_epochs=3,
                        lr=1e-4, lookback_window=8):
    """Regression MSE (continuous target); optional PH retrain.
    Collected from ``cstr/exp/fgl_cstr_regression.py``."""
    mse_loss = nn.MSELoss()
    model.eval()

    if not use_ph:
        total = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, lookback_window)
                y = y.float().to(device)
                total += mse_loss(model(x), y).item()
        return total / len(loader)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    state = {"m": 0.0, "PH": 0.0, "t": 0}
    window = deque(maxlen=window_size)
    errors = []
    for _, x, y in loader:
        x = x.float().to(device).view(-1, 1, lookback_window)
        y_float = y.float().to(device)
        with torch.no_grad():
            pred = model(x)
        err = mse_loss(pred, y_float).item()
        errors.append(err)
        state = page_hinkley_update(err, state, delta)
        window.append((x.cpu(), y_float.cpu()))
        if state["PH"] > lambda_thr and len(window) == window_size:
            model.train()
            for _ in range(retrain_epochs):
                for wx, wy in window:
                    wx, wy = wx.to(device), wy.to(device)
                    out = model(wx)
                    loss = mse_loss(out, wy)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            state = {"m": 0.0, "PH": 0.0, "t": 0}
            window.clear()
    return sum(errors) / len(errors)


def evaluate_seq(model, loader, output_steps):
    """Seq2Seq MSE: argmax bin averaged over steps. From ``fgl_cstr_seq2seq.py``."""
    mse_loss = nn.MSELoss()
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device).view(-1, 1, x.shape[1])
            y = y.long().to(device)
            pred = model(x)
            pred_bins = pred.argmax(dim=-1).float()
            total += mse_loss(pred_bins, y.float()).item()
            count += 1
    return total / count


def compute_per_sample_errors(model, loader, L):
    """Per-sample cross-entropy loss as a dict {sample_idx: error}.
    From ``adaptive_weight_exp.py`` (uses batch_size=1, no drop_last loaders)."""
    celoss = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    errors = {}
    with torch.no_grad():
        for indices, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device)
            logits = model(x)
            per_sample = celoss(logits, y_int)
            for j, idx in enumerate(indices):
                errors[idx.item()] = per_sample[j].item()
    return errors


def compute_per_sample_mse(model, loader, L):
    """Per-sample bin-index MSE as a dict {sample_idx: mse}.

    ``mse = (argmax_pred - true_bin) ** 2`` — consistent with :func:`evaluate`
    (classification MSE on predicted vs true bin index). Feeds the teacher–student
    MSE-gap weighting in :func:`run_adaptive_weight`; uses batch_size=1 loaders.
    """
    model.eval()
    errors = {}
    with torch.no_grad():
        for indices, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device).squeeze(-1)
            pred = model(x).argmax(dim=1).float()
            per_sample = (pred - y_int.float()).pow(2)
            for j, idx in enumerate(indices):
                errors[idx.item()] = per_sample[j].item()
    return errors


# ================================================================
#  Unified 3-stage FGL experiment
# ================================================================
def run_fgl_experiment(
    data,
    lookback_window,
    forecasting_horizon,
    model_fn=RNN,
    alpha=0.5,
    temperature=4,
    num_bins=50,
    val_size=0.2,
    test_size=0.2,
    epochs=30,
    batch_size=64,
    patience=5,
    lr=1e-4,
    hidden=128,
    num_layers=2,
    seed=42,
    regression=False,
    use_ph=False,
    ph_delta=0.005, ph_lambda=1.0, ph_window=50, ph_retrain_epochs=3,
    verbose=True,
    label="",
):
    """Unified 3-stage FGL: teacher(1-step, offset=H-1) → baseline(H-step) →
    student(H-step + distillation).

    Variant coverage (differences collapse into parameters):
      ``model_fn``   : model constructor (default ``RNN``; pass ``LSTMModel`` for the LSTM ablation).
      ``regression`` : True → ``RNNRegression`` + MSE loss + MSE distillation (no discretization).
      ``use_ph``     : Page-Hinkley drift-driven evaluation (the "drift" experiment).

    Numerically equivalent to ``cstr/exp/fgl_cstr.py`` (classification) and
    ``cstr/exp/fgl_cstr_regression.py`` (regression).
    """
    torch.manual_seed(seed)
    L, H = lookback_window, forecasting_horizon
    tag = f"[{label}] " if label else ""

    # ---- datasets ----
    if regression:
        teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
            data=data, lookback_window=L, forecasting_horizon=1,
            num_bins=50, val_size=val_size, test_size=test_size,
            offset=H - 1, batch_size=batch_size, MSE=True,
        )
        student_train, student_val, student_test, _, _ = create_time_series_dataset(
            data=data, lookback_window=L, forecasting_horizon=H,
            num_bins=50, val_size=val_size, test_size=test_size,
            offset=0, batch_size=batch_size, MSE=True,
        )
    else:
        bin_edges, y_min, y_max = compute_shared_bin_edges(data, L, num_bins)
        teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
            data=data, lookback_window=L, forecasting_horizon=1,
            num_bins=num_bins, val_size=val_size, test_size=test_size,
            offset=H - 1, batch_size=batch_size, bin_edges=bin_edges,
        )
        student_train, student_val, student_test, _, _ = create_time_series_dataset(
            data=data, lookback_window=L, forecasting_horizon=H,
            num_bins=num_bins, val_size=val_size, test_size=test_size,
            offset=0, batch_size=batch_size, bin_edges=bin_edges,
        )

    if verbose:
        kind = "Regression" if regression else model_fn.__name__
        print(f"\n{'=' * 50}")
        print(f"{tag}H={H:2d}  α={alpha:.2f}  T={temperature:.1f}  "
              f"Bins={'n/a' if regression else num_bins}  Epochs={epochs}  Lookback={L}  [{kind}]")
        if not regression:
            print(f"Bin edges: [{y_min:.4f}, {y_max:.4f}] → {num_bins} bins")
        print(f"{'=' * 50}")

    if regression:
        loss_fn = nn.MSELoss()
        mk = lambda: RNNRegression(L, hidden, num_layers).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss()
        mk = lambda: model_fn(L, hidden, num_bins, num_layers).to(device)

    def _to_y(y):
        return y.float().to(device) if regression else y.long().to(device)

    # ---- teacher ----
    teacher = mk()
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            y = _to_y(y)
            opt_t.zero_grad()
            loss_fn(teacher(x), y).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            vl = sum(loss_fn(teacher(x.float().to(device).view(-1, 1, L)), _to_y(y)).item()
                     for _, x, y in teacher_val) / len(teacher_val)
        if stop_t.step(vl, teacher):
            break
    stop_t.restore(teacher)

    # ---- baseline ----
    baseline = mk()
    opt_b = optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, L)
            y = _to_y(y)
            opt_b.zero_grad()
            loss_fn(baseline(x), y).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            vl = sum(loss_fn(baseline(x.float().to(device).view(-1, 1, L)), _to_y(y)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop_b.step(vl, baseline):
            break
    stop_b.restore(baseline)

    # ---- student + distillation ----
    student = mk()
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        student.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            targets = _to_y(y_s)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                t_out = teacher(x_t)
            if regression:
                loss = alpha * loss_fn(outputs, targets) + (1.0 - alpha) * loss_fn(outputs, t_out)
            else:
                loss = alpha * loss_fn(outputs, targets) + KL(outputs, t_out, temperature, alpha)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            vl = sum(loss_fn(student(x.float().to(device).view(-1, 1, L)), _to_y(y)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop_s.step(vl, student):
            break
    stop_s.restore(student)

    # ---- evaluation ----
    ev = evaluate_regression if regression else evaluate_with_ph
    ev_kw = dict(use_ph=use_ph, delta=ph_delta, lambda_thr=ph_lambda,
                 window_size=ph_window, retrain_epochs=ph_retrain_epochs, lr=lr,
                 lookback_window=L)
    teacher_mse = ev(teacher, teacher_test, **ev_kw)
    baseline_mse = ev(baseline, student_test, **ev_kw)
    student_mse = ev(student, student_test, **ev_kw)
    improvement = (baseline_mse - student_mse) / baseline_mse * 100 if baseline_mse > 0 else 0

    if verbose:
        print(f"  Teacher:  {teacher_mse:.4f}")
        print(f"  Baseline: {baseline_mse:.4f}")
        print(f"  Student:  {student_mse:.4f}  (Δ={improvement:+.1f}%)")

    return {"horizon": H, "lookback": L, "alpha": alpha, "temperature": temperature,
            "teacher": teacher_mse, "baseline": baseline_mse,
            "student": student_mse, "improvement": improvement}


# ================================================================
#  Adaptive distillation weights (A/B/C/D) — independent flow
# ================================================================
def run_adaptive_weight(data, L=20, H=15, alpha=0.5, temperature=4, num_bins=50,
                        epochs=30, batch_size=64, patience=5, seed=42,
                        variant='A', verbose=True):
    """FGL with per-sample weighted KL (variants A/B/C/D).
    Collected from ``cstr/exp/adaptive_weight_exp.py`` (run_experiment)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden, output, layers, lr = 128, num_bins, 2, 1e-4

    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    if verbose:
        print(f"  [{variant}] L={L} H={H} α={alpha} T={temperature} seed={seed}")

    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=H - 1, batch_size=batch_size, bin_edges=bin_edges)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=0, batch_size=batch_size, bin_edges=bin_edges)
    # batch_size=1, no drop_last loaders for per-sample error coverage
    student_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=0, batch_size=1, bin_edges=bin_edges)
    teacher_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=H - 1, batch_size=1, bin_edges=bin_edges)

    ce = nn.CrossEntropyLoss()

    def train_simple(model, loader, vloader):
        opt = optim.Adam(model.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(epochs):
            model.train()
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, L)
                opt.zero_grad()
                ce(model(x), y.long().to(device)).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = sum(ce(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                         for _, x, y in vloader) / len(vloader)
            if stop.step(vl, model):
                break
        stop.restore(model)
        model.eval()

    teacher = RNN(L, hidden, output, layers).to(device)
    train_simple(teacher, teacher_train, teacher_val)
    baseline = RNN(L, hidden, output, layers).to(device)
    train_simple(baseline, student_train, student_val)

    # Always train a preliminary standard-FGL student (uniform-weight KL) as the
    # "initial student" reference. Its per-sample MSE drives the gap weights
    # (B/C/D), and its test MSE is the baseline the final weighted student must
    # beat for adaptive weighting to count as helpful.
    def train_student_standard():
        stu = RNN(L, hidden, output, layers).to(device)
        opt = optim.Adam(stu.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(epochs):
            stu.train()
            for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
                xs = xs.float().to(device).view(-1, 1, L)
                ys = ys.long().to(device)
                out = stu(xs)
                xt = xt.float().to(device).view(-1, 1, L)
                with torch.no_grad():
                    tlog = teacher(xt)
                loss = alpha * ce(out, ys) + KL(out, tlog, temperature, alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()
            stu.eval()
            with torch.no_grad():
                vl = sum(ce(stu(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                         for _, x, y in student_val) / len(student_val)
            if stop.step(vl, stu):
                break
        stop.restore(stu)
        stu.eval()
        return stu

    prelim_student = train_student_standard()
    student_mse_init = evaluate(prelim_student, student_test, L)

    # per-sample weights — criterion = max(0, se_student − se_teacher) MSE gap,
    # taken on the preliminary student vs the teacher.
    student_train_indices = [indices[0].item() for indices, _, _ in student_train_full]
    if variant == 'A':
        student_errors, teacher_errors = {}, {}
    else:
        student_errors = compute_per_sample_mse(prelim_student, student_train_full, L)
        te_raw = compute_per_sample_mse(teacher, teacher_train_full, L)
        # teacher loader uses offset=H-1 → its tuple-idx j predicts the same raw
        # target as student idx j-(H-1). Re-key into student-index space so the
        # gap is on aligned targets (fixes the offset misalignment inherited from
        # the no-op is_teacher_loader in the original script).
        teacher_errors = {int(j - (H - 1)): e for j, e in te_raw.items()}
    weights_dict, raw_w, norm_w = compute_weights(variant, student_errors,
                                                  teacher_errors, student_train_indices)
    if verbose:
        print(f"    Raw weights: mean={raw_w.mean():.3f} std={raw_w.std():.3f} "
              f"min={raw_w.min():.3f} max={raw_w.max():.3f}")

    # student with weighted distillation
    student = RNN(L, hidden, output, layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for _ in range(epochs):
        student.train()
        for (idx_s, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            targets = y_s.long().to(device)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                logits = teacher(x_t)
            if variant == 'D':
                bw = torch.tensor([weights_dict.get(i.item(), 1.0) for i in idx_s],
                                  dtype=torch.float32, device=device)
                alpha_i = torch.clamp(alpha / bw, 0.01, 0.99)
                ce_ps = nn.functional.cross_entropy(outputs, targets, reduction='none')
                log_ps = nn.functional.log_softmax(outputs / temperature, dim=1)
                pt = nn.functional.softmax(logits / temperature, dim=1)
                kl_ps = nn.functional.kl_div(log_ps, pt, reduction='none').sum(dim=1)
                loss = (alpha_i * ce_ps + (1 - alpha_i) * (temperature ** 2) * kl_ps).mean()
            else:
                bw = torch.tensor([weights_dict.get(i.item(), 1.0) for i in idx_s],
                                  dtype=torch.float32, device=device)
                loss = alpha * ce(outputs, targets) + KL_weighted(outputs, logits, temperature, alpha, bw)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            vl = sum(ce(student(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop_s.step(vl, student):
            break
    stop_s.restore(student)
    student.eval()

    t_mse = evaluate(teacher, teacher_test, L)
    b_mse = evaluate(baseline, student_test, L)
    s_mse = evaluate(student, student_test, L)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0
    # vs the preliminary (initial) student — positive = adaptive weighting
    # descended below the initial standard-FGL student.
    init_delta = (student_mse_init - s_mse) / student_mse_init * 100 if student_mse_init > 0 else 0
    if verbose:
        print(f"  [{variant}] Baseline={b_mse:.1f}  initStudent={student_mse_init:.1f}  "
              f"Student={s_mse:.1f}  Δbase={improvement:+.1f}%  Δinit={init_delta:+.1f}%")
    return {"variant": variant, "L": L, "H": H, "seed": seed,
            "teacher_mse": t_mse, "baseline_mse": b_mse,
            "student_mse_init": student_mse_init, "student_mse": s_mse,
            "abs_improvement": b_mse - s_mse, "fgl_delta": improvement,
            "init_delta": init_delta}


# ================================================================
#  Iterative adaptive distillation — helpers
# ================================================================
def _should_stop(mse_history, eps, N_stall, max_rounds):
    """迭代蒸馏停止规则(纯函数)。

    Args:
        mse_history: 逐轮 val MSE 列表;index 0 = round-0(初始 student),
            index t = 第 t 轮后。len == 已完成轮数 + 1。
        eps: 相对改进低于此值视为"停滞"。
        N_stall: 连续停滞达此次数则停。
        max_rounds: 轮数上限(K);t >= max_rounds 即停。
    Returns:
        (stop, reason),reason ∈ {"cap","degradation","stall","continue"}。
    """
    t = len(mse_history) - 1
    if t >= max_rounds:
        return True, "cap"
    if t == 0:
        return False, "continue"
    cur, prev = mse_history[t], mse_history[t - 1]
    if cur > prev:
        return True, "degradation"
    stall = 0
    for s in range(t, 0, -1):
        p, c = mse_history[s - 1], mse_history[s]
        if p <= 0 or c > p:
            break
        if (p - c) / p < eps:
            stall += 1
        else:
            break
    if stall >= N_stall:
        return True, "stall"
    return False, "continue"


def _compute_arm_weights(variant, student, teacher, student_train_full,
                         teacher_train_full, student_train_indices, L, H):
    """单臂逐样本蒸馏权重(对齐目标)。

    variant='A' -> 恒为 1.0(对照臂:从不更新权重)。
    variant='E' -> max(0, se_student − se_teacher) 差距,零地板放大到 [0, W_MAX=4]
                   (由 compute_weights 处理)。
    teacher loader 因 offset=H−1,其原始 idx j 对齐到 student idx j−(H−1),此处重映射。
    """
    if variant == "A":
        return {idx: 1.0 for idx in student_train_indices}
    se = compute_per_sample_mse(student, student_train_full, L)
    te_raw = compute_per_sample_mse(teacher, teacher_train_full, L)
    te = {int(j - (H - 1)): e for j, e in te_raw.items()}
    weights, _, _ = compute_weights("E", se, te, student_train_indices)
    return weights


def _iterate_student(student_0, teacher, variant, max_rounds,
                     student_train, teacher_train, student_val, student_test,
                     student_train_full, teacher_train_full, student_train_indices,
                     L, H, alpha, temperature, round_epochs, patience,
                     eps, N_stall, lr):
    """单臂暖启动迭代蒸馏。

    student_0: 已训练的 round-0 student(共享,本函数不修改入参对象)。
    variant: 'A'(每轮恒均匀)或 'E'(每轮按当前 student 重估 gap 权重)。
    max_rounds: 该臂最大轮数(E-single=1;iter 臂=K)。
    返回 dict: {rounds_used, total_epochs, mse_curve_val, mse_curve_test, student}。
    student 为 keep-best-by-val 的模型实例。
    """
    ce = torch.nn.CrossEntropyLoss()
    student = copy.deepcopy(student_0)
    student.eval()

    mse_curve_val = [evaluate(student, student_val, L)]
    mse_curve_test = [evaluate(student, student_test, L)]
    best_val = mse_curve_val[0]
    best_state = {k: v.clone() for k, v in student.state_dict().items()}
    total_epochs = 0
    rounds_used = 0

    for r in range(1, max_rounds + 1):
        weights = _compute_arm_weights(variant, student, teacher,
                                       student_train_full, teacher_train_full,
                                       student_train_indices, L, H)
        opt = optim.Adam(student.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(round_epochs):
            student.train()
            for (idx_s, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
                x_s = x_s.float().to(device).view(-1, 1, L)
                targets = y_s.long().to(device)
                outputs = student(x_s)
                x_t = x_t.float().to(device).view(-1, 1, L)
                with torch.no_grad():
                    logits = teacher(x_t)
                bw = torch.tensor([weights.get(i.item(), 1.0) for i in idx_s],
                                  dtype=torch.float32, device=device)
                loss = alpha * ce(outputs, targets) + KL_weighted(outputs, logits, temperature, alpha, bw)
                opt.zero_grad(); loss.backward(); opt.step()
            student.eval()
            with torch.no_grad():
                vl = sum(ce(student(x.float().to(device).view(-1, 1, L)),
                            y.long().to(device)).item()
                         for _, x, y in student_val) / len(student_val)
            total_epochs += 1
            if stop.step(vl, student):
                break
        stop.restore(student)
        student.eval()

        mv = evaluate(student, student_val, L)
        mse_curve_val.append(mv)
        mse_curve_test.append(evaluate(student, student_test, L))
        if mv < best_val:
            best_val = mv
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
        rounds_used = r

        stop_flag, _ = _should_stop(mse_curve_val, eps, N_stall, max_rounds)
        if stop_flag:
            break

    student.load_state_dict(best_state)
    student.eval()
    return {"rounds_used": rounds_used, "total_epochs": total_epochs,
            "mse_curve_val": mse_curve_val, "mse_curve_test": mse_curve_test,
            "student": student}


# ================================================================
#  Inference-time adaptive blending — independent flow
# ================================================================
def run_adaptive_inference(data, student_horizon=12, base_alpha=0.5, num_bins=50,
                           val_size=0.2, test_size=0.2, epochs=30, temperature=4,
                           lookback_window=20, batch_size=64, patience=5, seed=0,
                           divergence_threshold=8.0, blend_strength=0.7,
                           verbose=True):
    """Inference-time adaptive blending (no weight modification).
    Collected from ``cstr/exp/fgl_cstr_adaptive.py``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    L, H = lookback_window, student_horizon
    hidden, output, layers, lr = 128, num_bins, 2, 1e-4

    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=H - 1, batch_size=batch_size, bin_edges=bin_edges)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=0, batch_size=batch_size, bin_edges=bin_edges)

    ce = nn.CrossEntropyLoss()

    def train_simple(model, loader, vloader):
        opt = optim.Adam(model.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(epochs):
            model.train()
            for _, x, y in loader:
                opt.zero_grad()
                ce(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).backward()
                opt.step()
            model.eval()
            vl = sum(ce(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in vloader) / len(vloader)
            if stop.step(vl, model):
                break
        stop.restore(model)
        model.eval()

    teacher = RNN(L, hidden, output, layers).to(device)
    train_simple(teacher, teacher_train, teacher_val)
    baseline = RNN(L, hidden, output, layers).to(device)
    train_simple(baseline, student_train, student_val)
    student = RNN(L, hidden, output, layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for _ in range(epochs):
        student.train()
        for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
            xs, ys = xs.float().to(device).view(-1, 1, L), ys.long().to(device)
            xtt = xt.float().to(device).view(-1, 1, L)
            out = student(xs)
            with torch.no_grad():
                tlog = teacher(xtt)
            loss = base_alpha * ce(out, ys) + KL(out, tlog, temperature, base_alpha)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        vl = sum(ce(student(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                 for _, x, y in student_val) / len(student_val)
        if stop_s.step(vl, student):
            break
    stop_s.restore(student)
    student.eval()

    mse = nn.MSELoss()
    # baseline / teacher / standard student MSE
    b_total = sum(mse(baseline(x.float().to(device).view(-1, 1, L)).argmax(dim=1).float(),
                      y.long().to(device).squeeze(-1).float()).item()
                  for _, x, y in student_test) / len(student_test)
    t_total = sum(mse(teacher(x.float().to(device).view(-1, 1, L)).argmax(dim=1).float(),
                      y.long().to(device).squeeze(-1).float()).item()
                  for _, x, y in teacher_test) / len(teacher_test)
    s_std_total = 0.0
    for _, x, y in student_test:
        x = x.float().to(device).view(-1, 1, L)
        y_int = y.long().to(device).squeeze(-1)
        pred = student(x).argmax(dim=1).float()
        s_std_total += mse(pred, y_int.float()).item()
    student_mse_standard = s_std_total / len(student_test)

    # adaptive blending over raw test windows
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    N_total = len(x_raw) - L - H + 1
    n_test = int(N_total * test_size)
    test_start = N_total - n_test

    blend_count = 0
    adaptive_errors = []
    for idx in range(test_start, test_start + n_test):
        s_win = torch.tensor(x_raw[idx:idx + L], dtype=torch.float32).view(1, 1, L).to(device)
        true_bin = np.digitize(y_raw[idx + L + H - 1], bin_edges)
        t_win = torch.tensor(x_raw[idx + H - 1:idx + H - 1 + L], dtype=torch.float32).view(1, 1, L).to(device)
        with torch.no_grad():
            s_logits = student(s_win)
            s_pred = s_logits.argmax(dim=1).item()
            s_conf = torch.softmax(s_logits / temperature, dim=1).max().item()
            t_logits = teacher(t_win)
            t_pred = t_logits.argmax(dim=1).item()
            t_conf = torch.softmax(t_logits / temperature, dim=1).max().item()
        divergence = abs(s_pred - t_pred)
        teacher_sure = t_conf > 0.3
        they_disagree = divergence > divergence_threshold
        if teacher_sure and they_disagree:
            w = blend_strength * t_conf
            final_pred = ((1 - w) * s_logits + w * t_logits).argmax(dim=1).item()
            blend_count += 1
        else:
            final_pred = s_pred
        adaptive_errors.append((final_pred - true_bin) ** 2)
    student_mse_adaptive = np.mean(adaptive_errors)

    imp_std = (b_total - student_mse_standard) / b_total * 100
    imp_adp = (b_total - student_mse_adaptive) / b_total * 100
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RESULTS: Inference-Time Adaptive Blending  L={L} H={H} α={base_alpha} T={temperature}")
        print(f"  Teacher MSE:         {t_total:.1f}")
        print(f"  Baseline MSE:        {b_total:.1f}")
        print(f"  Student (standard):  {student_mse_standard:.1f}  (Δ={imp_std:+.1f}%)")
        print(f"  Student (adaptive):  {student_mse_adaptive:.1f}  (Δ={imp_adp:+.1f}%)")
        print(f"  Blend triggered:     {blend_count}/{n_test} ({100 * blend_count / n_test:.1f}%)")
        print(f"{'=' * 60}")
    return {"teacher_mse": t_total, "baseline_mse": b_total,
            "student_mse_standard": student_mse_standard,
            "student_mse_adaptive": student_mse_adaptive,
            "blend_fraction": blend_count / n_test}


# ================================================================
#  Seq2Seq FGL — independent flow
# ================================================================
def run_seq2seq(data, student_horizon=72, teacher_steps=10, alpha=0.5, num_bins=50,
                val_size=0.2, test_size=0.2, epochs=50, temperature=4,
                lookback_window=8, batch_size=64, patience=5, verbose=True):
    """Seq2Seq FGL: teacher predicts K steps, student predicts H steps; KL on first K.
    Collected from ``cstr/exp/fgl_cstr_seq2seq.py``."""
    torch.manual_seed(42)
    hidden, layers, lr = 128, 2, 1e-4
    H, K = student_horizon, min(teacher_steps, student_horizon)
    L = lookback_window

    # shared bin edges over all sequence targets
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y = []
    for i in range(len(x_raw) - L - H + 1):
        all_y.extend(y_raw[i + L: i + L + H])
    all_y = np.array(all_y)
    shared_bin_edges = np.linspace(all_y.min(), all_y.max(), num_bins - 1)

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"[Seq2Seq] H={H}  K={K}  α={alpha:.2f}  T={temperature:.1f}  "
              f"Bins={num_bins}  Lookback={L}  Epochs={epochs}")
        print(f"{'=' * 55}")

    teacher_train, teacher_val, teacher_test, _ = create_seq_dataset(
        data=data, lookback_window=L, forecasting_horizon=K, num_bins=num_bins,
        val_size=val_size, test_size=test_size, batch_size=batch_size, bin_edges=shared_bin_edges)
    student_train, student_val, student_test, _ = create_seq_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=val_size, test_size=test_size, batch_size=batch_size, bin_edges=shared_bin_edges)

    ce = nn.CrossEntropyLoss()

    def train_simple(model, loader, vloader, out_steps):
        opt = optim.Adam(model.parameters(), lr=lr)
        stop = EarlyStopper(patience=patience)
        for _ in range(epochs):
            model.train()
            for x, y in loader:
                x = x.float().to(device).view(-1, 1, L)
                y = y.long().to(device)
                out = model(x)
                loss = ce(out.reshape(-1, num_bins), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = sum(ce(model(x.float().to(device).view(-1, 1, L)).reshape(-1, num_bins),
                            y.long().to(device).reshape(-1)).item()
                         for x, y in vloader) / len(vloader)
            if stop.step(vl, model):
                break
        stop.restore(model)

    teacher = SeqRNN(L, hidden, K, num_bins, layers).to(device)
    train_simple(teacher, teacher_train, teacher_val, K)
    baseline = SeqRNN(L, hidden, H, num_bins, layers).to(device)
    train_simple(baseline, student_train, student_val, H)

    student = SeqRNN(L, hidden, H, num_bins, layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for _ in range(epochs):
        student.train()
        for (x_s, y_s), (x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            y_s = y_s.long().to(device)
            s_out = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                t_out = teacher(x_t)
            ce_loss = ce(s_out.reshape(-1, num_bins), y_s.reshape(-1))
            kl_loss = seq_KL(s_out, t_out, temperature, alpha, K)
            loss = alpha * ce_loss + kl_loss
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            vl = sum(ce(student(x.float().to(device).view(-1, 1, L)).reshape(-1, num_bins),
                        y.long().to(device).reshape(-1)).item()
                     for x, y in student_val) / len(student_val)
        if stop_s.step(vl, student):
            break
    stop_s.restore(student)

    t_mse = evaluate_seq(teacher, teacher_test, K)
    b_mse = evaluate_seq(baseline, student_test, H)
    s_mse = evaluate_seq(student, student_test, H)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0
    if verbose:
        print(f"  Teacher (K={K}):  {t_mse:.4f}")
        print(f"  Baseline (H={H}): {b_mse:.4f}")
        print(f"  Student (H={H}):  {s_mse:.4f}  (Δ={improvement:+.1f}%)")
    return {"horizon": H, "teacher_steps": K, "teacher": t_mse,
            "baseline": b_mse, "student": s_mse, "improvement": improvement}
