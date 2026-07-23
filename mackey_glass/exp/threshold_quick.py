#!/usr/bin/env python
"""Quick threshold test: L=5,7,9,10,11,12 with τ=13, H=5, 3 seeds, 30 epochs."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
import torch, numpy as np
from utils.utils import MackeyGlass, RNN, create_time_series_dataset, KL

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
TAU, H = 13, 5
L_VALS = [5, 7, 9, 10, 11, 12]
SEEDS = [0, 1, 2]

mg = MackeyGlass(tau=TAU, constant_past=0.9, nmg=10, beta=0.2, gamma=0.1, dt=1.0,
                 splits=(10000., 0.), seed_id=42)
vals = [mg[idx][1].squeeze().item() for idx in range(len(mg))]
col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
data = torch.cat((col, col.clone()), dim=1)

class ES:
    def __init__(s, p=5, d=1e-4): s.p, s.d, s.b, s.c, s.bs = p, d, float('inf'), 0, None
    def step(s, l, m):
        if l + s.d < s.b: s.b = l; s.c = 0; s.bs = {k: v.cpu() for k, v in m.state_dict().items()}; return False
        s.c += 1; return s.c >= s.p
    def restore(s, m):
        if s.bs: m.load_state_dict(s.bs)

def run(L, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    hs, nl, lr, os_, vs, ts = 128, 2, 1e-4, 50, 0.2, 0.2
    xr = np.array([float(pt[0]) for pt in data])
    yr = np.array([float(pt[1]) for pt in data])
    ay = [yr[i + L] for i in range(len(xr) - L - 1 + 1)]
    se = np.linspace(np.array(ay).min(), np.array(ay).max(), os_ - 1)
    tt, tv, tte, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=os_,
        val_size=vs, test_size=ts, offset=H - 1, batch_size=128, bin_edges=se)
    st, sv, ste, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=os_,
        val_size=vs, test_size=ts, offset=0, batch_size=128, bin_edges=se)
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    def train_one(model, loader, val_loader, opt):
        es_ = ES()
        for _ in range(30):
            model.train()
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, L); y = y.long().to(device)
                opt.zero_grad(); ce(model(x), y).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = sum(ce(model(x.float().to(device).view(-1, 1, L)),
                            y.long().to(device)).item() for _, x, y in val_loader) / len(val_loader)
            if es_.step(vl, model): break
        es_.restore(model)

    teacher = RNN(L, hs, os_, nl).to(device)
    train_one(teacher, tt, tv, torch.optim.Adam(teacher.parameters(), lr=lr))

    baseline = RNN(L, hs, os_, nl).to(device)
    train_one(baseline, st, sv, torch.optim.Adam(baseline.parameters(), lr=lr))

    student = RNN(L, hs, os_, nl).to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=lr)
    es_s = ES()
    for _ in range(30):
        student.train()
        for (_, xs, ys), (_, xt, _) in zip(st, tt):
            xs = xs.float().to(device).view(-1, 1, L); ys = ys.long().to(device)
            out = student(xs)
            xt = xt.float().to(device).view(-1, 1, L)
            with torch.no_grad(): lt = teacher(xt)
            loss = 0.5 * ce(out, ys) + KL(out, lt, 4, 0.5)
            opt_s.zero_grad(); loss.backward(); opt_s.step()
        student.eval()
        with torch.no_grad():
            vl = sum(ce(student(x.float().to(device).view(-1, 1, L)),
                        y.long().to(device)).item() for _, x, y in sv) / len(sv)
        if es_s.step(vl, student): break
    es_s.restore(student)

    def ev(m, ld):
        m.eval(); t = 0.0
        with torch.no_grad():
            for _, x, y in ld:
                x = x.float().to(device).view(-1, 1, L)
                t += mse(m(x).argmax(1).float(), y.float().to(device).squeeze(-1)).item()
        return t / len(ld)

    bm = ev(baseline, ste); sm = ev(student, ste)
    return bm, sm

print(f"L-threshold quick test: τ={TAU}, H={H}")
print(f"{'L':>4s} {'Base':>8s} {'Stu':>8s} {'AbsImp':>8s} {'Δ%':>8s}")
print("-" * 42)

all_rows = []
for L in L_VALS:
    bms, sms = [], []
    for s in SEEDS:
        bm, sm = run(L, s)
        bms.append(bm); sms.append(sm)
        d = (bm - sm) / bm * 100 if bm > 0 else 0
        all_rows.append((L, s, bm, sm, bm - sm, d))
        print(f"  L={L} s={s}: {bm:.3f} {sm:.3f} {bm-sm:+.3f} {d:+.1f}%")

    bm_m = np.mean(bms); sm_m = np.mean(sms)
    ai_m = np.mean([b - s for b, s in zip(bms, sms)])
    d_m = np.mean([(b - s) / b * 100 if b > 0 else 0 for b, s in zip(bms, sms)])
    print(f"  → MEAN:   {bm_m:.3f} {sm_m:.3f} {ai_m:+.3f} {d_m:+.1f}%")
    print()

# Save CSV
import csv
outdir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "threshold_quick_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["L", "seed", "baseline_mse", "student_mse", "abs_improvement", "fgl_delta"])
    w.writerows(all_rows)
print(f"Saved to {outdir}/threshold_quick_results.csv")
