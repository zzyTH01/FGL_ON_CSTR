"""核实混沌 CSTR 上 E_iter vs A_iter 地板:K=5 时 A 未饱和(rounds_used 顶 cap)。
在 tau100 关键 cell 上 K=10(饱和)、多 seed 重跑 run_iterative_distillation,
捕获逐轮曲线 + 最终地板,看饱和后 E<A 是否仍成立。

用法:
  uv run python tmp_chaotic_verify.py --smoke        # 1 cell 1 seed 测速
  uv run python tmp_chaotic_verify.py               # 全量(后台)
"""
import argparse, csv, os, pickle, sys, time
import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
from fgl_common import run_iterative_distillation  # noqa: E402

DATA = os.path.join(_REPO, "cstr", "data")
TAG = "s1_A0.9_b0.03"
OUT = os.path.join(_REPO, "cstr", "results", "chaotic_floor_verify.csv")

# 关键 cell: 锚点(对照)+ E 大赢 + A 赢, 覆盖 L 与 H 两轴
CELLS = [(20, 15), (50, 15), (85, 15), (100, 30)]

COLUMNS = ["dataset", "L", "H", "seed", "arm", "baseline_mse", "student_mse",
           "rounds_used", "total_epochs", "mse_curve_test"]


def load(label):
    fn = f"data_delayed_stable_h2o_{label}_{TAG}.pkl"
    with open(os.path.join(DATA, fn), "rb") as f:
        return pickle.load(f)


def curve_str(c):
    return ";".join(f"{x:.3f}" for x in c) if c else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tau100")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--round_epochs", type=int, default=15)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()

    data = load(args.dataset)
    cells = CELLS[:1] if args.smoke else CELLS
    seeds = list(range(1 if args.smoke else args.seeds))

    print(f"[verify] dataset={args.dataset} K={args.K} cells={cells} seeds={seeds} "
          f"device 见上", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, "w", newline="")
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader(); f.flush()
    t0 = time.time()
    n_done = 0
    try:
        for (L, H) in cells:
            for s in seeds:
                r = run_iterative_distillation(
                    data, L=L, H=H, alpha=0.5, temperature=4.0, num_bins=50,
                    epochs=args.epochs, round_epochs=args.round_epochs, batch_size=64,
                    patience=5, K=args.K, seed=s, variant="E", verbose=False)
                base = r["A_iter"]["baseline_mse"]
                for arm in ("A_iter", "E_iter"):
                    a = r[arm]
                    row = {"dataset": args.dataset, "L": L, "H": H, "seed": s, "arm": arm,
                           "baseline_mse": base, "student_mse": a["student_mse"],
                           "rounds_used": a["rounds_used"], "total_epochs": a["total_epochs"],
                           "mse_curve_test": curve_str(a["mse_curve_test"])}
                    w.writerow(row)
                f.flush()
                n_done += 1
                el = time.time() - t0
                print(f"  L={L:3d} H={H:2d} s{s}: A={r['A_iter']['student_mse']:.1f} "
                      f"(rnd {r['A_iter']['rounds_used']})  E={r['E_iter']['student_mse']:.1f} "
                      f"(rnd {r['E_iter']['rounds_used']})  [{el:.0f}s, {n_done}/{len(cells)*len(seeds)}]",
                      flush=True)
    finally:
        f.close()
    print(f"[verify] wrote {OUT}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
