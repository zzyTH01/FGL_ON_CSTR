#!/usr/bin/env python
"""
Lorenz-63 L×H combined sweep — mirror of CSTR L×H sweep.

Fixed ρ=60 (strong chaos, Lyapunov ~1.6). Sweep L and H to map
the FGL effectiveness landscape.

Usage:
  cd /tmp && source .venv/bin/activate && python lorenz/lh_sweep.py
"""

import sys, os, csv, time, pickle, argparse
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils.utils import RNN, create_time_series_dataset, KL

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")

RHO = 60.0       # Strong chaos
SIGMA, BETA = 10.0, 8.0/3.0
L_VALUES = [8, 15, 25, 40, 60]
H_VALUES = [5, 15, 30, 45, 60]
SEEDS = [0, 1, 2]
EPOCHS = 30
ALPHA, TEMPERATURE = 0.5, 4
NUM_BINS, PATIENCE, BATCH_SIZE = 50, 5, 64
N_POINTS = 8000


def generate_lorenz():
    def ode(t, s): x,y,z=s; return [SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z]
    sol = solve_ivp(ode, [0, 500], [1.0, 1.0, 1.0], t_eval=np.arange(0, 500, 0.05),
                    method='RK45', rtol=1e-9, atol=1e-12)
    x = sol.y[0]; start = len(x)//5
    x = x[start:start+N_POINTS]
    col = torch.tensor(x, dtype=torch.float64).unsqueeze(1)
    return torch.cat((col, col.clone()), dim=1)


class EarlyStopper:
    def __init__(s, p=5, d=1e-4): s.p,s.d,s.b,s.c,s.bs=p,d,float('inf'),0,None
    def step(s,l,m):
        if l+s.d<s.b: s.b=l; s.c=0; s.bs={k:v.cpu() for k,v in m.state_dict().items()}; return False
        s.c+=1; return s.c>=s.p
    def restore(s,m):
        if s.bs: m.load_state_dict(s.bs)


def run_fgl(L, H, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    hs, nl, lr, os_, vs, ts = 128, 2, 1e-4, NUM_BINS, 0.2, 0.2
    xr=np.array([float(pt[0]) for pt in data]); yr=np.array([float(pt[1]) for pt in data])
    ay=[yr[i+L] for i in range(len(xr)-L-1+1)]
    se=np.linspace(np.array(ay).min(), np.array(ay).max(), os_-1)

    tt,tv,tte,_,_ = create_time_series_dataset(data=data,lookback_window=L,forecasting_horizon=1,
        num_bins=os_,val_size=vs,test_size=ts,offset=H-1,batch_size=BATCH_SIZE,bin_edges=se)
    st,sv,ste,_,_ = create_time_series_dataset(data=data,lookback_window=L,forecasting_horizon=H,
        num_bins=os_,val_size=vs,test_size=ts,offset=0,batch_size=BATCH_SIZE,bin_edges=se)
    ce,mse=nn.CrossEntropyLoss(),nn.MSELoss()

    def tm(m,ld,vld,opt):
        es=EarlyStopper(PATIENCE)
        for _ in range(EPOCHS):
            m.train()
            for _,x,y in ld: x=x.float().to(device).view(-1,1,L); y=y.long().to(device); opt.zero_grad(); ce(m(x),y).backward(); opt.step()
            m.eval()
            with torch.no_grad(): vl=sum(ce(m(x.float().to(device).view(-1,1,L)),y.long().to(device)).item() for _,x,y in vld)/len(vld)
            if es.step(vl,m): break
        es.restore(m)

    teacher=RNN(L,hs,os_,nl).to(device); tm(teacher,tt,tv,optim.Adam(teacher.parameters(),lr=lr))
    baseline=RNN(L,hs,os_,nl).to(device); tm(baseline,st,sv,optim.Adam(baseline.parameters(),lr=lr))

    student=RNN(L,hs,os_,nl).to(device); opt_s=optim.Adam(student.parameters(),lr=lr); es_s=EarlyStopper(PATIENCE)
    for _ in range(EPOCHS):
        student.train()
        for (_,xs,ys),(_,xt,_) in zip(st,tt):
            xs=xs.float().to(device).view(-1,1,L); ys=ys.long().to(device); out=student(xs)
            xt=xt.float().to(device).view(-1,1,L)
            with torch.no_grad(): lt=teacher(xt)
            loss=ALPHA*ce(out,ys)+KL(out,lt,TEMPERATURE,ALPHA); opt_s.zero_grad(); loss.backward(); opt_s.step()
        student.eval()
        with torch.no_grad(): vl=sum(ce(student(x.float().to(device).view(-1,1,L)),y.long().to(device)).item() for _,x,y in sv)/len(sv)
        if es_s.step(vl,student): break
    es_s.restore(student)

    def ev(m,ld):
        m.eval(); t=0.0
        with torch.no_grad():
            for _,x,y in ld: x=x.float().to(device).view(-1,1,L); t+=mse(m(x).argmax(1).float(),y.float().to(device).squeeze(-1)).item()
        return t/len(ld)
    return ev(baseline,ste),ev(student,ste),ev(teacher,tte)


def main():
    outdir=os.path.join(SCRIPT_DIR,"results"); os.makedirs(outdir,exist_ok=True)
    print(f"Lorenz ρ={RHO}, {N_POINTS} pts")
    print(f"L: {L_VALUES}  |  H: {H_VALUES}")
    total=len(L_VALUES)*len(H_VALUES)*len(SEEDS)
    print(f"Configs: {len(L_VALUES)*len(H_VALUES)}, Runs: {total}, Est: ~{total*0.5:.0f} min\n")

    print("Generating Lorenz data...")
    data=generate_lorenz()
    print(f"  Range: [{data[:,0].min():.1f}, {data[:,0].max():.1f}]\n")

    csv_path=os.path.join(outdir,"lorenz_lh_sweep.csv")
    rows=[]; t0=time.time()

    for L in L_VALUES:
        for H in H_VALUES:
            label=f"L={L:2d} H={H:2d} (L+H-1={L+H-1:3d})"
            print(f"\n{'='*45}\n  {label}\n{'='*45}")
            bms,sms,tms=[],[],[]
            for s in tqdm(SEEDS,desc="  Seeds"):
                bm,sm,tm=run_fgl(L,H,data,s)
                bms.append(bm); sms.append(sm); tms.append(tm)
                d=(bm-sm)/bm*100 if bm>0 else 0
                rows.append({"L":L,"H":H,"seed":s,"baseline_mse":bm,"teacher_mse":tm,"student_mse":sm,"abs_improvement":bm-sm,"fgl_delta":d})
            print(f"  Base={np.mean(bms):.1f}±{np.std(bms):.1f}  Stu={np.mean(sms):.1f}±{np.std(sms):.1f}  Δ={np.mean([(b-s)/b*100 if b>0 else 0 for b,s in zip(bms,sms)]):+.1f}%  ({(time.time()-t0)/60:.1f}m)")

    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["L","H","seed","baseline_mse","teacher_mse","student_mse","abs_improvement","fgl_delta"]); w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    agg=defaultdict(list)
    for r in rows: agg[(r["L"],r["H"])].append(r)
    L_s,H_s=sorted(set(r["L"] for r in rows)),sorted(set(r["H"] for r in rows))

    # Heatmaps
    grid_d=np.zeros((len(L_s),len(H_s))); grid_b=np.zeros_like(grid_d); grid_a=np.zeros_like(grid_d)
    for i,L in enumerate(L_s):
        for j,H in enumerate(H_s):
            rs=agg[(L,H)]
            grid_d[i,j]=np.mean([r["fgl_delta"] for r in rs])
            grid_b[i,j]=np.mean([r["baseline_mse"] for r in rs])
            grid_a[i,j]=np.mean([r["abs_improvement"] for r in rs])

    fig,axes=plt.subplots(1,3,figsize=(20,5.5))
    for ax,grid,title,cmap,label in [
        (axes[0],grid_d,"FGL Δ%","RdYlGn","Δ%"),
        (axes[1],grid_a,"Abs Improvement","RdYlGn","Abs Imp"),
        (axes[2],grid_b,"Baseline MSE","YlOrRd","Base MSE"),
    ]:
        im=ax.imshow(grid,aspect="auto",origin="lower",cmap=cmap,
                     extent=[H_s[0]-0.5,H_s[-1]+0.5,L_s[0]-0.5,L_s[-1]+0.5])
        ax.set_xticks(H_s); ax.set_yticks(L_s)
        ax.set_xlabel("H"); ax.set_ylabel("L")
        ax.set_title(title,fontweight="bold")
        for i in range(len(L_s)):
            for j in range(len(H_s)):
                v=grid[i,j]
                ax.text(H_s[j],L_s[i],f"{v:.0f}" if abs(v)<100 else f"{v:.0f}",ha="center",va="center",fontsize=8,
                        color="white" if abs(v)>(grid.max()+grid.min())/3 else "black")
        plt.colorbar(im,ax=ax,label=label)
    plt.suptitle(f"Lorenz-63 L×H Sweep (ρ={RHO}, strong chaos)",fontweight="bold",y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir,"lorenz_lh_sweep.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)

    # Console
    print("\n"+"="*70)
    print("  FGL Δ% Heatmap")
    print("="*70)
    lh = "L\\H"
    print(f"  {lh:>5s}",end="")
    for H in H_s: print(f"  {H:>8d}",end="")
    print("\n  "+"-"*(5+9*len(H_s)))
    for L in L_s:
        print(f"  {L:5d}",end="")
        for H in H_s:
            dm=np.mean([r["fgl_delta"] for r in agg[(L,H)]])
            print(f"  {dm:+7.1f}%",end="")
        print()

    all_aggs=[]
    for (L,H),rs in agg.items():
        all_aggs.append({"L":L,"H":H,"delta":np.mean([r["fgl_delta"] for r in rs]),
                         "abs":np.mean([r["abs_improvement"] for r in rs]),
                         "base":np.mean([r["baseline_mse"] for r in rs])})
    all_aggs.sort(key=lambda x:x["delta"],reverse=True)
    print("\nTop 5:")
    for a in all_aggs[:5]: print(f"  L={a['L']:2d} H={a['H']:2d}: Δ={a['delta']:+.1f}% Base={a['base']:.1f}")
    print("Bottom 5:")
    for a in all_aggs[-5:]: print(f"  L={a['L']:2d} H={a['H']:2d}: Δ={a['delta']:+.1f}% Base={a['base']:.1f}")

    # Report
    md_path=os.path.join(outdir,"lorenz_lh_sweep_report.md")
    with open(md_path,"w") as f:
        f.write(f"# Lorenz-63 L×H Combined Sweep\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Data:** Lorenz-63 ρ={RHO} (strong chaos, Lyapunov ~1.6), x(t), {N_POINTS} pts\n")
        f.write(f"**L:** {L_VALUES}  |  **H:** {H_VALUES}\n")
        f.write(f"**Seeds:** {SEEDS}  |  **Epochs:** {EPOCHS}  |  **α:** {ALPHA}\n\n")

        f.write("## FGL Δ% Heatmap\n\n")
        f.write("| L\\H | "+" | ".join(f"{h}" for h in H_s)+" |\n")
        f.write("|"+"---|"*(len(H_s)+1)+"\n")
        for L in L_s:
            f.write(f"| {L} | "+" | ".join(f"{np.mean([r['fgl_delta'] for r in agg[(L,H)]]):+.1f}%" for H in H_s)+" |\n")

        f.write("\n## Baseline MSE Heatmap\n\n")
        f.write("| L\\H | "+" | ".join(f"{h}" for h in H_s)+" |\n")
        f.write("|"+"---|"*(len(H_s)+1)+"\n")
        for L in L_s:
            f.write(f"| {L} | "+" | ".join(f"{np.mean([r['baseline_mse'] for r in agg[(L,H)]]):.1f}" for H in H_s)+" |\n")

        f.write(f"\n## Data: `lorenz_lh_sweep.csv` ({len(rows)} rows)\n")

    print(f"Report: {md_path}")


if __name__=="__main__":
    main()
