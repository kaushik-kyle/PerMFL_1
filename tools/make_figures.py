"""Convergence figures from results/*.h5. Run with .venv-figs/bin/python."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, h5py, glob, numpy as np, os, sys
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,
    "grid.alpha":0.3,"grid.linewidth":0.4,"axes.spines.top":False,
    "axes.spines.right":False,"figure.dpi":300,"savefig.bbox":"tight"})
C_P, C_F = "#4C4C4C", "#1F5FA8"
OUT = "report/figures"; os.makedirs(OUT, exist_ok=True)

CACHE={}
def load(ds):
    if ds in CACHE: return CACHE[ds]
    d={}
    for f in glob.glob(f"results/PerMFL/{ds}/**/*.h5", recursive=True):
        with h5py.File(f) as h:
            d[int(h['exp_no'][()])]={k:h[k][:] for k in h.keys() if h[k].shape}
    CACHE[ds]=d; return d

def band(ds, exps, key):
    d=load(ds); rs=[d[e][key] for e in exps if e in d and key in d[e]]
    if not rs: return None,None,None
    L=min(len(r) for r in rs); a=np.array([r[:L] for r in rs])
    return a.mean(0), a.min(0), a.max(0)

PANELS=[("CICIDS2017","Cicids",[906,908,910,912,914],[907,909,911,913,915]),
        ("TON-IoT","Toniot",[926,928,930,932,934],[927,929,931,933,935]),
        ("NSL-KDD","Nslkdd",[936,938,940,942,944],[937,939,941,943,945]),
        ("EMNIST-10","Emnist10",[2100,2102,2104],[2101,2103,2105])]

def convergence(key, ylab, fname):
    fig,axes=plt.subplots(1,4,figsize=(9.5,2.5),sharey=True)
    drew=False
    for ax,(title,ds,pe,fe) in zip(axes,PANELS):
        for exps,c,lbl in [(pe,C_P,"PerMFL"),(fe,C_F,"Fine Tuned")]:
            m,lo,hi=band(ds,exps,key)
            if m is None: continue
            x=np.arange(1,len(m)+1); drew=True
            ax.plot(x,m,color=c,lw=1.3,label=lbl)
            ax.fill_between(x,lo,hi,color=c,alpha=0.18,linewidth=0)
        ax.set_title(title,fontsize=9); ax.set_xlabel("Global round"); ax.set_ylim(0,1)
    if not drew: print(f"  SKIP {fname}: no data"); plt.close(fig); return
    axes[0].set_ylabel(ylab,fontsize=8)
    axes[0].legend(frameon=False,fontsize=8,loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)
    print(f"  wrote {fname}")

def sweep_curve():
    """Delta against heterogeneity: the central claim, as a line plot."""
    rows=[("CICIDS2017",[(0.1730,2000,2005),(0.1414,2006,2011),(0.0851,2012,2017),(0.0001,2018,2023)]),
          ("TON-IoT",[(0.7591,2024,2029),(0.3909,2030,2035),(0.0646,2036,2041)]),
          ("NSL-KDD",[(0.5379,2042,2047),(0.0333,2048,2053)])]
    dsmap={"CICIDS2017":"Cicids","TON-IoT":"Toniot","NSL-KDD":"Nslkdd"}
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.8))
    for ax,key,lbl in [(axes[0],"per_macro_f1","Personalised model"),
                       (axes[1],"global_macro_f1","Global model")]:
        for (name,pts),mk in zip(rows,["o","s","^"]):
            ds=dsmap[name]; d=load(ds); xs=[];ys=[]
            for jsd,lo,hi in pts:
                a=[np.nanmax(d[e][key]) for e in range(lo,hi+1,2) if e in d]
                b=[np.nanmax(d[e+1][key]) for e in range(lo,hi+1,2) if e+1 in d]
                if a and b: xs.append(jsd); ys.append(np.mean(b)-np.mean(a))
            if xs: ax.plot(xs,ys,marker=mk,lw=1.2,ms=4,label=name)
        ax.axhline(0,color="#999",lw=0.6,ls="--")
        ax.set_xlabel("Client heterogeneity (mean pairwise JSD)")
        ax.set_title(lbl,fontsize=9)
    axes[0].set_ylabel("Gain in macro F1\n(Fine Tuned minus PerMFL)",fontsize=8)
    axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_gain_vs_heterogeneity.png"); plt.close(fig)
    print("  wrote fig_gain_vs_heterogeneity.png")

def confusion():
    """Confusion matrices. Requires cm_global / cm_personal in the h5, which
    save_results does not yet write. See docs/backlog.md item B1."""
    d=load("Cicids")
    have=[e for e,v in d.items() if "cm_global" in v]
    if not have:
        print("  SKIP confusion matrices: no run has persisted a matrix (backlog B1)")
        return
    print(f"  {len(have)} runs carry matrices, plotting")

if __name__=="__main__":
    print("figures:")
    convergence("per_macro_f1","Personalised macro F1 (higher better)","fig_convergence_pm.png")
    convergence("global_macro_f1","Global macro F1 (higher better)","fig_convergence_gm.png")
    convergence("global_test_accuracy","Global accuracy (higher better)","fig_convergence_gm_acc.png")
    sweep_curve()
    confusion()
