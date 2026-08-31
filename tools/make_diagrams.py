"""Architecture and component diagrams. Run with .venv-figs/bin/python."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
plt.rcParams.update({"font.family":"serif","figure.dpi":300,"savefig.bbox":"tight"})
OUT="report/figures"; os.makedirs(OUT,exist_ok=True)
INK="#1A1A1A"; GREY="#6E6E6E"; BLUE="#1F5FA8"; FILL="#F2F4F7"; BFILL="#E3ECF7"

def box(ax,x,y,w,h,label,sub=None,fc=FILL,ec=GREY,fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2+(0.022 if sub else 0),label,ha="center",va="center",fontsize=fs,color=INK)
    if sub: ax.text(x+w/2,y+h/2-0.032,sub,ha="center",va="center",fontsize=fs-1.5,color=GREY,style="italic")

def arrow(ax,p,q,label=None,c=GREY,ls="-",rad=0.0,off=0.03,fs=6.5):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=8,lw=0.9,
                                 color=c,linestyle=ls,connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0]+q[0])/2+off,(p[1]+q[1])/2,label,fontsize=fs,color=c,ha="left",va="center")

# ---------------- Figure 1: three-tier architecture ----------------
fig,ax=plt.subplots(figsize=(8.4,4.2)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
DX=0.64  # diagram occupies 0..DX, equations live to the right of it

box(ax,0.20,0.79,0.26,0.13,"Global model  $x$","one per federation",fc=BFILL,ec=BLUE,fs=9)
teams=[(0.02,"Team 1"),(0.21,"Team 2"),(0.42,"Team $M$")]
for cx,lbl in teams:
    box(ax,cx,0.47,0.175,0.13,f"{lbl}",r"$w_m$",fs=8)
    arrow(ax,(cx+0.095,0.60),(0.31,0.79),rad=0.10)
    arrow(ax,(0.33,0.79),(cx+0.105,0.605),rad=0.10,c=BLUE,ls="--")
for cx0,tx in [(0.02,0.115),(0.21,0.305),(0.42,0.515)]:
    for k in range(3):
        cx=cx0+k*0.063
        box(ax,cx,0.16,0.050,0.115,r"$\theta$",fs=7.5)
        arrow(ax,(cx+0.028,0.275),(tx,0.47),rad=0.08)

ax.plot([DX+0.005,DX+0.005],[0.10,0.93],color="#D8DCE2",lw=0.8)
for y,txt,c in [(0.855,r"$x \leftarrow (1-\beta\gamma)\,x + \beta\gamma\,\bar{w}$",BLUE),
                (0.535,r"$w \leftarrow (1-\eta\lambda_{\mathrm{t}}-\eta\gamma)\,w$"+"\n"+r"$\qquad + \eta\gamma\,x + \eta\lambda_{\mathrm{t}}\,\bar{\theta}$",BLUE),
                (0.215,r"$\theta \leftarrow \theta - \alpha\nabla f - \alpha\lambda(\theta-w)$",GREY)]:
    ax.text(DX+0.03,y,txt,fontsize=8,color=c,ha="left",va="center")
ax.text(DX+0.03,0.93,"update rules",fontsize=7.5,color=INK,ha="left",va="center")
ax.text(DX+0.03,0.40,r"$\lambda_{\mathrm{t}}$ is $\lambda_{\mathrm{team}}$."+"\nUpstream it is\nfixed to $\lambda$.",
        fontsize=7,color=BLUE,ha="left",va="top")
ax.text(0.0,0.055,"Devices hold private data. Only parameters move.",fontsize=7,color=GREY,style="italic")
ax.text(0.32,0.97,"Three-tier structure of PerMFL",fontsize=10,ha="center",color=INK)
fig.savefig(f"{OUT}/fig_architecture.png"); plt.close(fig); print("wrote fig_architecture.png")

# ---------------- Figure 2: component diagram ----------------
fig,ax=plt.subplots(figsize=(7.4,4.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.36,0.86,0.28,0.10,"main.py","CLI, model, dispatch",fc=BFILL,ec=BLUE)
box(ax,0.02,0.60,0.28,0.13,"utils/","cicids, nslkdd, toniot,\nmodel_utils",fs=7.5)
box(ax,0.36,0.60,0.28,0.13,"servers/serverPerMFL.py","T x K x L loop, aggregation",fs=7.5)
box(ax,0.70,0.60,0.28,0.13,"clustering/team_former.py","MCTC + CFMD-i trigger",fs=7.5)
box(ax,0.36,0.33,0.28,0.11,"users/userPerMFL.py","local SGD",fs=7.5)
box(ax,0.70,0.33,0.28,0.11,"metrics.py","macro F1, recall, FPR",fs=7.5)
box(ax,0.02,0.33,0.28,0.11,"optimizers/","pFedMeOptimizer",fs=7.5)
box(ax,0.36,0.07,0.28,0.11,"results/*.h5","per-round series,\nconfusion matrices",fs=7.5)
for p,q in [((0.50,0.86),(0.50,0.735)),((0.30,0.665),(0.36,0.665)),
            ((0.50,0.60),(0.50,0.445)),((0.64,0.385),(0.70,0.385)),
            ((0.36,0.385),(0.30,0.385)),((0.64,0.665),(0.70,0.665)),
            ((0.50,0.33),(0.50,0.185))]:
    arrow(ax,p,q,None)
# edge labels placed clear of the boxes
for x,y,t,ha in [(0.515,0.795,"config","left"),(0.33,0.685,"clients, teams","center"),
                 (0.515,0.525,"team + device params","left"),(0.67,0.685,"signals","center"),
                 (0.515,0.255,"confusion matrices","left")]:
    ax.text(x,y,t,fontsize=6,color=GREY,ha=ha,va="bottom")
arrow(ax,(0.78,0.60),(0.62,0.665),None,rad=-0.25,c=BLUE,ls="--")
ax.text(0.80,0.545,"teams (group_division 3)",fontsize=6,color=BLUE,ha="center")
ax.text(0.5,0.985,"Component structure, PerMFL path only",fontsize=10,ha="center",color=INK)
ax.text(0.02,0.005,"Blue = entry point. Dashed = active only under --group_division 3. Seven baseline algorithms omitted.",
        fontsize=6.5,color=GREY,style="italic")
fig.savefig(f"{OUT}/fig_components.png"); plt.close(fig); print("wrote fig_components.png")

# ---------------- Figure 3: model architectures ----------------
def architectures():
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.0))
    def stack(ax,title,layers,note):
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        n=len(layers); top=0.90; bot=0.20; h=(top-bot)/n
        for i,(lbl,sub,fc) in enumerate(layers):
            y=top-(i+1)*h
            box(ax,0.08,y+0.014,0.84,h-0.028,lbl,sub,fc=fc,fs=8)
            if i<n-1:   # arrow points DOWN, from this box to the next
                ax.annotate("",xy=(0.5,y-0.014),xytext=(0.5,y+0.012),
                            arrowprops=dict(arrowstyle="-|>",lw=0.9,color=GREY))
        ax.set_title(title,fontsize=9.5,color=INK)
        ax.text(0.5,0.10,note,ha="center",va="top",fontsize=6.8,color=GREY,style="italic")
    stack(axes[0],"MCLR",
          [("input  79","one row of flow features",FILL),
           ("Linear  79 → 9","the only trainable layer",BFILL),
           ("log_softmax","log-probabilities, 9 classes",FILL)],
          "720 parameters. Convex: one minimum.")
    stack(axes[1],"DNN",
          [("input  79","",FILL),
           ("Linear  79 → 100","hidden layer 1",BFILL),
           ("ReLU","negatives set to zero",FILL),
           ("Linear  100 → 9","output layer",BFILL),
           ("log_softmax","log-probabilities",FILL)],
          "8,909 parameters. Non-convex.\nHIDDEN_LAYERS=2 inserts a second 100→100 pair.")
    stack(axes[2],"CNN",
          [("input  28 x 28","one image",FILL),
           ("Conv + pool","local patterns",BFILL),
           ("Conv + pool","larger patterns",BFILL),
           ("Linear → 128 → classes","",BFILL),
           ("log_softmax","log-probabilities",FILL)],
          "Image datasets only. Not used on\nany intrusion dataset in this project.")
    fig.suptitle("The three architectures, as implemented in FLAlgorithms/trainmodel/models.py",
                 fontsize=9,color=GREY)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_architectures_models.png"); plt.close(fig)
    print("wrote fig_architectures_models.png")

architectures()
