"""Plot the four preselected DEV trajectories on matched axes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"stix","font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,
    "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,"svg.fonttype":"none",
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6})
COLORS={"Ground truth":"#26313e","Frozen":"#6387a9","Simple":"#79569d","PostY":"#cc8b26","HOVER":"#d34d62"}
STYLE={"Ground truth":"-","Frozen":(0,(4,2)),"Simple":(0,(6,1,1,1)),"PostY":(0,(1,1.8)),"HOVER":"-"}
def draw(cases, seed_rows, z, output):
    fig,axes=plt.subplots(2,2,figsize=(178/25.4,108/25.4))
    fig.subplots_adjust(left=.077,right=.985,bottom=.17,top=.885,hspace=.80,wspace=.28)
    legend=[Line2D([],[],color=COLORS[a],ls=STYLE[a],lw=1.1,
                   label=r"$\delta$-Adapter PostY" if a=="PostY" else a) for a in COLORS]
    fig.legend(handles=legend,ncol=5,loc="upper center",bbox_to_anchor=(.52,1.0),
               frameon=False,fontsize=8,handlelength=2.2,columnspacing=1.0)
    for i,(case,ax) in enumerate(zip(cases,axes.ravel())):
        prefix=f"{case['dataset']}_{case['kind']}_951"
        ax.plot(np.arange(-48,0),z[prefix+"_history"][-48:],color=COLORS["Ground truth"],lw=1.0)
        for arm in COLORS:
            key="truth" if arm=="Ground truth" else arm
            ax.plot(np.arange(96),z[prefix+"_"+key],color=COLORS[arm],ls=STYLE[arm],
                    lw=1.1 if arm=="Ground truth" else 1.12 if arm=="HOVER" else .95,
                    zorder=4 if arm=="HOVER" else 5 if arm=="Ground truth" else 3)
        ax.axvline(-.5,color="#8795a3",lw=.7,ls=(0,(2,2)))
        ax.axvspan(case["zoom_start"]-.5,case["zoom_end"]+.5,color="#e9d5a9",alpha=.32,zorder=0)
        kind="Typical" if case["kind"]=="typical" else "Illustrative refinement"
        ax.set_title(f"({chr(97+i)}) {case['dataset']} · {kind}",loc="left",fontweight="bold",pad=5)
        ax.set_xlim(-48,95)
        ax.set_xticks([-48,0,24,48,72,95])
        ax.set_xlabel("Step relative to forecast origin",labelpad=3)
        if i%2==0:ax.set_ylabel("TRAIN-standardized value",labelpad=3)
        ax.grid(axis="y",color="#e3e8ed",lw=.55)
        mse=[float(next(row for row in seed_rows if row["dataset"]==case["dataset"] and row["kind"]==case["kind"] and row["seed"]=="951" and row["segment"]=="0" and row["system"]==arm)["MSE"]) for arm in ("Frozen","Simple","PostY","HOVER")]
        text="Window MSE (F/S/P/H)  "+" / ".join(f"{v:.5f}" for v in mse)
        ax.text(0,-.36,text,transform=ax.transAxes,fontsize=6.8,ha="left",va="top")
    for ext in ("pdf","svg","png"):
        fig.savefig(output/f"Figure7_Trajectories.{ext}",dpi=360 if ext=="png" else None)
    plt.close(fig)
    # Separate detailed views retain all forecast curves on one local target scale.
    fig,axes=plt.subplots(2,2,figsize=(178/25.4,95/25.4))
    fig.subplots_adjust(left=.077,right=.985,bottom=.11,top=.87,hspace=.55,wspace=.28)
    fig.legend(handles=legend,ncol=5,loc="upper center",bbox_to_anchor=(.52,1.0),frameon=False,fontsize=8)
    for i,(case,ax) in enumerate(zip(cases,axes.ravel())):
        prefix=f"{case['dataset']}_{case['kind']}_951";lo,hi=case["zoom_start"],case["zoom_end"]+1
        for arm in COLORS:
            key="truth" if arm=="Ground truth" else arm
            ax.plot(np.arange(lo,hi),z[prefix+"_"+key][lo:hi],color=COLORS[arm],ls=STYLE[arm],lw=1.15 if arm in ("Ground truth","HOVER") else .95)
        ax.set_title(f"({chr(97+i)}) {case['dataset']} · {case['kind']} · steps {lo}–{hi-1}",loc="left",fontsize=8.5)
        ax.set_xlabel("Forecast step")
        ax.grid(axis="y",color="#e3e8ed",lw=.55)
        if i%2==0:ax.set_ylabel("TRAIN-standardized value")
    fig.savefig(output/"Figure7_Local_Views.pdf")
    fig.savefig(output/"Figure7_Local_Views.png",dpi=300)



    plt.close(fig)
