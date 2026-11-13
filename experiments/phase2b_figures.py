"""Standalone statistical-audit plots from Task2B saved fitting results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output")
    args=parser.parse_args()
    path=Path(args.input)
    d=json.loads(path.read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    sizes=np.array(d["sizes"])
    means=np.array(d["means"])
    errors=np.array(d["standard_errors"])
    axes[0].errorbar(sizes,means,yerr=1.96*errors,fmt="o",capsize=3)
    axes[0].set(xscale="log",yscale="log",xlabel="Vertices n",ylabel="Mean uniform-pair graph distance",
                title=f"{d['records']} exact biased I3 graphs; MC 95% errors")
    for model in dict.fromkeys(r["model"] for r in d["mean_models"]):
        rows=[r for r in d["mean_models"] if r["model"]==model]
        axes[1].plot([r["minimum_n"] for r in rows],[r["dimension"] for r in rows],"o-",
                     label=model.replace("multiplicative_delta_","delta="))
    axes[1].set(xscale="log",xlabel="Smallest retained n",ylabel="Inverse fitted power",
                title="Correction-model and window sensitivity")
    axes[1].legend(fontsize=7)
    for ratio in [.5,1.,2.]:
        rows=[r for r in d["equation45_models"] if r["delta_ratio"]==ratio and r["b_bound"]==.15]
        axes[2].plot([r["minimum_n"] for r in rows],[r["dimension"] for r in rows],"o-",
                     label=f"delta/p={ratio:g}, |b|<=.15")
        active=[r for r in rows if r["b_at_boundary"]]
        axes[2].scatter([r["minimum_n"] for r in active],[r["dimension"] for r in active],
                        marker="x",s=70,color="black",zorder=5)
    axes[2].set(xscale="log",xlabel="Smallest retained n",ylabel="Equation45 inverse fitted power",
                title="Reference-normalized Eq45, shared-reference GLS")
    axes[2].legend(fontsize=7)
    from matplotlib.ticker import NullFormatter
    for axis in axes[1:]:
        ticks=[int(n) for n in sizes[:-3] if int(np.log2(n))%2==1]
        if int(sizes[-4]) not in ticks: ticks.append(int(sizes[-4]))
        axis.set_xticks(ticks,labels=[f"{n:,}" for n in ticks],rotation=25,ha="right")
        axis.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Central fits show model sensitivity; intervals do not include correction-model error")
    fig.tight_layout()
    fig.savefig(Path(args.output) if args.output else path.with_suffix(".png"),dpi=180)
    plt.close(fig)


if __name__=="__main__":main()
