"""Exact finite-grid GMC mass variance; no simulation is repeated.

The Gaussian exponential identity and translation invariance give
E[M_N^2] = N^-3 sum_j exp(gamma^2 C_N(j/N)). This exposes rare-tail
uncertainty that an empirical realization bootstrap can miss.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.log_field_3d import spectral_configuration


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("results/phase3a/field_validation/summary.json"))
    parser.add_argument("--output",type=Path,default=Path("results/phase3a/field_validation/mass_uncertainty_audit.json"))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    summary=json.loads(args.input.read_text()); samples=summary["realizations"]; result={}
    for n,rows in summary["gmc"].items():
        _,c,_=spectral_configuration(int(n)); result[n]={}
        for gamma,row in rows.items():
            second=float(np.exp(float(gamma)**2*c).mean()); variance=max(0.,second-1)
            result[n][gamma]={"exact_mean":1.,"exact_second_moment":second,
                             "exact_variance":variance,"exact_sample_mean_se":float(np.sqrt(variance/samples)),
                             "empirical_mean":row["total_mass"]["mean"],
                             "empirical_se":row["total_mass"]["se"]}
    payload={"identity":"E M_N² = mean_j exp(gamma² C_N(j/N)); finite-grid periodic quadrature only",
             "samples":samples,"results":result,"input":str(args.input),
             "input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "warning":"Exact SE does not imply a Gaussian CLT approximation is accurate for 256 intermittent fields"}
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    ns=sorted(int(n) for n in result)
    for gamma in result[str(ns[-1])]:
        means=[result[str(n)][gamma]["empirical_mean"] for n in ns]
        exact_se=[result[str(n)][gamma]["exact_sample_mean_se"] for n in ns]
        axes[0].errorbar(ns,means,yerr=2*np.array(exact_se),marker="o",label=f"γ={gamma}")
    axes[0].axhline(1.,color="black",linestyle="--")
    axes[0].set(xlabel="grid N",ylabel="mean unnormalized mass",title="Whiskers: ±2 exact finite-grid SE")
    axes[0].legend()
    gammas=[float(g) for g in result[str(ns[-1])]]
    axes[1].plot(gammas,[result[str(ns[-1])][str(g)]["exact_sample_mean_se"] for g in gammas],"o-",label="exact covariance identity")
    axes[1].plot(gammas,[result[str(ns[-1])][str(g)]["empirical_se"] for g in gammas],"s--",label="empirical sample SE")
    axes[1].set(xlabel="γ",ylabel="SE of mean, 256 fields",title=f"Rare-tail undercoverage, N={ns[-1]}")
    axes[1].legend();fig.tight_layout()
    for suffix in [".png",".pdf"]:
        figure=args.output.with_name(args.output.stem+suffix)
        if figure.exists(): raise FileExistsError(figure)
        fig.savefig(figure,dpi=160)
    plt.close(fig)
    print(json.dumps({"output":str(args.output),"n48_gamma2":result["48"]["2.0"]}))


if __name__=="__main__": main()
