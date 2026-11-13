"""Graph-cluster uncertainty budgets for saved Doob validation diagnostics."""
import hashlib
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from src.graph_metrics import metric_sample
from src.mated_crt import graph_from_minima


def matrix(records,family,width=None):
    if family in ("distance_cdf","degree_cdf","ball_fraction"):
        if family=="distance_cdf":
            curves=[np.asarray(r["distance_histogram"],float) for r in records]
            curves=[x.cumsum()/x.sum() for x in curves]
        elif family=="degree_cdf":
            curves=[np.asarray(r["degree"]["histogram"],float).cumsum()/r["n"] for r in records]
        else:
            curves=[np.asarray(r["mean_ball_volumes"],float)/r["n"] for r in records]
        width=max(map(len,curves)) if width is None else width
        return np.array([np.pad(x,(0,width-len(x)),constant_values=1.) for x in curves])
    out=[]
    for row in records:
        n=row["n"];minimum=np.asarray(row["cell_minima"])
        t=(np.arange(n)+.5)/n
        x=minimum[1:-1]/np.sqrt(t[1:-1]*(1-t[1:-1]))[:,None]
        if family=="marginal_minimum_cdf":
            out.append([np.mean(x[:,k]<=v) for k in range(x.shape[1]) for v in [.5,1.,1.5,2.]])
        elif family=="joint_minimum_cdf":
            out.append([np.mean((x[:,i]<=v)&(x[:,j]<=v))
                        for i,j in itertools.combinations(range(x.shape[1]),2) for v in [.5,1.,1.5,2.]])
        elif family=="adjacent_minimum_cdf":
            out.append([np.mean((x[:-1,k]<=v)&(x[1:,k]<=v)) for k in range(x.shape[1]) for v in [.5,1.,1.5,2.]])
        else:raise ValueError(family)
    return np.asarray(out)


def compare(left,right,seed=0,bootstrap=500):
    rng=np.random.default_rng(seed)
    result={"candidate_graphs":len(left),"reference_graphs":len(right),"families":{}}
    for family in ["distance_cdf","degree_cdf","ball_fraction","marginal_minimum_cdf", "joint_minimum_cdf","adjacent_minimum_cdf"]:
        L,R=matrix(left,family),matrix(right,family)
        if L.shape[1]!=R.shape[1]:
            width=max(L.shape[1],R.shape[1]);L=matrix(left,family,width);R=matrix(right,family,width)
        difference=L.mean(axis=0)-R.mean(axis=0)
        draws=[]
        for _ in range(bootstrap):
            delta=L[rng.integers(len(L),size=len(L))].mean(axis=0)-R[rng.integers(len(R),size=len(R))].mean(axis=0)
            draws.append(np.max(np.abs(delta-difference)))
        radius=float(np.quantile(draws,.95))
        observed=float(np.max(np.abs(difference)))
        result["families"][family]={"observed_sup_difference":observed,
            "centered_bootstrap_MC95_radius":radius,
            "difference_plus_MC95_radius":observed+radius,
            "signed_differences":difference.tolist(),
            "interpretation":"finite-CDF-grid error budget; systematic error is not certified"}
    for feature in ["distance_mean","degree_mean"]:
        x=np.array([r["features"][feature] for r in left]);y=np.array([r["features"][feature] for r in right])
        diff=float(x.mean()-y.mean());se=float(np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y)))
        result[feature]={"candidate_mean":float(x.mean()),"reference_mean":float(y.mean()),
            "difference":diff,"mc_se":se,"relative_difference":diff/float(y.mean()),
            "difference_mc95":[diff-1.96*se,diff+1.96*se]}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/phase2b/doob_error_budget_complete.json")
    args=parser.parse_args()
    paths=[Path("results/phase2b")/name/"records.jsonl" for name in
           ["doob_validation","doob_extra_controls","doob_one_factor"]]
    repeat=Path("results/phase2b/doob_tail_repeat/records.jsonl")
    if repeat.exists():paths.append(repeat)
    records=[]
    for path in paths:
        records.extend(dict(json.loads(line),source=str(path)) for line in path.read_text().splitlines())
    valid=[r for r in records if r["status"]=="success"]
    aliases=[]
    for row in valid:
        if row["case"]=="i3" and row["n"]==256:
            if row["configuration"] in ("fine","fine_repeat"):
                aliases.append(dict(row,configuration="fine_pooled"))
            if row["configuration"] in ("cutoff_refined","cutoff_repeat"):
                aliases.append(dict(row,configuration="cutoff_pooled"))
    valid.extend(aliases)
    comparisons=[]
    for case,n in [("i3",256),("a2",256),("i2",128),("a2_2d",128),("a2_permuted",128)]:
        reference=[r for r in valid if r["case"]==case and r["n"]==n and r["configuration"]=="reference"]
        configs=list(dict.fromkeys(r["configuration"] for r in valid if r["case"]==case and r["n"]==n and r["configuration"]!="reference"))
        for config in configs:
            candidate=[r for r in valid if r["case"]==case and r["n"]==n and r["configuration"]==config]
            comparisons.append(dict(case=case,n=n,configuration=config,reference="matching_law_special_case",
                **compare(candidate,reference,seed=330+len(comparisons))))
    # A2 label-permutation invariance: derive n128 graphs exactly by aggregating
    # the saved n256 cell minima, retaining the same physical realizations.
    original=[r for r in valid if r["case"]=="a2" and r["n"]==256 and r["configuration"]=="fine"]
    base=[]
    for row in original:
        minimum=np.asarray(row["cell_minima"]).reshape(128,2,3).min(axis=1)
        graph=graph_from_minima(minimum)
        metric=metric_sample(graph.adjacency,np.random.default_rng(row["seed"]+1),roots=8)
        base.append(dict(n=128,cell_minima=minimum.tolist(),distance_histogram=metric["distance_histogram"],
            mean_ball_volumes=metric["mean_ball_volumes"],degree=metric["degree"],
            features={"distance_mean":metric["mean_distance"],"degree_mean":metric["degree"]["mean"]}))
    perm=[]
    for row in valid:
        if row["case"]=="a2_permuted" and row["n"]==128 and row["configuration"]=="fine":
            perm.append(dict(row,cell_minima=np.asarray(row["cell_minima"])[:,[1,2,0]].tolist()))
    comparisons.append(dict(case="A2_coordinate_permutation",n=128,configuration="fine",reference="original_A2_n256_minima_aggregated_to_n128",**compare(perm,base,seed=777)))
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        bootstrap=500,failure_records=[r for r in records if r["status"]!="success"],comparisons=comparisons,
        pooled_comparisons="Fine/cutoff pooled aliases reuse original independent paths and must not be counted as added samples",
        scope="CIs quantify independent path Monte Carlo error per observable family; no uniform continuum-law guarantee"),indent=2))
    print("Saved",output)


if __name__=="__main__":main()
