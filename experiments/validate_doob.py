"""Independent controls for approximate biased Doob excursions.

Compare path, joint-minimum and local/global graph observables, with independent
graph Monte Carlo errors. This does not fit Hausdorff dimensions. Failures are
saved per attempt and are never replaced with successful draws.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lqg-matplotlib")
import numpy as np
from scipy.stats import invgamma,kstest

from src.brownian_excursions import sample_independent_excursion
from src.weyl_excursions import sample_weyl_excursion,thin_weyl_sample,weyl_covariance
from src.doob_excursions import sample_doob_excursion,analytic_cone
from src.mated_crt import cell_minima,graph_from_excursion,validate_graph
from src.graph_metrics import metric_sample
from src.provenance import environment_manifest


def cases():
    a2=weyl_covariance(3)
    perm=[2,0,1]
    positive=np.eye(3);positive[0,1]=positive[1,0]=.5
    return {"i3":(np.eye(3),"independent"),"a2":(a2,"weyl"),
            "i2":(np.eye(2),"independent"),"a2_2d":(weyl_covariance(2),"weyl"),
            "a2_permuted":(a2[np.ix_(perm,perm)],"weyl_permuted"),
            "positive_pair":(positive,None)}


def reference_sample(n,C,kind,rng):
    if kind=="independent":
        return sample_independent_excursion(n,len(C),rng,law="biased")
    if kind in ("weyl","weyl_permuted"):
        sample=sample_weyl_excursion(n,len(C),rng,law="biased",oversample=256)
        if kind=="weyl_permuted":
            permutation=[2,0,1]
            sample.values=sample.values[:,permutation]
            direction=np.asarray(sample.metadata["junction_direction_correlated"])[permutation]
            sample.metadata=dict(sample.metadata,covariance=C.tolist(),coordinate_permutation=permutation,
                junction_direction_correlated=direction.tolist(),junction_permutation_applied=True)
        return sample
    raise ValueError("No independently accessible reference for this case")


def observables(sample,n,metric_seed):
    C=np.asarray(sample.metadata["covariance"])
    d=len(C)
    minimum=cell_minima(sample,n)
    graph=graph_from_excursion(sample,n)
    metrics=metric_sample(graph.adjacency,np.random.default_rng(metric_seed),roots=8)
    times=np.array([.25,.5,.75])
    values=np.column_stack([np.interp(times,sample.times,sample.values[:,k]) for k in range(d)])
    whitened=np.linalg.solve(np.linalg.cholesky(C),values.T).T
    tcell=(np.arange(n)+.5)/n
    normalized=minimum[1:-1]/np.sqrt(tcell[1:-1]*(1-tcell[1:-1]))[:,None]
    features={"distance_mean":metrics["mean_distance"],
              "degree_mean":metrics["degree"]["mean"],
              "degree_variance":metrics["degree"]["variance"],
              "parallel_edge_fraction":float(np.mean(graph.multiplicities>1)),
              "PL_tie_fraction":float(np.mean(graph.metadata["symbolic_ties_by_coordinate"])/(n-2))}
    for j,t in enumerate(times):
        features[f"path_radius_squared_t{t}"]=float(np.sum(whitened[j]**2))
        for k in range(d):
            features[f"path_coordinate{k}_t{t}"]=float(values[j,k])
    for k in range(d):
        features[f"minimum_mean_{k}"]=float(normalized[:,k].mean())
        features[f"minimum_square_{k}"]=float((normalized[:,k]**2).mean())
        features[f"minimum_adjacent_product_{k}"]=float((normalized[1:,k]*normalized[:-1,k]).mean())
        for threshold in [.5,1.,1.5,2.]:
            features[f"minimum_cdf_{k}_{threshold}"]=float(np.mean(normalized[:,k]<=threshold))
    for i,j in itertools.combinations(range(d),2):
        features[f"joint_minimum_product_{i}{j}"]=float(np.mean(normalized[:,i]*normalized[:,j]))
        for threshold in [.5,1.,1.5,2.]:
            features[f"joint_minimum_cdf_{i}{j}_{threshold}"]=float(np.mean(
                (normalized[:,i]<=threshold)&(normalized[:,j]<=threshold)))
    histogram=np.asarray(metrics["degree"]["histogram"])
    for degree in [2,4,6,8,10]:
        features[f"degree_cdf_{degree}"]=float(histogram[:degree+1].sum()/n)
    for r in range(1,7):
        features[f"ball_fraction_r{r}"]=float(metrics["mean_ball_volumes"][r]/n) if r<len(metrics["mean_ball_volumes"]) else 1.
    spans=np.minimum(graph.edges[:,1]-graph.edges[:,0],n-(graph.edges[:,1]-graph.edges[:,0]))/n
    for threshold in [.02,.05,.1,.2]:
        features[f"edge_span_cdf_{threshold}"]=float(np.mean(spans<=threshold))
    return dict(features=features,cell_minima=minimum.tolist(),path_probe_times=times.tolist(),
                path_probe_values=values.tolist(),distance_histogram=metrics["distance_histogram"],
                mean_ball_volumes=metrics["mean_ball_volumes"],degree=metrics["degree"],
                graph_validation=validate_graph(graph),graph_metadata=graph.metadata,
                roots=metrics["roots"])


def summary(records,output,seed=552190):
    rng=np.random.default_rng(seed)
    success=[r for r in records if r["status"]=="success"]
    result={"groups":[],"comparisons":[],"failures":[r for r in records if r["status"]!="success"],
            "interpretation":"validation finite-observable errors, no dimension fits",
            "uncertainty":"graph-realization MC only, distinct from discretization and omitted-tail errors"}
    for case in dict.fromkeys(r["case"] for r in records):
        for config in dict.fromkeys(r["configuration"] for r in records if r["case"]==case):
            for n in sorted({r["n"] for r in success if r["case"]==case and r["configuration"]==config}):
                group=[r for r in success if r["case"]==case and r["configuration"]==config and r["n"]==n]
                if not group:
                    continue
                features={k:{"mean":float(np.mean([r["features"][k] for r in group])),
                             "se":float(np.std([r["features"][k] for r in group],ddof=1)/np.sqrt(len(group)))}
                          for k in group[0]["features"]}
                hit=np.array([r["sampler_metadata"]["hitting_times"] for r in group]).ravel()
                alpha=group[0]["sampler_metadata"]["alpha1"]
                ks=kstest(hit,invgamma(alpha,scale=.5).cdf)
                result["groups"].append(dict(case=case,configuration=config,n=n,graphs=len(group),features=features,
                    hitting_time_mean=float(hit.mean()),expected_hitting_time_mean=1/(2*(alpha-1)),
                    hitting_time_KS={"statistic":float(ks.statistic),"pvalue":float(ks.pvalue)},
                    generation_seconds_mean=float(np.mean([r["generation_seconds"] for r in group]))))
        for n in sorted({r["n"] for r in success if r["case"]==case}):
            reference=[r for r in success if r["case"]==case and r["configuration"]=="reference" and r["n"]==n]
            if not reference:
                continue
            for config in dict.fromkeys(r["configuration"] for r in success if r["case"]==case and r["configuration"]!="reference"):
                group=[r for r in success if r["case"]==case and r["configuration"]==config and r["n"]==n]
                if not group:
                    continue
                comparison={}
                for feature in group[0]["features"]:
                    x=np.array([r["features"][feature] for r in group]); y=np.array([r["features"][feature] for r in reference])
                    difference=x.mean()-y.mean()
                    se=np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
                    # Normal graph-level intervals are only diagnostics; complete
                    # raw graph features are retained for nonparametric reanalysis.
                    comparison[feature]=dict(difference=float(difference),mc_se=float(se),
                        difference_mc95=[float(difference-1.96*se),float(difference+1.96*se)],
                        standardized_difference=float(difference/se) if se>0 else None)
                result["comparisons"].append(dict(case=case,configuration=config,n=n,features=comparison))
    Path(output).write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/phase2b/doob_validation")
    parser.add_argument("--cases",nargs="+",default=["i3","a2"])
    parser.add_argument("--graphs",type=int,default=64)
    parser.add_argument("--coarse",type=int)
    parser.add_argument("--reference",type=int,default=256)
    parser.add_argument("--veryfine",type=int,default=32)
    parser.add_argument("--sizes",nargs="+",type=int,default=[64,256])
    parser.add_argument("--seed",type=int,default=2026111041)
    parser.add_argument("--one-factor",action="store_true")
    parser.add_argument("--tail-repeat",action="store_true")
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    path=output/"records.jsonl"
    if path.exists():raise FileExistsError(path)
    configs=[("coarse",args.graphs if args.coarse is None else args.coarse,dict(step_size=.02,relative_step=.2,origin_cutoff=.01)),
             ("fine",args.graphs,dict(step_size=.005,relative_step=.1,origin_cutoff=.003)),
             ("veryfine",args.veryfine,dict(step_size=.0025,relative_step=.05,origin_cutoff=.0015)),
             ("reference",args.reference,{})]
    if args.one_factor:
        configs=[("relative_refined",args.graphs,dict(step_size=.005,relative_step=.05,origin_cutoff=.003)),
                 ("cutoff_refined",args.graphs,dict(step_size=.005,relative_step=.1,origin_cutoff=.0015))]
    if args.tail_repeat:
        configs=[("fine_repeat",args.graphs,dict(step_size=.005,relative_step=.1,origin_cutoff=.003)),
                 ("cutoff_repeat",args.graphs,dict(step_size=.005,relative_step=.1,origin_cutoff=.0015))]
    meta=dict(provenance=environment_manifest(),seed=args.seed,cases=args.cases,configurations=configs,
              sizes=args.sizes,within_realization_sizes="coupled same physical path; not independent observations",
              reference_weyl_oversample=256,reference_law="biased")
    (output/"metadata.json").write_text(json.dumps(meta,indent=2))
    records=[];nmax=max(args.sizes)
    with path.open("w") as stream:
        for case_index,case in enumerate(args.cases):
            C,kind=cases()[case]
            for config_index,(config,count,parameters) in enumerate(configs):
                if config=="reference" and kind is None:continue
                start=time.perf_counter()
                failures=0
                for realization in range(count):
                    seed=int(np.random.SeedSequence([args.seed,case_index,config_index,realization]).generate_state(1)[0])
                    rng=np.random.default_rng(seed)
                    begun=time.perf_counter()
                    try:
                        sample=reference_sample(nmax,C,kind,rng) if config=="reference" else sample_doob_excursion(nmax,C,rng,**parameters)
                        generated=time.perf_counter()-begun
                        for n in args.sizes:
                            row=dict(case=case,configuration=config,realization=realization,seed=seed,n=n,
                                     covariance=C.tolist(),status="success",generation_seconds=generated,
                                     sampler_metadata=sample.metadata)
                            row.update(observables(sample,n,seed+1))
                            records.append(row);stream.write(json.dumps(row)+"\n");stream.flush()
                    except Exception as exc:
                        failures+=1
                        row=dict(case=case,configuration=config,realization=realization,seed=seed,n=nmax,
                                 covariance=C.tolist(),status="failed",error=repr(exc),
                                 generation_seconds=time.perf_counter()-begun)
                        records.append(row);stream.write(json.dumps(row)+"\n");stream.flush()
                print(json.dumps(dict(case=case,configuration=config,attempts=count,failures=failures,
                    seconds=time.perf_counter()-start)),flush=True)
    result=summary(records,output/"summary.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for axis,feature,label in zip(axes,["distance_mean","degree_mean","joint_minimum_product_01"],
                                   ["Mean pair distance","Mean simple degree","Normalized joint minimum product"]):
        for case in args.cases:
            groups=[r for r in result["groups"] if r["case"]==case and r["n"]==nmax]
            axis.errorbar([r["configuration"] for r in groups],
                          [r["features"][feature]["mean"] for r in groups],
                          yerr=[1.96*r["features"][feature]["se"] for r in groups],fmt="o-",label=case)
        axis.set(ylabel=label,title=f"n={nmax}; independent graph MC 95% errors")
        axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/"comparison.png",dpi=180);plt.close(fig)


if __name__=="__main__":main()
