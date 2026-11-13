"""Task2C isolated coupled minima and Brownian ball-clock experiments.

Run as ``python -m experiments.phase2c_path_validation --help``. One JSONL
checkpoint per independent physical realization; prior outputs are refused.
No covariance sweep or dimension fit is performed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import platform
import numpy as np
from src.brownian_excursions import sample_independent_excursion
from src.weyl_excursions import sample_weyl_excursion
from src.phase2c_paths import (thin_polygonal,ball_exit_laplace,coupled_gaussian_ball_walks,
    brownian_sphere_skeleton,ball_exit_cdf,ball_exit_quantile,sample_wos_doob)
from src.doob_excursions import isotropic_ball_walk
from src.mated_crt import cell_minima,graph_from_minima
from src.graph_metrics import metric_sample


TARGETS={"mean_distance_relative_bias_goal":.01,
         "joint_minimum_cdf_max_difference_goal":.02,
         "local_edges":"Report paired Jaccard; mean-distance agreement alone is insufficient",
         "uncertainty":"Independent physical-realization MC; nested resolutions and graph sizes are paired"}


def feature_record(sample,n,metric_seed):
    minima=cell_minima(sample,n)
    graph=graph_from_minima(minima)
    metric=metric_sample(graph.adjacency,np.random.default_rng(metric_seed),roots=8)
    root_hist=np.asarray(metric["root_histograms"])
    balls=root_hist.cumsum(axis=1)/n
    mid=(np.arange(n)+.5)/n
    normalized=minima[1:-1]/np.sqrt(mid[1:-1]*(1-mid[1:-1]))[:,None]
    feature={"mean_distance":metric["mean_distance"],
             "degree_mean":metric["degree"]["mean"],
             "degree_variance":metric["degree"]["variance"],
             "ties_fraction":float(np.mean(graph.metadata["symbolic_ties_by_coordinate"])/(n-2))}
    for r in [2,3,4,5,6]:
        b=balls[:,r] if balls.shape[1]>r else np.ones(len(balls))
        feature[f"ball_mean_r{r}"]=float(np.mean(b))
        feature[f"ball_root_variance_r{r}"]=float(np.var(b,ddof=1))
    for i in range(minima.shape[1]):
        feature[f"minimum_mean_{i}"]=float(normalized[:,i].mean())
        for t in [.5,1.,1.5,2.]:
            feature[f"minimum_cdf_{i}_{t}"]=float(np.mean(normalized[:,i]<=t))
        for j in range(i):
            for t in [.5,1.,1.5,2.]:
                feature[f"joint_cdf_{i}{j}_{t}"]=float(np.mean((normalized[:,i]<=t)&(normalized[:,j]<=t)))
    return {"features":feature,"minima":minima.tolist(),
            "edges":graph.edges.tolist(),"degree_histogram":metric["degree"]["histogram"],
            "root_histograms":metric["root_histograms"],
            "distance_histogram":metric["distance_histogram"],"roots":metric["roots"]}


def _padded_difference_hist(a,b):
    width=max(len(a),len(b))
    aa=np.pad(np.asarray(a,float),(0,width-len(a)));bb=np.pad(np.asarray(b,float),(0,width-len(b)))
    return aa/aa.sum()-bb/bb.sum()


def paired_difference(a,b):
    result={key:a["features"][key]-b["features"][key] for key in a["features"]}
    ea={tuple(e) for e in a["edges"]};eb={tuple(e) for e in b["edges"]}
    result["edge_jaccard"]=len(ea&eb)/len(ea|eb)
    result["distance_histogram_TV"]=float(abs(_padded_difference_hist(a["distance_histogram"],b["distance_histogram"])).sum()/2)
    ma,mb=np.asarray(a["minima"]),np.asarray(b["minima"])
    result["minimum_coordinate_max_absolute"]=float(abs(ma-mb).max())
    result["minimum_coordinate_rms"]=float(np.sqrt(np.mean((ma-mb)**2)))
    result["mean_distance_relative"]=(a["features"]["mean_distance"]-b["features"]["mean_distance"])/b["features"]["mean_distance"]
    return result


def summarize_minima(records):
    groups=[]
    for case in sorted({r["case"] for r in records}):
        selected=[r for r in records if r["case"]==case]
        for n in selected[0]["sizes"]:
            refkey="exact" if case=="i3" else str(selected[0]["fine"])
            for q in selected[0]["q"]:
                differences=[paired_difference(r["results"][str(n)][str(q)],r["results"][str(n)][refkey]) for r in selected]
                keys=differences[0]
                stats={k:{"mean":float(np.mean([d[k] for d in differences])),
                          "se":float(np.std([d[k] for d in differences],ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else None} for k in keys}
                groups.append({"case":case,"n":n,"q":q,"reference":refkey,"independent_paths":len(selected),"paired_differences":stats})
    return {"targets":TARGETS,"groups":groups,"warning":"Fine A2/A3 grids are not exact continuous minima; reference residual remains systematic."}


def minima_experiment(args,out):
    records=[]
    for case in args.cases:
        for i in range(args.samples):
            seed=args.seed+100000*args.cases.index(case)+i
            rng=np.random.default_rng(seed);start=time.perf_counter()
            if case=="i3":
                sample=sample_independent_excursion(max(args.n),3,rng,law="biased",oversample=args.fine)
            elif case=="a2":
                sample=sample_weyl_excursion(max(args.n),3,rng,law="biased",oversample=args.fine)
            else:
                from src.weyl_a3 import sample_a3_excursion
                sample=sample_a3_excursion(max(args.n),rng,law="biased",oversample=args.fine)
            row={"case":case,"realization":i,"seed":seed,"fine":args.fine,"q":args.q,"sizes":args.n,
                 "sampler_metadata":sample.metadata,"results":{}}
            for n in args.n:
                metric_seed=seed+90000000+n
                nested={}
                if case=="i3":
                    nested["exact"]=feature_record(sample,n,metric_seed)
                for q in sorted(set(args.q+[args.fine])):
                    nested[str(q)]=feature_record(thin_polygonal(sample,n,q),n,metric_seed)
                row["results"][str(n)]=nested
            row["seconds"]=time.perf_counter()-start
            with (out/"records.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            records.append(row)
            (out/"summary.json").write_text(json.dumps(summarize_minima(records),indent=2))
            print(json.dumps({"case":case,"realization":i,"seconds":row["seconds"]}),flush=True)


def summarize_clocks(records):
    results=[]
    for method in ["bounded_sphere","gaussian_nested"]:
        for scale in records[0][method]:
            clocks=np.array([r[method][scale] for r in records])
            entry={"method":method,"relative_step":float(scale),"samples":len(clocks),
                   "mean":float(clocks.mean()),"exact_mean":1/3,
                   "mean_se":float(clocks.std(ddof=1)/np.sqrt(len(clocks))) if len(clocks)>1 else None,
                   "relative_mean_bias":float(3*clocks.mean()-1),
                   "variance":float(clocks.var(ddof=1)) if len(clocks)>1 else None,"exact_variance":2/45,
                   "laplace":{str(s):{"mean":float(np.exp(-s*clocks).mean()),"exact":ball_exit_laplace(s)} for s in [1.,3.,10.]}}
            if method=="gaussian_nested":
                fine=min(map(float,records[0][method]))
                diff=clocks-np.array([r[method][str(fine)] for r in records])
                entry["paired_minus_finest_mean"]=float(diff.mean())
                entry["paired_minus_finest_se"]=float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else None
            results.append(entry)
    return {"exact_target":"3D Brownian ball exit from center, r=1, generator Delta/2", "groups":results,
            "warning":"Gaussian values are exact on nested grids, exits remain polygonal approximations; spherical scales are independent within a row."}


def clocks_experiment(args,out):
    records=[]
    scales=(.2,.1,.05,.025)
    for i in range(args.samples):
        seed=args.seed+i;start=time.perf_counter()
        row={"seed":seed,"realization":i,"bounded_sphere":{},"gaussian_nested":{}}
        rng=np.random.default_rng(seed)
        for scale in scales:
            _,times,_=isotropic_ball_walk(1.,3,rng,step_size=scale,relative_step=.99)
            row["bounded_sphere"][str(scale)]=float(times[-1])
        gaussian=coupled_gaussian_ball_walks(1.,3,rng,relative_steps=scales)
        row["gaussian_nested"]={str(scale):float(gaussian[scale][1][-1]) for scale in scales}
        row["seconds"]=time.perf_counter()-start
        with (out/"records.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        records.append(row)
        if (i+1)%128==0 or i+1==args.samples:
            (out/"summary.json").write_text(json.dumps(summarize_clocks(records),indent=2))
            print(json.dumps({"samples":i+1,"elapsed_seconds":sum(r["seconds"] for r in records)}),flush=True)


def sphere_clocks_experiment(args,out):
    rows=[]
    # Deterministic independent probes between interpolation knots.
    u=1/(1+np.exp(-np.linspace(-30.001,30.003,100003)))
    cdf_error=float(np.max(abs(ball_exit_cdf(ball_exit_quantile(u))-u)))
    for i in range(args.samples):
        start=time.perf_counter();seed=args.seed+i
        _,times,info=brownian_sphere_skeleton(1.,3,np.random.default_rng(seed),step_size=args.step_size)
        row={"seed":seed,"clock":float(times[-1]),"steps":info["steps"],
             "omitted_time_mean_bound":info["omitted_time_mean_bound"],"seconds":time.perf_counter()-start}
        with (out/"records.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        rows.append(row)
        if (i+1)%256==0 or i+1==args.samples:
            x=np.array([r["clock"] for r in rows])
            (out/"summary.json").write_text(json.dumps({"samples":len(x),"mean":float(x.mean()),
                "mean_se":float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
                "relative_mean_bias":float(3*x.mean()-1),"exact_mean":1/3,
                "variance":float(x.var(ddof=1)) if len(x)>1 else None,"exact_variance":2/45,
                "max_numerical_inverse_cdf_probability_defect":cdf_error,
                "cdf_probe_points":len(u),"total_seconds":sum(r["seconds"] for r in rows),
                "laplace":{str(s):{"mean":float(np.exp(-s*x).mean()),"exact":ball_exit_laplace(s)} for s in [1.,3.,10.]},
                "max_omitted_clock_mean_bound":max(r["omitted_time_mean_bound"] for r in rows)},indent=2))
            print(json.dumps({"samples":i+1,"seconds":sum(r["seconds"] for r in rows)}),flush=True)


def path_probes(sample):
    ts=np.array([.25,.5,.75])
    values=np.column_stack([np.interp(ts,sample.times,sample.values[:,i]) for i in range(3)])
    C=np.asarray(sample.metadata["covariance"])
    white=np.linalg.solve(np.linalg.cholesky(C),values.T).T
    features={f"radius_square_t{t}":float(x@x) for t,x in zip(ts,white)}
    features["multitime_dot_quarter_threequarter"]=float(white[0]@white[2])
    for j,t in enumerate(ts):
        for k in range(3):
            features[f"coordinate{k}_t{t}"]=float(values[j,k])
    return {"times":ts.tolist(),"values":values.tolist(),"features":features}


def independent_feature_difference(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    diff=x.mean()-y.mean()
    se=np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y)) if len(x)>1 and len(y)>1 else float("nan")
    return {"difference":float(diff),"mc_se":float(se),"mc95":[float(diff-1.96*se),float(diff+1.96*se)]}


def summarize_wos(records,reference):
    success=[r for r in records if r["status"]=="success"]
    result={"targets":TARGETS,"failures":[r for r in records if r["status"]!="success"],"comparisons":[]}
    for case in sorted({r["case"] for r in success}):
        group=[r for r in success if r["case"]==case]
        refs=[r for r in reference if r["case"]==case]
        for n in group[0]["sizes"]:
            reference_key="exact" if case=="i3" else str(refs[0]["fine"])
            common=group[0]["results"][str(n)]["features"]
            comparison={key:independent_feature_difference(
                [r["results"][str(n)]["features"][key] for r in group],
                [r["results"][str(n)][reference_key]["features"][key] for r in refs]) for key in common}
            baseline=np.mean([r["results"][str(n)][reference_key]["features"]["mean_distance"] for r in refs])
            path={key:independent_feature_difference([r["path"]["features"][key] for r in group],
                [r["path_reference"]["features"][key] for r in group]) for key in group[0]["path"]["features"]}
            result["comparisons"].append({"case":case,"n":n,"paths":len(group),"reference_paths":len(refs),
                "mean_distance_reference":float(baseline),
                "mean_distance_wos":float(np.mean([r["results"][str(n)]["features"]["mean_distance"] for r in group])),
                "mean_distance_relative_difference":comparison["mean_distance"]["difference"]/baseline,
                "mean_distance_relative_mc95":(np.array(comparison["mean_distance"]["mc95"])/baseline).tolist(),
                "feature_differences":comparison,"path_feature_differences":path})
    return result


def wos_extra_diagnostics(records):
    """Radial conditional moments remove variation from random leg clocks.

    Given T1,T2 and S, each leg's radius is a BES(2*alpha+2) bridge between
    zero and height H=1/sqrt(T1+T2). Its squared-radius conditional mean is
    the squared linear mean plus dimension times scalar bridge variance.
    """
    from scipy.stats import kstest,invgamma
    result=[]
    success=[r for r in records if r["status"]=="success"]
    for case in dict.fromkeys(r["case"] for r in success):
        group=[r for r in success if r["case"]==case]
        alpha=group[0]["sampler_metadata"]["alpha1"]
        clocks=np.array([r["sampler_metadata"]["hitting_times"] for r in group]).ravel()
        ks=kstest(clocks,invgamma(alpha,scale=.5).cdf)
        radial={}
        for t in [.25,.5,.75]:
            residual=[]
            for r in group:
                m=r["sampler_metadata"];total=sum(m["hitting_times"]);u=m["junction_time"];d=2*alpha+2
                expected=(t/u)**2/total+d*t*(u-t)/u if t<=u else ((1-t)/(1-u))**2/total+d*(t-u)*(1-t)/(1-u)
                residual.append(r["path"]["features"][f"radius_square_t{t}"]-expected)
            radial[str(t)]={"mean_residual":float(np.mean(residual)),
                "se":float(np.std(residual,ddof=1)/np.sqrt(len(residual))) if len(residual)>1 else None}
        boundary_bounds=[sum(d["boundary_omitted_time_conditional_mean_bound_sum"] for d in r["sampler_metadata"]["leg_diagnostics"])/sum(r["sampler_metadata"]["hitting_times"]) for r in group]
        tails=[max(d["tail_time"] for d in r["sampler_metadata"]["leg_diagnostics"])/sum(r["sampler_metadata"]["hitting_times"]) for r in group]
        result.append({"case":case,"physical_paths":len(group),"hitting_time_KS":float(ks.statistic),"hitting_time_KS_pvalue":float(ks.pvalue),
            "clock_mean":float(clocks.mean()),"clock_exact_mean":1/(2*(alpha-1)),"conditional_radial_moments":radial,
            "max_normalized_boundary_clock_mean_bound_sum":max(boundary_bounds),
            "max_normalized_cone_tail_time":max(tails),"generation_seconds":sum(r["seconds"] for r in group)})
    return {"groups":result,"warning":"Pointwise diagnostics, no simultaneous multiple-testing guarantee; tails/path projection remain approximations."}


def wos_experiment(args,out):
    if not args.reference:
        raise ValueError("--reference requires existing nested-control records.jsonl")
    reference=[json.loads(line) for line in Path(args.reference).read_text().splitlines()]
    records=[]
    for case in args.cases:
        for i in range(args.samples):
            seed=args.seed+100000*args.cases.index(case)+i;start=time.perf_counter()
            row={"case":case,"realization":i,"seed":seed,"sizes":args.n}
            try:
                if case=="a3":
                    from src.weyl_a3 import A3AnalyticCone,sample_a3_excursion
                    model=A3AnalyticCone()
                else:
                    from src.doob_excursions import analytic_cone
                    from src.weyl_excursions import weyl_covariance
                    model=analytic_cone(np.eye(3) if case=="i3" else weyl_covariance())
                sample=sample_wos_doob(max(args.n),rng=np.random.default_rng(seed),model=model,
                    step_size=args.step_size,relative_step=args.relative_step,trace_balls=args.trace_balls)
                rrng=np.random.default_rng(seed+44000000)
                if case=="i3":
                    control=sample_independent_excursion(max(args.n),3,rrng,law="biased")
                elif case=="a2":
                    control=sample_weyl_excursion(max(args.n),3,rrng,law="biased",oversample=1)
                else:
                    control=sample_a3_excursion(max(args.n),rrng,law="biased",oversample=1)
                row.update(status="success",sampler_metadata=sample.metadata,path=path_probes(sample),
                    path_reference=path_probes(control),results={str(n):feature_record(sample,n,seed+90000000+n) for n in args.n})
                if args.coarsen_pl:
                    row["coarsened_polygonal"]={}
                    for factor in [2,4,8]:
                        keep=np.arange(0,len(sample.times),factor)
                        required=np.unique(np.r_[np.linspace(0,1,max(args.n)+1),sample.metadata["junction_time"]])
                        indices=np.searchsorted(sample.times,required)
                        # Graph cell boundaries were inserted exactly.
                        indices=np.minimum(indices,len(sample.times)-1)
                        before=np.maximum(0,indices-1)
                        indices=np.where(abs(sample.times[before]-required)<abs(sample.times[indices]-required),before,indices)
                        keep=np.unique(np.r_[keep,indices,len(sample.times)-1])
                        from src.brownian_excursions import ExcursionSample
                        coarse=ExcursionSample(sample.times[keep],sample.values[keep],None,
                            dict(sample.metadata,coarsening_factor=factor,
                                warning="Same fine WOS path thinned, NOT a coarser-WOS sampler law"))
                        row["coarsened_polygonal"][str(factor)]={str(n):feature_record(coarse,n,seed+90000000+n) for n in args.n}
                    if i<args.save_fine_paths:
                        pathfile=out/f"{case}_{i:03d}_finest_path.npz"
                        np.savez_compressed(pathfile,times=sample.times,values=sample.values)
                        row["saved_finest_path"]=str(pathfile)
            except Exception as error:
                row.update(status="failure",error=repr(error))
            row["seconds"]=time.perf_counter()-start
            records.append(row)
            with (out/"records.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            (out/"summary.json").write_text(json.dumps(summarize_wos(records,reference),indent=2))
            (out/"extra_diagnostics.json").write_text(json.dumps(wos_extra_diagnostics(records),indent=2))
            if args.coarsen_pl:
                comparisons=[]
                success=[r for r in records if r["status"]=="success"]
                for label in dict.fromkeys(r["case"] for r in success):
                    group=[r for r in success if r["case"]==label]
                    for n in args.n:
                        for factor in [2,4,8]:
                            differences=[paired_difference(r["coarsened_polygonal"][str(factor)][str(n)],r["results"][str(n)]) for r in group]
                            stats={key:{"mean":float(np.mean([d[key] for d in differences])),
                                "se":float(np.std([d[key] for d in differences],ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else None} for key in differences[0]}
                            comparisons.append({"case":label,"n":n,"thinning_factor":factor,"physical_paths":len(group),"paired_differences":stats})
                (out/"coarsening_summary.json").write_text(json.dumps({"warning":"PL interpolation diagnostic, NOT coarse-WOS law or full discretization error bound","comparisons":comparisons},indent=2))
            print(json.dumps({"case":case,"realization":i,"status":row["status"],"seconds":row["seconds"]}),flush=True)


def make_figures(root="results/phase2c/paths"):
    os.environ.setdefault("MPLCONFIGDIR","/private/tmp/lqg-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root);output=root/"figures";output.mkdir(exist_ok=True)
    nested=json.loads((root/"nested_controls/summary.json").read_text())
    clocks=json.loads((root/"ball_clocks/summary.json").read_text())
    sphere=json.loads((root/"sphere_clock_control/summary.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for case in ["i3","a2","a3"]:
        groups=[g for g in nested["groups"] if g["case"]==case and g["n"]==256]
        q=[g["q"] for g in groups]
        axes[0].plot(q,[1-g["paired_differences"]["edge_jaccard"]["mean"] for g in groups],"o-",label=case.upper())
        x=[g["paired_differences"]["mean_distance_relative"]["mean"] for g in groups]
        se=[g["paired_differences"]["mean_distance_relative"]["se"] for g in groups]
        axes[1].errorbar(q,100*np.array(x),yerr=196*np.array(se),fmt="o-",label=case.upper())
    axes[0].set(xscale="log",yscale="log",xlabel="Grid points per cell q",ylabel="1 − paired edge Jaccard",title="Local edges remain sensitive (n=256)")
    axes[1].set(xscale="log",xlabel="Grid points per cell q",ylabel="Paired mean-distance change (%)",title="Distances hide local changes; MC95")
    axes[1].axhline(0,color="gray",lw=.8)
    for method,style in [("bounded_sphere","o-"),("gaussian_nested","s--")]:
        groups=[g for g in clocks["groups"] if g["method"]==method]
        axes[2].errorbar([g["relative_step"] for g in groups],
            [100*g["relative_mean_bias"] for g in groups],
            yerr=[100*3*1.96*g["mean_se"] for g in groups],fmt=style,label=method.replace("_"," "))
    axes[2].errorbar([.1],[100*sphere["relative_mean_bias"]],yerr=[196*3*sphere["mean_se"]],fmt="D",label="Exact sub-ball clocks")
    axes[2].axhline(0,color="gray",lw=.8)
    axes[2].set(xlabel="Step / outer radius",ylabel="Mean exit-time error (%)",title="Clipped clocks have first-order bias")
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/"coupled_minima_and_clock.png",dpi=180);fig.savefig(output/"coupled_minima_and_clock.pdf");plt.close(fig)
    if (root/"wos_controls/summary.json").exists():
        wos=json.loads((root/"wos_controls/summary.json").read_text())
        fig,ax=plt.subplots(figsize=(7,3.8))
        for j,n in enumerate([64,256]):
            groups=[g for g in wos["comparisons"] if g["n"]==n]
            groups.sort(key=lambda x:["i3","a2","a3"].index(x["case"]))
            centers=np.array([g["mean_distance_relative_difference"] for g in groups])*100
            lows=np.array([g["mean_distance_relative_mc95"][0] for g in groups])*100
            highs=np.array([g["mean_distance_relative_mc95"][1] for g in groups])*100
            ax.errorbar(np.arange(len(groups))+(j-.5)*.15,centers,yerr=[centers-lows,highs-centers],fmt="o",label=f"n={n}, graph MC95")
        ax.axhspan(-1,1,alpha=.12,color="green",label="±1% accuracy goal")
        ax.axhline(0,color="gray",lw=.8);ax.set(xticks=[0,1,2],xticklabels=["I3","A2 × half-line","A3"],ylabel="WOS − reference mean distance (%)",
            title="Exact-clock Doob controls: uncertainty still matters")
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(output/"wos_graph_control.png",dpi=180);fig.savefig(output/"wos_graph_control.pdf");plt.close(fig)
    fine=root/"finer_wos_interpolation/coarsening_summary.json"
    if fine.exists():
        coarsening=json.loads(fine.read_text())
        fig,axes=plt.subplots(1,2,figsize=(9,3.6))
        for case in ["i3","a3"]:
            groups=[g for g in coarsening["comparisons"] if g["case"]==case]
            factor=[g["thinning_factor"] for g in groups]
            axes[0].plot(factor,[g["paired_differences"]["edge_jaccard"]["mean"] for g in groups],"o-",label=case.upper())
            means=[100*g["paired_differences"]["mean_distance_relative"]["mean"] for g in groups]
            errors=[196*g["paired_differences"]["mean_distance_relative"]["se"] for g in groups]
            axes[1].errorbar(factor,means,yerr=errors,fmt="o-",label=case.upper())
        axes[0].set(xlabel="Retain every kth stopping vertex",ylabel="Paired edge Jaccard",title="Same WOS paths, coarser interpolation")
        axes[1].set(xlabel="Retain every kth stopping vertex",ylabel="Mean-distance change (%)",title="Paired graph MC95; n=256")
        axes[1].axhline(0,color="gray",lw=.8)
        for ax in axes:
            ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(output/"wos_paired_polygonal_coarsening.png",dpi=180);fig.savefig(output/"wos_paired_polygonal_coarsening.pdf");plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["minima","clocks","sphere_clocks","wos"])
    p.add_argument("--output",required=True)
    p.add_argument("--samples",type=int,default=8)
    p.add_argument("--seed",type=int,default=21731001)
    p.add_argument("--n",type=int,nargs="+",default=[64,256])
    p.add_argument("--q",type=int,nargs="+",default=[4,16,64,256])
    p.add_argument("--fine",type=int,default=1024)
    p.add_argument("--cases",nargs="+",choices=["i3","a2","a3"],default=["i3","a2","a3"])
    p.add_argument("--step-size",type=float,default=.005)
    p.add_argument("--relative-step",type=float,default=.1)
    p.add_argument("--reference")
    p.add_argument("--trace-balls",action="store_true")
    p.add_argument("--coarsen-pl",action="store_true")
    p.add_argument("--save-fine-paths",type=int,default=2)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        p.error("Output directory already exists; preserve previous checkpoints with a fresh name")
    out.mkdir(parents=True)
    (out/"metadata.json").write_text(json.dumps({"arguments":vars(args),"accuracy_targets":TARGETS,
        "python":platform.python_version(),"numpy":np.__version__,"rng":"numpy.PCG64",
        "path_bias_law":"Budd-Castro biased, never unbiased substituted",
        "cost_policy":"pilot first, total expected <15min per invocation, one realization/checkpoint"},indent=2))
    {"minima":minima_experiment,"clocks":clocks_experiment,
     "sphere_clocks":sphere_clocks_experiment,"wos":wos_experiment}[args.mode](args,out)


if __name__=="__main__":
    main()
