"""Saved-data finite-size, normalization and higher-point diagnostics."""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from experiments.phase3a_compare import (field_groups,graph_groups,save_json)
from src.metric_measure_3a import summarize,discrepancy,ensemble_median,RADII


def archive_graphs():
    paths=[Path("results/data/i3_biased_combined.jsonl"),Path("results/phase2b/data/i3_high_extension.jsonl")]
    groups={};seen=set()
    for path in paths:
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r["graph_seed"] in seen:raise ValueError("Duplicate archived physical graph")
            seen.add(r["graph_seed"])
            hist=np.asarray(r["root_histograms"],float)
            groups.setdefault(r["n"],[]).append({"grid":np.arange(hist.shape[1]),
                "cdfs":hist.cumsum(axis=1)/r["n"],"root_means":r["root_mean_distances"],
                "matrix":None,"discrete":True})
    return groups


def matrix_correlations(records):
    moments=[]
    for r in records:
        d=np.asarray(r["matrix"]);k=len(d);pairs=d[np.triu_indices(k,1)]
        shared=np.mean([d[i,j]*d[i,l] for i in range(k) for j,l in combinations([a for a in range(k) if a!=i],2)])
        disjoint=np.mean([v for a,b,c,e in combinations(range(k),4)
            for v in [d[a,b]*d[c,e],d[a,c]*d[b,e],d[a,e]*d[b,c]]])
        moments.append([pairs.mean(),(pairs**2).mean(),shared,disjoint])
    m=np.mean(moments,axis=0);var=m[1]-m[0]**2
    return {"shared_vertex_distance_correlation":float((m[2]-m[0]**2)/var),
            "disjoint_pair_distance_correlation":float((m[3]-m[0]**2)/var),
            "note":"Equal-realization 8-point matrix moments; finite-root and graph/field variation both present"}


def compact_summary(records):
    s=summarize(records)
    # Root-specific apparent slopes in a declared finite normalized-radius window.
    y=s["balls"][:,:,:3]
    slopes=np.array([np.polyfit(np.log(RADII[:3]),np.log(v),1)[0]
        for v in y.reshape(-1,3) if np.all(v>0)])
    result={k:s[k] for k in ["scale","ball_sd","ball_within_root_variance","ball_between_realization_variance",
        "triangle_small_mean","triangle_middle_mean","zero_distance_atom","scale_cv","tail_mass_at4"]}
    result["ball_slope_diagnostic"]={"radius_window":RADII[:3].tolist(),"root_quantiles":np.quantile(slopes,[.1,.5,.9]) if len(slopes) else [],
        "warning":"Finite-radius mass growth, not an asymptotic Hausdorff dimension"}
    result["diameter_lower_proxy_quantiles"]=np.quantile([float(np.asarray(r["grid"])[-1])/s["scale"] for r in records],[.1,.5,.9])
    result["diameter_note"]="Maximum root-to-all distance for eight measure roots, a lower proxy rather than exact diameter"
    if records[0].get("matrix") is not None:result.update(matrix_correlations(records))
    try:result["per_realization_normalization_discrepancy"]=discrepancy(s,summarize(records,per_realization=True))
    except ValueError:result["per_realization_normalization_discrepancy"]="Undefined: a field's pair median is zero; do not remove colliding samples"
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--output",required=True)
    p.add_argument("--confirmation",default="results/phase3a/confirmation")
    p.add_argument("--sensitivity",nargs="*",default=[])
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    choice=json.loads(Path("results/phase3a/selection/selection.json").read_text())["choices"]
    graphs=graph_groups("results/phase3a/graphs",split="confirmation")
    all_graphs=graph_groups("results/phase3a/graphs")
    fields=field_groups(args.confirmation)
    resolution=[]
    for (case,n),records in all_graphs.items():
        refined=[r for r in records if r["header"]["resolution"]]
        if not refined:continue
        delta=np.array([r["header"]["resolution"]["paired_relative_mean_change"] for r in refined])
        overlap=np.array([r["header"]["resolution"]["edge_jaccard"] for r in refined])
        resolution.append({"case":case,"n":n,"physical_paired_paths":len(refined),
            "relative_mean_coarse_minus_fine":delta.tolist(),"mean":float(delta.mean()),
            "paired_se":float(delta.std(ddof=1)/np.sqrt(len(delta))),"edge_jaccards":overlap.tolist()})
    summary={}
    exact_flat=next(iter(field_groups("results/phase3a/flat_exact").values()))
    for case in ["i3","a2","a3"]:
        pair=(choice[case]["xi"],choice[case]["gamma"])
        target=graphs[(case,8192)]
        summary[case]={"graph":compact_summary(target),"selected":compact_summary(fields[(48,"primitive2",*pair)]),
            "flat":compact_summary(fields[(48,"primitive2",0.,0.)])}
        summary[case]["exact_euclidean_flat_comparison"]=discrepancy(summarize(target),summarize(exact_flat))
        if pair[1]>0:
            gauged=[]
            for r in fields[(48,"primitive2",*pair)]:
                log_mass=r["header"]["gmc"][str(pair[1])]["log_total_mass"]
                factor=np.exp(-pair[0]/pair[1]*log_mass)
                gauged.append(dict(r,grid=np.asarray(r["grid"])*factor,
                    root_means=np.asarray(r["root_means"])*factor,matrix=np.asarray(r["matrix"])*factor))
            summary[case]["unit_mass_gauge_sensitivity"]={"formula":"D -> M^(-xi/gamma)*D, probability measure unchanged; not conditioning on M=1",
                "graph_comparison":discrepancy(summarize(target),summarize(gauged)),"statistics":compact_summary(gauged)}
        if case in ["a2","a3"]:
            summary[case]["nonzero_gamma_brackets"]={str(g):discrepancy(summarize(target),summarize(fields[(48,"primitive2",.2,g)])) for g in [.5,1.]}
    archives=archive_graphs()
    pair=(choice["i3"]["xi"],choice["i3"]["gamma"])
    reference=summarize(fields[(48,"primitive2",*pair)])
    large=[]
    for n,records in sorted(archives.items()):
        s=summarize(records)
        grid=np.asarray(records[0]["grid"])
        jumps=[]
        for r in records:
            c=np.asarray(r["cdfs"]).mean(axis=0)
            # Bisection represents an integer median a few ulps above its
            # support point. searchsorted would inspect the NEXT integer.
            i=int(round(s["scale"]))
            jumps.append(c[i]-(c[i-1] if i else 0) if i<len(c) else 0.)
        large.append({"n":n,"independent_graphs":len(records),"median":s["scale"],
            "cdf_jump_at_median":float(np.mean(jumps)),"selected_field_comparison":discrepancy(s,reference)})
    sensitivity={}
    for directory in args.sensitivity:
        data=field_groups(directory)
        values={}
        for key,records in data.items():
            n,stencil,xi,gamma=key
            base=fields.get((n,stencil,xi,gamma))
            if base is None:continue
            ids={r["realization"] for r in records};base=[r for r in base if r["realization"] in ids]
            values[str(key)]={"realizations":len(records),"paired_summary_shift":discrepancy(summarize(base),summarize(records)),
                "mean_raw_distance_ratio":float(np.mean([np.mean(r["root_means"]) for r in records])/np.mean([np.mean(r["root_means"]) for r in base]))}
        sensitivity[directory]=values
    smooth=field_groups("results/phase3a/smooth_control")[(32,"26",.3,1.5)]
    ids={r["realization"] for r in smooth}
    log=[r for r in fields[(32,"26",.3,1.5)] if r["realization"] in ids]
    graph=graphs[("i3",8192)]
    gs=summarize(graph)
    log_score=discrepancy(gs,summarize(log));smooth_score=discrepancy(gs,summarize(smooth))
    rng=np.random.default_rng(202611113911);delta=[]
    for _ in range(128):
        indices=rng.integers(len(smooth),size=len(smooth))
        graph_boot=[graph[i]for i in rng.integers(len(graph),size=len(graph))]
        g=summarize(graph_boot)
        delta.append(discrepancy(g,summarize([smooth[i]for i in indices]))["objective"]-
            discrepancy(g,summarize([log[i]for i in indices]))["objective"])
    smooth_probe={"paired_fields":8,"graph_realizations":len(graph),"log_score":log_score,"smooth_score":smooth_score,
        "smooth_minus_log_objective":smooth_score["objective"]-log_score["objective"],
        "paired_objective_difference_mc95":np.quantile(delta,[.025,.975]),
        "qualification":"Small common-field specificity probe, no new Gaussian draws; smooth exponential measure is not continuum GMC"}
    save_json(output/"diagnostics.json",{"graph_grid_refinement":resolution,"selected_statistics":summary,
        "archived_I3_unique_graphs":sum(map(len,archives.values())),"large_I3_finite_size":large,"sensitivity":sensitivity,
        "smooth_specificity_probe":smooth_probe})
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot([r["n"] for r in large],[r["selected_field_comparison"]["pair_w1"] for r in large],"o-")
    axes[0].set(xscale="log",xlabel="Archived exact I3 graph vertices",ylabel="W1 to selected N48 LFPP/GMC",title="No new large graph simulation")
    axes[1].plot([r["n"] for r in large],[r["cdf_jump_at_median"] for r in large],"o-")
    axes[1].set(xscale="log",xlabel="Graph vertices",ylabel="Pair CDF atom at ensemble median",title="Integer-distance finite-size effect")
    fig.savefig(output/"large_graph_finite_size.png",dpi=180);fig.savefig(output/"large_graph_finite_size.pdf");plt.close(fig)
    print(json.dumps({"output":str(output),"archive_graphs":sum(map(len,archives.values())),"sensitivity_directories":len(sensitivity)}))


if __name__=="__main__":main()
