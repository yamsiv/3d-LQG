"""Finalize Phase 3B from saved data; no fields or graphs are generated.

The finalist parameters are read from the pre-holdout frozen plan. Whole
physical realizations, including all roots and matrix subsets, are the units
of the bootstrap. These descriptive discrepancies are not goodness-of-fit
p-values or estimates of a continuum law.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.phase3a_compare import save_json
from experiments.phase3a_diagnostics import archive_graphs
from experiments.phase3b_search import graph_groups, field_groups, WEIGHTS
from src.metric_features_3b import (feature_vector, grouped_feature_distance,
                                  matrix_shape_samples, _gauge_records)
from src.metric_measure_3a import summarize, discrepancy, EVAL_GRID, RADII
from src.provenance import environment_manifest

ROOT = Path("results/phase3b")
KEYS = ["triangles_small_mean", "quartets_delta_mean", "quartets_aspect_mean",
        "ball_sd_r0.5", "root_mean_distance_cv", "finite_radius_growth_q0.5"]


def feature(records, gauge="grounded", normalization="median"):
    transformed = _gauge_records(records, gauge, False)
    scale = None if normalization == "median" else np.mean([
        np.mean(r["root_means"]) for r in transformed])
    return feature_vector(transformed, gauge=gauge, already_transformed=True, scale=scale)


def scalar_features(f):
    lookup = dict(zip(f["names"], f["vector"]))
    return {k: float(lookup[k]) for k in KEYS}


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def score(g, f):
    return grouped_feature_distance(g, f)["total"]


def bootstrap_finalist(graphs, probes, flat, gauge, rng, draws):
    bands = list(probes)
    assert all([r["realization"] for r in probes[b]] ==
               [r["realization"] for r in flat] for b in bands)
    g = feature(graphs, gauge)
    fs = {b: feature(rows, gauge) for b, rows in probes.items()}
    z = feature(flat, gauge)
    central = grouped_feature_distance(g, fs["central"])
    robust = max(score(g, f) for f in fs.values())
    point = dict(central=central["total"], robust=robust, flat=score(g, z),
                 improvement_over_flat=score(g, z)-robust,
                 components=central["components"])
    delta = {k: scalar_features(g)[k]-scalar_features(fs["central"])[k] for k in KEYS}
    samples = {k: [] for k in ["central", "robust", "flat", "improvement_over_flat"]}
    differences = {k: [] for k in KEYS}
    for _ in range(draws):
        gi = rng.integers(len(graphs), size=len(graphs))
        fi = rng.integers(len(flat), size=len(flat))
        bg = feature([graphs[i] for i in gi], gauge)
        bf = {b: feature([rows[i] for i in fi], gauge) for b, rows in probes.items()}
        bz = feature([flat[i] for i in fi], gauge)
        robust_b = max(score(bg, f) for f in bf.values())
        values = dict(central=score(bg, bf["central"]), robust=robust_b,
                      flat=score(bg, bz), improvement_over_flat=score(bg, bz)-robust_b)
        for k, value in values.items(): samples[k].append(value)
        sg, sf = scalar_features(bg), scalar_features(bf["central"])
        for k in KEYS: differences[k].append(sg[k]-sf[k])
    return dict(point=point, mc95={k: interval(v) for k,v in samples.items()},
                signed_graph_minus_field=delta,
                signed_mc95={k: interval(v) for k,v in differences.items()},
                graph_features=scalar_features(g), field_features=scalar_features(fs["central"]),
                graph_diagnostics=g["diagnostics"], field_diagnostics=fs["central"]["diagnostics"],
                gamma_probe_scores={b: score(g,f) for b,f in fs.items()},
                gamma_values={b:rows[0]["gamma"] for b,rows in probes.items()},
                draws=draws, gauge=gauge,
                interpretation="MC only; 16 independent graphs and 12 fields; field resamples shared across gamma probes and flat control; robust=max over valid-model probes, not worst-case over all systematic errors")


def split_floor(records, gauge, rng, draws=24):
    # Both halves are disjoint. Their smaller sample counts make this a
    # descriptive half-ensemble floor rather than a full-ensemble null test.
    values = []
    for _ in range(draws):
        order = rng.permutation(len(records)); cut=len(records)//2
        values.append(score(feature([records[i] for i in order[:cut]], gauge),
                            feature([records[i] for i in order[cut:]], gauge)))
    return dict(median=float(np.median(values)), central90=np.quantile(values,[.05,.95]).tolist(),
                half_counts=[len(records)//2,len(records)-len(records)//2], draws=draws)


def matched_controls(graphs, fields32, fields48, output, rng, draws):
    stencil=field_groups([ROOT/"holdout_stencil_control"],48)
    smooth=field_groups([ROOT/"holdout_smooth_control"],32)
    rows=[]
    for (xi,band), records in stencil.items():
        if band!="central": continue
        rich=fields48[(xi,"central")][:len(records)]
        ratios=np.array([np.mean(a["root_means"])/np.mean(b["root_means"])-1
                         for a,b in zip(records,rich)])
        draws_ratio=[float(np.mean(ratios[rng.integers(len(ratios),size=len(ratios))])) for _ in range(draws)]
        rows.append(dict(kind="paired_26_minus_98",xi=xi,fields=len(records),
            mean_distance_relative=float(ratios.mean()), mc95=interval(draws_ratio),
            feature_shift=score(feature(records),feature(rich)),
            note="same field and root uniforms; ensemble normalizers recomputed separately"))
    save_json(output/"stencil_controls.json",rows)
    rows=[]
    plan=json.loads((ROOT/"final_plan.json").read_text())
    for c in plan["candidates"]:
        identifier,xi=c["id"],c["xi"]
        log=fields32[(xi,"central")][:8]; sm=smooth[(xi,"central")]
        for gauge in ["grounded","unit_mass"]:
            g=feature(graphs[identifier],gauge)
            central_log=score(g,feature(log,gauge));central_smooth=score(g,feature(sm,gauge))
            diffs=[]
            for _ in range(draws):
                gi=rng.integers(len(graphs[identifier]),size=len(graphs[identifier]));fi=rng.integers(8,size=8)
                bg=feature([graphs[identifier][i] for i in gi],gauge)
                diffs.append(score(bg,feature([sm[i] for i in fi],gauge))-
                             score(bg,feature([log[i] for i in fi],gauge)))
            rows.append(dict(id=identifier,xi=xi,gauge=gauge,fields=8,
                log_score=central_log,smooth_score=central_smooth,
                smooth_minus_log=central_smooth-central_log, mc95=interval(diffs),
                note="same low-mode randomness and root uniforms; smooth filter fixed |k|<=2 with equal deterministic variance; not a GMC continuum approximation"))
    save_json(output/"smooth_controls.json",rows)


def archived_check(fields48, historic, output, rng, draws):
    archives=archive_graphs(); sizes=sorted(archives)
    candidates={"frozen_xi025":fields48[(.25,"central")],
                "historic_xi030":historic[(.3,"central")],
                "flat":fields48[(0.,"central")]}
    rows=[];largest={}
    for gauge in ["grounded","unit_mass"]:
        summaries={key:summarize(_gauge_records(records,gauge,False)) for key,records in candidates.items()}
        for n in sizes:
            g=summarize(archives[n])
            for label,f in summaries.items():
                rows.append(dict(n=n,graph_realizations=len(archives[n]),gauge=gauge,label=label,
                    xi=candidates[label][0]["xi"],gamma=candidates[label][0]["gamma"],
                    **discrepancy(g,f),graph_half_ball_sd=g["ball_sd"][1],field_half_ball_sd=f["ball_sd"][1]))
        n=sizes[-1];differences={key:[] for key in candidates if key!="flat"}
        for _ in range(draws):
            gi=rng.integers(len(archives[n]),size=len(archives[n])); fi=rng.integers(12,size=12)
            g=summarize([archives[n][i] for i in gi])
            fs={key:summarize(_gauge_records([records[i] for i in fi],gauge,False)) for key,records in candidates.items()}
            baseline=discrepancy(g,fs["flat"])["pair_w1"]
            for key in differences: differences[key].append(baseline-discrepancy(g,fs[key])["pair_w1"])
        largest[gauge]={key:dict(flat_minus_candidate_pair_w1_mc95=interval(value)) for key,value in differences.items()}
    save_json(output/"archived_i3.json",dict(rows=rows,largest_size=sizes[-1],paired_improvement=largest,
        note="648 archived exact I3 graphs reused; only pair/root-ball statistics, no unavailable higher-point matrices; xi.3 is a separately declared historical diagnostic, not heldout retuning"))
    return rows


def plots(output, graphs, fields, finalists, rows, archive_rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9})
    figs={};fig,axes=plt.subplots(1,4,figsize=(15,3.6),constrained_layout=True)
    figb,axesb=plt.subplots(1,4,figsize=(15,3.6),constrained_layout=True)
    figq,axesq=plt.subplots(1,4,figsize=(15,3.6),constrained_layout=True)
    for ax,ab,aq,c in zip(axes,axesb,axesq,finalists):
        identifier,xi=c["id"],c["xi"]
        series=[("Graph n8192",graphs[identifier],"black"),
                (f"LFPP/GMC {xi:g}, {c['gamma']:.3f}",fields[(xi,"central")],"tab:blue"),
                ("Flat torus",fields[(0.,"central")],"tab:orange")]
        for label,records,color in series:
            s=summarize(records)
            ax.plot(EVAL_GRID,s["cdf"],color=color,label=label)
            ab.plot(RADII,s["ball_sd"],color=color,label=label)
            qs=np.concatenate([matrix_shape_samples(r["matrix"])["quartets"][:,0] for r in records])
            qs.sort();aq.plot(qs,np.arange(1,len(qs)+1)/len(qs),color=color,label=label)
        ax.set(xlim=(0,2.1),ylim=(0,1),title=identifier,xlabel="Distance / ensemble median",ylabel="Pair CDF")
        ab.set(title=identifier,xlabel="Radius / ensemble median",ylabel="SD of root ball probability mass",xlim=(.2,1.6))
        aq.set(title=identifier,xlabel="Normalized quartet four-point delta",ylabel="Subset CDF",xlim=(0,.5))
        for a in [ax,ab,aq]:a.legend(fontsize=7)
    figs.update(pair_cdfs=fig,ball_mass_dispersion=figb,quartet_shapes=figq)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,gauge in zip(axes,["grounded","unit_mass"]):
        selected=[r for r in rows if r["gauge"]==gauge]
        x=np.arange(len(selected));y=np.array([r["robust"] for r in selected]);ci=np.array([r["robust_mc95"] for r in selected]).T
        # Bootstrap quantiles need not contain the plug-in estimate.
        ax.vlines(x,ci[0],ci[1],color="tab:blue");ax.scatter(x,y,color="tab:blue",label="Frozen candidate, robust gamma probes")
        ax.scatter(x,[r["flat"] for r in selected],marker="x",color="tab:orange",label="Flat control")
        ax.set(xticks=x,xticklabels=[r["id"] for r in selected],ylabel="Grouped natural-scale discrepancy",title=gauge)
        ax.legend(fontsize=7)
    figs["heldout_scores"]=fig
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,gauge in zip(axes,["grounded","unit_mass"]):
        for label in ["frozen_xi025","historic_xi030","flat"]:
            rr=[r for r in archive_rows if r["gauge"]==gauge and r["label"]==label]
            ax.plot([r["n"] for r in rr],[r["pair_w1"] for r in rr],"o-",label=label)
        ax.set(xscale="log",xlabel="Archived exact I3 graph size n",ylabel="Pair Wasserstein-1 / ensemble median",title=gauge)
        ax.legend(fontsize=8)
    figs["archived_i3_finite_size"]=fig
    design=json.loads((ROOT/"extended_design.json").read_text())["configurations"]
    coarse=json.loads((ROOT/"screen_analysis/scores.json").read_text())["rows"]
    intermediate=json.loads((ROOT/"intermediate_analysis/scores.json").read_text())["rows"]
    fig=plt.figure(figsize=(12,4.8),constrained_layout=True);ax=fig.add_subplot(121,projection="3d");bx=fig.add_subplot(122)
    for family in sorted({s["family"] for s in design}):
        specs=[s for s in design if s["family"]==family and s["id"]!="C022"]
        xyz=np.array([s["triple"] for s in specs]);ax.scatter(*xyz.T,label=family,s=28)
    failed=next(s for s in design if s["id"]=="C022");ax.scatter(*failed["triple"],marker="x",color="red",s=80,label="C022 certificate failure")
    for c in finalists:
        xyz=c["graph_spec"]["triple"];ax.text(*xyz,c["id"],fontsize=7)
    ax.set(xlabel="Canonical a",ylabel="Canonical b",zlabel="Canonical c",title="33 SPD configurations attempted (coordinate permutations quotiented)")
    ax.legend(fontsize=6,loc="upper left")
    for data,label,marker in [(coarse,"Coarse screen", "o"),(intermediate,"Intermediate", "s")]:
        for identifier in sorted({r["id"] for r in data}):
            eligible=[r for r in data if r["id"]==identifier and r["eligible_primary"]]
            if not eligible:continue
            best=min(eligible,key=lambda r:r["robust_score"])
            bx.scatter(best["xi"],best["robust_score"],marker=marker,color=".6" if marker=="o" else "tab:blue",s=30)
    for r in rows:
        if r["gauge"]=="grounded":bx.scatter(r["xi"],r["robust"],marker="*",s=110,label=r["id"]+" heldout")
    bx.set(xlabel="Calibrated LFPP xi",ylabel="Best/frozen robust natural-scale score",title="Different fidelities and independent cohorts; not a single response surface")
    bx.legend(fontsize=7)
    figs["covariance_candidate_map"]=fig
    for name,fig in figs.items():
        fig.savefig(output/(name+".png"),dpi=170);fig.savefig(output/(name+".pdf"));plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--output",required=True);p.add_argument("--bootstrap",type=int,default=64)
    a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():raise FileExistsError(output/"summary.json")
    plan=json.loads((ROOT/"final_plan.json").read_text());finalists=plan["candidates"]
    graphs=graph_groups([ROOT/"holdout_graphs"],8192)
    f48=field_groups([ROOT/"holdout_fields"],48);f32=field_groups([ROOT/"holdout_fields"],32)
    historic=field_groups([ROOT/"historical_candidate_control"],48)
    if any(len(x)!=16 for x in graphs.values()) or any(len(x)!=12 for x in f48.values()):raise ValueError("Holdout incomplete")
    rng=np.random.default_rng(202611113891);rows=[];fits={};sensitivity=[];noise={}
    for c in finalists:
        identifier,xi=c["id"],c["xi"]
        probes={band:records for (x,band),records in f48.items() if x==xi}
        for gauge in ["grounded","unit_mass"]:
            name=identifier+"_"+gauge;checkpoint=output/(name+"_bootstrap.json")
            if checkpoint.exists():fit=json.loads(checkpoint.read_text())
            else:
                fit=bootstrap_finalist(graphs[identifier],probes,f48[(0.,"central")],gauge,rng,a.bootstrap)
                save_json(checkpoint,fit)
            fits[name]=fit
            rows.append(dict(id=identifier,xi=xi,gamma=c["gamma"],gauge=gauge,
                **{k:fit["point"][k] for k in ["central","robust","flat","improvement_over_flat"]},
                **{k+"_mc95":fit["mc95"][k] for k in ["central","robust","flat","improvement_over_flat"]}))
            noise[name]={"graphs":split_floor(graphs[identifier],gauge,rng),
                         "fields":split_floor(probes["central"],gauge,rng)}
            for normalization in ["median","mean"]:
                gf=feature(graphs[identifier],gauge,normalization)
                ff={b:feature(rs,gauge,normalization) for b,rs in probes.items()}
                for weights_name,weights in WEIGHTS.items():
                    sensitivity.append(dict(id=identifier,xi=xi,gauge=gauge,normalization=normalization,weights=weights_name,
                        central=grouped_feature_distance(gf,ff["central"],weights=weights)["total"],
                        robust=max(grouped_feature_distance(gf,f,weights=weights)["total"] for f in ff.values())))
    save_json(output/"sensitivity.json",sensitivity);save_json(output/"same_model_half_split.json",noise)
    matched_controls(graphs,f32,f48,output,rng,a.bootstrap)
    archived=archived_check(f48,historic,output,rng,a.bootstrap)
    finite=[]
    mid=graph_groups([ROOT/"intermediate_graphs"],8192);mid_small=graph_groups([ROOT/"intermediate_graphs"],2048)
    for c in finalists:
        identifier,xi=c["id"],c["xi"]
        for label,records in [("intermediate_n2048",mid_small[identifier]),("intermediate_n8192",mid[identifier]),("holdout_n8192",graphs[identifier])]:
            for n,fs in [(32,f32),(48,f48)]:
                for gauge in ["grounded","unit_mass"]:
                    finite.append(dict(id=identifier,graph_stage=label,field_n=n,gauge=gauge,xi=xi,
                        score=score(feature(records,gauge),feature(fs[(xi,"central")],gauge)),
                        mean_graph_distance=float(np.mean([np.mean(r["root_means"]) for r in records]))))
    save_json(output/"finite_size.json",finite)
    historical=[]
    for gauge in ["grounded","unit_mass"]:
        g=feature(graphs["C002"],gauge);f=feature(historic[(.3,"central")],gauge)
        historical.append(dict(gauge=gauge,xi=.3,gamma=historic[(.3,"central")][0]["gamma"],
                               score=score(g,f),components=grouped_feature_distance(g,f)["components"]))
    save_json(output/"historical_candidate.json",historical)
    save_json(output/"summary.json",dict(provenance=environment_manifest(),configuration=vars(a),
        frozen_parameters="results/phase3b/final_plan.json",rows=rows,bootstrap=fits,
        interpretation="Grouped natural scales; independent realization uncertainty only; selected covariance and xi frozen before new holdout seeds; unit-mass gauge and alternate weights are sensitivities, never optimized jointly"))
    with (output/"heldout_candidates.csv").open("x") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plots(output,graphs,f48,finalists,rows,archived)
    print(json.dumps({"output":str(output),"frozen_finalists":len(finalists),"bootstrap_draws":a.bootstrap}),flush=True)


if __name__=="__main__":main()
