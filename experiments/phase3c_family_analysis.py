"""Freeze old-data tolerance, then analyze observed Phase3C family nodes.

Calibration uses ONLY old3B records and must precede new-score analysis. No
graphs/fields are generated. Runtime budget for32 disjoint partitions:<30s,
working memory<250MB, based on the previous feature-vector benchmarks.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.phase3b_search import graph_groups, field_groups
from src.family_search_3c import (
    calibrate_split_tolerance, observed_family_components, WEIGHTINGS, ADJACENCY,
    component_residual_profiles, direct_graph_levelset_comparisons,
    local_feature_rank, propose_family_adaptation,
)


SEED=202611123101
OLD=Path("results/phase3b")


def _jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_jsonable(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable(v)for v in value]
    return value


def _write(path,value):
    with Path(path).open("x")as stream:
        json.dump(_jsonable(value),stream,indent=2,allow_nan=False)
        stream.write("\n")


def calibrate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    complete=output/"tolerance_calibration.json"
    if complete.exists():
        print(f"Frozen old-data calibration already exists: {complete}",flush=True)
        return
    start=time.perf_counter()
    protocol=Path("docs/FAMILY_PROTOCOL_PHASE3C.md")
    sources=[Path(__file__),Path("src/family_search_3c.py"),Path("src/metric_features_3b.py"),protocol]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
    predeclaration={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
                    "seed":SEED,"partitions":32,"quantile":.95,
                    "rule":"max cohort-specific95th percentile of32 balanced disjoint same-model J partitions",
                    "cohorts":["old3B holdout I3/A3 n8192,16parents each",
                               "old3B holdout N48/98 centralxi0,.225,.25 andhistoricalxi.3,12parents each"],
                    "primary_gauge":"grounded","parallel_gauge":"unit_mass",
                    "primary_normalization":"mean","normalization_sensitivity":"median",
                    "weightings":WEIGHTINGS,"adjacency":ADJACENCY,
                    "sample_count_rescaling":False,"sampler_systematic_errors_separate":True,
                    "new_phase3c_score_files_loaded":False,
                    "runtime_estimate_seconds":30,"memory_estimate_mb":250,
                    "source_sha256":hashes}
    if not(output/"predeclaration.json").exists():
        _write(output/"predeclaration.json",predeclaration)
        (output/"protocol_snapshot.md").write_bytes(protocol.read_bytes())
        (output/"family_source_snapshot.py").write_bytes(Path("src/family_search_3c.py").read_bytes())
    graphs=graph_groups([OLD/"holdout_graphs"],8192)
    fields=field_groups([OLD/"holdout_fields"],48)
    historical=field_groups([OLD/"historical_candidate_control"],48)
    cohorts={"graph_i3":graphs["C002"],"graph_a3":graphs["C012"],
             "field_flat":fields[(0.,"central")],
             "field_xi0225":fields[(.225,"central")],
             "field_xi025":fields[(.25,"central")],
             "field_xi030":historical[(.3,"central")]}
    if any(len(rows) != (16 if name.startswith("graph") else 12) for name, rows in cohorts.items()):
        raise ValueError("Expected complete old3B calibration cohorts")
    if any(len(record["matrix"]) != 16 for rows in cohorts.values() for record in rows):
        raise ValueError("Calibration requires matched16-point matrices")
    fitted=calibrate_split_tolerance(cohorts,np.random.default_rng(SEED),draws=32,quantile=.95)
    fitted["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
    fitted["elapsed_seconds"]=time.perf_counter()-start
    fitted["seed"]=SEED
    fitted["predeclaration_sha256"]=hashlib.sha256((output/"predeclaration.json").read_bytes()).hexdigest()
    paths={record["source"]for rows in cohorts.values()for record in rows}
    _write(output/"old_input_manifest.json",{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in sorted(paths)})
    _write(complete,fitted)
    _plot_calibration(output,fitted)
    print(json.dumps({"epsilon":{mode:row["epsilon"]for mode,row in fitted["results"].items()},
                      "elapsed_seconds":fitted["elapsed_seconds"],"new_scores_read":False}),flush=True)


def analyze(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    tolerance=json.loads(Path(args.calibration).read_text())
    data=json.loads(Path(args.scores).read_text())
    rows=data["rows"]if isinstance(data,dict)else data
    nodes=[]
    for raw in rows:
        if raw.get("band","central")!="central":continue
        if raw.get("gauge",args.gauge)!=args.gauge:continue
        if raw.get("normalization",args.normalization)!=args.normalization:continue
        if raw.get("weighting",args.weighting)!=args.weighting:continue
        triple=raw.get("triple",raw.get("graph_spec",{}).get("triple"))
        if triple is None:raise ValueError("Node requires triple or graph_spec.triple")
        if "eligible"not in raw:
            raise ValueError("Explicit eligible mask including sampler and branch gates required")
        nodes.append(dict(raw,triple=triple,score=float(raw[args.score_key])))
    epsilon=tolerance["results"][f"{args.gauge}:{args.normalization}"]["epsilon"][args.weighting]
    design=[]
    if args.design:
        design_raw=json.loads(Path(args.design).read_text())
        design=design_raw["configurations"]if isinstance(design_raw,dict)else design_raw
    result=observed_family_components(nodes,epsilon=epsilon,tested_covariances=design)
    sensitivity={}
    for radius in ADJACENCY["radius_sensitivity"]:
        for diagonal in (False,True):
            sensitivity[f"radius{radius:g}_diagonal{int(diagonal)}"]=observed_family_components(
                nodes,epsilon=epsilon,covariance_radius=radius,diagonal_edges=diagonal,
                tested_covariances=design)
    _write(root/"observed_family_components.json",dict(result,nodes=nodes,
               gauge=args.gauge,normalization=args.normalization,weighting=args.weighting,
               score_source=str(args.scores),
               score_sha256=hashlib.sha256(Path(args.scores).read_bytes()).hexdigest(),
               tolerance_sha256=hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()))
    _write(root/"adjacency_sensitivity.json",sensitivity)
    strict=[dict(node,eligible=bool(node["eligible"]and node.get("componentwise_low",False)))for node in nodes]
    _write(root/"componentwise_secondary.json",observed_family_components(
        strict,epsilon=epsilon,tested_covariances=design))
    reference=tolerance["results"][f"{args.gauge}:{args.normalization}"]
    _write(root/"component_signed_profiles.json",component_residual_profiles(
        nodes,result["components"],reference["names"],reference["natural_scales"]))
    if args.graph_bootstrap:
        if args.gauge!="grounded"or args.normalization!="mean":
            raise ValueError("Provided cached bootstrap schema is primary grounded/mean only")
        _rank_and_levelsets(args,root,nodes,design,tolerance,epsilon)
    if args.propose_adaptive:
        if not design:raise ValueError("--design required for space-filling adaptive proposals")
        proposals=propose_family_adaptation(nodes,design,epsilon=epsilon,count=args.adaptive_count,
                                           first_identifier=args.first_new_id)
        _write(root/"adaptive_proposals.json",proposals)
    _plot_components(root,nodes,result)
    print(json.dumps({"nodes":len(nodes),"eligible_low_nodes":len(result["low_node_indices"]),
                      "components":len(result["components"]),
                      "observed_family_components":sum(c["observed_family_support"]for c in result["components"]),
                      "epsilon":epsilon}),flush=True)


def _rank_and_levelsets(args,root,nodes,design,tolerance,epsilon):
    """Use cached whole-parent draws, with no repeated metric/bootstrap work."""
    with np.load(args.graph_bootstrap,allow_pickle=False)as archive:
        graph={key:archive[key]for key in archive.files}
    required={"ids","vectors","valid","point_vectors","point_valid","scales","names"}
    if not required.issubset(graph):raise ValueError(f"Missing graph cache keys:{required-set(graph)}")
    identifiers=[str(value)for value in graph["ids"]]
    names=[str(value)for value in graph["names"]]
    reference=tolerance["results"]["grounded:mean"]
    if names!=reference["names"]:raise ValueError("Graph cached feature names differ from frozen calibration")
    specs={row["id"]:row for row in design}
    for node in nodes:specs.setdefault(node["id"],node)
    eligible={str(node["id"])for node in nodes if node["eligible"]}
    selected=[index for index,key in enumerate(identifiers)if key in eligible]
    identifiers=[identifiers[i]for i in selected]
    coordinates=np.array([specs[key]["triple"]for key in identifiers])
    point=np.asarray(graph["point_vectors"])[selected]
    vectors=np.asarray(graph["vectors"])[:,selected,:]
    validity=np.asarray(graph["point_valid"])[selected]&np.all(np.asarray(graph["valid"])[:,selected,:],axis=0)
    scales=np.asarray(graph["scales"])
    layouts={key:{"vector":point[i],"valid":np.asarray(graph["point_valid"])[selected][i],
                   "names":names,"scales":scales,"families":reference["families"]}
             for i,key in enumerate(identifiers)}
    levelsets=direct_graph_levelset_comparisons(nodes,layouts,tolerance,epsilon=epsilon,
                    normalization=args.normalization,weighting=args.weighting,
                    bootstrap_vectors={key:vectors[:,i,:]for i,key in enumerate(identifiers)},
                    bootstrap_valid={key:np.asarray(graph["valid"])[:,selected,:][:,i,:]for i,key in enumerate(identifiers)})
    _write(root/"direct_graph_levelsets.json",levelsets)
    field=None;field_keys={}
    if args.field_bootstrap:
        with np.load(args.field_bootstrap,allow_pickle=False)as archive:
            field={key:archive[key]for key in archive.files}
        for i,key in enumerate(field["keys"]):
            xi,band=ast.literal_eval(str(key))
            if band=="central"and xi>0:field_keys[float(xi)]=i
    field_xis=sorted(field_keys)
    ranks=[]
    for center,key in enumerate(identifiers):
        for radius in ADJACENCY["radius_sensitivity"]:
            fit=local_feature_rank(coordinates,point,center,scales,radius=radius,
                                   valid=validity,bootstrap_vectors=vectors)
            fit.update(id=key,triple=coordinates[center].tolist())
            if fit["status"]=="descriptive_local_fit":
                fit["rank_interpretation_usable"]=bool(not fit["near_permutation_quotient_wall"]and fit["design_condition"]<=30)
                projections=[]
                for j,xi in enumerate(field_xis):
                    if not any(node["id"]==key and node["xi"]==xi and node["eligible"]and node["score"]<=epsilon for node in nodes):continue
                    a,b=max(0,j-1),min(len(field_xis)-1,j+1)
                    if a==b:continue
                    ia,ib=field_keys[field_xis[a]],field_keys[field_xis[b]]
                    width=field_xis[b]-field_xis[a]
                    tangent=(field["point_vectors"][ib]-field["point_vectors"][ia])/width
                    tangent_draws=(field["vectors"][:,ib,:]-field["vectors"][:,ia,:])/width
                    tangent_valid=(field["point_valid"][ia]&field["point_valid"][ib]
                        &np.all(field["valid"][:,ia,:]&field["valid"][:,ib,:],axis=0))
                    projection_valid=validity&np.asarray(tangent_valid)[None,:]
                    projected=local_feature_rank(coordinates,point,center,scales,radius=radius,
                         valid=projection_valid,bootstrap_vectors=vectors,field_tangent=tangent,
                         bootstrap_field_tangents=tangent_draws)
                    projections.append({"xi":xi,"tangent_window":[field_xis[a],field_xis[b]],
                        "transverse_singular_values":projected.get("transverse_to_field_tangent_singular_values"),
                        "transverse_conditional_noise95":projected.get("transverse_graph_and_field_bootstrap_noise95"),
                        "field_tangent_uncertainty_included":projected.get("field_tangent_uncertainty_included",False)})
                fit["field_curve_projections"]=projections
            ranks.append(fit)
    _write(root/"local_feature_rank.json",{"rows":ranks,"input_graph_bootstrap":str(args.graph_bootstrap),
        "input_field_bootstrap":str(args.field_bootstrap),
        "qualification":"local finite response only; whole-parent bootstrap includes field-tangent MC when supplied; quotient walls, radius dependence and numerical biases prevent a continuum rank theorem"})
    _write(root/"rank_levelset_input_manifest.json",{str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in [args.graph_bootstrap,args.field_bootstrap]if path is not None})


def _plot_calibration(root,calibration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    modes=["grounded:mean","unit_mass:mean"]
    for ax,mode in zip(axes,modes):
        row=calibration["results"][mode]
        labels=list(row["cohorts"])
        x=np.arange(len(labels))
        values=[row["cohorts"][key]["epsilon_by_weight"]["primary"]for key in labels]
        ax.bar(x,values,color="tab:blue")
        ax.axhline(row["epsilon"]["primary"],color="black",ls="--",label="Frozen maximum")
        ax.set(xticks=x,xticklabels=labels,ylabel="95% quantile of disjoint self-split J",title=mode)
        ax.tick_params(axis="x",labelrotation=45,labelsize=7);ax.legend(fontsize=8)
    fig.savefig(root/"old_data_tolerance.png",dpi=180)
    fig.savefig(root/"old_data_tolerance.pdf");plt.close(fig)


def _plot_components(root,nodes,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    color={i:component for component,c in enumerate(result["components"])for i in c["indices"]}
    for a,b in result["edges"]:
        ax.plot([nodes[a]["xi"],nodes[b]["xi"]],
                [np.mean(nodes[a]["triple"]),np.mean(nodes[b]["triple"])],color=".8",lw=.6,zorder=0)
    for i in result["low_node_indices"]:
        ax.scatter(nodes[i]["xi"],np.mean(nodes[i]["triple"]),c=[color[i]],cmap="tab20",vmin=0,vmax=19,s=20)
    ax.set(xlabel="Measured linked xi",ylabel="Mean Brownian correlation (projection only)",
           title="Observed low-J adjacency; projected edges do not prove continuity")
    fig.savefig(root/"observed_family_projection.png",dpi=180)
    fig.savefig(root/"observed_family_projection.pdf");plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--stage",choices=["calibrate","analyze"],required=True)
    p.add_argument("--output",default="results/phase3c/families/statistics")
    p.add_argument("--calibration",default="results/phase3c/families/statistics/tolerance_calibration.json")
    p.add_argument("--scores")
    p.add_argument("--gauge",choices=["grounded","unit_mass"],default="grounded")
    p.add_argument("--normalization",choices=["mean","median"],default="mean")
    p.add_argument("--weighting",choices=list(WEIGHTINGS),default="primary")
    p.add_argument("--score-key",default="robust_score")
    p.add_argument("--design",default="results/phase3c/covariance_design/initial_design.json")
    p.add_argument("--graph-bootstrap")
    p.add_argument("--field-bootstrap")
    p.add_argument("--propose-adaptive",action="store_true")
    p.add_argument("--adaptive-count",type=int,default=12)
    p.add_argument("--first-new-id",type=int)
    args=p.parse_args()
    if args.stage=="calibrate":calibrate(args.output)
    else:
        if args.scores is None:p.error("--scores is required for observed family analysis")
        analyze(args)


if __name__=="__main__":main()
