"""Checkpointed field/LFPP/GMC and special-graph ensembles for Task 3A.

Example discovery:
  python -m experiments.phase3a_ensembles field --output results/phase3a/discovery
No existing checkpoint is overwritten. --resume checks the exact configuration.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest


def seed_for(base, *keys):
    return int(np.random.SeedSequence([base, *keys]).generate_state(1, dtype=np.uint64)[0])


def write_npz(path, **arrays):
    if path.exists(): raise FileExistsError(path)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.rename(path)


def configure(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = {k:v for k,v in vars(args).items() if k != "resume"}
    path = root/"metadata.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not args.resume: raise FileExistsError(path)
        if saved["configuration"] != config: raise ValueError("Resume configuration differs")
    else:
        path.write_text(json.dumps({"configuration":config,"provenance":environment_manifest(),
            "law":"biased for graphs; same physical field for all LFPP/GMC parameters",
            "independence":"realization seed; coupled resolutions and parameters are not extra samples",
            "cost_policy":"measured pilots; checkpoint every physical realization and grid; no invocation expected >15min"},indent=2)+"\n")
    return root


def parse_pairs(text):
    if text == "all":
        return [(xi,gamma) for xi in [0.,.1,.2,.3,.4,.6] for gamma in [0.,.5,1.,1.5,2.]]
    return [tuple(map(float,p.split(":"))) for p in text.split(",")]


def run_field(args):
    from src.log_field_3d import sample_coupled_fields, resample_field
    from src.gmc_3d import gmc_weights
    from src.lfpp_3d import build_lfpp_graph, root_distances, cube_crossing
    root=configure(args); pairs=parse_pairs(args.pairs)
    for realization in range(args.samples):
        seed=seed_for(args.seed,realization,1)
        fields=sample_coupled_fields(args.sizes,np.random.default_rng(seed),regularization=args.regularization)
        field_file=root/f"field_{realization:03d}.npz"
        if not field_file.exists():
            write_npz(field_file,**{f"h{n}":f.h for n,f in fields.items()},
                      header=json.dumps({"seed":seed,"fields":{str(n):f.metadata for n,f in fields.items()}}))
        for n,field in fields.items():
            if args.evaluate_sizes and n not in args.evaluate_sizes:
                continue
            if args.mesh_factor != 1:
                field=resample_field(field,n*args.mesh_factor)
            mesh_n=field.h.shape[0]
            measures={g:gmc_weights(field.h,g,field.variance) for g in sorted({p[1] for p in pairs})}
            point_sets={g:np.searchsorted(np.cumsum(mu.probabilities.ravel()),
                        np.random.default_rng(seed_for(args.seed,realization,int(g*100),77)).random(args.points),side="right")
                        for g,mu in measures.items()}
            for stencil in args.stencils:
                path=root/f"metric_{realization:03d}_n{n}_{stencil}.npz"
                if path.exists(): continue
                started=time.perf_counter(); results={}; crossings={}; metrics=[]
                for xi in sorted({p[0] for p in pairs}):
                    gammas=[g for x,g in pairs if x == xi]
                    selected=np.unique(np.concatenate([point_sets[g] for g in gammas]))
                    graph,metric_meta=build_lfpp_graph(field.h,xi,stencil=stencil,boundary=args.boundary)
                    distances=root_distances(graph,selected)
                    metrics.append(metric_meta)
                    for gamma in gammas:
                        points=point_sets[gamma]; indices=np.searchsorted(selected,points)
                        d=distances[indices]
                        cdf=weighted_root_cdfs(d,measures[gamma].probabilities.ravel(),bins=args.bins)
                        results[(xi,gamma)]={**cdf,"matrix":d[:,points],"points":points}
                    del distances,graph
                    if stencil == args.stencils[0] and args.crossings:
                        cube,_=build_lfpp_graph(field.h,xi,stencil="axis",boundary="cube")
                        edge=cube_crossing(cube,mesh_n)
                        site=cube_crossing(cube,mesh_n,site_weights=np.exp(xi*field.h))
                        crossings[str(xi)]={"edge":edge,"site":site,"edge_gap_corrected":edge/((mesh_n-1)/mesh_n)}
                        del cube
                header={"realization":realization,"field_seed":seed,"sampling_n":n,"n":mesh_n,
                    "mesh_factor":args.mesh_factor,"points":args.points,"pairs":pairs,"stencil":stencil,
                    "boundary":args.boundary,"field_metadata":field.metadata,"variance":field.variance,
                    "field_file":str(field_file),"metric_metadata":metrics,"crossings":crossings,
                    "gmc":{str(g):{"total_mass":m.total_mass,"log_total_mass":m.log_total_mass,
                        "ess":float(1/np.sum(m.probabilities**2)),"collision_probability":float(np.sum(m.probabilities**2)),
                        "max_atom":float(m.probabilities.max())} for g,m in measures.items()},
                    "seconds":time.perf_counter()-started}
                write_npz(path,grid=np.stack([results[p]["grid"] for p in pairs]),
                    cdfs=np.stack([results[p]["cdfs"] for p in pairs]).astype(np.float32),
                    root_means=np.stack([results[p]["root_means"] for p in pairs]),
                    matrices=np.stack([results[p]["matrix"] for p in pairs]),
                    points=np.stack([results[p]["points"] for p in pairs]),
                    bin_width=np.array([results[p]["distance_bin_width"] for p in pairs]),
                    header=json.dumps(header))
                print(json.dumps({"checkpoint":str(path),"seconds":header["seconds"]}),flush=True)
                gc.collect()


def graph_arrays(graph,points):
    from src.graph_metrics import bfs_distances
    csr=graph.to_sparse()
    d=np.array([bfs_distances(csr,int(p)) for p in points])
    data=weighted_root_cdfs(d,np.ones(graph.n)/graph.n,discrete=True)
    return dict(data,matrix=d[:,points],points=points)


def run_graph(args):
    from src.brownian_excursions import sample_excursion
    from src.weyl_excursions import sample_weyl_excursion
    from src.weyl_a3 import sample_a3_excursion
    from src.phase2c_paths import thin_polygonal
    from src.mated_crt import graph_from_excursion,validate_graph
    root=configure(args)
    archive=[json.loads(line) for line in Path("results/data/i3_biased_combined.jsonl").read_text().splitlines()]
    for case in args.cases:
        for n in args.sizes:
            old=sorted([r for r in archive if r["n"]==n],key=lambda r:r["realization"])
            for realization in range(args.samples):
                path=root/f"graph_{case}_n{n}_{realization:03d}.npz"
                if path.exists():continue
                started=time.perf_counter()
                resolution=[]
                if case=="i3":
                    row=old[realization]; seed=row["graph_seed"]
                    sample=sample_excursion(n,np.eye(3),np.random.default_rng(seed),law="biased")
                    points=np.asarray(row["roots"],int)
                else:
                    seed=seed_for(args.seed,1 if case=="a2" else 2,n,realization)
                    sampler=sample_weyl_excursion if case=="a2" else sample_a3_excursion
                    q=args.q*2 if realization<args.refine_samples else args.q
                    sample=sampler(n,rng=np.random.default_rng(seed),law="biased",oversample=q)
                    points=np.random.default_rng(seed_for(args.seed,n,realization,90)).integers(n,size=args.points)
                    if q!=args.q:
                        fine=graph_from_excursion(sample,n)
                        fine_data=graph_arrays(fine,points)
                        sample=thin_polygonal(sample,n,args.q)
                graph=graph_from_excursion(sample,n)
                data=graph_arrays(graph,points)
                if case=="i3":
                    hist=np.diff(np.c_[np.zeros(len(points)),data["cdfs"]],axis=1)*n
                    archived=np.asarray(row["root_histograms"])
                    width=min(hist.shape[1],archived.shape[1])
                    if not np.array_equal(np.rint(hist[:,:width]).astype(int),archived[:,:width]):
                        raise ValueError("Archived exact graph regeneration changed")
                elif realization<args.refine_samples:
                    e0,e1=set(map(tuple,graph.edges)),set(map(tuple,fine.edges))
                    resolution={"q":args.q,"fine_q":q,"paired_relative_mean_change":float(np.mean(data["root_means"])/np.mean(fine_data["root_means"])-1),
                        "edge_jaccard":len(e0&e1)/len(e0|e1),"fine_grid":fine_data["grid"].tolist(),
                        "fine_cdfs":fine_data["cdfs"].tolist(),"fine_matrix":fine_data["matrix"].tolist(),
                        "fine_root_means":fine_data["root_means"].tolist()}
                    del fine,fine_data
                header={"case":case,"n":n,"realization":realization,"seed":seed,"source":"regenerated archived exact graph for new distance matrix" if case=="i3" else "new special correlated graph",
                    "sampler_metadata":sample.metadata,"validation":validate_graph(graph),
                    "resolution":resolution,"seconds":time.perf_counter()-started}
                write_npz(path,**data,header=json.dumps(header))
                print(json.dumps({"checkpoint":str(path),"seconds":header["seconds"]}),flush=True)
                del graph,sample;gc.collect()


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest="kind",required=True)
    for name in ["field","graph"]:
        p=sub.add_parser(name);p.add_argument("--output",required=True);p.add_argument("--resume",action="store_true")
        p.add_argument("--sizes",type=int,nargs="+",default=[16,24,32] if name=="field" else [2048,8192])
        p.add_argument("--samples",type=int,default=24);p.add_argument("--points",type=int,default=8)
        p.add_argument("--seed",type=int,default=202611113001 if name=="field" else 202611113101)
    p=sub.choices["field"];p.add_argument("--pairs",default="all")
    p.add_argument("--stencils",nargs="+",default=["26"])
    p.add_argument("--regularization",default="sharp",choices=["sharp","heat"])
    p.add_argument("--boundary",default="torus",choices=["torus","cube"])
    p.add_argument("--evaluate-sizes",type=int,nargs="+")
    p.add_argument("--mesh-factor",type=int,default=1);p.add_argument("--bins",type=int,default=1024)
    p.add_argument("--crossings",action="store_true")
    p=sub.choices["graph"];p.add_argument("--cases",nargs="+",default=["i3","a2","a3"])
    p.add_argument("--q",type=int,default=32);p.add_argument("--refine-samples",type=int,default=4)
    args=parser.parse_args()
    if args.kind=="field":run_field(args)
    else:run_graph(args)


if __name__=="__main__":main()
