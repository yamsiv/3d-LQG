"""Checkpointed broad Phase 3C linked LFPP/GMC field ensembles.

Commands: inventory, run. The run is explicitly restricted by --field-ids;
completed matching checkpoints are verified and reused without overwriting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from src.phase3c_fields import (DEFAULT_XIS,seed_for,sha256_file,write_json_new,
    write_npz_new,field_seed,gamma_probes,sampled_points,compatible_metric)
from src.log_field_3d import sample_coupled_fields,spectral_configuration
from src.gmc_3d import gmc_weights
from src.lfpp_3d import build_lfpp_graph,root_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.parameter_conversion_3b import load_calibration
from src.provenance import environment_manifest


def registry(args):
    output=Path(args.registry)
    if output.exists():
        records=json.loads(output.read_text())["records"]
        for row in records:
            if sha256_file(row["source_field"])!=row["source_field_sha256"]:
                raise ValueError("Registered field source changed")
        return records
    output.parent.mkdir(parents=True,exist_ok=True)
    fresh=output.parent/"fresh_source_fields"
    fresh.mkdir(exist_ok=True)
    records=[]
    old_seeds=set()
    # Field archives, not graph or metric replicas, define the independent
    # family inventory. Duplicated source pointers do not add realizations.
    for phase in [Path("results/phase3a"),Path("results/phase3b")]:
        for path in phase.glob("*/field_*.npz"):
            try:old_seeds.add(field_seed(path))
            except KeyError:pass
    old_seeds.update(range(202611113510,202611113530))
    for identifier in range(48):
        if identifier<24:
            index=identifier
            path=Path(f"results/phase3a/confirmation/field_{index:03d}.npz")
            cohort="phase3a_confirmation_reused_for_screen"
            complete_tuple=[24,32,48]
            root_seed=seed_for(202611113701,index,2)
        elif identifier<36:
            index=identifier-24
            path=Path(f"results/phase3b/holdout_fields/field_{index:03d}.npz")
            cohort="phase3b_holdout_now_screen"
            complete_tuple=[32,48]
            root_seed=seed_for(202611113861,index,2)
        else:
            index=identifier-36
            path=fresh/f"field_{identifier:03d}.npz"
            cohort="phase3c_new_screen"
            complete_tuple=[32,48]
            seed=seed_for(args.seed,index,1)
            root_seed=seed_for(args.seed,index,2)
            if seed in old_seeds:raise ValueError("New field seed collides with an old recorded field seed")
            if not path.exists():
                draws=sample_coupled_fields(complete_tuple,np.random.default_rng(seed))
                write_npz_new(path,**{f"h{n}":field.h for n,field in draws.items()},
                    header=json.dumps({"seed":seed,"fields":{str(n):field.metadata for n,field in draws.items()}}))
        with np.load(path) as archive:
            checksums={str(n):__import__("hashlib").sha256(archive[f"h{n}"].tobytes()).hexdigest() for n in [32,48]}
        records.append({"realization":identifier,"cohort":cohort,"cohort_index":index,
            "field_seed":field_seed(path),"root_seed":root_seed,"source_field":str(path),
            "source_field_sha256":sha256_file(path),"array_sha256":checksums,
            "complete_source_resolution_tuple":complete_tuple,
            "independence":"one complete physical Gaussian field family"})
    seeds=[r["field_seed"]for r in records]
    if len(set(seeds))!=48:raise ValueError("Duplicate physical field seeds")
    for n in [32,48]:
        if len({r["array_sha256"][str(n)]for r in records})!=48:
            raise ValueError("Duplicate physical field arrays")
    write_json_new(output,{"records":records,"counts":{"phase3a":24,"phase3b":12,"new":12,"total":48},
        "source_seed_and_array_hashes_unique":True,"fresh_seed_base":args.seed,
        "reserved_holdout_seed_base_not_used":202611123900,
        "historical_field_seed_count_checked":len(old_seeds),
        "qualification":"reused prior confirmation/holdout fields are screening data in3C, never new holdout evidence"})
    return records


def cache_candidates(row,n,xi):
    index=row["cohort_index"]
    # The completed small3C pilot is a cache, not an extra family or repeat.
    yield Path(f"results/phase3c/fields/pilot/xi{xi:g}/metric_{row['realization']:03d}_n{n}.npz")
    if row["cohort"].startswith("phase3a"):
        if n==32 and index==0:
            yield Path(f"results/phase3b/field_pilot/xi{xi:g}/metric_{index:03d}_n{n}.npz")
        if n==32 and 1<=index<12:
            yield Path(f"results/phase3b/screen_fields/xi{xi:g}/metric_{index:03d}_n{n}.npz")
        if n==48 and index<12:
            yield Path(f"results/phase3b/intermediate_fields/xi{xi:g}/metric_{index:03d}_n{n}.npz")
    elif row["cohort"].startswith("phase3b") and n==48 and index<4:
        yield Path(f"results/phase3b/holdout_stencil_control/xi{xi:g}/metric_{index:03d}_n{n}.npz")


def run(args):
    records=registry(args)
    selected=[records[i]for i in args.field_ids]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    calibration=load_calibration(args.calibration)
    config={"registry":args.registry,"registry_sha256":sha256_file(args.registry),
        "calibration":args.calibration,"calibration_sha256":sha256_file(args.calibration),
        "points":args.points,"bins":args.bins,"stencil":args.stencil,"boundary":"torus",
        "probes":"central and valid-model low/high; duplicates retained as labels, not recomputed",
        "root_sampling":"same iid uniforms per physical field across all xi/gamma/resolutions",
        "gauge":"grounded raw metric; unit-mass Weyl gauge derived later",
        "cache_policy":"exact source hash, metric convention, probe gamma and16points must match"}
    source_paths=["src/phase3c_fields.py","experiments/phase3c_fields.py","src/log_field_3d.py",
        "src/gmc_3d.py","src/lfpp_3d.py","src/parameter_conversion_3b.py","src/metric_measure_3a.py"]
    if (output/"metadata.json").exists():
        saved=json.loads((output/"metadata.json").read_text())
        if saved["configuration"]!=config:raise ValueError("Existing campaign configuration differs")
        if any(sha256_file(p)!=h for p,h in saved["source_hashes"].items()):
            raise ValueError("Producer sources changed; use new output directory")
    else:
        snapshot=output/"source_snapshot";snapshot.mkdir()
        for path in source_paths:shutil.copyfile(path,snapshot/Path(path).name)
        write_json_new(output/"metadata.json",{"configuration":config,"provenance":environment_manifest(),
            "source_hashes":{p:sha256_file(p)for p in source_paths}})
    started=time.perf_counter();checkpoints=[]
    for row in selected:
        uniforms=np.random.default_rng(row["root_seed"]).random(args.points)
        with np.load(row["source_field"]) as archive:
            fields={n:archive[f"h{n}"]for n in args.sizes}
        for n,h in fields.items():
            _,_,variance=spectral_configuration(n)
            for xi in args.xis:
                folder=output/f"xi{xi:g}";folder.mkdir(exist_ok=True)
                path=folder/f"metric_{row['realization']:03d}_n{n}.npz"
                if path.exists():
                    with np.load(path)as archive:
                        old=json.loads(str(archive["header"]))
                        if old["source_field_sha256"]!=row["source_field_sha256"]:
                            raise ValueError("Existing checkpoint field identity changed")
                    continue
                tick=time.perf_counter()
                conversion=calibration.evaluate(xi)
                probes=gamma_probes(conversion)
                measures={gamma:gmc_weights(h,gamma,variance)for _,gamma in probes}
                points=[sampled_points(measures[g].probabilities,uniforms)for _,g in probes]
                cached=next((p for p in cache_candidates(row,n,xi)if compatible_metric(p,
                    source_hash=row["source_field_sha256"],n=n,xi=xi,stencil=args.stencil,
                    probes=probes,points=points)),None)
                if cached is not None:
                    with np.load(cached)as archive:
                        arrays={key:archive[key]for key in archive.files if key!="header" and key not in ["distance_roots","distances"]}
                        previous=json.loads(str(archive["header"]))
                    masses=previous["probes"]
                    metric_meta=previous.get("metric_metadata",{})
                    method="reused matching complete metric checkpoint"
                else:
                    unique_roots=np.unique(np.concatenate(points))
                    graph,metric_meta=build_lfpp_graph(h,xi,stencil=args.stencil,boundary="torus")
                    distances=root_distances(graph,unique_roots)
                    results={}
                    for gamma,measure in measures.items():
                        roots=points[next(i for i,(_,g)in enumerate(probes)if g==gamma)]
                        d=distances[np.searchsorted(unique_roots,roots)]
                        results[gamma]={**weighted_root_cdfs(d,measure.probabilities.ravel(),bins=args.bins),
                                        "matrix":d[:,roots]}
                    masses=[{"gamma":g,"band":band,"log_total_mass":measures[g].log_total_mass,
                        "total_mass":measures[g].total_mass,"collision_probability":float(np.sum(measures[g].probabilities**2)),
                        "maximum_atom":float(measures[g].probabilities.max()),
                        "observed_duplicate_point_fraction":float(1-len(np.unique(p))/len(p))}
                        for (band,g),p in zip(probes,points)]
                    arrays={"grid":np.stack([results[g]["grid"]for _,g in probes]),
                        "cdfs":np.stack([results[g]["cdfs"]for _,g in probes]).astype(np.float32),
                        "root_means":np.stack([results[g]["root_means"]for _,g in probes]),
                        "matrices":np.stack([results[g]["matrix"]for _,g in probes]),"points":np.stack(points),
                        "bin_width":np.array([results[g]["distance_bin_width"]for _,g in probes])}
                    del distances,graph
                    method="new Dijkstra roots; shared across duplicate gamma/point rows"
                # Reused masses get the same explicit collision metadata as new
                # checkpoints, without altering their distance/CDF arrays.
                for mass,p in zip(masses,points):
                    measure=measures[mass["gamma"]]
                    mass["maximum_atom"]=float(measure.probabilities.max())
                    mass["observed_duplicate_point_fraction"]=float(1-len(np.unique(p))/len(p))
                header={**row,"n":n,"sampling_n":n,"xi":xi,"probes":masses,
                    "conversion":conversion,"stencil":args.stencil,"boundary":"torus",
                    "field_variance":variance,"metric_metadata":metric_meta,
                    "seconds":time.perf_counter()-tick,"method":method,
                    "source_metric":str(cached)if cached else None,
                    "source_metric_sha256":sha256_file(cached)if cached else None,
                    "control":"calibrated_log_field","points":args.points,
                    "sampling_count":"one physical field; gamma probes/roots/xi are coupled observations"}
                write_npz_new(path,**arrays,header=json.dumps(header))
                checkpoints.append({"path":str(path),"seconds":header["seconds"],"reused":cached is not None})
                print(json.dumps({"field":row["realization"],"n":n,"xi":xi,
                    "seconds":header["seconds"],"reused":cached is not None}),flush=True)
    summary={"field_ids":args.field_ids,"sizes":args.sizes,"xis":args.xis,
        "new_checkpoints_this_invocation":len(checkpoints),"reused":sum(r["reused"]for r in checkpoints),
        "new_dijkstra":sum(not r["reused"]for r in checkpoints),"seconds":time.perf_counter()-started,
        "peak_rss_native":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"checkpoints":checkpoints}
    batch=output/"batches";batch.mkdir(exist_ok=True)
    label=args.label or f"ids{min(args.field_ids)}_{max(args.field_ids)}_N{'_'.join(map(str,args.sizes))}"
    write_json_new(batch/f"{label}.json",summary)
    print(json.dumps({k:v for k,v in summary.items()if k!="checkpoints"}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=["inventory","run"])
    parser.add_argument("--registry",default="results/phase3c/fields/registry.json")
    parser.add_argument("--seed",type=int,default=202611123100)
    parser.add_argument("--output",default="results/phase3c/fields/screen")
    parser.add_argument("--calibration",default="results/phase3b/calibration/analysis/calibration.json")
    parser.add_argument("--field-ids",nargs="+",type=int)
    parser.add_argument("--xis",nargs="+",type=float,default=list(DEFAULT_XIS))
    parser.add_argument("--sizes",nargs="+",type=int,default=[32])
    parser.add_argument("--points",type=int,default=16)
    parser.add_argument("--bins",type=int,default=1024)
    parser.add_argument("--stencil",choices=["26","primitive2","axis"],default="26")
    parser.add_argument("--label")
    args=parser.parse_args()
    if args.command=="inventory":
        rows=registry(args);print(json.dumps({"families":len(rows),"registry":args.registry}))
    elif not args.field_ids:
        parser.error("run requires explicit --field-ids")
    else:run(args)


if __name__=="__main__":main()
