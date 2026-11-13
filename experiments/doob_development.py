"""Small cost/failure pilot for the interior-ball sampler, not a sweep."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from src.doob_excursions import sample_doob_excursion
from src.mated_crt import graph_from_excursion
from src.graph_metrics import metric_sample
from src.provenance import environment_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/phase2b/doob_cost_pilot.json")
    parser.add_argument("--samples",type=int,default=4)
    parser.add_argument("--seed",type=int,default=2026111051)
    parser.add_argument("--fem",action="store_true",help="Tiny enriched-FEM attempt-cost pilot, not validation")
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if args.fem:
        from src.cone_sampling import enriched_cone_eigenpair, FastFEMMode
        matrices=[("I3",np.eye(3)),("unequal",np.array([[1.,-.2,.1],[-.2,1,.35],[.1,.35,1.]]))]
        attempts=[]
        for matrix_index,(name,C) in enumerate(matrices):
            for refinement in [4,5]:
                eigenpair=enriched_cone_eigenpair(C,refinement,strategy="edge")
                mode=FastFEMMode(eigenpair,C)
                for realization in range(args.samples):
                    seed=int(np.random.SeedSequence([args.seed,matrix_index,refinement,realization]).generate_state(1)[0])
                    started=time.perf_counter()
                    row=dict(name=name,covariance=C.tolist(),refinement=refinement,
                             seed=seed,realization=realization,n=64,
                             kernel_status="nonharmonic_interpolant_PROVISIONAL")
                    try:
                        sample=sample_doob_excursion(64,C,np.random.default_rng(seed),
                            step_size=.02,relative_step=.2,origin_cutoff=.01,
                            eigenpair=mode,max_exit_proposals=20000,max_balls=3000,
                            max_steps=300000)
                        graph=graph_from_excursion(sample,64)
                        row.update(status="success",sampler_metadata=sample.metadata,
                                   ties=graph.metadata["symbolic_ties_by_coordinate"])
                    except (ValueError,RuntimeError) as error:
                        row.update(status="failed",error_type=type(error).__name__,error=str(error))
                    row["seconds"]=time.perf_counter()-started
                    attempts.append(row)
                    print(json.dumps({k:row[k] for k in ["name","refinement","realization","status","seconds"]}),flush=True)
        path.write_text(json.dumps(dict(provenance=environment_manifest(),seed=args.seed,
            purpose="FEM_cost_and_failure_obstacle_pilot_no_dimension_inference",attempts=attempts),indent=2))
        return
    settings=[dict(step_size=.02,relative_step=.2,origin_cutoff=.01),
              dict(step_size=.01,relative_step=.15,origin_cutoff=.005),
              dict(step_size=.005,relative_step=.1,origin_cutoff=.003)]
    rows=[]
    for index,options in enumerate(settings):
        for realization in range(args.samples):
            seed=int(np.random.SeedSequence([args.seed,index,realization]).generate_state(1)[0])
            started=time.perf_counter()
            row=dict(settings=options,realization=realization,seed=seed,n=128)
            try:
                sample=sample_doob_excursion(128,np.eye(3),np.random.default_rng(seed),**options)
                sampled=time.perf_counter()
                graph=graph_from_excursion(sample,128)
                metrics=metric_sample(graph.adjacency,np.random.default_rng(seed+1),roots=4)
                row.update(status="success",sampler_seconds=sampled-started,
                    total_seconds=time.perf_counter()-started,
                    sampler_metadata=sample.metadata,
                    ties=graph.metadata["symbolic_ties_by_coordinate"],
                    mean_distance=metrics["mean_distance"],
                    degree_mean=metrics["degree"]["mean"])
            except (ValueError,RuntimeError) as error:
                # Preserve every failed attempt; never silently replace paths.
                row.update(status="failed",error_type=type(error).__name__,error=str(error),
                           total_seconds=time.perf_counter()-started)
            rows.append(row)
    summary=[]
    for options in settings:
        group=[r for r in rows if r["settings"]==options]
        success=[r for r in group if r["status"]=="success"]
        summary.append(dict(settings=options,attempts=len(group),successes=len(success),
            mean_sampler_seconds=float(np.mean([r["sampler_seconds"] for r in success])) if success else None,
            mean_hitting_time=float(np.mean([r["sampler_metadata"]["hitting_times"] for r in success])) if success else None,
            mean_steps=float(np.mean([sum(d["step_count"] for d in r["sampler_metadata"]["leg_diagnostics"]) for r in success])) if success else None,
            failures=[r for r in group if r["status"]=="failed"]))
    path.write_text(json.dumps(dict(provenance=environment_manifest(),seed=args.seed,
        purpose="small_unconditional_attempt_cost_pilot_not_distribution_validation",
        summary=summary,rows=rows),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
