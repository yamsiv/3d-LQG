"""Paired nested-grid local-observable error budget for the A2 reference."""
import json
from pathlib import Path
import time
import numpy as np
from src.weyl_excursions import sample_weyl_excursion,thin_weyl_sample
from src.mated_crt import graph_from_excursion
from src.provenance import environment_manifest
from experiments.validate_doob import observables


def main():
    output=Path("results/phase2b/weyl_minima_budget")
    output.mkdir(parents=True,exist_ok=True)
    path=output/"records.jsonl"
    if path.exists():raise FileExistsError(path)
    seed=2026111163;n=256;graphs=32
    records=[]
    start=time.perf_counter()
    with path.open("w") as stream:
        for realization in range(graphs):
            graph_seed=int(np.random.SeedSequence([seed,realization]).generate_state(1)[0])
            sample=sample_weyl_excursion(n,3,np.random.default_rng(graph_seed),law="biased",oversample=1024)
            finest_edges=set(map(tuple,graph_from_excursion(sample,n).edges))
            for q in [128,256,1024]:
                subsample=thin_weyl_sample(sample,n,q)
                row=observables(subsample,n,graph_seed+1)
                edges=set(map(tuple,graph_from_excursion(subsample,n).edges))
                row.update(n=n,oversample=q,realization=realization,seed=graph_seed,
                           edge_jaccard_vs_q1024=len(edges&finest_edges)/len(edges|finest_edges))
                records.append(row);stream.write(json.dumps(row)+"\n");stream.flush()
    comparisons=[]
    fine=[r for r in records if r["oversample"]==1024]
    for q in [128,256]:
        coarse=[r for r in records if r["oversample"]==q]
        features={}
        for name in fine[0]["features"]:
            difference=np.array([a["features"][name]-b["features"][name] for a,b in zip(coarse,fine)])
            features[name]={"coarse_minus_fine_mean":float(difference.mean()),
                            "paired_mc_se":float(difference.std(ddof=1)/np.sqrt(graphs))}
        comparisons.append(dict(coarse_q=q,fine_q=1024,features=features,
            edge_jaccard_mean=float(np.mean([r["edge_jaccard_vs_q1024"] for r in coarse]))))
    (output/"summary.json").write_text(json.dumps(dict(provenance=environment_manifest(),
        n=n,independent_paths=graphs,finest_q=1024,seed=seed,comparisons=comparisons,
        elapsed_seconds=time.perf_counter()-start,
        status="paired_P1_refinement_error_diagnostic_not_certified_continuum_error"),indent=2))
    print("Saved",output,"seconds",time.perf_counter()-start,flush=True)


if __name__=="__main__":main()
