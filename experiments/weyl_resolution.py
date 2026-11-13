"""Coupled nested-grid error diagnostic for the special Weyl cone sampler."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from src.weyl_excursions import sample_weyl_excursion
from src.brownian_excursions import ExcursionSample
from src.mated_crt import graph_from_excursion, cell_minima
from src.graph_metrics import bfs_distances
from src.provenance import environment_manifest


def nested_sample(sample,n,oversample):
    grid = sample.times*n*oversample
    mask = np.isclose(grid,np.rint(grid),atol=1e-8,rtol=0)
    if "junction_time" in sample.metadata:
        mask |= sample.times == sample.metadata["junction_time"]
    metadata = dict(sample.metadata,oversample=oversample,
                    coupling="nested_from_same_finest_continuum_skeleton")
    return ExcursionSample(sample.times[mask],sample.values[mask],None,metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/weyl_resolution")
    parser.add_argument("--seed",type=int,default=2026111004)
    parser.add_argument("--graphs",type=int,default=24)
    parser.add_argument("--n",type=int,default=512)
    parser.add_argument("--roots",type=int,default=8)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    records, arrays, summary = [],{},[]
    for dimension in (2,3):
        all_minima = []
        for index in range(args.graphs):
            path_seed = int(np.random.SeedSequence([args.seed,dimension,index]).generate_state(1)[0])
            fine = sample_weyl_excursion(args.n,dimension,np.random.default_rng(path_seed),
                                         law="biased",oversample=128)
            roots = np.random.default_rng(path_seed+1).integers(args.n,size=args.roots)
            per_graph = []
            minima_by_level = []
            for factor in (8,32,128):
                sampled = nested_sample(fine,args.n,factor)
                graph = graph_from_excursion(sampled,args.n)
                adjacency = graph.to_sparse()
                means = [float(bfs_distances(adjacency,int(root)).mean()) for root in roots]
                per_graph.append({"dimension":dimension,"realization":index,"path_seed":path_seed,
                    "n":args.n,"law":"biased","oversample":factor,"roots":roots.tolist(),
                    "root_means":means,"mean_distance":float(np.mean(means)),
                    "ties":graph.metadata["symbolic_ties_by_coordinate"],
                    "edges":set(map(tuple,graph.edges))})
                minima_by_level.append(cell_minima(sampled,args.n))
            finest_edges = per_graph[-1]["edges"]
            for row in per_graph:
                edges = row.pop("edges")
                row["edge_jaccard_vs_128"] = len(edges&finest_edges)/len(edges|finest_edges)
                records.append(row)
            all_minima.append(minima_by_level)
        arrays[f"cell_minima_dimension_{dimension}"] = np.asarray(all_minima)
        for factor in (8,32,128):
            coarse = [r for r in records if r["dimension"]==dimension and r["oversample"]==factor]
            fine = [r for r in records if r["dimension"]==dimension and r["oversample"]==128]
            difference = np.array([a["mean_distance"]-b["mean_distance"] for a,b in zip(coarse,fine)])
            ratios = np.array([a["mean_distance"]/b["mean_distance"]-1 for a,b in zip(coarse,fine)])
            summary.append({"dimension":dimension,"oversample":factor,"graphs":args.graphs,
                            "mean_paired_distance_difference_vs_128":float(difference.mean()),
                            "paired_difference_standard_error":float(difference.std(ddof=1)/np.sqrt(args.graphs)),
                            "mean_relative_distance_difference_vs_128":float(ratios.mean()),
                            "relative_difference_standard_error":float(ratios.std(ddof=1)/np.sqrt(args.graphs)),
                            "mean_edge_jaccard_vs_128":float(np.mean([r["edge_jaccard_vs_128"] for r in coarse])),
                            "mean_tie_count_per_coordinate":float(np.mean([r["ties"] for r in coarse]))})
    np.savez_compressed(output/"nested_minima.npz",**arrays)
    record = {"seed":args.seed,"n":args.n,"roots_per_graph":args.roots,"independent_graphs_per_dimension":args.graphs,
              "law":"paper_biased_special_A2_cone","levels":[8,32,128],"records":records,"summary":summary,
              "provenance":environment_manifest(),"elapsed_seconds":time.perf_counter()-started,
              "limitation":"128 remains a finite grid; nested differences do not certify continuum error"}
    (output/"resolution.json").write_text(json.dumps(record,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    for dimension in (2,3):
        rows = [r for r in summary if r["dimension"]==dimension]
        axes[0].errorbar([r["oversample"] for r in rows],
                        [100*r["mean_relative_distance_difference_vs_128"] for r in rows],
                        yerr=[196*r["relative_difference_standard_error"] for r in rows],
                        fmt="o-",label=f"{dimension} coordinates")
        axes[1].plot([r["oversample"] for r in rows],[r["mean_edge_jaccard_vs_128"] for r in rows],
                     "o-",label=f"{dimension} coordinates")
    axes[0].set(xscale="log",xlabel="time samples per graph cell",ylabel="paired mean-distance change vs128 (%)")
    axes[1].set(xscale="log",xlabel="time samples per graph cell",ylabel="edge Jaccard similarity vs128")
    for axis in axes: axis.legend()
    fig.suptitle(f"Coupled Weyl minima resolution; n={args.n}, {args.graphs} independent graphs per case")
    fig.savefig(output/"resolution.png",dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
