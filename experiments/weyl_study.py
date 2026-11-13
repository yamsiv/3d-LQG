"""Coupled PL-resolution scaling diagnostic for the tractable A2 Weyl cones."""
import json
from pathlib import Path
import time
import numpy as np

from src.brownian_excursions import ExcursionSample
from src.weyl_excursions import sample_weyl_excursion, weyl_covariance
from src.mated_crt import graph_from_excursion
from src.graph_metrics import metric_sample, apparent_ball_exponent
from src.provenance import environment_manifest
from experiments.reproduce_budd_castro import graph_seeds, analyze_records, make_figures


def coarsen(sample, n, oversample):
    grid = np.linspace(0, 1, n*oversample+1)
    if "junction_time" in sample.metadata:
        grid = np.unique(np.r_[grid, sample.metadata["junction_time"]])
    indices = np.searchsorted(sample.times, grid)
    assert np.allclose(sample.times[indices], grid, rtol=0, atol=1e-13)
    metadata = dict(sample.metadata, oversample=oversample,
                    coupling="nested skeleton of one finest matrix bridge")
    return ExcursionSample(grid, sample.values[indices], None, metadata)


def main():
    seed = 2026111030
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    graphs = 16
    prefix = Path("results/data")
    path = prefix/"weyl_coupled.jsonl"
    if path.exists():
        raise FileExistsError(path)
    prefix.mkdir(parents=True, exist_ok=True)
    meta = dict(provenance=environment_manifest(), seed=seed, sizes=sizes,
                graphs_per_size_dimension=graphs, dimensions=[2,3],
                finest_oversample=128, nested_oversamples=[32,128], roots_per_graph=8,
                law="biased", status="exact_grid_law_PL_minima_resolution_diagnostic")
    path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    records = []
    with path.open("w") as stream:
        for dimension in [2, 3]:
            for n in sizes:
                start = time.perf_counter()
                for realization in range(graphs):
                    graph_seed, metric_seed = graph_seeds(seed+dimension, n, realization)
                    sample = sample_weyl_excursion(n, dimension, np.random.default_rng(graph_seed),
                                                  law="biased", oversample=128)
                    previous = None
                    for oversample in [32,128]:
                        graph = graph_from_excursion(coarsen(sample,n,oversample), n)
                        row = metric_sample(graph.adjacency, np.random.default_rng(metric_seed), roots=8)
                        row.update(dimension=dimension, oversample=oversample,
                                   realization=realization, graph_seed=graph_seed, metric_seed=metric_seed,
                                   covariance=weyl_covariance(dimension).tolist(),
                                   sampler_metadata=sample.metadata, graph_metadata=graph.metadata,
                                   ball_exponent_diagnostic=apparent_ball_exponent(row["mean_ball_volumes"], n),
                                   interpretation="finite_size_PL_minima_effective_exponent_only")
                        if previous is not None:
                            coarse_edges = set(map(tuple,previous["edges"]))
                            fine_edges = set(map(tuple,graph.edges))
                            row["coupled_resolution"] = {
                                "coarse_oversample": 32, "fine_oversample": 128,
                                "mean_distance_change": row["mean_distance"]-previous["mean_distance"],
                                "relative_mean_distance_change": row["mean_distance"]/previous["mean_distance"]-1,
                                "edge_jaccard":len(coarse_edges&fine_edges)/len(coarse_edges|fine_edges)}
                        else:
                            previous = {"edges":graph.edges,"mean_distance":row["mean_distance"]}
                        records.append(row)
                        stream.write(json.dumps(row)+"\n")
                        stream.flush()
                print(json.dumps({"dimension":dimension,"n":n,"graphs":graphs,
                                  "seconds":time.perf_counter()-start}),flush=True)
    summaries = []
    for dimension in [2,3]:
        for oversample in [32,128]:
            subset = [r for r in records if r["dimension"]==dimension and r["oversample"]==oversample]
            output = prefix/f"weyl_d{dimension}_o{oversample}.analysis.json"
            analysis = analyze_records(subset, output, bootstrap=300, seed=seed+dimension+oversample)
            make_figures(subset, analysis, Path("results/figures")/f"weyl_d{dimension}_o{oversample}")
            summaries.append(dict(dimension=dimension,oversample=oversample,
                                  effective_mean_fit=analysis["distance_fits"]["mean"][0],
                                  collapse=analysis["histogram_collapse"]))
    path.with_suffix(".summary.json").write_text(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
