"""Small diagnostic covariance pilot, gated against asymptotic interpretation.

The general-covariance sampler targets only a positive *grid* Gaussian bridge,
with finite MCMC burn-in and piecewise-linear minima. These records test whether
it is ready for a future scaling study. No Hausdorff exponents are fitted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from src.brownian_excursions import sample_conditioned_bridge, sample_independent_excursion
from src.mated_crt import graph_from_excursion, validate_graph
from src.graph_metrics import metric_sample
from experiments.reproduce_budd_castro import source_metadata, graph_seeds


def covariance_points():
    points = []
    for rho in (-.49, -.3, 0., .3, .6):
        c = np.full((3, 3), rho)
        np.fill_diagonal(c, 1.)
        points.append((f"equicorr_{rho:+.2f}", c, "equicorrelated"))
    # Analytic birectangular cones (paper Eq.29); original numerical table has
    # birectangular rows at alpha=pi/3,pi/2,2pi/3. Only finite-grid geometry here.
    for alpha in (np.pi / 3, 2 * np.pi / 3):
        c = np.eye(3)
        c[0, 1] = c[1, 0] = -np.cos(alpha)
        points.append((f"birectangular_alpha_{alpha:.5f}", c, "birectangular"))
    # Deliberately non-isosceles, unequal correlations, checked SPD below.
    for values in ((-.2, .1, .35), (-.35, -.1, .2)):
        a, b, c23 = values
        c = np.array([[1., a, b], [a, 1., c23], [b, c23, 1.]])
        points.append(("general_" + "_".join(f"{v:+.2f}" for v in values), c, "three_parameter"))
    for _, c, _ in points:
        np.linalg.cholesky(c)
    return points


def run_pilot(output, *, sizes=(128, 256, 512), graphs=4, roots=4,
              oversample=2, sweeps=32, seed=76213, points=None,
              force_grid_identity=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = covariance_points() if points is None else points
    metadata = source_metadata()
    metadata.update({"sizes": list(sizes), "graphs_per_size_per_covariance": graphs,
                     "roots_per_graph": roots, "oversample": oversample, "sweeps": sweeps,
                     "seed": seed, "law": "finite_grid_unbiased_approximation",
                     "purpose": "sampler_diagnostic_covariance_pilot_NOT_Hausdorff_estimation",
                     "force_grid_identity": force_grid_identity,
                     "points": [{"name": name, "covariance": c.tolist(), "family": family}
                                for name, c, family in points]})
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with output.open("w") as stream:
        for point_index, (name, c, family) in enumerate(points):
            start = time.perf_counter()
            for n in sizes:
                for realization in range(graphs):
                    graph_seed, metric_seed = graph_seeds(seed + point_index, n, realization)
                    rng = np.random.default_rng(graph_seed)
                    t = time.perf_counter()
                    if np.array_equal(c, np.eye(3)) and not force_grid_identity:
                        sample = sample_independent_excursion(n, 3, rng, law="unbiased")
                    else:
                        sample = sample_conditioned_bridge(n, c, rng, oversample=oversample, sweeps=sweeps)
                    graph = graph_from_excursion(sample, n)
                    generated = time.perf_counter()
                    record = metric_sample(graph.adjacency, np.random.default_rng(metric_seed), roots=roots,
                                           exact_diameter_limit=32)
                    record.update({"name": name, "family": family, "realization": realization,
                                   "covariance": c.tolist(), "eigenvalues": np.linalg.eigvalsh(c).tolist(),
                                   "condition_number": float(np.linalg.cond(c)),
                                   # Paper order alpha=C12,beta=C23,theta=C13.
                                   "cone_angles": np.arccos(-c[[0, 1, 0], [1, 2, 2]]).tolist(),
                                   "graph_seed": graph_seed, "metric_seed": metric_seed,
                                   "sampler_metadata": sample.metadata, "graph_metadata": graph.metadata,
                                   "generation_seconds": generated - t,
                                   "metric_seconds": time.perf_counter() - generated,
                                   "graph_validation": validate_graph(graph),
                                   "interpretation": "finite_grid_diagnostic_only"})
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
            print(json.dumps({"point": name, "seconds": time.perf_counter() - start}), flush=True)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    summarize(records, output.with_suffix(".summary.json"))
    return records


def summarize(records, output):
    summary = {"gate": "CLOSED", "extensive_parameter_sweep_performed": False,
               "reason": "No certified MCMC equilibration or continuous-excursion/grid convergence for general covariance; PL minima ties remain.",
               "asymptotic_dimensions_fitted": False, "groups": []}
    for name in dict.fromkeys(r["name"] for r in records):
        for n in sorted({r["n"] for r in records if r["name"] == name}):
            group = [r for r in records if r["name"] == name and r["n"] == n]
            values = np.array([r["mean_distance"] for r in group])
            ties = np.array([r["graph_metadata"]["symbolic_ties_by_coordinate"] for r in group])
            summary["groups"].append({"name": name, "n": n, "graphs": len(group),
                                       "mean_distance": float(values.mean()),
                                       "distance_mc_se": float(values.std(ddof=1) / np.sqrt(len(group))),
                                       "mean_simple_degree": float(np.mean([r["degree"]["mean"] for r in group])),
                                       "mean_tie_fraction": float(ties.mean() / (n - 2)),
                                       "covariance": group[0]["covariance"],
                                       "condition_number": group[0]["condition_number"],
                                       "cone_angles": group[0]["cone_angles"]})
    Path(output).write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in dict.fromkeys(g["name"] for g in summary["groups"]):
        group = [g for g in summary["groups"] if g["name"] == name]
        axes[0].errorbar([g["n"] for g in group], [g["mean_distance"] for g in group],
                         yerr=[1.96*g["distance_mc_se"] for g in group], marker="o", label=name)
        axes[1].plot([g["n"] for g in group], [g["mean_tie_fraction"] for g in group], "o-", label=name)
    axes[0].set(xscale="log", xlabel="Vertices n", ylabel="Mean pair graph distance")
    axes[1].set(xscale="log", xlabel="Vertices n", ylabel="Fraction of tied PL cell minima")
    axes[0].legend(fontsize="x-small")
    fig.suptitle("Finite-grid MCMC covariance diagnostics; unvalidated continuous-law approximation")
    fig.tight_layout()
    figure_dir = Path("results/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / (Path(output).stem + ".png"), dpi=170)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/data/covariance_pilot.jsonl")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--graphs", type=int, default=4)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=76213)
    args = parser.parse_args()
    run_pilot(args.output, sizes=args.sizes, graphs=args.graphs, roots=args.roots,
              oversample=args.oversample, sweeps=args.sweeps, seed=args.seed)


if __name__ == "__main__":
    main()
