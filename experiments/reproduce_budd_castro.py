"""Reproducible, graph-cluster distance study (Budd--Castro §4.2).

Run from the repository root with ``python -m experiments.reproduce_budd_castro``.
Every realization has a deterministic separate RNG seed and is saved immediately
as one JSONL record. Reanalysis never requires resimulating a graph. This program
does not claim to reproduce the paper's sampling volume or asymptotic accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lqg-matplotlib")

from src.graph_metrics import metric_sample, apparent_ball_exponent
from src.scaling_analysis import (window_stability, pooled_histogram,
                                  collapse_histograms, fixed_offset_sensitivity)


def source_metadata():
    source_files = (list(Path("src").glob("*.py")) + list(Path("experiments").glob("*.py"))
                    + list(Path("tests").glob("*.py")) + [Path("papers/Budd_Castro.pdf")])
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except subprocess.CalledProcessError:
        commit, status = None, None
    import scipy
    import matplotlib
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "git_status": status, "source_sha256": hashes, "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "rng": "numpy.random.PCG64 via default_rng, SeedSequence per realization",
            "primary_source": "Budd and Castro, arXiv:2207.05355v1 (12 July 2022)",
            "argv": sys.argv, "working_directory": str(Path.cwd()),
            "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}}


def graph_seeds(seed, n, realization):
    states = np.random.SeedSequence([seed, n, realization]).generate_state(2, dtype=np.uint64)
    return int(states[0]), int(states[1])


def generate_graph(n, covariance, rng, law, oversample, sampler_options=None):
    """Generator adapter: kept in one place as the sampler API evolves."""
    from src.brownian_excursions import sample_excursion
    from src.mated_crt import graph_from_excursion
    sample = sample_excursion(n, covariance, rng, law=law, oversample=oversample,
                              **(sampler_options or {}))
    graph = graph_from_excursion(sample, n)
    return sample, graph


def run_study(*, sizes, graphs, roots, covariance, law, seed, output,
              oversample=1, resume=False, sampler_options=None,
              secondary=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix(".metadata.json")
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; select a new output or use --resume")
    old_records = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if old_records:
        old_meta = json.loads(meta_path.read_text())
        for key, value in {"sizes": list(sizes), "graphs_per_size": graphs,
                           "roots_per_graph": roots, "covariance": np.asarray(covariance).tolist(),
                           "law": law, "seed": seed, "oversample": oversample,
                           "sampler_options": sampler_options or {}}.items():
            if old_meta.get(key) != value:
                raise ValueError(f"resume configuration differs: {key}")
    else:
        metadata = source_metadata()
        metadata.update({"sizes": list(sizes), "graphs_per_size": graphs, "roots_per_graph": roots,
                         "covariance": np.asarray(covariance).tolist(), "law": law, "seed": seed,
                         "oversample": oversample, "sampler_options": sampler_options or {},
                         "sampling_design": "independent graphs at each size; all targets for uniform roots",
                         "phase": "graph_only"})
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    completed = {(r["n"], r["realization"]) for r in old_records}
    with output.open("a") as stream:
        for n in sizes:
            start_size = time.perf_counter()
            for realization in range(graphs):
                if (n, realization) in completed:
                    continue
                graph_seed, metric_seed = graph_seeds(seed, n, realization)
                start = time.perf_counter()
                sample, graph = generate_graph(n, covariance, np.random.default_rng(graph_seed),
                                               law, oversample, sampler_options)
                generated = time.perf_counter()
                record = metric_sample(graph.adjacency, np.random.default_rng(metric_seed), roots=roots)
                record.update({"realization": realization, "graph_seed": graph_seed,
                               "metric_seed": metric_seed, "law": law,
                               "sampler_metadata": sample.metadata, "graph_metadata": graph.metadata,
                               "generation_seconds": generated - start,
                               "metric_seconds": time.perf_counter() - generated,
                               "ball_exponent_diagnostic": apparent_ball_exponent(record["mean_ball_volumes"], n)})
                if secondary and realization == 0 and n == max(sizes):
                    from src.secondary_observables import monte_carlo_returns, spectral_dimension_pilot
                    returns = monte_carlo_returns(graph.adjacency, np.random.default_rng(metric_seed + 1),
                                                  walkers=10000, max_steps=128)
                    record["lazy_walk_returns"] = returns
                    record["spectral_dimension_diagnostic"] = spectral_dimension_pilot(returns)
                if realization == 0 and n in (min(sizes), max(sizes)):
                    np.savez_compressed(output.with_name(output.stem + f"_graph_n{n}.npz"),
                                        edges=np.asarray(graph.edges),
                                        multiplicities=np.asarray(graph.multiplicities),
                                        covariance=np.asarray(covariance),
                                        graph_seed=np.asarray(graph_seed, dtype=np.uint64))
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            print(json.dumps({"n": n, "graphs": graphs, "seconds": time.perf_counter() - start_size}), flush=True)
    return [json.loads(line) for line in output.read_text().splitlines()]


def analyze_records(records, output, *, bootstrap=500, seed=0):
    analysis = {"distance_fits": {}, "diameter_fits": {}, "ball_diagnostics": {},
                "uncertainty_note": "95% graph-cluster bootstrap covers Monte Carlo variability only"}
    for i, statistic in enumerate(("mean", "median", "q25", "q75")):
        analysis["distance_fits"][statistic] = window_stability(records, statistic=statistic,
                                                               bootstrap=bootstrap, seed=seed + 100 * i)
    for i, observable in enumerate(("diameter_lower", "diameter_upper")):
        analysis["diameter_fits"][observable] = window_stability(records, observable=observable,
                                                                 bootstrap=bootstrap, seed=seed + 1000 + i)
    analysis["histogram_collapse"] = collapse_histograms(records)
    analysis["fixed_offset_sensitivity"] = fixed_offset_sensitivity(records)
    for n in sorted({r["n"] for r in records}):
        group = [r for r in records if r["n"] == n]
        slopes = [r["ball_exponent_diagnostic"]["slope"] for r in group
                  if r.get("ball_exponent_diagnostic") is not None]
        means = [r["mean_distance"] for r in group]
        within = [r["within_graph_root_mean_variance"] for r in group]
        observed_variance = float(np.var(means, ddof=1))
        root_noise = float(np.mean([r["within_graph_root_mean_variance"] / len(r["roots"]) for r in group]))
        analysis["ball_diagnostics"][n] = {"graphs_with_usable_window": len(slopes),
                                           "mean_apparent_slope": float(np.mean(slopes)) if slopes else None,
                                           "between_graph_sd": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else None}
        analysis.setdefault("graph_variability", {})[n] = {
            "between_graph_sd_of_root_averaged_distance": float(np.std(means, ddof=1)),
            "mean_within_graph_root_mean_variance": float(np.mean(within)),
            "mean_estimated_variance_from_finite_root_sampling": root_noise,
            "estimated_between_graph_variance_after_root_noise_subtraction": max(0., observed_variance - root_noise),
            "roots_per_graph": len(group[0]["roots"])}
    Path(output).write_text(json.dumps(analysis, indent=2) + "\n")
    return analysis


def make_figures(records, analysis, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    groups = {n: [r for r in records if r["n"] == n] for n in sorted({r["n"] for r in records})}
    first = records[0]
    covariance = np.asarray(first["sampler_metadata"]["covariance"])
    law = first["sampler_metadata"]["law"]
    counts = [len(g) for g in groups.values()]
    count_label = str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
    model_label = f"{len(covariance)}-tree {law} excursion, C=I" if np.array_equal(covariance, np.eye(len(covariance))) else f"{len(covariance)}-tree {law} excursion"
    design_label = f"{count_label} graphs/size, {len(first['roots'])} uniform roots/graph"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for statistic in ("mean", "median", "q25", "q75"):
        fit = analysis["distance_fits"][statistic][0]
        n, y = np.asarray(fit["sizes"]), np.asarray(fit["values"])
        ci = np.asarray(fit["values_ci95"])
        axes[0, 0].errorbar(n, y, yerr=np.maximum(0, np.array([y - ci[:, 0], ci[:, 1] - y])),
                            marker="o", capsize=2, label=statistic)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Vertices n", ylabel="Uniform-pair graph distance")
    axes[0, 0].legend()
    for n, group in groups.items():
        hist = pooled_histogram(group)
        r = np.arange(len(hist))
        mean = np.dot(r, hist)
        axes[0, 1].plot(r / mean, hist * mean, label=str(n))
    axes[0, 1].set(xlabel="Distance / mean distance", ylabel="Mean × pair-distance probability")
    axes[0, 1].legend(title="n", fontsize="small")
    for n, group in groups.items():
        width = max(len(r["mean_ball_volumes"]) for r in group)
        volumes = np.mean([np.pad(r["mean_ball_volumes"],
                                  (0, width - len(r["mean_ball_volumes"])), constant_values=n) for r in group], axis=0)
        axes[1, 0].plot(np.arange(1, width), volumes[1:], label=str(n))
    axes[1, 0].set(xscale="log", yscale="log", xlabel="Graph radius r", ylabel="Mean intrinsic ball vertex count")
    axes[1, 0].legend(title="n", fontsize="small")
    nvalues = np.array(list(groups))
    lo = [np.mean([r["diameter"]["lower"] for r in groups[n]]) for n in nvalues]
    hi = [np.mean([r["diameter"]["upper"] for r in groups[n]]) for n in nvalues]
    axes[1, 1].plot(nvalues, lo, "o-", label="Mean certified lower bound")
    axes[1, 1].plot(nvalues, hi, "s--", label="Mean certified upper bound")
    axes[1, 1].fill_between(nvalues, lo, hi, alpha=.15)
    axes[1, 1].set(xscale="log", yscale="log", xlabel="Vertices n", ylabel="Graph diameter bounds")
    axes[1, 1].legend()
    fig.suptitle(model_label + "; " + design_label + "\nSimple-graph distances; 95% bootstrap bars are Monte Carlo only", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(prefix) + "_geometry.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for statistic in ("mean", "median", "q25", "q75"):
        fits = analysis["distance_fits"][statistic]
        x = [f["sizes"][0] for f in fits]
        y = np.asarray([f["effective_dimension"] for f in fits])
        ci = np.asarray([f["dimension_ci95"] for f in fits])
        axes[0].errorbar(x, y, yerr=np.maximum(0, np.array([y - ci[:, 0], ci[:, 1] - y])),
                         fmt="o-", capsize=2, label=statistic)
    axes[0].set(xscale="log", xlabel="Smallest included graph n", ylabel="Effective dimension 1 / log slope")
    axes[0].legend()
    group = groups[max(groups)]
    width = max(len(r["degree"]["histogram"]) for r in group)
    degree_hist = np.mean([np.pad(r["degree"]["histogram"], (0, width-len(r["degree"]["histogram"]))) / r["n"]
                           for r in group], axis=0)
    axes[1].semilogy(np.arange(width), degree_hist, ".-")
    axes[1].set(xlabel="Simple-graph degree", ylabel="Probability", title=f"n = {max(groups)}")
    fig.suptitle(model_label + "; " + design_label + "\n95% graph-bootstrap bars omit finite-size bias", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(prefix) + "_reliability.png", dpi=170)
    plt.close(fig)


def export_summary(records, output):
    """Readable per-size numerical table; raw graph records remain the source."""
    from src.scaling_analysis import histogram_statistic
    rows = []
    for n in sorted({r["n"] for r in records}):
        group = [r for r in records if r["n"] == n]
        hist = pooled_histogram(group)
        slopes = [r["ball_exponent_diagnostic"]["slope"] for r in group
                  if r.get("ball_exponent_diagnostic") is not None]
        means = np.asarray([r["mean_distance"] for r in group])
        rows.append({"n": n, "graphs": len(group), "roots_per_graph": len(group[0]["roots"]),
                     "mean_distance": histogram_statistic(hist),
                     "mean_distance_mc_se": means.std(ddof=1)/np.sqrt(len(means)),
                     "median_distance": histogram_statistic(hist, "median"),
                     "q25_distance": histogram_statistic(hist, "q25"),
                     "q75_distance": histogram_statistic(hist, "q75"),
                     "mean_diameter_lower_bound": np.mean([r["diameter"]["lower"] for r in group]),
                     "mean_diameter_upper_bound": np.mean([r["diameter"]["upper"] for r in group]),
                     "mean_simple_degree": np.mean([r["degree"]["mean"] for r in group]),
                     "mean_ball_window_slope_diagnostic": np.mean(slopes) if slopes else "",
                     "graphs_with_ball_window": len(slopes),
                     "mean_generation_seconds": np.mean([r["generation_seconds"] for r in group]),
                     "mean_metric_seconds": np.mean([r["metric_seconds"] for r in group])})
    with Path(output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--graphs", type=int, default=32)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=3)
    parser.add_argument("--correlations", type=float, nargs=3, metavar=("C12", "C13", "C23"))
    parser.add_argument("--rho", type=float)
    parser.add_argument("--law", choices=("biased", "unbiased"), default="biased")
    parser.add_argument("--oversample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--output", default="results/data/independent_i3_biased.jsonl")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--secondary", action="store_true")
    args = parser.parse_args()
    if args.correlations and args.rho is not None:
        parser.error("choose correlations or rho")
    covariance = np.eye(args.dimension)
    if args.rho is not None:
        covariance = np.full((args.dimension, args.dimension), args.rho)
        np.fill_diagonal(covariance, 1.)
    if args.correlations:
        if args.dimension != 3:
            parser.error("--correlations requires dimension 3")
        a, b, c = args.correlations
        covariance = np.array([[1., a, b], [a, 1., c], [b, c, 1.]])
    if not np.allclose(covariance, np.eye(args.dimension)):
        parser.error("non-independent covariance studies require the gated parameter_sweep runner")
    if args.analyze_only:
        records = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
    else:
        records = run_study(sizes=sorted(args.sizes), graphs=args.graphs, roots=args.roots,
                            covariance=covariance, law=args.law, seed=args.seed,
                            output=args.output, oversample=args.oversample, resume=args.resume,
                            secondary=args.secondary)
    analysis_path = Path(args.output).with_suffix(".analysis.json")
    analysis = analyze_records(records, analysis_path, bootstrap=args.bootstrap, seed=args.seed + 1)
    make_figures(records, analysis, Path("results/figures") / Path(args.output).stem)
    export_summary(records, Path(args.output).with_suffix(".summary.csv"))
    print(json.dumps({"analysis": str(analysis_path), "mean_fit": analysis["distance_fits"]["mean"][0]}), flush=True)


if __name__ == "__main__":
    main()
