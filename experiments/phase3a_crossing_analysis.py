"""Analyze saved coupled LFPP crossings; never generate a field or graph.

The regression is a finite-resolution proxy on a periodic field restricted to
an open cube. It is not an estimate certified to equal the whole-space LFPP
exponent. All bootstrap rows are complete physical field realizations across
resolutions. Output files refuse overwriting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.provenance import environment_manifest


KINDS = ("site", "edge_gap_corrected", "edge")


def load_crossings(directory, sizes, expected_fields):
    by_seed = {}
    sources = {}
    xis = None
    for path in sorted(Path(directory).glob("metric_*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path) as arrays:
            header = json.loads(str(arrays["header"]))
        n = header["sampling_n"]
        crossing = header.get("crossings", {})
        if n not in sizes or not crossing:
            continue
        if header["n"] != n:
            raise ValueError("This analysis requires mesh_factor=1")
        available = tuple(sorted(map(float, crossing)))
        if xis is None:
            xis = available
        elif available != xis:
            raise ValueError(f"Different xi sets in {path}")
        seed = int(header["field_seed"])
        key = (seed, n)
        if key in sources:
            raise ValueError(f"Duplicate crossing realization: {key}")
        by_seed.setdefault(seed, {})[n] = [
            [float(crossing[str(xi)][kind]) for kind in KINDS] for xi in xis
        ]
        sources[key] = str(path)
    if len(by_seed) != expected_fields:
        raise ValueError(f"Need {expected_fields} complete fields; found {len(by_seed)}")
    incomplete = {s: sorted(set(sizes)-set(rows)) for s, rows in by_seed.items()
                  if set(rows) != set(sizes)}
    if incomplete:
        raise ValueError(f"Incomplete coupled resolution rows: {incomplete}")
    seeds = sorted(by_seed)
    values = np.array([[by_seed[s][n] for n in sizes] for s in seeds])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Positive finite crossings required")
    return seeds, np.array(xis), values, sorted(sources.values())


def fit_lambda(values, sizes, indices):
    """Negative OLS slope of log ensemble median physical distance vs log N."""
    medians = np.median(values[:, indices], axis=0)
    x = np.log(np.asarray(sizes)[indices])
    centered = x-x.mean()
    return -(centered[:, None, None]*np.log(medians)).sum(axis=0)/(centered@centered)


def implicit_quantities(exponent, xi):
    if xi == 0:
        return None, None, None
    q = (1-exponent)/xi
    if q <= np.sqrt(6):
        return float(q), None, None
    gamma = 6/(q+np.sqrt(q*q-6))
    return float(q), float(gamma), float(gamma/xi)


def analyze(args):
    start = time.perf_counter()
    sizes = sorted(set(args.sizes))
    if len(sizes) < 2:
        raise ValueError("At least two resolutions required")
    output = Path(args.output)
    destinations = {suffix: output/f"{args.label}.{suffix}"
                    for suffix in ("json", "csv", "npz", "png", "pdf")}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(path)
    seeds, xis, values, sources = load_crossings(args.input, sizes, args.expected_fields)
    windows = [list(range(len(sizes)))] + [[i, i+1] for i in range(len(sizes)-1)]
    # Avoid duplicate full/adjacent windows for the two-resolution special case.
    windows = list(dict.fromkeys(tuple(w) for w in windows))
    rng = np.random.default_rng(args.seed)
    resamples = rng.integers(len(seeds), size=(args.bootstrap, len(seeds)))
    point = np.stack([fit_lambda(values, sizes, list(w)) for w in windows])
    boot = np.stack([[fit_lambda(values[indices], sizes, list(w)) for w in windows]
                     for indices in resamples])
    rows = []
    for iw, window in enumerate(windows):
        for ix, xi in enumerate(xis):
            for ik, kind in enumerate(KINDS):
                samples = boot[:, iw, ix, ik]
                estimate = point[iw, ix, ik]
                q, gamma, dimension = implicit_quantities(estimate, xi)
                row = {
                    "window": "-".join(str(sizes[i]) for i in window),
                    "kind": kind, "xi": float(xi), "independent_fields": len(seeds),
                    "lambda_finite_proxy": float(estimate),
                    "lambda_mc95_low": float(np.quantile(samples, .025)),
                    "lambda_mc95_high": float(np.quantile(samples, .975)),
                    "Q_finite_proxy": q, "gamma_implicit_proxy": gamma,
                    "dimension_implicit_proxy": dimension,
                    "gamma_valid_bootstrap_fraction": None,
                    "Q_mc95_low": None, "Q_mc95_high": None,
                    "gamma_conditional_mc95_low": None, "gamma_conditional_mc95_high": None,
                    "dimension_conditional_mc95_low": None, "dimension_conditional_mc95_high": None,
                }
                if xi:
                    qs = (1-samples)/xi
                    valid = qs > np.sqrt(6)
                    row["gamma_valid_bootstrap_fraction"] = float(valid.mean())
                    row["Q_mc95_low"], row["Q_mc95_high"] = np.quantile(qs, [.025, .975]).tolist()
                    if valid.any():
                        gs = 6/(qs[valid]+np.sqrt(qs[valid]**2-6))
                        row["gamma_conditional_mc95_low"], row["gamma_conditional_mc95_high"] = np.quantile(gs, [.025, .975]).tolist()
                        row["dimension_conditional_mc95_low"], row["dimension_conditional_mc95_high"] = np.quantile(gs/xi, [.025, .975]).tolist()
                rows.append(row)
    medians = np.median(values, axis=0)
    median_boot = np.median(values[resamples], axis=1)
    metadata = {
        "label": args.label, "input": args.input, "sizes": sizes,
        "xis": xis.tolist(), "kinds": KINDS, "seed": args.seed,
        "bootstrap_draws": args.bootstrap, "independent_fields": len(seeds),
        "windows": [list(w) for w in windows],
        "estimator": "lambda=-OLS slope(log ensemble median physical crossing versus log N)",
        "bootstrap_unit": "complete independent field, all coupled resolutions retained",
        "target": "finite-resolution proxy; periodic torus field restricted to open cube",
        "parameter_conversion": "conditional implicit proxy Q=(1-lambda)/xi; gamma=6/(Q+sqrt(Q^2-6)) only if Q>sqrt6",
        "ci_warning": "MC95 excludes finite-size, infrared, regularization and discretization error; gamma CIs condition on the valid discriminant subset, whose fraction is reported",
        "normalization": "site includes both endpoints; edge_gap_corrected divides edge crossing by (N-1)/N; raw edge retains known finite-grid gap",
        "provenance": environment_manifest(),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "seconds": time.perf_counter()-start,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destinations["npz"],
        field_seeds=np.asarray(seeds, dtype=np.uint64), sizes=np.array(sizes), xis=xis,
        raw_crossings=values, windows=np.array([list(w)+[-1]*(len(sizes)-len(w)) for w in windows]),
        lambda_point=point, lambda_bootstrap=boot, resampling_indices=resamples,
        median_crossings=medians, median_crossings_bootstrap=median_boot,
        header=json.dumps(metadata))
    destinations["json"].write_text(json.dumps({"metadata": metadata, "rows": rows,
        "median_crossings": medians.tolist()}, indent=2)+"\n")
    with destinations["csv"].open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot(destinations, sizes, xis, medians, median_boot, point, boot, windows)
    print(json.dumps({"label": args.label, "fields": len(seeds), "seconds": time.perf_counter()-start,
        "site_full_window": [{k:r[k] for k in ["xi", "lambda_finite_proxy", "lambda_mc95_low", "lambda_mc95_high", "gamma_implicit_proxy", "gamma_valid_bootstrap_fraction"]}
                             for r in rows if r["kind"] == "site" and r["window"] == "-".join(map(str, sizes))]}))


def plot(destinations, sizes, xis, medians, median_boot, point, boot, windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), constrained_layout=True)
    for ix, xi in enumerate(xis):
        q = np.quantile(median_boot[:, :, ix, 0], [.025, .975], axis=0)
        axes[0].plot(sizes, medians[:, ix, 0], "o-", label=f"xi={xi:g}")
        axes[0].fill_between(sizes, q[0], q[1], alpha=.12)
    axes[0].set(xscale="log", yscale="log", xlabel="Field resolution N", ylabel="Median site crossing", title="Open cube in periodic field; paired MC95")
    axes[0].legend(fontsize=7, ncol=2)
    for iw, window in enumerate(windows):
        q = np.quantile(boot[:, iw, :, 0], [.025, .975], axis=0)
        axes[1].plot(xis, point[iw, :, 0], "o-", label="N="+"/".join(str(sizes[i]) for i in window))
        axes[1].fill_between(xis, q[0], q[1], alpha=.12)
    axes[1].set(xlabel="LFPP xi", ylabel="Finite proxy lambda = -slope", title="Window sensitivity and Monte Carlo uncertainty")
    axes[1].legend(fontsize=7)
    for kind, marker, label in [(0, "o", "Site, endpoint included"), (1, "s", "Edge, gap corrected"), (2, "^", "Edge, raw gap")]:
        axes[2].plot(xis, point[0, :, kind], marker+"-", label=label)
    axes[2].set(xlabel="LFPP xi", ylabel="Full-window finite lambda proxy", title="Length-convention sensitivity")
    axes[2].legend(fontsize=7)
    figure.savefig(destinations["png"], dpi=180)
    figure.savefig(destinations["pdf"])
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/phase3a/crossing_analysis")
    parser.add_argument("--label", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--expected-fields", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202611113940)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
