"""Checkpointed LFPP stencil, quadrature, and cost validation for Task 3A.

Use a new --output directory; existing records are never replaced. This is a
small validation experiment, separate from the main cross-model ensembles.
"""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.lfpp_3d import build_lfpp_graph, root_distances, cube_crossing, exact_flat_distances
from src.provenance import environment_manifest


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row)+"\n")


def benchmark(output, seed):
    rng = np.random.default_rng(seed)
    path = output / "benchmark.jsonl"
    if path.exists():
        raise FileExistsError(path)
    pilot_cost = None
    rows = []
    for n in (16, 24, 32, 48):
        coordinate = np.arange(n)/n
        x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
        # A deterministic smooth field benchmarks nonconstant edge weights.
        # It is not used as a log-field scientific sample.
        h = np.sin(2*np.pi*x)+.7*np.cos(2*np.pi*y)+.4*np.sin(4*np.pi*z)
        roots = rng.integers(n**3, size=40)
        for stencil in ("axis", "26", "primitive2"):
            if stencil == "primitive2" and n == 48:
                # Richest 48^3 construction is not required by main pipeline.
                continue
            start = time.perf_counter()
            graph, metadata = build_lfpp_graph(h, .4, stencil=stencil)
            built = time.perf_counter()
            distances = root_distances(graph, roots)
            end = time.perf_counter()
            row = {"n": n, "stencil": stencil, "roots": len(roots),
                   "field": "deterministic smooth trigonometric cost control, not log field",
                   "build_seconds": built-start, "search_seconds": end-built,
                   "seconds": end-start, "csr_bytes": metadata["csr_bytes"],
                   "distance_array_bytes": distances.nbytes,
                   "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "finite": bool(np.isfinite(distances).all())}
            rows.append(row); append(path, row)
            if n == 16 and stencil == "26":
                pilot_cost = row["seconds"]
                (output/"cost_estimate.json").write_text(json.dumps({
                    "pilot_n16_26_seconds": pilot_cost,
                    "n32_26_estimated_seconds": pilot_cost*8*np.log(32**3)/np.log(16**3),
                    "n48_26_estimated_seconds": pilot_cost*27*np.log(48**3)/np.log(16**3),
                    "design": "40 roots; 11 individual graph builds; expected each below one minute",
                    "memory_estimate_n48_26": "about 35MB CSR +35MB distances, construction arrays additional",
                    "reason": "select practical ensemble sizes before main runs"}, indent=2)+"\n")
            print(json.dumps(row), flush=True)
            del graph, distances
    return rows


def flat_validation(output, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (16, 24, 32):
        roots = np.array([0])
        euclidean = exact_flat_distances(n, roots)[0]
        valid = euclidean > 0
        for stencil in ("axis", "26", "primitive2"):
            graph, meta = build_lfpp_graph(np.zeros((n,)*3), 0, stencil=stencil)
            dist = root_distances(graph, roots)[0]
            error = dist[valid]/euclidean[valid]-1
            rows.append({"n": n, "stencil": stencil, "mean_relative_error": float(error.mean()),
                         "max_relative_error": float(error.max()),
                         "quantile_relative_error": np.quantile(error, [.5, .9, .95, .99]).tolist(),
                         "negative_errors_below_tolerance": int(np.sum(error < -1e-12)),
                         "distances_exact_to_stencil": True,
                         "interpretation": "fixed stencil anisotropy is not removed by increasing N"})
    (output/"flat_validation.json").write_text(json.dumps(rows, indent=2)+"\n")
    return rows


def nonzero_validation(output, seed):
    """Independent small log-field controls when field module is available."""
    from src.log_field_3d import sample_coupled_fields
    rows = []
    path = output/"nonzero.jsonl"
    if path.exists():
        raise FileExistsError(path)
    for realization in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([seed, realization]))
        fields = sample_coupled_fields([16, 24], rng)
        for n in (16, 24):
            field = fields[n]
            h = field.h
            roots = rng.integers(n**3, size=8)
            for xi in (.2, .4, .6):
                saved = {}
                for stencil, quadrature in (("axis", 2), ("26", 2), ("primitive2", 2), ("primitive2", 4)):
                    graph, meta = build_lfpp_graph(h, xi, stencil=stencil,
                                                   quadrature_per_grid=quadrature)
                    distance = root_distances(graph, roots)
                    label = f"{stencil}_q{quadrature}"
                    saved[label] = distance
                    if stencil == "axis":
                        cube, _ = build_lfpp_graph(h, xi, stencil="axis", boundary="cube")
                        edge_cross = cube_crossing(cube, n)
                        paper_cross = cube_crossing(cube, n, site_weights=np.exp(xi*h))
                    if label != "axis_q2":
                        reference = saved["axis_q2"] if stencil == "26" else saved["26_q2"]
                        nonzero = reference > 0
                        diff = distance[nonzero]/reference[nonzero]-1
                        reference_label = "axis_q2" if stencil == "26" else "26_q2"
                    else:
                        diff = np.array([0.]); reference_label = "self"
                    row = {"seed": seed, "realization": realization, "n": n, "xi": xi,
                           "label": label, "reference": reference_label,
                           "distance_mean": float(distance.mean()),
                           "mean_pointwise_relative_difference": float(diff.mean()),
                           "p95_abs_pointwise_relative_difference": float(np.quantile(abs(diff), .95)),
                           "edge_cube_crossing": edge_cross, "paper_cube_crossing": paper_cross,
                           "field_sha256": sha256(h.tobytes()).hexdigest(),
                           "build_seconds": meta["construction_seconds"]}
                    if label == "primitive2_q4":
                        ref = saved["primitive2_q2"]
                        mask = ref > 0
                        row["q4_vs_q2_mean_relative"] = float(np.mean(distance[mask]/ref[mask]-1))
                        row["q4_vs_q2_p95_abs_relative"] = float(np.quantile(abs(distance[mask]/ref[mask]-1), .95))
                    rows.append(row); append(path, row)
                    del graph, distance
            np.savez_compressed(output/f"field_{realization}_n{n}.npz", h=h, roots=roots)
    return rows


def interpolation_validation(output):
    """Same saved fields/roots; distinguish two finite interpolation targets."""
    path = output/"interpolation.jsonl"
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for realization in range(4):
        for n in (16, 24):
            raw = np.load(output/f"field_{realization}_n{n}.npz")
            h, roots = raw["h"], raw["roots"]
            for xi in (.2, .4, .6):
                distances = {}
                for rule in ("weight", "field"):
                    graph, _ = build_lfpp_graph(h, xi, stencil="26", interpolation=rule,
                                               quadrature_per_grid=4)
                    distances[rule] = root_distances(graph, roots)
                    del graph
                mask = distances["weight"] > 0
                difference = distances["field"][mask]/distances["weight"][mask]-1
                # Rowwise sorted distributions also detect shape after mean scaling.
                a = np.sort(distances["weight"].ravel())/distances["weight"].mean()
                b = np.sort(distances["field"].ravel())/distances["field"].mean()
                row = {"realization": realization, "n": n, "xi": xi,
                       "comparison": "field vs weight trilinear interpolation, both quadrature4",
                       "mean_pointwise_relative_difference": float(difference.mean()),
                       "p95_abs_pointwise_relative_difference": float(np.quantile(abs(difference), .95)),
                       "distance_mean_ratio_minus_one": float(distances["field"].mean()/distances["weight"].mean()-1),
                       "realization_mean_normalized_pair_W1": float(np.mean(abs(a-b))),
                       "field_sha256": sha256(h.tobytes()).hexdigest()}
                rows.append(row); append(path, row)
    return rows


def summarize(output):
    """Aggregate saved paired controls across independent fields, not roots."""
    target = output/"paired_control_summary.json"
    if target.exists():
        raise FileExistsError(target)
    nonzero = [json.loads(line) for line in (output/"nonzero.jsonl").read_text().splitlines()]
    interpolation = [json.loads(line) for line in (output/"interpolation.jsonl").read_text().splitlines()]
    rows = []
    for n in (16, 24):
        for xi in (.2, .4, .6):
            select = lambda data: [r for r in data if r["n"] == n and r["xi"] == xi]
            controls = {
                "26_vs_axis": [r["mean_pointwise_relative_difference"] for r in select(nonzero) if r["label"] == "26_q2"],
                "primitive2_vs_26": [r["mean_pointwise_relative_difference"] for r in select(nonzero) if r["label"] == "primitive2_q2"],
                "quadrature4_vs_2": [r["q4_vs_q2_mean_relative"] for r in select(nonzero) if r["label"] == "primitive2_q4"],
                "field_vs_weight_interpolation": [r["distance_mean_ratio_minus_one"] for r in select(interpolation)],
                "interpolation_shape_W1": [r["realization_mean_normalized_pair_W1"] for r in select(interpolation)],
            }
            for label, values in controls.items():
                values = np.asarray(values)
                rows.append({"n": n, "xi": xi, "contrast": label, "independent_fields": len(values),
                             "mean": float(values.mean()), "SE_between_fields": float(values.std(ddof=1)/np.sqrt(len(values))),
                             "minimum": float(values.min()), "maximum": float(values.max())})
    target.write_text(json.dumps(rows, indent=2)+"\n")
    with (output/"paired_control_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for n, style in ((16, "--"), (24, "-")):
        for contrast, color, label in (("26_vs_axis", "C0", "26 / axis"),
                                       ("primitive2_vs_26", "C1", "98 / 26"),
                                       ("field_vs_weight_interpolation", "C2", "field / weight interpolation")):
            sub = [r for r in rows if r["n"] == n and r["contrast"] == contrast]
            axes[0].errorbar([r["xi"] for r in sub], [100*r["mean"] for r in sub],
                             yerr=[100*r["SE_between_fields"] for r in sub], color=color,
                             linestyle=style, marker="o", label=f"{label}, N={n}")
        for ax, contrast, scale in ((axes[1], "quadrature4_vs_2", 100), (axes[2], "interpolation_shape_W1", 1)):
            sub = [r for r in rows if r["n"] == n and r["contrast"] == contrast]
            ax.errorbar([r["xi"] for r in sub], [scale*r["mean"] for r in sub],
                         yerr=[scale*r["SE_between_fields"] for r in sub], linestyle=style,
                         marker="o", label=f"N={n}")
    axes[0].set(ylabel="Paired distance change (%)", title="Stencil and interpolation sensitivity")
    axes[1].set(ylabel="Distance change (%)", title="Doubling long-edge quadrature")
    axes[2].set(ylabel="Within-realization mean-normalized W1", title="Interpolation shape sensitivity")
    for ax in axes:
        ax.set_xlabel("LFPP xi"); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("Four independent fields; error bars are between-field SE")
    fig.tight_layout()
    fig.savefig(output/"paired_discretization_controls.png", dpi=180)
    fig.savefig(output/"paired_discretization_controls.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/phase3a/lfpp_validation")
    parser.add_argument("--seed", type=int, default=20261111311)
    parser.add_argument("--mode", choices=["benchmark", "nonzero", "interpolation", "summarize"], default="benchmark")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output/f"metadata_{args.mode}.json"
    if metadata.exists():
        raise FileExistsError(metadata)
    metadata.write_text(json.dumps({"arguments": vars(args), "provenance": environment_manifest(),
                         "stopping_policy": "no individual experiment expected >15min; preserve every row"}, indent=2)+"\n")
    if args.mode == "benchmark":
        benchmark_rows = benchmark(output, args.seed)
        flat_rows = flat_validation(output, args.seed+1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for stencil in ("axis", "26", "primitive2"):
            rows = [r for r in benchmark_rows if r["stencil"] == stencil]
            axes[0].plot([r["n"] for r in rows], [r["seconds"] for r in rows], "o-", label=stencil)
            rows = [r for r in flat_rows if r["stencil"] == stencil]
            axes[1].plot([r["n"] for r in rows], [100*r["mean_relative_error"] for r in rows], "o-", label=stencil)
        axes[0].set(xlabel="Grid side N", ylabel="Build + 40-root search seconds", yscale="log")
        axes[1].set(xlabel="Grid side N", ylabel="Mean flat distance excess over Euclidean (%)")
        for ax in axes:
            ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output/"stencil_cost_and_anisotropy.png", dpi=180)
        plt.close(fig)
    elif args.mode == "nonzero":
        nonzero_validation(output, args.seed)
    elif args.mode == "interpolation":
        interpolation_validation(output)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
