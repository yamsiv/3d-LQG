"""Small checkpointed controls for the explicit provisional Task3B sampler.

Preserves completed checkpoints; --resume reuses their exact saved records.
No existing Phase2C/3A files are modified. Full paths saved for first two
realizations per group; every graph's minima/root-distance matrix is saved.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import shortest_path
from src.covariance import from_correlations
from src.provisional_excursions_3b import make_provisional_model, sample_provisional_excursion
from src.brownian_excursions import sample_independent_excursion, ExcursionSample
from src.weyl_excursions import sample_weyl_excursion
from src.weyl_a3 import sample_a3_excursion
from src.mated_crt import cell_minima, graph_from_minima, validate_graph
from src.provenance import environment_manifest
from experiments.phase2c_path_validation import feature_record, path_probes
from src.corner_ritz import spherical_quadrature


CASES = {"i3": (0., 0., 0.), "a2": (-.5, 0., 0.), "a3": (-.5, 0., -.5),
         "unequal": (-.2, .1, .35), "equi02": (.2, .2, .2)}


def _json(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def metrics(sample, n, seed, out, key):
    stats = feature_record(sample, n, seed)
    minima = np.asarray(stats["minima"])
    graph = graph_from_minima(minima)
    points = np.random.default_rng(seed+389).integers(n, size=16)
    distances = shortest_path(graph.to_sparse(), directed=False, unweighted=True,
                              indices=points)
    matrix = distances[:, points]
    result = {"features": stats["features"], "degree_histogram": stats["degree_histogram"],
              "root_histograms": stats["root_histograms"],
              "distance_histogram": stats["distance_histogram"],
              "matrix": matrix.tolist(), "graph_checks": validate_graph(graph)}
    np.savez_compressed(out/(key+f"_n{n}.npz"), minima=minima, edges=graph.edges,
                        points=points, distances=distances, matrix=matrix)
    return result


def _mean_se(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "se": float(values.std(ddof=1)/np.sqrt(len(values)))
            if len(values) > 1 else None, "n": len(values)}


def summarize(out):
    rows = [json.loads(p.read_text()) for p in sorted((out/"records").glob("*.json"))]
    groups = []
    successes = [r for r in rows if r["status"] == "success"]
    for case, setting in sorted({(r["case"], r["setting"]) for r in successes}):
        group = [r for r in successes if r["case"] == case and r["setting"] == setting]
        entry = {"case": case, "setting": setting, "independent_paths": len(group),
                 "seconds": sum(r["seconds"] for r in group), "sizes": {}}
        for n in group[0]["results"]:
            features = group[0]["results"][n]["features"]
            entry["sizes"][n] = {k: _mean_se([r["results"][n]["features"][k] for r in group])
                                   for k in features}
        if setting != "reference":
            legs = [l for r in group for l in r["sampler_metadata"]["leg_diagnostics"]]
            entry["mean_value_diagnostics"] = {"maximum_sum_abs_log_defect": max(l["sum_abs_log_mean_value_ratio"] for l in legs),
                "maximum_abs_signed_sum_log_defect": max(abs(l["sum_log_mean_value_ratio"]) for l in legs),
                "maximum_quadrature_log_difference": max(l["maximum_audited_quadrature_log_difference"] for l in legs)}
        groups.append(entry)
    comparisons = []
    for group in groups:
        if group["setting"] == "reference":
            continue
        refs = next((r for r in groups if r["case"] == group["case"] and r["setting"] == "reference"), None)
        if refs is None:
            continue
        for n, features in group["sizes"].items():
            ref = refs["sizes"][n]
            diff = features["mean_distance"]["mean"]-ref["mean_distance"]["mean"]
            se = np.sqrt((features["mean_distance"]["se"] or 0)**2+(ref["mean_distance"]["se"] or 0)**2)
            comparisons.append({"case": group["case"], "setting": group["setting"], "n": int(n),
                "relative_mean_distance_difference": diff/ref["mean_distance"]["mean"],
                "relative_mc95": ((np.array([-1.96, 1.96])*se+diff)/ref["mean_distance"]["mean"]).tolist(),
                "max_joint_cdf_difference": max(abs(features[k]["mean"]-ref[k]["mean"]) for k in features if k.startswith("joint_cdf")),
                "degree_variance_difference": features["degree_variance"]["mean"]-ref["degree_variance"]["mean"]})
    value = {"groups": groups, "comparisons": comparisons,
             "failures": [r for r in rows if r["status"] != "success"],
             "notes": ["Independent-realization standard errors, no independent-pair fiction",
                       "Two resolutions are independent paths, not coupled Brownian refinements",
                       "Exact A2/A3 grid references have approximate polygonal joint minima"]}
    _json(out/"summary.json", value)
    return value


def angular_diagnostics(out, *, samples=1024, seed=7171384):
    """Isolate finite-SIR angular error cheaply; independent draws and SEs.

    Each realization here is only a direction, not another graph or path.
    D6 spherical quadrature provides a numerical m_tilde² target reference,
    distinct from accuracy of m_tilde as a true cone eigenfunction.
    """
    path = out/"angular_diagnostics.json"
    if path.exists():
        return json.loads(path.read_text())
    start = time.perf_counter()
    result = []
    for case in ["unequal", "equi02"]:
        C = from_correlations(*CASES[case])
        z, area_weights = spherical_quadrature(C, 64)
        model = make_provisional_model(C, degree=6)
        importance = model.angular_mode(z)**2
        probability = area_weights*importance
        probability /= probability.sum()
        # Features use correlated direction components and all their products.
        x = z@model.R.T
        features = np.column_stack([x, x*x, x[:,0]*x[:,1], x[:,0]*x[:,2], x[:,1]*x[:,2]])
        target = probability@features
        nu = area_weights/area_weights.sum()
        mu = nu@importance
        leading_numerator = target*(nu@(importance*importance))-(nu*(importance*importance))@features
        for K in [512, 2048]:
            model.initial_candidates = K
            rng = np.random.default_rng(np.random.SeedSequence([seed, list(CASES).index(case), K]))
            directions = np.array([model.sample_direction(rng) for _ in range(samples)])
            d = directions@model.R.T
            f = np.column_stack([d, d*d, d[:,0]*d[:,1], d[:,0]*d[:,2], d[:,1]*d[:,2]])
            result.append({"case": case, "initial_candidates": K, "independent_directions": samples,
                "ritz_degree": 6, "sphere_target_quadrature": 64, "seed_basis": seed,
                "feature_names": ["x0", "x1", "x2", "x0sq", "x1sq", "x2sq", "x0x1", "x0x2", "x1x2"],
                "numerical_target": target.tolist(), "empirical_mean": f.mean(axis=0).tolist(),
                "mc_se": (f.std(axis=0, ddof=1)/np.sqrt(samples)).tolist(),
                "leading_1_over_K_bias_prediction": (leading_numerator/(K*mu*mu)).tolist(),
                "warning": "Moment MC and asymptotic 1/K diagnostic do not bound total initial angular-law error"})
    value = {"groups": result, "total_seconds": time.perf_counter()-start}
    _json(path, value)
    return value


def distribution_diagnostics(out):
    """Full retained observables with realization-level uncertainty."""
    from src.metric_measure_3a import matrix_features
    rows = [json.loads(p.read_text()) for p in sorted((out/"records").glob("*.json"))]
    rows = [r for r in rows if r["status"] == "success"]
    result = []
    for case in CASES:
        ref = [r for r in rows if r["case"] == case and r["setting"] == "reference"]
        coarse = [r for r in rows if r["case"] == case and r["setting"] == "coarse"]
        fine = [r for r in rows if r["case"] == case and r["setting"] == "fine"]
        contrasts = [("coarse-reference", coarse, ref), ("fine-reference", fine, ref),
                     ("fine-coarse", fine, coarse)]
        for label, a, b in contrasts:
            if not a or not b:
                continue
            for n in [128, 256]:
                key = str(n)
                feature_names = list(a[0]["results"][key]["features"])
                stats = {}
                for feature in feature_names:
                    x = np.array([r["results"][key]["features"][feature] for r in a])
                    y = np.array([r["results"][key]["features"][feature] for r in b])
                    se = np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
                    stats[feature] = {"difference": float(x.mean()-y.mean()), "mc_se": float(se)}
                joint = [k for k in feature_names if k.startswith("joint_cdf")]
                largest = max(joint, key=lambda k: abs(stats[k]["difference"]))
                matrices = {}
                for name in ["small_over_large", "middle_over_large"]:
                    x = np.array([matrix_features(r["results"][key]["matrix"])[name].mean() for r in a])
                    y = np.array([matrix_features(r["results"][key]["matrix"])[name].mean() for r in b])
                    matrices[name] = {"difference": float(x.mean()-y.mean()),
                        "mc_se": float(np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y)))}
                result.append({"case": case, "contrast": label, "n": n,
                    "independent_paths": [len(a), len(b)], "features": stats,
                    "largest_joint_cdf_difference": {"feature": largest, **stats[largest]},
                    "matrix_triangle_features": matrices})
    path_stats = []
    for case, setting in sorted({(r["case"], r["setting"]) for r in rows}):
        group = [r for r in rows if r["case"] == case and r["setting"] == setting]
        names = group[0]["path"]["features"]
        path_stats.append({"case": case, "setting": setting,
            "path_features": {k: _mean_se([r["path"]["features"][k] for r in group]) for k in names},
            "hitting_times": _mean_se([np.mean(r["sampler_metadata"]["hitting_times"]) for r in group]),
            "coupled_polygonal_q8_edge_jaccard": _mean_se([r["coupled_polygonal_thinning_edge_jaccard_q8"] for r in group])
                 if setting != "reference" else None})
    value = {"comparisons": result, "path_groups": path_stats,
        "warning": "Independent realization uncertainty; multiple correlated joint-CDF thresholds, no multiplicity-free p-values"}
    _json(out/"distribution_diagnostics.json", value)
    return value


def audit_saved_controls(out):
    """Reuse old controls; separate confirmatory A3 seeds from the first pilot."""
    import hashlib
    from scipy.stats import invgamma, kstest
    rows = [json.loads(p.read_text()) for p in (out/"records").glob("*.json")]
    rows = [r for r in rows if r["status"] == "success"]
    confirmation = out/"a3_fine_confirmation"/"records"
    if confirmation.exists():
        extra = [json.loads(p.read_text()) for p in confirmation.glob("*.json")]
        rows += [dict(r, setting="fine_confirmation") for r in extra if r["status"] == "success"]
    previous = Path("results/phase2c/paths/nested_controls/records.jsonl")
    old = [json.loads(line) for line in previous.open()]
    comparisons = []
    for case in ["i3", "a2", "a3"]:
        oldref = [r for r in old if r["case"] == case]
        newref = [r for r in rows if r["case"] == case and r["setting"] == "reference"]
        reference = [r["results"]["256"]["exact" if case == "i3" else "1024"]["features"] for r in oldref]
        reference += [r["results"]["256"]["features"] for r in newref]
        for setting in ["coarse", "fine", "fine_confirmation"]:
            group = [r for r in rows if r["case"] == case and r["setting"] == setting]
            if not group:
                continue
            stats = {}
            for feature in reference[0]:
                a = np.array([r["results"]["256"]["features"][feature] for r in group])
                b = np.array([r[feature] for r in reference])
                stats[feature] = {"difference": float(a.mean()-b.mean()),
                    "mc_se": float(np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)))}
            joint = max((k for k in stats if k.startswith("joint_cdf")), key=lambda k: abs(stats[k]["difference"]))
            baseline = np.mean([r["mean_distance"] for r in reference])
            d = stats["mean_distance"]
            comparisons.append({"case": case, "setting": setting, "n": 256,
                "independent_paths": len(group), "reference_independent_paths": len(reference),
                "relative_mean_distance_difference": d["difference"]/baseline,
                "relative_mean_distance_mc95": ((d["difference"]+np.array([-1.96, 1.96])*d["mc_se"])/baseline).tolist(),
                "largest_joint_cdf_difference": {"feature": joint, **stats[joint]}, "features": stats})
    clocks = []
    for case, setting in sorted({(r["case"], r["setting"]) for r in rows}):
        group = [r for r in rows if r["case"] == case and r["setting"] == setting]
        alpha = group[0]["sampler_metadata"]["alpha1"]
        t = np.array([r["sampler_metadata"]["hitting_times"] for r in group])
        ks = kstest(t.ravel(), invgamma(alpha, scale=.5).cdf)
        residuals = []
        for row in group:
            m = row["sampler_metadata"]
            total = sum(m["hitting_times"])
            u = m["junction_time"]
            dim = 2*alpha+2
            residuals.append([row["path"]["features"][f"radius_square_t{s}"]
                -((s/u)**2/total+dim*s*(u-s)/u if s <= u else
                  ((1-s)/(1-u))**2/total+dim*(s-u)*(1-s)/(1-u))
                for s in [.25, .5, .75]])
        residuals = np.asarray(residuals)
        clocks.append({"case": case, "setting": setting, "independent_paths": len(group),
            "mean_leg_lifetime": float(t.mean()), "target_mean": float(.5/(alpha-1)),
            "empirical_cluster_mean_se": float(t.mean(axis=1).std(ddof=1)/np.sqrt(len(group))),
            "target_mean_se_if_exact_independent_legs": float(np.sqrt(.25/((alpha-1)**2*(alpha-2))/t.size)) if alpha > 2 else None,
            "inverse_lifetime_zmean": float(((.5/t-alpha)/np.sqrt(alpha)).mean()),
            "inverse_lifetime_zmean_target_se": float(1/np.sqrt(t.size)),
            "nominal_marginal_KS_statistic": float(ks.statistic), "nominal_KS_pvalue": float(ks.pvalue),
            "conditional_radial_square_mean_residual": residuals.mean(axis=0).tolist(),
            "conditional_radial_square_cluster_se": (residuals.std(axis=0, ddof=1)/np.sqrt(len(group))).tolist()})
    value = {"pooled_reference": {"previous_dataset": str(previous),
        "sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "old_paths_per_case": 64, "new_paths_per_case": 24,
        "note": "Old controls reused, never recomputed or counted as new draws; A2/A3 q1024 remain polygonal references"},
        "comparisons": comparisons, "clock_and_conditional_radial_diagnostics": clocks,
        "warning": "Pointwise diagnostics over many correlated features, not simultaneous inference; generic clock target only intended, not established"}
    _json(out/"augmented_control_audit.json", value)
    return value


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((out/"summary.json").read_text())
    diagnostics = json.loads((out/"distribution_diagnostics.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for j, case in enumerate(["i3", "a2", "a3"]):
        for i, setting in enumerate(["coarse", "fine"]):
            row = next(r for r in summary["comparisons"] if r["case"] == case and r["setting"] == setting and r["n"] == 256)
            lo, hi = np.array(row["relative_mc95"])*100
            y = 100*row["relative_mean_distance_difference"]
            axes[0].errorbar(j+(i-.5)*.15, y, yerr=[[y-lo], [hi-y]], fmt="o", color=f"C{i}",
                             label=setting if j == 0 else None)
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set_xticks(range(3), ["I3", "A2", "A3"])
    axes[0].set_ylabel("Mean-distance difference (%)")
    axes[0].set_title("n256 controls: independent MC95")
    axes[0].legend()
    for case in ["unequal", "equi02"]:
        selected = [next(g for g in summary["groups"] if g["case"] == case and g["setting"] == s) for s in ["coarse", "fine"]]
        means = [g["sizes"]["256"]["mean_distance"]["mean"] for g in selected]
        se = [1.96*g["sizes"]["256"]["mean_distance"]["se"] for g in selected]
        axes[1].errorbar([0, 1], means, yerr=se, fmt="o-", label=case)
        defects = [g["mean_value_diagnostics"]["maximum_sum_abs_log_defect"] for g in selected]
        axes[2].plot([0, 1], defects, "o-", label=case)
    for axis in axes[1:]:
        axis.set_xticks([0, 1], ["D4 / coarse", "D6 / finer"])
        axis.legend()
    axes[1].set_ylabel("Mean graph distance")
    axes[1].set_title("Generic n256:12 paths/setting")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Maximum leg sum |log(1+R)|")
    axes[2].set_title("Observed harmonic defects; no bound")
    fig.tight_layout()
    fig.savefig(out/"pilot_validation.pdf")
    fig.savefig(out/"pilot_validation.png", dpi=180)
    plt.close(fig)


def conditional_reference(case, metadata, rng, *, oversample=128):
    """Exact-grid bridge given the same S,T, not an independent full ensemble.

    For a true harmonic inward process conditioned on its lifetime, reversing
    the leg gives the killed-cone bridge0->S of that duration. Its special-cone
    matrix/Bessel representation remains available for fixed junction data.
    Therefore this tests the conditional path approximation but cannot test
    the numerical S law or lifetime distribution.
    """
    from src.brownian_excursions import _bessel_bridge, positive_bridge_minima
    from src.weyl_excursions import _eigenvalue_bridge
    from src.weyl_a3 import _matrix_bridge_eigenvalues
    C = np.array(metadata["covariance"])
    total = sum(metadata["hitting_times"])
    u = metadata["junction_time"]
    height = np.linalg.cholesky(C)@metadata["junction_direction"]/np.sqrt(total)
    times = np.unique(np.r_[np.linspace(0, 1, 256*oversample+1), u])
    middle = np.searchsorted(times, u)
    if case == "i3":
        values = np.empty((len(times), 3))
        for k in range(3):
            values[:middle+1, k] = _bessel_bridge(times[:middle+1], 0., height[k], rng)
            values[middle:, k] = _bessel_bridge(times[middle:], height[k], 0., rng)
        minima = positive_bridge_minima(values[:-1], values[1:], np.diff(times)[:, None], rng)
    else:
        count = 3 if case == "a2" else 4
        particles = np.r_[0., np.cumsum(np.sqrt(2)*height[:count-1])]
        particles -= particles.mean()
        function = _eigenvalue_bridge if case == "a2" else _matrix_bridge_eigenvalues
        eigenvalues = np.empty((len(times), count))
        eigenvalues[:middle+1] = function(times[:middle+1], np.zeros(count), particles, rng)
        eigenvalues[middle:] = function(times[middle:], particles, np.zeros(count), rng)
        values = np.diff(eigenvalues, axis=1)/np.sqrt(2)
        if case == "a2":
            extra = np.empty(len(times))
            extra[:middle+1] = _bessel_bridge(times[:middle+1], 0., height[2], rng)
            extra[middle:] = _bessel_bridge(times[middle:], height[2], 0., rng)
            values = np.column_stack([values, extra])
        minima = None
    values[[0, -1]] = 0.
    return ExcursionSample(times, values, minima, {**metadata,
        "sampler": "special_cone_conditional_bridge_same_provisional_junction",
        "law": "conditional_bridge_reference", "grid_values_exact": True,
        "cell_minima_exact": case == "i3", "oversample": oversample,
        "warning": "Conditions on the original provisional S,T; cannot validate their marginal laws"})


def conditional_bridge_audit(out):
    directory = out/"conditional_bridge_audit"
    for name in [directory, directory/"records", directory/"arrays"]:
        name.mkdir(exist_ok=True)
    source_paths = list((out/"records").glob("*.json"))
    confirmation = out/"a3_fine_confirmation"/"records"
    if confirmation.exists():
        source_paths += list(confirmation.glob("*.json"))
    pairs = []
    for source in sorted(source_paths):
        original = json.loads(source.read_text())
        if original["case"] not in {"i3", "a2", "a3"} or original["setting"] == "reference" or original["status"] != "success":
            continue
        setting = original["setting"]+("_confirmation" if "a3_fine_confirmation" in str(source) else "")
        key = f"{original['case']}_{setting}_{original['realization']:03d}"
        dest = directory/"records"/(key+".json")
        if dest.exists():
            paired = json.loads(dest.read_text())
        else:
            start = time.perf_counter()
            seed = int(np.random.SeedSequence([original["seed"], 737101]).generate_state(1, dtype=np.uint64)[0])
            reference = conditional_reference(original["case"], original["sampler_metadata"], np.random.default_rng(seed))
            paired = {"case": original["case"], "setting": setting,
                "source": str(source), "conditional_seed": seed,
                "realization": original["realization"], "reference_oversample": 128,
                "results": {str(n): metrics(reference, n, (original["seed"]+n) % (2**64), directory/"arrays", key) for n in [128, 256]},
                "path": path_probes(reference), "seconds": time.perf_counter()-start}
            _json(dest, paired)
        pairs.append((original, paired))
    groups = []
    for case, setting in sorted({(r[1]["case"], r[1]["setting"]) for r in pairs}):
        group = [(a, b) for a, b in pairs if b["case"] == case and b["setting"] == setting]
        for n in [128, 256]:
            key = str(n)
            feature_names = group[0][0]["results"][key]["features"]
            differences = {feature: _mean_se([a["results"][key]["features"][feature]-b["results"][key]["features"][feature] for a, b in group]) for feature in feature_names}
            largest = max((k for k in feature_names if k.startswith("joint_cdf")), key=lambda k: abs(differences[k]["mean"]))
            reference_mean = np.mean([b["results"][key]["features"]["mean_distance"] for a, b in group])
            d = differences["mean_distance"]
            groups.append({"case": case, "setting": setting, "n": n,
                "independent_junction_pairs": len(group), "paired_feature_differences": differences,
                "mean_distance_relative_difference": d["mean"]/reference_mean,
                "mean_distance_relative_mc95": ((d["mean"]+np.array([-1.96, 1.96])*d["se"])/reference_mean).tolist(),
                "largest_joint_cdf_paired_difference": {"feature": largest, **differences[largest]}})
    value = {"groups": groups, "total_reference_seconds": sum(b["seconds"] for _, b in pairs),
        "notes": ["New reference bridges share the provisional S,T, not their internal Brownian paths",
            "Variance estimated across physical junction/path pairs; this is not a nested Brownian refinement",
            "Cannot validate initial angular or lifetime laws; A2/A3 residual q128 minima error remains"]}
    _json(directory/"summary.json", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/phase3b/sampler_pilot"))
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--settings", nargs="+", default=["reference", "coarse", "fine"])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--generic-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=38371001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--angular-only", action="store_true")
    parser.add_argument("--angular-samples", type=int, default=1024)
    args = parser.parse_args()
    out = args.output
    if out.exists() and not args.resume:
        raise FileExistsError("Existing output: explicitly --resume or use a new output")
    for folder in [out, out/"records", out/"arrays", out/"paths"]:
        folder.mkdir(parents=True, exist_ok=True)
    if not (out/"manifest.json").exists():
        _json(out/"manifest.json", {"configuration": vars(args) | {"output": str(out)},
            "cost": "First generic path .72s; planned ~200 small paths, 3-8 minutes, memory <200MB per path.",
            "provenance": environment_manifest()})
    if args.angular_only:
        angular_diagnostics(out, samples=args.angular_samples)
        distribution_diagnostics(out)
        audit_saved_controls(out)
        conditional_bridge_audit(out)
        make_figures(out)
        return
    for setting in args.settings:
        for case in args.cases:
            generic = case not in {"i3", "a2", "a3"}
            if generic and setting == "reference":
                continue
            count = args.generic_samples if generic else args.samples
            C = from_correlations(*CASES[case])
            degree = (4 if setting == "coarse" else 6) if generic else {"i3": 0, "a2": 0, "a3": 1}[case]
            model = None if setting == "reference" else make_provisional_model(C, degree=degree,
                initial_candidates=512 if setting == "coarse" else 2048)
            for realization in range(count):
                key = f"{case}_{setting}_{realization:03d}"
                dest = out/"records"/(key+".json")
                if dest.exists():
                    continue
                seed = int(np.random.SeedSequence([args.seed, list(CASES).index(case),
                    ["reference", "coarse", "fine"].index(setting), realization]).generate_state(1, dtype=np.uint64)[0])
                start = time.perf_counter()
                row = {"case": case, "setting": setting, "realization": realization, "seed": seed,
                       "covariance": C.tolist(), "sizes": [128, 256]}
                try:
                    rng = np.random.default_rng(seed)
                    if setting == "reference":
                        if case == "i3":
                            sample = sample_independent_excursion(256, 3, rng, law="biased", oversample=128)
                        elif case == "a2":
                            sample = sample_weyl_excursion(256, 3, rng, law="biased", oversample=128)
                        else:
                            sample = sample_a3_excursion(256, rng, law="biased", oversample=128)
                    else:
                        sample = sample_provisional_excursion(256, rng=rng, model=model,
                            step_size=.01 if setting == "coarse" else .005,
                            relative_step=.2 if setting == "coarse" else .1,
                            exit_order=8 if setting == "coarse" else 12,
                            trace_balls=realization < 2)
                    row.update(status="success", sampler_metadata=sample.metadata,
                               path=path_probes(sample), results={})
                    for n in row["sizes"]:
                        row["results"][str(n)] = metrics(sample, n, (seed+n) % (2**64), out/"arrays", key)
                    if realization < 2:
                        np.savez_compressed(out/"paths"/(key+".npz"), times=sample.times,
                            values=sample.values, segment_minima=sample.segment_minima
                            if sample.segment_minima is not None else np.empty((0, 3)))
                    if setting != "reference":
                        # Coupled *polygonal* thinning isolates grid-minimum
                        # loss on this same retained path, not unknown bridges.
                        coarse_times = np.unique(np.r_[np.linspace(0, 1, 256*8+1), sample.metadata["junction_time"]])
                        coarse_values = np.column_stack([np.interp(coarse_times, sample.times, sample.values[:, j]) for j in range(3)])
                        thin = ExcursionSample(coarse_times, coarse_values, None, sample.metadata)
                        e_full = {tuple(e) for e in graph_from_minima(cell_minima(sample, 256)).edges}
                        e_thin = {tuple(e) for e in graph_from_minima(cell_minima(thin, 256)).edges}
                        row["coupled_polygonal_thinning_edge_jaccard_q8"] = len(e_full & e_thin)/len(e_full | e_thin)
                except Exception as error:
                    row.update(status="failure", error=repr(error))
                row["seconds"] = time.perf_counter()-start
                _json(dest, row)
                print(json.dumps({k: row[k] for k in ["case", "setting", "realization", "status", "seconds"]}), flush=True)
            summarize(out)
    summarize(out)


if __name__ == "__main__":
    main()
