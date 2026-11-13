"""Stage-I sampler reliability gate; old code/data are read-only inputs.

Run plan -> pilot -> (coordinator review) -> generate -> analyze. Every physical
path is checkpointed. Nested time-grid comparisons use the SAME retained
polygonal realization; independently seeded angular/path levels do not.
"""
from __future__ import annotations
import argparse
from itertools import permutations
import json
import hashlib
from pathlib import Path
import time
import numpy as np

from src.brownian_excursions import ExcursionSample
from src.provisional_excursions_3b import make_provisional_model, sample_provisional_excursion
from src.mated_crt import cell_minima, graph_from_minima, validate_graph
from src.graph_metrics import bfs_distances
from src.metric_measure_3a import weighted_root_cdfs, evaluate_cdf
from src.metric_features_3b import feature_vector
from src.provenance import environment_manifest


IDS = ["C002", "C007", "C012", "C006", "C010", "C000", "C004", "C016", "C019", "C028"]
MISSING = ["C002", "C007", "C012", "C004", "C010", "C016", "C019"]
BASE = 202611123001
SIZES = [512, 2048]
LEVELS = {1: dict(degree=6, base_step=.005, relative_step=.1, initial_candidates=1024, exit_order=12),
          2: dict(degree=8, base_step=.0025, relative_step=.05, initial_candidates=2048, exit_order=16)}
TARGETS = {"mean_relative": .02, "normalized_pair_W1": .02, "joint_minimum_CDF": .05,
           "rank_copula_CDF": .025,
           "ball_sd": .015, "triangle_mean": .01, "quartet_mean": .005,
           "matrix_correlation": .08}


def save_json(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def seed_for(identifier, level, realization, extra=0):
    return int(np.random.SeedSequence([BASE, int(identifier[1:]), level, realization, extra]).generate_state(1, dtype=np.uint64)[0])


def specs():
    return {s["id"]: s for s in json.loads(Path("results/phase3b/extended_design.json").read_text())["configurations"] if s["id"] in IDS}


def aligned_minima(minima, source_covariance, target_covariance):
    """Coordinate labels must align even though graph unions are invariant."""
    source = np.asarray(source_covariance)
    target = np.asarray(target_covariance)
    for perm in permutations(range(3)):
        if np.allclose(source[np.ix_(perm, perm)], target, atol=1e-11, rtol=0):
            return np.asarray(minima)[:, perm]
    raise ValueError("Source covariance is not a coordinate permutation of target")


def grid_thinning(sample, n=2048, oversample=8):
    times = np.unique(np.r_[np.linspace(0., 1., n*oversample+1), sample.metadata["junction_time"]])
    values = np.column_stack([np.interp(times, sample.times, sample.values[:, k]) for k in range(3)])
    return ExcursionSample(times, values, None, dict(sample.metadata,
        thinning_scope="same retained polygonal path, not a coarser WOS law", oversample=oversample))


def graph_record(minima, points):
    graph = graph_from_minima(minima)
    distances = np.array([bfs_distances(graph.to_sparse(), int(p)) for p in points])
    data = weighted_root_cdfs(distances, np.ones(len(minima))/len(minima), discrete=True)
    data.update(matrix=distances[:, points], points=points, minima=minima,
        edges=graph.edges, degree_histogram=np.bincount([len(a) for a in graph.adjacency]),
        graph_validation=validate_graph(graph), discrete=True)
    return data


def minima_features(minima):
    n = len(minima)
    t = (np.arange(n)+.5)/n
    x = minima[1:-1]/np.sqrt(t[1:-1]*(1-t[1:-1]))[:, None]
    result = {}
    for k in range(3):
        result[f"minimum_mean_{k}"] = float(x[:, k].mean())
        for threshold in [.5, 1., 1.5, 2.]:
            result[f"marginal_{k}_{threshold}"] = float(np.mean(x[:, k] <= threshold))
            for j in range(k):
                result[f"joint_{j}{k}_{threshold}"] = float(np.mean((x[:, k] <= threshold)&(x[:, j] <= threshold)))
    ranks = np.empty_like(x)
    for k in range(3):
        order = np.lexsort((np.arange(len(x)), x[:, k]))
        ranks[order, k] = (np.arange(len(x))+.5)/len(x)
        increments = np.diff(minima[1:-1, k])
        result[f"turning_fraction_{k}"] = float(np.mean(increments[:-1]*increments[1:] < 0))
        for j in range(k):
            for q in [.25, .5, .75]:
                result[f"copula_{j}{k}_{q}"] = float(np.mean((ranks[:, k] <= q)&(ranks[:, j] <= q)))
    return result


def metric_features(records, normalization="mean"):
    scale = float(np.mean([np.mean(r["root_means"]) for r in records])) if normalization == "mean" else None
    feature = feature_vector(records, scale=scale)
    names = [n.replace("over_median", "over_mean") if normalization == "mean" else n for n in feature["names"]]
    return {name: float(value) for name, value, valid in zip(names, feature["vector"], feature["valid"]) if valid}


def pair_discrepancy(a, b, normalization="mean"):
    from src.metric_measure_3a import ensemble_median
    norm = lambda rows: float(np.mean([np.mean(r["root_means"]) for r in rows])) if normalization == "mean" else ensemble_median(rows)
    sa, sb = norm(a), norm(b)
    grid = np.unique(np.r_[np.concatenate([r["grid"]/sa for r in a]), np.concatenate([r["grid"]/sb for r in b])])
    def on_atom_union(record, scale):
        values = grid*scale
        # Multiplication back to integer graph support must preserve its right
        # CDF value when normalization roundoff places an atom a few ulps low.
        nearest = np.rint(values)
        close = abs(values-nearest) <= 32*np.finfo(float).eps*np.maximum(1., abs(values))
        return evaluate_cdf(record, np.where(close, nearest, values)).mean(axis=0)
    ca = np.mean([on_atom_union(r, sa) for r in a], axis=0)
    cb = np.mean([on_atom_union(r, sb) for r in b], axis=0)
    # CDFs are right-continuous piecewise constant on this exact atom union.
    return {"W1": float(np.sum(np.abs(ca[:-1]-cb[:-1])*np.diff(grid))),
            "KS": float(np.max(np.abs(ca-cb))), "left_scale": sa, "right_scale": sb}


def save_arrays(path, data, header):
    payload = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    payload["header"] = json.dumps(header)
    np.savez_compressed(path, **payload)


def reuse_inventory(out):
    source_specs = specs()
    old = []
    hashes = {}
    for stage in ["screen_graphs", "anchor_controls", "intermediate_graphs"]:
        for identifier in IDS:
            for path in sorted((Path("results/phase3b")/stage/identifier).glob("graph_*_n2048.npz")):
                with np.load(path) as a:
                    h = json.loads(str(a["header"]))
                old.append({"source": str(path), "stage": stage, "id": identifier,
                    "seed": h["seed"], "sampler": h["sampler_metadata"]["sampler"],
                    "degree": h["sampler_metadata"].get("ritz_degree"),
                    "exact_grid": h["sampler_metadata"].get("grid_values_exact"),
                    "continuous_minima_exact": h["sampler_metadata"].get("cell_minima_exact")})
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    pilot = Path("results/phase3b/sampler_pilot")
    supplemental = []
    for name in ["summary.json", "distribution_diagnostics.json", "augmented_control_audit.json", "conditional_bridge_audit/summary.json"]:
        p = pilot/name
        supplemental.append(str(p))
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    plan = {"configurations": list(source_specs.values()), "reuse_records": old,
        "small_graph_supplemental_evidence": supplemental,
        "new_missing_fine_groups": MISSING, "new_paths_per_group": 8,
        "pilot": [{"id": c, "level": level, "realization": 0} for c, level in [("C019", 1), ("C016", 1), ("C002", 1), ("C019", 2)]],
        "pilot_wall_time_cap_seconds": 30, "planned_remaining_batch_seconds": [120, 240],
        "memory_budget_mb": 250, "output_budget_mb": 250,
        "primary_normalization": "ensemble mean pair distance", "sensitivity_normalization": "ensemble median",
        "targets_are_goals_not_claimed_bounds": TARGETS, "seed_base": BASE,
        "levels": LEVELS, "provenance": environment_manifest(), "input_sha256": hashes}
    path = out/"stage_plan.json"
    if not path.exists():
        save_json(path, plan)
    return plan


def simulate_one(out, identifier, level, realization, *, pilot=False):
    key = f"{identifier}_L{level}_{realization:03d}"
    destination = out/"records"/(key+".json")
    if destination.exists():
        return json.loads(destination.read_text())
    spec = specs()[identifier]
    settings = LEVELS[level]
    seed = seed_for(identifier, level, realization)
    start = time.perf_counter()
    row = {"id": identifier, "level": level, "realization": realization,
        "seed": seed, "spec": spec, "settings": settings, "pilot_checkpoint": pilot}
    try:
        model = make_provisional_model(spec["covariance"], degree=settings["degree"],
            initial_candidates=settings["initial_candidates"])
        step = settings["base_step"]*np.sqrt(2.5/(model.alpha1-1))
        sample = sample_provisional_excursion(max(SIZES), rng=np.random.default_rng(seed), model=model,
            step_size=step, relative_step=settings["relative_step"], exit_order=settings["exit_order"],
            audit_order=2*settings["exit_order"], max_steps=400_000 if pilot else 2_000_000)
        row.update(status="success", sampler_metadata=sample.metadata, sizes={}, coupled={})
        u = np.random.default_rng(seed_for(identifier, level, realization, 91)).random(16)
        full = {}
        for n in SIZES:
            points = np.floor(u*n).astype(int)
            minima = cell_minima(sample, n)
            data = graph_record(minima, points)
            header = {**row, "n": n, "physical_path_id": key, "source": str(destination)}
            save_arrays(out/"arrays"/(key+f"_n{n}.npz"), data, header)
            row["sizes"][str(n)] = {"minima_features": minima_features(minima),
                "mean_distance": float(data["root_means"].mean()), "graph_validation": data["graph_validation"]}
            full[n] = data
        # These checkpoints pair observation grids on ONE retained trajectory.
        for q in [8, 32, 128]:
            thinned = grid_thinning(sample, max(SIZES), q)
            row["coupled"][str(q)] = {}
            for n in SIZES:
                data = graph_record(cell_minima(thinned, n), full[n]["points"])
                if np.any(data["minima"] < full[n]["minima"]-1e-12):
                    raise AssertionError("Thinning cannot lower retained-path cell minima")
                a, b = set(map(tuple, data["edges"])), set(map(tuple, full[n]["edges"]))
                diff = data["minima"]-full[n]["minima"]
                row["coupled"][str(q)][str(n)] = {
                    "edge_jaccard": len(a&b)/len(a|b),
                    "minima_rms": float(np.sqrt(np.mean(diff*diff))),
                    "mean_distance_relative": float(data["root_means"].mean()/full[n]["root_means"].mean()-1),
                    "minima_feature_differences": {k: v-minima_features(full[n]["minima"])[k] for k, v in minima_features(data["minima"]).items()}}
                save_arrays(out/"arrays"/(key+f"_q{q}_n{n}.npz"), data,
                    {"physical_path_id": key, "n": n, "q": q, "coupling_scope": "same retained polygonal path"})
        if realization < 2:
            np.savez_compressed(out/"paths"/(key+".npz"), times=sample.times,
                values=sample.values, metadata=json.dumps(sample.metadata))
        row["retained_path_points"] = len(sample.times)
    except Exception as error:
        row.update(status="failure", error=repr(error))
    row["seconds"] = time.perf_counter()-start
    save_json(destination, row)
    print(json.dumps({k: row[k] for k in ["id", "level", "realization", "status", "seconds"]}), flush=True)
    return row


def load_graph_record(path, target, *, n=None):
    with np.load(path) as a:
        header = json.loads(str(a["header"]))
        result = {k: a[k].copy() for k in ["grid", "cdfs", "root_means", "matrix", "minima"]}
        points = a["points"].copy()
    covariance = header.get("sampler_metadata", {}).get("covariance", target)
    result["minima"] = aligned_minima(result["minima"], covariance, target)
    original_n = len(result["minima"])
    if n is not None and n != original_n:
        if original_n % n:
            raise ValueError("Graph cells must nest for minimum coarsening")
        factor = original_n//n
        minima = result["minima"].reshape(n, factor, 3).min(axis=1)
        result = graph_record(minima, points//factor)
    graph = graph_from_minima(result["minima"])
    degrees = np.array([len(v) for v in graph.adjacency])
    result.update(source=str(path), header=header, discrete=True,
        basic={**minima_features(result["minima"]), "degree_mean": float(degrees.mean()),
            "degree_variance": float(degrees.var()), "degree_q90": float(np.quantile(degrees, .9)),
            "ties_fraction": float(np.mean([len(result["minima"])-2-len(np.unique(result["minima"][1:-1, k])) for k in range(3)])/(len(result["minima"])-2))})
    return result


def get_groups(out, identifier, n):
    C = specs()[identifier]["covariance"]
    stage = "anchor_controls" if identifier in {"C002", "C007", "C012"} else "screen_graphs"
    coarse = [load_graph_record(p, C) for p in sorted((Path("results/phase3b")/stage/identifier).glob(f"graph_*_n{n}.npz"))]
    if identifier in MISSING:
        fine_paths = sorted((out/"arrays").glob(f"{identifier}_L1_[0-9][0-9][0-9]_n{n}.npz"))
    else:
        # Retained2048-cell minima coarsen exactly into512-cell minima.
        fine_paths = sorted((Path("results/phase3b/intermediate_graphs")/identifier).glob("graph_*_n2048.npz"))
    fine = [load_graph_record(p, C, n=n) for p in fine_paths]
    return coarse, fine


def compare_groups(left, right, *, rng, draws=32, paired=False):
    if paired and len(left) != len(right):
        raise ValueError("Paired cohorts must have matching physical paths")
    fl, fr = metric_features(left), metric_features(right)
    names = sorted(fl.keys() & fr.keys())
    basic_names = sorted(left[0].get("basic", {}).keys() & right[0].get("basic", {}).keys())
    a = np.array([[r["basic"][k] for k in basic_names] for r in left])
    b = np.array([[r["basic"][k] for k in basic_names] for r in right])
    basic_difference = a.mean(axis=0)-b.mean(axis=0)
    if paired:
        basic_se = (a-b).std(axis=0, ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.zeros(len(basic_names))
    else:
        basic_se = np.sqrt(a.var(axis=0, ddof=1)/len(a)+b.var(axis=0, ddof=1)/len(b))
    differences = []
    relative = []
    w1 = []
    for _ in range(draws):
        il = rng.integers(len(left), size=len(left))
        ir = il if paired else rng.integers(len(right), size=len(right))
        ll, rr = [left[i] for i in il], [right[i] for i in ir]
        xl, xr = metric_features(ll), metric_features(rr)
        differences.append([xl.get(k, np.nan)-xr.get(k, np.nan) for k in names])
        relative.append(np.mean([r["root_means"].mean() for r in ll])/np.mean([r["root_means"].mean() for r in rr])-1)
        w1.append(pair_discrepancy(ll, rr)["W1"])
    differences = np.asarray(differences)
    stats = {}
    for j, name in enumerate(names):
        valid = differences[:, j][np.isfinite(differences[:, j])]
        stats[name] = {"difference": fl[name]-fr[name],
            "bootstrap_mc95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "bootstrap_se": float(valid.std(ddof=1)) if len(valid)>1 else None}
    basic_stats = {name: {"difference": float(d), "mc_se": float(s)}
        for name, d, s in zip(basic_names, basic_difference, basic_se)}
    joint_names = [k for k in basic_names if k.startswith("joint_")]
    joint = max(joint_names, key=lambda k: abs(basic_stats[k]["difference"])) if joint_names else None
    copula_names = [k for k in basic_names if k.startswith("copula_")]
    copula = max(copula_names, key=lambda k: abs(basic_stats[k]["difference"])) if copula_names else None
    median_left, median_right = metric_features(left, "median"), metric_features(right, "median")
    return {"left_paths": len(left), "right_paths": len(right), "paired": paired,
        "normalization": "ensemble_mean", "metric_features": stats, "basic_features": basic_stats,
        "joint_CDF_max_observed": {"feature": joint, **basic_stats[joint]} if joint is not None else None,
        "rank_copula_max_observed": {"feature": copula, **basic_stats[copula]} if copula is not None else None,
        "mean_distance_relative_difference": float(np.mean([r["root_means"].mean() for r in left])/np.mean([r["root_means"].mean() for r in right])-1),
        "mean_relative_bootstrap_mc95": np.quantile(relative, [.025, .975]).tolist(),
        "pair": {**pair_discrepancy(left, right), "W1_bootstrap_mc95": np.quantile(w1, [.025, .975]).tolist()},
        "median_sensitivity": {"pair": pair_discrepancy(left, right, "median"),
            "metric_feature_differences": {k: v-median_right.get(k, v) for k, v in median_left.items()}},
        "bootstrap_draws": draws,
        "scope": "Whole physical realizations; independent levels are not coupled Brownian refinements"}


def harmonic_diagnostics(records):
    result = {}
    for r in records:
        h = r["header"]
        metadata = h.get("sampler_metadata", {})
        # Paired graph sizes repeat the same physical metadata; callers pass one n.
        result.setdefault(h.get("seed", r["source"]), metadata)
    legs = [l for m in result.values() for l in m.get("leg_diagnostics", [])]
    return {"physical_paths": len(result),
        "alpha1": [float(m["alpha1"]) for m in result.values() if "alpha1" in m],
        "max_abs_signed_leg_defect": max((abs(l["sum_log_mean_value_ratio"]) for l in legs), default=None),
        "max_abs_leg_defect_sum": max((l["sum_abs_log_mean_value_ratio"] for l in legs), default=None),
        "max_quadrature_log_difference": max((l["maximum_audited_quadrature_log_difference"] for l in legs), default=None),
        "mean_initial_ess_fraction": float(np.mean([m["initial_direction"]["importance_ess"]/m["initial_direction"]["candidates"] for m in result.values() if "initial_direction" in m])) if any("initial_direction" in m for m in result.values()) else None,
        "max_clock_interval_to_n2048_cell": max((m["max_normalized_dt"]*2048 for m in result.values() if "max_normalized_dt" in m), default=None)}


def same_model_floor(records, rng, splits=12):
    values = []
    for _ in range(splits):
        order = rng.permutation(len(records))
        half = len(order)//2
        values.append(pair_discrepancy([records[i] for i in order[:half]], [records[i] for i in order[half:2*half]])["W1"])
    return {"parent_count": len(records), "split_parents": len(records)//2,
        "W1_median": float(np.median(values)), "W1_range": [float(min(values)), float(max(values))],
        "scope": "Dependent disjoint re-splits of a small same-method cohort; conservative descriptive MC floor, not independent tests"}


def legacy_coupled_diagnostics(out, identifier):
    """Reuse the one full saved D6 path for each legacy fine configuration."""
    folder = Path("results/phase3b/intermediate_graphs")/identifier
    with np.load(folder/"first_path.npz") as a:
        metadata = json.loads(str(a["metadata"]))
        sample = ExcursionSample(a["times"].copy(), a["values"].copy(), None, metadata)
    with np.load(folder/"graph_000_n2048.npz") as a:
        old_points = a["points"].copy()
        if not np.array_equal(cell_minima(sample, 2048), a["minima"]):
            raise ValueError("Saved full legacy path does not match saved graph minima")
    full = {n: graph_record(cell_minima(sample, n), old_points//(2048//n)) for n in SIZES}
    result = []
    for q in [8, 32, 128]:
        sample_q = grid_thinning(sample, 2048, q)
        for n in SIZES:
            thin = graph_record(cell_minima(sample_q, n), full[n]["points"])
            a, b = set(map(tuple, thin["edges"])), set(map(tuple, full[n]["edges"]))
            result.append({"id": identifier, "n": n, "q": q,
                "source": str(folder/"first_path.npz"), "paired_physical_paths": 1,
                "mean_edge_jaccard": len(a&b)/len(a|b),
                "mean_minima_rms": float(np.sqrt(np.mean((thin["minima"]-full[n]["minima"])**2))),
                "mean_distance_relative": float(thin["root_means"].mean()/full[n]["root_means"].mean()-1),
                "mean_distance_relative_cluster_se": None,
                "coupling_scope": "same retained legacy polygonal path; only one full path saved, no population uncertainty estimate"})
    return result


def make_figures(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gates = json.loads((directory/"region_gate.json").read_text())
    coupled = json.loads((directory/"same_path_thinning.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    index = np.arange(len(gates))
    mean = np.array([g["coarse_fine_n2048"]["mean_distance_relative_difference"] for g in gates])*100
    interval = np.array([g["coarse_fine_n2048"]["mean_relative_bootstrap_mc95"] for g in gates])*100
    # Percentile endpoints can be asymmetric; draw intervals directly, since
    # the point estimate need not lie inside every small bootstrap interval.
    axes[0, 0].vlines(index, interval[:, 0], interval[:, 1], color="C0")
    axes[0, 0].scatter(index, mean, color=["C3" if g["id"] == "C019" else "C0" for g in gates])
    axes[0, 0].axhline(0, color="black", lw=.7)
    axes[0, 0].axhspan(-2, 2, color="grey", alpha=.12, label="working 2% goal")
    axes[0, 0].set_ylabel("Fine-minus-coarse mean distance (%)")
    axes[0, 0].set_title("Independent cohorts: 8 fine / 6 coarse parents")
    axes[0, 0].legend(fontsize=8)
    for stage, color in [("coarse", "C1"), ("fine", "C0")]:
        values = [g["harmonic"][stage]["max_abs_signed_leg_defect"] for g in gates]
        axes[0, 1].plot(index, values, "o-", color=color, label=stage)
    axes[0, 1].axhline(.03, color="C3", ls="--", label="warning .03; no bound")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Largest observed |signed leg log defect|")
    axes[0, 1].set_title("Harmonic normalization alone is insufficient")
    axes[0, 1].legend(fontsize=8)
    for axis in axes[0]:
        axis.set_xticks(index, [g["id"] for g in gates], rotation=50)
    q_values = [8, 32, 128]
    for column, name, label in [(0, "mean_edge_jaccard", "Edge Jaccard against full retained path"),
                                (1, "mean_distance_relative", "Same-path mean-distance change (%)")]:
        rows = [[r[name] for r in coupled if r["n"] == 2048 and r["q"] == q] for q in q_values]
        rows = [np.asarray(r)*(100 if column else 1) for r in rows]
        axes[1, column].plot(q_values, [np.median(r) for r in rows], "o-", label="median over 10 regions")
        axes[1, column].fill_between(q_values, [min(r) for r in rows], [max(r) for r in rows], alpha=.2, label="range over regions")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks(q_values, [str(q) for q in q_values])
        axes[1, column].set_xlabel("Uniform observations per graph cell")
        axes[1, column].set_ylabel(label)
        axes[1, column].set_title("Paired retained-polyline thinning; not Brownian error")
        axes[1, column].legend(fontsize=8)
    fig.text(.5, .008, "Coupled lower panels: 7 regions with 8 paths each; 3 legacy regions with 1 saved path each. Shading is a region range.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(directory/"sampler_reliability_gate.pdf")
    fig.savefig(directory/"sampler_reliability_gate.png", dpi=170)
    plt.close(fig)


def analyze(out):
    import csv
    start = time.perf_counter()
    destination = out/"analysis"
    destination.mkdir(exist_ok=True)
    if (destination/"summary.json").exists():
        raise FileExistsError("Completed analysis exists; preserve it before a new analysis version")
    rng = np.random.default_rng([BASE, 9901])
    comparisons = []
    gates = []
    budgets = []
    for identifier in IDS:
        for n in SIZES:
            coarse, fine = get_groups(out, identifier, n)
            comparison = {"id": identifier, "n": n, "contrast": "independent_D6_fine_minus_D4_coarse",
                **compare_groups(fine, coarse, rng=rng)}
            comparison["same_method_pair_MC_floor"] = {"coarse": same_model_floor(coarse, rng), "fine": same_model_floor(fine, rng)}
            comparisons.append(comparison)
            if n != 2048:
                continue
            hd = {"coarse": harmonic_diagnostics(coarse), "fine": harmonic_diagnostics(fine)}
            unsafe = identifier == "C019" or (hd["coarse"]["max_abs_signed_leg_defect"] or 0) > .1
            gates.append({"id": identifier, "spec": specs()[identifier], "harmonic": hd,
                "status": "unsafe_for_family_interpretation" if unsafe else "provisional_empirical_region",
                "reason": "Large observed harmonic accumulation and D8 positivity-certificate failure" if unsafe else "Observed shifts and MC uncertainty retained; full continuous joint-minimum error unbounded",
                "coarse_fine_n2048": comparison})
            metrics = comparison["metric_features"]
            rows = [("mean_relative", comparison["mean_distance_relative_difference"], comparison["mean_relative_bootstrap_mc95"], TARGETS["mean_relative"]),
                ("pair_W1", comparison["pair"]["W1"], comparison["pair"]["W1_bootstrap_mc95"], TARGETS["normalized_pair_W1"])]
            joint = comparison["joint_CDF_max_observed"]
            rows.append(("joint_minimum_CDF", joint["difference"], (np.array([-1.96, 1.96])*joint["mc_se"]+joint["difference"]).tolist(), TARGETS["joint_minimum_CDF"]))
            copula = comparison["rank_copula_max_observed"]
            rows.append(("rank_copula_CDF", copula["difference"], (np.array([-1.96, 1.96])*copula["mc_se"]+copula["difference"]).tolist(), TARGETS["rank_copula_CDF"]))
            for name in ["ball_sd_r0.5", "triangles_small_mean", "triangles_middle_mean", "quartets_delta_mean", "quartets_aspect_mean", "matrix_shared_product_correlation", "matrix_disjoint_product_correlation"]:
                family = "ball_sd" if name.startswith("ball") else "triangle_mean" if name.startswith("triangles") else "quartet_mean" if name.startswith("quartets") else "matrix_correlation"
                rows.append((name, metrics[name]["difference"], metrics[name]["bootstrap_mc95"], TARGETS[family]))
            for name, value, interval, goal in rows:
                envelope = max(abs(v) for v in interval)
                budgets.append({"id": identifier, "observable": name, "observed_difference": value,
                    "MC95": interval, "working_goal": goal, "empirical_sensitivity_envelope": envelope,
                    "status": "unsafe_region" if unsafe else "uncertainty_exceeds_goal" if envelope > goal else "stability_observed_within_goal",
                    "qualification": "Pointwise empirical sensitivity; not a uniform/systematic-error bound"})
    save_json(destination/"coarse_fine_comparisons.json", comparisons)
    save_json(destination/"region_gate.json", gates)
    with (destination/"observable_error_budget.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(budgets[0]))
        writer.writeheader()
        for row in budgets:
            writer.writerow({**row, "MC95": json.dumps(row["MC95"])})
    # New fine physical paths retain every coupled q observation checkpoint.
    coupled = []
    for identifier in MISSING:
        originals = [json.loads(p.read_text()) for p in sorted((out/"records").glob(f"{identifier}_L1_*.json"))]
        originals = [r for r in originals if r["status"] == "success"]
        for n in SIZES:
            C = specs()[identifier]["covariance"]
            full = [load_graph_record(out/"arrays"/f"{identifier}_L1_{r['realization']:03d}_n{n}.npz", C) for r in originals]
            for q in [8, 32, 128]:
                primitive = [r["coupled"][str(q)][str(n)] for r in originals]
                result = {"id": identifier, "n": n, "q": q, "paired_physical_paths": len(primitive),
                    "mean_edge_jaccard": float(np.mean([r["edge_jaccard"] for r in primitive])),
                    "mean_minima_rms": float(np.mean([r["minima_rms"] for r in primitive])),
                    "mean_distance_relative": float(np.mean([r["mean_distance_relative"] for r in primitive])),
                    "mean_distance_relative_cluster_se": float(np.std([r["mean_distance_relative"] for r in primitive], ddof=1)/np.sqrt(len(primitive))),
                    "coupling_scope": "same retained polygonal paths, not independent/coarse WOS laws"}
                if q == 32 and n == 2048:
                    thin = [load_graph_record(out/"arrays"/f"{identifier}_L1_{r['realization']:03d}_q{q}_n{n}.npz", C) for r in originals]
                    result["higher_point_and_ball_paired_comparison"] = compare_groups(thin, full, rng=rng, paired=True, draws=16)
                coupled.append(result)
    for identifier in ["C000", "C006", "C028"]:
        coupled.extend(legacy_coupled_diagnostics(out, identifier))
    save_json(destination/"same_path_thinning.json", coupled)
    exact_controls = []
    for identifier, case in [("C002", "i3"), ("C007", "a2"), ("C012", "a3")]:
        _, fine = get_groups(out, identifier, 2048)
        fine8 = [{**r, "cdfs": r["cdfs"][:8], "root_means": r["root_means"][:8], "matrix": r["matrix"][:8, :8]} for r in fine]
        references = []
        for path in sorted(Path("results/phase3a/graphs").glob(f"graph_{case}_n2048_*.npz")):
            with np.load(path) as array:
                references.append({k: array[k].copy() for k in ["grid", "cdfs", "root_means", "matrix"]})
                references[-1].update(discrete=True, basic={}, source=str(path))
        exact_controls.append({"id": identifier, "case": case, "n": 2048,
            "contrast": "new_D6_provisional_vs_preserved_Phase3A_reference",
            "matrix_root_points": 8, "reference_minima_scope": "I3 exact continuous; A2/A3 q32 polygonal",
            **compare_groups(fine8, references, rng=rng)})
    save_json(destination/"exact_reference_metric_controls.json", exact_controls)
    failures = [json.loads(p.read_text()) for p in (out/"records").glob("*.json") if json.loads(p.read_text())["status"] != "success"]
    summary = {"new_successful_physical_paths": sum(json.loads(p.read_text())["status"] == "success" for p in (out/"records").glob("*.json")),
        "new_model_or_path_failures": failures, "reused_fine_physical_paths": 24,
        "reused_coarse_physical_paths": 60, "region_gate_count": len(gates),
        "unsafe_regions": [g["id"] for g in gates if g["status"].startswith("unsafe")],
        "all_other_regions": "provisional, per-observable uncertainty and unbounded continuous-path error remain",
        "analysis_seconds": time.perf_counter()-start, "primary_normalization": "ensemble mean pair distance",
        "prospective_warning_gates": {"absolute_signed_leg_defect": .03, "sum_absolute_leg_defects": .1,
            "scope": "Warning screens, not uniform guarantees; large observed values preclude family interpretation"},
        "no_global_validated_claim": True}
    save_json(destination/"summary.json", summary)
    make_figures(destination)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["plan", "pilot", "generate", "analyze"])
    parser.add_argument("--output", type=Path, default=Path("results/phase3c/calibration/sampler"))
    parser.add_argument("--ids", nargs="+", default=MISSING)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    out = args.output
    for folder in [out, out/"records", out/"arrays", out/"paths"]:
        folder.mkdir(parents=True, exist_ok=True)
    plan = reuse_inventory(out)
    if args.mode == "pilot":
        start = time.perf_counter()
        records = []
        for item in plan["pilot"]:
            if time.perf_counter()-start > 27:
                break
            records.append(simulate_one(out, item["id"], item["level"], item["realization"], pilot=True))
        save_json(out/"pilot_cost.json", {"wall_seconds": time.perf_counter()-start,
            "attempted_physical_paths": len(records), "successful_physical_paths": sum(r["status"] == "success" for r in records), "failures": [r for r in records if r["status"] != "success"],
            "mean_seconds_per_path": float(np.mean([r["seconds"] for r in records])),
            "next_step": "Coordinator reviews cost/status before generate mode; no automatic large run"})
    elif args.mode == "generate":
        for identifier in args.ids:
            for realization in range(args.samples):
                simulate_one(out, identifier, 1, realization)
    elif args.mode == "analyze":
        analyze(out)


if __name__ == "__main__":
    main()
