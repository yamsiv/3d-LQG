"""Saved-data sampler reliability audit for Task3B; never launches simulation.

Every physical path is counted once per stage, irrespective of graph sizes.
Different algorithms sharing a seed are explicitly flagged, not treated as
independent paths or collapsed merely because their legacy ID strings match.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from src.metric_features_3b import matrix_shape_samples
from src.provenance import environment_manifest


STAGES = ["screen_graphs", "adaptive_coarse_graphs", "intermediate_graphs",
          "birect_extension_graphs", "holdout_graphs", "anchor_controls"]
ANCHORS = {"C002": "i3", "C007": "a2", "C012": "a3"}
RADII = np.array([.35, .5, .75, 1., 1.5])


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def save_csv(path, rows):
    names = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def analytic_alpha(C):
    values = np.sort([C[0, 1], C[0, 2], C[1, 2]])
    if np.allclose(values, [-.5, -.5, 0], atol=1e-12, rtol=0):
        return 6.5
    nonzero = values[abs(values) > 1e-12]
    if len(nonzero) <= 1:
        rho = float(nonzero[0]) if len(nonzero) else 0.
        return float(np.pi/np.arccos(-rho)+1.5)
    return None


def _mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else None


def inventory(root):
    records, failures, source_hashes = [], [], {}
    for stage in STAGES:
        folder = root/stage
        if not folder.exists():
            continue
        configuration = json.loads((folder/"metadata.json").read_text())["configuration"]
        for path in sorted(folder.glob("C*/graph_*_n*.npz")):
            with np.load(path) as array:
                header = json.loads(str(array["header"]))
                data = {k: array[k].copy() for k in ["grid", "cdfs", "root_means", "matrix"]}
                minima = array["minima"]
                ties = [len(minima)-2-len(np.unique(minima[1:-1, i])) for i in range(3)]
            data.update(header=header, discrete=True, source=str(path), stage=stage,
                        ties=ties, physical_key=(stage, header["id"], header["seed"]))
            records.append(data)
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(list(folder.glob("C*/model_failure.json"))+list(folder.glob("C*/failure_*.json"))):
            failure = json.loads(path.read_text())
            spec = failure.get("spec")
            C = np.array(spec["covariance"]) if spec is not None else None
            failures.append({"stage": stage, "id": path.parent.name, "source": str(path),
                "type": "model_construction_failure" if path.name == "model_failure.json" else "path_generation_failure",
                "error": failure["error"], "planned_samples": configuration["samples"],
                "sizes": configuration["sizes"], "covariance_spd": bool(np.linalg.eigvalsh(C).min() > 0) if C is not None else None,
                "triple": spec.get("triple") if spec else None,
                "condition": spec.get("condition") if spec else None,
                "explanation": "Positive Ritz polynomial certificate failed, not covariance invalidity" if "positivity" in failure["error"].lower() else "See raw failure"})
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records, failures, source_hashes


def quality_rows(records):
    table, path_rows = [], []
    unique = {}
    for record in records:
        key = record["physical_key"]
        unique.setdefault(key, record)
    for key, record in sorted(unique.items()):
        h = record["header"]
        m = h["sampler_metadata"]
        C = np.array(m["covariance"])
        alpha = float(m["alpha1"])
        oracle = analytic_alpha(C)
        legs = m.get("leg_diagnostics", [])
        times = np.array(m.get("hitting_times", []))
        provisional = not m.get("grid_values_exact", False)
        physical = {"stage": key[0], "id": key[1], "seed": key[2],
            "legacy_physical_path_id": h.get("physical_path_id"),
            "alpha1": alpha, "analytic_alpha1": oracle,
            "alpha_error_if_analytic": alpha-oracle if oracle is not None else None,
            "ritz_degree": m.get("ritz_degree"),
            "sampling_law": m.get("law"), "sampler": m["sampler"],
            "reliability": "provisional" if provisional else ("exact_continuous_minima" if m.get("cell_minima_exact") else "exact_grid_polygonal_minima"),
            "analytic_h_available": oracle is not None,
            "lifetime_mean_observed": float(times.mean()) if len(times) else None,
            "target_lifetime_mean": float(.5/(alpha-1)) if alpha > 1 else None,
            "target_lifetime_moment_scope": "Exact intended cone law; generic numerical alpha makes this a diagnostic, not proof of sampler law",
            "target_lifetime_variance_finite": alpha > 2,
            "lifetime_heavy_tail_flag": "infinite_second_moment" if alpha <= 2 else ("near_second_moment_threshold" if alpha < 2.5 else "inverse_gamma_heavy_tail"),
            "initial_ess": m.get("initial_direction", {}).get("importance_ess"),
            "initial_candidates": m.get("initial_direction", {}).get("candidates"),
            "initial_ess_fraction": m.get("initial_direction", {}).get("importance_ess", 0)/m.get("initial_direction", {}).get("candidates", 1) if provisional else None,
            "max_normalized_dt": m.get("max_normalized_dt"),
            "step_size": m.get("step_size"), "relative_step": m.get("relative_step"),
            "exit_order": m.get("exit_order"), "oversample": m.get("oversample"),
            "sum_log_mean_value_ratio_both_legs": float(sum(l["sum_log_mean_value_ratio"] for l in legs)) if legs else None,
            "maximum_abs_signed_leg_defect": max((abs(l["sum_log_mean_value_ratio"]) for l in legs), default=None),
            "sum_abs_log_mean_value_ratio_both_legs": float(sum(l["sum_abs_log_mean_value_ratio"] for l in legs)) if legs else None,
            "maximum_leg_abs_defect_sum": max((l["sum_abs_log_mean_value_ratio"] for l in legs), default=None),
            "maximum_quadrature_log_difference": max((l["maximum_audited_quadrature_log_difference"] for l in legs), default=None),
            "observed_graph_sizes": sorted(r["header"]["n"] for r in records if r["physical_key"] == key)}
        path_rows.append(physical)
    for stage, identifier, n in sorted({(r["stage"], r["header"]["id"], r["header"]["n"]) for r in records}):
        selected = [r for r in records if (r["stage"], r["header"]["id"], r["header"]["n"]) == (stage, identifier, n)]
        physical = [p for p in path_rows if p["stage"] == stage and p["id"] == identifier]
        first = selected[0]["header"]
        row = {"stage": stage, "id": identifier, "n": n,
            "triple": first["spec"]["triple"], "family": first["spec"]["family"],
            "covariance_spd": True, "condition": first["spec"]["condition"],
            "independent_paths_within_stage": len(selected),
            "reliability": physical[0]["reliability"],
            "alpha1": physical[0]["alpha1"], "analytic_alpha1": physical[0]["analytic_alpha1"],
            "ritz_degree": physical[0]["ritz_degree"], "oversample": physical[0]["oversample"],
            "sampler": physical[0]["sampler"], "sampling_law": physical[0]["sampling_law"],
            "grid_values_exact": first["sampler_metadata"].get("grid_values_exact", False),
            "continuous_minima_exact": first["sampler_metadata"].get("cell_minima_exact", False),
            "effective_oversample_at_graph_n": float(first["sampler_metadata"]["oversample"]*max(first["configuration"]["sizes"])/n)
                if first["sampler_metadata"].get("grid_values_exact") and not first["sampler_metadata"].get("cell_minima_exact") else None,
            "target_lifetime_mean": physical[0]["target_lifetime_mean"],
            "lifetime_heavy_tail_flag": physical[0]["lifetime_heavy_tail_flag"],
            "target_lifetime_variance_finite": physical[0]["target_lifetime_variance_finite"],
            "lifetime_observed_mean": float(np.mean([p["lifetime_mean_observed"] for p in physical])),
            "lifetime_cluster_se": _mean_se([p["lifetime_mean_observed"] for p in physical])[1],
            "all_graph_invariants_pass": all(r["header"]["validation"]["connected"] and r["header"]["validation"]["hamiltonian_cycle"]
                and r["header"]["validation"]["coordinate_diagonal_counts"] == [n-3]*3 for r in selected),
            "ties_fraction_mean": float(np.mean([np.mean(r["ties"])/(n-2) for r in selected])),
            "ties_fraction_max": float(np.max([max(r["ties"])/(n-2) for r in selected])),
            "root_count": len(selected[0]["cdfs"]), "matrix_points": len(selected[0]["matrix"])}
        for field in ["initial_ess", "initial_ess_fraction", "maximum_abs_signed_leg_defect", "maximum_leg_abs_defect_sum", "maximum_quadrature_log_difference", "sum_log_mean_value_ratio_both_legs", "sum_abs_log_mean_value_ratio_both_legs"]:
            values = [p[field] for p in physical if p[field] is not None]
            row[field+"_mean"] = float(np.mean(values)) if values else None
            row[field+"_min"] = float(min(values)) if values else None
            row[field+"_max"] = float(max(values)) if values else None
        normalized_dt = [p["max_normalized_dt"]*n for p in physical if p["max_normalized_dt"] is not None]
        row["max_normalized_dt_times_n_mean"] = float(np.mean(normalized_dt)) if normalized_dt else None
        row["max_normalized_dt_times_n_max"] = float(max(normalized_dt)) if normalized_dt else None
        table.append(row)
    return table, path_rows


class Ensemble:
    """Fast realization-bootstrap features, using exactly eight measure points."""
    def __init__(self, records):
        self.records = records
        size = max(len(r["grid"]) for r in records)
        self.curves = np.ones((len(records), 8, size))
        self.means = np.empty((len(records), 8))
        shape_rows = []
        moments = []
        self.shape_names = []
        for i, record in enumerate(records):
            if not np.array_equal(record["grid"], np.arange(len(record["grid"]))):
                raise ValueError("Graph integer supports required")
            self.curves[i, :, :len(record["grid"])] = record["cdfs"][:8]
            self.means[i] = record["root_means"][:8]
            shape = matrix_shape_samples(record["matrix"][:8, :8])
            values = []
            names = []
            for key, labels in [("triangles", ["small", "middle"]), ("quartets", ["delta", "aspect"])]:
                for axis, label in enumerate(labels):
                    values.extend([np.mean(shape[key][:, axis]), *np.quantile(shape[key][:, axis], [.1, .5, .9])])
                    names.extend([f"{key}_{label}_mean", *[f"{key}_{label}_q{q:g}" for q in [.1, .5, .9]]])
            shape_rows.append(values)
            self.shape_names = names
            moments.append([shape[k] for k in ["pair_mean", "pair_second", "shared_product", "disjoint_product"]])
        self.shapes = np.asarray(shape_rows)
        self.moments = np.asarray(moments)

    def summarize(self, indices=None):
        indices = np.arange(len(self.records)) if indices is None else indices
        curves = self.curves[indices]
        means = self.means[indices]
        cdf = curves.mean(axis=(0, 1))
        norm = int(np.searchsorted(cdf, .5, side="left"))
        if norm <= 0:
            raise ValueError("Positive median required")
        result = {"mean_distance": float(means.mean()), "ensemble_pair_median": float(norm),
            "mean_over_median": float(means.mean()/norm),
            "root_mean_distance_cv": float(means.mean(axis=1).std(ddof=1)/means.mean())}
        for q in [.1, .25, .75, .9, .95]:
            result[f"pair_q{q:g}_over_median"] = float(np.searchsorted(cdf, q, side="left")/norm)
        for radius in RADII:
            index = min(int(np.floor(radius*norm)), curves.shape[2]-1)
            ball = curves[:, :, index].ravel()
            result[f"ball_mean_r{radius:g}"] = float(ball.mean())
            result[f"ball_sd_r{radius:g}"] = float(ball.std())
            result[f"ball_centered_q10_r{radius:g}"] = float(np.quantile(ball, .1)-ball.mean())
            result[f"ball_centered_q90_r{radius:g}"] = float(np.quantile(ball, .9)-ball.mean())
        for radius in [2, 3, 4, 5, 6]:
            ball = curves[:, :, min(radius, curves.shape[2]-1)].ravel()
            result[f"fixed_integer_ball_mean_r{radius}"] = float(ball.mean())
            result[f"fixed_integer_ball_sd_r{radius}"] = float(ball.std())
        result.update({name: float(value) for name, value in zip(self.shape_names, self.shapes[indices].mean(axis=0))})
        mu, second, shared, disjoint = self.moments[indices].mean(axis=0)
        variance = second-mu*mu
        result["matrix_shared_correlation"] = float((shared-mu*mu)/variance)
        result["matrix_disjoint_correlation"] = float((disjoint-mu*mu)/variance)
        return result


def compare(a, b, *, bootstrap, rng):
    x, y = Ensemble(a), Ensemble(b)
    fx, fy = x.summarize(), y.summarize()
    keys = list(fx)
    samples = np.empty((bootstrap, len(keys)))
    relative_mean_samples = []
    for i in range(bootstrap):
        xx = x.summarize(rng.integers(len(a), size=len(a)))
        yy = y.summarize(rng.integers(len(b), size=len(b)))
        samples[i] = [xx[k]-yy[k] for k in keys]
        relative_mean_samples.append(xx["mean_distance"]/yy["mean_distance"]-1)
    return {"left_paths": len(a), "right_paths": len(b), "roots_and_matrix_points": 8,
        "left": fx, "right": fy,
        "features": {k: {"difference": fx[k]-fy[k],
            "bootstrap_mc95": np.quantile(samples[:, i], [.025, .975]).tolist(),
            "bootstrap_se": float(samples[:, i].std(ddof=1))} for i, k in enumerate(keys)},
        "mean_distance_relative_difference": fx["mean_distance"]/fy["mean_distance"]-1,
        "mean_distance_relative_bootstrap_mc95": np.quantile(relative_mean_samples, [.025, .975]).tolist(),
        "bootstrap": bootstrap,
        "notes": ["Whole-realization bootstrap, independent seed ensembles; no pair/root pseudo-replication",
            "First eight roots/matrix points used in BOTH groups; per-realization shape quantiles then ensemble average",
            "Ensemble median re-estimated in each bootstrap; integer graph atoms may cause discontinuous ball-radius changes",
            "Cross-resolution differences bundle angular, clock, path and finite-sample changes; not a pure bias estimate"]}


def comparisons(records, *, bootstrap, seed):
    rng = np.random.default_rng(seed)
    anchor, refinement = [], []
    for identifier, case in ANCHORS.items():
        left = [r for r in records if r["stage"] == "anchor_controls" and r["header"]["id"] == identifier and r["header"]["n"] == 2048]
        right = []
        for path in sorted(Path("results/phase3a/graphs").glob(f"graph_{case}_n2048_*.npz")):
            with np.load(path) as array:
                right.append({k: array[k].copy() for k in ["grid", "cdfs", "root_means", "matrix"]})
                right[-1]["header"] = json.loads(str(array["header"]))
                right[-1]["source"] = str(path)
        if left and right:
            if set(r["header"]["seed"] for r in left) & set(r["header"]["seed"] for r in right):
                raise ValueError("Anchor and Phase3A reference seeds overlap")
            anchor.append({"id": identifier, "case": case, "n": 2048,
                "contrast": "provisional_D4_minus_saved_Phase3A_reference", **compare(left, right, bootstrap=bootstrap, rng=rng)})
    mids = sorted({r["header"]["id"] for r in records if r["stage"] == "intermediate_graphs"})
    for identifier in mids:
        left = [r for r in records if r["stage"] == "intermediate_graphs" and r["header"]["id"] == identifier and r["header"]["n"] == 2048]
        right = [r for r in records if r["stage"] in {"screen_graphs", "adaptive_coarse_graphs"} and r["header"]["id"] == identifier and r["header"]["n"] == 2048]
        if left and right:
            if set(r["header"]["seed"] for r in left) & set(r["header"]["seed"] for r in right):
                raise ValueError("Coarse and intermediate seeds overlap")
            exact = left[0]["header"]["sampler_metadata"].get("grid_values_exact", False)
            continuous_minima = left[0]["header"]["sampler_metadata"].get("cell_minima_exact", False)
            refinement.append({"id": identifier, "n": 2048,
                "contrast": ("independent_exact_continuous_minimum_anchor_seed_cohorts" if continuous_minima else
                    "independent_exact_grid_anchor_seed_cohorts_with_effective_polygonal_grid_refinement") if exact else "D6_finer_minus_D4_coarse_independent_seeds",
                **compare(left, right, bootstrap=bootstrap, rng=rng)})
    return anchor, refinement


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/phase3b"))
    parser.add_argument("--output", type=Path, default=Path("results/phase3b/quality"))
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=202611113989)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Use a new quality output; prior results preserved")
    args.output.mkdir(parents=True)
    records, failures, hashes = inventory(args.root)
    table, physical = quality_rows(records)
    failure_rows = [{"stage": f["stage"], "id": f["id"], "n": n, "triple": f["triple"],
        "condition": f["condition"], "covariance_spd": f["covariance_spd"],
        "independent_paths_within_stage": 0, "planned_samples": f["planned_samples"],
        "reliability": "model_construction_failed_positive_Ritz_certificate", "failure": f["error"]}
        for f in failures for n in f["sizes"]]
    save_csv(args.output/"sampler_accuracy_by_stage_configuration_size.csv", table+failure_rows)
    save_csv(args.output/"physical_path_diagnostics.csv", physical)
    save_json(args.output/"failures.json", failures)
    seed_groups = {}
    for p in physical:
        seed_groups.setdefault((p["id"], p["seed"]), []).append(p)
    shared = [{"id": key[0], "seed": key[1], "stages": [p["stage"] for p in group],
        "samplers": [p["sampler"] for p in group],
        "explanation": "Distinct paths/algorithms but same RNG seed; not independent across these stage comparisons"}
        for key, group in seed_groups.items() if len(group) > 1]
    anchor, refinement = comparisons(records, bootstrap=args.bootstrap, seed=args.seed)
    save_json(args.output/"anchor_comparisons_n2048.json", anchor)
    save_json(args.output/"coarse_refined_comparisons_n2048.json", refinement)
    status = []
    for identifier in sorted({r["header"]["id"] for r in records} | {f["id"] for f in failures}):
        selected = [t for t in table if t["id"] == identifier]
        if not selected:
            status.append({"id": identifier, "status": "failed_positive_Ritz_certificate_for_SPD_covariance",
                "stage": next(f["stage"] for f in failures if f["id"] == identifier),
                "conclusion": "No samples; numerical certificate failure is not mathematical inadmissibility"})
            continue
        exact = [t for t in selected if t["reliability"] != "provisional"]
        approximate = [t for t in selected if t["reliability"] == "provisional"]
        largest = max((t["maximum_abs_signed_leg_defect_max"] or 0 for t in approximate), default=0)
        status.append({"id": identifier, "triple": selected[0]["triple"],
            "status": exact[0]["reliability"] if exact else "provisional_arbitrary_covariance",
            "stages": sorted({t["stage"] for t in selected}),
            "exact_reference_available": bool(exact), "analytic_h_available": any(p["analytic_h_available"] for p in physical if p["id"] == identifier),
            "largest_observed_absolute_signed_leg_defect": largest,
            "large_observed_h_defect_flag": largest > .1,
            "conclusion": "Coarse kernel has substantial observed nonharmonic normalization effects; exploratory ranking is unreliable without independent improvement" if largest > .1 else
                ("Use exact reference and retain its stated minimum discretization" if exact else "Provisional only; small observed harmonic defects do not control full path/minimum law")})
    save_json(args.output/"configuration_reliability.json", status)
    brief = []
    for label, groups in [("anchor_minus_reference", anchor), ("refined_minus_coarse", refinement)]:
        for group in groups:
            row = {"comparison": label, "id": group["id"], "n": 2048,
                "left_paths": group["left_paths"], "right_paths": group["right_paths"],
                "mean_distance_relative_difference": group["mean_distance_relative_difference"]}
            row["mean_distance_relative_mc95"] = group["mean_distance_relative_bootstrap_mc95"]
            for name in ["mean_distance", "ball_sd_r0.5", "triangles_small_mean", "quartets_delta_mean", "matrix_shared_correlation"]:
                row[name+"_difference"] = group["features"][name]["difference"]
                row[name+"_mc95"] = group["features"][name]["bootstrap_mc95"]
            brief.append(row)
    save_csv(args.output/"leading_quality_comparisons.csv", brief)
    summary = {"physical_paths_stage_qualified": len(physical), "saved_graphs": len(records),
        "counts_by_stage": {stage: sum(p["stage"] == stage for p in physical) for stage in STAGES},
        "unique_legacy_id_seed_pairs": len(seed_groups), "shared_seed_cross_stage_groups": shared,
        "failure_count": len(failures), "failures": failures,
        "all_saved_graph_invariants_pass": all(t["all_graph_invariants_pass"] for t in table),
        "runtime_seconds": time.perf_counter()-start,
        "bootstrap": args.bootstrap, "seed": args.seed,
        "warnings": ["All arbitrary-C graph laws remain provisional",
            "Mean lifetime formula for a generic numerical cone is a target diagnostic, not a validated sampler identity",
            "max_normalized_dt*n is a worst retained clock interval, not a uniform omitted-bridge error bound",
            "No precise Hausdorff dimensions inferred from this audit",
            "C022 failure is a numerical positivity-certificate failure for an SPD covariance"]}
    save_json(args.output/"summary.json", summary)
    manifest = {"provenance": environment_manifest(), "input_sha256": hashes,
        "external_Phase3A_reference_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for case in ANCHORS.values() for p in sorted(Path("results/phase3a/graphs").glob(f"graph_{case}_n2048_*.npz"))},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(args.output/"manifest.json", manifest)
    (args.output/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
