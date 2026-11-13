"""Saved-data audit of the109-C Stage-II graph campaign; no simulation.

The final audit refuses incomplete batch markers. Every graph/parent, including
unsafe numerical paths, remains in the registry. Graph sizes are paired
observations, never extra parents. Four deterministic parents/config optionally
supply secondary degree histograms after a saved reconstruction benchmark.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time
import numpy as np

from src.mated_crt import graph_from_minima
from src.provenance import environment_manifest

ROOT = Path("results/phase3c")
SIGNED_WARNING = .03
ABSOLUTE_WARNING = .10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def save_csv(path, rows):
    names = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})


def campaign_inputs():
    campaign = json.loads((ROOT/"screen/campaign.json").read_text())
    design = json.loads((ROOT/"covariance_design/initial_design.json").read_text())
    return campaign, {s["id"]: s for s in design["configurations"]}


def readiness(campaign):
    missing = [b["label"] for b in campaign["batches"] if not (Path(b["output"])/"BATCH_COMPLETE.json").exists()]
    return {"expected_batches": len(campaign["batches"]),
        "complete_batches": len(campaign["batches"])-len(missing), "missing_batches": missing,
        "ready_for_final_audit": not missing}


def degree_benchmark():
    paths = sorted(Path("results/phase3b/screen_graphs").glob("C*/graph_*_n2048.npz"))[:5]
    start = time.perf_counter()
    for p in paths:
        with np.load(p) as a:
            graph_from_minima(a["minima"])
    elapsed = time.perf_counter()-start
    return {"graphs": len(paths), "seconds": elapsed,
        "projected_4_parents_times109_seconds": elapsed/max(1, len(paths))*436,
        "decision": "include_first4_degree_histograms" if elapsed/max(1, len(paths))*436 < 30 else "skip_secondary_degrees",
        "scope": "Saved minima reconstructed; no Brownian/path generation"}


def equivalent_covariance(actual, declared):
    return any(np.allclose(np.asarray(actual)[np.ix_(p, p)], declared, atol=1e-11, rtol=0)
               for p in permutations(range(3)))


def path_diagnostics(header, source_stage):
    m = header["sampler_metadata"]
    legs = m.get("leg_diagnostics", [])
    signed = [float(l["sum_log_mean_value_ratio"]) for l in legs]
    absolute = [float(l["sum_abs_log_mean_value_ratio"]) for l in legs]
    alpha = float(m["alpha1"])
    initial = m.get("initial_direction", {})
    return {"id": header["id"], "seed": header["seed"], "source_stage": source_stage,
        "physical_path_id": header.get("physical_path_id"), "sampler": m["sampler"],
        "law": m["law"], "target_law": m.get("target_law", m["law"]),
        "grid_values_exact": m.get("grid_values_exact", False),
        "continuous_minima_exact": m.get("cell_minima_exact", False),
        "alpha1": alpha, "ritz_degree": m.get("ritz_degree"),
        "signed_leg_log_defects": signed, "absolute_leg_log_defect_sums": absolute,
        "max_abs_signed_leg_log_defect": max(map(abs, signed), default=0.),
        "max_absolute_leg_log_defect_sum": max(absolute, default=0.),
        "harmonic_warning": any(abs(x)>SIGNED_WARNING for x in signed) or any(x>ABSOLUTE_WARNING for x in absolute),
        "max_exit_quadrature_log_difference": max((l["maximum_audited_quadrature_log_difference"] for l in legs), default=None),
        "initial_ess": initial.get("importance_ess"), "initial_candidates": initial.get("candidates"),
        "initial_ess_fraction": initial["importance_ess"]/initial["candidates"] if initial else None,
        "hitting_times": m.get("hitting_times"), "target_leg_lifetime_mean": .5/(alpha-1) if alpha>1 else None,
        "target_lifetime_variance_finite": alpha>2,
        "lifetime_scope": "Intended true cone law; approximate generic eigenfunction/kernel need not satisfy this identity",
        "max_normalized_wos_dt": m.get("max_normalized_dt"),
        "sampler_limitations": m.get("limitations", [])}


def check_arrays(path, expected_n, expected_points=16):
    with np.load(path) as a:
        h = json.loads(str(a["header"]))
        grid, cdf, means = a["grid"], a["cdfs"], a["root_means"]
        matrix, points, minima = a["matrix"], a["points"], a["minima"]
    checks = {"header_size": h["n"] == expected_n,
        "root_count": cdf.shape[0] == expected_points and len(means) == expected_points,
        "matrix_shape": matrix.shape == (expected_points, expected_points),
        "point_count": len(points) == expected_points,
        "point_range": bool(np.all((points>=0)&(points<expected_n))),
        "integer_support": bool(np.array_equal(grid, np.arange(len(grid)))),
        "cdf_finite": bool(np.all(np.isfinite(cdf))),
        "cdf_monotone": bool(np.all(np.diff(cdf, axis=1)>=-1e-12)),
        "cdf_total_mass": bool(np.allclose(cdf[:, -1], 1., atol=1e-12)),
        "cdf_root_atom": bool(np.allclose(cdf[:, 0], 1/expected_n, atol=1e-12)),
        "mean_tail_identity": bool(np.allclose(np.sum(1-cdf, axis=1), means, atol=1e-10)),
        "matrix_finite_nonnegative": bool(np.all(np.isfinite(matrix)) and np.all(matrix>=0)),
        "matrix_symmetric": bool(np.allclose(matrix, matrix.T, atol=1e-12)),
        "matrix_zero_diagonal": bool(np.allclose(np.diag(matrix), 0., atol=1e-12)),
        "matrix_zero_iff_vertex_collision": bool(np.array_equal(matrix==0, points[:, None]==points[None, :])),
        "matrix_triangle_inequality": bool(np.all(matrix[:, None, :]<=matrix[:, :, None]+matrix[None, :, :]+1e-12)),
        "minima_shape": minima.shape == (expected_n, 3),
        "minima_finite": bool(np.all(np.isfinite(minima))),
        "endpoint_minima": bool(np.all(minima[[0, -1]] == 0) and np.all(minima[1:-1]>0)),
        "graph_invariants_recorded": bool(h["validation"]["connected"] and h["validation"]["hamiltonian_cycle"]
            and h["validation"]["coordinate_diagonal_counts"] == [expected_n-3]*3)}
    ties = [expected_n-2-len(np.unique(minima[1:-1, k])) for k in range(3)]
    diagnostics = {"source": str(path), "id": h["id"], "seed": h["seed"], "n": expected_n,
        "root_count": len(points), "all_checks_pass": all(checks.values()),
        "failed_checks": [k for k, v in checks.items() if not v],
        "maximum_mean_tail_identity_error": float(np.max(abs(np.sum(1-cdf, axis=1)-means))),
        "ties_by_coordinate": ties, "maximum_tie_fraction": max(ties)/(expected_n-2),
        "max_wos_clock_interval_over_cell": h["sampler_metadata"].get("max_normalized_dt", 0)*expected_n
            if "max_normalized_dt" in h["sampler_metadata"] else None,
        "exact_grid_observation_interval_over_cell": expected_n/(max(h["configuration"]["sizes"])*h["sampler_metadata"]["oversample"])
            if h["sampler_metadata"].get("grid_values_exact") else None,
        "grid_interval_scope": "Observation-grid diagnostic only; continuous exact minima remain exact when supplied"}
    return h, minima, points, diagnostics


def preservation(campaign):
    gate_root = ROOT/"calibration/sampler"
    gate = json.loads((gate_root/"final_manifest.json").read_text())
    gate_mismatches = [str(gate_root/p) for p, digest in gate["artifact_sha256"].items()
        if not (gate_root/p).exists() or sha(gate_root/p)!=digest]
    old_sources = {p: h for p, h in campaign["provenance"]["source_sha256"].items() if "3c" not in p.lower()}
    source_mismatches = [p for p, h in old_sources.items() if not Path(p).exists() or sha(p)!=h]
    reused_mismatches = [p["path"] for p in campaign["reused_input_hashes"] if sha(p["path"])!=p["sha256"]]
    return {"closed_StageI_artifacts_checked": len(gate["artifact_sha256"]), "closed_StageI_mismatches": gate_mismatches,
        "old_source_files_checked": len(old_sources), "old_source_mismatches": source_mismatches,
        "historical_n2048_inputs_checked": len(campaign["reused_input_hashes"]), "historical_input_mismatches": reused_mismatches,
        "producer_hash_matches": sha("experiments/phase3b_graphs.py")==campaign["producer_sha256"],
        "design_hash_matches": sha(ROOT/"covariance_design/initial_design.json")==campaign["initial_design_sha256"]}


def perform_audit(out, degree_parents=4):
    campaign, specs = campaign_inputs()
    ready = readiness(campaign)
    if not ready["ready_for_final_audit"]:
        raise RuntimeError(f"Final audit requires every BATCH_COMPLETE marker: {ready}")
    if (out/"summary.json").exists():
        raise FileExistsError("Completed audit exists; choose a new output version")
    started = time.perf_counter()
    directories = [(Path(d), "reused_"+Path(d).name) for d in campaign["reused_directories"]]
    directories += [(Path(b["output"]), b["label"]) for b in campaign["batches"]]
    parents = {}
    size_rows = []
    input_hashes = {}
    duplicate_errors = []
    failures = []
    check_failures = []
    for directory, stage in directories:
        for path in sorted(directory.glob("C*/graph_*_n*.npz")):
            n = int(path.stem.rsplit("_n", 1)[1])
            if n not in (512, 2048):
                continue
            h, minima, points, diagnostics = check_arrays(path, n)
            identifier = h["id"]
            if identifier not in specs:
                raise ValueError(f"Undeclared covariance {identifier}")
            actual = h["sampler_metadata"]["covariance"]
            if not equivalent_covariance(actual, specs[identifier]["covariance"]):
                diagnostics["all_checks_pass"] = False
                diagnostics["failed_checks"].append("actual_covariance_not_permutation_of_design")
            key = identifier, h["seed"]
            if key not in parents:
                parents[key] = {"diagnostics": path_diagnostics(h, stage), "sizes": {}}
            if n in parents[key]["sizes"]:
                duplicate_errors.append({"id": identifier, "seed": h["seed"], "n": n, "duplicate": str(path)})
            parents[key]["sizes"][n] = {"minima": minima, "points": points, "source": str(path), "header": h}
            size_rows.append(diagnostics)
            input_hashes[str(path)] = sha(path)
            if not diagnostics["all_checks_pass"]:
                check_failures.append(diagnostics)
        for path in sorted(list(directory.glob("C*/model_failure.json"))+list(directory.glob("C*/failure_*.json"))):
            raw = json.loads(path.read_text())
            failures.append({"id": path.parent.name, "stage": stage, "source": str(path),
                "type": "model_failure" if path.name=="model_failure.json" else "path_failure",
                "error": raw["error"], "seed": raw.get("seed"), "realization": raw.get("realization")})
            input_hashes[str(path)] = sha(path)
    # A single model-construction failure accounts for its planned parent set;
    # path failures and partially written paired sizes remain individually visible.
    paired_errors = []
    path_rows = []
    for key, parent in sorted(parents.items()):
        d = parent["diagnostics"]
        sizes = parent["sizes"]
        complete = set(sizes)=={512, 2048}
        d.update(observed_sizes=sorted(sizes), complete_paired_sizes=complete,
            source_files=[r["source"] for r in sizes.values()])
        if complete:
            small, large = sizes[512], sizes[2048]
            d["nested_minima_identity"] = bool(np.array_equal(small["minima"], large["minima"].reshape(512, 4, 3).min(axis=1)))
            d["paired_measure_points"] = bool(np.array_equal(small["points"], large["points"]//4))
            if not d["nested_minima_identity"] or not d["paired_measure_points"]:
                paired_errors.append({"id": key[0], "seed": key[1], "source_files": d["source_files"]})
        else:
            paired_errors.append({"id": key[0], "seed": key[1], "observed_sizes": sorted(sizes)})
        path_rows.append(d)
    # Check recorded per-batch hashes and each planned attempt, beyond merely
    # trusting a producer exit code or aggregate size-record count.
    attempt_accounting = []
    for batch in campaign["batches"]:
        folder = Path(batch["output"])
        marker = json.loads((folder/"BATCH_COMPLETE.json").read_text())
        hash_mismatch = [p for p, h in marker["graph_n2048_hashes"].items() if input_hashes.get(p)!=h]
        missing = []
        for identifier in batch["ids"]:
            model_failure = (folder/identifier/"model_failure.json").exists()
            for i in range(batch["parents_per_config"]):
                pair = all((folder/identifier/f"graph_{i:03d}_n{n}.npz").exists() for n in [512, 2048])
                failure = (folder/identifier/f"failure_{i:03d}.json").exists()
                if not (pair or failure or model_failure):
                    missing.append({"id": identifier, "realization": i})
        attempt_accounting.append({"batch": batch["label"], "all_attempts_accounted": not missing,
            "unaccounted_attempts": missing, "recorded_n2048_hash_mismatches": hash_mismatch,
            "marker_records": marker["saved_size_records"]})
    registry = []
    for identifier, spec in sorted(specs.items()):
        ps = [r for r in path_rows if r["id"]==identifier]
        complete = [r for r in ps if r["complete_paired_sizes"]]
        fs = [f for f in failures if f["id"]==identifier]
        signed = max((r["max_abs_signed_leg_log_defect"] for r in ps), default=0.)
        absolute = max((r["max_absolute_leg_log_defect_sum"] for r in ps), default=0.)
        unsafe = identifier in {"C019", "C023"} or signed>SIGNED_WARNING or absolute>ABSOLUTE_WARNING
        if not ps:
            status = "numerical_model_failure_for_SPD_covariance" if fs else "missing_unaccounted_data"
        elif len(complete)!=campaign["target_parents"]:
            status = "incomplete_paired_parent_count"
        elif unsafe:
            status = "unsafe_harmonic_excluded_from_family_claims"
        elif all(r["continuous_minima_exact"] for r in ps):
            status = "exact_continuous_minimum_reference"
        elif all(r["grid_values_exact"] for r in ps):
            status = "exact_grid_reference_polygonal_minima"
        else:
            status = "provisional_exploratory_graph_law"
        stage_counts = {stage: sum(r["source_stage"]==stage for r in complete) for stage in sorted({r["source_stage"] for r in ps})}
        eig = np.linalg.eigvalsh(spec["covariance"])
        local = [r for r in size_rows if r["id"]==identifier]
        registry.append({"id": identifier, "spec": spec, "covariance_spd": bool(eig.min()>0),
            "minimum_eigenvalue": float(eig.min()), "status": status,
            "complete_paired_parents": len(complete), "all_observed_parents": len(ps),
            "parents_by_source_stage": stage_counts, "parents_with_harmonic_warning": sum(r["harmonic_warning"] for r in ps),
            "maximum_abs_signed_leg_sum": signed, "maximum_absolute_leg_sum": absolute,
            "maximum_quadrature_difference": max((r["max_exit_quadrature_log_difference"] for r in ps if r["max_exit_quadrature_log_difference"] is not None), default=None),
            "mean_initial_ess_fraction": float(np.mean([r["initial_ess_fraction"] for r in ps if r["initial_ess_fraction"] is not None])) if any(r["initial_ess_fraction"] is not None for r in ps) else None,
            "alpha1": ps[0]["alpha1"] if ps else None,
            "lifetime_variance_finite_for_intended_model": ps[0]["target_lifetime_variance_finite"] if ps else None,
            "maximum_n2048_tie_fraction": max((r["maximum_tie_fraction"] for r in local if r["n"]==2048), default=None),
            "maximum_n2048_wos_clock_ratio": max((r["max_wos_clock_interval_over_cell"] for r in local if r["n"]==2048 and r["max_wos_clock_interval_over_cell"] is not None), default=None),
            "sampler_allowed_for_provisional_scoring": len(complete)==12 and not unsafe,
            "failures": fs, "all_saved_graph_checks_pass": all(r["all_checks_pass"] for r in local),
            "interpretation": "No path discarded; small observed h defects do not validate the Brownian path/minimum law. Exact special references retain their own grid limits."})
    degrees = []
    plan = json.loads((out/"plan.json").read_text())
    if degree_parents and plan["degree_reconstruction_benchmark"]["decision"].startswith("include"):
        start_degree = time.perf_counter()
        for identifier in specs:
            chosen = sorted((r for r in path_rows if r["id"]==identifier and r["complete_paired_sizes"]), key=lambda r:r["seed"])[:degree_parents]
            for r in chosen:
                saved = parents[(identifier, r["seed"])]["sizes"][2048]
                graph = graph_from_minima(saved["minima"])
                degree = np.array([len(v) for v in graph.adjacency])
                degrees.append({"id": identifier, "seed": r["seed"], "n": 2048,
                    "mean_degree": float(degree.mean()), "degree_variance": float(degree.var()),
                    "maximum_degree": int(degree.max()), "degree_histogram": np.bincount(degree).tolist(),
                    "simple_edge_count_matches_header": len(graph.edges)==saved["header"]["validation"]["simple_edge_count"],
                    "scope": "First four smallest complete parent seeds per C; secondary four-parent summaries, not all12"})
        degree_seconds = time.perf_counter()-start_degree
    else:
        degree_seconds = 0.
    preservation_result = preservation(campaign)
    save_json(out/"configuration_registry.json", registry)
    save_csv(out/"configuration_registry.csv", [{k:v for k,v in r.items() if k!="spec"} | {"triple": r["spec"]["triple"], "family": r["spec"]["family"]} for r in registry])
    save_json(out/"physical_parent_registry.json", path_rows)
    save_csv(out/"physical_parent_registry.csv", path_rows)
    save_csv(out/"graph_size_checks.csv", size_rows)
    save_json(out/"failure_registry.json", failures)
    save_json(out/"batch_attempt_accounting.json", attempt_accounting)
    save_json(out/"degree_histograms.json", degrees)
    save_json(out/"preservation.json", preservation_result)
    statuses = {status: sum(r["status"]==status for r in registry) for status in sorted({r["status"] for r in registry})}
    summary = {"configurations_attempted": len(specs), "configuration_status_counts": statuses,
        "stage_qualified_distinct_parents": len(path_rows), "complete_paired_parents": sum(r["complete_paired_sizes"] for r in path_rows),
        "saved_size_records": len(size_rows), "parents_by_stage": {stage: sum(r["source_stage"]==stage for r in path_rows) for stage in sorted({r["source_stage"] for r in path_rows})},
        "sampler_excluded_ids": [r["id"] for r in registry if not r["sampler_allowed_for_provisional_scoring"]],
        "numerical_model_failure_ids": [r["id"] for r in registry if r["status"]=="numerical_model_failure_for_SPD_covariance"],
        "all_batch_attempts_accounted": all(r["all_attempts_accounted"] for r in attempt_accounting),
        "all_batch_hashes_match": all(not r["recorded_n2048_hash_mismatches"] for r in attempt_accounting),
        "graph_check_failures": check_failures, "paired_record_errors": paired_errors,
        "duplicate_parent_size_errors": duplicate_errors, "degree_records": len(degrees),
        "degree_reconstruction_seconds": degree_seconds, "total_seconds": time.perf_counter()-started,
        "warnings": ["Unsafe parents retained; configurations masked rather than conditioned by rejection",
            "Every other generic law remains provisional, including few-percent and ordinal-minimum limitations from StageI",
            "No new simulations; full ZIP sweep deferred to final campaign integrity check"]}
    save_json(out/"summary.json", summary)
    save_json(out/"manifest.json", {"provenance": environment_manifest(), "input_sha256": input_hashes,
        "script_sha256": sha(__file__), "campaign_sha256": sha(ROOT/"screen/campaign.json")})
    (out/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["plan", "ready", "audit"])
    parser.add_argument("--output", type=Path, default=ROOT/"screen/audit")
    parser.add_argument("--degree-parents", type=int, default=4)
    args = parser.parse_args()
    campaign, specs = campaign_inputs()
    if args.mode=="ready":
        print(json.dumps(readiness(campaign), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output/"plan.json").exists():
        benchmark = degree_benchmark()
        save_json(args.output/"plan.json", {"scientific_question": "Complete coverage, reliability and preservation of109 attempted graph covariances",
            "new_simulations": False, "required_batch_markers": len(campaign["batches"]), "configurations": len(specs),
            "estimated_seconds": [8, 25], "working_memory_mb_estimate": 200,
            "degree_reconstruction_benchmark": benchmark,
            "secondary_degree_parent_selection": args.degree_parents,
            "scope": "Read saved arrays; pair n512/2048; retain all failures and unsafe paths; no whole-dataset ZIP sweep",
            "provenance": environment_manifest()})
    if args.mode=="audit":
        perform_audit(args.output, args.degree_parents)


if __name__=="__main__":
    main()
