"""Reuse Phase 3A checkpoints to calibrate the richer Phase 3B diagnostics.

No graph or field is resimulated. Target cost: under 2 minutes, <150 MB, from
measured feature-vector costs .025 s/24 graphs and .057 s/24 fields. Physical
realizations, not roots or triangles, are the bootstrap and split units.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.phase3a_compare import field_groups, graph_groups, jsonable
from src.metric_features_3b import (
    feature_vector, grouped_feature_distance, split_noise_scales,
    bootstrap_feature_difference, matrix_shape_samples, DEFAULT_WEIGHTS,
)


BASE = Path("results/phase3a")
SEED = 202611113801


def save_json(path, data):
    with Path(path).open("x") as f:
        json.dump(jsonable(data), f, indent=2)
        f.write("\n")


def field_pair_records(records, pair):
    return [dict(record, xi=pair[0], gamma=pair[1]) for record in records]


def _weight_variants():
    base = dict(DEFAULT_WEIGHTS)
    variants = {"default": base, "equal_families": {key: 1. for key in base},
                "mesoscopic_and_higher_point": dict(base, balls=3., quartets=3., mesoscopic=2.),
                "without_global_scale": dict(base, global_scale=0.),
                "without_slope_and_atoms": dict(base, mesoscopic=0., atoms=0.)}
    for family in base:
        variants[f"double_{family}"] = dict(base, **{family: 2*base[family]})
        variants[f"half_{family}"] = dict(base, **{family: .5*base[family]})
    return variants


def write_power_if_missing(out):
    """Independent-realization scalar planning calculation, not post hoc power.

    Per-space triangle/quartet means are independent between physical spaces,
    while all matrix subsets within one space remain a single scalar row.
    """
    path = out/"power_summary.json"
    if path.exists():
        return
    graphs = graph_groups(BASE/"graphs", split="confirmation")
    fields = field_groups(BASE/"confirmation", n=48, stencil="primitive2")
    pairs = {"i3": (.3,1.5), "a2": (.2,0.), "a3": (.2,0.)}
    rows = []
    for case, pair in pairs.items():
        for label, selected in (("frozen_3a", pair),("flat", (0.,0.))):
            g = [matrix_shape_samples(r["matrix"]) for r in graphs[(case,8192)]]
            f = [matrix_shape_samples(r["matrix"]) for r in fields[(48,"primitive2",*selected)]]
            for key, coordinate, feature in (("triangles",0,"triangle_small_mean"),
                                             ("triangles",1,"triangle_middle_mean"),
                                             ("quartets",0,"quartet_delta_mean"),
                                             ("quartets",1,"quartet_aspect_mean")):
                a = np.array([r[key][:,coordinate].mean() for r in g if len(r[key])])
                b = np.array([r[key][:,coordinate].mean() for r in f if len(r[key])])
                se = float(np.sqrt(np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b)))
                rows.append(dict(case=case, label=label, xi=selected[0],gamma=selected[1],feature=feature,
                                 graph_realizations=len(a),field_realizations=len(b),
                                 graph_mean=float(a.mean()),field_mean=float(b.mean()),
                                 signed_difference=float(a.mean()-b.mean()),independent_realization_SE=se,
                                 gaussian_80pct_power_MDE=2.80*se))
    save_json(path, {"rows":rows,
                    "method":"SE=sqrt(s_graph^2/n_graph+s_field^2/n_field), one shape-mean row per physical realization; MDE=(1.96+.84)*SE",
                    "qualification":"Unadjusted Gaussian scalar planning approximation, not observed power or a corrected hypothesis test; ignores systematic errors and parameter selection",
                    "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     (Path(__file__),Path('src/metric_features_3b.py'))}})
    with (out/"power_summary.csv").open("x") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase3b/statistics")
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--split-draws", type=int, default=24)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"summary.json").exists():
        write_power_if_missing(out)
        print(f"Completed statistics audit already exists: {out}/summary.json")
        return
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    graphs = graph_groups(BASE/"graphs")
    confirmed = graph_groups(BASE/"graphs", split="confirmation")
    field48 = field_groups(BASE/"confirmation", n=48, stencil="primitive2")
    field32_old = field_groups(BASE/"discovery", n=32, stencil="26")
    field32_new = field_groups(BASE/"confirmation", n=32, stencil="26")
    baseline_pair = (.3, 1.5)
    baseline_fields = field_pair_records(field32_old[(32, "26", *baseline_pair)], baseline_pair)
    baseline_fields += field_pair_records(field32_new[(32, "26", *baseline_pair)], baseline_pair)
    layout = feature_vector(graphs[("i3", 8192)])
    config = {"seed": SEED, "bootstrap_draws": args.bootstrap, "split_draws": args.split_draws,
              "uses_only_existing_phase3a_checkpoints": True,
              "noise_reference_graph": "48 I3 n8192 graphs, disjoint groups of 24",
              "noise_reference_field": "48 independent N32,26-direction fields at xi=.3,gamma=1.5; disjoint groups of 24",
              "fixed_noise_scaling": "sqrt((graph_split_RMS^2+field_split_RMS^2)/2), floor .05*natural_scale",
              "weights": _weight_variants(),
              "time_estimate_seconds": 120, "memory_estimate_mb": 150,
              "normalization": "ensemble median per compared finite ensemble; recomputed inside every split/bootstrap"}
    if not (out/"config.json").exists():
        save_json(out/"config.json", config)
    if not (out/"noise_calibration.npz").exists():
        graph_noise = split_noise_scales(graphs[("i3",8192)], rng, draws=args.split_draws, split_size=24)
        field_noise = split_noise_scales(baseline_fields, rng, draws=args.split_draws, split_size=24)
        noise = np.sqrt((graph_noise["scales"]**2+field_noise["scales"]**2)/2)
        np.savez_compressed(out/"noise_calibration.npz", noise_scales=noise,
                            graph_rms=graph_noise["raw_rms"], field_rms=field_noise["raw_rms"],
                            graph_differences=graph_noise["differences"],
                            field_differences=field_noise["differences"],
                            graph_valid=graph_noise["valid"], field_valid=field_noise["valid"],
                            natural_floor=graph_noise["floor"])
    with np.load(out/"noise_calibration.npz") as a:
        noise = a["noise_scales"]
        split_deltas = {label: (a[label+"_differences"], a[label+"_valid"])
                        for label in ("graph", "field")}
    if not (out/"feature_layout.json").exists():
        save_json(out/"feature_layout.json", {key: layout[key] for key in ("names", "families", "scales")})

    # Both volume gauges are declared before ranking, rather than selecting one
    # after viewing its scores. gamma0,xi>0 has no unit-mass Weyl gauge.
    fields = {}
    for (_, _, xi, gamma), records in sorted(field48.items()):
        records = field_pair_records(records, (xi, gamma))
        for gauge in ("grounded", "unit_mass"):
            if gamma == 0 and xi > 0 and gauge == "unit_mass":
                continue
            fields[(xi, gamma, gauge)] = feature_vector(records, gauge=gauge)
    graph_features = {case: feature_vector(confirmed[(case,8192)]) for case in ("i3", "a2", "a3")}
    rows = []
    for case, target in graph_features.items():
        for (xi, gamma, gauge), features in fields.items():
            natural = grouped_feature_distance(target, features)
            calibrated = grouped_feature_distance(target, features, noise_scales=noise)
            variants = {name: grouped_feature_distance(target, features, noise_scales=noise, weights=weights)["total"]
                        for name, weights in _weight_variants().items()}
            rows.append(dict(case=case, xi=xi, gamma=gamma, gauge=gauge,
                             natural_score=natural["total"], calibrated_score=calibrated["total"],
                             components=calibrated["components"], weight_variants=variants,
                             graph_realizations=len(confirmed[(case,8192)]), field_realizations=24))
    if not (out/"feature_comparisons.json").exists():
        save_json(out/"feature_comparisons.json", {"rows": rows, "graph_features": graph_features,
                  "field_features": {str(key): value for key,value in fields.items()}})
    if not (out/"feature_comparisons.csv").exists():
        with (out/"feature_comparisons.csv").open("x") as f:
            columns = ["case", "xi", "gamma", "gauge", "natural_score", "calibrated_score"]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: row[k] for k in columns} for row in rows)
    # Conditional same-model re-splits use the same family weights/scales as
    # graph-field comparisons; these are not p-values or extra independent data.
    floors = {}
    for label, (deltas, masks) in split_deltas.items():
        scores = []
        for delta, mask in zip(deltas, masks):
            a = dict(layout, vector=delta, valid=mask)
            b = dict(layout, vector=np.zeros_like(delta), valid=mask)
            scores.append(grouped_feature_distance(a, b, noise_scales=noise)["total"])
        floors[label] = {"scores": scores, "quantiles_10_50_90": np.quantile(scores,[.1,.5,.9]),
                         "realizations_per_split": 24,
                         "warning": "Repeated dependent splits of a finite dataset; descriptive noise floor only"}

    frozen_pairs = {"i3": (.3,1.5), "a2": (.2,0.), "a3": (.2,0.)}
    signed = {}
    for case, pair in frozen_pairs.items():
        path = out/f"signed_{case}_frozen_3a.json"
        if path.exists():
            signed[case] = json.loads(path.read_text())
            continue
        records = field_pair_records(field48[(48,"primitive2",*pair)], pair)
        fit = bootstrap_feature_difference(confirmed[(case,8192)], records,
                                          np.random.default_rng(SEED+100+["i3","a2","a3"].index(case)),
                                          draws=args.bootstrap)
        save_json(path, fit)
        signed[case] = fit
    # Keep refinement shifts alongside MC uncertainty; none is absorbed into a
    # nominally tighter bootstrap interval.
    refinement = {}
    for case, pair in frozen_pairs.items():
        small = feature_vector(confirmed[(case,2048)])
        large = graph_features[case]
        old_field = feature_vector(field_pair_records(field32_new[(32,"26",*pair)],pair))
        new_field = fields[(*pair,"grounded")]
        refinement[case] = {"graph_2048_to_8192": grouped_feature_distance(small,large,noise_scales=noise),
                            "field_N32_26_to_N48_98": grouped_feature_distance(old_field,new_field,noise_scales=noise),
                            "note": "Field refinement jointly changes cutoff and directions; paired physical seeds, not independent data"}
    winners = {}
    for case in graph_features:
        for gauge in ("grounded","unit_mass"):
            eligible = [row for row in rows if row["case"]==case and row["gauge"]==gauge]
            wins = {}
            for variant in _weight_variants():
                best = min(eligible,key=lambda row:row["weight_variants"][variant])
                label = f"xi={best['xi']:g},gamma={best['gamma']:g}"
                wins[label] = wins.get(label,0)+1
            winners[f"{case}_{gauge}"] = wins
    summary = {"status": "completed cached-data audit; no new graphs or fields generated",
               "seed": SEED, "noise_floors": floors, "refinement": refinement,
               "weight_sensitive_winner_counts": winners,
               "comparisons": len(rows), "weight_scenarios": len(_weight_variants()),
               "warning": "These five Phase3A confirmation pairs are diagnostics, mostly not theoretically calibrated; not a new Phase3B selection/confirmation",
               "elapsed_seconds": time.perf_counter()-start,
               "signed_feature_MC_intervals": [f"signed_{case}_frozen_3a.json" for case in frozen_pairs],
               "decision_scope": "Same finite summaries can fit several parameters; no unique optimum, scalar map or continuum identification follows"}
    _plots(out, rows, floors)
    paths = [Path(__file__), Path("src/metric_features_3b.py"), Path("src/metric_measure_3a.py")]
    paths += list((BASE/"graphs").glob("graph_*.npz"))
    paths += list((BASE/"confirmation").glob("metric_*.npz"))
    paths += list((BASE/"discovery").glob("metric_*_n32_26.npz"))
    if not (out/"input_manifest.json").exists():
        save_json(out/"input_manifest.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save_json(out/"summary.json", summary)
    write_power_if_missing(out)
    print(json.dumps(jsonable(summary)), flush=True)


def _plots(out, rows, floors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax, case in zip(axes,("i3","a2","a3")):
        for gauge, marker in (("grounded","o"),("unit_mass","s")):
            data = [row for row in rows if row['case']==case and row['gauge']==gauge]
            labels = [f"{r['xi']:g}/{r['gamma']:g}" for r in data]
            ax.plot(range(len(data)),[r['calibrated_score'] for r in data],marker=marker,label=gauge)
            ax.set_xticks(range(len(data)),labels,rotation=35)
        for label, color in (("graph","black"),("field","gray")):
            lo,mid,hi = floors[label]["quantiles_10_50_90"]
            ax.axhspan(lo,hi,color=color,alpha=.08)
            ax.axhline(mid,color=color,lw=.8,ls='--',label=f"{label} self-splits")
        ax.set(title=f"{case.upper()}: cached Phase 3A data",xlabel="xi/gamma (not all theory matched)",
               ylabel="Noise-scaled rich-feature discrepancy")
        ax.legend(fontsize=7)
    fig.savefig(out/"rich_feature_audit.png",dpi=180)
    fig.savefig(out/"rich_feature_audit.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
