"""Independent-seed grid/burn-in sensitivity at identity and near degeneracy.

Uses only small graphs to diagnose the general-covariance approximation before
permitting asymptotic parameter studies. Every configuration has its own files.
"""
from pathlib import Path
import json
import numpy as np

from experiments.parameter_sweep import run_pilot
from experiments.reproduce_budd_castro import run_study


def main():
    covariance = np.full((3, 3), -.49)
    np.fill_diagonal(covariance, 1.)
    points = [("identity", np.eye(3), "control"),
              ("near_degenerate_rho_-0.49", covariance, "near_boundary")]
    configurations = [(2, 32), (2, 128), (8, 128)]
    all_records = []
    for index, (oversample, sweeps) in enumerate(configurations):
        path = Path(f"results/data/sensitivity_grid_o{oversample}_s{sweeps}.jsonl")
        records = run_pilot(path, sizes=(256, 512), graphs=6, roots=8,
                            oversample=oversample, sweeps=sweeps, seed=382100 + index * 100,
                            points=points)
        for record in records:
            record["configuration"] = f"grid_o{oversample}_s{sweeps}"
        all_records.extend(records)
    path = Path("results/data/sensitivity_exact_unbiased_identity.jsonl")
    records = run_study(sizes=(256, 512), graphs=12, roots=8, covariance=np.eye(3),
                        law="unbiased", seed=382700, output=path, oversample=1)
    for record in records:
        record["configuration"] = "exact_unbiased"
        record["name"] = "identity"
    all_records.extend(records)
    summary = {"gate": "CLOSED", "asymptotic_exponents_fitted": False, "groups": []}
    for config in dict.fromkeys(r["configuration"] for r in all_records):
        for name in dict.fromkeys(r["name"] for r in all_records if r["configuration"] == config):
            for n in (256, 512):
                group = [r for r in all_records if (r["configuration"], r["name"], r["n"]) == (config, name, n)]
                values = np.array([r["mean_distance"] for r in group])
                traces = [np.asarray(r["sampler_metadata"].get("trace_area_midpoint_radius", [])) for r in group]
                summary["groups"].append({"configuration": config, "name": name, "n": n,
                    "graphs": len(group), "mean_distance": float(values.mean()),
                    "mc_se": float(values.std(ddof=1) / np.sqrt(len(group))),
                    "mean_tie_fraction": float(np.mean([r["graph_metadata"]["symbolic_ties_by_coordinate"] for r in group]) / (n-2)),
                    "trace_first_quarter_area": float(np.mean([t[:max(1,len(t)//4),0].mean() for t in traces])) if traces[0].size else None,
                    "trace_last_quarter_area": float(np.mean([t[-max(1,len(t)//4):,0].mean() for t in traces])) if traces[0].size else None})
    Path("results/data/sampler_sensitivity.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, ("identity", "near_degenerate_rho_-0.49")):
        for config in dict.fromkeys(g["configuration"] for g in summary["groups"] if g["name"] == name):
            group = [g for g in summary["groups"] if g["configuration"] == config and g["name"] == name]
            ax.errorbar([g["n"] for g in group], [g["mean_distance"] for g in group],
                         yerr=[1.96*g["mc_se"] for g in group], marker="o", capsize=3, label=config)
        ax.set(xlabel="Vertices n", ylabel="Mean uniform-pair distance", title=name)
        ax.legend(fontsize="small")
    fig.suptitle("Grid and burn-in sensitivity; independent-seed configurations, Monte Carlo errors only")
    fig.tight_layout()
    fig.savefig("results/figures/sampler_sensitivity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
