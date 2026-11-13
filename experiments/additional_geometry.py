"""Graph drawings, independent-graph walk diagnostics and small resistances."""
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.brownian_excursions import sample_independent_excursion
from src.mated_crt import graph_from_excursion, validate_graph
from src.graph_metrics import diameter_bounds, metric_sample
from src.secondary_observables import monte_carlo_returns, spectral_dimension_pilot, effective_resistances
from src.provenance import environment_manifest


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    output = Path("results/additional")
    output.mkdir(parents=True, exist_ok=True)
    path = output/"observables.json"
    if path.exists():
        raise FileExistsError(path)
    start = time.perf_counter()
    seed = 2026111020
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    examples = []
    for ax, d in zip(axes, [2, 3]):
        graph_seed = seed+d
        sample = sample_independent_excursion(32, d, np.random.default_rng(graph_seed))
        graph = graph_from_excursion(sample, 32)
        angle = np.linspace(0, 2*np.pi, 32, endpoint=False)
        positions = np.column_stack((np.cos(angle), np.sin(angle)))
        cycle = np.column_stack((np.arange(32), (np.arange(32)+1)%32))
        ax.add_collection(LineCollection(positions[cycle], colors="black", linewidths=1.4))
        for k, edges in enumerate(graph.coordinate_edges):
            ax.add_collection(LineCollection(positions[edges], colors=[f"C{k}"], alpha=.65,
                                            linewidths=.8, label=f"Tree {k+1}"))
        ax.scatter(*positions.T, color="black", s=12, zorder=3)
        ax.set(xlim=(-1.1,1.1), ylim=(-1.1,1.1), aspect="equal", title=f"{d}-tree graph, n=32")
        ax.axis("off")
        ax.legend(fontsize=8, loc="upper right")
        examples.append({"dimension": d, "seed": graph_seed, "edges": graph.edges,
                         "multiplicities": graph.multiplicities,
                         "validation": validate_graph(graph),
                         "diameter": diameter_bounds(graph.adjacency, exact_limit=128)})
    fig.suptitle("Hamiltonian circular layout; vertex positions have no geometric interpretation")
    fig.tight_layout()
    fig.savefig(output/"small_graphs.png", dpi=180)
    plt.close(fig)
    walk_rows = []
    for n in [8192, 32768]:
        for realization in range(8):
            graph_seed = int(np.random.SeedSequence([seed, n, realization]).generate_state(1)[0])
            graph = graph_from_excursion(sample_independent_excursion(n, 3,
                                         np.random.default_rng(graph_seed)), n)
            for multi in [False, True]:
                walk_seed = graph_seed + 100 + int(multi)
                result = monte_carlo_returns(graph.to_sparse(multigraph=multi),
                         np.random.default_rng(walk_seed), max_steps=256, walkers=12000)
                walk_rows.append({"n": n, "realization": realization,
                                  "graph_seed": graph_seed, "walk_seed": walk_seed,
                                  "conductance": "multiplicities" if multi else "simple_unit",
                                  "returns": result,
                                  "slope_8_64": spectral_dimension_pilot(result, 8, 64),
                                  "slope_16_128": spectral_dimension_pilot(result, 16, 128)})
    resistance_rows = []
    for realization in range(8):
        graph_seed = seed+300+realization
        rng = np.random.default_rng(graph_seed)
        graph = graph_from_excursion(sample_independent_excursion(128, 3, rng), 128)
        pairs = rng.integers(128, size=(32, 2))
        resistance_rows.append({"graph_seed": graph_seed, "n": 128, "pairs": pairs,
                               "unit_simple_resistance": effective_resistances(graph.to_sparse(), pairs),
                               "multigraph_resistance": effective_resistances(graph.to_sparse(True), pairs)})
    path.write_text(json.dumps(json_ready({"provenance": environment_manifest(),
        "seed": seed, "small_graphs": examples, "walks": walk_rows,
        "resistances": resistance_rows, "elapsed_seconds": time.perf_counter()-start,
        "interpretation": "finite_graph_diagnostics_only_no_asymptotic_spectral_dimension"}), indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n in [8192, 32768]:
        for multi in [False, True]:
            label = "multiplicities" if multi else "simple_unit"
            rows = [r for r in walk_rows if r["n"] == n and r["conductance"] == label]
            curves = np.array([r["returns"]["return_probability"] for r in rows])
            mean = curves.mean(axis=0)
            se = curves.std(axis=0, ddof=1)/np.sqrt(len(rows))
            t = np.arange(len(mean))
            axes[0].loglog(t[1:], mean[1:], label=f"n={n}, {label}")
            axes[0].fill_between(t[1:], np.maximum(mean[1:]-1.96*se[1:], 1e-6),
                                 mean[1:]+1.96*se[1:], alpha=.1)
            axes[1].plot(t[1:], t[1:]*mean[1:], label=f"n={n}, {label}")
    axes[0].set(xlabel="Lazy-walk steps t", ylabel="Uniform-root return probability",
                title="Eight independent graphs, 12000 walks per graph")
    axes[0].legend(fontsize=7)
    axes[1].set(xscale="log", xlabel="Lazy-walk steps t", ylabel="t × return probability",
                title="A plateau would be consistent with a t^-1 window")
    fig.tight_layout()
    fig.savefig(output/"walk_returns.png", dpi=180)
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    main()
