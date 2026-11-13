"""Fresh-process timing and peak RSS of exact independent graph generation."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from src.brownian_excursions import sample_independent_excursion
from src.mated_crt import graph_from_excursion, validate_graph
from src.graph_metrics import metric_sample
from src.provenance import environment_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    if args.child:
        n = args.child
        seed = 2026111040+n
        start = time.perf_counter()
        sample = sample_independent_excursion(n, 3, np.random.default_rng(seed))
        sampled = time.perf_counter()
        graph = graph_from_excursion(sample, n)
        generated = time.perf_counter()
        metrics = metric_sample(graph.adjacency, np.random.default_rng(seed+1), roots=4)
        measured = time.perf_counter()
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps(dict(n=n,seed=seed,sampling_seconds=sampled-start,
                              graph_seconds=generated-sampled,metric_seconds=measured-generated,
                              peak_rss_bytes=maxrss if sys.platform=="darwin" else maxrss*1024,
                              validation=validate_graph(graph),
                              simple_edges=len(graph.edges),
                              symbolic_ties=graph.metadata["symbolic_ties_by_coordinate"])))
        return
    rows = []
    for n in [1024,16384,262144,524288]:
        output = subprocess.check_output([sys.executable,"-m","experiments.benchmark","--child",str(n)], text=True)
        rows.append(json.loads(output))
        print(output.strip(),flush=True)
    path = Path("results/validation/runtime_memory.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"rows":rows,"provenance":environment_manifest(),
                               "memory_method":"fresh_process_peak_RSS_including_python_and_libraries"},indent=2))


if __name__ == "__main__":
    main()
