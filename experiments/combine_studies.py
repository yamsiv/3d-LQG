"""Combine independent saved runs while preserving their provenance and seeds."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.reproduce_budd_castro import (analyze_records, make_figures,
                                              source_metadata, export_summary)
from src.finite_size_corrections import shifted_mean_fit
from src.scaling_analysis import bootstrap_collapse


def combine(inputs, output, *, expected=None, bootstrap=500,
            collapse_bootstrap=100, seed=941700):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if expected is not None and len(expected) != len(inputs):
        raise ValueError("expected counts must match input count")
    lines = []
    sources = []
    all_records = []
    reference_model = None
    for i, filename in enumerate(inputs):
        path = Path(filename)
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            raise ValueError(f"{path}: incomplete final JSONL line")
        records = [json.loads(line) for line in data.splitlines()]
        if expected is not None and len(records) != expected[i]:
            raise ValueError(f"{path}: {len(records)} records, expected {expected[i]}; wait for run completion")
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        model = {key: metadata[key] for key in ("covariance", "law", "oversample", "roots_per_graph")}
        if reference_model is None:
            reference_model = model
        elif model != reference_model:
            raise ValueError("runs differ in covariance, excursion law, grid or root count")
        if any(r["sampler_metadata"].get("mcmc") or not r["sampler_metadata"].get("cell_minima_exact") for r in records):
            raise ValueError("this aggregation is restricted to validated exact-minima samples")
        sources.append({"file": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                        "records": len(records), "original_metadata": metadata})
        lines.append(data)
        all_records.extend(records)
    keys = [(r["n"], r["graph_seed"]) for r in all_records]
    if len(keys) != len(set(keys)):
        raise ValueError("repeated graph seed at the same size; samples are not independent")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"".join(lines))
    metadata = source_metadata()
    metadata.update({"aggregation": "byte-preserving concatenation of independently seeded runs",
                     "sources": sources, "graphs_per_size": dict(sorted(Counter(r["n"] for r in all_records).items())),
                     "records": len(all_records), **reference_model})
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    analysis = analyze_records(all_records, output.with_suffix(".analysis.json"), bootstrap=bootstrap, seed=seed)
    make_figures(all_records, analysis, Path("results/figures") / output.stem)
    export_summary(all_records, output.with_suffix(".summary.csv"))
    collapse = bootstrap_collapse(all_records, bootstrap=collapse_bootstrap, seed=seed + 1)
    output.with_suffix(".collapse.json").write_text(json.dumps(collapse, indent=2) + "\n")
    sizes = sorted({r["n"] for r in all_records})
    corrections = {"input": str(output), "source_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                   "shifted_mean_fits": [shifted_mean_fit(all_records, min_n=n, bootstrap=bootstrap, seed=seed + 100 + i)
                                          for i, n in enumerate(sizes[:-3])]}
    output.with_suffix(".corrections.json").write_text(json.dumps(corrections, indent=2) + "\n")
    print(json.dumps({"records": len(all_records), "graphs_per_size": metadata["graphs_per_size"],
                      "naive_dimension": analysis["distance_fits"]["mean"][0]["effective_dimension"],
                      "collapse_dimension": collapse["central"]["effective_dimension"],
                      "collapse_mc95": collapse["dimension_ci95"],
                      "shifted_mean_dimension": corrections["shifted_mean_fits"][0]["effective_dimension"]}), flush=True)
    return all_records, analysis, collapse, corrections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect", nargs="+", type=int)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--collapse-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=941700)
    args = parser.parse_args()
    combine(args.inputs, args.output, expected=args.expect, bootstrap=args.bootstrap,
            collapse_bootstrap=args.collapse_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
