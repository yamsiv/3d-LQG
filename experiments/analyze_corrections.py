"""Reanalyze existing raw data under additive-shift finite-size sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
from src.finite_size_corrections import shifted_mean_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    for filename in args.inputs:
        path = Path(filename)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        sizes = sorted({r["n"] for r in records})
        fits = [shifted_mean_fit(records, min_n=n, bootstrap=args.bootstrap, seed=2026111010+i)
                for i, n in enumerate(sizes[:-3])]
        result = {"input": filename, "source_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                  "shifted_mean_fits": fits,
                  "comparison": "Compare original analysis naive mean/window and histogram-collapse diagnostics"}
        path.with_suffix(".corrections.json").write_text(json.dumps(result, indent=2))
        print(filename, [(f["sizes"][0], round(f["effective_dimension"], 3),
                          np.round(f["dimension_ci95"], 3).tolist()) for f in fits])


if __name__ == "__main__":
    main()
