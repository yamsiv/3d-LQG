"""Bootstrap shifted histogram-collapse estimates for saved independent graphs."""
import argparse
import json
from pathlib import Path
from src.scaling_analysis import bootstrap_collapse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=44279)
    args = parser.parse_args()
    for i, path in enumerate(map(Path, args.inputs)):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        analysis = bootstrap_collapse(records, bootstrap=args.bootstrap, seed=args.seed + i)
        target = path.with_suffix(".collapse.json")
        target.write_text(json.dumps(analysis, indent=2) + "\n")
        print(json.dumps({"input": str(path), "effective_dimension": analysis["central"]["effective_dimension"],
                          "MC_ci95": analysis["dimension_ci95"],
                          "windows": [(w["minimum_n"],w["effective_dimension"]) for w in analysis["window_sensitivity"]]}), flush=True)


if __name__ == "__main__":
    main()
