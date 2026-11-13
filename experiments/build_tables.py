"""Export finite-graph summaries from saved records without resimulating."""
import csv
import json
from pathlib import Path
import numpy as np
from src.scaling_analysis import pooled_histogram, histogram_statistic


def main():
    directory = Path("results/tables")
    directory.mkdir(parents=True,exist_ok=True)
    rows = []
    names = ["i3_biased","i3_biased_high","i3_biased_combined","i2_biased","i3_unbiased"]
    for name in names:
        path = Path("results/data")/(name+".jsonl")
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for n in sorted({r["n"] for r in records}):
            group = [r for r in records if r["n"]==n]
            histogram = pooled_histogram(group)
            mean = np.array([r["mean_distance"] for r in group])
            ball = [r["ball_exponent_diagnostic"]["slope"] for r in group
                    if r["ball_exponent_diagnostic"] is not None]
            rows.append(dict(study=name,n=n,graphs=len(group),roots_per_graph=len(group[0]["roots"]),
                mean_distance=mean.mean(),mean_distance_mc_se=mean.std(ddof=1)/np.sqrt(len(group)),
                median_distance=histogram_statistic(histogram,"median"),
                q25_distance=histogram_statistic(histogram,"q25"),
                q75_distance=histogram_statistic(histogram,"q75"),
                mean_simple_degree=np.mean([r["degree"]["mean"] for r in group]),
                mean_diameter_lower=np.mean([r["diameter"]["lower"] for r in group]),
                mean_diameter_upper=np.mean([r["diameter"]["upper"] for r in group]),
                mean_apparent_ball_slope=np.mean(ball) if ball else None))
    with (directory/"metric_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pilot = json.loads(Path("results/data/covariance_pilot.summary.json").read_text())
    with (directory/"covariance_pilot.csv").open("w") as stream:
        fields = list(pilot["groups"][0])
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(pilot["groups"])
    print("Saved",directory)


if __name__ == "__main__":
    main()
