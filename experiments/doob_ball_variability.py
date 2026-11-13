"""Recover non-mean ball diagnostics from saved minima and original roots."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.mated_crt import graph_from_minima
from src.graph_metrics import sparse_adjacency,bfs_distances


def main():
    paths=[Path("results/phase2b")/name/"records.jsonl" for name in
           ["doob_validation","doob_extra_controls","doob_one_factor"]]
    output=Path("results/phase2b/doob_ball_variability.json")
    rows=[]
    for path in paths:
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row["status"]!="success":continue
            if row["case"] in ("i3","a2") and row["n"]!=256:continue
            graph=sparse_adjacency(graph_from_minima(np.asarray(row["cell_minima"])).adjacency)
            fractions=[]
            for root in row["roots"]:
                distance=bfs_distances(graph,root)
                fractions.append([float(np.mean(distance<=r)) for r in [2,3,4]])
            fractions=np.asarray(fractions)
            rows.append(dict(source=str(path),case=row["case"],configuration=row["configuration"],
                n=row["n"],realization=row["realization"],seed=row["seed"],roots=row["roots"],
                root_ball_fractions_r2_r3_r4=fractions.tolist(),
                within_graph_root_variance=fractions.var(axis=0,ddof=1).tolist()))
    groups=[]
    for case in dict.fromkeys(r["case"] for r in rows):
        for configuration in dict.fromkeys(r["configuration"] for r in rows if r["case"]==case):
            group=[r for r in rows if r["case"]==case and r["configuration"]==configuration]
            values=np.array([r["within_graph_root_variance"] for r in group])
            groups.append(dict(case=case,configuration=configuration,n=group[0]["n"],graphs=len(group),
                mean_within_graph_root_variance=values.mean(axis=0).tolist(),
                graph_cluster_mc_se=(values.std(axis=0,ddof=1)/np.sqrt(len(group))).tolist()))
    output.write_text(json.dumps(dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        method="Regenerate graph from saved cell minima; use original recorded uniform roots; no new path sampling",
        radii=[2,3,4],records=rows,groups=groups,
        limitation="Eight-root variance diagnostics remain noisy and are not independent graph samples"),indent=2))
    print("Saved",output,"records",len(rows))


if __name__=="__main__":main()
