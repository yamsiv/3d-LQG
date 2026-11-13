"""Paired holdout improvement over flat controls, without new simulations."""
from pathlib import Path
import json
import numpy as np
from experiments.phase3a_compare import field_groups,graph_groups,save_json
from src.metric_measure_3a import summarize,discrepancy
from src.provenance import environment_manifest


def main():
    root=Path("results/phase3a/paired_improvement");root.mkdir(exist_ok=True)
    if (root/"summary.json").exists():raise FileExistsError(root)
    choices=json.loads(Path("results/phase3a/selection/selection.json").read_text())["choices"]
    graphs=graph_groups("results/phase3a/graphs",split="confirmation")
    fields=field_groups("results/phase3a/confirmation",n=48,stencil="primitive2")
    flat=fields[(48,"primitive2",0.,0.)];results={}
    for case in ["i3","a2","a3"]:
        rng=np.random.default_rng([202611113922,["i3","a2","a3"].index(case)])
        g=graphs[(case,8192)];choice=choices[case]
        selected=fields[(48,"primitive2",choice["xi"],choice["gamma"])]
        score=lambda a,b,c:discrepancy(summarize(a),summarize(b))["objective"]-discrepancy(summarize(a),summarize(c))["objective"]
        draws=[]
        for _ in range(160):
            gi=rng.integers(len(g),size=len(g));fi=rng.integers(len(flat),size=len(flat))
            draws.append(score([g[i] for i in gi],[flat[i] for i in fi],[selected[i]for i in fi]))
        np.savez_compressed(root/f"{case}_bootstrap.npz",flat_minus_selected=np.array(draws))
        results[case]={"flat_minus_selected":score(g,flat,selected),"mc95":np.quantile(draws,[.025,.975]).tolist(),
            "positive_favors_selected":True,"draws":160,"physical_fields":len(flat),"physical_graphs":len(g)}
    save_json(root/"summary.json",{"provenance":environment_manifest(),"results":results,
        "design":"Same graph bootstrap and paired field identities for selected/flat, separate deterministic normalizers recomputed",
        "qualification":"Frozen selected holdout comparison to a flat torus; not a test identifying logarithmic correlations, no multiplicity correction"})
    print(json.dumps(results))


if __name__=="__main__":main()
