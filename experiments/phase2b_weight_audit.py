"""Compare fixed versus resampled variance weights using saved bootstrap design."""
import json
from pathlib import Path
import numpy as np
from src.phase2b_scaling import fit_mean_model


def main():
    directory=Path("results/phase2b")
    path=directory/"bootstrap_weight_audit.json"
    if path.exists():raise FileExistsError(path)
    result=json.loads((directory/"updated_independent_scaling.json").read_text())
    cache=np.load(directory/"updated_independent_scaling.npz")
    records=[]
    for source in result["sources"]:
        records.extend(json.loads(line) for line in Path(source["file"]).read_text().splitlines())
    sizes=np.array(result["sizes"])
    groups=[np.array([r["mean_distance"] for r in records if r["n"]==n]) for n in sizes]
    rng=np.random.default_rng(result["seed"])
    fixed=[]; reweighted=[]
    for i in range(result["bootstrap"]):
        samples=[group[rng.integers(len(group),size=len(group))] for group in groups]
        means=np.array([group.mean() for group in samples])
        np.testing.assert_allclose(means,cache["bootstrap_means"][i],rtol=1e-13,atol=1e-13)
        errors=np.array([group.std(ddof=1)/np.sqrt(len(group)) for group in samples])
        fixed.append(fit_mean_model(sizes,means,cache["se"],"offset")["dimension"])
        reweighted.append(fit_mean_model(sizes,means,errors,"offset")["dimension"])
    output=dict(parent="updated_independent_scaling.json",bootstrap=result["bootstrap"],
        seed=result["seed"],fixed_weight_mc95=np.quantile(fixed,[.025,.975]).tolist(),
        reestimated_weight_mc95=np.quantile(reweighted,[.025,.975]).tolist(),
        fixed_weight_dimensions=fixed,reestimated_weight_dimensions=reweighted,
        interpretation="Same graph clusters/draws; addresses estimated-variance-weight sensitivity, not model error")
    path.write_text(json.dumps(output,indent=2))
    print({k:v for k,v in output.items() if not k.endswith("dimensions")})


if __name__=="__main__":main()
