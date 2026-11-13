"""Audit saved bootstrap draws and finish diagnostics without resimulation."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from src.phase2b_scaling import fit_mean_model,fit_equation45
from src.scaling_analysis import collapse_histograms
from src.provenance import environment_manifest


def main():
    source=Path("results/phase2b/updated_independent_scaling.json")
    output=source.with_name("updated_independent_scaling_audited.json")
    if output.exists():raise FileExistsError(output)
    result=copy.deepcopy(json.loads(source.read_text()))
    cache=np.load(source.with_suffix(".npz"))
    sizes=cache["sizes"];se=cache["se"]
    bootmeans=cache["bootstrap_means"];bootscales=cache["bootstrap_scales"]
    covariance=cache["scale_covariance"]
    for row in result["mean_models"]:
        row["positive_leading_amplitude"]=bool(row["coefficients"][0]>0)
        if row["minimum_n"]==int(sizes[0]):
            draws=[fit_mean_model(sizes,m,se,row["model"]) for m in bootmeans]
            row["bootstrap_nonpositive_leading_fraction"]=float(np.mean([
                not draw["positive_leading_amplitude"] for draw in draws]))
    for row in result["equation45_models"]:
        row["positive_predictions"]=bool(np.all(np.array(row["prediction"])>0))
        if row["minimum_n"]==int(sizes[0]):
            draws=[fit_equation45(sizes,k,covariance,delta_ratio=row["delta_ratio"],
                                  b_bound=row["b_bound"]) for k in bootscales]
            row["bootstrap_p_boundary_fraction"]=float(np.mean([x["p_at_boundary"] for x in draws]))
            row["bootstrap_nonpositive_prediction_fraction"]=float(np.mean([
                not x["positive_predictions"] for x in draws]))
    boot_dimensions=np.array([-1/np.polyfit(np.log(sizes),np.log(k),1)[0] for k in bootscales])
    result["histogram_collapse_mc_ci95"]=np.quantile(boot_dimensions,[.025,.975]).tolist()
    result["histogram_collapse_bootstrap_dimensions"]=boot_dimensions.tolist()
    records=[]
    for item in result["sources"]:
        path=Path(item["file"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError("Input changed after original analysis")
        records.extend(json.loads(l) for l in path.read_text().splitlines())
    result["full_pipeline_collapse_window_checks"]=[dict(minimum_n=int(n),
        **collapse_histograms([r for r in records if r["n"]>=n])) for n in sizes[:-2]]
    result["equation45_window_convention"]=("Regression windows reuse full-range common shift and scales; "
        "full_pipeline_collapse_window_checks separately recompute central collapses after dropping sizes")
    result["finalization"]={"provenance":environment_manifest(),"parent_file":str(source),
        "parent_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "method":"Reuse saved bootstrap means/scales; no graph sampling or new bootstrap draws"}
    output.write_text(json.dumps(result,indent=2))
    print(output, "collapse MC95",result["histogram_collapse_mc_ci95"])


if __name__=="__main__":main()
