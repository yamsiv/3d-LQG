"""Task 2B fitting audit; preserves earlier analyses and raw graph records."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.phase2b_scaling import fit_mean_model,fit_equation45
from src.scaling_analysis import collapse_histograms,bootstrap_scaling
from src.provenance import environment_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+")
    parser.add_argument("--output",required=True)
    parser.add_argument("--bootstrap",type=int,default=100)
    parser.add_argument("--seed",type=int,default=2026111051)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    sources=[]
    for filename in args.inputs:
        path=Path(filename)
        records.extend(json.loads(l) for l in path.read_text().splitlines())
        sources.append(dict(file=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    keys=[(r["n"],r["graph_seed"]) for r in records]
    if len(set(keys))!=len(keys):raise ValueError("Repeated graph seeds")
    if any(r["sampler_metadata"]["law"]!="biased" or not r["sampler_metadata"]["cell_minima_exact"] for r in records):
        raise ValueError("Analysis requires matching validated biased exact-minimum records")
    sizes=np.asarray(sorted({r["n"] for r in records}))
    groups={n:[r for r in records if r["n"]==n] for n in sizes}
    means=np.array([np.mean([r["mean_distance"] for r in groups[n]]) for n in sizes])
    se=np.array([np.std([r["mean_distance"] for r in groups[n]],ddof=1)/np.sqrt(len(groups[n])) for n in sizes])
    rng=np.random.default_rng(args.seed)
    start=time.perf_counter()
    central=collapse_histograms(records)
    central_scales=np.array([r["scale"] for r in central["second_pass"]]+[1.])
    boot_means=[]
    boot_scales=[]
    for index in range(args.bootstrap):
        sample=[]
        m=[]
        for n in sizes:
            group=groups[n]
            resampled=[group[i] for i in rng.integers(len(group),size=len(group))]
            sample.extend(resampled)
            m.append(np.mean([r["mean_distance"] for r in resampled]))
        boot_means.append(m)
        collapse=collapse_histograms(sample)
        boot_scales.append([r["scale"] for r in collapse["second_pass"]]+[1.])
        if (index+1)%25==0:print(json.dumps({"collapse_bootstrap":index+1,"seconds":time.perf_counter()-start}),flush=True)
    boot_means,boot_scales=np.asarray(boot_means),np.asarray(boot_scales)
    models=["power","offset","offset_inverse_power","multiplicative_delta_0.1",
            "multiplicative_delta_0.2","multiplicative_delta_0.4"]
    mean_fits=[]
    for drop in range(len(sizes)-3):
        for model in models:
            if model=="offset_inverse_power" and len(sizes)-drop<5:continue
            fit=fit_mean_model(sizes[drop:],means[drop:],se[drop:],model)
            draws=[fit_mean_model(sizes[drop:],draw[drop:],se[drop:],model) for draw in boot_means]
            fit.update(minimum_n=int(sizes[drop]),sizes=sizes[drop:].tolist(),
                       mc_ci95=np.quantile([d["dimension"] for d in draws],[.025,.975]).tolist(),
                       bootstrap_nonpositive_leading_fraction=float(np.mean([not d["positive_leading_amplitude"] for d in draws])),
                       bootstrap_boundary_fraction=float(np.mean([d["boundary"] for d in draws])))
            mean_fits.append(fit)
    eq45=[]
    covariance=np.cov(boot_scales[:,:-1],rowvar=False)
    for drop in range(len(sizes)-3):
        cov=covariance[drop:,drop:]
        for ratio in [.5,1.,2.]:
            for bound in [.05,.15]:
                fit=fit_equation45(sizes[drop:],central_scales[drop:],cov,
                                  delta_ratio=ratio,b_bound=bound)
                draws=[fit_equation45(sizes[drop:],draw[drop:],cov,delta_ratio=ratio,b_bound=bound)
                       for draw in boot_scales]
                fit.update(minimum_n=int(sizes[drop]),sizes=sizes[drop:].tolist(),
                    mc_ci95=np.quantile([d["dimension"] for d in draws],[.025,.975]).tolist(),
                    bootstrap_p_boundary_fraction=float(np.mean([d["p_at_boundary"] for d in draws])),
                    bootstrap_nonpositive_prediction_fraction=float(np.mean([not d["positive_predictions"] for d in draws])),
                    bootstrap_b_boundary_fraction=float(np.mean([d["b_at_boundary"] for d in draws])))
                eq45.append(fit)
    batches=[]
    for b in range(8):
        batch=[r for n in sizes for r in groups[n][b::8]]
        collapse=collapse_histograms(batch)
        batchmean=np.array([np.mean([r["mean_distance"] for r in batch if r["n"]==n]) for n in sizes])
        batches.append(dict(batch=b,offset=fit_mean_model(sizes,batchmean,se,"offset"),
                            collapse_dimension=collapse["effective_dimension"]))
    result=dict(provenance=environment_manifest(),sources=sources,records=len(records),
        sizes=sizes.tolist(),graphs_per_size=[len(groups[n]) for n in sizes],seed=args.seed,
        bootstrap=args.bootstrap,means=means.tolist(),standard_errors=se.tolist(),
        naive_ols=bootstrap_scaling(records,bootstrap=args.bootstrap,seed=args.seed+1),
        histogram_collapse=central,mean_models=mean_fits,equation45_models=eq45,
        full_pipeline_collapse_window_checks=[dict(minimum_n=int(n),
            **collapse_histograms([r for r in records if r["n"]>=n])) for n in sizes[:-2]],
        batches=batches,elapsed_seconds=time.perf_counter()-start,
        limitations=["MC intervals condition on ansatz and fit window", "No unique asymptotic exponent established",
            "Eq45 normalized a+b=1; b bound and delta/p are explicit model assumptions",
            "GLS covariance is estimated from resampled graphs and held fixed across fits",
            "Equation45 fit-window diagnostics reuse full-range collapsed scales/common shift; separate central checks rerun the complete collapse pipeline",
            "All models and windows retained; none selected to match the published value"])
    output.write_text(json.dumps(result,indent=2))
    np.savez_compressed(output.with_suffix(".npz"),sizes=sizes,means=means,se=se,
                        bootstrap_means=boot_means,bootstrap_scales=boot_scales,scale_covariance=covariance)
    print(json.dumps({"output":str(output),"records":len(records),"seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
