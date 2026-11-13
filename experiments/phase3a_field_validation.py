"""Checkpointed, small unit-log Fourier field and GMC validation ensembles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
from scipy.stats import skew, kurtosis, normaltest

from src.log_field_3d import (sample_coupled_fields, sample_field, spectral_configuration,
                              log_coefficient_fit, resample_field)
from src.gmc_3d import (gmc_weights, coarse_cell_masses, moment_threshold,
                        euclidean_mass_exponent)
from src.provenance import environment_manifest


def basic_summary(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1)/np.sqrt(len(a))),
            "sd": float(a.std(ddof=1)), "quantiles": np.quantile(a, [.05, .25, .5, .75, .95]).tolist()}


def field_record(field, gammas):
    h = field.h
    n = h.shape[0]
    coefficients = np.fft.fftn(h)/n**3
    empirical = np.fft.ifftn(abs(np.fft.fftn(h))**2).real/n**3
    separations = sorted(set([1, 2, 3, n//8, n//4, n//2]))
    modes = [(1, 0, 0), (1, 1, 0), (2, 1, 0)]
    result = {"n": n, "variance": field.variance, "point": float(h[0, 0, 0]),
              "spatial_variance": float(np.mean(h*h)),
              "mean": float(h.mean()), "imaginary_max": field.metadata["imaginary_roundoff_max"],
              "separations": separations,
              "empirical_axial_covariance": empirical[separations, 0, 0].tolist(),
              "theoretical_axial_covariance": field.covariance[separations, 0, 0].tolist(),
              "standardized_modes": [[float((coefficients[k]/np.sqrt(field.spectrum[k])).real),
                                       float((coefficients[k]/np.sqrt(field.spectrum[k])).imag)] for k in modes],
              "gmc": {}}
    for gamma in gammas:
        mu = gmc_weights(h, gamma, field.variance)
        coarse = {str(b): coarse_cell_masses(mu, b).ravel().tolist() for b in [1, 2, 4, 8]}
        p = mu.probabilities.ravel()
        result["gmc"][str(gamma)] = {
            "total_mass": mu.total_mass, "log_total_mass": mu.log_total_mass,
            "normalized_effective_fraction": float(1/(len(p)*np.sum(p*p))),
            "max_atom": float(p.max()), "entropy": float(-np.sum(p*np.log(p))),
            "coarse_masses": coarse,
        }
    return result


def summarize(records, sizes, gammas):
    out = {"realizations": len(records), "field": {}, "gmc": {}, "coupled": {}}
    for n in sizes:
        rows = [row["fields"][str(n)] for row in records]
        empirical = np.asarray([r["empirical_axial_covariance"] for r in rows])
        coeff = np.asarray([r["standardized_modes"] for r in rows])
        ntest = normaltest(np.asarray([r["point"] for r in rows]))
        out["field"][str(n)] = {
            "point": basic_summary([r["point"] for r in rows]),
            "point_skew": float(skew([r["point"] for r in rows])),
            "point_excess_kurtosis": float(kurtosis([r["point"] for r in rows])),
            "normaltest_p": float(ntest.pvalue),
            "spatial_variance": basic_summary([r["spatial_variance"] for r in rows]),
            "ensemble_variance": rows[0]["variance"], "separations": rows[0]["separations"],
            "empirical_covariance_mean": empirical.mean(axis=0).tolist(),
            "empirical_covariance_graph_se": (empirical.std(axis=0, ddof=1)/np.sqrt(len(rows))).tolist(),
            "theoretical_covariance": rows[0]["theoretical_axial_covariance"],
            "mode_component_means": coeff.mean(axis=0).tolist(),
            "mode_component_variances": coeff.var(axis=0, ddof=1).tolist(),
            "mode_correlation_matrix": np.corrcoef(coeff.reshape(len(rows), -1), rowvar=False).tolist(),
        }
        out["gmc"][str(n)] = {}
        for gamma in gammas:
            g = [r["gmc"][str(gamma)] for r in rows]
            threshold = moment_threshold(gamma)
            moment_fits = {}
            for q in [.5, 1., 1.25, 2.]:
                values = [np.mean([np.mean(np.asarray(r["coarse_masses"][str(b)])**q) for r in g])
                          for b in [2, 4, 8]]
                fit = np.polyfit(np.log(1/np.array([2, 4, 8])), np.log(values), 1)[0]
                moment_fits[str(q)] = {"moment_values": values, "finite_grid_slope": float(fit),
                                       "continuum_zeta": float(euclidean_mass_exponent(q, gamma)),
                                       "continuum_moment_admissible": q < threshold,
                                       "warning": "Three block scales, cutoff effects; no asymptotic claim"}
            out["gmc"][str(n)][str(gamma)] = {
                "total_mass": basic_summary([r["total_mass"] for r in g]),
                "effective_fraction": basic_summary([r["normalized_effective_fraction"] for r in g]),
                "max_atom": basic_summary([r["max_atom"] for r in g]),
                "continuum_positive_moment_threshold": threshold if np.isfinite(threshold) else "infinite",
                "moments": moment_fits,
            }
    for gamma in gammas:
        ref = [row["fields"][str(max(sizes))]["gmc"][str(gamma)] for row in records]
        out["coupled"][str(gamma)] = {}
        for n in sizes[:-1]:
            small = [row["fields"][str(n)]["gmc"][str(gamma)] for row in records]
            diff = [a["total_mass"]-b["total_mass"] for a, b in zip(small, ref)]
            coarse_l1 = []
            for a, b in zip(small, ref):
                av = np.asarray(a["coarse_masses"]["4"])/a["total_mass"]
                bv = np.asarray(b["coarse_masses"]["4"])/b["total_mass"]
                coarse_l1.append(np.sum(abs(av-bv)))
            out["coupled"][str(gamma)][str(n)] = {
                "total_mass_difference": basic_summary(diff), "coarse_probability_l1": basic_summary(coarse_l1)}
    return out


def plots(output, summary, exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    nmax = max(int(k) for k in summary["field"])
    row = summary["field"][str(nmax)]
    r = np.array(row["separations"])/nmax
    axes[0].errorbar(r, row["empirical_covariance_mean"], yerr=np.array(row["empirical_covariance_graph_se"])*1.96,
                     fmt="o", label="empirical MC95")
    axes[0].plot(r, row["theoretical_covariance"], label="exact spectrum")
    axes[0].set(xscale="log", xlabel="torus axial separation", ylabel="covariance", title=f"Field covariance, N={nmax}")
    axes[0].legend()
    ns = np.array(sorted(int(k) for k in summary["field"]))
    axes[1].plot(np.log(ns), [summary["field"][str(n)]["ensemble_variance"] for n in ns], "o-", label="spectral sum")
    axes[1].set(xlabel="log N", ylabel="point variance", title=f"Variance slope {exact['variance_log_slope']:.4f}")
    axes[1].legend()
    for gamma in summary["gmc"][str(nmax)]:
        means = [summary["gmc"][str(n)][gamma]["total_mass"]["mean"] for n in ns]
        se = [summary["gmc"][str(n)][gamma]["total_mass"]["se"] for n in ns]
        axes[2].errorbar(ns, means, yerr=1.96*np.array(se), marker="o", label=f"γ={gamma}")
    axes[2].axhline(1., color="k", linestyle="--")
    axes[2].set(xlabel="N", ylabel="unnormalized total mass", title="GMC means; rare tails limit precision")
    axes[2].legend()
    fig.tight_layout(); fig.savefig(output/"field_gmc_validation.png", dpi=160); fig.savefig(output/"field_gmc_validation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/phase3a/field_validation"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202611110301)
    args = parser.parse_args()
    output = args.output
    if (output/"records.jsonl").exists(): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = [16, 24, 32, 48]; gammas = [0., .5, 1., 1.5, 2.]
    tracemalloc.start(); start = time.perf_counter()
    for j in range(8):
        fields = sample_coupled_fields(sizes, np.random.default_rng(args.seed+j))
        for f in fields.values(): field_record(f, gammas)
    pilot_seconds = time.perf_counter()-start
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    budget = {"pilot_realizations": 8, "pilot_seconds": pilot_seconds, "python_traced_peak_bytes": peak,
              "expected_main_seconds_linear": pilot_seconds*args.samples/8,
              "reason": "Validate spectrum, GMC means and paired resolution effects; no LFPP solves"}
    (output/"budget.json").write_text(json.dumps(budget, indent=2))
    metadata = {"arguments": vars(args) | {"output": str(output)}, "sizes": sizes, "gammas": gammas,
                "provenance": environment_manifest(), "normalization": "S(k)=1/(4*pi)*|k|^-3",
                "regularization": "sharp spherical |k|<N/3; common Fourier randomness",
                "replicate_unit": "independent complete common-field realization"}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2))
    records=[]; start=time.perf_counter()
    with (output/"records.jsonl").open("x") as handle:
        for j in range(args.samples):
            seed = int(np.random.SeedSequence([args.seed, j, 17]).generate_state(1, dtype=np.uint64)[0])
            fields = sample_coupled_fields(sizes, np.random.default_rng(seed))
            row = {"realization": j, "seed": seed, "fields": {str(n): field_record(f, gammas) for n, f in fields.items()}}
            if j < 2:
                np.savez_compressed(output/f"fields_{j:03d}.npz", **{f"h{n}": f.h for n, f in fields.items()})
            # Same seed, same maximum grid -> same underlying Gaussian modes.
            if j < 64:
                heat = sample_coupled_fields(sizes, np.random.default_rng(seed), regularization="heat")
                row["heat_sensitivity"] = {str(n): {str(g): gmc_weights(heat[n].h, g, heat[n].variance).total_mass
                                                   for g in gammas} for n in sizes}
            if j < 64:
                row["fixed_field_quadrature"]={}
                for n in [16, 24]:
                    fine = resample_field(fields[n], 2*n)
                    row["fixed_field_quadrature"][str(n)]={str(g): {
                        "mass_coarse": gmc_weights(fields[n].h, g, fields[n].variance).total_mass,
                        "mass_fine": gmc_weights(fine.h, g, fine.variance).total_mass} for g in gammas}
            handle.write(json.dumps(row)+"\n"); handle.flush(); records.append(row)
    summary=summarize(records, sizes, gammas)
    summary["main_seconds"]=time.perf_counter()-start
    summary["fixed_field_quadrature"]={}
    summary["regularization_sensitivity"]={}
    for n in [16,24]:
        summary["fixed_field_quadrature"][str(n)]={}
        for gamma in gammas:
            values=[r["fixed_field_quadrature"][str(n)][str(gamma)] for r in records[:64]]
            rel=[v["mass_coarse"]/v["mass_fine"]-1 for v in values]
            summary["fixed_field_quadrature"][str(n)][str(gamma)]={"relative_mass_difference":basic_summary(rel)}
    for n in sizes:
        summary["regularization_sensitivity"][str(n)]={str(g):basic_summary([
            r["heat_sensitivity"][str(n)][str(g)]-r["fields"][str(n)]["gmc"][str(g)]["total_mass"]
            for r in records[:64]]) for g in gammas}
    exact={"variance":{},"isotropy":{},"coefficient_fits":[]}
    for n in [16,24,32,48,64,96,128]:
        _, c, v=spectral_configuration(n)
        exact["variance"][str(n)]=v
        exact["isotropy"][str(n)]={"same_radius_3_axis":float(c[3,0,0]),
                                   "same_radius_3_oblique":float(c[2,2,1]),
                                   "absolute_difference":float(c[3,0,0]-c[2,2,1])}
    ns=np.array([16,24,32,48,64,96,128])
    exact["variance_log_slope"]=float(np.polyfit(np.log(ns),[exact["variance"][str(n)] for n in ns],1)[0])
    for n in [64,96,128]:
        for window in [(.02,.08),(.03,.12),(.04,.2)]:
            exact["coefficient_fits"].append(log_coefficient_fit(n,distance_window=window))
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    (output/"exact_spectral_diagnostics.json").write_text(json.dumps(exact,indent=2))
    plots(output,summary,exact)
    print(json.dumps({"output":str(output),"samples":args.samples,"seconds":summary["main_seconds"],"pilot":budget}))


if __name__=="__main__": main()
