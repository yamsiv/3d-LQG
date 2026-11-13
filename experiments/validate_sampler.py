"""Independent distribution checks and forced-I3 finite-grid MCMC diagnostics.

Run: .venv/bin/python -m experiments.validate_sampler --output results/validation
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy import stats

from src.brownian_excursions import (
    sample_independent_excursion, sample_conditioned_bridge, positive_bridge_minima,
)


def exact_grid_independent_bridge(steps, dimension, rng):
    """Exact discrete-time positive Gaussian bridge by coordinate Vervaat shifts.

    A continuous-increment bridge has a unique grid minimum. Cyclic invariance
    of Gaussian bridge increments gives one positive cyclic rotation per orbit.
    This is valid for independent coordinates only; separate rotations would
    destroy a nontrivial intercoordinate covariance.
    """
    increments = rng.normal(size=(steps,dimension))/np.sqrt(steps)
    increments -= increments.mean(axis=0)
    values = np.vstack([np.zeros(dimension),np.cumsum(increments,axis=0)])
    for k in range(dimension):
        shift = np.argmin(values[:-1,k])
        values[1:,k] = np.cumsum(np.roll(increments[:,k],-shift))
    values[-1] = 0
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/validation")
    parser.add_argument("--seed", type=int, default=2026111003)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--mcmc-chains", type=int, default=48)
    parser.add_argument("--mcmc-grid", type=int, default=64)
    parser.add_argument("--mcmc-sweeps", default="16,64,256")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    hitting = np.empty((args.samples,2))
    junction = np.empty(args.samples)
    directions = np.empty((args.samples,3))
    radial = np.empty(args.samples)
    for index in range(args.samples):
        sample = sample_independent_excursion(4,3,rng,law="biased")
        hitting[index] = sample.metadata["hitting_times"]
        junction[index] = sample.metadata["junction_time"]
        directions[index] = sample.metadata["junction_direction"]
        unbiased = sample_independent_excursion(4,3,rng,law="unbiased")
        radial[index] = np.sum(unbiased.values[2]**2)/.25
    minima = positive_bridge_minima(np.full(100000,.7),1.1,.25,rng)
    levels = np.array([.05,.2,.4,.6])
    survivor = (1-np.exp(-2*(.7-levels)*(1.1-levels)/.25))/(1-np.exp(-2*.7*1.1/.25))
    empirical = np.array([np.mean(minima>level) for level in levels])
    checks = {
        "hitting_inv_gamma_ks": stats.kstest(hitting.ravel(),"invgamma",args=(3.5,0,.5))._asdict(),
        "junction_beta_ks": stats.kstest(junction,"beta",args=(3.5,3.5))._asdict(),
        "unbiased_radial_chi2_ks": stats.kstest(radial,"chi2",args=(9,))._asdict(),
        "hitting_mean": float(hitting.mean()), "hitting_expected_mean": .2,
        "direction_squared_means": np.mean(directions**2,axis=0).tolist(),
        "direction_squared_expected_mean": 1/3,
        "minimum_levels": levels.tolist(), "minimum_survival_expected": survivor.tolist(),
        "minimum_survival_observed": empirical.tolist(),
    }
    # MCMC diagnostic is deliberately distinct from continuum exact-law checks.
    reference = np.array([exact_grid_independent_bridge(args.mcmc_grid,3,rng)[args.mcmc_grid//2]
                          for _ in range(max(4000,args.samples))])
    mcmc_results, mcmc_raw = [], {}
    for sweeps in map(int,args.mcmc_sweeps.split(",")):
        midpoints, area, traces = [],[],[]
        for chain in range(args.mcmc_chains):
            sample = sample_conditioned_bridge(args.mcmc_grid,np.eye(3),rng,
                                               oversample=1,sweeps=sweeps)
            midpoints.append(sample.values[args.mcmc_grid//2])
            area.append(float(sample.values[1:-1].mean()))
            traces.append(sample.metadata["trace_area_midpoint_radius"])
        midpoints = np.asarray(midpoints)
        observed = np.mean(midpoints,axis=1)
        ref_observed = np.mean(reference,axis=1)
        mcmc_results.append({
            "sweeps": sweeps, "independent_chains": args.mcmc_chains,
            "grid": args.mcmc_grid,
            "mean_midpoint_coordinate": float(observed.mean()),
            "between_chain_standard_error": float(observed.std(ddof=1)/np.sqrt(len(observed))),
            "reference_mean": float(ref_observed.mean()),
            "two_sample_ks": stats.ks_2samp(observed,ref_observed)._asdict(),
            "interpretation": "finite_grid_mixing_diagnostic_only_no_general_C_validation",
        })
        mcmc_raw[f"mcmc_midpoints_sweeps_{sweeps}"] = midpoints
        mcmc_raw[f"mcmc_traces_sweeps_{sweeps}"] = np.asarray(traces)
    np.savez_compressed(output/"sampler_raw.npz",hitting_times=hitting,junction_times=junction,
                        junction_directions=directions,unbiased_midpoint_radial=radial,
                        positive_bridge_minima=minima,discrete_orthant_reference_midpoints=reference,
                        **mcmc_raw)
    record = {"seed":args.seed,"rng":"numpy.default_rng PCG64","samples":args.samples,
              "python":platform.python_version(),"numpy":np.__version__,
              "source":"Budd-Castro arXiv:2207.05355v1 equations32-41; reflection min formula",
              "checks":checks,"mcmc_diagnostics":mcmc_results,
              "elapsed_seconds":time.perf_counter()-started,
              "limitations":["KS checks are diagnostics, not proofs",
                             "I3 MCMC test does not validate correlated cones",
                             "finite-grid bridge target differs from continuous excursion"]}
    (output/"sampler_validation.json").write_text(json.dumps(record,indent=2,default=float)+"\n")
    os.environ.setdefault("MPLCONFIGDIR",str(output/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(10,8),layout="constrained")
    axes[0,0].hist(hitting.ravel(),bins=70,density=True,range=(0,1),alpha=.6,label="sampled T")
    grid = np.linspace(.01,1,500)
    axes[0,0].plot(grid,stats.invgamma.pdf(grid,3.5,scale=.5),label="InvGamma(3.5, .5)")
    axes[0,0].set(xlabel="unscaled hitting time",ylabel="density"); axes[0,0].legend()
    axes[0,1].hist(junction,bins=40,density=True,alpha=.6,label="junction U")
    grid = np.linspace(0,1,500)
    axes[0,1].plot(grid,stats.beta.pdf(grid,3.5,3.5),label="Beta(3.5,3.5)")
    axes[0,1].set(xlabel="unit-time junction",ylabel="density"); axes[0,1].legend()
    axes[1,0].hist(radial,bins=60,density=True,alpha=.6,label="unbiased midpoint")
    grid = np.linspace(0,35,500)
    axes[1,0].plot(grid,stats.chi2.pdf(grid,9),label="chi-square(9)")
    axes[1,0].set(xlabel="4 × squared midpoint radius",ylabel="density"); axes[1,0].legend()
    if mcmc_results:
        sweeps = [row["sweeps"] for row in mcmc_results]
        means = [row["mean_midpoint_coordinate"] for row in mcmc_results]
        errors = [row["between_chain_standard_error"] for row in mcmc_results]
        axes[1,1].errorbar(sweeps,means,yerr=1.96*np.array(errors),fmt="o",label="ESS chains, 95% mean error")
        axes[1,1].axhline(np.mean(reference),color="k",label="exact finite-grid reference")
        axes[1,1].set(xscale="log",xlabel="ESS sweeps",ylabel="mean midpoint coordinate")
        axes[1,1].legend(fontsize=8)
    fig.suptitle(f"Independent sampler checks; {args.samples} exact-law samples, seed {args.seed}")
    fig.savefig(output/"sampler_validation.png",dpi=180)
    plt.close(fig)
    print(json.dumps(record,indent=2,default=float))


if __name__ == "__main__":
    main()
