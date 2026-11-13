"""Small independent A3 law controls; no graph scaling or parameter sweep.

Run: MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.validate_a3
All raw arrays and seeds are retained; output directories must be new.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
from scipy.stats import chi2, invgamma, ks_2samp, kstest
from src.provenance import environment_manifest
from src.weyl_a3 import (a3_covariance, a3_polynomial, a3_junction_direction,
                         sample_a3_excursion)


def independent_angles(count, rng):
    """Uniform trace-free sphere folded into the chamber, rejected by u².

    Independent of the GUE eigenvalue construction. The sharp maximum of
    Vandermonde on a fixed-radius sphere is attained at normalized H4 roots.
    """
    roots = np.sort(np.polynomial.hermite_e.hermeroots([0,0,0,0,1]).real)
    roots /= np.linalg.norm(roots)
    bound = a3_polynomial(np.diff(roots)/np.sqrt(2))**2
    retained, proposals = [], 0
    while sum(len(x) for x in retained) < count:
        values = rng.normal(size=(max(256, count//2), 4))
        values -= values.mean(axis=1, keepdims=True)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        values.sort(axis=1)
        weight = a3_polynomial(np.diff(values, axis=1)/np.sqrt(2))**2
        if np.any(weight > bound*(1+1e-12)):
            raise RuntimeError('Sharp angular rejection envelope violated')
        keep = rng.random(len(values)) < weight/bound
        retained.append(values[keep])
        proposals += len(values)
    return np.concatenate(retained)[:count], proposals, float(bound)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=2400)
    parser.add_argument('--angular-samples', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=20261111031)
    parser.add_argument('--output', type=Path, default=Path('results/phase2c/a3/law_validation'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing outputs; select a new directory')
    args.output.mkdir(parents=True)
    base = args.seed
    rng = np.random.default_rng(base)
    tracemalloc.start()
    pilot_start = time.perf_counter()
    for _ in range(8):
        sample_a3_excursion(128, rng, 'biased', 64)
    pilot_seconds = time.perf_counter()-pilot_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    estimated = args.samples*2*pilot_seconds/8*9/8194
    print(json.dumps({'pilot_eight_paths_n128_q64_seconds': pilot_seconds,
                      'pilot_python_peak_bytes': peak,
                      'law_experiment_estimated_seconds': estimated}), flush=True)
    chol = np.linalg.cholesky(a3_covariance())
    inverse = np.linalg.inv(a3_covariance())
    raw = {'unbiased_r2': [], 'biased_r2': [], 'biased_conditional_expected_r2': [],
           'biased_times': [], 'biased_u': [], 'biased_junction_r2': []}
    start = time.perf_counter()
    # Separate SeedSequence-derived streams per independent graph/path.
    seeds = np.random.SeedSequence(base+1).spawn(args.samples*2)
    for index in range(args.samples):
        original = sample_a3_excursion(8, np.random.default_rng(seeds[2*index]), 'unbiased', 1)
        white = np.linalg.solve(chol, original.values[[2,4,6]].T).T
        raw['unbiased_r2'].append(np.sum(white*white, axis=1))
        biased = sample_a3_excursion(8, np.random.default_rng(seeds[2*index+1]), 'biased', 1)
        times = np.array(biased.metadata['hitting_times'])
        u = biased.metadata['junction_time']
        point = biased.values[np.argmin(abs(biased.times-.5))]
        duration = u if u >= .5 else 1-u
        expected = (.5/duration)**2/times.sum()+15*.5*(duration-.5)/duration
        raw['biased_r2'].append(point@inverse@point)
        raw['biased_conditional_expected_r2'].append(expected)
        raw['biased_times'].append(times)
        raw['biased_u'].append(u)
        raw['biased_junction_r2'].append(1/times.sum())
    angle_rng = np.random.default_rng(base+2)
    gue = np.stack([a3_junction_direction(angle_rng) for _ in range(args.angular_samples)])
    alternate, proposals, bound = independent_angles(args.angular_samples, np.random.default_rng(base+3))
    raw.update({'gue_directions': gue, 'independent_rejection_directions': alternate})
    raw = {key: np.asarray(value) for key, value in raw.items()}
    np.savez_compressed(args.output/'raw_arrays.npz', **raw)
    def estimate(values):
        return {'mean': float(values.mean()), 'independent_sample_se': float(values.std(ddof=1)/np.sqrt(len(values)))}
    def ks(result):
        return {'statistic': float(result.statistic), 'pvalue': float(result.pvalue)}
    standardized = raw['unbiased_r2']/np.array([.1875,.25,.1875])
    residual = raw['biased_r2']-raw['biased_conditional_expected_r2']
    angular_tests = {}
    for coordinate in range(4):
        angular_tests[f'eigenvalue_{coordinate}'] = ks(ks_2samp(gue[:,coordinate], alternate[:,coordinate]))
    angular_tests['u_squared'] = ks(ks_2samp(a3_polynomial(np.diff(gue,axis=1)/np.sqrt(2))**2,
                                           a3_polynomial(np.diff(alternate,axis=1)/np.sqrt(2))**2))
    summary = {'configuration': vars(args) | {'output': str(args.output)},
               'environment': environment_manifest(), 'covariance': a3_covariance().tolist(),
               'ensemble_counts': {'unbiased_paths': args.samples, 'biased_paths': args.samples,
                                   'angles_each_method': args.angular_samples},
               'pilot': {'seconds_eight_n128_q64': pilot_seconds, 'python_peak_bytes': peak,
                         'estimated_small_law_seconds': estimated},
               'runtime_seconds_excluding_pilot': time.perf_counter()-start,
               'original_chi15_squared_means': [estimate(standardized[:,i]) for i in range(3)],
               'original_chi15_squared_ks': [ks(kstest(standardized[:,i], chi2(15).cdf)) for i in range(3)],
               'original_r2_covariance': np.cov(raw['unbiased_r2'].T).tolist(),
               'original_r2_covariance_theory': (30*(np.minimum.outer([.25,.5,.75],[.25,.5,.75])-
                                                     np.outer([.25,.5,.75],[.25,.5,.75]))**2).tolist(),
               'biased_conditional_r2_residual': estimate(residual),
               'biased_junction_r2': estimate(raw['biased_junction_r2']),
               'biased_junction_r2_theory': 2*6.5**2/14,
               'biased_hitting_time_mean': estimate(raw['biased_times'].mean(axis=1)),
               'biased_hitting_time_theory': 1/11,
               'biased_hitting_time_first_leg_ks': ks(kstest(raw['biased_times'][:,0], invgamma(6.5,scale=.5).cdf)),
               'biased_junction_u_variance': float(raw['biased_u'].var(ddof=1)),
               'biased_junction_u_variance_theory': 1/56,
               'angular_two_sample_tests': angular_tests,
               'angular_rejection': {'proposals': proposals, 'sharp_u_squared_bound': bound,
                                     'target_acceptance': 6/(180180*bound)},
               'interpretation': 'Exact finite-grid benchmark. KS diagnostics are not proofs or joint-minimum validation.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3, figsize=(13,3.7))
    x = np.sort(standardized[:,1])
    axes[0].plot(x, np.arange(1,len(x)+1)/len(x), label='A3 midpoint samples')
    axes[0].plot(x, chi2(15).cdf(x), '--', label='chi-square(15)')
    axes[0].set(xlabel='Squared radius / t(1−t)', ylabel='CDF', title='Original law')
    for values, label in [(raw['biased_r2'], 'paper-biased'), (raw['unbiased_r2'][:,1], 'original')]:
        x = np.sort(values)
        axes[1].plot(x, np.arange(1,len(x)+1)/len(x), label=label)
    axes[1].set(xlabel='Squared midpoint radius', ylabel='CDF', title='Distinct excursion ensembles')
    for values, label in [(gue, 'GUE eigenvalues'), (alternate, 'uniform-angle rejection')]:
        x = np.sort(values[:,3])
        axes[2].plot(x, np.arange(1,len(x)+1)/len(x), label=label)
    axes[2].set(xlabel='Largest junction eigenvalue (unit radius)', ylabel='CDF', title='Independent angular check')
    for axis in axes:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output/'law_validation.png', dpi=170)
    print(json.dumps({key: summary[key] for key in ['runtime_seconds_excluding_pilot','original_chi15_squared_means',
                                                 'biased_conditional_r2_residual','angular_two_sample_tests']}), flush=True)


if __name__ == '__main__':
    main()
