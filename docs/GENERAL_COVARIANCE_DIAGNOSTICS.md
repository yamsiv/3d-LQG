# General-covariance sampler diagnostics and validation gate

The finite-grid orthant Gaussian-bridge sampler accepts arbitrary SPD
correlations, including the user convention `a=C12, b=C13, c=C23`. It uses
multiscale elliptical-slice updates preserving the finite-grid conditional
Gaussian law. Its finite burn-in and its continuous-path approximation remain
separate uncertainties. It is not the paper's biased Doob-transform sampler for
a general cone, and the following data do not establish continuum geometry.

The validation gate remains **closed for extensive covariance-dependent
Hausdorff-dimension studies**. Two observed problems justify this decision:
grid-dependent graph distances and unresolved mixing near degeneracy.

## Diagnostic covariance pilot

`python -m experiments.parameter_sweep` generated 108 small graphs: sizes
128, 256 and 512; four independently seeded chains at each size and covariance;
four independent uniform roots per graph; 32 multiscale sweeps; two fine-grid
segments per graph cell. The nine matrices are:

- Equicorrelations `rho=-0.49,-0.30,0,0.30,0.60`.
- Birectangular cones `alpha=pi/3` and `2pi/3`, with
  `C12=-cos(alpha), C13=C23=0`.
- Unequal triples `(C12,C13,C23)=(-0.2,0.1,0.35)` and `(-0.35,-0.1,0.2)`.

The zero-correlation point is deliberately sampled using the same finite-grid
MCMC method, so it is a matched surrogate control. Every graph passed
connectedness, Hamiltonian-cycle and coordinate-diagonal-count checks. At
`n=512`, mean distances for the equicorrelated points were respectively 4.758,
4.986, 4.976, 5.044 and 6.040. Approximately 14% of cell minima were tied in the
piecewise-linear paths and required the documented symbolic tie convention.
These are finite-grid observations. No asymptotic dimension is fitted, and the
apparent trend at this resolution is not promoted to a continuum claim.

Raw records, full traces, matrices, angles, condition numbers, seeds and source
hashes are in `results/data/covariance_pilot.jsonl` and its metadata file. The
summary is `results/data/covariance_pilot.summary.json`; its figure is
`results/figures/covariance_pilot.summary.png`.

## Independent grid/burn-in sensitivity

`python -m experiments.sampler_sensitivity` generated 96 additional graphs:
identity and `rho=-0.49` at sizes 256 and 512, six independent chains for each of
three `(oversample,sweeps)` configurations `(2,32),(2,128),(8,128)`, and twelve
exact unbiased identity graphs at each size. Every graph has eight uniform
roots. Configurations use independent seeds. The exact control samples the
continuous excursion's cell minima rather than a piecewise-linear surrogate.

At `n=512`, the identity results are:

| Sampler | Mean pair distance | Monte Carlo SE | Fraction of tied interior cell minima |
| --- | ---: | ---: | ---: |
| Grid, oversample 2, 32 sweeps | 4.9069 | 0.0484 | 0.1413 |
| Grid, oversample 2, 128 sweeps | 4.9757 | 0.0619 | 0.1426 |
| Grid, oversample 8, 128 sweeps | 5.1965 | 0.1443 | 0.0376 |
| Exact continuous unbiased identity | 5.1781 | 0.0560 | 0 |

The refined control agrees within sampling uncertainty, whereas the coarse grid
changes distances and has many tied minima. Agreement at one refined grid and
these sample counts is useful evidence, but it is not a convergence certificate.

Near `rho=-0.49`, the covariance condition number is 74.5. The path-area
diagnostic (mean of positive path coordinates) drifts substantially during 128
sweeps. At `n=512`, its first-quarter versus last-quarter mean is 0.3757 versus
0.2216 for oversample 2, and 0.3719 versus 0.2278 for oversample 8. This drift
persists after increasing resolution and prevents claiming equilibrium. Small
Monte Carlo standard errors across these finitely burned-in chains would not
cover this systematic error.

Raw records and per-sweep traces are in:

- `results/data/sensitivity_grid_o2_s32.jsonl`
- `results/data/sensitivity_grid_o2_s128.jsonl`
- `results/data/sensitivity_grid_o8_s128.jsonl`
- `results/data/sensitivity_exact_unbiased_identity.jsonl`

The combined summary is `results/data/sampler_sensitivity.summary.json`; the
figure is `results/figures/sampler_sensitivity.png`. Metadata sidecars give the
full run configurations and source hashes.

## Consequences for the next graph study

The generator can be exercised on the full three-parameter domain, but a
reliable covariance-to-Hausdorff-dimension catalogue has not been produced by
this approximation. Prioritize exact special cones or the paper's validated
spherical eigenfunction/Doob-transform approach. For finite-grid MCMC, require
multiple dispersed positive initial paths, substantially longer traces,
integrated autocorrelation/effective-sample-size checks, several grid
refinements, and direct comparisons with exact independent and special-cone
controls. No polynomial-time mixing guarantee is claimed. In particular, do
not use the near-degenerate pilot to infer a minimum dimension or a map to an
LQG coupling.
