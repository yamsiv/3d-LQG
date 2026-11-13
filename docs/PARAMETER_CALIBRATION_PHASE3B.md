# Phase 3B LFPP/GMC parameter calibration

This workstream uses the published parameter identities, with numerical
finite-resolution crossing slopes in place of the unknown asymptotic exponent.
It does not calibrate the Brownian covariance matrix. Every inferred nonzero
gamma is provisional, including those with narrow sampling intervals.

## Mathematical conventions checked in the original sources

The local source is Contreras Hip–Zhuang,
[arXiv:2504.09141v1](https://arxiv.org/abs/2504.09141v1), equations (1.1)–(1.3),
Definition 1.7 and Proposition 1.8. The path weight is
`epsilon * sum_vertices exp(xi*h_epsilon(v))`, including endpoints. The physical
epsilon factor is essential. If median crossings behave as `N^(-lambda)`, then

```
lambda = -slope(log median D_N versus log N),
Q = (1-lambda)/xi,
gamma = Q - sqrt(Q^2-6) = 6/(Q+sqrt(Q^2-6)),
d_gamma = gamma/xi.
```

The final two relations apply to the subcritical branch `Q>sqrt(6)`.
Contreras Hip–Zhuang,
[arXiv:2512.06292v1](https://arxiv.org/abs/2512.06292v1), equation (1.5) and
Corollary 1.9, relates the metric Hausdorff dimension to that branch. Gamma and
xi determine each other in the theoretical subcritical parameterization; no
closed-form map is known in dimension three. The numerical finite proxies do
not establish that the simulated torus/cube model has reached its limiting
exponent. We never set xi=gamma or equate lambda with dimension.

At xi=0 use the separate exact flat endpoint gamma=0, dimension=3. Q is
undefined/infinite there and is serialized as null rather than a fictitious
finite value. The endpoint does not justify interpolating nonzero gamma below
the smallest calibrated positive xi. For `Q<=sqrt(6)`, retain lambda and Q,
but return no subcritical gamma or dimension.

## Finite model and experiment

The previous `src/log_field_3d.py` and `src/lfpp_3d.py` are reused unchanged.
The field is mean-zero unit-log Fourier noise on the unit three-torus with
covariance spectrum `1/(4*pi*|k|^3)` and sharp spherical cutoff `|k|<N/3`.
Common integer Fourier modes have the same randomness across all requested
sizes. Coupling requires the complete tuple **(32,48,64,96,128)** and the seed;
requesting a smaller maximum grid with the same seed gives a different coupling.

Paths are restricted to an open cube: there are no periodic path shortcuts.
Axis trapezoidal edge weights and source/target half-site costs give exactly
the CHZ vertex-sum rule on the supplied grid. N grid nodes at spacing 1/N span
`[0,1-1/N]^3`; at xi=0 the site-sum crossing is exactly one. This is an
O(1/N) endpoint/domain difference from CHZ's lattice intersection with the
closed unit cube. In addition, the periodic Fourier field and sharp cutoff
differ from the paper's whole-space grounded field and box-average mollifier.
They are finite-volume/mollifier systematics, not implementation bugs.

The main run uses twenty independent complete field families, seeds
202611113510 through 202611113529, five coupled sizes and eleven xi values:
0, .1, .15, .2, .225, .25, .275, .3, .325, .35, .375. Shared parameters and
sizes do not create additional independent realizations. A pilot was saved
separately and is not added to the main sampling count.

Each field/grid/xi/orientation result is independently checkpointed. The source
and configuration are frozen before the run. Every field has a SHA256 checksum;
full grids are saved for the first two main families, and all others reproduce
from seeds plus the complete tuple and frozen source. Calibration field seeds
are distinct from the later metric-measure search/confirmation seeds.

## Cost and uncertainty

A persisted N64/96/128 pilot at xi .275 took2.01 seconds in total (including
field generation) and peaked near1.09GB RSS. N128 has158MB of CSR arrays.
The main design was estimated at6–8 minutes and about1.2GB sequential memory;
graphs at different xi are built sequentially, not retained together.

The primary estimator is ordinary least squares on log **ensemble medians**
over all five sizes. Two thousand bootstrap replicates resample complete
physical field families, preserving all within-family size/parameter coupling.
These intervals quantify finite-ensemble sampling error conditional on the
specified model and fitting rule. Individual grid points, xi values and
orientation crossings are never treated as independent samples.

Systematic sensitivity is reported separately, using the point-estimate
envelope from five prespecified models:

- All five sizes, pure log power law.
- Drop N32, pure log power law.
- Largest three sizes, pure log power law.
- All sizes with fixed `B*N^(-1/2)` additive correction to log distance.
- All sizes with fixed `B/N` additive correction to log distance.

Adjacent two-size estimates and their bootstrap intervals are retained as
diagnostics, not allowed to dominate the core envelope. The correction models
have only five sizes and three fitted coefficients; their extrapolation can be
poorly identified. Their envelope is **not** a rigorous bias bound or an
additional confidence interval. If any correction model leaves the subcritical
branch, that fact is flagged rather than dropped silently.

For a bootstrap distribution crossing `Q=sqrt(6)`, the proportion permitting
subcritical inversion is recorded. Gamma intervals conditional on valid draws
are explicitly conditional; they do not turn a branch-ambiguous estimate into
a validated parameter. A Monte Carlo interval cannot cover unmeasured
regularization, boundary, infrared or continuum path-direction errors.

## Reproduction and API

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_crossings --output results/phase3b/calibration/main
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_calibrate --input results/phase3b/calibration/main --output results/phase3b/calibration/analysis
```

The producer resumes matching configurations by skipping complete checkpoints;
it rejects a changed configuration or producer source. Analysis refuses to
overwrite an existing output directory.

```
from src.parameter_conversion_3b import load_calibration
table = load_calibration('results/phase3b/calibration/analysis/calibration.json')
candidate = table.evaluate(.275)
```

The API interpolates crossing slopes and uncertainty endpoints linearly between
neighboring calibrated xi values; it does not extrapolate, impose monotonicity,
bridge invalid uncertainty nodes or force a subcritical inversion. Returned
`valid=True` means the finite central proxy admits the real subcritical branch,
**not** that the asymptotic map has been validated. All nonzero candidates carry
`asymptotic_map_validated=False` and an explicit provisional status.

## Executed findings

The main run completed in **322.40s**, peaking at1.26GB RSS. All1,100 expected
crossing checkpoints parsed and passed positive/finite checks. The first two
saved complete field families passed ZIP CRC and array-hash checks. Frozen
producer hashes match their current source files. The saved pilot is separate
and is not an additional independent family. Four-seed heat/infrared/orientation
controls took13.86s,13.92s and28.54s, respectively; all200 expected control
checkpoints are complete. Orientations and filters reuse the first four main
physical seeds and do not increase the independent-family count.

| xi | lambda proxy | gamma proxy | Paired MC95 gamma | Valid-model/window gamma envelope | d proxy |
| --- | ---: | ---: | --- | --- | ---: |
| .10 | .02547 | .31286 | [.31130,.31554] | [.30158,.31286] | 3.1286 |
| .15 | .05588 | .49619 | [.48917,.50413] | [.48098,.49794] | 3.3079 |
| .20 | .09283 | .71827 | [.70839,.73158] | [.71827,.76411] | 3.5913 |
| .225 | .11027 | .85001 | [.83689,.88929] | [.85001,.90544] | 3.7778 |
| .25 | .12977 | 1.00771 | [.99296,1.07561] | [1.00771,1.11605] | 4.0308 |
| .275 | .15262 | 1.21191 | [1.19328,1.31586] | [1.21191,1.51861] | 4.4069 |
| .30 | .17460 | 1.49840 | [1.44956,1.69732] | [1.49840,1.74131], plus invalid model | 4.9947 |
| .325 | .20039 | 2.22965 | [1.87258,2.36890] **conditional on44.15% valid draws** | Only full-window central proxy valid | 6.8604 |
| .35 | .22799 | unavailable | No valid draws | No valid model | unavailable |
| .375 | .25237 | unavailable | No valid draws | No valid model | unavailable |

The table's envelope is conditional on models with real inversion; the explicit
invalid-model flag is part of the result. In particular, the .325 point is
**unreliable as a matched subcritical calibration**, despite a finite central
gamma. The annotated figure marks this and shades the invalid central branch.

At xi=.3 the pure-power gamma estimates are1.498(all sizes),1.529(drop32),
1.601(N64–128), with dimension proxies4.995,5.095,5.336. Adjacent-window
gamma values are1.465,1.370,1.570,1.653. Thus increasing the upper resolution
supports the earlier finite proxy near gamma1.5, while the remaining drift is
still material. The fixed1/N correction gives gamma1.741. The fixed1/sqrt(N)
correction gives lambda.2714, with MC95[.0546,.3820], and leaves the real
subcritical branch. Five-size extrapolation cannot distinguish those ansatzes
reliably. No fit was selected to agree with a graph dimension.

Paired controls, using only the first four fields, show:

| Control | xi | Mean N128 crossing change | Change in full-window lambda | Paired MC95 lambda change |
| --- | ---: | ---: | ---: | --- |
| Heat versus sharp filter | .2 | +2.22% | −.01039 | [−.02423,.00410] |
| Heat versus sharp filter | .3 | +4.11% | −.01799 | [−.03860,.00543] |
| Delete Fourier modes0<\|k\|≤1 | .2 | +11.38% | −.00344 | [−.01986,.01100] |
| Delete Fourier modes0<\|k\|≤1 | .3 | +16.92% | +.00180 | [−.01565,.01156] |
| Cross along axis1 instead of0 | .2 | +7.65% | −.00602 | [−.01978,.00597] |
| Cross along axis1 instead of0 | .3 | +13.53% | −.00427 | [−.01099,.02305] |
| Cross along axis2 instead of0 | .2 | +4.36% | +.01152 | [−.01268,.01994] |
| Cross along axis2 instead of0 | .3 | +6.20% | +.00648 | [−.00209,.02560] |

These are paired finite-volume sensitivities with low power, not estimates of
an anisotropy in the field law: coordinate permutation preserves the ensemble.
They show that a few samples' crossing levels fluctuate more than their
multi-size slopes. The axis0 repeat matches the main checkpoint exactly.

A deterministic coordinate/epsilon sensitivity maps the identical N-node field
samples to a closed unit cube with spacing1/(N−1), multiplies crossing lengths
byN/(N−1), and fits againstlog(N−1). It shifts lambda by+.01511 at xi.2 and
+.01373 at xi.3. This is a change of finite domain/regularization convention,
not a new field simulation; its flat site-sum length becomes1+epsilon. The
finite endpoint effect is comparable to sampling uncertainty and should not
be confused with an exponent discrepancy.

The observed regularizer/endpoint sensitivities are comparable to the primary
slope's sampling error. The conversion is usable for **exploratory linked
candidate selection through approximately xi=.3**, with explicit model and
parameter brackets. It is not a trustworthy asymptotic parameter map. Do not
take gamma.325 as a routine primary candidate or interpret the missing gamma
at .35 as a measured location of the theoretical critical endpoint.

## Saved outputs and checks

- [Main raw checkpoints](../results/phase3b/calibration/main/), including
  immutable config, producer snapshots, provenance, seeds and COMPLETE marker.
- [Calibration table](../results/phase3b/calibration/analysis/calibration.json),
  [CSV](../results/phase3b/calibration/analysis/conversion.csv),
  [all models](../results/phase3b/calibration/analysis/models.csv), and
  [paired bootstrap arrays](../results/phase3b/calibration/analysis/bootstrap.npz).
- [Annotated figure](../results/phase3b/calibration/annotated_figures/calibration.png)
  and [PDF](../results/phase3b/calibration/annotated_figures/calibration.pdf).
  Original unannotated analysis figures are preserved.
- [Paired controls](../results/phase3b/calibration/controls_summary/controls.json)
  and [integrity audit](../results/phase3b/calibration/audit.json).

The parameter identities, known power/correction-law fits, invalid-branch
handling and no-extrapolation behavior pass the new tests. Combined with the
existing field/LFPP checks, **19 relevant tests passed**. The API additionally
returns `gamma_mc95_is_conditional` and
`gamma_systematic_has_invalid_models`, so clients cannot mistake conditional
intervals or partial envelopes for a complete uncertainty statement.
