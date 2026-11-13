# A deliberately provisional arbitrary-covariance sampler, Task 3B

The new module is `src/provisional_excursions_3b.py`. It leaves every Phase 2C
and Phase 3A sampler unchanged. The current user explicitly allows monitored
approximations for exploratory covariance selection. This does **not** turn the
previous unvalidated generic excursion law into a validated one.

## Target and retained mathematical construction

The reference is the local `papers/Budd_Castro.pdf`, arXiv:2207.05355v1,
§§3.3–3.4, equations 32–43 and §4.1. With C=RRᵀ, W={z:Rz>0}, and the principal
spherical Dirichlet mode m, write p=alpha1−1/2 and

```
h(z)=|z|^(-p-1) m(z/|z|).
```

Draw S on W∩S² with density m², run two independent inward Doob-h Brownian
legs from S to 0, reverse the first, concatenate, and rescale time and space
by their total duration T and sqrt(T). For the exact harmonic the durations
are independent inverse-gamma(alpha1,1/2), independently of S. This produces
the paper's **biased** unit-duration excursion, not the original unbiased
cone bridge. The graph uses minima in equal time cells and the existing
tested adjacency implementation.

The new code substitutes the positive corner-factor Ritz function from Task
2C. The stored polynomial must pass its exact-rational Bernstein positivity
certificate. A faster value-only evaluator agrees with the existing
gradient-bearing evaluator in tests; it does not clip or repair its values.
Positive formula evaluation, eigenfunction accuracy, harmonicity and path-law
accuracy are separate properties.

## Continuous finite-SIR initial direction

Uniform Gaussian directions are accepted when they lie in W; the first K
accepted directions are iid uniform spherical-triangle proposals ν. Select
one by weights w(s)=m_tilde(s)². This is a **finite self-normalized importance
resampling approximation**, not exact rejection. Both analytic controls and
generic runs use this same initial-direction algorithm.

If μ=Eνw, the exact density ratio of this finite-SIR law to the normalized
target wν/μ is

```
q_K(s)/p(s) = K μ E[1/(w(s)+sum_{j=1}^{K-1} w(S_j))].
```

It tends to 1 as K→∞ but is not identically 1. For a bounded test f, the usual
ratio expansion gives leading bias

```
[ E_p(f)*Eν(w²) - Eν(w² f) ] / (K μ²).
```

This asymptotic expression is a diagnostic, not a finite-K bound. Candidate
count, effective sample size and maximum normalized weight are saved. K512
and2048 are the coarse/refined defaults. Independent angular experiments
compare these to spherical quadrature moments.

## Randomly rotated sphere quadrature

For each strict interior ball, construct a tensor Gauss–Legendre/azimuth
sphere rule with 2q² nodes ω_i and positive weights w_i summing to 1. Draw an
independent Haar orthogonal frame R_j. Select exit node x+rR_jω_i with weights

```
w_i h_tilde(x+rR_jω_i) / Z_R,
Z_R=sum_i w_i h_tilde(x+rR_jω_i).
```

This is an explicitly approximate transition kernel. Random rotation makes
the unconditional exit law continuous and avoids a fixed directional grid;
it does not eliminate its random-normalizer bias. No Harnack rejection bound
is asserted for the nonharmonic numerical h.

For clarity, if Z=integral h_tilde dμ and **uniformly over every frame**
|Z_R/Z−1|≤δ<1, averaging rotated quadrature gives

```
P_tilde(A)/(1+δ) <= Q_q(A) <= P_tilde(A)/(1−δ),
P_tilde(dy)=h_tilde(y) μ(dy)/Z.
```

Indeed E_R sum_i w_i f(Rω_i)=integral f dμ. The implementation compares q8
or 12 to q16 at every 16th ball. These are sampled numerical diagnostics,
**not** the required uniform δ. Their errors may accumulate over a leg.

Numerical harmonic defects Z/h_tilde(x)−1 are a different issue. Their signed
and absolute logarithmic sums are recorded for each complete leg. The exact
telescoping identity and its missing endpoint/angular/tail terms are explained
in `docs/LIKELIHOOD_PHASE2C.md`; tiny observed sums do not prove a uniform
likelihood bound. In particular, inverse-gamma tail completion is exact only
as a marginal for an actual harmonic Doob process; it remains an approximation
for the locally normalized Ritz chain.

## Brownian paths and minima

After drawing an outer endpoint, the existing Task2C walk-on-sub-spheres
Brownian stopping skeleton is rotated onto it. Its center-to-sphere clock
has Laplace transform sqrt(2s)/sinh(sqrt(2s)); numerical inverse-CDF error,
outer boundary projection and omitted time are retained in metadata. Interior
sub-ball radius is min(step_size,relative_step*outer_radius). Both exact-law
stopping values and clocks still omit Brownian paths between skeleton points.
The origin cutoff adds a straight terminal segment with an approximate
remaining clock. All skeleton points and requested graph-cell boundaries are
retained, and **joint minima remain polygonal approximations**. No independent
coordinate-minimum correction is used for correlated models.

Independent coarse/fine chains are not coupled Brownian refinements. A separate
paired thinning check compares the full saved polygonal path to its q8 regular
time interpolation, documenting additional observation-grid loss on that
same polygonal realization. It cannot quantify the unknown paths inside the
sub-balls. Rootwise metric-ball laws, full root-distance records and sampled
16×16 distance matrices are saved alongside the minima and edges.

## API and law labeling

```python
from src.provisional_excursions_3b import (
    make_provisional_model, sample_provisional_excursion,
)
model = make_provisional_model(C, degree=4, quadrature_order=48,
                               initial_candidates=512)
sample = sample_provisional_excursion(
    2048, rng=rng, model=model, step_size=.01, relative_step=.2,
    exit_order=8, audit_order=16, audit_stride=16,
)
```

Metadata always says `law='biased_approximation'`, `target_law='biased'`,
`validation_status='provisional'`, `grid_values_exact=False` and
`cell_minima_exact=False`. The prior exact independent generator remains
the reference. Finite-grid MCMC and unbiased bridges are not substituted.

The model exposes covariance, R, dimension, p, alpha1, angular_mode, log_h
and sample_direction. It deliberately has no fake `log_exit_bound` and is
not passed through the old analytic-only WOS guard. SPD failure, loss of
stored polynomial positivity, nonfinite evaluations and computational caps
raise errors. The experiment logs failures rather than retrying until success.

## Cost and experiment design

The first unequal generic D4 fit took .016s and one n256 path .721s, retaining
56,056 points across602 outer balls. Typical path memory is a few MB;
exponential arrays are never dense graph matrices. Fits are cached per
configuration by the calling experiment. The initial pilot budgets under
200 MB/path and 3–8 minutes for roughly 200 paths. Every realization saves an
atomic JSON checkpoint; source/environment provenance is recorded before the
experiment. Existing output directories require explicit `--resume`.

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3b_sampler_pilot
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_provisional_excursions_3b.py
```

Pilot controls use I3(D0), A2×R+(D0), A3(D1), where the basis contains the
analytic eigenmode; each receives 24 independent coarse and 24 refined paths,
plus 24 independent exact-law reference paths. New references use q128 for
A2/A3 and exact scalar minima for I3. Two generic cases use 12 paths per
combined setting: unequal(−.2,.1,.35) and equicorrelated.2. Coarse settings
are D4,K512,q8,step.01,relative.2; refined D6,K2048,q12,step.005,relative.1.
This joint change assesses overall stability; it does not identify each
component's individual bias. Both n128 and256 are observed on each path.

The main coordinator may scale absolute step size by
sqrt(2.5/(alpha1−1)) to keep the expected normalized spatial scale comparable
across cones. This uses E(T1+T2)=1/(alpha1−1) for the intended biased law,
and is a design choice, not correction of an exact finite-time law. Larger
n requires separate matched controls and resolution checks. Mean-distance
stability alone never validates local edge/minimum laws.

## Pilot numerical results

The main pilot completed **264 independent full excursions in 196.98 s**, with
no failures: 72 special-cone exact-law references, 144 special-cone provisional
paths and 48 generic provisional paths. One targeted A3 confirmation adds 24
independent provisional paths in 37.52 s. Conditional reference bridges below
are additional conditional draws and are **not** independent full-law samples.

At n=256, comparisons against the 24 new independent reference paths give:

| Control | Coarse mean-distance change, MC95 | Refined change, MC95 |
|---|---:|---:|
| I3 | −1.59% [−5.24%, +2.06%] | −0.32% [−4.19%, +3.55%] |
| A2×R+ | −0.03% [−3.40%, +3.34%] | −1.11% [−4.28%, +2.05%] |
| A3 | −0.68% [−3.07%, +1.71%] | +0.52% [−2.37%, +3.41%] |

The intervals describe between-realization Monte Carlo uncertainty, not
discretization bias bounds. Largest individual joint-minimum CDF differences
are 0.06–0.165 in these small ensembles. Therefore neither the 1% mean-bias
goal nor 0.02 joint-CDF goal is established. Most ball and matrix-shape
differences are smaller than their realization-level uncertainty; all raw
diagnostics are retained rather than declaring validation from the means.

Analytic-control maximum absolute signed cumulative quadrature defects are
3.74e−7, 3.35e−8 and 1.59e−9 for coarse I3/A2/A3; refined values are below
2e−11. Generic observed harmonic defects improve under D4→D6:

| Generic C | Largest leg sum of absolute log defects D4→D6 | Largest absolute signed sum D4→D6 |
|---|---:|---:|
| (−.2,.1,.35) | .02956 → .01005 | .00648 → .00263 |
| Equicorrelated .2 | .00732 → .00176 | .00213 → .000254 |

Refined q12 versus q16 normalizer differences are below 8e−11 at audited
balls. These improvements concern the angular kernel; they do not control
the remaining path and minimum errors. Generic mean distances change
4.1619→4.3393 (unequal) and 4.4699→4.3218 (equi.2), with differences
0.1773±0.1261 SE and −0.1481±0.1463 SE. Twelve paths per setting cannot
distinguish these changes confidently from Monte Carlo variation.

An isolated initial-direction audit used 1,024 independent directions for
each K512/2048 and each of the two generic cones (4,096 directions total,
8.80 s). Nine first/second moments all lie within 2.11 MC SE of a separate
order64 spherical-quadrature reference for the numerical D6 m² law. The
largest leading-1/K predicted moment bias is 1.28e−4 at K512; this prediction
is not a finite-sample distributional certificate. Initial angular error
appears smaller than the unresolved path/minimum error at pilot precision.

## An anomalous A3 cohort and a sharper conditional control

The first refined A3 cohort has mean leg lifetime .07510 versus exact .09091,
with nominal marginal KS p=.053. Its inverse-lifetime mean is 2.48 theoretical
standard errors high. Adding 64 preserved Phase2C references to the 24 new
references gives one A3 joint-CDF residual −.1404±.0360 SE. This cannot simply
be dismissed because graph mean distances agree.

A prespecified response was one independent 24-path repeat, not a series of
repeats until a convenient answer appeared. Its lifetime mean is .08531,
nominal KS p=.221; its largest joint-CDF residual is −.0774±.0413 SE. These
changes suggest cohort fluctuation, while leaving local-law uncertainty.
The many correlated diagnostic thresholds are not simultaneous tests.

To separate random junction/time variation from conditional path error, a new
audit draws exact Bessel or matrix cone bridges **conditional on the same saved
S,T1,T2** as each of the 168 provisional special-cone paths. Reversing a true
inward Doob leg conditional on lifetime gives the killed-cone bridge from 0
to S, for which the special-cone constructions apply. A2/A3 use q128 grid
values and polygonal minima; I3 includes exact conditional scalar minima.
The reference shares the junction/time data, while its interior Brownian
randomness is independent. This pairing is not a coupled Brownian refinement
and cannot validate the marginal angular or lifetime law. Its total cost is
9.61 s; all conditional reference minima, edges, root distances and matrices
are checkpointed separately.

At n=256, paired provisional-minus-conditional-reference mean changes are
coarse I3 −.94%, A2 −.37%, A3 −.057%; refined I3 +2.80%
[−.26%, +5.85%], A2 −.001%, A3 +.10%; A3 confirmation +.55%. The A3 refined
and confirmation largest joint-CDF residuals become −.0673±.0445 and
+.0330±.0281 SE. Junction/time imbalance explains a substantial part of the
original A3 discrepancy, but the I3 refined discrepancy and remaining joint
uncertainty preclude a claim of uniformly accurate continuous-minimum laws.

Paired q8 regular-time thinning gives edge Jaccard values about .80–.83 against
the retained full polygonal graphs. This is an observation-grid warning, not
the fraction of unknown true edges that are wrong. It reinforces why the main generator
retains all stopping points and why a stable average distance is insufficient.

## Decision and artifacts

The efficient positive formula and angular-quadrature construction overcome
the previous **rejection-envelope cost obstruction** sufficiently for an
explicitly provisional, moderate-condition-number covariance screen. They
do **not** overcome all continuous-path/minimum and accumulated-law errors.
Use independent graph selection/confirmation seeds, matched special-cone
anchors at each graph size and separate degree/path refinements; interpret
close optimizer choices relative to those sensitivities. Retain the validated
exact independent generator and exact-grid A2/A3 references for final claims.
No precise arbitrary-covariance Hausdorff exponent is justified by this pilot.

All pilot artifacts live under `results/phase3b/sampler_pilot/`:

- `summary.json`, `distribution_diagnostics.json`: independent cohort summaries.
- `augmented_control_audit.json`: saved-data reuse, clock and conditional-radial
  diagnostics, explicitly preserving the original A3 anomaly.
- `angular_diagnostics.json`: isolated finite-SIR moment experiment.
- `conditional_bridge_audit/`: all 168 conditional bridge controls and paired
  summaries; these must not be pooled as independent full-law graph samples.
- `records/`, `arrays/`, `paths/`: immutable realization checkpoints, all graph
  minima/root distances/matrices, and selected complete stopping paths.
- `a3_fine_confirmation/`: separately seeded, targeted A3 repeat.
- `pilot_validation.pdf`, `pilot_validation.png`: primary numerical diagnostics.
- `source_snapshot/`, `final_source_snapshot/`, `final_sampler_manifest.json`:
  original/final source versions and provenance. No Phase2C/3A artifact changed.

The supplementary saved-data and angular audit is reproduced by:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_sampler_pilot --resume --angular-only
```

Already completed conditional/angular checkpoints are reused. All 11 new
sampler tests pass, including positivity, exact stored-formula agreement,
harmonic control normalization, graph invariants and matched junction values
for the three conditional reference constructions.
