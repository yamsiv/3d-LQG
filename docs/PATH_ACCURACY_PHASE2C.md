# Brownian clocks, coupled minima and Task 2C path controls

The target is the **paper-biased** excursion of Budd–Castro
[arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355), §§3.3–3.4 and
4.1, equations (32), (33), (41). Existing samplers and all Phase 2B data are
preserved. New code is in `src/phase2c_paths.py`; experiments are in
`experiments/phase2c_path_validation.py`. This document distinguishes
mathematical stopping-skeleton identities from finite-graph convergence.

## Preregistered targets and experiment design

Before the main runs, metadata recorded a **1% mean-distance bias goal**, a
**0.02 joint-minimum CDF goal**, and the requirement to report local edges
separately. These are goals, not thresholds retrospectively adjusted to pass.
Monte Carlo uncertainty is across independent physical excursions. Nested
graph sizes and grid refinements of one excursion are paired observations,
never extra independent paths.

The two-path-per-cone pilot took about 0.24–0.53 seconds per nested-grid
realization. The 192-realization main minimum experiment (64 each of I3,
A2 × half-line and A3) was therefore budgeted below two minutes. A 128-path
clock pilot took 0.13 seconds; 16,384 clocks were budgeted below 30 seconds
and took about 13 seconds. The six-path new-Doob pilot took 0.48–1.35 seconds
per realization; 64 each of three cones were budgeted below five minutes.
Every physical realization is checkpointed to JSONL. Failed paths are logged
and never replaced silently.

## An isolated clock error in the old path implementation

For standard Brownian motion with generator Δ/2, started at the center of a
three-dimensional ball of radius r, let τ be its first sphere hitting time.
Solving the radial boundary-value problem `(Δ/2)f=s f`, with boundary value
one, gives

```
E exp(−sτ) = z/sinh(z),  z=r sqrt(2s),
Eτ = r²/3,     Var τ = 2r⁴/45.
```

The old spherical random walk uses fixed step ell and clock ell²/3, clips
the last spatial segment, and clips its clock by the same linear fraction.
This does not simulate the continuous first passage time. The untrimmed walk
has overshoot; clipping the final line does not remove its first-order
stopping-time error. In 16,384 unit-ball trials, its relative mean-clock
bias was +11.34%, +6.06%, +3.39%, +1.05% for ell/r=.2,.1,.05,.025.
These are Monte Carlo estimates, not universal asymptotic constants.

Replacing the increments by exact Gaussian Brownian grid increments is
insufficient. Coupled nested Gaussian grids gave +13.13%, +6.63%, +3.39%,
+1.88% biases at the same scales. Coarse-minus-finest clock differences were
measured on the **same Brownian grid realization**. Polygonal exits may occur
before or after a finer polygonal exit within one coarse interval; the code
does not incorrectly assert pathwise monotonicity of clipped times.

For a Brownian bridge between two interior points over dt, a conservative
missed-ball-exit bound is

```
P(missed exit | endpoints) <= min(1, 2d exp(−2 δ²/(d dt))),
δ = r − max(|start|,|end|).
```

The linear segment stays at least δ from the sphere by convexity. Exiting
requires some coordinate fluctuation of the centered bridge to exceed
δ/sqrt(d); the bound follows from the scalar one-sided bridge formula and a
union bound. It becomes useless near the sphere. Thus exact Gaussian grid
values and a small nominal dt do not certify a stopped Brownian path.

## A new Brownian stopping skeleton with exact-law sub-ball clocks

`brownian_sphere_skeleton` implements Brownian walk on **sub-spheres** inside
the outer ball. At current position x choose

```
a = min(ell, .9*(r−|x|)).
```

The sub-ball is strictly inside the outer ball. Brownian exit from its center
has independent uniform direction and clock a²τ₁. This independence follows
from rotational invariance of the joint direction/time law; its Markov
iteration supplies genuine Brownian positions and times at the successive
stopping times. Fixed-radius interior steps are vectorized until their first
approach to the outer boundary; discarded future increments do not alter
the retained law. Adaptive boundary steps follow separately.

The unit exit-time CDF is evaluated using two equivalent series:

```
F(t) = 1 − 2 sum[k>=1] (−1)^(k+1) exp(−k²π²t/2),
F(t) = 2 sqrt(2/(πt)) sum[m>=0] exp(−(2m+1)²/(2t)).
```

The second is the Jacobi transform and prevents small-time cancellation.
Twelve terms suffice to floating-point accuracy at the t=.3 switch. The
inverse CDF uses monotone interpolation on 16,385 logit-spaced probabilities
between logistic(−32) and logistic(32); exterior tail probabilities are
inverted directly rather than clipped. Independent deterministic probes
gave maximum CDF residual 2.714e−7 over 100,003 points and maximum relative
quantile error 5.306e−7 over 10,003 points. These are measured numerical
errors, not a proved uniform likelihood bound.

The last boundary layer stops at gap δ <= 1e−7*r and projects to the outer
sphere, omitting the residual clock. Starting from x, optional stopping gives

```
E residual_time = (r²−|x|²)/3 <= 2rδ/3,
E |true_exit−x|² = r²−|x|² <= 2rδ.
```

These bounds, and their sum over balls, are saved. The residual path may
still wander: a small boundary gap is not a deterministic spatial bound.
The old 6% mean-clock bias at ell/r=.1 disappears in the new 4,096-path clock
control: mean .331967 ± .003220 SE versus 1/3; maximum omitted mean clock
bound 6.666e−8. The observed −0.41% discrepancy is compatible with Monte
Carlo fluctuation and is not evidence for a corrected exact law.

Unknown Brownian portions remain within each sub-ball. Before the final
boundary projection, interpolation has a coupling with uniform spatial error
at most 2ell. This does **not** immediately give a 2ell bound after the
outer-endpoint rotation, concatenation, duration normalization and graph-cell
cutting. In particular the Householder rotation is ill-conditioned when its
two directions coincide. Those later errors need separate control.

`sample_wos_doob` uses this interior skeleton, exact analytic h-weighted outer
sphere exits and the paper's two-leg construction. It accepts the existing
I3/A2 analytic model and the new `A3AnalyticCone`. Numerical angular modes are
deliberately refused by this control API. Straight terminal cone tails retain
the exact marginal inverse-gamma lifetime but approximate their paths. All
cell minima remain polygonal, and **no independent coordinate-minimum
corrections are applied to correlated paths**.

## Coupled exact-grid and minimum controls

For I3, the exact Bessel-bridge sampler supplies both the fine grid and exact
continuous scalar segment minima. The same fine realization is thinned to
q=4,16,64,256,1024 points per graph cell, retaining the random biased junction.
Continuous minima over coarse cells are obtained by merging the exact fine
minima, not by interpolating them. Thus each polygonal graph is compared
directly with a continuum-law graph on the same physical excursion.

For A2 and A3, exact finite-time matrix-bridge distributions are available,
but joint continuous gap minima are not. The paired reference is q=1024,
explicitly still approximate. Neither matrix bridges nor the new WOS method
make finite-grid gap minima exact.

At n=256, the q=256 paired mean-distance changes relative to their references
were:

| Cone | Relative mean-distance change ± graph SE | Edge Jaccard | Largest absolute mean joint-CDF change |
|---|---:|---:|---:|
| I3 | −0.017% ± .112% | .9637 | .00468 |
| A2 × half-line | −0.018% ± .083% | .9662 | .00228 |
| A3 | −0.028% ± .097% | .9643 | .00277 |

These are 64 independent physical paths per row, with paired roots across
resolutions. Mean-distance stability hides several percent local edge
changes. The q=4 exact-grid bias was about −0.4% (I3), −1.0% (A2), −1.1%
(A3); it therefore does not explain the earlier **positive** 2% Doob offset
by itself. This does not prove that minima errors in the old stopped random
walk have the same sign or magnitude as in these exact-grid controls.

## Full new-Doob validation and remaining gate

See `results/phase2c/paths/wos_controls/summary.json` for the 64-path-per-cone
comparison at n=64,256, and the accompanying figures. Paths at both sizes are
the same physical realization. Reference graphs are independent exact I3 or
q=1024 Weyl controls. JSONL records retain minima, edges, degree and pair
histograms, rootwise ball histograms, one- and multitime path probes, inverse-
gamma leg clocks, and outer ball centers/radii for accumulated h diagnostics.

Root-averaged ball fractions are cumulative pair-distance histograms, not an
independent validation law. Across-root ball variances supply an additional
rootwise diagnostic. Histogram TV entries in the paired grid comparison are
mean within-realization histogram changes, not estimates of ensemble-law
total variation. CDF comparisons at several thresholds have no automatic
simultaneous coverage guarantee.

The clock repair is a mathematically justified improvement. A finite small-
sample comparison cannot establish a 1% error bound on the complete graph
law, and joint-minimum or relative corner errors must not be hidden inside
graph Monte Carlo bars. No dimension estimates or generic covariance sweep
are produced by these experiments.

The completed main control has zero failures among 192 paths. At n=256,
the new WOS mean distances are I3 4.21291, A2 4.14486 and A3 4.12231.
Relative to the 64-path matched references their changes and pointwise graph
MC95 intervals are +1.30% [−1.02%,3.62%], −.21% [−2.33%,1.91%], and
+1.27% [−.07%,2.62%]. The corresponding largest joint-CDF differences are
.0389 ± .0387 SE, .0594 ± .0348 SE and .0608 ± .0269 SE. The requested
0.02 joint-CDF accuracy is therefore **not established**. Several pointwise
path/radial diagnostics have 2–3 SE deviations; these are retained as warnings
and are not discarded because other observables agree.

An additional focused experiment uses 24 I3 and 24 A3 paths at ell=.0025,
relative step .05, n=256. It saves the first two full finest paths per cone
as compressed NPZ files, all graph minima/edges/roots, and all realization
seeds. Its independent I3 mean comparison is +3.91% [+.61%,7.21%], whereas
A3 is −.05% [−2.00%,1.91%]. This does **not** demonstrate monotone convergence
or justify a 1% accuracy claim. It is an unresolved small-sample discrepancy.

The same finest paths are coarsened by retaining every second, fourth or
eighth stopping vertex, together with every graph-cell boundary and junction.
This preserves the Brownian skeleton clock, endpoint rotation, normalization
and metric roots. It is a **coupled polygonal interpolation diagnostic**,
not a sample from a coarser-WOS algorithm. The edge Jaccards against the
finest retained path are .974/.952/.935 (I3) and .966/.945/.919 (A3), while
all mean-distance changes stay below .21%, with paired SE .13–.24%.
Thus the lack of local-edge precision persists even with the repaired clock.

For a less variable path diagnostic, condition on the sampled T1,T2 and
junction. The whitened radius is a Bessel bridge of dimension D=2alpha1+2.
Writing H²=1/(T1+T2), its squared-radius conditional mean before junction U is
`(t/U)²H² + D*t*(U−t)/U`, with the corresponding reversed formula after U.
These residuals and inverse-gamma hitting-time KS diagnostics are saved in
`extra_diagnostics.json`. The main A3 t=.25 residual is −.320 ± .124 SE;
the finer independent run gives −.081 ± .252 SE and does not reproduce it.
This reduces concern about that particular deviation but does not certify
the complete path law. No simultaneous significance claim is made.

Figure files under `results/phase2c/paths/figures/` show the separate clock,
local-edge, metric and coupled-interpolation diagnostics.

## Reproduction

Run each command in a fresh output directory; the script refuses existing
directories to preserve checkpoints. Seeds and software versions are saved.

```
.venv/bin/python -m pytest -q tests/test_phase2c_paths.py
.venv/bin/python -m experiments.phase2c_path_validation minima --output results/phase2c/paths/nested_controls --samples 64 --seed 21732001
.venv/bin/python -m experiments.phase2c_path_validation clocks --output results/phase2c/paths/ball_clocks --samples 16384 --seed 21733001
.venv/bin/python -m experiments.phase2c_path_validation sphere_clocks --output results/phase2c/paths/sphere_clock_control --samples 4096 --seed 21735001 --step-size .1
.venv/bin/python -m experiments.phase2c_path_validation wos --output results/phase2c/paths/wos_controls --samples 64 --seed 21736001 --reference results/phase2c/paths/nested_controls/records.jsonl --step-size .005 --relative-step .1 --trace-balls
.venv/bin/python -m experiments.phase2c_path_validation wos --output results/phase2c/paths/finer_wos_interpolation --samples 24 --seed 21737001 --reference results/phase2c/paths/nested_controls/records.jsonl --step-size .0025 --relative-step .05 --trace-balls --cases i3 a3 --n 256 --coarsen-pl --save-fine-paths 2
.venv/bin/python -c 'from experiments.phase2c_path_validation import make_figures; make_figures()'
```
