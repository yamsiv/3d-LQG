# Independent correlated-sampler validation, Task 2B

Handoff date: 2026-11-11. This work audits and tests the biased cone law of
Budd–Castro, arXiv:2207.05355v1, §§3.3–3.4 and 4.1. The original/unbiased
finite-grid MCMC implementation remains separate. No field, GMC or LFPP
comparison was performed.

## Mathematical and numerical targets

For a positive harmonic h, the exit law from an interior ball centered at x
has density h(y)/h(x) relative to uniform sphere area. Positivity and the
harmonic mean-value property are both essential. Sampling an exit proportional
to an arbitrary positive interpolant is not automatically a Doob transform:
its normalizer is Z(x)=average_sphere h, which need not equal h(x).

The new analytic sampler uses exact angular and h-weighted exit distributions
for wedges and birectangular cones. Its Brownian paths inside balls are bounded
isotropic random walks with physical step duration ell²/d, not ell². Clipping
the final segment introduces clock and path errors. Global step size, step size
relative to ball radius, and the terminal cutoff are independent controls.
Adding graph-cell boundaries to a retained polyline does not recover missed
Brownian minima. Validation therefore compares path statistics, individual and
joint minima, local graph statistics, distances, and balls across actual walk
refinements.

## Spherical FEM audit and attempted repair

The existing P1 angular solver has three corner facets whose vertices are all
Dirichlet boundary nodes. Its interpolated eigenmode is identically zero on
these **open interior patches**, although the true principal mode is positive.
At identity covariance and refinement 5, m([.01,.01,1])=0. Eigenvalue convergence
alone did not expose this problem. This did not affect the preceding exact
independent-coordinate or Weyl samplers; they did not use the FEM mode.

An attempted centroid-only enrichment does not repair it: the new degrees of
freedom couple only to zero boundary nodes, so they remain isolated from the
interior principal mode. The optional conforming repair in
`enriched_cone_eigenpair(...,strategy="edge")` instead splits the corner's
opposite interior edge and its neighboring facet. This introduces three
connected interior nodes and removes the open zero patches. The old mesh,
solver and previous datasets are preserved.

Strict positivity still does not control corner asymptotics. For identity at
refinement 5, the enriched mode normalized at an interior reference point is
about 165% too large at [.01,.01,1], and about 26,522 times the true value at
[1e-6,1e-6,1]. P1 interpolation vanishes linearly in this corner region, while
the true orthant mode vanishes quadratically. At refinement 6 the latter
relative error remains enormous. Refinement shrinks the unresolved region;
it supplies no uniform relative-error bound.

For balls of half the distance to the cone faces, the enriched refinement-5
mean-value defect near [.02,.02,1] is approximately −15.4% for the orthant,
−0.86% for the A2 birectangular cone, and −7.25% for the positive-correlation
birectangular cone. The corresponding analytic quadrature defects are below
1e-7. Interior-center FEM defects of about 0.16–0.29% are much smaller but
cannot replace boundary checks.

`FastFEMMode` caches the same ray/facet interpolant using an explicit projected
triangulation; tests compare it to the original evaluator at interior points.
It accelerates diagnostic sampling without asserting harmonicity. It refuses
negative modes rather than clipping a sampling density. The optional enriched
mode remains **provisional**.

Artifacts:

- `results/phase2b/fem_audit/audit.json`, `fem_boundary.png`
- `results/phase2b/fem_enriched_audit/audit.json`, `fem_boundary.png`
- `tests/test_cone_sampling.py`

The original enriched plot inherited an inaccurate zero-patch title from the
un-enriched template. Its data and image are preserved. A corrected replot
from the same saved JSON is
`results/phase2b/fem_enriched_audit/fem_boundary_corrected.png`; it identifies
the repaired support and remaining mode-ratio/harmonic errors correctly.

## Executed matched-law controls

All controls use the **biased** excursion ensemble. The identity reference
uses exact Bessel-3 grid values and exact continuous cell minima. The A2
reference uses Hermitian-matrix bridges with exact grid values and polygonal
joint minima; the main reference has 256 grid segments per graph cell. It is
not mislabeled as an exact continuum graph reference.

For each of I3 and A2×R+, the main study generated 64 coarse paths, 64 fine
paths, 32 very fine paths and 256 reference paths: 416 independent paths per
cone. Each is observed at n=64 and 256, so 1,664 raw rows represent **832
independent paths**, not 1,664 independent samples. Coupled observations at
different graph sizes are used only for validation, with no dimension fit.

| Configuration | Global step | Relative step | Terminal cutoff |
| --- | ---: | ---: | ---: |
| Coarse | .02 | .2 | .01 |
| Fine | .005 | .1 | .003 |
| Very fine | .0025 | .05 | .0015 |
| Relative-only refinement | .005 | .05 | .003 |
| Cutoff-only refinement | .005 | .1 | .0015 |

The last two configurations used 32 new independent paths per 3D control.
All main and one-factor attempts succeeded; none were replaced after failure.
Additional two-dimensional wedges at angles pi/2 and pi/3 and a coordinate
permutation of A2×R+ were tested at n=128, each with 32 coarse, 64 fine, 16
very fine and 128 matching reference paths. These 720 paths also had zero
failures. The relevant analytic spectral parameters are alpha1=2 and 3 for
the two wedges, and 4.5 for A2×R+; I3 has alpha1=3.5.

Each record retains the original path seed, parameters, hitting times, angular
junction, raw cell minima, path probes, graph and root diagnostics. The output
includes individual normalized minimum moments/CDFs, cross-coordinate joint
minimum CDFs and products, adjacent-cell joint minimum diagnostics, degree
distributions, multiplicities, edge spans, pair-distance histograms and ball
profiles. Minima are normalized by sqrt(t(1−t)) at cell centers for comparison;
endpoint cells are excluded from these minimum diagnostics.

## Numerical error budget

`doob_error_budget_complete.json` resamples whole independent paths within each
arm, using 500 graph-cluster bootstrap replicates. For each finite vector of
CDF events it reports the maximum absolute observed difference and the 95th
percentile of the centered bootstrap sup-norm error. These are Monte Carlo
precision statements per observable family; they are neither certified
discretization bounds nor simultaneous tests over every family.

The following figures use the original 64-path fine arms at n=256. Entries
are **observed sup difference / centered MC 95% radius**, in probability units:

| Diagnostic family | I3 versus exact reference | A2×R+ versus q=256 reference |
| --- | ---: | ---: |
| Pair-distance CDF | .0203 / .0206 | .0043 / .0187 |
| Degree CDF | .0059 / .0077 | .0043 / .0074 |
| Marginal minimum CDF events | .0526 / .0791 | .0593 / .0738 |
| Cross-coordinate joint minimum events | .0764 / .0726 | .0480 / .0698 |
| Adjacent-cell joint minimum events | .0554 / .0815 | .0582 / .0755 |

Thus local degree distributions are probed at roughly percent-level precision,
but these experiments do **not** certify percent-level joint-minimum accuracy.
The finite event grid also cannot certify the law of every minimum ordering
that determines graph edges. Failure to reject a distribution difference must
not be interpreted as an equivalence certificate.

At n=256 the fine I3 mean pair distance is 4.26632±.03195 graph SE, versus
4.19033±.01593 for the exact reference, a difference .07599 with 95% Monte
Carlo interval [.00601,.14597]. The A2 fine value is 4.11530±.02903 versus
4.11335±.01285; difference .00195, interval [−.06027,.06416]. The very-fine
arms are smaller and noisier; they do not establish monotone convergence.

### Independent repeat of a surprising cutoff contrast

The first 32-path I3 cutoff-only arm gave 4.05711±.03033 SE, substantially
below the reference; the relative-only arm gave 4.19057±.03465 SE. This
surprising contrast was not discarded. An independently seeded repeat used
64 paths at each of fine and cutoff-only settings:

| Arm | Graphs | Mean distance | Graph MC SE |
| --- | ---: | ---: | ---: |
| Original fine | 64 | 4.26632 | .03195 |
| Independent fine repeat | 64 | 4.27764 | .03086 |
| Original cutoff-only | 32 | 4.05711 | .0303 |
| Independent cutoff repeat | 64 | 4.18835 | .02894 |
| Exact reference | 256 | 4.19033 | .01593 |

The extreme low cutoff result did not reproduce, while the fine arm's positive
offset did. Pooling original and repeat fine paths gives 4.27198 and a
reference difference .08165 with MC 95% interval [.02821,.13510], about 1.95%
relative difference. Pooled cutoff paths give 4.14461, difference −.04572
with interval [−.09992,.00848]. Pooled arms reuse the original independent
paths and must not be counted as additional simulations. Multiple diagnostics
were inspected, so isolated small p-values do not establish a unique source
of bias. The repeated fine offset nonetheless prevents claiming precision
validation of this walk discretization.

For analytic h the remaining radial hitting time is sampled exactly
conditionally on the stopping point. Every saved omitted-tail duration lies
inside its corresponding endpoint graph cell. At n=256 the largest normalized
I3 fine duration is 2.52e-5, versus cell width .003906. Because the minimum of
an endpoint cell is already zero, the omitted tail's spatial wiggles cannot
alter its cell minimum in these observed realizations. This does not remove
earlier bounded-walk or clock errors. A true tail can leave the cutoff radius;
the cutoff is not a deterministic spatial bound. For analytic h its probability
of reaching radius R before zero is (r/R)^(2 alpha1). This argument does not
validate an FEM kernel or its radial-tail approximation.

### Reference-grid error and local edges

Thirty-two additional A2×R+ reference paths at n=256 were generated at q=1024
and thinned to nested q=128 and 256 grids. With identical roots, q=256 minus
q=1024 changes mean distance by .00554±.00840 paired SE, mean degree by
−.00513±.00224, and the largest observed joint-minimum-CDF event by
.002584±.000488. Edge Jaccard agreement is .9656. Consequently even this
fine reference has detectable local discretization error, although much smaller
than current unpaired minimum-distribution Monte Carlo uncertainty. Stable
mean distances do not mean all edges have converged. No certified continuum
error is inferred from only these grid levels.

### Coordinate labels and ball variability

The permutation experiment moves the correlated pair from coordinates 1,2 to
2,3. The fine n=128 graph mean differs from the original-label fine arm by
.03093±.03270 SE; pair-distance CDF sup difference is .0081 with centered
MC 95% radius .0230. Original-label n=128 graphs were reconstructed exactly
by aggregating saved n=256 cell minima. Coordinate columns were reordered back
before comparing minimum CDFs. These observations support label equivariance
at present precision, not equality proven by a statistical test.

A metadata-only issue was found during final review of the new permutation
reference helper: it permuted path values and covariance correctly but left
`junction_direction_correlated` in canonical Weyl column order. The helper is
fixed and an independent regression test compares metadata to the actual
junction value. The 128 existing reference records are preserved; their
recoverable annotation is `permutation_metadata_correction.json`, with the
source hash and corrected directions. No graph, minimum or reported
comparison changes. A future angular reanalysis must apply that annotation.

The recorded hitting-time diagnostics are not uniformly perfect pass/fail
tests. Fine two-dimensional pi/2 and pi/3 wedge KS p-values are .481 and
.128; the permuted A2 fine arm has p=.033, and the original coarse I3 arm
has p=.041. Many dependent features were inspected. These values and the
finite-step approximation prevent interpreting isolated p-values as either
proof of correctness or definitive bug discoveries. The pi/2 wedge's
inverse-gamma shape is two, so its hitting time has infinite variance and a
small-sample mean is particularly unstable.

Mean uniform-root ball profiles are cumulative pair-distance histograms, so
they are not independent evidence for validation. To add a distinct diagnostic,
graphs were reconstructed from saved minima and their original eight roots
reused to estimate within-graph root variation of ball fractions at radii
2,3,4. At r=4 the mean conditional-root variance was .02778±.00164 graph SE
for fine I3 and .02745±.00082 for its reference; A2 values were
.02402±.00144 and .02496±.00071. These are noisy finite-root diagnostics.

## Validation status and remaining obstruction

| Construction/configuration | Status | Reason |
| --- | --- | --- |
| Exact independent biased/unbiased sampler | Validated baseline | Analytical grid and minima law; independent audit/tests |
| Weyl exact grid-value sampler | Validated grid law | Joint minima still polygonal; measured residual reference-grid error |
| Analytic-h walk-on-balls, I3/A2/wedges/permutations | Provisional numerical sampler | Mathematically correct target; finite-step discrepancy and insufficiently tight minimum-law error budget |
| Original un-enriched FEM h | Unreliable for sampling | Open interior zero patches |
| Enriched FEM h for arbitrary C | Provisional, dimension-study gate closed | Positivity repaired, relative corner accuracy/harmonicity uncontrolled |
| Existing general finite-grid MCMC | Separately labeled provisional alternative | Unbiased finite-grid law; previous mixing/grid issues persist |

No three-parameter production sweep or covariance-dependent dimension catalogue
was generated. A trustworthy next sampler needs boundary-aware relative
eigenmode control and a bound on accumulated kernel error, plus tighter walk
and joint-minimum refinement against exact special cases. Exact or independently
simulable families remain preferable while that mathematical/numerical problem
is unresolved.

## Artifacts and reproduction

Raw data and metadata are under:

- `results/phase2b/doob_validation/`
- `results/phase2b/doob_extra_controls/`
- `results/phase2b/doob_one_factor/`
- `results/phase2b/doob_tail_repeat/`
- `results/phase2b/weyl_minima_budget/`

Derived budgets are `doob_error_budget_complete.json`,
`doob_ball_variability.json` and `doob_tail_budget.json` in `results/phase2b`.
`permutation_metadata_correction.json` preserves the diagnostic metadata fix.
Previous raw data and figures were not replaced. Main and extra validation
directories include comparison figures. For fresh outputs:

```bash
.venv/bin/python -m experiments.validate_doob --output results/reproduction/doob_main
.venv/bin/python -m experiments.validate_doob --cases i2 a2_2d a2_permuted --sizes 128 --graphs 64 --coarse 32 --reference 128 --veryfine 16 --seed 2026111161 --output results/reproduction/doob_extra
.venv/bin/python -m experiments.validate_doob --cases i3 a2 --sizes 256 --graphs 32 --one-factor --seed 2026111162 --output results/reproduction/doob_one_factor
.venv/bin/python -m experiments.validate_doob --cases i3 --sizes 256 --graphs 64 --tail-repeat --seed 2026111164 --output results/reproduction/doob_repeat
```

The Weyl budget and derived reanalysis scripts have explicit input/output
paths in their source. Saved metadata preserve seeds, actual parameters,
source hashes and software versions. No failed attempts were hidden or redrawn.
