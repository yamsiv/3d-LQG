# Phase 2B: independent audit and correlated-excursion validation

**Completed 11 November 2026, America/New_York.** Work began on 10 November
and resumed after a usage interruption. Scope: Brownian excursions, graph
construction and intrinsic metric geometry. No Gaussian-field, GMC or LFPP
implementation or comparison was undertaken.

## Decision and completion status

The independent-coordinate implementation survived an independent mathematical,
source and data audit. Its three previous headline estimates were reproduced.
An additional 80 independent large graphs improve Monte Carlo precision but
do not settle correction-model dependence.

The paper's biased interior-ball/Doob construction is now implemented with
analytic harmonics for every two-dimensional wedge and every three-dimensional
birectangular cone, including coordinate permutations. Its path approximation
has been tested against independent Bessel and Weyl controls using individual
and joint minima, local edges/degrees, histograms and balls. These are useful
validation data, but they do not establish a tight, uniform numerical error
budget for continuum correlated graphs.

**Arbitrary-correlated sampling remains unvalidated.** The angular FEM
interpolant was found to have open interior zero patches. A conforming mesh
repair removes those patches, but not the wrong corner vanishing rate or its
large harmonic mean-value defects. Locally normalized sphere sampling with
this nonharmonic function is not the desired exact Doob transform. This is
a specific mathematical/numerical obstruction, not a reason to substitute
the old finite-grid MCMC ensemble.

**Part D's general covariance dimension study was not opened.** No new
covariance-dependent Hausdorff exponents were inferred from the small
validation graphs. The requested stopping rule applies. The full correlated
model is not yet ready for a reliable field/metric comparison program.

## 1. Recovery inventory and preservation

The resumed session inspected Git status, instructions, research logs, source,
tests, saved metadata, raw JSONL records and generated figures. The working
tree contains uncommitted research files; no commit, reset or data replacement
was performed.

Already completed before interruption:

- Independent audit of the original 568-graph dataset and exact constructions.
- Eighty new exact biased I3 graphs: forty at each of n=262,144 and 524,288.
- Reanalysis of the resulting 648 observations and alternative correction fits.
- Analytic Doob sampler, FEM support/mean-value audits and optional corner repair.
- Main Doob validation: 832 independently generated physical paths, each
  observed at two coupled sizes, n=64 and 256.

The new Phase 2B report and log entry were absent. The resumed work completed
missing wedge/permutation and one-factor checks, reused saved bootstrap draws
for additional statistical diagnostics, tightened rejection of defective FEM
support, and completed the handoff. Prior raw datasets and figures were
preserved. The 648 observations consist of the old 568 plus the new 80;
combined/derived files are not additional independent simulations.

## 2. Independent mathematical and implementation audit

Primary authority: Budd–Castro, *Scale-invariant random geometry from mating of
trees: a numerical study*, local `papers/Budd_Castro.pdf`,
[arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1), §§2.5, 3.1–3.4,
4.1–4.2, equations (28), (32)–(45). The audit read these definitions directly
and inspected the implementations and original run metadata.

**Exact I3 and the two ensembles.** An unbiased scalar excursion is a
Bessel-3 bridge, obtained as the norm of a three-component Gaussian bridge
from zero to zero. Coordinates factorize only for the unbiased independent
orthant. For the paper-biased law, the spherical mode is proportional to
`product(s_i)`, `alpha1=3d/2-1`, and its squared-coordinate angular law is
Dirichlet(3/2,...,3/2). Two independent times have law
InvGamma(alpha1,scale=1/2). Conditional on their normalized junction and
height, scalar bridge legs are independent Bessel-3 bridges. Fixed endpoint
orientations are legitimate because rotational invariance at a zero endpoint
makes the entire radial path independent of that orientation.

The paper's bias is proportional to
`integral |X(t)|^(2 alpha1-2) dt` in **whitened cone coordinates**. Original
and biased laws differ at finite volume. No code silently exchanges them.
The new Doob API targets the biased law; finite-grid MCMC remains a separately
labeled approximation to an unbiased discrete-time positive bridge.

**Conditional minima.** For positive scalar bridge endpoints x,y over dt,

```
P(M>z | positive,x,y) =
  [1-exp(-2*(x-z)*(y-z)/dt)] / [1-exp(-2*x*y/dt)].
```

Doob endpoint factors cancel when both endpoints are fixed. The inversion,
conditional independence of segment minima, insertion of the random junction,
and combination of the two pieces of a straddling cell are correct. Thus
independent-coordinate graph minima have the continuous target law, up to
floating-point arithmetic. This claim does not extend to polygonal Weyl or
Doob paths.

**Covariances, adjacency and metrics.** The interface uses
`(a,b,c)=(C12,C13,C23)`, with paper angles
`(alpha,beta,theta_angle)=acos(-(a,c,b))`. Positive definiteness is checked
directly. For generic cell minima, a diagonal i<j satisfies
`max(m_i,m_j)<min(m_(i+1),...,m_(j-1))`. The shared Hamiltonian cycle is
counted once, while coincident diagonals retain multiplicity. Reference and
stack implementations agree with an independent small-graph oracle.
Shortest paths use unit edges on the underlying simple graph, with uniform
independent vertices including self-pairs. No hidden metric normalization was
introduced; multiplication by n powers occurs only in analysis.

**Actual sample independence.** The original 568 and updated 648 records
have unique graph seeds and unique metric seeds, matching their saved
SeedSequence design. Archive hashes and the byte-preserving old concatenation
were checked. Saved means equal their stored histograms. Reconstructing
selected actual graphs from each original run reproduced their root distance
histograms and diameter bounds exactly. Coupled observations of one physical
path at different n or resolutions are explicitly not independent graphs.

An additional analytical biased-law check gives
`E[|X(1/2)|²]=2.9814083642` for biased I3, versus 2.25 for the original law.
Four thousand new biased samples gave 2.94745 ± 0.02041 independent-sample SE.
This checks path values and ensemble distinction, beyond checking drawn
hitting-time metadata.

No bug invalidating the original exact I3 data was found. Detailed derivations,
tests and independent reruns are in
[the audit](../docs/AUDIT_PHASE_2B.md) and
[saved audit evidence](phase2b/audit/original_data_reanalysis.json).

Final review found a metadata-only bug in the new permutation reference
helper: its path values and covariance were permuted correctly, but its
recorded correlated junction direction retained canonical Weyl column order.
The helper is corrected and covered by a regression test. The 128 existing
reference records remain unchanged; a hash-linked
[correction sidecar](phase2b/permutation_metadata_correction.json) supplies
the reordered directions. Graphs, minima and reported comparisons are
unaffected; future angular analyses must use that annotation.

## 3. Reproducing and revisiting the dimension estimates

The old central estimates were independently recovered as:

| Estimator on the old 568 graphs | Recovered value |
| --- | ---: |
| Ordinary log–mean regression, inverse slope | 4.189503 |
| Weighted additive-offset mean fit | 4.833297 |
| Two-stage shifted histogram collapse, inverse scale slope | 5.070283 |

A separate three-parameter nonlinear fit reproduced the offset result, and
multistart histogram optimization changed its dimension by only about 0.00004.
The differences between these estimators are not primarily an optimizer bug.

### New large-size data and computational cost

Before computation, the old benchmark predicted about nine minutes and a
peak memory requirement near its previously observed 0.8 GB. The eighty new
graphs actually took about 152+319=471 seconds for generation and eight-root
metrics. Memory was not remeasured during this run. No old sample was replaced.

The updated counts at n=2^11,...,2^19 are
`64,64,64,64,88,88,88,64,64`. Every realization has eight uniform roots.
The new independent run has base seed 2026111021, with full per-graph seeds,
histograms, metric records and provenance in
[i3_high_extension.jsonl](phase2b/data/i3_high_extension.jsonl).

| Estimator on 648 graphs | Central estimate | Monte Carlo 95% interval |
| --- | ---: | --- |
| Naive ordinary log–mean inverse slope | 4.215 | [4.171,4.245] |
| Weighted additive-offset mean model | 4.910 | [4.643,5.159] |
| Shifted histogram collapse, without final Eq. (45) correction | 4.916 | [4.541,5.440] |

The corrected intervals remain compatible with the paper's printed
4.83±0.06, whose uncertainty is not reinterpreted as a 95% interval.
Compatibility does not identify a unique extrapolation or reproduce the
paper's precision.

### What equation (45) actually corrects

The old additive-offset fit uses `E[d_n]=A*(n/nmax)^p+B`. It is a mean-distance
ansatz, equivalent to a particular relative correction with exponent p. The
paper first finds histogram scales k_n and shifts, then fits

```
k_n = (n/n0)^(-p) * [a + b*(n/n0)^(-delta)],  p=1/d_H,
```

with a near one, small b, and delta of order p. It varies size windows and
correction assumptions, and uses eight batches for statistical checks. The
previous implementation stopped before this final fit.

Task 2B implements and audits a **reference-normalized adaptation**:
`a=1-b`, because the stored reference scale k(n0)=1 is deterministic. This
restriction is stated explicitly and differs from the paper's free a near
one. The deterministic reference entry is omitted from residuals. Generalized
least squares uses the complete bootstrap covariance of the other scales,
including their shared random reference histogram.

With `delta=p`, `|b|<=.15`, the updated result is
**4.856 [4.650,5.123]**, with b≈−.0282. Other explicit assumptions give:

| Eq. (45) assumption | Central inverse power | MC 95% interval |
| --- | ---: | --- |
| delta=.5p, |b|<=.15 | 4.665 | [4.444,5.240] |
| delta=p, |b|<=.15 | 4.856 | [4.650,5.123] |
| delta=2p, |b|<=.15 | 4.969 | [4.732,5.232] |

At delta=.5p, tightening |b| to .05 makes the central fit hit its amplitude
bound; 88.5% of its bootstrap fits hit that bound. Even at .15, 49% hit it.
Those intervals are constrained-model diagnostics, not unconstrained precision
measurements. The delta=p/.15 and delta=2p alternatives do not hit those
bootstrap amplitude bounds. No exponent was constrained to agree with 4.83.

### Correction-model, window and bootstrap sensitivity

All models use the full nine-size data in this comparison:

| Mean correction model | Central inverse power | MC 95% interval |
| --- | ---: | --- |
| A*x^p+B | 4.910 | [4.643,5.159] |
| A*x^p+B+D*x^(-p) | 4.093 | [3.416,5.128] |
| A*x^p+D*x^(p-.1) | 5.399 | [4.990,5.732] |
| A*x^p+D*x^(p-.2) | 4.923 | [4.666,5.140] |
| A*x^p+D*x^(p-.4) | 4.597 | [4.461,4.732] |

Here x=n/nmax. These are different hypotheses, not separate measurements of a
known common asymptotic parameter. The offset fit has chi-square 7.389 on six
residual degrees of freedom; the offset-plus-inverse-power fit has 5.236 on
five. The more flexible fit is not uniquely justified by that small reduction.
Some fixed-delta bootstrap fits have nonpositive leading amplitudes
(0.5–1%); these branches are flagged as nonphysical asymptotic behavior.
Central fits have positive amplitudes and positive Eq. (45) predictions.

Dropping successively smaller sizes gives offset estimates
`4.910,4.674,4.710,4.873,4.420,3.415`; the last four-size fit has very broad
uncertainty and one residual degree of freedom. It is not evidence for a new
small dimension. Repeating the full central histogram-collapse pipeline for
successive windows gives approximately
`4.916,4.950,4.922,4.904,4.919,4.863,4.664`.
The Eq. (45) regression-window study separately holds the full-range collapse
convention fixed; it must not be mislabeled as a full-pipeline refit.

Whole graphs, not bins, target vertices or roots, are bootstrap clusters.
Reference graphs are resampled along with smaller graphs. A naive extra root
bootstrap would double-count root noise already present in whole-record
resampling. Re-estimating the size variance weights in each bootstrap changed
the offset interval from [4.643,5.159] to [4.627,5.171], much less than changing
the correction model. Eight independent batches also show substantial spread.
None of these Monte Carlo intervals covers unknown asymptotic corrections.

The original and cached updated analyses were preserved. The final annotations
reuse saved bootstrap draws rather than rerunning graph experiments:
[audited analysis](phase2b/updated_independent_scaling_audited.json),
[weight audit](phase2b/bootstrap_weight_audit.json).

![Updated fitting and window diagnostics](phase2b/updated_independent_scaling_readable.png)

**Conclusion for I3:** the exact graph dataset is trustworthy. A precise,
ansatz-independent Hausdorff dimension has not been established. The published
value is compatible with several corrected fits, while the naive fit remains
inadequate for that comparison.

## 4. New biased Doob/interior-ball sampler

For C=RRᵀ, work in W=R⁻¹R₊ᵈ. A positive homogeneous harmonic u of degree p
gives `alpha1=p+d/2-1` and
`h(x)=u(x)/|x|^(2p+d-2)`. The new sampler draws the paper's angular m1² law,
two vertex-hitting legs, and performs its reversal/concatenation and duration
normalization. There is no excursion MCMC or burn-in in this construction.

Analytic u is available for every wedge and wedge times a half-line:
`u=s^(pi/alpha)*sin(pi*theta/alpha)` in 2D, multiplied by the positive
perpendicular coordinate in 3D. This covers the entire birectangular
three-dimensional covariance family and its permutations, not arbitrary
three-parameter cones.

Rather than maximal interior balls, radius `eta*distance_to_faces` with
eta=1/2 is used. This leaves the Brownian target unchanged and gives a valid
analytic rejection bound
`h(exit)/h(center) <= (1+eta)/(1-eta)^(d-1)`.
An isotropic bounded walk approximates the interior Brownian path. A
Householder orthogonal map sends its uniform exit direction to an independent
h-weighted exit. Rotational invariance justifies this path mechanism.

Approximation layers are recorded separately:

1. **Spatial and time approximation.** A fixed step length ell has physical
   dt=ell²/d for standard Brownian covariance. The final ball step is clipped
   with a linear time fraction. Both the global step and its maximum fraction
   of a ball radius must shrink; reducing only one does not control all balls.
2. **Vertex cutoff.** The residual hitting time has exact radial law
   `r²*InvGamma(alpha1,1/2)` for analytic h, but its spatial path is replaced
   by a line. The true omitted path can leave the cutoff ball; its chance of
   reaching R>r before zero is `(r/R)^(2*alpha1)`. Localization is probabilistic.
   For the saved I3 fine controls, the maximum normalized tail duration is
   2.53e-5, below one n=256 cell (0.003906); the cutoff-refined value is smaller.
   These omitted tails lie within endpoint cells whose minima already equal
   zero, conditional on the recorded duration. This does not remove interior
   walk/clock errors or justify all future sizes.
3. **Minima.** Every polygonal vertex is retained and graph boundaries are
   inserted. This computes minima of that polygonal approximation. It does
   not simulate missing joint Brownian minima. Increasing inserted grid
   density alone cannot fix them.
4. **General FEM mode.** A supplied numerical mode introduces angular,
   spectral, boundary and nonharmonic-kernel errors in addition to the above.

Failed attempts are saved without silently drawing replacements. Numerical
mode/covariance mismatch and unresolved zero-support facets now raise errors
before path sampling. Roundoff positivity adjustments and zero proposals are
recorded; a positive floor is never substituted for the eigenmode.
See [sampler methods](../docs/DOOB_SAMPLER.md).

## 5. Independent cross-validation and observable error budget

The main experiment used 64 coarse, 64 fine, 32 very fine and 256 independent
reference paths **per case**, for I3 and the A2-times-half-line cone. Each path
was observed at n=64 and 256, giving 1,664 rows from 832 independent paths.
There were no failed attempts. Settings (global step, relative step, cutoff):

- Coarse: (.02,.20,.01).
- Fine: (.005,.10,.003).
- Very fine: (.0025,.05,.0015).
- References: exact biased Bessel values/minima for I3; exact biased Weyl
  values with 256 samples per cell for A2.

The diagnostics include hitting times, path coordinates/radii at three times,
marginal and adjacent-cell minima, cross-coordinate joint minima, several
CDF thresholds, degree distribution/variance, multiedge fraction, edge-span
distribution, pair-distance histograms and six ball radii. Uncertainty uses
independent physical paths; the coupled size rows are not pooled as replicates.

At n=256:

| Case and method | Mean pair distance ± graph MC SE | Mean simple degree ± SE |
| --- | ---: | ---: |
| I3 fine Doob | 4.2663 ± .0320 | 7.4918 ± .0075 |
| I3 very fine Doob | 4.2507 ± .0432 | 7.4792 ± .0094 |
| I3 exact reference | 4.1903 ± .0159 | 7.4911 ± .0040 |
| A2 fine Doob | 4.1153 ± .0290 | 7.5570 ± .0074 |
| A2 very fine Doob | 4.1515 ± .0481 | 7.5496 ± .0090 |
| A2 Weyl reference | 4.1134 ± .0129 | 7.5528 ± .0035 |

Close mean degrees and distances do not certify joint minima. For fine I3,
the joint minimum CDF at a stored threshold differed by +.0764 with combined
SE .0301. Across the probed features, joint/marginal minimum CDF precision is
only on the order of 0.1 probability units. The very fine run moves some
discrepancies toward zero but remains statistically broad. These checks do
not establish percent-level or uniform distributional error control, and
individual feature discrepancies are not interpreted as formal discoveries
after many correlated comparisons.

For fine I3, the mean ball fraction at radius 4 differs from the exact
reference by −.01894 ± .01011 combined SE; for A2 the difference is
−.00119 ± .00928. The edge-span CDF at span/n=.1 differs by +.00488 ± .00259
for I3 and −.00135 ± .00257 for A2. These help describe the finite-observable
budget, but the uniform-root mean ball fraction is exactly the cumulative
pair-distance law. It is not an additional independent validation statistic;
rootwise ball-volume fluctuations supply further information.

Such rootwise variation was reconstructed from the saved minima and original
roots. At radius 4, mean conditional-root variance of ball fractions was
.02778 ± .00164 graph SE for fine I3 and .02745 ± .00082 for its reference;
A2 values were .02402 ± .00144 and .02496 ± .00071. These are additional,
noisy finite-root diagnostics, saved in
[doob_ball_variability.json](phase2b/doob_ball_variability.json).

A separate 32-path reference study generated Weyl values at 1,024 samples per
cell and thinned the same paths to 128 and 256. The coarse-minus-fine
(256 minus 1,024) difference was +.00554 ± .00840 paired SE in mean distance
and −.00513 ± .00224 in mean degree,
and the largest measured joint-minimum CDF change by about .00258 ± .00049.
Edge Jaccard agreement was about .966: local edge sets remain more sensitive
than the mean metric. This residual reference discretization is distinct from
the Doob walk errors and graph Monte Carlo errors.

Missing wedge/permutation checks and one-factor refinements were completed
after recovery, with new seeds and separate outputs. The permutation check
at n=128 had a mean-distance difference .0309 ± .0327 SE between two analytic
representations, consistent with symmetry at that precision. It is a finite
observable check, not a proof based on a nonsignificant test. A surprising
cutoff-only I3 result was retained and independently repeated rather than
used to select a sampler configuration. The full outcome, including both
runs, is in the independent validation note.

Specifically, the original 32-path cutoff-only mean was 4.0571, whereas its
independent 64-path repeat gave 4.1884 ± .0289 SE, close to the reference.
The baseline fine repeat instead gave 4.2776 ± .0309, agreeing with its
original fine run. Pooling the two independent fine groups gives about
4.2720 ± .0221 versus exact 4.1903 ± .0159, a roughly 2% offset with about
three combined SEs (difference MC 95% interval about [.0282,.1351]). This is
evidence to retain possible finite-step bias,
not to declare a precision validation pass. The cutoff-only anomaly did not
replicate; both outcomes remain saved.

The complete error budget and comparisons are in
[doob_error_budget_complete.json](phase2b/doob_error_budget_complete.json),
[validation records](phase2b/doob_validation/records.jsonl), and
[the independent validation note](../docs/CORRELATED_VALIDATION_PHASE2B.md).

![Doob versus independently accessible laws](phase2b/doob_validation/comparison.png)

No general covariance gate was opened on the basis of matching mean distances.

## 6. The general angular-mode obstruction and attempted repair

The old spherical P1 mesh has three corner triangles with all vertices on
the Dirichlet boundary. Its interpolated eigenmode is identically zero on
these **open interior patches**, although the true first eigenmode is strictly
positive. At I3 refinement 5, there are three such triangles among 1,024;
the mode at correlated coordinates (.01,.01,1) is zero. The old eigenvalue
tests and interior checks did not reveal this. The old exact I3 results did
not use the FEM sampler, so this defect does not invalidate that dataset.

A centroid-only enrichment failed: its new degree of freedom connects only
to zero boundary nodes and does not fix the principal mode. A conforming split
of each corner's opposite interior edge and its neighboring triangle adds
connected interior degrees of freedom and removes the open zero patches.
The original mesh/results remain available; enrichment is a separate option.

**Positivity repair is not harmonic accuracy.** At I3 refinement 5 the repaired
mode overestimates the angular value near (.01,.01,1) by about 165%; at
(1e-6,1e-6,1) its relative error exceeds two million percent. The true corner
behavior is quadratic, while a P1 facet gives linear vanishing. Its mean-value
defect at (.02,.02,1) and half-distance interior-ball radius is about −15.4%.
The analytic quadrature control is many orders of magnitude smaller. Thus
these defects cannot be explained by Monte Carlo sampling noise.

For an exact harmonic h, sphere-average h equals center h. For the FEM
interpolant it does not. Sampling a normalized sphere density proportional to
that interpolant therefore inserts state-dependent normalization factors
across the many balls of each leg. The resulting kernel is an approximate
locally reweighted walk, not the desired exact Doob process. A converged
principal eigenvalue or nonnegative nodal values does not bound this path-law
error. Fast cached interpolation changes evaluation cost, not the function
or its accuracy.

The FEM-driven pilot demonstrates that this approximate kernel can be run;
it does not validate its distribution. It has not been used for a covariance
dimension sweep. Diagnostics and attempted repairs:
[original mesh](phase2b/fem_audit/audit.json),
[enriched mesh](phase2b/fem_enriched_audit/audit.json).

![Repaired support still has boundary and harmonic errors](phase2b/fem_enriched_audit/fem_boundary_corrected.png)

## 7. Covariance registry and trustworthy-data table

All requested moderate matrices are positive definite, with condition numbers
between 1 and 3. Their angles are in paper order (alpha,beta,theta_angle),
not the user ordering of correlations. Values marked FEM are approximate
cone spectral information, not graph-dimension estimates.

| Configuration (a,b,c) | Angles, radians | alpha1 | Status for continuum graph geometry |
| --- | --- | ---: | --- |
| (0,0,0) | (1.57080,1.57080,1.57080) | 3.5 exact | **Validated** exact Bessel/minimum generator; asymptotic extrapolation provisional |
| (−.5,0,0) | (1.04720,1.57080,1.57080) | 4.5 exact | **Provisional** continuum minima; validated Weyl grid law and analytic-h controls |
| (+.5,0,0) | (2.09440,1.57080,1.57080) | 3.0 exact | **Provisional** analytic-h implementation available; no new trusted multi-size exponent |
| (−.2,−.2,−.2) | (1.36944,1.36944,1.36944) | ≈4.59593 FEM | **Unreliable for controlled exponents** with present general sampler |
| (+.2,+.2,+.2) | (1.77215,1.77215,1.77215) | ≈2.90014 FEM | **Unreliable for controlled exponents** with present general sampler |
| (+.4,+.4,+.4) | (1.98231,1.98231,1.98231) | ≈2.49268 FEM | **Unreliable for controlled exponents** with present general sampler |
| (−.2,+.1,+.35) | (1.36944,1.92837,1.67096) | ≈3.25927 FEM | **Unreliable for controlled exponents** with present general sampler |

Here “unreliable” describes current numerical inference, not invalidity of the
mathematical covariance model. The full matrices, eigenvalues, mesh refinements,
lambda1 and susceptibility `gamma_s=1-alpha1` are saved in
[the registry](phase2b/covariance_registry.json). Cone spectral exponents are
not Hausdorff dimensions.

No genuine general covariance-dependent geometric trend can be claimed from
this task. In particular, equal-apparent-dimension normalized-shape comparisons
were not performed using unvalidated samples. Near rho=−1/2 was not prioritized.
The old MCMC results and their nonstationary traces are retained as an
explicit alternative approximation, not merged with biased Doob observations.

## 8. Concrete alternatives and next-phase decision

1. **Restrict first to analytic families.** All birectangular cones and 2D
   wedges have known harmonics. Improve walk/clock and joint-minimum refinement
   controls against Bessel and Weyl special cases before a larger restricted
   family study. The exact I3 generator should remain a regression control.
2. **Use boundary-aware angular approximation.** Resolve the correct
   `r^(pi/angle)*sin(pi*theta/angle)` corner behavior using enriched singular
   functions, graded/higher-order meshes or a validated spectral method.
   Require relative boundary and ball mean-value checks, not eigenvalues alone.
   Establish how local kernel defects accumulate over whole legs.
3. **Improve special-cone minima.** Investigate stable killed-cone heat kernels
   and joint Weyl gap-minimum sampling, or a coupled adaptive Brownian
   refinement that quantifies local-edge as well as metric errors.
4. **Treat finite-grid MCMC separately.** If revisited, require dispersed starts,
   longer traces, effective-sample and grid convergence checks, and the correct
   ensemble comparison. Do not reinterpret that approximation as the paper's
   biased law or hide burn-in within Monte Carlo error bars.
5. **Keep statistical model uncertainty visible.** More large graphs can
   reduce sampling error, but cannot select a correction law by themselves.
   Retain all windows, correction exponents, amplitude constraints and
   reference-curve covariance diagnostics. Do not tune them toward 4.83.

The code and feasible validation work are complete to this documented stop.
The general three-parameter covariance metric program remains obstructed by
the angular-kernel error and unresolved joint-minimum precision. The model is
not ready for the intended general covariance field/GMC/LFPP comparison.
Existing exact or well-defined special families can support future separately
authorized work, but no such next stage was started here.

## 9. Reproducibility and audit trail

All new artifacts are under `results/phase2b/`; earlier Task 1 results remain
under their original paths. The environment remains Python 3.14.6, NumPy
2.5.1, SciPy 1.18.1, Matplotlib 3.11.1 and pytest 9.2.0 on macOS arm64.
Sources, configuration, random seeds and package versions are captured in
run metadata. Unit and independent-audit tests were rerun after recovery;
the final suite has **83 passing tests** (4.40 seconds), recorded in
[final_tests.xml](phase2b/final_tests.xml) and the final manifest.

Reanalysis without graph regeneration:

```
.venv/bin/python -m experiments.phase2b_finalize_analysis
.venv/bin/python -m experiments.phase2b_figures results/phase2b/updated_independent_scaling_audited.json
.venv/bin/python -m experiments.phase2b_weight_audit
.venv/bin/python -m pytest -q
```

The finalization and weight-audit scripts refuse to overwrite existing outputs;
the displayed commands document their provenance, not an instruction to
replace completed results. Original updated fits were generated by
`experiments.phase2b_scaling` with 200 bootstrap draws and seed 2026111052.
Raw control metadata records independent path counts, paired resolutions and
failure policies. The final manifest records hashes of old and new datasets
so preservation can be checked independently.
