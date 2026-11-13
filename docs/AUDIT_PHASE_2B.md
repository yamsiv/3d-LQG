# Independent mathematical and data audit for Phase 2B

Audit date: 10 November 2026, America/New_York. This audit was conducted from
the source, original paper, saved JSONL records, and experiment metadata before
relying on the previous report's conclusions. The primary authority is
Budd–Castro, `papers/Budd_Castro.pdf`, arXiv:2207.05355v1, especially §§2.5,
3.1–3.4 and 4.1–4.2, equations (28), (32)–(45). The repository instructions,
research context/log, previous report, sampler/metric code, tests and raw
combined independent-coordinate data were inspected directly.

## Audit outcome

No mathematical bug was found in the independent-coordinate excursion laws,
conditional minima, covariance/angle transformations, generic-minimum graph
construction, or unweighted metric calculations. The old general-covariance
MCMC target is correctly identified as a **finite-grid positive Gaussian
bridge**, which is not the desired continuous excursion at a fixed grid size
and finite burn-in. Its numerical error was not controlled in the old runs.
The earlier report's general-covariance stopping decision was appropriate.

The three headline independent-coordinate estimates can be reproduced from
the original saved data. The corrected value 4.833 is strongly conditional on
an additive-offset mean ansatz and is not the paper's complete correction
procedure. Compatibility with the printed 4.83±0.06 is useful but cannot
validate the extrapolation or be treated as a precision replication.

## 1. Excursion construction and the two laws

An ordinary scalar duration-one Brownian excursion has the Bessel-3 bridge
law, so the norm of a three-component Gaussian bridge 0→0 supplies its exact
finite-dimensional distributions. In the independent orthant, unbiased
coordinates factorize. This use of Gaussian bridges is valid because norms
are taken in the scalar excursion construction; it does not replace a vector
orthant excursion by an unconstrained correlated bridge.

For the biased law, the paper's positive spherical mode is proportional to
`prod_i s_i`. Its homogeneous degree is d, giving
`alpha1=3d/2-1`. Uniform orthant-sphere directions have squared coordinates
Dirichlet(1/2,...,1/2); weighting by `prod_i s_i²` gives
Dirichlet(3/2,...,3/2). The implemented angular law is therefore correct.
The two hitting times are independently `InvGamma(alpha1,1/2)`.

Given the common direction and hitting times, the h-transform endpoint
factors cancel in bridge probabilities. Killed Brownian bridges in the
orthant factorize into scalar positive bridges, and taking an endpoint to the
vertex yields Bessel-3 bridges. A Gaussian vector bridge with one zero
endpoint and the other fixed at `(height,0,0)` gives the required radial
bridge: rotational invariance at the zero endpoint means fixing the terminal
orientation does not alter the radial path law. Reversing the second such
construction is equally valid.

Thus the implemented conditional junction U and height H construction follows
the paper's biased law. Under that law the coordinates are independent
**conditional on the junction**, and generally dependent after mixing over
it. The original and biased finite-volume ensembles are not interchangeable.
The bias is evaluated in whitened-cone coordinates for nonidentity covariance.

An additional independent analytical check concerns the biased midpoint
radius. Write `T_i=.5/G_i` for iid Gamma(alpha1) variables. Then
`U~Beta(alpha1,alpha1)` and, conditional on U,
`|H|²=2 U(1-U) Q`, with `Q~Gamma(2 alpha1)` independent of U. Conditional
Gaussian-bridge second moments give, for `t<U`,

```
E[|X(t)|² | U] = 4 alpha1 t² (1-U)/U + 3d t (1-t/U).
```

The other half is its time reversal. Integrating at `t=1/2`, `d=3`,
`alpha1=3.5` gives **2.9814083642**, versus **2.25** for the unbiased law.
Four thousand newly generated biased samples with seed 41415 gave
**2.94745 ± 0.02041** independent-sample SE, consistent with this formula.
This tests the sampled path values and the distinction between laws, rather
than only checking sampled hitting-time metadata.

Absolute continuity of the two excursion laws does not, by itself, prove
equality of every asymptotic metric exponent. The previous documents
correctly retained this distinction as part of the paper's scaling motivation.

## 2. Conditional minima and graph construction

For positive endpoints x,y over duration dt, a Bessel-3 bridge is a killed
one-dimensional Brownian bridge after its fixed-endpoint Doob factors cancel.
Reflection therefore gives

```
P(M>z | x,y,positive) =
 [1-exp(-2(x-z)(y-z)/dt)] / [1-exp(-2xy/dt)].
```

The implemented quadratic inversion is correct, including its stable smaller
root. `expm1`/`log1p` handle small probabilities. Endpoint-zero segments have
minimum zero. Conditional on the skeleton and junction, bridge subsegments
are independent, and scalar coordinates factorize; the vector of sampled
cell minima is consequently exact for the independent original/biased laws,
up to floating-point arithmetic. The junction is explicitly inserted and a
cell containing it combines both subsegment minima.

The paper's horizontal segment condition reduces to
`max(m_i,m_j)<min(m_(i+1),...,m_(j-1))` for nonneighboring cells with generic
distinct interior minima. Both endpoint-cell minima are zero and are cyclic
neighbors; the boundary edge is inserted separately. The reference and
optimized stack construction agree with the independently coded literal
oracle for every interior-rank ordering at n=3,...,8. Each coordinate
contributes n−3 noncrossing diagonals. The common cycle has multiplicity one;
coincident coordinate diagonals preserve parallel-edge multiplicity.

For polygonal approximations, lexicographic resolution of equal cell minima is
an intentional additional model approximation. It can break some finite-grid
symmetries (for example time reflection through its index tie convention).
Passing triangulation invariants after resolving ties does not certify the
continuous excursion law. This is material for MCMC and Weyl polygonal data,
but the old independent-coordinate production records have zero such ties.

## 3. Covariance, angles and coordinate labels

The code follows the user's `(a,b,c)=(C12,C13,C23)` ordering, rather than the
different `b,c` labels in the historical research context. Paper equation (28)
gives `(alpha,beta,theta_angle)=acos(-(C12,C23,C13))`. Cholesky validates strict
positive definiteness. The geometric cone is `W=R^-1 R_+^d` for `C=RR^T`;
rows of R are inward face normals and have unit norm for a correlation matrix.
The transformations round-trip correctly on unequal-correlation examples.

Permuting the columns of a **given** cell-minimum array leaves the union graph
and multiplicities exactly unchanged, while permuting coordinate provenance.
This deterministic symmetry was independently tested. It is distinct from
testing whether a numerical correlated sampler has the correct law after
permuting its input covariance; the latter remains a required sampler check.

## 4. Metrics and sampling independence

The sparse searches use a symmetric adjacency with unit weights, so a directed
search on that symmetric matrix returns the correct undirected distances.
List BFS agrees with independent small-graph metric controls. Parallel edges
of unit length cannot change an unweighted shortest path; they do change
conductances for random walks/resistance, which are separately labeled.

A uniform root and all targets estimate the law of independent uniform
vertices, including self-pairs. No extra factor of n or edge multiplicity is
inserted in the reported raw graph distances. Every old root histogram sums
to n; pooled histograms sum to `8n`, have eight zero distances, and recover
the saved graph mean exactly. Ball volumes are cumulative root histograms.
The eccentricity/triangle-inequality diameter bounds are certified finite-graph
bounds, not exact large-graph diameters or bounds on a limiting exponent.

The combined I3 data contain **568 distinct graph seeds and 568 distinct
metric seeds**, with no overlap between those sets. They are a byte-preserving
concatenation of the 448-record original run and the separately seeded
120-record high-size run; input hashes match the saved aggregation metadata.
Counts per size are `64,64,64,64,88,88,88,24,24` for n=2^11,...,2^19.
All records say biased law, exact minima, and no MCMC. Two actual graph and
distance realizations, one from each original run, were regenerated from
their saved seeds and matched the saved root lists, histograms and degrees.

Independent pseudorandom streams are a conventional reproducible Monte Carlo
design, not a mathematical proof of stochastic independence. There is no
evidence of duplicate production samples. The combined file must not be
counted as additional graphs on top of its two source files. Likewise nested
Weyl resolutions are the same paths, not independent graph realizations.

## 5. Independent reanalysis of the three reported exponents

Using the original raw records, direct log regression of per-size graph means
gives **4.189502624088149**. A separate three-parameter SciPy least-squares
optimization, rather than production's profile over p, fits

```
mean(n) = A*(n/nmax)^p + B
```

and gives `A=30.14972`, `p=0.20689809`, `B=-2.13999`,
**1/p=4.83329746**, chi-square **4.95003398** on six residual degrees of
freedom. This agrees with the saved profile estimate **4.83329714** to
optimization tolerance. The existing two-stage histogram collapse was rerun
from raw graph histograms and gives **5.07028326008**, as reported.

An independent multistart check used first-pass shift starts
`-3,0,1,5,10,18`, selecting by residual rather than a target exponent. It gave
**5.07023933** and common shift **3.14240687**, a negligible change in the
central estimate. Extremely large initial shifts can lead to a zero-overlap
stationary solution with very poor residual; successful termination alone is
not sufficient to identify a good nonlinear fit.

The additive-offset fit is **not equation (45)** of Budd–Castro. Their
procedure first obtains shifts/scales by collapsing the full distance
histograms, averages shifts, refits scales, then fits

```
k_n = (n/n0)^(-1/d_H) [a+b*(n/n0)^(-delta)]
```

while varying windows/correction assumptions and requiring `a≈1`, small b,
and delta of order `1/d_H`. They estimate statistical errors using eight
independent batches and assess fitting systematics separately. The old code
implements the two-stage collapse but only a pure log slope of its scales;
it explicitly did not fit the final correction ansatz. A correction to the
mean and a correction to the full histogram scale can be related under
additional shape/shift assumptions, but they are not interchangeable fits.

The original offset-model windows illustrate the sensitivity:

| Smallest n | Offset-model inverse exponent | Saved MC 95% interval |
| ---: | ---: | --- |
| 2,048 | 4.8333 | [4.5266,5.1330] |
| 4,096 | 4.5380 | [4.1972,4.9239] |
| 8,192 | 4.5697 | [4.1140,5.1710] |
| 16,384 | 4.8144 | [3.9582,5.9036] |
| 32,768 | 4.4716 | [3.4507,6.4440] |
| 65,536 | 3.7064 | [2.4849,7.3266] |

Short windows have too little information to identify three parameters
reliably. A good full-window residual does not establish the ansatz or
asymptotic stability. The proximity of 4.833 to the printed paper value is
not independent evidence for that particular correction model.

## 6. Bootstrap interpretation

Resampling whole graph records within size is appropriate for the actual
design. Each record includes its finite set of random roots, so graph-cluster
resampling captures both graph variability and the finite-root contribution
to that estimator. Treating all target distances as independent would greatly
understate uncertainty; the code does not do that. Adding another naive
root bootstrap on top of complete-record resampling can double-count root
noise and is not automatically an improvement.

For histogram collapse, resampling the reference-size graphs as well as the
smaller-size graphs correctly propagates the common-reference uncertainty.
The bins and pair distances remain correlated; least-squares collapse is a
diagnostic loss, not an independent-bin likelihood. Its graph bootstrap does
not require treating bins as independent.

The original offset bootstrap holds the original estimated size weights
fixed while resampling graphs. This is a conditional weighting choice and
does not account fully for uncertainty in those variance weights. All
reported bootstrap intervals remain conditional on the correction model,
fitting window, optimizer choices and retained sizes. They do not cover
unknown asymptotic corrections or, for polygonal/general-C data, sampler
systematics. Dropping negative bootstrap slopes would require reporting the
fraction; the old main mean fit had zero such slopes.

## 7. Mathematical review of the new interior-ball sampler

`src/doob_excursions.py` was reviewed separately from implementation. For its
analytic wedge and wedge-times-half-line cones, the orthonormal wedge basis,
positive harmonic polynomial, Kelvin transform, and spherical angular density
are correct. In d=3, writing wedge power k, squared perpendicular height has
`Beta(3/2,k+1)` law. Its rejection sampling matches `m1²`.

For a strict interior ball of radius `r=eta*distance_to_boundary`, positive
harmonic Harnack gives
`sup h(y)/h(x) <= (1+eta)/(1-eta)^(d-1)`. This supplies a valid rejection
envelope. The angular exit law is proportional to h on the sphere, with
normalizer h(x) for an exact harmonic function. A Brownian path started at
the ball center has an exit time independent of its uniform exit direction.
Rotating/reflection-mapping its endpoint to the independently sampled
h-weighted direction therefore gives the correct conditional path mechanism
in the Brownian limit. Householder reflections are legitimate orthogonal maps.

A fixed-length isotropic step ell has covariance `ell² I/d`, so the new
physical time increment `ell²/d` is correct. Its clipped final-step time is
explicitly approximate. The radial h-transform generator is
`(1/2) f''+(1-2 alpha1)/(2r) f'`; its residual hitting time from radius r is
`InvGamma(alpha1,scale=r²/2)`. The sampled tail time is therefore correct for
an exact cone h, while replacing its spatial path by a straight segment is
another controlled-by-refinement approximation.

Systematic checks must refine **global step size, relative step/radius, and
origin cutoff separately**. Since the implementation caps ell by a fixed
fraction of a small ball's radius, reducing global step size alone does not
remove the relative final-step error in every small ball. Tail normalized
duration should be much smaller than a graph cell, and adding graph-grid
points by interpolation does not recover unresolved Brownian minima.
Rejection/path budget failures must be reported and must not silently trigger
successful-only resampling, which would select the excursion ensemble.

For a numerical FEM angular interpolant, local exit normalization is not an
exact Doob transform because the interpolated h is not harmonic. The code
correctly avoids applying analytic Harnack to it and uses a global interpolant
bound instead. The original all-boundary corner triangles also make the
piecewise-linear eigenmode vanish on an open interior patch, despite the true
first mode being strictly positive. Interior enrichment may repair support,
but cannot alone certify harmonicity, boundary relative error, the radial
law, or the resulting excursion ensemble. A passing eigenvalue convergence
test and positive nodal values are insufficient for a sampler validation gate.

## Reproducible audit artifacts

- `tests/test_audit_phase2b.py`: six new archival, independent-regression,
  regeneration, symmetry and analytical biased-moment tests; **6 passed in
  1.84 seconds** in the isolated audit run.
- Selected pre-existing covariance, excursions, adjacency, metrics and
  independent-oracle tests: **38 passed in 2.02 seconds**.
- `results/phase2b/audit/original_data_reanalysis.json`: source data hash,
  seed/count audit, independent central fits, multistart check and old window
  results. Original data and figures were preserved.

These conclusions audit the completed code and old data. Numerical acceptance
of the new general sampler must additionally use the independently executed
path/minimum/edge/metric cross-checks in the Phase 2B report.

## 8. Continuation and final checks, 11 November 2026

The updated independent data add 80 new graphs, 40 at each of the two largest
sizes. The combined 648-record inputs have 648 distinct graph seeds and 648
distinct metric seeds; both input hashes match their recorded provenance. The
largest-size counts are now 64 at n=262,144 and 524,288. Old raw records and
the previous report were preserved.

The new saved analysis gives naive OLS inverse slope **4.21458**, additive
mean-offset inverse exponent **4.91014 [4.64347,5.15922]**, and
offset-plus-inverse-power inverse exponent **4.09319 [3.41618,5.12764]**.
Fixed multiplicative correction delta=0.1,0.2,0.4 gives **5.39853,4.92303,
4.59687** respectively. These values reinforce model sensitivity rather than
isolating a unique dimension. The intervals here are the saved conditional
Monte Carlo intervals, not error bars covering choice of model.

The new equation-(45) implementation was independently reviewed. It uses
`a=1-b` so the deterministic reference scale remains exactly one, and
`delta=(delta/p)*p` with preselected ratios. Omitting the deterministic
reference from GLS residuals is correct. The whitening matrix constructed
from the full bootstrap scale covariance has the correct orientation and
accounts for the shared-reference correlations. An additional audit test
compares it to an independent Cholesky-whitened optimization with non-diagonal
covariance and perturbed synthetic data. This normalization is an explicitly
restricted adaptation of the paper's free-a-near-one fit, not a literal
reproduction of every fitting choice.

On updated data, the ratio=1, `|b|<=.15` reference-normalized fit gives
**4.85594 [4.65033,5.12285]**, with b≈−0.02824. The ratio=.5/.15 and
ratio=2/.15 choices give **4.66529** and **4.96886**. No central reported
mean fit has a nonpositive leading amplitude or an exponent at its optimizer
bound. No central equation-(45) fit has nonpositive predicted scales or an
exponent at its bound. However, at ratio=.5, bootstrap b bounds are reached
in **88.5%** of replicates for bound .05 and **49%** for bound .15; these
intervals remain substantially constrained by the correction assumption even
where the central value is interior. The ratio=1/.15 and ratio=2 choices had
zero b-boundary bootstrap fits in this run.

The equation-(45) size-window diagnostics reuse the full-data collapsed scales
and their common shift, then restrict the regression window. They measure
**regression-window sensitivity at fixed collapse convention**. They do not
replace a complete re-estimation of shifts/scales after dropping each size.
The GLS covariance is estimated from bootstrap draws and held fixed in later
fits, so its estimation error is not fully propagated.

Final source review confirms guards against a supplied eigenpair belonging to
the wrong whitened covariance cone, nonfinite/zero h at an interior-ball
center, and unrecorded negative-coordinate roundoff corrections. Analytic
omitted-tail maximum estimates use
`P_r(sup radius>=R)=(r/R)^(2 alpha1)` for R>r. A sampled residual time is exact
for the analytic radial law, but a straight spatial tail is still approximate
and the true omitted tail can leave its cutoff ball. FEM runs correctly do
not inherit the analytic tail bound or an exact-Doob claim.

The original P1 zero-corner patch is a genuine mathematical sampling defect,
not just slow code: all-boundary facets have a mode identically zero on an
open interior set. Conforming enrichment repairs that support issue but
still vanishes linearly near a corner where the true mode generally has a
higher corner power. Thus its relative error can diverge arbitrarily close
to the corner. Whether such regions are sufficiently rare for a particular
observable requires a demonstrated error budget; positivity and central
agreement alone do not close the general-covariance validation gate.

Post-run checks, including flags missing from the version loaded by an
already-running analysis process, are preserved in
`results/phase2b/audit/updated_fit_flags.json`. They do not modify the saved
analysis or raw samples. The final audit suite contains seven tests, including
the independent GLS objective check.
