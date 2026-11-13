# Phase 3A comparison protocol and parameter conventions

This document separates published results from the finite-resolution comparisons
in this repository. The phase is exploratory: no further Brownian sampler
repair is required before comparing the qualified I3/A2/A3 graph controls.
Agreement would be numerical compatibility of specified finite ensembles,
not identification of scaling limits.

## 1. Primary sources actually consulted

The local PDFs were read directly, including their equations and theorem
statements. Text extraction is only an aid; the square-root in the distance
exponent bound below was checked on a rendered original page.

| Source | Local version | Definitions/results used |
| --- | --- | --- |
| [Budd–Castro](https://arxiv.org/abs/2207.05355v1) | `papers/Budd_Castro.pdf`, v1 | §§3.1–3.4, equations (26), (28), (32)–(43); §4.2 finite-size graph distances |
| [Contreras Hip–Zhuang, distance bounds](https://arxiv.org/abs/2504.09141v1) | `papers/HighDim_Exponents.pdf`, v1 | §1 equations (1.1)–(1.3), Remark 1.1, Theorem 1.3, Definition 1.7, Proposition 1.8, Corollary 1.9 |
| [Contreras Hip–Zhuang, weak metrics](https://arxiv.org/abs/2512.06292v1) | `papers/Weak_ExpMetrics.pdf`, v1 | equations (1.3)–(1.6), Theorem 1.3, Theorem 1.8 and Corollary 1.9 |
| [Ding–Gwynne–Zhuang](https://arxiv.org/abs/2310.03996v5) | `papers/ExpMetric_Tightness.pdf`, v5 | equations (1.4)–(1.10), Proposition 1.1, Theorem 1.2, Remark 1.4 and Lemma 3.7 |
| [Dello Schiavo–Herry–Kopfer–Sturm](https://arxiv.org/abs/2302.02963v1) | `papers/PolyharmonicFields.pdf`, v1 | Lemma 1.4; §§1.1, 3.1 and 4.1; Fourier kernels (18)–(19); Proposition 4.1 |

The latter paper treats several different discrete approximations. Its full
subcritical statement for Fourier approximations must not be confused with
its smaller parameter range for some natural discrete-Laplacian schemes.

## 2. Distinct parameters

| Symbol here | Meaning | Not interchangeable with |
| --- | --- | --- |
| `(a,b,c)=(C12,C13,C23)` | Three Brownian encoding correlations | A scalar field or metric parameter |
| `(alpha,beta,theta_angle)` | Cone angles, with `C12=-cos(alpha)`, `C23=-cos(beta)`, `C13=-cos(theta_angle)` | GMC gamma |
| `lambda1` | Principal spherical Dirichlet eigenvalue | LFPP crossing exponent |
| `alpha1=sqrt(lambda1+1/4)` | Three-dimensional Brownian cone spectral index | Metric dimension |
| `gamma_s=1-alpha1` | Cone/string susceptibility | GMC gamma |
| `gamma` | GMC exponential coupling in unit-log normalization | LFPP xi |
| `xi` | LFPP path-length exponential coupling | gamma or a graph dimension |
| `lambda_cross(3,xi)` | CHZ crossing-distance exponent | DGZ's sequence of normalizers `lambda_n` |
| `Q(xi)` | LFPP distance exponent in DGZ/weak-metric convention | `lambda_cross` |
| `d_gamma` | Hausdorff dimension within the subcritical weak-metric theory | A graph dimension fitted without a parameter identification |

The repository covariance order follows the code. A display in the historical
research context uses a different arrangement of letters; use the explicit
matrix entries whenever recording parameters.

## 3. The field normalization

Use the unit torus `T^3=(R/Z)^3`, Lebesgue volume one, no zero Fourier mode,
and the complex basis `exp(2*pi*i*k.x)`. The polyharmonic paper defines

```
a_d = 2/[Gamma(d/2)*(4*pi)^(d/2)],
Covariance operator = [a_d*(-Delta)^(d/2)]^(-1).
```

Consequently `a_3=1/(2*pi^2)`, and the coefficient of Fourier covariance is

```
S(k) = 1/[a_3*(2*pi*|k|)^3] = 1/(4*pi*|k|^3),  k != 0.
```

If a conjugate pair has complex coefficient `c_k`, require
`E|c_k|^2=S(k)` and independent real/imaginary parts of variance `S(k)/2`;
set `c_-k=conj(c_k)`. In a real cosine/sine basis with basis norm one,
each real coefficient has variance `S(k)`. These are two equivalent
conventions, not an extra factor of two. NumPy's inverse FFT divides by the
number of sites, so stored FFT coefficients must include that factor.

The shell integral `4*pi*r^2*S(r) dr=dr/r` gives variance growth
`log(cutoff)+O(1)` and the covariance logarithm has coefficient one.
The precise bounded infrared correction is fixed by grounding the torus
mean. The finite variance for GMC subtraction is the exact discrete spectral
sum for the chosen filter, not the empirical variance of one realization.

Coupled resolutions retain common Fourier Gaussians, with explicit rejection
of ambiguous Nyquist modes. Sharp spectral cutoff and smooth heat filtering
are different regularizations; they can have the same limiting logarithmic
coefficient and visibly different coarse-grid observables. Field validation
must separate the exact finite spectral covariance from an asymptotic log fit.

## 4. GMC and its moments

For the same field used by LFPP, define finite-grid cell masses

```
m_i = N^(-3)*exp(gamma*h_i - gamma^2*V_N/2),
M = sum_i m_i,  p_i = m_i/M.
```

At every finite cutoff, `E M=1` exactly under the Gaussian model. This is
checked before normalization. The continuum nontrivial range is
`|gamma|<sqrt(6)`. Positive mass moments exist for `0<q<6/gamma^2`;
the Euclidean small-ball exponent is
`zeta(q)=(3+gamma^2/2)*q-(gamma^2/2)*q^2` in that range.
See [Rhodes–Vargas, Theorems 2.11 and 2.14](https://arxiv.org/html/1305.6221),
which state these GMC moment facts and their scaling assumptions.

Thus q=2 is a suitable continuum moment only when `gamma^2<3`.
At gamma=2, q must be below 1.5: a finite-grid sample variance or bootstrap
interval for total mass does not estimate a finite continuum variance.
Heavy-tail under-sampling can make the empirical mean appear too low even
when the formula has expectation one. Report quantiles, maximum contributions
and the finite cutoff alongside any Monte Carlo interval. A q=1.25 diagnostic
is preferable to q=2 for the gamma=2 moment-scaling pilot.

The sampled probability measure is a distinct object. Global total mass and
the deterministic renormalization cancel in `p_i`, but neither should be
discarded from validation. At finite grid scale, `sum_i p_i^2` is the
probability that two independently sampled points coincide. Preserve this
zero-distance atom and test its refinement dependence; removing duplicate
points changes the measure-sampling experiment.

## 5. LFPP definitions, signs and subcritical qualifications

### The lattice and continuum conventions

CHZ equations (1.1)–(1.2) define a nearest-neighbor vertex path length
`epsilon*sum_{v in P} exp(xi*h_epsilon(v))`, including endpoints. The
physical factor epsilon is already present. An edge-trapezoid implementation
with edge cost `epsilon*(w_u+w_v)/2` differs, for a fixed endpoint pair, by
the additive endpoint cost `epsilon*(w_start+w_end)/2`. Adding these half-site
costs gives the exact vertex-sum convention; face-to-face optimization also
needs the correct source and target costs. An edge midpoint or a richer
directional stencil is a different finite discretization.

The continuum target is `inf_P integral exp(xi*h_epsilon(P))*ds`. A fixed
six-direction lattice has L1 geometry at xi=0; a fixed richer stencil has a
polyhedral metric. Refining N alone does not make either direction set
Euclidean. Validate direction errors separately, and compare increasingly
rich stencils using line integration of the field along longer edges rather
than just their endpoints. Treat path quadrature as another error component.

CHZ Remark 1.1 and DGZ Lemma 3.7 address invariance of the exponent under
specified regularizations/path approximations in the whole-space framework.
They do not assert equality of these small-grid distributions on the torus.

### The two exponent conventions

CHZ equation (1.3) defines the crossing exponent by a high-probability
threshold `D_cross(epsilon) <= epsilon^alpha`. Heuristically,

```
median D_cross(epsilon) = epsilon^(lambda_cross+o(1)),
lambda_cross = 1-xi*Q(xi).
```

Therefore a regression against `log(N)` at epsilon proportional to 1/N
has slope `-lambda_cross`; against `log(epsilon)` it has slope
`+lambda_cross`. Do not fit the unscaled lattice sum and use the same formula.
At xi=0 the physical continuum crossing is one and the exponent is zero.

DGZ equation (1.7) uses the symbol `lambda_n` for a median distance between
two fixed points inside a box. Proposition 1.1 says
`lambda_n=2^(-(1-xi*Q)*n+o(n))` at epsilon=`2^(-n)`. This sequence is a
normalizer, not the CHZ exponent. The weak-metric paper equation (1.4) uses
`a_epsilon` for another median normalizer.

For an N-node open cube with spacing 1/N, opposite grid faces are only
`(N-1)/N` apart. Either use the exact site-sum whose xi=0 crossing is one,
or record and divide out this known edge-metric length in a sensitivity fit.
Tiny deterministic finite-grid length changes can otherwise contaminate a
three- or four-size crossing slope.

### Which xi values are plausibly subcritical?

DGZ defines `xi_crit=sup{xi:Q(xi)>sqrt(6)}` and proves tightness throughout
that phase, with Euclidean topology for subsequential limits. Uniqueness is
not asserted. CHZ Theorem 1.3 gives
`rho_lower(xi)<=lambda_cross(3,xi)<=2*xi`; the square-root in the general
bound is `sqrt(2*d-2)`, not `sqrt(2*d)-2`.

Using `Q=(1-lambda_cross)/xi`, these bounds yield the following diagnostics
for the theoretical exponent, not estimates from the simulation:

| xi | Lower Q bound | Upper Q bound | Subcritical assessment |
| --- | ---: | ---: | --- |
| .1 | 8 | 10 | Inside from these bounds |
| .2 | 3 | 5 | Inside from these bounds |
| .3 | 1.3333 | 3.2010 | Bounds do not decide |
| .4 | .5 | 2.1075 | Outside the `Q>sqrt(6)` phase |
| .6 | −.3333 | 1.55 | Outside the `Q>sqrt(6)` phase |

Here `rho_lower(xi)=max[(sqrt(5/2)-1/sqrt(6))*xi-(sqrt(15)-2)/6,0]`
for xi<=1/sqrt(6), and `max[1/4-xi^2/2,0]` above that threshold.
In particular xi<1/(sqrt(6)+2)≈.2247 is sufficient, while the sharper
lower bound confines the critical endpoint to at most about .3623 under
this exponent identification. Values .4 and .6 remain useful finite-grid
stress tests; they must not be advertised as subcritical GMC-matched metrics.

### The implicit gamma–xi map

CHZ Definition 1.7 and Proposition 1.8 parameterize a unique dimension
exponent using `Q=3/gamma+gamma/2` and `xi=gamma/d_gamma`.
The later weak-metric paper states this relation in equation (1.5) and
identifies `d_gamma` as the weak metric's Hausdorff dimension in Corollary 1.9,
using its KPZ theorem. This is a result within the weak-metric framework,
not a proven Budd–Castro identification or a uniqueness theorem for a
canonical 3D LQG metric.

If Q>sqrt(6), the subcritical solution is
`gamma=Q-sqrt(Q^2-6)`, which is numerically better evaluated as
`gamma=6/[Q+sqrt(Q^2-6)]`. Then `d_gamma=gamma/xi`.
If a finite slope gives Q<=sqrt(6), gives no real gamma, or is unstable
under size/stencil/infrared changes, no theoretically matched pair has been
established. In particular, near criticality the inversion is highly sensitive.
The gamma/xi scan must remain explicitly unmatched unless this step succeeds.

Even broad bounds constrain matching: from
`3+gamma^2/2 <= d_gamma <= 3+gamma^2/2+2*gamma`, the admissible xi intervals
for gamma=.5,1,1.5,2 lie respectively inside
[.1212,.1600], [.1818,.2857], [.2105,.3636], [.2222,.4000]. These are
outer bounds, not proposed parameter estimates. gamma=0,xi=0 is the ordinary
reference; gamma=0 with nonzero xi is metric plus volume measure, not a
subcritical gamma-parameter match.

## 6. Intrinsic random metric-measure comparison

Use the underlying simple graph for unweighted graph distance and normalized
vertex counting measure. All graph controls must target the same explicitly
selected original/biased law; the saved Budd–Castro benchmarks mainly target
the paper-biased law. At finite volume these laws differ. A2/A3 have exact
finite-time bridge values but discretized joint continuous minima. Their
coupled minimum-resolution checks remain relevant to any apparent match.

The periodic LFPP torus and an open cube are different random spaces.
Use the torus for the principal intrinsic distribution comparison, and the
open cube for crossing diagnostics and a boundary sensitivity check. A
periodic field restricted to a cube has special long-range face correlations;
it is not literally the sphere-grounded whole-space field of CHZ. Local
unit-log behavior motivates exploration, not a precision replication of its
normalization exponent. Unknown graph topology is another independent
limitation; matching counts does not match resolution or topology.

For each fixed finite ensemble, choose a deterministic ensemble median of
two independently measure-sampled distances. Give each realization equal
probability and each within-realization pair law its probability weight.
Normalize graph and LFPP distances by this same type of statistic. If the
finite-grid atom forces median zero, record failure of this normalization
rather than silently deleting collisions. Per-realization normalization is a
separate sensitivity analysis because it removes random global-scale variation.

Retain independent realizations' pair CDFs/quantiles, rootwise ball masses,
and eight-point distance matrices. Triangles use three distinct sample
indices, not necessarily distinct locations. Preserve degenerate triangles
or explicitly report how many ratios cannot be defined. Pair laws alone
do not determine sampled matrix laws. Mean ball mass duplicates the pair CDF;
its across-root distribution adds information.

The same h supplies every LFPP and GMC member for a realization. Uniform
volume roots are a separate diagnostic from GMC roots. Reuse common random
uniform variables for categorical measure sampling and preserve coupled
resolutions when feasible. These couplings improve difference estimates;
they do not create extra independent fields.

## 7. Selection, uncertainty and research claims

Split independent field and graph seeds before parameter selection and
confirmation. Count every tested `(gamma,xi,stencil,boundary)` choice.
Use weighted Wasserstein/CDF objectives for pairs and balls, plus at least
one sampled-matrix or triangle observable. A joint score is a ranking device,
not an uncalibrated omnibus p-value or evidence of a unique parameter map.
Include xi=gamma=0 controls and finite-size/stencil sensitivities.

Bootstrap entire independent fields and graphs, carrying all roots,
parameters and resolutions together. Recompute ensemble normalization inside
the bootstrap if its uncertainty is included. Monte Carlo confidence intervals
do not cover discretization, finite-volume corrections or model selection;
show these separately. If computation fixes the exploratory normalization
for confirmation, state that this is conditional on the training estimate.

A selected match needs independent confirmation in several observables and
refinement stability. A mismatch only on one coarse distribution is not a
disproof of a continuum correspondence. Neither outcome identifies a
Brownian covariance-to-gamma map. Do not call graph GFFs log-correlated
3D fields without a separate definition and validation.

The [independent statistics audit](STATISTICS_PHASE3A.md) documents the
implemented objective, bootstrap, collision handling and noise floors. In
the I3 discovery score, most improvement over the flat control comes from
root-ball dispersion rather than pair or triangle shape. A separate paired
smooth-field control addresses whether this improvement is specific to
logarithmic correlations. Preserve the component-level interpretation and
the frozen choices; do not count a lower sum as several independent matches.
