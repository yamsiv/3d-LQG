# Phase 2C: A3 benchmark, corner-aware harmonics and Brownian path accuracy

Research date: **11 November 2026, America/New_York**. Four workstreams handled
coordination/positive-factor variational methods, A3 mathematics, independent
spectral methods, and path/minimum validation. Earlier code, datasets and
reports were preserved. New modules and checkpoints are labeled Phase 2C.

**Decision: Outcome C for the requested complete general sampler.** A reliable
arbitrary-covariance excursion/graph sampler has not been established. This is
not a claim that the model is unsimulable: this task makes substantial progress
on the angular obstruction and repairs a demonstrated clock error, but the
complete path and joint-minimum accuracy gates still fail. No generic graph
sweep or new Hausdorff-dimension fit was performed.

The concrete advances are an independently accessible A3 exact-grid benchmark;
two genuinely different corner-aware angular methods with strong cross-method
agreement and positive-factor structure; a telescoping path-likelihood identity;
and a Brownian stopping skeleton with exact-law sub-ball clocks. The recommended
next research phase is an independent Euclidean 3D field/GMC/LFPP pilot, with
graph comparisons restricted to exact independent and carefully discretized
special matrix-bridge controls. That next phase is proposed here, **not begun**.

## Part I. Mathematical developments

### 1. What was audited and kept distinct

Repository instructions, context/log, Phase 2B report, independent audit, Doob
methods, correlated-validation notes, original paper and relevant source/tests
were read. The primary graph authority remains Budd–Castro,
[arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1), §§3.2–3.4 and 4.1.

Five separate errors were retained throughout: zero support patches; wrong
corner asymptotics; nonzero harmonic mean-value defects; accumulated error
from local normalization of a nonharmonic function; and path/clock/minimum
discretization even with analytic harmonics. A successful eigenvalue or mean
distance alone is not used to pass any of the other gates.

The original and paper-biased excursion laws remain separate. All new complete
Doob controls target the **biased** law. Matrix benchmarks expose an explicit
`law="unbiased"` or `law="biased"` option. No finite-grid MCMC samples were
silently reused as paper-biased continuous excursions.

### 2. A3 Weyl chamber: verified identities

For four independent standard Brownian particles, the three first differences
divided by sqrt(2) have covariance

```
C_A3 = [[1,-1/2,0],[-1/2,1,-1/2],[0,-1/2,1]].
```

Thus `(a,b,c)=(C12,C13,C23)=(-1/2,0,-1/2)` in this repository. Its eigenvalues
are `1−1/sqrt(2),1,1+1/sqrt(2)`, condition number about 5.83, and paper-ordered
angles `(alpha,beta,theta_angle)=(pi/3,pi/3,pi/2)`. This spherical triangle has
area pi/6 and is genuinely non-birectangular.

For ordered zero-sum particle positions lambda and normalized gaps x,

```
Delta(lambda)=product_{i<j}(lambda_j-lambda_i)=8*u(x),
u(x)=x1*x2*x3*(x1+x2)*(x2+x3)*(x1+x2+x3).
```

The gap generator is `L_C=(1/2)sum Cij partial_i partial_j`. Exact rational
polynomial differentiation in the tests gives `L_C u=0` coefficient by
coefficient. This is not a finite-difference harmonicity check. For C=RRᵀ,
the map from the zero-sum particle plane to white coordinates z=R⁻¹x is an
isometry: `|z|²=xᵀC⁻¹x=sum lambda_i²`. Therefore U(z)=u(Rz) is positive
harmonic, homogeneous of degree six, and zero on cone faces. Its spherical
restriction is the principal mode, with

```
lambda1=6*7=42, alpha1=sqrt(42+1/4)=13/2,
gamma_s=1-alpha1=-11/2,
h(z)=U(z)/|z|^13.
```

The spectral parameter 6.5 is **not** a graph Hausdorff dimension. The corner
powers, in ray order R⁻¹e1,R⁻¹e2,R⁻¹e3, are 3,2,3, consistent with their
respective interior angles.

The random-matrix ingredient follows noncolliding Brownian/Dyson bridge theory;
the covariance, normalization and biased specialization are derived explicitly
in [WEYL_A3_PHASE2C.md](../docs/WEYL_A3_PHASE2C.md). The underlying primary
reference is Katori–Tanemura,
[Noncolliding Brownian Motion and Determinantal Processes](https://arxiv.org/abs/0705.2460).

### 3. A3 original and biased exact-grid constructions

A Hermitian 4×4 Brownian matrix uses variance dt on diagonal coordinates and
dt/2 on each independent off-diagonal real/imaginary coordinate. Removing the
trace leaves 15 independent real Gaussian directions. Ordered eigenvalue gaps
of its zero-to-zero bridge give the unbiased A3 excursion at every retained
time. In particular,

```
|Z(t)|²/[t(1-t)] ~ chi-square(15),
Cov(|Z(s)|²,|Z(t)|²)=30*[min(s,t)-st]².
```

For the **biased** law, normalize the centered eigenvalues of a traceless GUE
matrix to unit norm. Their angular density is the required m1². The exact
normalization is

```
integral_(W cap S²) U(s)² d_sigma(s)=pi/180180,
m1(s)=sqrt(180180/pi)*U(s).
```

An independent rational Gaussian-moment calculation gives
`E[u(G_C)²]=9/2`, agreeing with that normalization. The radial chi(15) variable
is independent of the angle. Draw independent
`T1,T2~InvGamma(13/2,scale=1/2)`, let `T=T1+T2`, `U_time=T1/T`, and set
the normalized junction eigenvalues to `s/sqrt(T)`. Generate independent
Hermitian bridges zero→diag(s/sqrt(T)) over [0,U_time] and back to zero over
[U_time,1]. Their gaps give the paper-biased grid law, whose tilt involves
`integral |Z(t)|^11 dt`. Fixing endpoint eigenvector orientation does not change
the eigenvalue-path law, by conjugation invariance at the zero endpoint.

These constructions give **exact finite-time joint distributions**, up to
floating-point Gaussian/eigenvalue computations. They do not simulate exact
continuous joint gap minima. Every matrix-bridge graph retains that explicit
qualification. The new analytic A3 cone also supplies an independent harmonic
oracle for interior-ball algorithms.

### 4. Two independent corner-aware angular methods

**Spectral method of particular solutions (MPS).** Fractional-order Ferrers
modes centered at each spherical vertex incorporate its opening angle;
regular interior modes stabilize multiple singular corners. A stacked
boundary/interior QR/SVD formulation prevents spurious functions that are
small everywhere. This follows the numerical strategy studied by
[Dahne–Salvy](https://arxiv.org/abs/2003.08095), but this implementation does
not reproduce their interval-certified eigenvalue enclosures. A local
Fourier/Ferrers continuation from an interior arc restores corner evaluation
where global cancellation loses digits. Interface defects are measured.

**Independent positive corner-factor Galerkin/Ritz method.** Let d_i=n_i·z
be positive face distances. For vertex i between faces j,k, write its local
wedge coordinates r_i,theta_i, opening omega_i and nu_i=pi/omega_i. Define

```
K_i=r_i^(nu_i-2)*sin(nu_i*theta_i)/[sin(theta_i)*sin(omega_i-theta_i)],
F(z)=d0*d1*d2*product_i K_i(z).
```

F is positive inside, has exact face zeros, and has leading corner behavior
`r_i^nu_i*sin(nu_i*theta_i)`. Its degree is beta=sum nu_i−3. The quotient is
evaluated stably from the nearer face, without subtracting nearly equal angles.
It reproduces all birectangular harmonic factors exactly. For A3 it is the
degree-five product missing only the final positive factor x1+x2+x3.

Trial modes are `F(s)*q_D(d(s))`, where q_D is a homogeneous degree-D
polynomial. Analytic tangent gradients and independent Duffy/Gauss spherical
quadrature assemble a variational eigenproblem. D=0 contains orthant and
birectangular controls; D=1 contains the A3 mode. No P1 interpolation of the
result is used. Mixed polynomial coefficients are not clipped. Instead,
exact rational Bernstein degree elevation certifies positivity of the
polynomial represented by the stored binary-float coefficients. Combined
with F>0, this proves positivity of that real-valued formula. It is not an
eigenfunction-error, rounding-error or path-law certificate.

The methods have complementary weaknesses: MPS solves the interior equation
but has small boundary leakage and cancellation; Ritz enforces boundary/corner
structure but is only approximately harmonic. This is a genuinely independent
cross-check, not a second mesh size of the old P1 solver.
See [MPS methods](../docs/CORNER_METHODS_PHASE2C.md) and
[positive-factor methods](../docs/CORNER_RITZ_PHASE2C.md).

### 5. Harmonicity and accumulation over a leg

For the true harmonic h and a positive candidate h_tilde, define
`Z(x)=average_sphere h_tilde`, `R(x)=Z(x)/h_tilde(x)-1`, and
`e(x)=h_tilde(x)/h(x)`. For a common stopped interior-ball exit chain,

```
dQ/dP = e(xN)/e(x0) * product_j[1+R(xj)]^(-1).
```

Intermediate relative h errors telescope; multiplying their worst-case error
separately at every step is unnecessarily pessimistic. However, the cumulative
log defects remain. If separately established **uniform** hypotheses give
endpoint |e−1|<=epsilon and sum|log(1+R)|<=eta, then

```
|log(dQ/dP)| <= log((1+epsilon)/(1-epsilon)) + eta.
```

A complete excursion additionally needs initial angular-density normalization,
both legs, stopping/tail bounds and the conditional Brownian paths inside
balls. A spectral error Delta p contributes a radial factor involving
`Delta p*log(radius)`, so a unit-sphere comparison alone is insufficient near
the vertex. Observed quadrature defects do not establish these uniform
hypotheses. This identity and its limits are tested and documented in
[the likelihood audit](../docs/LIKELIHOOD_PHASE2C.md).

## Part II. Numerical implementation and computational budget

All additions are new modules: `weyl_a3`, `corner_spectral`, `corner_ritz`,
`harmonic_likelihood`, `local_polynomial_bounds`, and `phase2c_paths` under
`src/`. Experiments checkpoint individual fits or physical realizations under
`results/phase2c/`. Existing Bessel, A2, FEM, finite-grid MCMC and old Doob
implementations were not replaced.

Pilot costs were measured before the main experiments. The nested-minimum
pilot cost about .24–.53 seconds per path; the main 192 paths took 74 seconds.
A 128-clock pilot took .13 seconds; the 16,384-clock experiment took 13 seconds.
The new-Doob pilot cost .48–1.35 seconds per path; 192 controls took 178 seconds.
The specific finer/coarsening experiment took 69 seconds for 48 paths.
No individual main experiment approached 15 minutes. Angular fits and
independent grids took seconds to tens of seconds. The full output tree is
about 164 MB at the integration checkpoint, including four finest-path NPZ
files totaling roughly 40 MB. Memory estimates preceded runs; a new portable
peak-RSS claim is not made.

Matrix-bridge work and storage grow linearly with retained path points, with
fixed-size 4×4 eigenproblems. Ritz has about (D+1)(D+2)/2 coefficients and
quadrature assembly plus a small dense generalized eigenproblem. MPS uses
small dense QR/SVD systems and special-function evaluation. High trial degree
can become ill-conditioned; it is not an unrestricted convergence knob.
The new stopping skeleton scales with inverse squared spatial resolution
away from boundaries, plus adaptive boundary steps. No production graph
parameter sweep or expensive MCMC burn-in campaign was run.

## Part III. Validation and error diagnosis

### 6. Exact A3 controls

Tests verify C=DDᵀ, rational harmonicity, Vandermonde/Frobenius identities,
spherical normalization, corner powers, both excursion laws and matching
analytic exit weights. Independent GUE and uniform-chamber rejection angular
samplers used 8,000 draws each. Observed coordinate KS discrepancies were
about .0075–.0213; these support, rather than prove, the derived identity.
The rejection envelope comes from the normalized Hermite-root maximizer,
not an empirical maximum of the GUE sample.

For 2,400 unbiased paths, normalized squared-radius means at three times
were 14.880,14.875,15.059, each with SE about .11, versus exact 15. Multitime
and biased conditional-moment tests were also performed. The biased radial
moment residual was .0082±.0244 SE. This checks actual paths, not merely
inverse-gamma metadata. Data:
[A3 law validation](phase2c/a3/law_validation/summary.json).

### 7. Angular evidence beyond eigenvalues

Thirty MPS fits, independent boundary samples, 300 harmonicity grid rows and
interface checks were saved. For the generic unequal triple, refined MPS
gives p≈2.7573085701973 and lambda≈10.3600591214806; for equicorrelation .2,
p≈2.3982968874890 and lambda≈8.1501248480282. These are numerical estimates.

Global MPS still becomes negative at very small gaps for some cases: e.g.
equicorrelation .2 at gaps near 1e-8, and A3 near a corner at 1e-6, despite
essentially exact eigenvalues. The local arc continuation restores the tested
corner powers down to about 1e-9. On exact controls its largest observed
relative discrepancy was about 1.75e-7. But the patched representation has
interface mismatch: peak-normalized derivatives differ by less than about
6.2e-9, while endpoint-relative differences can reach .00166. It is not
promoted automatically to a globally smooth positive sampler mode.

The independent Ritz method has exact face zeros and the correct corner
powers algebraically. Generic degree 10 fits have the following observed
cross-method differences, aligned by one interior normalization:

| Generic covariance | Maximum on 1,000 interior points | Maximum on corner rays to epsilon 1e-10 |
| --- | ---: | ---: |
| (−.2,.1,.35) | .0490% | .387% |
| (.2,.2,.2) | .00210% | .0298% |

These differences decrease over degrees4,6,8,10. The generic reference is the
independently checked MPS/local continuation, **not an assumed exact solution**.
Degree12 Ritz eigenvalues are 10.360059121928 and 8.150124848036, respectively.
Their positive-polynomial certificates pass, but mass conditions are already
about 4.0e12 and 1.2e12. Blindly raising degree in double precision is unjustified.
Near-machine-scale face distances also lose relative digits when forming Rz.

At selected face/corner probes and ball fractions .25,.5,.75, maximum Ritz
mean-value defects decrease to about 2.48e-6 and 4.72e-6 at degree 12. A wider
independent spatial audit finds degree 10 maxima at fraction .5 about 4.64e-5
and 4.94e-6; the unequal case reaches 1.19e-4 at fraction .8. Different probe
sets give different maxima; neither is a certified supremum. Sphere order 16
versus 32 changes the largest-ball log means by up to 1.66e-7, so integration
error is recorded separately.

The path audit reuses 12 analytic legs, covering 5,467 existing balls, rather
than generating fresh paths. For Ritz trial spaces containing the exact
I3/A2/A3 harmonics, sphere order 16 gives maximum one-step defect 3.2e-14 and
maximum accumulated absolute log defect 1.8e-12. Order 8→16 changes one
accumulated result by 2.45e-7. This validates the diagnostic on exact controls;
it does not bound all paths of a generic numerical kernel.

A conservative global rejection envelope was initially too loose for generic
chains. A separate exact-rational local polynomial-box bound improves its
polynomial factor dramatically: at an unequal-cone interior point and D10,
the .5-ball ratio falls from about 208,376 to 62; near an edge, to about 1.11.
This removes one avoidable proposal-cost problem. It does not validate the
geometric evaluator's machine enclosure, the approximate eigenfunction, or
an entire path law. No generic chain is sampled merely because the bound
improves. [Local feasibility data](phase2c/local_envelope.json).

![Independent angular formulations](phase2c/ritz/ritz_convergence.png)

![Mean-value and accumulated-likelihood diagnostics](phase2c/likelihood/mean_value_defects.png)

### 8. A concrete clock error and a new stopping skeleton

For a standard three-dimensional Brownian ball exit from its center,

```
E exp(-s*tau)=z/sinh(z), z=r*sqrt(2s),
E tau=r²/3, Var tau=2r⁴/45.
```

The old bounded walk clips a final straight segment and its time fraction.
That leaves first-passage overshoot error. At relative steps .2,.1,.05,.025,
16,384 trials give mean-clock biases +11.34%,+6.06%,+3.39%,+1.05%.
Exact Gaussian time-grid increments still give polygonal-exit biases
+13.13%,+6.63%,+3.39%,+1.88%. Thus Gaussian increments alone do not fix the
stopped-path problem. This isolates a real error source, without claiming it
alone explains the earlier graph offset.

The new method walks on sub-spheres strictly inside the outer ball. Each
sub-ball center has independent uniform exit direction and its correct
Brownian exit-time law. Iterating these samples gives a genuine Brownian
stopping skeleton, up to the numerical inverse CDF. The CDF uses Jacobi-
related series to avoid short-time cancellation. Measured probability residual
is 2.714e-7 and relative quantile error 5.306e-7 on independent probe grids.
These are numerical checks, not a proved uniform likelihood enclosure.

Near the outer boundary the residual gap delta is at most 1e-7*r; projection
omits a conditional mean clock bounded by 2r*delta/3. The genuine endpoint
has mean-square displacement at most 2r*delta from the stopping point. The
new 4,096-clock control gives .331967±.003220 SE for a unit ball, consistent
with 1/3, unlike the old approximately 6% error at relative step .1.

Unknown Brownian segments are confined in their sub-balls, giving a coupling
with polygonal spatial error at most2ell **before** outer projection,
endpoint rotation and excursion normalization. That bound must not be
advertised as a uniform bound on final excursion minima: the rotation can
be ill-conditioned and time cuts introduce further issues. The new full
Doob wrapper deliberately accepts only validated analytic harmonic models.

The cone-vertex cutoff, outer-ball boundary layer, interior spatial resolution,
clock inversion and minima interpolation remain separate parameters/errors.
No independent coordinate-minimum correction is applied to correlated paths.
See [path methods and controls](../docs/PATH_ACCURACY_PHASE2C.md).

### 9. Coupled minima and complete graph controls

Numerical goals were recorded before the main experiments: roughly1% or
smaller mean-distance bias at benchmark sizes, .02 joint-minimum CDF precision,
and explicit local-edge diagnostics. They were not changed retrospectively.
A 1% size-dependent distance error can still be significant for an exponent:
over a factor256 in n, opposite-sign1% biases perturb the distance log slope
by about .0036, corresponding to dimension error about .08 near d=4.83.
Constant multiplicative bias and size-dependent bias are different concerns.

The same fine exact-law realization is observed at n=64/256 and nested grid
resolutions q=4,16,64,256,1024. For I3 the reference has exact continuum
minima; A2/A3 reference q1024 still has approximate minima. There are 64
independent physical paths per cone, **not** independent coarse/fine copies.

At n=256, q=256 minus its coupled reference gives:

| Cone | Relative mean-distance difference ± paired graph SE | Edge Jaccard | Largest measured joint-CDF mean change |
| --- | ---: | ---: | ---: |
| I3 | −.017% ± .112% | .9637 | .00468 |
| A2×half-line | −.018% ± .083% | .9662 | .00228 |
| A3 | −.028% ± .097% | .9643 | .00277 |

Average-distance stability conceals several percent changing local edges.
At q=4, mean distances change by about −.4%,−1.0%,−1.1%, with Jaccards near
.73. These particular exact-grid minimum errors do not explain a **positive**
2% stopped-walk offset by themselves, and are not assumed to equal the
errors of an adaptively stopped trajectory.

The improved-clock full Doob experiment has 192 independent physical paths,
64 per cone, observed at n=64 and 256, with no failures. Against independent
matched-law references, n256 mean-distance results are:

| Cone | New mean distance | Difference from reference, MC95 |
| --- | ---: | --- |
| I3 | 4.21291 | +1.30% [−1.02%,3.62%] |
| A2×half-line | 4.14486 | −.21% [−2.33%,1.91%] |
| A3 | 4.12231 | +1.27% [−.07%,2.62%] |

The corresponding largest joint-CDF differences are .0389±.0387 SE,
.0594±.0348 and .0608±.0269. These do **not** establish the .02 target.
One-time, multitime and conditional radial-bridge diagnostics are retained;
some have2–3 SE deviations. Isolated inspected deviations are not presented
as simultaneous significance tests, nor dropped because other statistics agree.

A specific finer test used 24 I3 and 24 A3 paths at ell=.0025, relative
step .05. I3 differs by +3.91% [.61%,7.21%]; A3 by−.05% [−2.00%,1.91%].
This does not establish monotone convergence or a1% pass. The I3 discrepancy
is unresolved rather than tuned away or attributed confidently to one cause.

Those same finest paths are also coarsened by retaining every2nd,4th,8th
stopping vertex while preserving cell boundaries, junction, clock, endpoint
rotation and roots. This is a paired interpolation diagnostic, **not** the
law of a coarser WOS algorithm. Mean distances change by at most .21%, but
edge Jaccards are .974/.952/.935 for I3 and .966/.945/.919 for A3. Rootwise
ball histograms and across-root variability are saved; mean ball fractions
alone duplicate cumulative pair-distance information.

All raw records, failures, seeds and first two complete finest paths per cone
are retained. Monte Carlo intervals use independent physical paths; paired
resolution changes use within-path differences. No small-size dimension
extrapolation was attempted.

![Coupled minima versus clock errors](phase2c/paths/figures/coupled_minima_and_clock.png)

![Improved-clock graph controls](phase2c/paths/figures/wos_graph_control.png)

## Part IV. Generic-covariance feasibility

| Covariance | Angular/harmonic status | Path/graph status |
| --- | --- | --- |
| I3 | Analytic and exact-control trial spaces verified | Existing exact Bessel/minimum generator validated; new WOS pipeline provisional |
| (−.5,0,0) | Analytic A2×half-line verified | Exact matrix-grid law; continuum minima discretized; new WOS pipeline provisional |
| (−.5,0,−.5) | New A3 polynomial, normalization, spectral data verified | New original/biased matrix-grid laws validated; joint continuum minima approximate |
| (.2,.2,.2) | Positive Ritz and independent MPS relative convergence; observed harmonicity evidence | No validated general excursion/graph samples generated |
| (−.2,.1,.35) | Same, with larger remaining corner/mean-value discrepancies | No validated general excursion/graph samples generated |

The generic angular stage is substantially improved: no P1 zero patches are
inherited, the leading singular factors are correct, positivity has an
algebraic certificate for the stored formula, and independent relative
comparisons converge. But there is no uniform true-eigenfunction/path error
enclosure, and the improved path scheme does not yet pass its exact-control
accuracy goals. Therefore the prerequisites for a generic sampler demonstration
are not met. We do not bypass this gate with the old finite-grid MCMC, an
empirical rejection envelope, or small-n Hausdorff fits.

The stopping decision is specific to the requested accuracy in this focused
task. It is not an impossibility theorem. Useful next numerical work would
require a genuinely coupled complete-path error analysis and sharper uniform
boundary/likelihood estimates, rather than another blind mesh or sample-count
increase. Individual 1% mean tests alone would still be inadequate for local
graphs or a reliable covariance-dependent exponent catalogue.

## Part V. Research decision and the LQG/LFPP/GMC project

### 10. What is accessible now

I3 finite-volume biased and unbiased graphs have an exact continuous-minimum
control. A2 and now A3 provide independent correlated **exact grid laws** with
measured paired minimum/edge sensitivity. Analytic harmonics and the new angular
methods enable potential-theory diagnostics without pretending to have a
generic metric ensemble. Intrinsic pair-distance laws, ball masses, graph
counting measure, degree/multiplicity, return probabilities and resistance can
be investigated on these baselines with their sampling qualifications.

A generic dimension study still requires matching-law path accuracy across
several sizes; systematic distance/minimum errors separately from graph Monte
Carlo variance; enough independent graphs; stable finite-size corrections;
and shape/ball/rootwise checks beyond one fitted exponent. Nothing here proves
metric-measure convergence, a three-manifold topology or 3D LQG equivalence.

### 11. A concrete independent Euclidean fallback phase

**Recommendation, not work executed in Task2C:** temporarily prioritize an
independent Euclidean 3D reference pipeline, with special-family graph controls.

1. Define a unit-log-correlated Gaussian field on a specified 3D torus or box,
   using normalized critical fractional covariance `(-Delta)^(-3/2)` with a
   documented zero/infrared mode. A Fourier realization has variance spectrum
   proportional to `|k|^(-3)`; field amplitudes have its square root. Validate
   the logarithmic covariance coefficient, mollification and cutoff behavior
   before interpreting exponential parameters.
2. Construct renormalized GMC measures
   `exp(gamma*h_epsilon - gamma²*Var(h_epsilon)/2) dx`. In unit-log dimension 3,
   the subcritical range is |gamma|<sqrt(6). Check unnormalized expected total
   mass, fluctuations and coarse moments before samplewise normalization.
   Arbitrary-dimensional discrete polyharmonic/GMC constructions have rigorous
   foundations; see [Dello Schiavo–Herry–Kopfer–Sturm](https://arxiv.org/abs/2302.02963).
3. Independently approximate LFPP path lengths `integral exp(xi*h_epsilon) ds`
   and normalize by a declared crossing-distance statistic. Specify the domain,
   endpoints and discretization. A six-neighbor lattice gives an L1 metric at
   xi=0, so it must not be mislabeled as an isotropic Euclidean approximation;
   investigate angular-stencil or continuum path-solver convergence. Start
   with small grids and measured resource costs, then several independent
   resolutions and field samples. The continuum weak-metric theory supplies
   relevant structure, not an automatic identification with the graphs:
   [Contreras Hip–Zhuang](https://arxiv.org/abs/2512.06292).
4. Keep parameters separate: (a,b,c) controls Brownian encoding covariance;
   the cone's angles and alpha1 are geometric/spectral data; gamma controls
   the GMC exponential; xi controls LFPP lengths. No established map from
   covariance to either scalar has been derived. Any theoretical relation
   between gamma, xi and a metric dimension needs its own normalization and
   assumptions, not substitution of a fitted graph dimension.
5. Compare normalized intrinsic pair-distance distributions, ball-mass laws,
   multiscale fluctuations and diffusion/resistance observables on well-defined
   finite-volume metric-measure ensembles. Restrict graphs to validated I3
   or explicitly discretized A2/A3 references. Keep Euclidean domain/topology
   and compact-volume conventions visible; do not optimize a unique covariance
   map from matching one scalar exponent.

An ordinary 3D GFF has covariance `(-Delta)^(-1)` and a 1/r singularity, not
the required logarithm. A graph GFF conditional on a graph is extra Gaussian
randomness with covariance the graph Laplacian inverse/pseudoinverse; its
behavior depends on resistance and spectral geometry. Neither its name nor
three Brownian encoding coordinates makes it the desired Euclidean 3D
log-correlated field. Choosing a fractional graph Laplacian to force a logarithm
would define another model, not discover a canonical encoding field.

Graph distance with normalized vertex counting measure can be compared as an
abstract metric-measure object without a canonical Euclidean embedding.
Uniform-root pair and ball laws remain meaningful. Euclidean covariance,
conformal coordinates and direct spatial field matching do require additional
structure that these graphs do not presently provide. Matching Hausdorff
dimensions would be weak compatibility evidence, not equality of models.

### 12. Handoff and reproducibility

**Outcome C is the recommendation for the full general sampler.** Validated
components and the exact-grid A3 advance should be retained, while a generic
covariance sweep stays closed. The next suggested phase is the independent
Euclidean reference program above, not another blind correlated simulation
campaign. No such field/GMC/LFPP code was started in this task.

Methods and source links:

- [A3 derivation](../docs/WEYL_A3_PHASE2C.md)
- [MPS and local continuation](../docs/CORNER_METHODS_PHASE2C.md)
- [Positive-factor Ritz formulation](../docs/CORNER_RITZ_PHASE2C.md)
- [Likelihood accumulation](../docs/LIKELIHOOD_PHASE2C.md)
- [Path and joint-minimum diagnostics](../docs/PATH_ACCURACY_PHASE2C.md)

The saved path inventory lists exact counts, timings and failure counts in
[completed_experiment_inventory.json](phase2c/paths/completed_experiment_inventory.json).
Individual experiment directories hold seeds, software versions, parameters,
checkpoints and raw numerical data. The final manifest captures current source
hashes. Some path sidecars lack a historical per-run source hash, so they do
not independently certify a frozen code snapshot for every earlier control.
PNG/PDF plots are standalone artifacts.
No original Phase1/2B dataset or source implementation was overwritten.
The final manifest verifies preservation, test results and local links.

### 13. Interruption recovery and final checks

On resumption, the three specialist contributions and all declared numerical
experiments were already complete. The report draft existed; final logging,
integrity checks and the handoff remained. Recovery parsed all nine JSONL
files, matched every path-inventory row count, and checked CRCs of all five
NPZ archives. No saved numerical experiment required restarting. The
[resumption audit](phase2c/resumption_audit.json) records the recovered hashes.

| Task 2C stage | Final state |
| --- | --- |
| Audit and A3 mathematical/grid-law benchmark | Completed; analytic identities and distribution controls pass |
| Two corner-aware angular strategies and independent comparisons | Completed as numerical methods; uniform relative-error certification unresolved |
| Harmonic/likelihood accumulation analysis | Completed identity and diagnostics; generic whole-path bounds unresolved |
| Improved clocks and coupled path/minimum controls | Completed; full-pipeline accuracy goals not established |
| Generic covariance excursion/graph demonstration | Withheld under the specified validation gate |
| Limited metric scaling study | Not started because prerequisites failed |
| Research decision, documentation and reproducible handoff | Completed at the documented obstruction |

The full resumed suite has **121 passing tests**, with zero failures, errors
or skips. The earlier 121-test XML is also preserved. Numerical distribution
tests support the derived constructions; passing unit tests do not supersede
the failed complete-sampler accuracy gate. See the
[resumed test record](phase2c/test_results_resumed.xml) and
[final artifact/preservation manifest](phase2c/final_manifest.json).

To rerun the inexpensive validation suite, use
`.venv/bin/python -m pytest -q`. For integrity verification, the executed
command is `.venv/bin/python -m experiments.phase2c_manifest`; it refuses
overwriting the manifest. Saved experiment metadata and the method documents
give their exact seeds, parameter settings and commands. Existing datasets,
including failed validations, remain the record for this phase.
