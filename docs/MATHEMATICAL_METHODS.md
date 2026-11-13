# Mathematical conventions and validation gates

Primary authority: Budd–Castro, *Scale-invariant random geometry from mating of
trees: a numerical study*, arXiv:2207.05355v1 (local `papers/Budd_Castro.pdf`),
§§2.5, 3, 4, equations (22)–(45), Tables 1–2. The arXiv landing page was checked
on 2026-11-10 and lists v1. The local title page also identifies v1.

## Correlations and cones

The user-facing convention is `(a,b,c)=(C12,C13,C23)`:

```
C = [[1,a,b], [a,1,c], [b,c,1]].
alpha=acos(-a), beta=acos(-c), theta_angle=acos(-b).
```

This differs in the ordering of `b,c` from an earlier project-context formula.
`theta_angle` replaces the paper's third angle gamma. Angles are radians. Check
positive definiteness directly; the condition that their sum exceeds pi alone
is insufficient. With C=RRᵀ (Cholesky), standard Brownian motion in
W=R⁻¹R³₊ maps to covariance-C Brownian motion by R. Each positive coordinate
rescaling leaves the graph law unchanged. Equicorrelation is admissible for
−1/2<rho<1; at −1/2 the rank is two and the excursion limit is singular.

The principal Dirichlet angular eigenmode satisfies −ΔS² m1=lambda1 m1.
alpha1=sqrt(lambda1+1/4) and gamma_s=1−alpha1. These are spectral and
susceptibility exponents, not cone angles or a Hausdorff dimension. The optional
surface-FEM module computes approximate eigenpairs on an inscribed triangular
mesh. Refinement studies check (29) for birectangular cones. This calculation
does not by itself validate a general correlated excursion sampler.

## Original and biased finite-volume laws

The original law is a Brownian excursion of duration one in W. The paper's
biased law weights it by

```
I(X) / E I(X),   I(X)=integral_0^1 |X(t)|^(2 alpha1−2) dt,
E I(X)=2^(alpha1−1) Gamma(alpha1)/alpha1.
```

The norm is in the *whitened cone*, before multiplying by R. Under this biased
law, sample S on W∩S² with density m1², and independently two paths from S
conditioned to hit the cone boundary at its vertex. Reverse the first,
concatenate, and normalize duration and amplitude as in (32). Their hitting
times are independently InvGamma(alpha1, scale=1/2), (33). The relevant Doob
function is h(x)=|x|^(−alpha1−1/2)m1(x/|x|), (41). The paper approximates the
paths by rotated bounded-step walks within interior balls (§4.1).

These finite-volume laws differ. Absolute continuity does not alone prove
invariance of all asymptotic metric exponents. The paper motivates preservation
of local critical behavior; this remains part of its scaling interpretation.

## Exact independent-coordinate specialization used here

For the independent d-dimensional orthant, the positive homogeneous harmonic
function is product(x_i) (degree d), m1(s) is proportional to product(s_i),
lambda1=d(2d−2), and alpha1=3d/2−1. Thus alpha1=3.5 in d=3 and 2 in d=2.

An unbiased scalar excursion is the norm of a three-component Gaussian
Brownian bridge from zero to zero (Bessel-3 bridge). Coordinates are independent.
For the biased law, S_i² has the Dirichlet(3/2,...,3/2) distribution. Draw
T1,T2 as above, U=T1/(T1+T2), and H=S/sqrt(T1+T2). Conditional on (U,H),
each coordinate is independently a Bessel-3 bridge 0→H_i on [0,U], followed by
a Bessel-3 bridge H_i→0 on [U,1]. A norm of a Gaussian vector bridge constructs
each leg because one of its endpoints is zero. This is an exact specialization
of the paper's (32)–(40), avoiding approximate walk-on-balls paths.

Inside a segment of duration dt with positive scalar endpoints x,y, the
conditional Bessel-3 bridge equals a scalar Brownian bridge conditioned to stay
positive: endpoint Doob factors cancel. Its minimum M obeys

```
P(M>z | x,y) = [1−exp(−2(x−z)(y−z)/dt)] / [1−exp(−2xy/dt)],
0 <= z <= min(x,y).
```

Invert this survival function using an independent uniform variate. Conditional
segment minima are independent given the skeleton and junction. Insert U into
the grid so a cell straddling U includes minima from both legs. This yields
exact joint cell minima for the independent biased and unbiased laws, up to
floating point error. Increasing oversampling is unnecessary for these laws;
separate generated samples will change because the random-number consumption
changes, but their graph distributions agree.

## General-SPD approximation and restricted interpretation

The general sampler targets a Gaussian bridge conditioned positive at a finite
grid using multiscale elliptical-slice transitions for blocks conditional on
their endpoints. It accepts arbitrary SPD C. Positivity is a hard likelihood;
each exact block transition preserves the finite-grid target. Initial states
are independent-coordinate excursions, so finite burn-in requires validation.

The retained curve is piecewise linear. Grid conditioning misses continuous
boundary crossings. Repeated minima at shared grid endpoints are resolved by
explicit lexicographic infinitesimals, producing triangulations but adding a
further approximation. Tie frequencies, traces, sweep sensitivity, independent
chain variability and grid sensitivity are reported. This sampler implements
neither exact continuum excursions nor the paper's general biased h-transform.
General-correlation dimension fitting is disabled in the gated runner.

## Graph and metric conventions

Let m_i be the coordinate minimum on cell i. Non-neighboring cells i<j have a
diagonal if max(m_i,m_j)<min(m_(i+1),...,m_(j−1)). Add the cyclic boundary
once. End cells share the zero minimum and are cyclic neighbors. Distinct
interior minima yield exactly n−3 noncrossing diagonals per tree. The reference
implementation tests all pairs; the optimized monotone stack uses horizontal
visibility. Tie preprocessing and canonical edge sorting make the current
implementation O(d n log n) time, O(d n) storage. No dense n×n array is used.

Diagonals retain their contributing tree and multiplicity. The shared cycle has
multiplicity one. Unweighted shortest paths use the underlying simple graph;
parallel edges of unit length cannot shorten such a path. Random walks must
separately specify conductance conventions.

Uniform independent vertices include self-pairs. A uniform root followed by
all target vertices estimates this law. Independent graphs are uncertainty
clusters; roots and targets within a graph are not independent replicates.
Diameter bounds are certified finite-graph bounds from pivot eccentricities
and triangle inequalities. A lower/upper bound scaling slope is diagnostic and
does not bound a limiting dimension.

Log-distance slopes and inverse slopes are finite-size effective exponents.
Bootstrap intervals account for graph Monte Carlo uncertainty, not unknown
asymptotic corrections. The shifted histogram collapse follows §4.2 but its
final correction ansatz (45) is not identified by a small pilot and is not
silently substituted by a naive log fit.
