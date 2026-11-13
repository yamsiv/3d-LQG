# A3 Weyl chamber: derivation and an independent exact-grid benchmark

This new Task 2C control is implemented in `src/weyl_a3.py`. It supplements
the existing independent and A2 controls and leaves them unchanged. Both
the original and Budd–Castro paper-biased laws are available, explicitly
selected by `law`. Exact finite-time distributions do **not** imply exact
continuous-time coordinate or joint cell minima.

## Sources and scope of the derivation

Budd–Castro [arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1),
§§3.2–3.4, equations (23)–(26), (32)–(41), define the excursion, its biased
law, and the inward Doob transform. The locally supplied PDF was consulted.
Katori–Tanemura,
[arXiv:0705.2460v3](https://arxiv.org/html/0705.2460), §1.2, equations
(1.9)–(1.13), give the Dyson diffusion, Karlin–McGregor killed kernel,
and Vandermonde transform. The latter source was consulted online on
2026-11-11. The four-particle gap normalization, angular normalization,
conditional matrix-bridge implementation and tests below are our explicit
specialization and derivation, rather than claimed numerical results from
those papers.

## Covariance, harmonic polynomial and whitening

Let D be the 3×4 first-difference matrix divided by sqrt(2). For four
independent standard Brownian particles B, X=DB has covariance

```
C=DDᵀ = [[1, -.5, 0],[-.5, 1, -.5],[0, -.5, 1]].
```

The repository order is (C12,C13,C23)=(-.5,0,-.5). The eigenvalues of C
are 1−1/sqrt(2), 1, 1+1/sqrt(2), so it is SPD. Its paper-ordered spherical
angles (alpha,beta,theta) are (pi/3,pi/3,pi/2); the spherical area is pi/6.
This is not a birectangular triangle.

For ordered particle locations lambda with zero sum, set
Xi=(lambda[i+1]−lambda[i])/sqrt(2). Their Vandermonde is

```
Delta(lambda) = product_{i<j}(lambda[j]−lambda[i]) = 8 u(X),
u(x)=x1 x2 x3 (x1+x2)(x2+x3)(x1+x2+x3).
```

The factor 8 is (sqrt(2))^6. Independent particle motion has generator
one-half the Euclidean Laplacian. In gap coordinates its generator is
L_C=(1/2)sum Cij partial_i partial_j. Translation invariance removes the
center-of-mass direction. Vandermonde harmonicity therefore yields L_C u=0.
The test suite independently expands the polynomial over rational numbers
and verifies **every coefficient** of L_C u vanishes, with no finite
differencing or eigenfunction solver.

For C=RRᵀ, white coordinates are z=R^-1x and W=R^-1 R_+^3. The map
R^-1D from the zero-sum particle plane to R^3 is an isometry because its
rows are orthonormal. Hence

```
|z|² = xᵀ C^-1 x = sum_i lambda[i]².
```

The positive homogeneous white harmonic U(z)=u(Rz) has degree p=6. Its
spherical restriction is positive inside W and zero on the faces, so it is
the principal Dirichlet mode. In dimension three,

```
lambda1=p(p+1)=42; alpha1=sqrt(lambda1+1/4)=13/2;
gamma_s=1−alpha1=−11/2.
```

The inward harmonic function is the Kelvin transform
`h(z)=U(z)/|z|^13`, homogeneous of degree −7. It differs from the
outward Vandermonde Doob harmonic U. Conditional bridges below are valid
for either transform because endpoint h factors cancel in a bridge law.

Near each spherical vertex the exact corner powers are 3, 2, 3 in the
order of rays R^-1e1, R^-1e2, R^-1e3. At a ray supported on x1,
the three small factors are x2,x3,x2+x3; at the x2 ray only x1,x3 vanish.
These agree with pi/omega for vertex angles pi/3,pi/2,pi/3. Tests check
ratios down to correlated coordinate offsets 1e-7 after white normalization.

## Exact angular normalization and independent angular sampling

For a Hermitian 4×4 Gaussian matrix, use independent N(0,1) diagonal
coordinates and off-diagonal real and imaginary parts with variance 1/2.
The density is proportional to exp(−Tr(H²)/2). Trace removal leaves a
standard 15-dimensional Gaussian matrix. Its ordered centered eigenvalue
density is proportional to `exp(−sum lambda²/2) Delta(lambda)^2` on the
three-dimensional zero-sum plane. Thus its radial density is proportional
to `r^14 exp(−r²/2)`, a chi(15) variable, independent of angle. Dividing
the centered eigenvalues by their Euclidean norm yields exactly the
paper's density m1² on the white spherical chamber.

For completeness, the four-variable Gaussian Vandermonde integral is
`(2pi)^2 product_{j=1}^4 j! =1152 pi²`. Divide by 24 for ordered
eigenvalues, by sqrt(2pi) for the center-of-mass Gaussian, and by 64
because Delta²=64u². Removing the radial integral
`2^(13/2) Gamma(15/2)` gives

```
integral_{W cap S²} U(s)² d_sigma(s) = pi/180180,
m1(s)=sqrt(180180/pi) U(s).
```

An independent exact-rational Wick calculation in the tests finds
`E[u(G_C)^2]=9/2`. Dividing its full-space Gaussian integral into 24
Weyl chambers gives the same angular normalization.

The validation experiment also uses an independent angular algorithm:
sample a four-dimensional real Gaussian, remove its mean, normalize and
sort, giving uniform spherical chamber angle. Reject with probability
proportional to u². The sharp maximum is obtained at the normalized roots
of probabilists' Hermite polynomial H4. A Lagrange calculation gives
`sum_{j!=i}1/(lambda_i−lambda_j)=6lambda_i`; the H4 roots normalized to
sum lambda²=1 satisfy it. Strict concavity of log Delta minus a positive
quadratic on the ordered chamber identifies the maximum. This provides a
rejection envelope independent of the GUE matrix sampler, with theoretical
acceptance about 0.23017. No eigenvalue FEM is used by either algorithm.

## Original unit-time excursion and exact grid values

Generate a Hermitian Brownian matrix bridge H(t) from zero to zero over
[0,1], using 16 independent real Gaussian bridges in the orthonormal
coordinates above. Take its ordered eigenvalues, remove their mean, and
form the three gaps divided by sqrt(2). The Dyson eigenvalue process is
the outward Vandermonde transform of killed independent particles;
conditioning on endpoints cancels the transform factors. The coalescing
zero-endpoint limit is therefore the unbiased A3 cone excursion.

This constructs the joint law at **all retained times**, not merely its
one-time marginal. In particular

```
|Z(t)|²/[t(1−t)] ~ chi-square(15),
Cov(|Z(s)|²,|Z(t)|²)=30[min(s,t)−st]^2.
```

The radial formulas follow directly from the 15 independent traceless
matrix bridge coordinates, independently of eigenvalue computations.

## Paper-biased law: angular density, inverse-gamma clocks and legs

The bias of the original excursion is proportional to
`integral_0^1 |Z(t)|^11 dt`, not to a coordinate norm in the correlated
octant. Its normalizing expectation is
`2^(11/2) Gamma(13/2)/(13/2)` (Budd–Castro equation (36)).

Draw unit centered eigenvalues s from the angular algorithm, and independent
`T1,T2~InvGamma(13/2,scale=1/2)`. Set T=T1+T2, U=T1/T and
the normalized junction eigenvalues v=s/sqrt(T). Generate independent
Hermitian matrix bridges from zero to diag(v) on [0,U] and from diag(v)
to zero on [U,1]. Their gaps give the paper-biased excursion, and U is
inserted into the requested grid exactly.

Why conditioning the matrix endpoint to a fixed diagonal orientation is
valid: a bridge starting at the zero matrix is invariant under simultaneous
unitary conjugation. Conditional on its endpoint eigenvalues, the endpoint
eigenvectors are Haar and independent of its entire eigenvalue path.
Selecting the diagonal orientation consequently leaves the eigenvalue path
law unchanged. Time reversal handles the return leg. Trace fluctuations
are independent and disappear from every gap.

Useful independent checks are `E[T_i]=1/11`,
`U~Beta(13/2,13/2)`, `Var(U)=1/56`, and
`E[|Z(U)|²]=2(13/2)^2/14`. Conditional on T1,T2, for t on a leg of
duration D measured from its zero endpoint, the squared radial mean is

```
E[|Z(t)|² | T1,T2,s] = (t/D)^2/T + 15 t(D−t)/D.
```

These expressions are used without fitting the observations. Original and
biased midpoint distributions are visibly different in the saved figure.

## Numerical validation actually run

Command: `MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.validate_a3`.
Seed 20261111031, SeedSequence-derived independent path streams, 2,400
original and 2,400 biased complete paths, plus 8,000 angles per independent
angular method. Full parameters, software/source hashes, raw arrays, summary
and figure are in `results/phase2c/a3/law_validation/`. The pilot generated
eight n=128,q=64 paths in 0.107 seconds, Python-traced peak 2.97 MB. A
linear grid-length cost extrapolation underestimated small-call overhead:
the complete law experiment took 1.01 seconds instead of 0.07 seconds.
Neither estimate motivates a large run.

At t=1/4,1/2,3/4, empirical chi-square(15) radial means were respectively
14.8797±0.1113, 14.8752±0.1129, 15.0594±0.1107 (independent-path SE).
The biased conditional squared-radius residual was 0.00819±0.02443.
Two-sample angular KS distances range from 0.0075 to 0.02125 over four
eigenvalues and u² (the latter p≈0.054). These tests support the exact-law
derivation; they are neither a proof nor a joint-minimum accuracy guarantee.
Nine initial unit tests passed in 0.60 seconds. A tenth test subsequently
checks the analytic-cone interface, Harnack exit bound and product spherical
quadrature mean-value identity, including points 1e-5 from two faces. All
ten passed in 0.54 seconds before integration with the full suite.
No Hausdorff dimension is inferred from these controls.

## Remaining error and cost

`sample_a3_excursion(n_steps,rng,law,oversample)` returns ExcursionSample
with `grid_values_exact=True`, `cell_minima_exact=False`,
`segment_minima=None`. The graph constructor therefore uses polygonal
minima, whose joint error must be investigated through coupled refinements.
`thin_a3_sample` retains a nested coarse grid and the same junction.
Resolution rows remain one physical realization, not independent samples.

Gaussian draws and fixed-size 4×4 eigensolves cost O(nq) time and storage.
The implementation currently materializes matrices at every retained time.
Strong path/minimum control remains necessary before this benchmark can
support precision correlated-graph exponents; exact grid values alone do
not remove the Task 2B graph-discretization obstacle.

`A3AnalyticCone()` also exposes the existing interior-ball model interface
with R, covariance, p, alpha1, eigenvalue, homogeneity, analytic flag,
`log_u`, `log_h`, normalized `angular_mode`, exact `sample_direction`, and
the positive-harmonic Harnack `log_exit_bound`. It permits an independent
comparison of approximate Doob paths with the matrix-bridge law without
using a numerical angular approximation. This is an analytic oracle, not a
claim that the rest of an interior-ball path algorithm is exact.
