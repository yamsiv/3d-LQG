# Positive corner-factor Galerkin method (Task 2C)

This is an independent numerical formulation, rather than a refinement of the
old P1 facet interpolation. The implementation is `src/corner_ritz.py`.
It complements the particular-solution formulation documented in
`CORNER_METHODS_PHASE2C.md`. All errors below concern angular functions,
not proof of an accurate correlated excursion sampler.

## A positive factor with the right corner geometry

Let C=RRᵀ, let n_i be row i of R, and d_i(z)=n_i·z. The whitened cone is
d_i>0; each n_i has unit norm. Vertex i is the positive ray where the other
two faces meet. For those faces j,k, use orthonormal planar axes e0,e1 with

```
rho=C_jk, omega=acos(-rho), nu=pi/omega,
e0=(n_j-rho*n_k)/sqrt(1-rho²), e1=n_k,
a=e0·z, b=e1·z, r=sqrt(a²+b²), theta=atan2(b,a).
```

Then d_k=r*sin(theta), d_j=r*sin(omega-theta). Define

```
K_i(z) = r^(nu-2) * sin(nu*theta) / [sin(theta)*sin(omega-theta)],
F(z) = d_0*d_1*d_2 * product_i K_i(z).
```

Every factor is positive in the cone interior. Along a face F vanishes
linearly. Near vertex i, the d_j*d_k factor cancels its denominator, giving
the desired leading r^nu*sin(nu*theta), times a smooth positive coefficient
from the remaining factors. Its homogeneous degree is beta=sum_i nu_i−3.
Thus it fixes both the corner exponent and leading angular shape, not only
the exponent. Multiplicative normalization at an interior point is harmless.

For an orthant K_i=2, so F is proportional to product d_i, the exact harmonic.
For every birectangular cone, the two right-angle factors are constant;
F is proportional to the known wedge harmonic times the perpendicular height.
For A3, F is proportional to
`x1*x2*x3*(x1+x2)*(x2+x3)`; multiplication by the positive degree-one
polynomial `x1+x2+x3` gives the exact degree-six harmonic. These algebraic
control cases distinguish a sound factor construction from an eigenvalue-only
test.

## Stable evaluation and derivatives

The factor uses log products, and evaluates the angle from the *closer face*
directly from the two positive d coordinates. Subtracting a nearly equal
angle from omega is avoided. Write delta=min(theta,omega−theta); then

```
log K = (nu-2)log r + log nu + log sinc(nu*delta)
        - log sinc(delta) - log sin(omega-delta).
```

The angular log derivative is
`nu*cot(nu*theta)-cot(theta)+cot(omega-theta)`.
Near a face the first cotangent cancellation uses its series, starting with
`-(nu²−1)*theta/3`. This improves finite-precision relative evaluation but
does not certify arbitrary proximity to a face: forming d=Rz itself loses
relative digits when d is comparable to matrix-product roundoff.

Analytic log gradients avoid differencing a tiny eigenmode. On the unit sphere,
the tangent gradient of a homogeneous trial function is its Euclidean gradient
minus its degree times the function times z. Tests independently check the
gradient and Euler homogeneity identity.

## Rayleigh--Ritz trial space

At trial degree D use all monomials q(d)=sum_{|b|=D} c_b*d^b. The angular
trial functions are F(s)*q(d(s)) on |s|=1. The variational stiffness and mass
matrices integrate the tangent gradients and products. A Duffy/Gauss simplex
rule followed by z=R⁻¹t/|R⁻¹t| uses surface Jacobian
`abs(det R⁻¹)/|R⁻¹t|³`. The spherical area control agrees with the angle excess.
Mass-diagonal column scaling improves conditioning; a large-condition-number
gate rejects unresolved double-precision systems rather than hiding them.

This is a Dirichlet trial space. Unlike the MPS basis, it enforces the boundary
and leading corner structure exactly as a real-valued formula, but it solves
the differential equation only approximately. Quadrature variation is separate
from polynomial-degree variation. Homogeneous degrees D and D+2 yield nested
angular spaces via multiplication by dᵀC⁻¹d=1 on the sphere.

## Positivity certificate and its scope

Write t=d/sum(d), so q(d)=sum(d)^D q(t). Homogeneous polynomial degree elevation
on the simplex gives Bernstein coefficients bounding q(t) on the whole closed
simplex. The code interprets every stored binary-float coefficient as an exact
rational and performs elevation with `Fraction`. If every resulting coefficient
is strictly positive, the stored polynomial is positive throughout the simplex.
Together with F>0 this proves positivity of the real-valued trial formula.
Mixed Bernstein coefficients are reported as inconclusive; no mode is clipped.

This certificate does **not** enclose assembly/eigenfunction errors, every
rounded evaluation, harmonicity, or the excursion law. It cannot replace
independent relative-value comparisons and mean-value diagnostics.

## Executed evidence

Dedicated tests check exact orthant/birectangular/A3 trial-space
identities, analytic gradients, quadrature area and stable tiny-corner
evaluation. Analytic controls are represented at D=0, or D=1 for A3.
`experiments/phase2c_ritz.py` saves every fit before continuing; it varies
degree and quadrature independently. Outputs are in `results/phase2c/ritz/`.

For unequal(a,b,c)=(-.2,.1,.35), D=4,6,8,10,12 at quadrature80 gives
lambda approximately 10.360074359,10.360060173,10.360059197,
10.360059127,10.360059122. Independent MPS gives 10.360059121480615.
The largest probed mean-value defect falls from about7.41e-5 at D4 to
2.48e-6 at D12, across edge/corner locations and ball fractions .25,.5,.75.
For equicorrelation .2, D12 gives8.150124848036, versus independent MPS
8.150124848028; the largest corresponding probed defect is4.72e-6.

Eigenvalue agreement is only one check. Independent MPS/local-corner
continuation comparisons, normalized at a common interior point, give the
following maximum observed relative discrepancies at D10:

| Case | 1,000 random interior points | Corner rays down to epsilon1e-10 |
| --- | ---: | ---: |
| Unequal triple | 0.0490% | 0.387% |
| Equicorrelation .2 | 0.00210% | 0.0298% |

These improve from coarser degrees and are observed cross-method evidence,
not a certified supremum or exact generic reference. D12 mass conditions are
about4.0e12 and1.2e12; continuing to raise D indefinitely in double precision
would not be a justified validation strategy. Global MPS corner leakage is
excluded from a purported exact reference by using local continuation only
where its own independent checks support it.

## Kernel envelopes and whole-path limits

The polynomial is globally positive, but a global Bernstein upper/lower ratio
can give extremely pessimistic local exit envelopes. A focused alternative
`src/local_polynomial_bounds.py` expands the stored polynomial exactly around
the current coordinate box center. For d=center+width*u, |u_i|<=1,
`q(d)=sum c_gamma*u^gamma`, so `q<=sum abs(c_gamma)`.
Exact rational algebra bounds this stored polynomial over the box. The small
feasibility experiment in `results/phase2c/local_envelope.json` compares it
with the global range bound; it samples no Brownian paths or graphs.

The box bound is not a proof of the full h evaluator's machine-rounding error,
nor of a true eigenfunction approximation. The separate likelihood audit
shows why endpoint relative error and the *accumulated* log mean-value defects
are required for a whole conditioned leg. No general covariance graph dataset
is validated by the Ritz convergence table alone.
