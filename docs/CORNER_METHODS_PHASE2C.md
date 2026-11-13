# Corner-aware angular methods, Task 2C

This document describes the new method of particular solutions (MPS) and its
local corner continuation. The independent positive-factor Ritz method is in
`src/corner_ritz.py`; their generic-covariance comparison is a separate
cross-check. None of these developments certifies a general excursion sampler.

## Mathematical construction

Write C=RRᵀ and let v_j be the normalized columns of R⁻¹. These are the
spherical triangle's vertices. At vertex j choose an orthonormal frame
(v_j,e_j,f_j) so that the two incident edges have polar angles 0 and ω_j.
The angle is determined by the two opposite inward face normals:
ω_j=acos(−C_kl), where {j,k,l}={1,2,3}. This vertex indexing differs from
the ordered list (C12,C13,C23); tests check it explicitly.

For λ=p(p+1), separation of the spherical Laplacian gives local solutions

```
F_jk(r,theta) = P_p^(-mu)(cos r) sin(mu theta),  mu=k*pi/omega_j,
P_p^(-mu)(cos r) = tan(r/2)^mu / Gamma(1+mu)
                  * 2F1(-p,p+1;1+mu;sin(r/2)^2).
```

The gamma factor can be removed by column rescaling. Near a corner the
first term has precisely r^(pi/ω) sin(pi theta/ω), with higher admissible
powers for the remaining terms. Expansions at all three vertices are
supplemented by integer-order sine/cosine expansions at an interior point.
Each term solves the Helmholtz equation **before boundary fitting**. The
corresponding homogeneous functions r^p m and r^(−p−1)m are harmonic inside
the three-dimensional cone wherever these expansions are regular.

Following the MPS approach of [Dahne and Salvy, arXiv:2003.08095v2,
§§1,3.2,3.9](https://arxiv.org/html/2003.08095v2), boundary and interior
evaluations are jointly orthonormalized. The smallest singular value of
the boundary block supplies an eigenvalue-search objective. Interior
expansions are essential in multiply singular triangles. Our implementation
uses ordinary floating point, not their interval-based certification.

`src/corner_spectral.py` uses a column-scaled SVD of the stacked evaluations,
discards singular values below 2e-13 of the largest, and minimizes the
smallest singular value of the orthonormal boundary block. This is the
rank-revealing analogue of the stacked QR construction. Columns whose
linear combinations vanish numerically in both regions cannot masquerade as
eigenfunctions. The mode is normalized to one at the normalized vertex sum.
The initial search interval is motivated by an independent coarse FEM
eigenvalue, solely to locate the first eigenvalue; the FEM mode is not used.
The scalar minimization is recentered to remove scipy's order-one argument
stopping floor. No positivity clipping or exact boundary override is used.

## What changed relative to P1

This strategy is different from the old P1 mesh and from the independent
positive-factor variational method. It satisfies the differential equation
termwise and approximates the remaining boundary conditions. The factor
method enforces boundary values and leading corner powers by construction,
then approximates the differential equation. These complementary error
mechanisms make their comparison useful.

The original repaired P1 mode had positive support but incorrect corner
powers and large interior harmonicity defects. MPS reproduces known modes
and generic eigenvalues very accurately without mesh refinement. Nevertheless
**absolute boundary residuals do not establish relative boundary accuracy**.
In a global sum, a roundoff-sized leakage dominates an exact value of order
epsilon^3. Individual basis functions have the correct behavior at their own
vertex; terms centered elsewhere can cancel only to finite precision.

At order 20 the maximum residual on independent boundary probes is about
1.45e-13 for rho=.2 and 5.85e-13 for the unequal case. Yet rho=.2's global
mode becomes negative at small correlated gaps of size 1e-8. At the A3
corner (epsilon,epsilon,1), epsilon=1e-6, its sphere mean-value diagnostic
is −1.069 despite using the exact eigenvalue 42. This is floating-point
cancellation of a tiny value, not a reliable normalized Doob kernel.

## Stable local continuation and its limits

A stable *local* representation is obtained independently from the global
mode's trace on the geodesic arc r=r0=.12 around a vertex. Project this
trace onto the sine functions sin(k pi theta/ω) using Gauss–Legendre
quadrature (160 nodes, 18 terms). Divide by the radial function at r0 and
evaluate its separated expansion inward. This satisfies Helmholtz and both
adjacent Dirichlet edges termwise and has the correct corner exponent.
Using sin(r/2) from tangential coordinates rather than 1−cos(r) avoids losing
small radii to floating-point cancellation.

For exact controls, local continuation restores relative corner errors below
1.75e-7 on the sampled epsilon=1e-2,...,1e-9 rays. At the smallest epsilons,
the conversion between correlated gaps and rotated tangent coordinates itself
loses relative precision. The mean-value defects on interior balls contained
in these local patches are at approximately machine precision. Generic local
exponents converge to the predicted values as epsilon decreases.

This does **not** authorize swapping the local patch into a global sampler.
Matching values on one arc does not ensure exact matching of normal
derivatives. At order20 the peak-normalized arc mismatches are ≤3.2e-11,
but relative arc errors close to incident edges reach approximately 1e-4
in the unequal case at angular fraction 1e-7. Noncorner edges still have
global leakage. A patched function would need independent interface and
relative edge analysis, or a globally controlled positive formulation.

`interface_derivatives.json` additionally records one-sided local and centered
global finite-difference radial derivatives at steps 1e-4,3e-5,1e-5. At the
smallest step the peak-normalized mismatch is below 6.2e-9 in these cases,
but the largest *relative* mismatch near arc endpoints reaches 0.00166 for
the unequal cone. Finite-difference truncation and cancellation are included
in that diagnostic; it is not a rigorous derivative-error bound.

## Recorded numerical evidence

`results/phase2c/angular/` contains 30 checkpoint JSON files (six cones,
orders4,8,12,16,20), all raw corner/edge values, local slopes, interface
residuals, and mean-value tests. The four analytic controls are I3,
C12=−.5 with other correlations zero, C12=+.5 with others zero, and A3.
Known controls use their exact degree for the detailed mode diagnostics;
the independently searched degree is stored separately. Generic modes use
the searched degree throughout. No Brownian paths or graph samples are
generated in this audit.

| Generic covariance | p, order20 | lambda=p(p+1) | Boundary tension, orders8/12/20 |
| --- | ---: | ---: | --- |
| rho=.2 | 2.398296887488958 | 8.150124848028183 | 2.18e-7 / 1.38e-12 / 2.00e-13 |
| (a,b,c)=(-.2,.1,.35) | 2.757308570197275 | 10.360059121480615 | 6.16e-7 / 1.11e-10 / 4.07e-13 |

These are floating-point numerical convergence observations, not certified
digits or validated excursion laws. Some order20 matrices have retained
condition numbers near 1e12; further increasing the order blindly is not a
reliable way to gain relative accuracy.

The additional append-only `harmonicity_grid.jsonl` varies interior/edge/corner
locations, ball fractions .25/.5/.8, angular quadratures16/32, and orders8/20.
This separates numerical quadrature error from evaluator cancellation. A
single small defect at an interior point is not used as a global bound.

All 30 fits and primary diagnostics took approximately ten seconds with
one BLAS thread. Eight mathematical tests check vertex-angle indexing,
birectangular eigenspaces, A3 corner continuation, generic refinement and
explicit rejection of negative modes, including an independent Ritz comparison.
Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.validate_corner_solver --output results/reproduction/phase2c_angular --extra-mean-value --extra-interface
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.compare_corner_methods --output results/reproduction/phase2c_angular/comparison
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_corner_spectral.py
```

Existing output JSON files are skipped and retained on continuation. The
figure `mps_diagnostics.png` displays exact-control corner relative errors
and generic boundary tensions. No old source module or Phase2B dataset was
replaced.

## Independent comparison with the positive-factor Ritz method

The eight checkpoints under `angular/comparison/` use polynomial degrees
4,6,8,10 at quadrature order80. They normalize both methods at the normalized
sum of the spherical vertices, evaluate 1,000 independently fixed random
interior directions (Dirichlet(2,2,2) weights, seed114732), and compare all
three edge/corner rays at epsilon=1e-2,...,1e-10. Global MPS values are
retained even where cancellation is severe; near-corner comparisons use its
independently continued local representation. Neither numerical method is
called exact for generic covariance.

| Covariance | Ritz degree | Maximum interior relative discrepancy | Maximum corner relative discrepancy |
| --- | ---: | ---: | ---: |
| rho=.2 | 4 | .002355 | .005325 |
| rho=.2 | 6 | .0005794 | .002061 |
| rho=.2 | 8 | .0001361 | .0008070 |
| rho=.2 | 10 | .00002097 | .0002977 |
| unequal | 4 | .01563 | .02758 |
| unequal | 6 | .005982 | .01554 |
| unequal | 8 | .001934 | .008049 |
| unequal | 10 | .0004895 | .003867 |

The degree10 eigenvalues are 8.150124848159056 and 10.360059127141907,
respectively, compared with MPS values 8.150124848028183 and
10.360059121480615. The scaled mass condition numbers are 1.77e10 and
5.32e10. Thus eigenvalues agree to much higher relative precision than the
corner amplitudes. Degree10 edge ratios at epsilon≥1e-6 have maximum
discrepancies 8.05e-6 and 8.20e-5, respectively; at smaller epsilon global
MPS boundary leakage eventually contaminates this comparator.

These distinct methods show encouraging generic eigenfunction convergence,
including the relative corner amplitudes that the P1 audit missed. The
sampled comparisons supply no uniform relative-error theorem or path-law
bound. In particular, local harmonic mean-value defects of the globally
positive Ritz evaluator must be examined separately before any sampler claim.

## Sampling decision

MPS is a strong independent interior/spectral reference and provides useful
local corner data. The present global floating-point representation fails
uniform positivity/relative-boundary requirements and is deliberately not
connected to the excursion sampler. Passing harmonicity on a finite set of
balls cannot certify all conditioned transitions, and a numerical eigenvalue
alone gives neither the boundary law nor accumulated path-law control.
