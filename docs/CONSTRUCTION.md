# Construction, sampling law, and APIs

The source authority is Budd and Castro, *Scale-invariant random geometry from
mating of trees: a numerical study*, [arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1),
local `papers/Budd_Castro.pdf` (28 pages). The implemented definitions were
checked against §§2.5, 3.1–3.4 and 4.1, equations (28), (32)–(41). The local
research context is reference material; this task stops before fields/GMC/LFPP.

## Parameters

`from_correlations(a,b,c)` uses the user's convention

```
[[1,a,b],
 [a,1,c],
 [b,c,1]].
```

The paper's angles satisfy `C12=-cos(alpha)`, `C23=-cos(beta)`,
`C13=-cos(theta_angle)`; its symbol gamma is renamed theta_angle here. Thus
`from_angles(alpha,beta,theta_angle)` does **not** pair its second argument with
the user's second correlation. All inputs undergo Cholesky positive-definiteness
validation. `equicorrelated(rho)` requires `-1/2<rho<1` in dimension three.
At rho approaching -1/2 the smallest covariance eigenvalue tends to zero while
the other two approach 3/2. No exactly singular covariance is accepted.

Writing `C=R R^T`, the whitened cone is `W={z:R z>0}`. Its inward unit face
normals are the rows of R. The cone's first spherical Dirichlet eigenpair
`(-Delta)m1=lambda1*m1` determines
`alpha1=sqrt(lambda1+(d/2-1)^2)` and string susceptibility `1-alpha1`.
The harmonic function for hitting the cone vertex is
`h(z)=|z|^(-alpha1-d/2+1)*m1(z/|z|)` (paper equation 41).
These spectral quantities are not graph dimension estimates or LQG couplings.

## Exact independent-coordinate construction

For `C=I_d`, the unit-time unbiased excursion consists of independent scalar
Brownian excursions. Each scalar excursion is a Bessel(3) bridge and is sampled
at grid times as the norm of three independent ordinary Gaussian bridges from
zero to zero. This representation applies to a scalar excursion, rather than
substituting an unconditioned bridge for the desired positive vector excursion.

The paper's numerical graph uses a **biased** excursion law. Relative to the
unbiased excursion X in the whitened cone, its density is proportional to

`integral_0^1 |X(t)|^(2*alpha1-2) dt`.

For the independent orthant, the positive spherical eigenmode is proportional
to the product of direction coordinates, `m1(s) proportional prod_i s_i`.
Its homogeneous degree is d, so `lambda1=d*(2*d-2)`,
`alpha1=3*d/2-1`; in d=3 these are 12 and 3.5. The code uses a conditional
bridge version of the paper's construction, avoiding a discretized walk on balls:

1. Draw positive S on the unit sphere with `S_i^2` distributed as
   `Dirichlet(3/2,...,3/2)`; this is exactly the `m1(S)^2` density.
2. Draw independent `T1,T2 ~ InvGamma(alpha1,scale=1/2)`.
3. Set `U=T1/(T1+T2)` and `H=S/sqrt(T1+T2)`.
4. Conditional on U and H, draw independent scalar Bessel(3) bridges from
   0 to H_i over [0,U], then H_i to 0 over [U,1]. A Bessel(3) bridge with one
   zero endpoint is the norm of a three-dimensional Gaussian vector bridge
   with endpoints `(0,0,0)` and `(H_i,0,0)` (or reversed).

These steps follow by conditioning equation (32)'s two h-transform paths on
their hitting times, as in equations (33)–(40). Given the endpoint and duration,
their Doob factors cancel in bridge probabilities, leaving the killed
Brownian bridges to the vertex. In the orthant these factor coordinate-wise.
The random junction U is inserted into the regular sampling grid. This is
mathematically equivalent to the paper's biased-law construction for I_d;
it does not claim to implement a general-cone walk-on-balls sampler.

### Exact interval minima

The graph only needs minima over its time cells. Given positive scalar
excursion values x,y over a duration dt, the intervening Bessel(3) bridge has
the same bridge law as ordinary Brownian motion conditioned to stay positive.
For `0<=z<=min(x,y)`, reflection gives

```
P(minimum>z | x,y, positive bridge)
 = [1-exp(-2*(x-z)*(y-z)/dt)]/[1-exp(-2*x*y/dt)].
```

`positive_bridge_minima` inverts this distribution, using `expm1`/`log1p`
and the stable quadratic root. Any segment touching the zero endpoint has
minimum zero. Conditional on all sampled values and U, disjoint segment bridges
are independent, including across coordinates in this independent case.
Taking the minimum over the subsegments of each graph cell therefore produces
**exact continuous-law cell minima**, up to floating-point arithmetic.
There is no time-step or spatial-grid bias in this branch. Oversampling is
optional and changes the random-number consumption, not the target law.

This distinction matters: minima of straight lines through a sampled excursion
would miss Brownian valleys and can change adjacency even with correct endpoint
samples. The exact branch does not use that approximation. The stored values
can be joined for illustration, but that illustrated polygonal curve is not
used to determine minima; the separately sampled minima are.

## General SPD construction: explicit approximation

For arbitrary positive-definite C, `sample_conditioned_bridge` targets the
finite-grid Gaussian bridge density

```
pi(x_1,...,x_(M-1)) proportional
 exp[-sum_j (x_(j+1)-x_j)^T C^-1 (x_(j+1)-x_j)/(2*dt)]
 * product_j 1{x_j>0},  x_0=x_M=0, dt=1/M.
```

It uses multiscale elliptical-slice updates. Conditional on the two endpoints
of a block, a Gaussian bridge has linear mean and covariance C times the
scalar bridge covariance. The update rotates the current centered block with
an independent bridge draw and shrinks an angular bracket until all interior
grid values are positive. Each update preserves this finite-grid target.
Randomly shifted block partitions at powers-of-two scales avoid permanently
fixed dyadic boundaries. Initialization is independent positive scalar
excursions, multiplied by optional `initial_scale` for overdispersed chain tests.

The finite-grid target is **not** the exact continuous excursion distribution:
between-grid crossings are ignored. The final retained state also has finite
burn-in error, and stationarity/mixing is not certified. Recorded traces are
mean coordinate height and whitened midpoint radius after each sweep. Multiple
chains, overdispersed starts, increasing sweep counts, and grid refinement must
be checked before using these samples to infer covariance dependence.

The graph for this branch uses piecewise-linear cell minima. A sampled knot
can be a local minimum exactly at a cell boundary, causing adjacent cell minima
to tie. A documented symbolic perturbation orders tied interior minima by cell
index; metadata reports the number of such ties per coordinate. This guarantees
a triangulation of the approximate path but adds a grid artifact. It vanishes
only after a suitable refinement limit has been established; no numerical rate
is assumed. End cells intentionally share minimum zero and are never separated
by this tie rule.

`sample_excursion(...,law='biased')` rejects general C explicitly. The general
biased law requires a sufficiently accurate cone eigenfunction and a validated
h-transform path sampler, which are not replaced by independent scalar
excursions, clipping, reflecting, or ordinary unconditioned Gaussian bridges.
The implemented FEM eigenpair is useful mathematical infrastructure, not a
claim that that sampler has been validated.

## Graph rule and edge conventions

Let m_i be the minimum on cell `[i/n,(i+1)/n]`, with zero-based labels.
For non-neighbors i<j other than (0,n-1), a coordinate contributes a diagonal iff

`max(m_i,m_j) < min(m_(i+1),...,m_(j-1))`.

This is precisely the paper's horizontal-segment criterion for generic continuous
minima. `triangulation_reference` checks it using a running minimum in O(n²)
time. `triangulation_fast` uses a monotone stack: pop larger minima when a new
smaller minimum arrives, connect each popped vertex and the surviving top to
the current vertex, and omit neighboring/cycle pairs. The stack traversal is
O(n). Current canonical ranking and sorted edge output make the overall
implementation O(d*n*log n) time and O(d*n) storage. Each coordinate has exactly
n-3 diagonals. Tests independently check hand examples, the reference rule,
noncrossing, the shared Hamiltonian cycle, connectedness, and multiedges.

The boundary Hamiltonian cycle is included once. A diagonal contributed by k
coordinates has multiplicity k; `coordinate_edges` retains all provenance.
Unit-edge shortest-path distances are measured in the underlying simple graph:
parallel unit edges give the same distances, while changing conductance and
random-walk transition probabilities. For degree/random-walk statistics, state
which convention is used.

## API examples

```python
import numpy as np
from src.covariance import from_correlations, from_angles, equicorrelated
from src.brownian_excursions import sample_excursion, sample_conditioned_bridge
from src.mated_crt import graph_from_excursion, validate_graph

rng = np.random.default_rng(20261110)  # NumPy default PCG64; save seed and version.
sample = sample_excursion(8192, np.eye(3), rng, law="biased")
graph = graph_from_excursion(sample, 8192)
assert validate_graph(graph)["connected"]
simple_sparse = graph.to_sparse()
multigraph_sparse = graph.to_sparse(multigraph=True)

# Supported, explicitly approximate; inspect diagnostics before metric fitting.
covariance = from_correlations(-.2, .1, .3)
approx = sample_conditioned_bridge(256, covariance, rng,
                                    oversample=8, sweeps=256, initial_scale=2)
approx_graph = graph_from_excursion(approx, 256)
```

`ExcursionSample` contains `times`, `values`, optional `segment_minima`, and
`metadata`. `MatedCRTGraph` contains adjacency lists, unique edge pairs,
multiplicities, coordinate diagonal arrays, and copied sampling metadata.
Metadata includes covariance, law, exactness flags, junction/hitting times for
the biased independent law, and MCMC diagnostics where applicable. Experiments
must additionally save independent realization seeds, software versions,
sample counts, and observable definitions; the sampler cannot infer a seed
from an already advanced NumPy generator.
