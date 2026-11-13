# Interior-ball construction of the biased excursion (Task 2B)

The target is Budd–Castro arXiv:2207.05355v1 §§3.3–3.4, 4.1, equations
(32), (33) and (41). `src/doob_excursions.py` supplies this construction as
a new, explicitly approximate sampler. Existing Bessel, Weyl and finite-grid
MCMC samplers remain separate and unchanged.

## Target and analytic harmonics

Let C=RRᵀ, W=R⁻¹R₊ᵈ and let u be the positive homogeneous harmonic function
vanishing on the cone faces, of degree p. The angular mode is m(s)=u(s),
alpha1=p+d/2−1, and the inward harmonic function is

```
h(x)=u(x)/|x|^(2p+d−2)=|x|^(−alpha1−d/2+1) m(x/|x|).
```

The biased construction samples S on the spherical section with density m²,
then two independent h-transformed Brownian motions from S to the vertex.
Reverse the first path, concatenate and normalize time to one and amplitude
by the square root of the total time. This is the paper's biased law, not
the original unweighted excursion law. The new API provides no silently
substituted `unbiased` option.

Analytical models cover every 2D wedge, and every 3D birectangular cone,
including all permutations of the independent coordinate. For a correlated
pair with rho and opening alpha=acos(−rho), choose orthonormal planar axes
e0,e1 so the wedge angle is 0<theta<alpha. With k=pi/alpha, its harmonic
function is `s^k sin(k theta)`. In d=3 multiply by the positive perpendicular
coordinate y. Thus p=k in d=2 and p=k+1 in d=3.

The spherical starting density can be sampled exactly: theta has density
`(2/alpha) sin²(k theta)` (uniform rejection, mean two proposals). In d=3,
independently sample `y²~Beta(3/2,k+1)` and set planar radius sqrt(1−y²).
This agrees with the independent and A2 special-cone angular distributions.

## Sphere exits and why rejection is controlled for analytic h

At x choose radius r=eta*D, where D=min_i(Rx)_i is the distance to the closest
face (the rows of R have unit length). The implementation takes eta<1,
default 1/2, so the ball is strictly inside W. Using smaller interior balls
than the paper's maximal balls does not change the exact Brownian h-transform;
it changes efficiency and numerical approximation errors.

The desired exit density on the sphere is proportional to h(x+r*omega).
For analytic harmonic h its normalizer is h(x). Positive harmonic-function
Harnack bounds on the larger ball of radius D give the rigorous envelope

```
h(x+r*omega)/h(x) <= (1+eta)/(1−eta)^(d−1).
```

Consequently default analytic 3D rejection accepts one in six proposals on
average, independently of proximity to a face. An observed envelope violation
raises an error. There is no exit-direction MCMC or burn-in.

## Interior paths, rotation and physical time

An isotropic fixed-length random walk from the ball center is stopped when
it first exits the ball. Its final segment is clipped to the sphere. Because
the walk is invariant under orthogonal maps, its exit direction is uniform
and independent of its exit time. An orthogonal Householder map rotates or
reflects the whole path so its exit agrees with the h-weighted sampled point.
Orthogonal reflection is valid as well as rotation by the symmetry of the
Brownian target and the discrete increment law.

A step of length ell in d dimensions has covariance `(ell²/d)I`, so its
Brownian clock is **dt=ell²/d**. The final clipped step receives its linear
time fraction. That last-step clock is an approximation to a Brownian
boundary-crossing time, and is refined rather than treated as exact.
The local length is `min(step_size,relative_step*ball_radius)`; reducing only
the global step does not refine arbitrarily small balls near the faces.
Both controls need sensitivity tests. No Gaussian bridge-minimum correction
is added independently across correlated coordinates.

The path is stopped inside `origin_cutoff`. For an exact harmonic cone model,
the remaining hitting time at radius r has exact law
`r²*InvGamma(alpha1,scale=1/2)`, independent of angular direction. The sampler
draws this remaining time but replaces that small remaining path by a straight
line to zero. The omitted true path can wander back outside the cutoff ball;
localization is probabilistic rather than deterministic. Its radial Bessel
index is −alpha1, so starting at radius r the exact probability that this
tail reaches radius R>r before zero is `(r/R)^(2*alpha1)`. Metadata records
this bound at `R=2*origin_cutoff` for analytical cones. Both this tail
approximation and the duration normalization require sensitivity checks. With
a numerical angular mode, even the radial-time parameter and this bound are
not certified (the latter is stored as null).

All raw polygonal vertices are retained. Graph-cell boundaries are inserted
by interpolation, without discarding finer walk vertices. Their union gives
the exact minima of this **polygonal path**, not the continuous Brownian path.
Increasing `oversample` alone cannot improve these minima: the random walks
and the cutoff must be refined. Metadata stores individual leg times, junction
directions, step/ball/proposal counts, cutoff radii and tail times, maximal
physical and normalized time steps, and angular-mode provenance.

## Numerical FEM modes remain provisional

An explicitly supplied eigenpair selects a numerical 3D angular mode. The
model uses a safe global envelope
`max(m_FEM)*(|x|−r)^(-alpha1−1/2)`, since nonnegative barycentric interpolation
is bounded by its nodal maximum. **The Harnack bound is not used for FEM**:
the interpolant is not harmonic. Its sphere average generally differs from
h(x), so rejection samples its locally normalized kernel rather than a
mathematically exact Doob transform. Accurate eigenvalues alone cannot
control this discrepancy or angular relative errors near the boundary.

The original P1 angular mesh also has corner triangles with all three
vertices on the boundary. Its eigenmode is identically zero in those open
interior patches. The sampler rejects meshes with unresolved zero-support
facets before drawing paths; it never fills those values with a positive
floor. Zero proposals are counted, and a zero-h ball center also raises an
explicit error. Adding a centroid alone failed because its degree of freedom
connects only to boundary nodes. Conforming splits of the opposite interior
edges and their neighboring triangles remove the zero patches. This repair
still fails near-corner relative-error and harmonic mean-value checks, so it
does not validate the general sampler.

Therefore arbitrary covariance sampling is an investigational feature until
angular mesh, local-kernel and full-path cross-checks pass. It must not be
used for a large parameter sweep based solely on positive outputs.

## Failure policy and current checks

The sampler raises on exhausted walk, ball or rejection budgets; it does not
retry the entire path and silently condition the sample on computational
success. Experiment scripts record failed attempts explicitly. The development
pilot is `python -m experiments.doob_development`; it measures cost and failures,
not distributional correctness. Distributional validation is a separate
workstream using Bessel/Weyl controls and multiple resolutions.

Unit tests check analytic harmonicity, orthant h and cone
spectral data, all coordinate permutations, spherical angular moments,
h-weighted exit first moments from an independent harmonic-Taylor identity,
Brownian ball clock, seed reproducibility, graph invariants and explicit
failure on an interior FEM zero patch.

```python
import numpy as np
from src.doob_excursions import sample_doob_excursion
from src.mated_crt import graph_from_excursion

C=np.array([[1.,-.5,0],[-.5,1.,0],[0,0,1.]])
sample=sample_doob_excursion(128,C,np.random.default_rng(123),
    step_size=.01,relative_step=.15,origin_cutoff=.005)
graph=graph_from_excursion(sample,128)
```
