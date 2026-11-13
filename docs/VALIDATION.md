# Independent validation notes

Source: Timothy Budd and Alicia Castro, *Scale-invariant random geometry from
mating of trees: a numerical study*, [arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1),
local `papers/Budd_Castro.pdf`, 28 pages. Sections 2.5, 3.1–3.4 and 4 were read
directly; Table 2 on PDF page 24 was also rendered and visually checked because
plain-text extraction interleaves its columns. These notes concern graph metrics
only.

## Independent graph oracle

Let `m[i]` be the minimum of a scalar excursion on graph cell `i`. For
nonneighboring `i<j`, the horizontal segment condition of §2.5 is equivalent,
for generic distinct interior minima, to

```
max(m[i], m[j]) < min(m[i+1:j]).
```

The distinguished edges `(i,i+1)` and `(0,n-1)` are inserted separately. The
first and final cell have minimum zero. Their tie is harmless because their
connecting edge is already the boundary edge. A cubic oracle evaluates every
interior comparison literally, without the production monotonic stack.

Hand checks for six vertices:

| Cell minima | Diagonals |
| --- | --- |
| `[0,5,2,7,3,0]` | `(0,2), (2,4), (2,5)` |
| `[0,3,7,2,5,0]` | `(0,3), (1,3), (3,5)` |

`tests/test_independent_validation.py` checks the oracle for all interior-rank
permutations at `n=3,…,8`, including `n-3` diagonals and noncrossing. This is a
deterministic combinatorial check; it cannot establish the sampler's law.

The boundary cycle is shared once by the polygons. Coincident interior
diagonals from different coordinates are distinct parallel edges. Unweighted
shortest-path distance is unchanged on passing to the underlying simple graph;
random-walk conductances need an explicit choice.

## Independent-coordinate exact construction

For the positive orthant in dimension `d`, the homogeneous positive harmonic
function for survival is `u(x)=prod(x_i)`, of degree `d`. Thus the paper's
spectral parameter is `alpha1=3d/2-1`, the spherical mode is proportional to
`prod(s_i)`, and its inward harmonic function (41) is
`h(x)=prod(x_i)/|x|^(3d-2)`.

The biased construction (32) can therefore be sampled without discretizing a
Doob stochastic differential equation: sample `S_i²` from a symmetric
Dirichlet distribution with concentration `3/2`, sample independent
`T1,T2 ~ InvGamma(alpha1,scale=1/2)` as in (33), and condition on these variables.
Each coordinate of each leg is then a Bessel-3 bridge between zero and `S_i`
(one leg is reversed). Its finite-dimensional distributions can be sampled as
norms of three-dimensional Gaussian bridges with vector endpoints zero and
`(S_i,0,0)`. The scalar bridges are independent conditional on the junction,
but the full coordinates need not be independent under the biased law.

For positive scalar endpoints `x,y`, a Bessel-3 bridge of duration `dt` is a
Brownian bridge conditioned positive. Its minimum `M` has survival function

```
P(M>z | x,y, positive) =
  [1-exp(-2*(x-z)*(y-z)/dt)] / [1-exp(-2*x*y/dt)],
  0 <= z <= min(x,y).
```

This follows from reflection for killed one-dimensional Brownian motion; Doob
endpoint factors cancel upon conditioning on both endpoints. It permits exact
cell-minimum sampling. A cell containing the random junction must be split at
the junction, with its two subinterval minima combined. A cell containing an
excursion endpoint has minimum zero. This removes finite-time-grid error in the
independent-coordinate graph construction; it does not remove finite-graph-size
effects from estimating metric exponents.

Useful distribution checks derived independently:

- The normalized junction time `U=T1/(T1+T2)` has distribution
  `Beta(alpha1,alpha1)`.
- `E[T_i]=1/(2*(alpha1-1))` when `alpha1>1`.
- For the unbiased independent excursion at fixed `t`,
  `sum_i X_i(t)²/[t*(1-t)] ~ chi²(3d)`, consistent with (35).

## Parameter and benchmark checks

The user convention is `(a,b,c)=(C12,C13,C23)`. Paper equation (28) uses
`(alpha,beta,theta)=acos(-(C12,C23,C13))`, where `theta` replaces the paper's
third spherical angle called gamma. Positive definiteness, rather than only
an angle-sum inequality, is the numerical admissibility check.

Selected Table 2 targets on the analytically tractable birectangular slice are:

| `alpha` | `beta=theta` | Paper `d_H` |
| --- | --- | --- |
| `pi/4` | `pi/2` | `4.64 ± 0.13` |
| `pi/3` | `pi/2` | `4.67 ± 0.04` |
| `pi/2` | `pi/2` | `4.83 ± 0.06` |
| `2*pi/3` | `pi/2` | `5.15 ± 0.09` |

The near-corner equilateral case `alpha=beta=theta=pi/3+0.005` gives
`4.08 ± 0.04` in Table 2. Table 1 gives `3.59 ± 0.07` for the independent
two-coordinate case and `4.07 ± 0.14` for the Brownian-map control.

The paper's §4.2 uses volumes `2^11,…,2^19`, uniform roots and all distances
from those roots. It applies shifted histogram collapse, then a leading
correction-to-scaling fit, and combines batching uncertainty with fitting-range
systematics. Its text says 80,000 independent graphs per volume while Figure
12's caption says 10,000 graphs with 10 roots for that illustrated case; this
internal sample-count difference should not be silently resolved. A modest
log-mean-distance regression is a pilot diagnostic with a different error budget.

## Secondary observables and tests

`src/secondary_observables.py` supplies exact small-graph normalized trace
returns, weighted Monte Carlo returns, a windowed spectral slope and small-graph
effective resistance. The default lazy walk has holding probability `1/2` to
suppress parity artifacts. Roots are uniform, so the finite-volume plateau of
the averaged return probability is `1/n` for a connected lazy graph. Choosing a
stationary root instead gives plateau `sum_v pi(v)^2`.

Monte Carlo standard errors in that module condition on one graph. They do not
measure between-graph fluctuations. A spectral slope is only reported when a
window remains above five times the uniform-root plateau; it is explicitly a
finite-graph window slope, not an asymptotic spectral-dimension claim.

Exact unit controls include the lazy triangle spectrum `(1,1/4,1/4)`, its unit
edge resistance `2/3`, and a weighted triangle Monte Carlo comparison. Final
independent test execution: 12 passed in 0.93 seconds.

## Executed distribution diagnostics

Command:

```
.venv/bin/python -m experiments.validate_sampler --output results/validation
```

Seed `2026111003`, NumPy PCG64, 4,000 independent exact-law samples. Raw draws,
parameters and summaries are in
[`sampler_raw.npz`](../results/validation/sampler_raw.npz) and
[`sampler_validation.json`](../results/validation/sampler_validation.json);
the comparison figure is
[`sampler_validation.png`](../results/validation/sampler_validation.png).

| Check | Observed result |
| --- | --- |
| `T ~ InvGamma(3.5,0.5)` | KS statistic `0.01064`, p-value `0.323`; mean `0.198879` vs `0.2` |
| `U ~ Beta(3.5,3.5)` | KS statistic `0.01588`, p-value `0.262` |
| Unbiased squared midpoint radius divided by `0.25` | chi-square(9) KS statistic `0.00746`, p-value `0.978` |
| Exact positive-bridge minimum survival at four levels | maximum empirical discrepancy `0.000639` over 100,000 draws |

These checks are consistent with the independent-coordinate formulas. A p-value
is not the probability that an implementation is correct.

For the generic ESS implementation, an additional independent oracle samples
the **finite-grid** positive Gaussian bridge at `C=I3` by cyclically shifting
each coordinate's Gaussian bridge increments at their unique grid minimum.
Cyclic invariance gives one positive rotation per orbit, hence the exact
discrete-time conditioned target. Independent rotations are valid only at
identity covariance. At 64 steps, 4,000 such reference paths have mean midpoint
coordinate `0.720389`. Comparing 48 independent ESS chains for each run length:

| Sweeps | Mean midpoint coordinate | Between-chain SE | Two-sample KS p-value |
| --- | --- | --- | --- |
| 16 | `0.72170` | `0.03132` | `0.405` |
| 64 | `0.71115` | `0.02973` | `0.941` |
| 256 | `0.70868` | `0.02499` | `0.820` |

This supports the implementation of the ESS transitions in one small known
case. It does not certify burn-in for general correlations or larger grids.
The continuum unbiased midpoint-coordinate mean is `sqrt(2/pi)=0.797885`,
which also shows why positivity only at 64 grid points cannot be equated with
the continuous excursion law.

**Validation gate:** the exact independent-coordinate generator is suitable
for preliminary multi-size metric experiments. General covariance graphs remain
finite-grid, finite-burn-in approximations and need additional grid/mixing
validation before interpreting fitted exponents as Budd–Castro dimensions.

## Special correlated A2 Weyl-cone construction

The extension `src/weyl_excursions.py` uses three ordered Hermitian-matrix
eigenvalues. With diagonal Brownian variance `dt` and off-diagonal real and
imaginary variances `dt/2`, adjacent eigenvalue gaps divided by `sqrt(2)` have
Brownian covariance `[[1,-1/2],[-1/2,1]]`. Their cone is a wedge of angle
`pi/3`. Subtracting the trace removes a common independent scalar motion and
does not change gaps. An optional independent Bessel-3 bridge supplies the
third positive coordinate.

The foundational identification is the Vandermonde Doob transform of killed
Brownian motion with Hermitian Brownian eigenvalues; see Katori–Tanemura,
[*Noncolliding Brownian motions and Harish-Chandra formula*](https://arxiv.org/pdf/math/0306386),
v2, introduction and equations (1.1)–(1.2). The Hermitian normalization is given
explicitly on its first page. The following bridge and bias specialization is
our derivation using that identification and Budd–Castro (32)–(41).

Because a matrix Brownian path starting at zero is invariant under a fixed
unitary conjugation, its entire eigenvalue path is independent of the final
eigenvector orientation. Conditioning on a fixed diagonal matrix endpoint
therefore gives the eigenvalue bridge conditioned only on its final ordered
eigenvalues. Reversing gives the other leg. At fixed time, the traceless GUE
eigenvalue density is a Gaussian times the squared Vandermonde polynomial.
Its radial dimension is 8. Adding an independent chi-3 coordinate increases
the radial dimension to 11. Normalizing these Gaussian-polynomial samples
gives exactly the angular `m1²` density needed by the biased construction.
The corresponding `alpha1` values are 3 (two coordinates) and 4.5 (three).

Independent tests in `tests/test_weyl_validation.py` verify:

- the wedge angular CDF `theta/alpha - sin(2*pi*theta/alpha)/(2*pi)`;
- the squared extra-axis coordinate's `Beta(1.5,4)` spherical marginal;
- unbiased midpoint radial chi-square laws with 8 and 11 degrees of freedom;
- short-time quadratic variation, including covariance `C12=-1/2`;
- actual junction heights and bias spectral parameters.

All five tests passed. Exact grid values do not give exact cell minima for the
correlated eigenvalue gaps. The implementation uses piecewise-linear minima
and records symbolic tie resolutions. Its numerical graph validation therefore
also requires nested time-grid refinement.

Executed command:

```
.venv/bin/python -m experiments.weyl_resolution --output results/weyl_resolution
```

This used 24 independent biased paths per dimension at `n=512`, seed
`2026111004`, with eight identical uniform roots across each path's nested
time grids (8, 32 and 128 time samples per graph cell). Data and graph-cluster
paired comparisons are in
[`resolution.json`](../results/weyl_resolution/resolution.json), the cell minima
are in [`nested_minima.npz`](../results/weyl_resolution/nested_minima.npz), and
the figure is [`resolution.png`](../results/weyl_resolution/resolution.png).

| Coordinates | Samples/cell | Mean paired distance change vs 128 | SE across paths | Edge Jaccard vs 128 |
| --- | --- | --- | --- | --- |
| 2 | 8 | `+0.092%` | `0.399%` | `0.817` |
| 2 | 32 | `−0.082%` | `0.393%` | `0.914` |
| 3 | 8 | `−0.808%` | `0.357%` | `0.808` |
| 3 | 32 | `+0.155%` | `0.249%` | `0.905` |

The coarse 8-sample grid exhibits a detectable distance change in the
three-coordinate case. The 32-versus-128 comparison is consistent with small
mean-distance error at this size, while about 9–10% edge disagreement remains.
Thus average distances can stabilize earlier than local edge observables.
The mean number of tie resolutions per coordinate falls from about 19.6 (8)
to 5.1 (32) to 1.1 (128) in the three-coordinate case. A 128-sample grid is
still finite; these results do not certify its continuum error or its behavior
at every graph size. They support a cautious special-cone pilot when accompanied
by a multi-size resolution comparison.

The production special-cone pilot's 100-replicate graph-cluster shifted-collapse
bootstrap is saved in
[`weyl_d2_o128.collapse.json`](../results/data/weyl_d2_o128.collapse.json) and
[`weyl_d3_o128.collapse.json`](../results/data/weyl_d3_o128.collapse.json). It uses
16 independent paths per size `256,…,8192` and the 128-sample grid. For two
coordinates, the effective dimension is `3.200`, Monte Carlo 95% interval
`[2.731,4.117]`, with fitting-window centers `3.186–3.274`. For three coordinates,
the value is `4.916`, interval `[4.249,6.272]`, with centers `4.674–5.016`.
These are statistically compatible controls relative to the paper's `3.13`
and `4.67`, but their broad intervals do not constitute precision replication.
The final correction ansatz (45) was not fitted, and Monte Carlo intervals
exclude remaining discretization and asymptotic-fit errors.

## Independent interpretation of additional observables

The separately executed
[`additional/observables.json`](../results/additional/observables.json) contains
eight independent `I3` biased graphs at each of `n=8192,32768`, 12,000 lazy
walks per graph, and both simple and multiplicity conductances. Simple-walk
window slopes `−2 d log p(t)/d log t` are:

| Size | Window | Mean slope | Between-graph SE |
| --- | --- | --- | --- |
| 8192 | 8–64 | `2.272` | `0.016` |
| 8192 | 16–128 | `2.285` | `0.034` |
| 32768 | 8–64 | `2.233` | `0.014` |
| 32768 | 16–128 | `2.291` | `0.025` |

Multiplicity-walk slopes lie in the same approximate range, `2.22–2.32`.
The returns at time 128 are about `0.0022`, still well above the uniform-root
finite-volume plateau (`0.0000305` at `n=32768`). Nevertheless, 12,000 walkers
give only about 26 returns per graph at time 128, and the diagnostic filter
retaining counts at least 20 can select noisy high-count observations at late
times. These are short finite-window slopes; neither a stable asymptotic
spectral dimension nor a contradiction of spectral dimension 2 is established.

Eight independent `n=128` graphs with 32 uniform vertex pairs each give mean
simple-graph effective resistance `0.47158` (between-graph SE `0.02041`), and
multiplicity-conductance resistance `0.45018` (SE `0.01924`). The reduction is
consistent with Rayleigh monotonicity. These pair averages are finite-graph
diagnostics, not resistance-scaling exponents.
