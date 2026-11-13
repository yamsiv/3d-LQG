# Finite-grid 3D LFPP: conventions and independent validation

The new implementation is `src/lfpp_3d.py`; the checkpointed validation runner
is `experiments/phase3a_lfpp_validation.py`. Earlier graph samplers and results
are unchanged. Local original PDFs consulted: Contreras Hip–Zhuang
[arXiv:2504.09141v1](https://arxiv.org/abs/2504.09141v1), equations (1.1)–(1.3)
and Remark 1.1; Ding–Gwynne–Zhuang
[arXiv:2310.03996v5](https://arxiv.org/abs/2310.03996v5), equations (1.4)–(1.6)
and Theorem 1.2. The spectral torus field is a separate finite-volume model,
not the exact whole-space field or mollifier in those papers.

## Physical spacing and two path-cost conventions

For an N³ array, sites have coordinates i/N. Periodic distances use the unit
flat torus; an open grid spans [0,(N−1)/N]³. These settings are never silently
interchanged. A tagged periodic CSR is rejected by the cube-crossing helper.

For a fixed field h and xi, write w_i=exp(xi*h_i). No GMC variance subtraction
is applied to metric weights. Default axis-edge costs are

```
c(i,j)=(w_i+w_j)/(2N).
```

They define a proper finite weighted graph metric with zero diagonal. CHZ
equation (1.1) instead uses epsilon times the sum of **all vertex weights**
on a nearest-neighbor path, including both endpoints. With epsilon=1/N,

```
L_vertex(P)=L_trapezoid(P)+(w_start+w_end)/(2N).
```

`cube_crossing(graph,N,site_weights=exp(xi*h))` handles the unequal source
costs with one directed auxiliary source, then adds target half-site costs.
On an axis graph this exactly reproduces the paper's vertex-sum convention
on the supplied open grid. `paper_vertex_distances` does the same for point
costs; its finite-grid diagonal is w_i/N. That convention is kept separate
from the proper metric used for sampled metric-measure comparisons. For
xi=0, proper face distance is (N−1)/N and vertex-sum face cost is exactly 1.

If median crossings behave like N^(−lambda), the finite-resolution regression
is log(crossing) versus log N with slope −lambda. In the paper's conventions
lambda=1−xi*Q. DGZ's median normalizer notation is distinct from this exponent.
The implementation does not identify xi with gamma or infer subcriticality
from an arbitrary positive xi. See the separate parameter-convention protocol.

## Directional approximation and quadrature

`build_lfpp_graph(h,xi,stencil=...,boundary=...)` returns CSR plus metadata.
Available undirected-neighbor direction counts are 6 (`axis`),26 (`26`), and98
(`primitive2`, all primitive integer offsets with maximum coordinate 2).
Every undirected edge is evaluated once and inserted symmetrically.

An edge with offset v has physical length |v|/N. Its weight is integrated
along the straight segment; long edges do not skip intermediate high values.
The default `interpolation='weight'` uses the trilinear interpolant of exp(xi*h).
Axis integrals are then exactly trapezoidal. Other directions use composite
trapezoidal quadrature with ceil(q*|v|) subdivisions, default q=2.

An explicit alternative, `interpolation='field'`, integrates
exp(xi*trilinear(h)). Along an axis edge this integral is analytic: the
endpoint weight multiplied by exprel(xi*(h_end−h_start))/N. Other directions
use the same composite quadrature. These interpolation targets differ at
finite cutoff; changing the target is recorded in metadata. Paper vertex-sum
endpoint corrections apply to the default axis rule, not the field rule.

At xi=0 the axis metric is exactly L1. The richer stencils produce a more
isotropic **polyhedral** metric, not an exact Euclidean metric. Increasing N
with the same stencil does not make its angular norm Euclidean. This is a
controlled numerical comparison between directional approximations, not an
implementation of an isotropic eikonal continuum solver.

For N=32, all torus targets from one root give the following excess over the
exact flat Euclidean distance (excluding the root):

| Stencil | Mean excess | Maximum excess |
| --- | ---: | ---: |
| axis |54.49%|73.21%|
|26|8.08%|12.80%|
|primitive2|2.04%|4.93%|

N=16/24 controls show the same persistent anisotropy. Ensemble scale
normalization can remove much of a global distance excess but cannot remove
all directional or distance-shape differences.

## Executed cost and accuracy controls

The benchmark used a deterministic smooth trigonometric field to isolate
runtime, not as scientific log-field data. It performed 40-root searches:

| N | axis build+search (s) |26 build+search (s)|primitive2 build+search (s)|
| ---: | ---: | ---: | ---: |
|16|.020|.079|.346|
|24|.059|.260|1.151|
|32|.144|.650|2.923|
|48|.613|2.643|not needed/run|

The N=16 pilot and extrapolated costs were saved before larger cases. At
N=48,26-direction CSR uses 34.95MB and40-root distance output 35.39MB. Whole
process peak RSS was about 335MB after several graph constructions. Individual
searches are not independent samples of a random geometry. The memory/time
figures are machine-specific. Work is O(s*N³*q) for edge construction and
roughly O(R*(E+V)*log V) for R Dijkstra roots, with O(s*N³+R*N³) storage.

Four independent log-field realizations were coupled at N=16/24, with eight
uniform roots per field and xi=.2,.4,.6. Every comparison uses identical field
values and roots. The96 saved rows are not96 independent fields. Relative to
axis distances,26 directions reduce average pointwise distances by about
18–23%; primitive2 reduces them another 1.9–3.2% relative to26. Doubling
primitive2 quadrature from 2 to 4 changes average distances by about +.06–.16%.
These raw-scale changes must not be confused with ensemble-normalized
cross-model goodness-of-fit statistics.

A separate 24-row check reuses those saved fields and roots, comparing the
two interpolation targets with quadrature 4. At N=24, the field-interpolation
mean distance is lower by about .38%,1.45%,3.35% for xi=.2,.4,.6. After
normalizing each realization by its own mean, observed pair-law W1 differences
are about .00015,.00084,.00489 respectively. This latter normalization is only
a discretization diagnostic; the primary cross-model analysis retains
deterministic ensemble scales and global random fluctuations. Four fields
are insufficient for a precision systematic-error claim.

## Correctness tests and reproducibility

Ten focused tests cover exact L1 torus/cube distances, stencil directions,
flat anisotropy, symmetric weights, constant-field shifts, integration across
an intermediate high-weight site, exact axis exponential integrals, error
handling, and paper crossings against an independently assembled directed
vertex-cost graph. They pass; this does not prove continuum convergence.

Saved raw/checkpoint records, fields, roots, configurations and source hashes
are under `results/phase3a/lfpp_validation/`. Reproduce into a new output
directory, because each mode refuses to replace existing records:

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3a_lfpp_validation --mode benchmark --output results/reproduction/phase3a_lfpp
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3a_lfpp_validation --mode nonzero --output results/reproduction/phase3a_lfpp
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3a_lfpp_validation --mode interpolation --output results/reproduction/phase3a_lfpp
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3a_lfpp_validation --mode summarize --output results/reproduction/phase3a_lfpp
```

Remaining systematics are the field cutoff and interpolation, fixed directional
anisotropy, spatial resolution, finite domain/topology and measure sampling.
They are distinct from between-field Monte Carlo uncertainty. No finite-grid
fit is asserted to be an established continuum exponent.
