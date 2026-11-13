# Budd–Castro mated-CRT graphs: construction, validation and preliminary metric geometry

Research date: **10 November 2026, America/New_York**. Scope: graph construction,
sampling, intrinsic metrics and numerical scaling only. Four agents divided
coordination, construction, metrics and independent validation. No field,
LFPP or GMC implementation was started.

The independent-coordinate construction is validated and scales to 524,288
vertices. A second, correlated special cone has a validated exact grid law and
quantified polygonal-minimum error. The main independent 3D benchmark is
compatible with the published dimension after finite-size corrections, with
substantially broader uncertainty. **General covariance-dependent dimensions
have not been reliably estimated:** finite-grid and mixing diagnostics prevent
opening the extensive-sweep gate. This report delivers the completed work and
the specific obstacle permitted by the task's completion criteria.

## 1. Mathematical definitions and source authority

The primary source is Budd and Castro, *Scale-invariant random geometry from
mating of trees: a numerical study*, [arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1),
published as Phys. Rev. D 107, 026010 (2023). The local PDF is v1, dated July
2022; §§2.5, 3 and 4 were read directly. Table 2 was also visually checked
because PDF text extraction interleaves its columns. The arXiv landing page
listed one version when checked. Mathematical definitions below follow this
paper; additional sampling derivations are identified separately.

The user convention is

```math
C=\begin{pmatrix}1&a&b\\a&1&c\\b&c&1\end{pmatrix},
\qquad (a,b,c)=(C_{12},C_{13},C_{23}).
```

This ordering differs from the earlier context document's labels for `b,c`.
The paper's angles satisfy `alpha=acos(-a)`, `beta=acos(-c)`,
`theta_angle=acos(-b)`, equation (28). All angles are radians;
`theta_angle` replaces the paper's third angle gamma. Covariances are checked
by Cholesky decomposition; angle-sum conditions alone are insufficient.
Equicorrelation is admissible for `-1/2<rho<1`. The boundary at `rho=-1/2`
has rank two, and is never sampled as positive definite.

Write `C=RR^T` and `W=R^{-1}R_+^3`. Standard Brownian motion in W maps under R
to covariance-C motion in the octant. The original finite-volume process is a
duration-one excursion from the cone vertex to itself, positive in the
interior. The principal spherical Dirichlet eigenpair obeys
`-Delta_S² m1=lambda1 m1`; the paper's spectral parameter is
`alpha1=sqrt(lambda1+1/4)` and susceptibility is `gamma_s=1-alpha1`.
Neither is a cone angle or graph dimension.

**Paper's numerical ensemble.** Section 3.3 weights the original excursion
law by `I(X)/E[I(X)]`, where
`I(X)=integral_0^1 |X(t)|^(2 alpha1-2) dt` and
`E[I(X)]=2^(alpha1-1) Gamma(alpha1)/alpha1`, equations (32)–(40).
The norm is measured in the whitened cone. Draw `S` on its spherical section
with density `m1²`, and two paths starting at S conditioned to hit the boundary
at the vertex. Their independent hitting times are
`InvGamma(alpha1,scale=1/2)`. Reverse the first path, concatenate, and rescale
time by their total duration and space by its square root. The Doob function is
`h(x)=|x|^(-alpha1-1/2) m1(x/|x|)`, (41).

Biased and original laws have different finite-volume distributions. The paper
motivates the same local critical exponents; absolute continuity alone does
not prove an asymptotic metric-identification theorem.

**Graph definition.** Each scalar coordinate supplies n cell minima `m_i`.
Non-neighboring i<j have a diagonal precisely when
`max(m_i,m_j)<min(m_(i+1),...,m_(j-1))`, with the generic-minimum convention
of §2.5. Add the shared Hamiltonian boundary cycle. Each tree contributes
exactly n−3 noncrossing diagonals. Coincident diagonals from different trees
retain multiplicity and coordinate provenance; the common cycle is counted
once. Distances use unit edges on the underlying simple graph, which leaves
unweighted shortest paths unchanged.

Full derivations and conventions are in
[MATHEMATICAL_METHODS.md](../docs/MATHEMATICAL_METHODS.md).

## 2. Implementation and approximation hierarchy

The source is organized into covariance, excursions, triangulations, metrics,
scaling and secondary-observable modules under `src/`. Experiments are separate
from reusable code. No dense adjacency matrix is allocated for production.

**Independent coordinates, exact continuous-law graphs.** A scalar unbiased
excursion is a Bessel-3 bridge, sampled as the norm of a three-component
Gaussian bridge. In the d-dimensional orthant, `m1(s)` is proportional to
`product(s_i)`, with `alpha1=3d/2-1`. For the biased law, draw
`S_i²~Dirichlet(3/2,...,3/2)`, the two inverse-gamma times, and their normalized
junction U and height H. Conditional on these, each coordinate consists of two
independent Bessel-3 bridge legs, 0→H_i→0. This is an analytically tractable
specialization of the paper's law, rather than its bounded-step approximation.

For positive scalar endpoints x,y over a segment of duration dt, reflection
gives the conditional minimum survival probability

```math
\Pr(M>z\mid x,y,\mathrm{positive})=
\frac{1-\exp[-2(x-z)(y-z)/dt]}{1-\exp[-2xy/dt]},
\quad 0\le z\le\min(x,y).
```

Doob endpoint factors cancel for a fixed-endpoint Bessel bridge. Inverting this
formula gives the exact segment minima. The random junction is inserted into
the skeleton; both subsegments of a junction-straddling graph cell are included.
Consequently the independent graphs have **no time-grid/minimum approximation**
apart from floating-point arithmetic. Their biased coordinates are independent
conditional on the junction, not necessarily unconditionally independent.

**Correlated special cone, exact grid law.** An A2 Weyl-chamber construction
provides adjacent ordered eigenvalue gaps of a 3×3 Hermitian Gaussian matrix
bridge, divided by sqrt(2). Their covariance is `C12=-1/2`. For d=3 add a
positive independent coordinate, giving
`C=[[1,-.5,0],[-.5,1,0],[0,0,1]]` and angles `(pi/3,pi/2,pi/2)`.
The unbiased and paper-biased grid distributions are sampled directly;
`alpha1=3` in d=2 and `4.5` in d=3. The biased angular sample uses normalized
traceless GUE eigenvalues, plus an independent chi-3 coordinate in d=3.
Conditional matrix bridges produce the two legs. This implementation derives
from Dyson/noncolliding-bridge theory, not a prescription quoted from
Budd–Castro. See [WEYL_CONSTRUCTION.md](../docs/WEYL_CONSTRUCTION.md) and the
primary [Katori–Tanemura paper](https://arxiv.org/abs/math/0306386).
Joint gap minima remain **polygonal approximations** and require refinement.

**General SPD, provisional finite-grid law.** Multiscale elliptical-slice
updates target the Gaussian bridge conditioned positive at a finite grid.
The exact block transitions preserve that discrete target, but finite burn-in
and missed continuous boundary crossings remain. Piecewise-linear minima
have ties at shared endpoints; explicit lexicographic infinitesimals resolve
them. This is a labeled approximation, not the general biased h-transform
sampler. Requesting an unsupported general biased law raises an error.

**Cone eigenpairs.** A separate surface-FEM module computes approximate
spherical modes and h values for arbitrary SPD cones. Birectangular eigenvalue
checks at alpha=pi/3,pi/2,2pi/3 converge toward 20,12,8.75. At refinement 6,
relative errors were about 0.11–0.13%. These are numerical approximations,
not certified eigenvalue bounds; this module is not a validated general
excursion sampler. [Data and figure](cone_diagnostics/data.json),
[eigenvalue plot](cone_diagnostics/cone_eigenvalues.png).

**Complexity and resources.** The pairwise reference adjacency routine is
O(d n²). The optimized visibility stack is linear after preprocessing;
ranking ties and canonical sorting make the current complete implementation
O(d n log n), with O(d n) memory. A list-based BFS costs O(n+m); production
uses SciPy's compiled unweighted sparse search. Pivot diameter bounds require
several such searches. The special correlated sampler additionally costs
O(q n) small-matrix operations for q samples per cell.

A fresh-process benchmark at n=524,288 took 0.27 s to sample independent minima,
5.90 s to construct the graph, and 1.55 s for four-root metric measurements
including diameter pivots. Peak process RSS was about **799 MB**; the graph
had 1,985,078 simple edges. These are observed single-machine timings, not
portable performance guarantees. [Runtime/memory records](validation/runtime_memory.json).

## 3. Validation and comparison with the paper

The test suite currently has **54 passing tests**. Checks include all interior
rank permutations for n=3,...,8 against an independent adjacency oracle,
hand examples, noncrossing diagonals, exact diagonal counts, Hamiltonian cycle,
connectedness, sparse multiplicities, seed reproducibility, covariance and
angle conversion, metric comparison to an independent all-pairs algorithm,
and synthetic scaling controls. Distribution diagnostics used 4,000 independent
exact-law samples. Inverse-gamma hitting times had mean 0.19888 vs 0.2 and KS
p=0.323; junction times matched Beta(3.5,3.5), p=0.262; the unbiased midpoint
radius matched chi-square(9), p=0.978. These checks support implementation
correctness without constituting proofs based on p-values.

The correlated control also passed independent angular-CDF, radial,
quadratic-variation, junction and eigenvalue-normalization checks. At n=512,
24 independently sampled fine paths per dimension were thinned to nested grids.
For 32→128 samples per cell, paired mean-distance changes were −0.08%±0.39%
SE (d=2) and +0.16%±0.25% SE (d=3). Edge Jaccard agreement was only about
0.91 and 0.90: stable average distances do not establish convergence of local
degree/edge observables. The six-size production control below kept the mean
relative resolution changes within about 0.65% at every size.

The paper fits uniform-pair histograms on n=2^11,...,2^19, first collapsing
them with additive shifts and then fitting correction ansatz (45). It reports
80,000 graphs per size in the text; Figure 12 instead describes 10,000 graphs
and ten roots for that example. This ambiguity is retained rather than
inventing a resolution. Our sample counts are much smaller. We reproduce the
two-stage shifted collapse but do **not** claim an identified fit of (45).

| Model, biased law | Sizes and independent graphs | Paper dimension | Shifted-collapse diagnostic, MC 95% interval |
| --- | --- | --- | --- |
| Three-tree C=I3; all angles pi/2 | 9 sizes 2,048–524,288; 568 total; 8 roots/graph | 4.83 ± 0.06 | 5.070 [4.254, 6.008] |
| Two-tree C=I2; angle pi/2, gamma_2=sqrt(2) | 6 sizes 2,048–65,536; 48/size; 8 roots | 3.59 ± 0.07 | 3.312 [2.963, 3.729] |
| Three-tree C12=-.5, C13=C23=0; angles pi/3,pi/2,pi/2 | 6 sizes 256–8,192; 16/size; 8 roots; q=128 | 4.67 ± 0.04 | 4.916 [4.249, 6.272] |
| Two-tree C12=-.5; angle pi/3, gamma_2=sqrt(4/3) | 6 sizes 256–8,192; 16/size; 8 roots; q=128 | 3.13 ± 0.05 | 3.200 [2.731, 4.117] |

The printed paper uncertainties are not assumed to be 95% intervals. Our
intervals are independent-graph bootstrap Monte Carlo intervals only. These
comparisons are compatible controls, **not precision replications**. The
additional positive-correlation 3D Table 2 point C12=.5 has only provisional
finite-grid data here. The rho≈−.5 dimension minimum and the Brownian-map
two-tree C12=.5 control have not been reproduced reliably.

See [independent validation](../docs/VALIDATION.md),
[sampler diagnostics](validation/sampler_validation.json), and
[coupled correlated resolution data](weyl_resolution/resolution.json).

## 4. Intrinsic distances, diameter and ball growth

All pair laws choose independent uniform vertices, including self-pairs.
Eight uniform roots per graph are followed by all targets; the full per-root
histograms are saved. Graph realizations are uncertainty clusters. The primary
combined study joins two independently seeded runs and checks for duplicate
graph seeds. Per-size sample counts are below; it contains no synthetic or
reweighted observations.

| n | Graphs | Mean distance ± graph MC SE | Mean diameter lower–upper bounds | Mean apparent ball slope |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 64 | 7.392 ± 0.048 | 16.47–22.73 | 2.920 |
| 4,096 | 64 | 9.003 ± 0.068 | 19.95–27.55 | 3.003 |
| 8,192 | 64 | 10.661 ± 0.066 | 23.70–32.78 | 3.246 |
| 16,384 | 64 | 12.512 ± 0.078 | 28.33–39.33 | 3.378 |
| 32,768 | 88 | 14.832 ± 0.079 | 33.41–46.47 | 3.578 |
| 65,536 | 88 | 17.504 ± 0.102 | 39.16–56.05 | 3.687 |
| 131,072 | 88 | 20.409 ± 0.123 | 45.88–64.72 | 3.817 |
| 262,144 | 24 | 23.982 ± 0.206 | 54.58–76.21 | 3.875 |
| 524,288 | 24 | 28.179 ± 0.269 | 62.63–88.71 | 3.976 |

**Fit dependence is the main numerical finding.** A naive log-mean fit across
all nine sizes gives inverse slope **4.190 [4.141,4.239]**. Excluding small
sizes moves it toward about 4.36. This narrow Monte Carlo interval excludes
the printed paper central value, but does not include finite-size bias.
A weighted additive-offset fit
`E[d]=A*(n/nmax)^p+B` gives **1/p=4.833 [4.527,5.133]**, with
`B=-2.140`, chi-square 4.950 on six residual degrees of freedom and zero
bootstrap fits at the allowed p boundaries. Its numerical proximity to the
paper was not a fitting constraint. Excluding the smallest size gives
4.538 [4.197,4.924]; shorter windows become weakly identified. These are
model-sensitive effective exponents rather than a final asymptotic estimate.

The independently fitted histogram method gives 5.070 with common shift 3.143,
and broad graph-bootstrap interval [4.254,6.008]. All correction models,
windows and bootstrap settings are retained. No fitting window was selected
for closeness to 4.83. The combination of compatible corrected estimates and
unstable naive slopes explains the discrepancy more plausibly than treating
the naive result as a new dimension measurement.

Median, lower-quartile and upper-quartile naive inverse slopes are 4.179,
4.011 and 4.349. Integer-valued quantiles cause discrete bootstrap intervals;
they offer shape checks, not independent high-precision measurements.
Mean-normalized distance histograms approximately align, with residual shape
and lattice effects. An additional unbiased C=I3 study has 288 independent
graphs over 2,048–65,536 vertices; its naive inverse slope is 3.986 and shifted
collapse 4.224 [3.762,4.776]. Its different finite-size fits are not a proof of
different asymptotic exponents for the two laws.

Diameter estimates use certified eccentricity/triangle-inequality bounds;
large-graph values are never labeled exact. Exact n=32 examples had diameters
7 for two trees and 4 for three trees. In the main study, naive inverse slopes
for average lower and upper diameter bounds were 4.156 and 4.072. Their slopes
do not themselves bound an asymptotic dimension.

Ball profiles count vertices, with uniform roots. The slope diagnostic uses
r≥2 and volumes between 2% and 25% of the whole graph, requiring at least three
radii. Its strong size drift from about 2.92 to 3.98 shows that these radii do
not deliver a stable Hausdorff estimate. Compact-volume saturation, microscopic
radius shifts, few available radii and root variability all matter. There is
no Euclidean boundary in these cycle-glued graphs; the distinguished excursion
root is not substituted for a uniform metric root.

![Distance, histogram, ball and diameter results](figures/i3_biased_combined_geometry.png)

![Fit-window and degree diagnostics](figures/i3_biased_combined_reliability.png)

The complete [metric table](tables/metric_summary.csv),
[fits](data/i3_biased_combined.analysis.json),
[shifted means](data/i3_biased_combined.corrections.json), and
[collapse bootstrap](data/i3_biased_combined.collapse.json) preserve more
digits and all fitting windows.

## 5. Covariance dependence and the validation checkpoint

A deliberately small coarse diagnostic used three sizes 128,256,512, four
independently seeded chains per matrix/size and nine matrices: equicorrelations
rho=−.49,−.30,0,.30,.60; the birectangular C12=−.5 and +.5 with the other
correlations zero; and unequal triples (−.2,.1,.35), (−.35,−.1,.2).
Full matrices, cone angles, eigenvalues, condition numbers and traces are
saved. All graph combinatorial invariants passed. This is 108 small graphs,
not an extensive dimension sweep.

At n=512, the five equicorrelation surrogate mean distances were respectively
4.758,4.986,4.976,5.044,6.040. These values are retained as finite-grid data
only: approximately 14% of cell minima were tied at two samples per cell.
They cannot support a continuum monotonicity claim or d_H(rho) catalogue.

An independent-seed sensitivity study generated 96 more small graphs.
At identity, n=512, mean distances were 4.907±.048 SE for q=2/32 sweeps,
4.976±.062 for q=2/128 sweeps, 5.196±.144 for q=8/128 sweeps, and
5.178±.056 from the exact unbiased continuum sampler. Tie fractions dropped
from roughly 14% to 3.8%, and to zero for exact minima.

At rho=−.49 the covariance condition number is 74.5. For n=512 and 128
sweeps, the area diagnostic drifted from first-quarter to last-quarter means
0.3757→0.2216 at q=2 and 0.3719→0.2278 at q=8. Increasing resolution did
not remove the visible transient. A small between-chain standard error would
not cover this burn-in error. The near-degenerate angle triple is
`(acos(.49),acos(.49),acos(.49))`, not the paper's still narrower
`pi/3+.005` point.

**Decision: keep the general-covariance dimension gate closed.** The code
accepts the full three-parameter domain, and the family has been explored as
a sampler diagnostic, but credible d_H(rho) values and equal-dimension
shape comparisons remain unfinished. No unique relationship to any LQG
coupling is inferred. [Detailed diagnosis](../docs/GENERAL_COVARIANCE_DIAGNOSTICS.md).

![Covariance sampler diagnostics](figures/covariance_pilot.summary.png)

![Independent grid and sweep sensitivity](figures/sampler_sensitivity.png)

## 6. Additional geometric observables

The main independent three-tree simple degree has mean about 7.571 at large n;
the two-tree control has mean about 5.849. The multigraph has exactly
`n+d*(n-3)` edge occurrences, giving mean degree approaching 8 in d=3 and
6 in d=2. The difference is due to coincident diagonals. Degree histograms
are saved graph by graph. Hamiltonian cycles rule out cut vertices; more
delicate two-vertex separators/bottleneck statistics were not estimated here.

Lazy random walks with holding probability 1/2 were measured on eight
independent biased I3 graphs at each of n=8,192 and 32,768, using 12,000
uniform-root walks per graph to 256 steps. Both simple unit conductances and
diagonal multiplicities were tested. Window slopes corresponding to spectral
dimensions were around **2.2–2.3** over 8–64 or 16–128 steps. These are only
finite-time diagnostics. Return counts become small at later times, adjacent
times share paths, and the reliable-count filter can select noisy high returns.
No stable long-time/asymptotic spectral dimension has been established.
The uniform-root finite-volume plateau is 1/n; stationary-root conventions
would have a different averaged plateau.

For eight n=128 graphs and 32 uniform vertex pairs each, mean effective
resistance was 0.4716 for unit simple conductances and 0.4502 with
multiplicities. The reduction is consistent with Rayleigh monotonicity; it is
not a resistance scaling estimate. Dense pseudoinverses are restricted to
small graphs.

![Small examples with tree provenance](additional/small_graphs.png)

![Return probability diagnostics](additional/walk_returns.png)

All [additional numerical records](additional/observables.json) include
seeds, walk counts, conditional errors and conductance conventions. Drawings
use Hamiltonian circular layouts and imply no Euclidean embedding or topology.

## 7. Numerical reliability and reproducibility

Python 3.14.6, NumPy 2.5.1, SciPy 1.18.1, Matplotlib 3.11.1 and pytest 9.2.0
were used on macOS arm64. Direct versions are pinned in
`requirements-lock.txt`; this is not a transitive cross-platform lock.
NumPy Generator/PCG64 and explicit per-graph/per-metric seeds are saved.
Main run seeds are 2026111001 (biased I3), 2026111002 (biased I2),
2026111003 (unbiased I3), 2026111004 (independent high-size biased I3),
and 2026111030 for the coupled Weyl study. Other seeds and commands are in
script defaults and metadata. Separate studies may use the same scalar base
seed but different seed derivations; they are not treated as independent
replicates of an identical construction when overlapping.

Each JSONL realization retains per-root histograms, pooled distances, ball
profiles, degrees, diameter bounds, generator metadata and timings. Selected
full graphs are saved as NPZ; all can be regenerated from their seeds.
Software versions, Git base and source hashes were captured for each run;
the working tree was uncommitted. The combined metadata embeds its two
original run metadata files and hashes instead of pretending all code was
unchanged throughout development. Later fixes to non-dyadic grid membership
do not affect these dyadic independent runs. Original research documents and
existing source artifacts were preserved.

Bootstrap units are whole graph realizations, resampled within size. Root
averages and targets inside a graph are not independent graph samples.
Per-root variance and estimated root-averaging noise are also saved;
subtracting that noise gives a noisy finite-sample estimate of between-graph
variance and does not establish exact separation. Most mean/window fits use
500 bootstrap replicates; correlated controls use 300; nonlinear histogram
collapse uncertainty uses 100. Bootstrap intervals exclude model error,
unknown asymptotic corrections and, for the correlated case, residual minima
discretization. The special-cone q=32/128 comparisons use coupled paths and
identical root seeds; they are paired systematic diagnostics, not additional
independent graph samples.

Remaining material limitations are: low counts at the largest two sizes;
nonlinear correction-model dependence; no identified full equation (45) fit;
nonconverged local edges in the polygonal correlated control; and unresolved
general-cone MCMC/grid errors. No reported matching exponent proves a common
metric law, convergence, manifold topology, or 3D LQG identification.

Reproduction instructions are in the [README](../README.md).

## 8. Research interpretation

**Established implementation facts:** adjacency agrees with the paper's
definition and an independent oracle; the independent law/minima construction
is exact by the stated conditional-bridge derivation; arbitrary admissible
covariances are supported by a separately labeled approximate generator.

**New numerical evidence:** sparse graphs at the independent point show
growing intrinsic distances across a factor 256 in vertex count. Shift-aware
fits are compatible with the published dimension but have substantial
uncertainty. Ball profiles and naive distance slopes display large finite-size
corrections. A correlated cone provides a second broadly compatible control.
The general MCMC pilot exposes a concrete near-degenerate sampling obstacle.

**Unresolved conjectures:** existence and uniqueness of a limiting random
metric-measure space, its Hausdorff and spectral dimensions, covariance
dependence, and its topology. Even a convincing power law for uniform-pair
distances would not prove tightness of the whole metric space. Three encoded
trees are not evidence of a topological three-manifold.

## 9. Open questions

1. Can exact independent cell minima and the paper's walk-on-balls method be
   compared directly, including the full corrected histogram fit, at higher
   independent sample counts?
2. Which correction model and size window become stable when graph counts at
   2^18 and 2^19 increase? Does the additive offset stabilize across statistics?
3. Can a stable determinant/heat-kernel method sample joint Weyl gap minima,
   removing the correlated control's polygonal error?
4. Can the validated FEM angular mode be integrated into a properly controlled
   Doob/walk-on-balls sampler for general cones? Mesh, spatial step, hitting
   time and angular sampling errors all need separate checks.
5. Are near-degenerate covariance transients avoidable without changing the
   target law? Longer chains alone require convincing convergence diagnostics.
6. Do return probabilities, resistance growth and separator statistics show
   stable transport behavior beyond the current short windows?

## 10. Recommendations and phase handoff

First strengthen the independent and A2 special-cone graph baselines: increase
independent large-size samples, compare histogram/offset corrections, retain
graph-cluster uncertainty, and resolve correlated minima. Next implement and
cross-validate the paper's general biased sampler against both exact special
cases. Initial general-correlation testing should stay in a moderate condition
number region; rho≈−.5 needs its own sampling study before exponent work.
Keep arbitrary unequal-correlation matrices in the tests so the isosceles slice
does not become an implicit restriction.

For future candidate-field comparisons, preserve normalized pair-distance
histograms, uniform-root ball-mass distributions, degree/multiplicity data,
walk returns, resistance and separator diagnostics as separate observables.
These offer stronger tests than one fitted dimension. A later comparison
should begin with reliable moderate-correlation graph samples, not a parameter
map selected from the current unvalidated sweep. The present work does not
choose such a map or initiate that next stage.

The phase handoff is therefore: validated exact baseline and special-cone
controls, reproducible preliminary metric results, and a documented stop at
the general-covariance sampler gate. Review is required before starting the
field/GMC/LFPP stage, as requested by the user.
