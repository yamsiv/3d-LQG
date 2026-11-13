# Intrinsic metric-measure statistics and independent audit — Phase 3A

Implementation: `src/metric_measure_3a.py`; immutable ensemble checkpoints:
`experiments/phase3a_ensembles.py`; selection and confirmation:
`experiments/phase3a_compare.py`. Mathematical conventions and the distinction
between an exploratory parameter grid and theoretically matched gamma/xi
are in [COMPARISON_PROTOCOL_PHASE3A.md](COMPARISON_PROTOCOL_PHASE3A.md).

## Statistical unit and actual design

One physical field or graph is a statistical realization. Its roots,
all-target distances, 56 triangles from eight sampled points, parameter
choices, stencils and coupled resolutions are dependent observations.
They never multiply the independent realization count.

The saved graph metadata specify 48 realizations per model per size,
three models and n=2,048/8,192: 288 checkpoints. I3 checkpoints regenerate
archived exact graphs to obtain new distance matrices, checking the old
root histograms; these are not additional independent I3 simulations.
A2/A3 are newly sampled biased matrix-bridge graphs with q=32 polygonal
minima, with four coupled q=64 checks per model and size. The split is
realizations 0–23 for discovery and 24–47 for confirmation.

Discovery uses 24 independent field seeds, coupled N=16/24/32, axis and
26-direction metrics, and all 30 pairs from six xi values and five gamma
values. Selection uses only N=32 and the 26-direction stencil; three graph
targets give 90 graph–parameter comparisons. The additional sizes and
axis metrics are discretization sensitivities, not independent parameter
replicates. Confirmation is designed for 24 new field seeds at N=24/32/48,
26- and 98-direction metrics, the frozen selected pairs and the flat control.
Two graph models selected the same pair, so these give three distinct pairs.
The saved confirmation configuration additionally includes `(xi,gamma)`
`(.2,.5)` and `(.2,1)` as bracketing probes motivated by the separate coarse
crossing conversion. Thus it has five distinct pairs; those two additions
are diagnostics, not a re-selection of the locked graph fits.

The graph/discovery metadata and saved selection determine actual counts;
the original design file contains an obsolete I3 split description. Final
results must use observed checkpoint counts rather than that preliminary
count. A covariance family, grid size and sampler law remain part of the
ensemble label; pooling graph sizes would define a different ensemble.

## Pair laws and deterministic ensemble normalization

Let j index realizations and let `mu_j` be its probability measure: normalized
vertex count for graphs, normalized GMC for a coupled field experiment.
Eight points are sampled independently with replacement from `mu_j`.
For each root, the second point is integrated over all target vertices using
their exact finite-grid probability weights. Thus a stored root curve
estimates `mu_j{y:d_j(root,y)<=r}`; it is not an unweighted target histogram
when GMC is nonuniform.

For each finite ensemble use

```
F(r) = mean_j mean_roots F_(j,root)(r),
s = inf{r:F(r)>=1/2},
D_normalized = D/s.
```

Every realization has equal weight, regardless of its total unnormalized
GMC mass. This is a deterministic ensemble normalization estimated from
the available independent realizations. No graph-by-graph or field-by-field
scale removal is used in the principal result. Per-realization normalization
is only a separate sensitivity and removes genuine random scale variation
as well as finite-root noise.

Normalizing the measure by total GMC mass M does not also fix the metric's
volume gauge. For gamma>0, the field shift `h -> h-log(M)/gamma` produces
the same probability measure and, by the finite LFPP Weyl identity,
multiplies distances by `M^(-xi/gamma)`. This is another possible physically
motivated sensitivity, distinct from both per-realization median scaling
and conditioning the original field law on M=1. It is not the principal
Phase 3A convention and is not automatically justified for unmatched pairs.
Global-scale discrepancies should be interpreted with this distinction in
mind before postulating an additional geometric parameter.

Graph CDFs have exact integer-distance support and right continuity. LFPP
root CDFs are retained on 1,025 evenly spaced distance coordinates and
interpolated linearly for analysis. Their one-bin bound concerns displacement
in distance, not a universal bound on absolute CDF error. The zero-distance
atom is recorded exactly before float32 CDF storage. Record the maximum
bin width divided by s alongside every finite comparison.

The numerical W1 approximation integrates `|F_G-F_F|` over a grid covering
the recorded normalized supports, extending past four when needed. The
CDF plots keep a compact display window; `tail_mass_at4` exposes any mass
outside that window. Trapezoidal integration and the interpolation grid
remain approximation errors. The squared-CDF integral is a descriptive
Lebesgue integral, not a distribution-free Cramér–von Mises test statistic.

Finite graph distance atoms are especially important for KS. At a jump of
size J, any continuous comparison CDF must have sup error at least J/2.
For discovery n=8,192, all three graph medians are 11; the masses at that
median are approximately .12611 (I3), .12661 (A2) and .13208 (A3). The
associated lower bounds are about .063–.066 before any geometric mismatch.
Because the LFPP interpolated CDF equals .5 near its median while the
graph CDF jumps past .5, the measured median overshoot is about .113–.116.
A KS discrepancy of this size does not alone establish continuum mismatch.
W1 is less sensitive to jumps, but still has graph finite-size effects.

## Rootwise ball laws and fluctuations

For radii `.25,.5,.75,1,1.5,2` in ensemble-median units, save the entire
array `B_(j,k)(r)=mu_j(ball(root_(j,k),r*s))`. Its average over roots and
realizations equals the pair CDF at r, so the objective does not count the
ball mean as independent evidence. Rootwise quantiles and dispersion add
information beyond that mean.

The saved empirical variance decomposition is

```
pooled Var(B) = mean_j Var_roots(B|j) + Var_j(mean_roots B|j).
```

This is an exact empirical identity with population-divisor variances and
equal root counts. The second term still contains finite-root Monte Carlo
noise; it is not a deconvolved estimate of pure field/graph randomness.
Similarly the coefficient of variation of root-averaged mean distance
contains both realization scale variation and eight-root sampling error.
The whole-realization bootstrap retains both, rather than claiming to have
separated them perfectly.

## Finite sampled distance matrices and collisions

The eight-point matrix is formed from the same independently measure-sampled
points used as roots. It preserves diagonal zeros and off-diagonal zeros
from collisions. The same gamma-specific points are used across xi values
and stencils on the same field, providing useful coupling for sensitivity
comparisons. Different gamma values use their own deterministic sampling
streams; they share the field, not necessarily the same categorical uniforms.

For each triple of distinct sample indices, sort its distances as
`d1<=d2<=d3` and use `(d1/d3,d2/d3)` when d3>0. A valid triangle with
two coincident locations has ratios (0,1) and remains included. A triple
with all locations coincident has d3=0 and undefined ratios; report its
frequency separately.

The objective averages valid triangle ratios within each realization and
then gives every eligible realization equal weight. It records the eligible
realization count. The plotted raw triangle CDF pools all valid triangles,
so it is weighted by the number of valid triangles, conditional on d3>0.
These summaries coincide when all fields have the same number of valid
triangles, and differ when collision rates vary. The CDF is therefore a
descriptive shape plot, not a second independently calibrated test.

Normalized GMC on a finite grid is atomic, with collision probability
`sum_i p_i^2`. For gamma=2 it can be appreciable and cutoff dependent even
though the subcritical continuum GMC is nonatomic. Do not delete collisions
to improve agreement, compare gamma=2 to a diffuse continuum measure without
qualification, or hide a zero-median normalization failure.

## Objective, frozen selection and confirmation

The declared ranking score is

```
J = W1(pair laws)
    + mean_declared_radii |SD_graph(rootball)-SD_field(rootball)|
    + |mean_graph(d1/d3)-mean_field(d1/d3)|
    + |mean_graph(d2/d3)-mean_field(d2/d3)|.
```

All distances are in their respective ensemble-median units. The weights
are explicit descriptive choices, not a likelihood, unique optimum theorem,
calibrated p-value or inferred covariance-to-gamma map. Inspect every term
and the raw distributions; a low sum may hide disagreement in one observable.
The global-scale CV difference and collision atom are supplementary diagnostics,
not optimized terms.

The discovery I3 score illustrates why the components matter. The selected
score is .06993 versus .12573 for the flat torus. Of the .05579 improvement,
about .05196 comes from root-ball dispersion, .00334 from pair W1 and only
.00050 from triangle means. The flat pair and triangle laws were already
similarly close by these summaries. Thus the main improvement reproduces
random local mass heterogeneity; it is not yet a distinctive identification
of a log-correlated metric-measure structure. Confirmation must inspect the
same decomposition rather than treating every term as a separate success.

A focused smooth-field negative control is analyzed separately: retain only
`0<|k|<=2` from eight saved confirmation h32 fields and rescale its theoretical
point variance to the original V32, then couple its metric and measure at
the locked I3 pair. This is a finite smooth nonlogarithmic control, not a
new independent field ensemble or an alternative parameter search. Comparing
it with the same eight log fields tests whether generic heterogeneity alone
can explain the descriptive improvement. It does not establish a continuum
limit for either model; conclusions require the saved control results.

The frozen discovery choices are I3 `(xi,gamma)=(.3,1.5)` and A2/A3
`(.2,0)`. The latter is LFPP with ordinary volume and is not a theoretically
matched nonzero-gamma metric-measure pair. The I3 choice also remains an
unmatched exploratory pair until crossing-exponent evidence supports the
implicit parameter relation. No uniqueness claim follows from selection on
this coarse 30-pair grid.

Selection is followed by independent field and graph confirmation, without
reselecting parameters. N=24/32/48 and two rich stencils test refinement
sensitivity. The final comparison code bootstraps the principal N=48,
98-direction comparisons; other refinement rows retain physical seed
coupling and report shifts. Those shifts are systematic diagnostics rather
than new independent sample counts.

## Bootstrap, noise floors and realistic power

`bootstrap_comparison` independently resamples entire graph and field
realizations with replacement, keeps every root and sampled matrix, and
recomputes ensemble medians inside each draw. Its percentile MC95 intervals
address finite-realization uncertainty at the fixed grid, topology, sampler
and parameter choice. They do not cover finite-size bias, path/minimum
discretization, stencil anisotropy, parameter selection or the existence of
a continuum limit.

W1 and the joint score are nonnegative empirical distances. A positive
percentile lower endpoint is not a valid rejection of equality of continuum
laws; these distance functionals are nonsmooth at their null. A formal
two-sample test would need a suitable null/permutation calibration, including
the estimated normalizers and clustering, and would still test only the
specified finite ensembles.

The script reports graph discovery-versus-confirmation discrepancies and
field discovery-versus-confirmation discrepancies at N=24/32 as split-sample
noise floors. These are individual independent-split diagnostics, not a
fully estimated distribution of power. Compare selected-versus-control
improvements against those floors, graph n=2,048→8,192 changes, field
resolution/stencil changes, and paired q=32→64 graph changes. Effects of the
same size as these shifts are not identifiable with 24 independent fields
and eight roots. Additional roots primarily reduce conditional Monte Carlo
noise; more independent fields are needed to characterize genuine field
fluctuations. High-gamma mass tails require separate treatment because some
continuum mass moments do not exist.

An additional scalar power calculation, saved in
`results/phase3a/theory/statistics_audit.json`, uses realization-level mean
triangle ratios in the discovery sample. For a difference of two such means,
the normal approximation to a two-sided 5% test with 80% power has detectable
effect approximately `2.80*sqrt(s_graph^2/n_graph+s_field^2/n_field)`.
Across the three graph models, selected/flat comparisons and the two ratios,
these finite-ensemble detectable differences range from about .015 to .046.
This calculation has no multiplicity or finite-size allowance and is not
power for W1 or for a continuum identification. It shows why ratio differences
of one or two hundredths can remain inconclusive with 24 realizations.

## Crossing proxies, topology and independent checks

### Crossing-exponent proxy analysis

`experiments/phase3a_crossing_analysis.py` reuses crossing costs in saved
checkpoint headers and generates no fields. It requires 24 physical seeds
at every requested resolution before analysis, ignores unfinished temporary
checkpoints, and pairs bootstrap resampling across N. Five hundred draws
are saved with the raw crossing arrays and source hashes under
`results/phase3a/crossing_analysis/`. A snapshot of the executed discovery
script preserves its historical source hash; the confirmation version adds
temporary-file handling and a neutral plot caption without changing the fit.

The estimator is minus the slope of log ensemble median physical crossing
versus log N. It saves the full three-size fit, adjacent two-size windows,
exact site-sum, gap-corrected edge and raw-edge conventions. xi=0 produces
the expected zero site exponent to floating-point roundoff. Conversions use
`Q=(1-lambda)/xi` and the subcritical quadratic solution only when Q>sqrt(6).
The reported gamma intervals condition on bootstrap draws satisfying this
inequality and explicitly record the valid fraction.

Discovery full-window site fits at N=16/24/32 give lambda approximately
.03794, .09873, .14350, .23597 and .40229 at xi=.1/.2/.3/.4/.6. The
corresponding valid gamma proxies for the first three are .3171, .7239 and
1.3884; xi=.4/.6 have no valid subcritical conversion in these draws.

Independent confirmation at N=24/32/48 gives:

| xi | Finite site lambda, MC95 | Implicit gamma proxy, conditional MC95 | Dimension proxy gamma/xi, conditional MC95 |
| --- | --- | --- | --- |
| .2 | .06728 [.04617,.12659] | .69508 [.67711,.75164] | 3.4754 [3.3855,3.7582] |
| .3 | .17855 [.12776,.21230] | 1.51440 [1.34114,1.68011] | 5.0480 [4.4705,5.6004] |

Both full-window valid fractions are one. This apparent xi=.3/gamma≈1.5
compatibility with the selected I3 pair is interesting, but the adjacent
confirmation windows give gamma proxies 1.2904 (24→32) and 1.7459 (32→48),
with dimension proxies 4.301 and 5.820. The first window's valid fraction is
.988. Finite-window and infrared/discretization effects are excluded from
the MC intervals; there is no stable asymptotic parameter calibration yet.
Independent exact-power synthetic checks verify the regression sign,
paired-resampling invariance and algebraic gamma inversion.

### Topology and implementation checks

An abstract metric-measure comparison requires no graph embedding. Torus
LFPP has fixed topology; finite three-tree graphs have no established
three-manifold scaling topology. An intrinsic distribution match cannot
resolve that distinction. The open-cube crossing diagnostic and periodic
pair law also have different boundary conditions and should not be pooled.

An independent audit ran eight statistics tests, including analytic equal-field
weighting despite unequal root counts, exact graph CDF right continuity,
distance-bin bracketing of weighted LFPP CDFs, normalization scale invariance,
whole-realization bootstrap, coincident-location triangles, synthetic crossing
power laws and the valid/invalid implicit-conversion branches. A saved N=32
discovery checkpoint was independently inspected: CDF shape `(30,8,1025)`,
matrix shape `(30,8,8)`, shared gamma-specific points across xi, monotone CDFs
ending at one, and matrix asymmetry below `7e-16`. Its field checkpoint holds
the shared-seed h16/h24/h32 arrays. These are implementation checks, not a
certificate of asymptotic statistical power.
