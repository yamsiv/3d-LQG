# Adaptive intrinsic comparison protocol — Phase 3B

Implementation: [metric_features_3b.py](../src/metric_features_3b.py).
Cached independent audit: [phase3b_statistics_audit.py](../experiments/phase3b_statistics_audit.py)
and [results/phase3b/statistics](../results/phase3b/statistics/).
The Phase 3A datasets remain unchanged. This document specifies statistical
interpretation; actual simulation counts and sampler settings come from each
Phase 3B checkpoint and the main report.

## 1. Independent parameter-convention review

The local primary PDFs were checked directly through their extracted texts:

- Contreras Hip–Zhuang, *Bounds on the distance exponent for higher-dimensional
  LFPP*, local arXiv:2504.09141v1, equations (1.1)–(1.3), Definition 1.7,
  Proposition 1.8 and Section 4. Their lattice path cost includes physical
  spacing epsilon at **every vertex**, including endpoints. Their crossing
  exponent is defined by a high-probability small-distance threshold, not by
  definition a finite-resolution mean or median regression. Median regressions
  in this project are finite-resolution proxies motivated by the scaling theory.
- Contreras Hip–Zhuang, *Weak exponential metrics*, local arXiv:2512.06292v1,
  equation (1.5), Weyl-scaling axiom (1.6), Theorem 1.3 and Corollary 1.9.
  The gamma–xi relation is implicit and one-to-one in the subcritical regime;
  the latter result identifies d_gamma as metric Hausdorff dimension. It does
  not identify a Brownian covariance or an arbitrary finite torus ensemble.
- Ding–Gwynne–Zhuang, *Tightness of exponential metrics*, local
  arXiv:2310.03996v5, Proposition 1.1 and equations (1.7)–(1.8). Their
  lambda_n is a **sequence of median normalizers**, not the scalar CHZ crossing
  exponent. It scales as 2^[-(1-xi*Q)n+o(n)]. Uniqueness of the general
  higher-dimensional limiting metric is not supplied by tightness alone.

In unit-log dimension three, the conversion to use is

```
lambda_cross = 1 - xi*Q,
Q = (1-lambda_cross)/xi,
gamma = 6 / (Q + sqrt(Q^2-6)),
d_gamma = gamma/xi,
```

for xi>0 and Q>sqrt(6). This is the subcritical root gamma<sqrt(6).
The xi=gamma=0 flat reference is handled separately. A regression of physical
crossing lengths against log N has slope **minus lambda_cross**. A slope of
the unscaled vertex sum has a different sign/offset conversion. The existing
axis implementation with half-endpoint corrections reproduces the vertex-sum
normalization; rich stencils are used separately for intrinsic metric shape.

The theoretical whole-space field is grounded by a spherical average, whereas
the project's compact torus has its zero Fourier mode removed. The torus is
a useful finite ensemble, not literally the source's whole-space finite-volume
law. Sharp Fourier cutoffs are not the source's box-average mollification.
Universality of an asymptotic local exponent does not eliminate finite-volume
or finite-cutoff differences. Report regularization/window changes separately
from whole-field Monte Carlo uncertainty.

If an MC interval or correction window crosses Q=sqrt(6), retain its invalid
portion as an explicit failure/partial conversion. Do not clip Q to the
subcritical boundary, silently discard invalid bootstrap draws, or interpolate
an apparently precise gamma curve through an unstable region. The full family
of fitted windows/correction models is a systematic envelope, not additional
independent data. Gamma=0 with xi>0 remains a diagnostic metric plus ordinary
volume, outside the matched nonzero-gamma metric-measure family.

Brownian parameters in source code are `(a,b,c)=(C12,C13,C23)`. The sorted
triple is a valid canonical representative under coordinate permutations:
permuting the three Brownian coordinates permutes the three off-diagonal
entries and leaves the unlabeled graph metric-measure law unchanged. All
matrices must still pass positive-definiteness/conditioning checks. The cone
index alpha1 and susceptibility 1-alpha1 are not LFPP/GMC parameters.

## 2. Statistical unit, scales, gauges and comparability

One physical graph or Gaussian field is an independent realization. Its roots,
pair distances, triangles, quartets, coupled resolutions, path stencils,
gamma choices and xi choices are dependent observations. Preserve all those
within-cluster dependencies when resampling. Reused historical seeds supply
new observables of old graphs, not new independent graph samples.

The principal metric normalization is the median of the equal-realization
mixture of root-to-measure distance distributions. The probability measure
is vertex count/n for graphs and GMC/total GMC mass for fields. Recompute the
ensemble median in each bootstrap or independent split. Per-realization median
normalization is only a sensitivity: it removes global randomness and finite
root noise together.

Prespecify two field ensembles and report both:

1. **Grounded:** retain the zero-mode field normalization used by the generator.
2. **Unit-mass Weyl gauge, gamma>0:** shift h by -log(M)/gamma and multiply all
   distances by M^(-xi/gamma), keeping the normalized GMC probability measure.
   This changes the ensemble, not its normalized measure. It is not conditioning
   the original law on M=1. It is undefined for gamma=0,xi>0; the flat (0,0)
   reference has no shift.

Use the same gauge for every selection candidate within each reported analysis.
Never choose the gauge after inspecting the best fit. Gauge changes affect
ensemble-scale fluctuations but leave each individual triangle/quartet ratio
unchanged. Independently selected graph/field domain topologies are unknown
versus torus; a large-scale shape discrepancy need not be a local universality
obstruction. Identify mesoscopic and domain-scale observables explicitly.

Eight-point versus sixteen-point matrices need care. Means of triangle/quartet
functions and exchangeable product U-statistics have the same expectation at
either point count. Empirical per-realization quantiles have finite-point bias.
Use matched point counts for principal comparisons, or take independent
eight-index subsets from sixteen-point matrices as a sensitivity. More subsets
of one matrix never provide more independent realizations.

## 3. Rich feature layout and its limitations

`feature_vector(records, gauge='grounded')` returns:

```
vector: 47 finite values,
names/families: stable ordered feature labels,
scales: declared natural units for descriptive scoring,
valid: explicit mask for defined estimates,
diagnostics: normalization, sample counts, atoms, bin bounds, noise terms.
```

Field records need xi/gamma/log_total_mass for the unit-mass gauge; alternatively
provide already transformed records and `already_transformed=True`. Graphs
already have unit counting mass. Invalid shape or slope estimates are stored
as value zero with `valid=False`; zero is never evidence of an exact shape.

The eight feature families are:

| Family | Features | Scientific content / limitation |
|---|---|---|
| Pair | Quantiles .1,.25,.75,.9,.95 divided by pair median; mean/median | Global intrinsic pair shape; integer graph quantization persists |
| Balls | SD and centered .1/.9 quantiles at median radii .35,.5,.75,1,1.5 | Root heterogeneity beyond the pair law; dependent radii |
| Triangles | Per-realization means and .1/.5/.9 empirical quantiles of two sorted-side ratios | Three-point shape, preserving repeated sampled locations |
| Quartets | Means and .1/.5/.9 quantiles of two sorted opposite-pair-sum ratios | Four-point shape, including a hyperbolicity-like coordinate |
| Matrix dependence | Shared-edge and disjoint-edge standardized product covariance | Dependence of distances sharing a point / only sharing the random space |
| Global scale | CV of root-averaged all-target mean distance | Contains finite-root noise; not pure global geometry |
| Mesoscopic | .1/.5/.9 finite-radius mass-growth slopes plus eligible fraction | Finite diagnostic only, never a Hausdorff-dimension estimator |
| Atoms | Pair zero atom, all-coincident triangle/quartet fractions | Finite grid/counting-measure diagnostics; may vanish at different rates |

For a triangle sort distances d1<=d2<=d3, retaining `(d1/d3,d2/d3)` when d3>0.
If exactly two sampled locations coincide the valid shape (0,1) remains.
All-coincident samples have undefined ratios and a separately reported mass.

For four sampled indices sort their three opposite-edge sums s1<=s2<=s3.
Use `((s3-s2)/s3,(s2-s1)/s3)` when s3>0. The first is twice the usual four-point
Gromov delta divided by s3; it vanishes for a tree metric. It is a finite sampled
shape statistic, not a global hyperbolicity certificate. Matrix shared/disjoint
products average all appropriate triples/quartets within each physical matrix.
Their covariance estimates subtract the square of the ensemble pair mean;
finite U-statistic estimates can have sampling fluctuations and are not a full
distance-matrix-law comparison.

Ball **means** are excluded as a separately weighted family because
`E_root mu(B(root,r))` is precisely the pair CDF. Centered root quantiles/SD
provide additional information, but are still correlated with pair features.
Quantiles of matrix shapes are computed within realization then averaged;
these differ from quantiles of a pooled triangle/quartet mixture. Equal physical
weights are retained even if collision counts or root counts differ.

Finite-radius slopes use
`log[B(.75s)/B(.35s)]/log(.75/.35)` for roots with positive small-ball mass,
large-ball mass below .8 and zero atom below .1 of the smaller mass. The
eligible fraction is retained; conditioning can itself change across models.
These radii may correspond to only a few graph hops, so integer floors and
finite-size shifts are essential. A disagreement is not an asymptotic dimension
discrepancy. No ad hoc independent correlated-coordinate minimum correction
is introduced by the statistics layer.

## 4. Quantization, interpolation and root-noise diagnostics

Graph root CDFs are right-continuous on exact graph-distance support. Field
CDF interpolation has a distance displacement bound equal to one saved bin,
not a universal vertical CDF error bound. At a graph CDF atom J any continuous
comparison CDF has KS error at least J/2. The module computes the **maximum
ensemble** atom on union support, rather than averaging individual maxima.
Report this floor and normalized distance-bin width alongside finite scores.

If m_j is the mean over K iid roots of their all-target mean distances, then

```
Var(m_j) = Var(E[m_j | space]) + E[Var_root(mean distance | space)/K].
```

The diagnostics report the observed between-realization variance, estimated
within-root sampling contribution and their raw difference. This subtraction
is unbiased in expectation under iid measure-sampled roots, but can be negative
and noisy at small sample counts. Keep its sign; clipped zero is not proof of
absent global fluctuations. The principal CV feature remains the directly
observed CV so that a noisy deconvolution is not optimized as if exact.

## 5. Objective, noise floors and sensitivity

`grouped_feature_distance` computes RMS standardized feature differences within
each family and takes their weighted mean. Natural scales are explicit; an
alternative uses externally calibrated fixed same-model split scales. Default
family weights are pair1, balls1.5, triangles1, quartets1.5, dependence1,
global scale.5, mesoscopic1, atoms.25. These descriptive weights give substantial
weight to higher-point/multiscale evidence. They are not a likelihood or a
chi-square statistic. Averaging within family prevents adding many correlated
features from silently increasing its total weight.

Same-model calibration repeatedly splits independent physical records into
disjoint groups. Featurewise RMS split differences estimate the conditional
finite-sample noise scale. The module floors them at .05 of their natural scale
to avoid huge weights from accidentally zero empirical noise. All parameter
candidates use the **same** reference scales; recalibrating each candidate to
its own variance can spuriously reward noisy models. For equal n graph/field
comparisons combine reference graph and field split scales as
`sqrt((scale_graph^2+scale_field^2)/2)`. Unequal counts require explicit scaling
or a fresh matched-count reference. The Gaussian `n^(-1/2)` scaling is only an
approximation for atom-sensitive quantiles and CVs.

Repeated re-splits reuse one dataset; their distribution is a descriptive
noise floor, not independent replicates or a formal null distribution.
A positive percentile interval for a nonnegative distance is not a valid
rejection of equal laws. Full covariance/Mahalanobis whitening is ill-conditioned
with 47 features and only 6–24 physical samples; use family aggregation or an
explicitly regularized, independently validated covariance estimate.

The cached audit reports 21 prespecified weight variants: default, equal family,
higher-point/multiscale emphasis, removal of global scale, removal of slopes/
atoms, and half/double each family. Report rank instability, not only the
best selected score. Also retain the signed feature residuals and whole-cluster
MC intervals, the full pair and ball laws, and the raw sampled matrices.

The field/graph bootstrap independently resamples whole physical realizations,
recomputing ensemble normalizations. Paired refinement/gauge/stencil analyses
require matching physical seeds and use common bootstrap indices. Parameters
sharing one field must remain coupled when comparing their objectives. MC
intervals do not include sampler law bias, path minima, spatial stencil,
cutoff, domain, graph-size effects or selection among many candidates.

An approximate scalar 80%-power minimum detectable effect is
`(1.96+.84)*SE = 2.80*SE`, using an independent-realization SE and an unadjusted
two-sided 5% test. This is only a rough planning calculation. Multiple feature/
parameter selection, non-Gaussian tails and systematic errors increase the
needed effect or sample size. More triangles from one graph cannot cure a
small number of independent graphs.

## 6. Adaptive design and stopping interpretation

The intended design is an initial moderate-SPD, permutation-deduplicated
space-filling covariance screen (target20–40, practical initial20–30), modest
coarse graphs (about6–10 realizations/configuration), approximately5–8 regions
with about12 independent refinements, then2–4 finalists with at least16 held-out
physical fields/graphs when measured runtime allows. These are budget/design
targets, **not claims that those ensembles have run**. Each stage must save its
actual candidate list and seed ledger before advancing. Root-level experiment
configuration/manifest determines the actual design, not this illustrative count.

Use a surrogate for feature vectors or family discrepancies, with variance
from physical-cluster uncertainty. Add new configurations where expected
agreement and uncertainty are promising; do not simply minimize one noisy point
estimate. Reserve final seeds and never refit covariance or gamma on them.
If held-out data motivate a new candidate, label that analysis exploratory
and preserve an unused confirmation set for the next phase.

The main field curve uses crossing-calibrated gamma(xi) with MC and window/
regularization envelopes. Independently varied gamma/xi pairs are diagnostic
controls only. Approximate generic graph samplers retain harmonic, path/minimum
resolution and analytic-control errors as explicit configuration metadata.
Large artifacts trigger a restricted-family interpretation, not a large sweep
through an invalid substitute. Strong candidate fits must survive at least
one graph-size and field/path refinement and flat/smooth negative controls.

## 7. Curves, level sets and identifiability

Let Phi(C) be the selected finite intrinsic feature vector and Psi(gamma) the
calibrated field curve, under a fixed gauge/domain/discretization. Matching
several features means only that Phi(C) lies near Psi(gamma) at that resolution.
It does not imply a common continuum law, manifold topology or embedding.

To distinguish the proposed alternatives, examine covariance-direction changes
after accounting for same-model noise and paired sampler/size shifts. A local
Jacobian of Phi in the three covariance directions can be examined using
regularized singular values in calibrated feature units. Singular directions
must exceed MC **and** systematic scales before being interpreted. A robust
rank above one would oppose a scalar gamma explanation of the entire tested
neighborhood, while still allowing a distinguished matching curve. An apparent
rank one is compatible with level sets but does not establish them. A restricted
one-dimensional analytic family alone cannot distinguish a special curve from
two-dimensional many-to-one level sets in the full covariance domain.

Likewise, optimizing gamma to match one dimension or one mean pair distance
and finding a higher-point residual motivates an additional geometric parameter
only if that residual survives size, discretization, gauge and topology checks.
Near-equal finite features cannot establish equality of continuum geometries.
The report should map **candidate regions plus reliability**, leaving unresolved
parameter directions and indistinguishable optima explicit.

## 8. Executed cached audit (11 November 2026)

The saved audit reused all Phase3A graph records, the24 N48/98-direction
confirmation fields and independent discovery/confirmation N32/26 fields.
It generated no new graph or field. Wall time15.81s for24 conditional split
draws,64 cluster-bootstrap draws per frozen graph comparison and27 diagnostic
graph/field/gauge comparisons. Input/source SHA256 hashes are saved.

The fixed noise-scaled score's conditional same-model24-versus24 split
medians are .626 (graph) and .763 (field), with10–90% ranges .483–.846 and
.624–.994. These reuse finite records and are not calibrated acceptance bands.

| Graph | Rich grounded best among five cached pairs | Score | Flat score |
|---|---|---:|---:|
| I3 | xi=.3,gamma=1.5 |1.739|3.956|
| A2 | xi=.3,gamma=1.5 |1.596|3.643|
| A3 | xi=.2,gamma=.5 |1.447|3.191|

The A2 ranking differs from Phase3A's simpler objective, illustrating feature
dependence; this is a post hoc diagnostic on previously used confirmation
data, not new independent selection evidence. Across21 weight scenarios the
grounded winners are I3(.3,1.5)19 times/(.2,0)2; A2(.3,1.5)20/(.2,0)1;
A3(.2,.5)21. The two gauges produce separately reported rankings.

The graph n2048→8192 shifts are1.70–1.98 noise units, substantially larger
than the coupled field N32/26→N48/98 shifts .52–.63. The graph shift is
especially large in finite-radius growth (5.45–5.89 family units), so these
small-graph mesoscopic discrepancies must not be interpreted asymptotically.
At the old frozen pairs, quartet-delta signed differences graph-minus-field
are I3 .0070 MC95[-.0008,.0149], A2 -.0263[-.0343,-.0185],
A3 -.0183[-.0247,-.0082]. These useful higher-point residuals require fresh,
matched-size, theoretically coupled and discretization-controlled investigation.

The saved independent-realization scalar planning calculation
`power_summary.{json,csv}` uses one triangle/quartet mean per physical space.
For the I3 frozen pair, unadjusted Gaussian80%-power MDEs are .0579 for the
smallest-side ratio, .0235 for the middle-side ratio, .0129 for quartet delta
and .0499 for quartet aspect. A2/A3 quartet-delta MDEs are .0131/.0117.
These calculations expose substantial feature-specific power differences;
they exclude multiplicity, optimization and systematic errors and are not
post hoc claims about the probability that a selected hypothesis is true.

Reproduce without replacing completed outputs:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_statistics_audit --bootstrap 64 --split-draws 24
.venv/bin/python -m pytest tests/test_metric_features_3b.py tests/test_metric_measure_3a.py -q
```
