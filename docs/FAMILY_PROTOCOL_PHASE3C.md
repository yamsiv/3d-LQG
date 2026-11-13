# Phase 3C finite-family analysis: predeclared protocol

Written before loading/scoring new Phase3C comparisons. Source:
[`family_search_3c.py`](../src/family_search_3c.py). Calibration and analysis:
[`phase3c_family_analysis.py`](../experiments/phase3c_family_analysis.py).
The final raw calibration/checkpoint timestamps, source hashes and protocol
snapshot under `results/phase3c/families/statistics/` identify the frozen version.
Later exploratory changes must be separately labeled, never substituted for
this primary analysis.

## Scope and scientific status

Search for several empirically supported low-discrepancy regions across the
canonical covariance domain and the crossing-linked xi/gamma family. Do not
optimize only the global argmin or turn a surrogate prediction into an observed
connection. A finite connected set of low-J neighboring samples is evidence
about this sampling design and resolution, not a continuous covariance curve,
continuum common law, manifold topology or scalar covariance-to-gamma theorem.

The exact I3 and matrix-grid special cones remain references. Generic Doob
samples are explicitly provisional. The sampler workstream supplies numerical
budget/gate labels separately; a low J never overrides a failed or unknown
sampler gate. Monte Carlo tolerance does not cover joint-minimum, nonharmonic
transition, cutoff, clock or other systematic errors.

The source parameter conventions remain Budd–Castro2207.05355v1 §§3.3–3.4/4.1
for the biased excursion ensemble; Contreras Hip–Zhuang2504.09141v1(1.1)–(1.3),
Definition1.7/Proposition1.8 and2512.06292v1(1.5)/Corollary1.9 for
lambda_cross=1-xi Q, Q=3/gamma+gamma/2 and xi=gamma/d_gamma in the relevant
subcritical regime. Finite crossing fits are proxies. Invalid/unstable branches
retain their status and cannot become primary matches just because J is small.

## Normalization, gauges and features

**Primary:** grounded torus field, normalized GMC probability measure and
**ensemble mean** measure-pair distance normalization. This choice precedes
the new results and reduces propagation of integer-median hop changes into
every ball radius. Each graph retains counting probability measure and the
paper-biased finite-volume excursion law. Recompute the ensemble normalizer
inside every whole-parent resample.

**Fixed parallel:** unit-mass Weyl gauge, D -> M^(-xi/gamma)D, with the same
normalized probability measure and the same candidate set. This is not
conditioning M=1 and is not a gauge selected for best agreement. It is undefined
for gamma0,xi>0. The flat(0,0) control has no shift.

**Sensitivity:** ensemble median normalization, in both gauges. Neither
normalizer removes individual-realization global fluctuations. The47 Phase3B
features and their natural scales are retained, including centered root-ball
quantiles/dispersion, triangle/quartet shape coordinates, shared/disjoint-pair
dependence and finite-radius growth. Ball means are not counted independently
from pair CDFs. Missing values retain masks; they are not geometric zeros.
All principal matrices use16 points; roots/matrix subsets never add independent
sample counts.

For backward layout compatibility the legacy name `pair_mean_over_median`
stores mean/selected_scale. It equals1 under mean normalization and acts as a
normalization check, not additional shape information. The ball/shape summaries
are still scaled by the chosen common ensemble mean. No old source is modified.

Four weightings are fixed:

| Family | Primary | Equal | Higher-point | Without mesoscopic/atoms |
|---|---:|---:|---:|---:|
| Pair |1|1|.5|1|
| Ball fluctuations |1.5|1|1|1.5|
| Triangles |1|1|1.5|1|
| Quartets |1.5|1|2|1.5|
| Matrix dependence |1|1|1.5|1|
| Global scale |.5|1|.5|.5|
| Finite-radius growth |1|1|1|0|
| Atoms |.25|1|.25|0|

Each component is RMS standardized residual within its family; J is the
weighted family average. These are descriptive weights, not a likelihood or
independent-feature chi-square test. Keep signed47-dimensional residuals,
valid masks and per-family discrepancies alongside every aggregate J.

## Frozen epsilon from OLD data only

Use the following saved Phase3B cohorts, all with16-point matrices:

- I3 C002 held-out graphs:16 parents at n8192, split8 versus8.
- A3 C012 held-out graphs:16 parents at n8192, split8 versus8.
- Held-out N48/98-direction fields at central xi0,.225,.25:12 parents per
  parameter, each split6 versus6.
- Separately declared historical xi.3 linked central fields at N48/98
  directions:12 parents, split6 versus6.

These field cohorts share underlying physical fields. They are not six
independent ensembles, and their resplit indices remain shared across xi.
Within a partition each half is disjoint;32 balanced random partitions reuse
the finite dataset and are dependent. The seed is202611123101. For every
gauge g, normalization n and weighting w, fix

```
epsilon[g,n,w] = max over the six cohorts of
                quantile_.95( J_w(A_b,B_b), b=1,...,32 ).
```

No Phase3C J, covariance design optimum or desired curve enters this rule.
No unproved n^(-1/2) rescaling is applied. The reference has smaller parent
counts than the new target12 graphs/48 fields or later24-plus-parent stages;
the same epsilon remains a deliberately conservative finite-precision screen
reference. It is not a calibrated continuum-acceptance band or familywise test.
New MC intervals and numerical budgets remain separate instead of retuning
epsilon until a component appears.

Also retain the maximum-cohort95th-percentile reference for each family and
each absolute signed feature difference. Individual feature thresholds have a
fixed .05 natural-scale floor to avoid flags from accidentally zero resplit
noise. Per-family/feature flags are unadjusted diagnostics of tradeoffs within
a small total J; they are not simultaneous rejection decisions. A candidate
can be low aggregate J while visibly disagreeing in a particular component.

**Prespecified secondary subgraph:** keep the exact same epsilon and additionally
require every valid family RMS to fall below its already-frozen max-cohort95%
self-split component threshold, for all available gamma probes. This is the
existing `componentwise_low` criterion, approved before Phase3C scores; no
new smaller epsilon is introduced. Persistent same-sign residual flags are
adverse evidence, not a reward for aligned residuals. A component profile marks
persistence when the equal-covariance-weighted flag fraction is at least.75 and
positive-sign fraction is at least.75 or at most.25. These descriptive fractions
never treat repeated xi or shared fields as independent trials.

For robust calibration-probe analysis, take the maximum J across valid
central/low/high gamma probes at each xi, using the same frozen epsilon.
Report the central value and the probe envelope. A finite set of gamma probes
is not an enclosure of all asymptotic parameter uncertainty. Invalid probes
are recorded, not silently made valid.

## Observed adjacency and family support

The primary family graph uses only observed, sampler-allowed, calibration-
eligible nodes with J<=epsilon at the chosen fixed gauge/weight/normalization.
Define covariance neighborhoods on the **full tested covariance design**, not
only the low-J subset:

- Mutual4-nearest canonical covariance neighbors, Euclidean distance<=.25.
- Between different covariances, primary links require the **same measured xi**.
- At a fixed covariance, link adjacent tested xi only when their gap<=.0375.
  High-J/failed intermediate observations are retained while defining adjacency,
  then excluded, so they cannot be jumped over to manufacture a connection.

The primary product graph has no diagonal/interpolated links. Prespecified
sensitivities use covariance radii .18/.25/.32, and separately permit diagonal
neighbor links at xi gap<=.0375. If topology changes under these choices,
report that sensitivity. A gap is unresolved coverage, not proof that a true
family is disconnected. Conversely a graph connection does not prove all
unobserved parameters between its endpoints have low J.

Call a connected component **observed family support** only if it contains
at least4 distinct covariances,3 measured xi values and xi extent>=.05.
Retain every other component as isolated/local low-J support, rather than
discarding it or inventing a continuous curve. Report multiple components,
xi/gamma extents, covariance spans, blocked/poor-sampler nodes and component
flags. Do not reduce this output to a unique best covariance.

Direct graph-to-graph comparisons within candidate regions use the corresponding
graph-only old-cohort noise references and signed shape residuals. Similar J
against a field is not evidence that two graph laws are equal. Covariance
permutations are quotiented, and near-equal correlations are explicitly treated
as walls of that quotient when assessing local parameter directions.

## Local response, level sets and adaptive acquisition

Fit a weighted local linear feature response in a radius .25 around a tested
covariance, using at most12 neighbors and requiring at least5 observations and
full affine covariance-design rank. The actual design condition and common valid
features are reported. Compare radii .18/.25/.32 as sensitivity. Do not substitute
global PCA rank for local structure: even a curved one-dimensional family can
need several global PCs.

When available, bootstrap the **whole parents separately at each covariance**,
keeping the field's resample common across xi/probes. Refit the same local
regression in each draw. Report singular values and the conditional95th-percentile
operator norm of the bootstrap Jacobian error. Singular directions exceeding
that noise are only a conditional finite-data diagnostic, not a rank theorem.
Feature changes can arise from covariance-dependent numerical sampler bias.
Small-radius versus larger-radius instability can reflect curvature, sparse
design, discontinuous graph quantization or MC error; none is automatically an
additional continuum parameter.

At sorted-correlation gaps below .03, flag proximity to a coordinate-permutation
quotient wall. Symmetry can remove first-order parameter responses there;
an apparent rank deficiency does not identify a many-to-one level set.
If a finite field-curve tangent is supplied, project the local graph Jacobian
onto it and report transverse singular values. Interpreting a transverse response
requires it to survive MC, radius and independently quantified sampler errors.

Reserve approximately12 adaptive covariance points across observed branches,
different xi ranges and poorly resolved directions. Include uncertain/high-coverage
regions as well as low-J regions; a global-argmin-only batch cannot discover
multiple families. The coordinating design records its actual branch/coverage
allocations before those new scores are inspected. Surrogates can propose where
to sample, but only new observed records can join the family graph.

## Raw loader requirements and reproducibility

Graph records provide grid, root CDFs, root means,16x16 matrix, discrete=True,
and headers with covariance ID/triple, size, numerical settings, physical path
identity and seed. Shared sizes of one path are paired observations. Field
records provide the same metric arrays plus xi/gamma/log_total_mass and a
physical field identity/source hash. Common field resampling across xi requires
matching source identities/order, not merely reusing integer realization labels.

Every analysis stores source/input hashes, epsilon calibration, chosen gauge,
normalization, weights, adjacency, masks, cohort counts, raw residuals and MC
draws/indices. Source3B files and datasets remain immutable. A sampler/systematic
gate is mandatory; low J with absent gate is not a trusted candidate.

The coordinator's observed numerical policy excludes C019/C023 and the known
D4 positivity failure C022. Any retained path with an absolute signed leg
log-mean-value sum>.03 or accumulated absolute leg sum>.10 excludes that
configuration/settings from family claims. These diagnostics are not a total-law
bound; remaining generic settings are still provisional. Never remove only a
bad realization and average the convenient survivors.

## Executed old-data calibration

The frozen calibration completed in13.91seconds without reading Phase3C scores.
Primary grounded/mean epsilon is **.26041420**; unit-mass/mean epsilon is
**.20661248**. Median sensitivities are.25754259/.21535207. Grounded/mean
epsilon is driven by the historical xi.3 field6-versus6 split noise (.26041),
versus I3 .20661, A3 .10768 and xi.225/.25 fields .11258/.15589. Thus the
chosen rule is deliberately loose at new12-graph/48-field precision. A connected
low-J cloud under this threshold cannot identify a correspondence by itself.
Componentwise residuals, independent MC/refinement and sampler status remain
essential; the threshold is preserved rather than narrowed after seeing results.

The six-cohort partitions correctly share the underlying field identities across
xi, gauges and normalizations. All4 weight thresholds and47 signed-feature
reference thresholds are saved in `tolerance_calibration.json`; the original
method snapshot and later pre-score secondary/gate addendum remain separate.

## Prespecified full16-point distance-matrix diagnostic

An optional secondary diagnostic, declared before family selection, compares
the full upper triangle of each16-point matrix rather than47 feature projections.
Each120-coordinate panel is **one** physical graph/field parent. Normalize all
distances with the common ensemble mean (or the fixed median sensitivity),
after the selected Weyl gauge. Let K(x,y)=||x-y||_2/sqrt(120). Report

```
E_U = 2*mean_{i,j} K(X_i,Y_j)
      - mean_{i!=j} K(X_i,X_j)
      - mean_{i!=j} K(Y_i,Y_j).
```

Retain negative finite estimates; do not clip or take a square root. Because
the ensemble scale is estimated from these same parents, this is a descriptive
signed U-form, not a strictly unbiased fixed-population kernel estimator.
Bootstrap whole parents and recompute scale/gauge in every draw. Use common
field/graph resampling indices for paired comparisons with flat/smooth controls.
This diagnostic does **not** change J, its weights or epsilon.

Iid sampled point labels make the population matrix law exchangeable. A common
point permutation across all panels is exactly a coordinate permutation and
leaves the estimator unchanged; independently relabeling individual panels can
change the finite estimate. Repeated sampled locations are retained. Euclidean
distance here is only a kernel on intrinsic distance-matrix coordinates, not
an assumed Euclidean embedding of the graph. Quantization, finite GMC atoms,
domain geometry and sampler errors still affect the panel law.

Positive percentile endpoints are not a null-calibrated energy test, especially
near its degenerate equality null. Use this as a family/direct-levelset and
control diagnostic with signed MC uncertainty, never as proof of equality or
an alternative hidden optimization target.
