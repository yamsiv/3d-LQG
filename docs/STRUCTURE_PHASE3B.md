# Covariance structure and independent statistical review — Phase 3B

Script: [phase3b_structure.py](../experiments/phase3b_structure.py).
Artifacts: [results/phase3b/structure](../results/phase3b/structure/).
The analysis reads saved graphs and scores only. It launches no simulations.
The initial28-configuration analysis took6.86seconds, including24 whole-path
feature bootstrap draws per covariance and64 paired two-size growth draws.
The completed adaptive augmentation adds three coarse configurations, preserving
the original28-point outputs separately. Total successful covariance count is
**31 of32 attempted**; C022 failed the positive-polynomial certificate and has
no invented/imputed graph records.

## Data and units

The initial screen has six independent physical paths for each covariance,
paired graph sizes512/2048 and16 iid counting-measure sampled points per graph.
Its47 features use a deterministic **ensemble** pair median. Intermediate
records have eight new physical paths for each of nine covariances and paired
sizes2048/8192, also16 points. Provisional paths use finer angular/path settings.
Exact I3 and matrix-grid A2/A3 references retain their separate reliability labels.

Three distinct score units must not be mixed:

1. The main adaptive score and robust rankings use declared **natural feature
   scales**, as in the saved search files, with weighted family RMS differences.
2. The structural far-covariance scores use one **fixed noise-scale vector**
   from `feature_bootstrap.npz`, based on conditional whole-path bootstrap
   variances of the six-path I3/A2/A3 screen controls. The feature noise vector
   estimates six-versus-six differences and has a .05 natural-scale floor.
3. Phase3A's cached noise audit uses different reference ensembles/sample
   counts. Its scores and noise-floor numbers are not numerically comparable
   to either main adaptive scores or the six-path structure scores.

Within item2, far-covariance scores and the structural self-comparison scores
use the **identical** saved noise vector. The control conditional self-comparison
medians are .750(I3), .517(A2), .500(A3). They are bootstrap descriptions of
finite records, not formal null distributions, acceptance thresholds or
independent new experiments.

Physical identity across field xi/gamma choices was checked using the source
NPZ SHA256: all12 initial screen field IDs have exactly one source hash across
every xi and uncertainty probe. Cached field headers have `field_seed=None`,
so source hash, rather than that null field, supplies the identity evidence.
See `source_identity_check.json`.

## Finite feature variation

PCA divides features by their declared natural scales, then centers across
covariance configurations. Initial PC1/2/3 explain62.4%,25.9%,4.11% of finite
between-configuration variance. Projecting the whole-path bootstrap draws on
that fixed basis gives observed-to-conditional-MC variance ratios2.05,3.05,1.86;
PC4 is about1.01. The corresponding31-configuration explained fractions are
63.5%,25.1%,3.85%. The PCA includes correlated features; ball and finite-radius
features dominate the leading loadings. It is a descriptive projection under
explicit feature weights, not an intrinsic manifold-dimension estimator.

Several global PCs **do not refute a one-dimensional nonlinear geometry
family**: curvature of a one-dimensional curve can require multiple linear
principal components. Conversely a dominant PC is not evidence that all
covariances collapse onto scalar level sets. The feature bootstrap does not
include covariance-dependent Doob error, continuous minima, graph quantization
or field discretization. No continuum rank is identified.

For orientation, PC2's Spearman correlation with mean Brownian correlation is
-.749; PC1's is-.343. Leading PCs also weakly correlate with the monitored
mean-value diagnostics; this neither removes nor bounds the sampler bias.
An uncalibrated global linear covariance-response fit has multiple nonzero
singular values, but it does not establish local rank or a scalar parameter map.

Figures: `feature_pca.{png,pdf}`, `covariance_map.{png,pdf}` and the separate
31-point `adaptive31/covariance_map.{png,pdf}`. Red outlined squares on covariance
maps denote I3/A2/A3 references; zero mean-value defects for those reference
records mean the provisional-kernel diagnostic is inapplicable.

## Candidate degeneracies and higher-point distinctions

Among the378 initial covariance pairs, C001=(-.15,-.15,-.15) and
C028=(-.53549,-.21041,.53225) are far apart in canonical covariance space
(Euclidean separation .786), yet have fixed-noise rich-feature discrepancy
.356. I3 C002 and C028 have separation .784 and discrepancy .481. These are
comparable with the **same-unit** conditional control medians .50–.75 above.
This identifies useful candidate near-degeneracies for later investigation,
not equal metric-measure laws or demonstrated continuum level sets. Six paths,
coarse quantization, selection among378 pairs and C028's provisional sampler
leave substantial ambiguity.

Finite mean-distance growth does not determine even the observed matrix shape.
For each physical path the512/2048 graphs remain paired when bootstrapping

```
beta_finite = log(mean_distance_2048/mean_distance_512) / log(4).
```

This is a **two-size finite growth proxy**, never a Hausdorff-dimension estimate.
For example A3 C012 and generic C026 have growth difference .0114 with paired
MC95[-.0056,.0298], while quartet-delta means differ by .02024 with independent
physical-space SE .00244. C012 versus C018 has growth difference-.00232
MC95[-.01799,.01258], yet quartet-delta difference .02374(SE .00306).
These are selected descriptive examples, not multiplicity-adjusted hypothesis
tests. The generic samplers can cause correlated path/minimum artifacts;
the discrepancies cannot be promoted to different continuum geometries.
All pair comparisons, rather than only striking examples, are saved in
`pair_structure.json`.

## Gauges, weights and refinement

Across21 declared weight scenarios the initial **global robust winner** is A3
C012 in both separately specified gauges. Grounded xi choices are .225 in18
scenarios/.2 in3; unit-mass xi choices are .225 in14/.25 in6/.2 in1. The robust
criterion first takes the maximum over gamma uncertainty probes at each xi,
then minimizes over eligible central xi. This stability concerns a finite,
prespecified score and tested grid; it is not evidence for a unique optimum or
an identified gamma. Covariance-specific xi choices and full ranking changes
are saved in `sensitivity.json` and `weight_gauge_sensitivity.{png,pdf}`.

The three adaptive coarse proposals select C029 xi=.25, C030 xi=.25 and
C031 xi=.225 in the original coarse field comparison; their natural robust
scores are .158,.153,.141. They were deliberately selected using earlier
scores, so these are exploratory observations, not fresh confirmation.

All nine intermediate ensembles were complete when analyzed. Their2048/8192
growth proxies range approximately .251–.262. The matched-n2048 initial
versus intermediate natural feature discrepancies range .084–.244 for the
six original configurations. New adaptive proposal comparisons are saved in
`adaptive_fidelity_addendum.json`: C029 .233, C030 .093, C031 .073.
**Numerical settings and physical seeds both changed**, so these shifts do not
isolate method bias. Exact reference shifts demonstrate the appreciable
finite-sample/quantization contribution.

At n2048 the estimated ensemble median is only seven or eight graph hops.
Moving it by one hop changes several normalized ball radii across integer
distance jumps. C029's coarse/refined medians, for example, are7 and8. Those
discontinuities amplify ball/growth score sensitivity. Inspect signed raw
features and graph CDF atoms; a separate ensemble-mean normalization sensitivity
can diagnose this without silently changing the frozen principal criterion.

## Independent code review and resolution

`independent_code_review.json` records the reviewed source versions and
statistical issues. The principal finding was a criterion mismatch: the saved
initial selection bootstrap minimized central-probe scores while adaptive
ranking/finalist selection used maximum-over-probe scores. The coordinating
agent corrected future search bootstraps to use the robust criterion, preserving
the original screen outputs as explicitly labeled **central-score sensitivity**.
Those old argmin frequencies are not confidence in the robust finalists.

Other reviewed limitations remain explicit: GP diagonal fixed-noise acquisition
does not model shared fields or sampler bias; a loader's minimum-four-record
guard is weaker than the actual stage completion requirements; eligibility's
MC subcritical fraction does not cover finite-window systematic error; and
maximizing a few gamma-probe scores is a sensitivity device rather than an
error enclosure. None invalidates a clearly labeled exploratory design, but
each matters when interpreting an apparently small discrepancy.

## Research interpretation

The saved screen provides both candidate near-degeneracies at well-separated
covariances and examples where similar finite growth hides distinct higher-point
shape. Neither proves nor disproves a distinguished matching curve or many-to-one
level sets. The most useful next tests retain several such configurations,
matching matrix point counts and gauge, then separate graph-size effects and
coupled numerical refinement from independent-realization uncertainty.
General numerical harmonics remain provisional. No continuum equivalence,
canonical Euclidean embedding or topological identification follows from these
finite summaries.

Reproduce/complete analyses without simulations or replacing finished artifacts:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_structure
.venv/bin/python -m experiments.phase3b_structure --intermediate-only
.venv/bin/python -m pytest tests/test_phase3b_structure.py tests/test_metric_features_3b.py tests/test_metric_measure_3a.py -q
```

## Held-out independent review

`heldout_independent_review.json` independently recomputes all frozen held-out
probe scores from raw records and reproduces them to1e-12 in both gauges.
All47 point features are valid. The frozen pairs exactly match stored field
parameters: C012 xi=.225/gamma=.85001258; C030,C029,C002
xi=.25/gamma=1.00770920. There are16 graphs per covariance and12 fields/probe,
64 unique held-out graph seeds with no overlap with intermediate graph seeds,
and one source-field SHA per physical field across all xi/gamma probes. The
final plan predates the first held-out graph checkpoint. No current calculation
bug was identified in `phase3b_results.py`.

The final bootstrap uses independent graph resamples and common field indices
across gamma probes and the flat control. It recomputes the Weyl gauge and
ensemble normalization in every draw. Half-split floors use8-versus8 graphs
and6-versus6 fields, so they are descriptive and not acceptance bands for the
full16-versus12 comparisons. The26-minus98 stencil controls have the correct
positive sign: about3.1–3.3% mean-distance shifts at the frozen positive xi,
and5.79% at xi0. Smooth controls share the actual cached low-mode randomness
and root uniforms; they test a finite matched-variance nonlogarithmic field,
not a second continuum GMC construction.

Grounded robust held-out scores for C012/C030 are .1355/.1381 with MC95
[.0829,.1779]/[.1135,.1698]; these do not identify a unique winner.
C030's quartet-delta mean agrees within MC error (.00101 signed graph-minus-field,
MC95[-.00435,.00507]), while quartet aspect differs .01824[.00251,.02744]
and smallest triangle ratio differs-.02898[-.04280,-.01190]. C012 likewise
retains signed quartet-delta/aspect and ball-dispersion residuals. These
intervals describe physical-realization MC error without multiplicity,
sampler-law, finite-size or field/path errors. Agreement of one higher-point
coordinate is not agreement of the entire sampled distance-matrix law.

The crossing-derived dimension proxies gamma/xi are3.778 and4.031 at the
frozen xi values. In particular the latter remains in tension with larger
exact-I3 correction-sensitive graph estimates around4.8–4.9. Low normalized
pair discrepancy cannot resolve that exponent tension. The separately declared
historical I3 xi=.3 diagnostic has a closer quartet family but different
global/mesoscopic residuals; the small total-score gap should not be promoted
to rejection of that parameter. Overall the independent review supports the
report's partial/ambiguous interpretation and a targeted next study, not a
continuum correspondence or unique covariance-to-gamma relationship.
