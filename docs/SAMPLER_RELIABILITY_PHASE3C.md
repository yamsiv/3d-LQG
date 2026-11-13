# Phase 3C Stage I sampler reliability gate

Status: Stage I completed, 12 November 2026. The plan below was saved before
new computation; results follow it. This workstream
adds only `experiments/phase3c_reliability.py` and new artifacts under
`results/phase3c/calibration/sampler/`. All previous samplers/data stay intact.

The intended ensemble remains Budd–Castro's biased excursion, local primary
`papers/Budd_Castro.pdf` (arXiv:2207.05355v1), §§3.3–3.4, equations32–43 and
§4.1. The provisional positive-Ritz/sphere-quadrature method approximates this
ensemble; it does not supply a validated general-covariance Doob process.
The independent Bessel and special matrix-bridge references remain separate.

## Prespecified ten configurations

| ID | Correlations (a,b,c) | Role |
|---|---|---|
| C002 | (0,0,0) | Exact continuous-minimum I3 reference |
| C007 | (−.5,0,0) | A2×R+ matrix reference |
| C012 | (−.5,−.5,0) | A3 up to coordinate permutation |
| C006 | (−.7,0,0) | Negative analytic birectangular |
| C010 | (0,0,.5) | Positive analytic birectangular |
| C000 | (−.3,−.3,−.3) | Negative equicorrelated |
| C004 | (.3,.3,.3) | Positive equicorrelated |
| C016 | (−.2,.1,.35) | Previous asymmetric pilot control |
| C019 | (−.527752,.299752,.356044) | Known coarse harmonic-defect problem |
| C028 | (−.535493,−.210409,.532253) | Mixed-sign intermediate control |

All are strictly SPD. Configurations are read from the saved Phase3B design,
not reconstructed by rounding this display.

## Reuse and missing data

Reuse six coarse paths/configuration from the main Phase3B screen. For I3/A2/A3,
also reuse the six separately labeled D4 provisional anchor controls; the
main screen anchors themselves use the exact reference construction.

Reuse eight D6 finer paths for C000/C006/C028 at matched n2048, originally
sampled at n8192. Their smaller n has the same physical WOS path and therefore
does not create additional independent samples. Existing exact I3/A3
intermediate cohorts are references, not demonstrations of the finer general
sampler. Add only the missing D6 provisional groups for seven configurations:
C002/C007/C012/C004/C010/C016/C019, eight paths each (56 total). The planned
fine setting is degree6, base step .005 times sqrt(2.5/(alpha1−1)), relative
step .1, initial SIR1024, exit sphere order12 with order24 audit. Observe n512
and2048 on the same realization. Existing fine C000/C006/C028 already use
these settings. No precise dimensions are fitted.

The Phase3B n128/256 sampler pilot and conditional-bridge audit are reused as
additional small-scale evidence for I3/A2/A3/C016. They are not relabeled as
n2048 validation. The exact special references include known polygonal-gap
minimum limitations for A2/A3. Source hashes identify all reused checkpoints.

## Pilot and resource budget

Before the full missing-cohort run, a pilot capped at30 seconds will draw four
paths: C019 D6, C016 D6, I3 D6, and one difficult-case C019 D8/finer probe.
The first three checkpoints count toward their already planned eight-path
cohorts. The fourth tests whether an additional angular/path refinement is
computationally practical; one path cannot establish convergence.

Previous measured cost is approximately .7–2 seconds/path with 50–200k retained
stopping points, memory typically tens of MB and capped by the existing
two-million-step budget. The full new batch is provisionally budgeted at
2–4 minutes CPU, <250MB working memory and <250MB saved arrays. The coordinator
will review the measured pilot cost before authorizing the remaining batch.
No experiment is planned to exceed15 minutes. Failures are saved and never
silently retried with replacement seeds.

Seeds derive from base202611123001, configuration ID, refinement level and
realization index. Main-search and holdout seed families remain reserved for
the coordinator. Each completed physical path receives an immutable record
and array checkpoint, with raw minima, graph summaries and selected full
paths. Source/environment metadata accompany each stage.

## Observables and empirical error budget

The primary distance normalization is the deterministic **ensemble mean pair
distance**. Median normalization is a sensitivity. All root arrays and
16×16 iid measure-sampled matrices retain realization clustering. Assess:

- Mean distances, full normalized pair CDFs and quantiles.
- Rootwise normalized ball masses, including dispersion across roots/graphs.
- Triangle/quartet shape distributions and shared/disjoint distance dependence.
- Marginal/joint minima CDFs, degree laws, edge counts and tied-minimum rates.
- Initial SIR ESS, exit quadrature refinement, accumulated signed/absolute
  harmonic defects, retained clock interval/cell ratio and failure frequency.

Independent coarse/fine seed cohorts compare methods but are **not** coupled
Brownian refinements. Separately, the same retained polygonal path is observed
on nested q8/q32/q128 uniform-time grids, retaining its junction. Minima and
edges are compared to the full retained path, giving genuinely paired
observation-grid diagnostics. These do not bound the Brownian paths omitted
inside WOS sub-balls. No independent coordinate-minimum correction is used.

The error budget separates (1) between-realization uncertainty, (2) observed
coarse/fine shifts with their uncertainty, (3) same-path observation-grid
effects, and (4) unresolved true-path/harmonic error. An absolute shift plus
its MC uncertainty is an **empirical sensitivity envelope**, not a rigorous
upper bound or proof of uniform regional accuracy. Large observed defects
flag unreliable settings; small observed defects alone do not validate them.

Provisional targets to discuss with the coordinator are2% mean shifts, .02
normalized pair-W1, .05 joint-minimum CDF, .015 ball-SD changes, .01 triangle
means, .005 quartet means and .08 distance correlations. These goals concern
the precision of exploratory discrimination. Failed or inconclusive targets
will remain visible; they are not reasons to force agreement.

The gate produces per-region/per-observable statuses, not one binary claim
that all admissible covariance matrices are now reliable. Broad sampling
must wait for the coordinator's review of this Stage I evidence.

## Executed pilot and resource check

The four-attempt pilot took **5.33 seconds**. It saved C019/C016/I3 degree6
paths, counted toward their planned cohorts. The C019 degree8 probe failed its
stored-polynomial positivity certificate in .031 seconds. That is a numerical
certificate failure for an SPD covariance; no clipping, repair, alternate
seed or repeated degree search followed. `pilot_cost_audited.json` clarifies
that the original pilot counter counted four **attempts**, including the one
failure, rather than four successful paths. Both records remain preserved.

The coordinator approved the remaining53 degree6 paths after reviewing these
costs. All56 planned degree6 paths succeeded, with no replacement attempts.
Generation plus saved coupled diagnostics took **92.97 seconds** summed over
the57 attempted records. The largest retained path had293,712 points. Outputs
occupy about84MB before final figures/source manifests, below the250MB disk
estimate; the memory estimate was not presented as a measured RSS maximum.
The final saved-data analysis took17.45 seconds; legacy-path coupling added
1.04 seconds. No covariance sweep or new sampling method was launched by
this workstream.

The exact original pilot source was recovered from its saved implementation
snapshot and checked against the recorded SHA256. Subsequent analysis helpers
and tests are saved separately in the final source snapshot. Source hashes,
seeds and configurations distinguish all cohorts.

## What the two-level comparisons show

At n2048, each of the ten configurations has six coarse and eight finer
provisional parents. Three finer cohorts are reused unchanged from Phase3B;
the others are new. All graph sizes on a physical path remain paired. The
following are signed finer-minus-coarse shifts using ensemble **mean** scale:

| C | Mean-distance change (MC95) | Normalized pair W1 | Largest absolute-minimum joint-CDF shift ± SE |
|---|---:|---:|---:|
| C002 I3 | −4.28% [−11.47%, +1.88%] | .0554 | −.275 ± .123 |
| C007 A2 | +4.09% [−.71%, +8.90%] | .0516 | −.259 ± .113 |
| C012 A3 | +2.78% [.13%, +5.66%] | .0405 | −.051 ± .074 |
| C006 birect−.7 | +1.21% [−1.65%, +4.96%] | .0260 | +.096 ± .113 |
| C010 birect+.5 | −3.06% [−6.55%, +.81%] | .0448 | +.169 ± .127 |
| C000 equi−.3 | −.97% [−5.43%, +2.76%] | .0232 | +.111 ± .101 |
| C004 equi+.3 | −4.60% [−9.74%, +1.28%] | .0557 | −.135 ± .158 |
| C016 unequal | +3.98% [−.22%, +8.93%] | .0526 | +.099 ± .138 |
| C019 difficult | +4.23% [−3.09%, +12.32%] | .0535 | −.273 ± .100 |
| C028 mixed | +2.05% [−2.65%, +6.52%] | .0335 | −.322 ± .083 |

These are small independent method cohorts, with degree, spatial step,
relative step, initial SIR count and exit quadrature changing together. The
32-draw whole-realization bootstrap gives descriptive pointwise MC intervals,
not precision confidence statements or a separation of every systematic term.
Same-method disjoint-split median W1 values range .024–.059 using smaller
three/four-parent halves. Thus much observed pair-law variation is comparable
to the current MC floor. Means alone do not establish the intended2% goal.

Absolute cell-minimum distributions retain real random path amplitudes; their
large shifts cannot simply be dismissed after mean-distance agreement. An
additional **rank-copula** diagnostic was declared before computing it:
coordinatewise minima ranks have joint CDFs at .25/.5/.75. They are invariant
under separate strictly increasing coordinate changes, just as graph
adjacency is; they isolate graph-relevant joint rank dependence while leaving
the absolute path-law diagnostic intact. A working .025 copula goal is not
an asserted error bound. Some copula shifts remain sizeable: C028 −.0605
±.0184 SE, C016 −.0731±.0291, A2 +.0907±.0334. These are maxima over several
correlated thresholds, so they are not multiplicity-free hypothesis tests.
Temporal local-turning fractions and degree summaries are also saved.

The compact error-budget CSV covers110 region/observable combinations:
16 have empirical pointwise intervals within their working goals,83 are
uncertainty-limited relative to those goals, and11 belong to unsafe C019.
Every entry remains qualified as empirical sensitivity, not a uniform
numerical error bound. All valid signed features from the declared47-feature
layout and median-scale sensitivities are saved in JSON; unavailable features
are omitted from a contrast rather than interpreted as geometric zeros.

## Independent exact-reference controls at the screening size

To test the complete general pipeline, new degree6 provisional control graphs
were compared to the48 preserved Phase3A graphs/model at n2048. Both sides
use the first eight roots and8×8 sampled matrices, eliminating finite-point
quantile differences. I3 references have exact continuous minima; A2/A3 have
q32 polygonal gap minima. No reference paths were regenerated.

| Model | General-pipeline mean change (MC95) | Mean-normalized pair W1 |
|---|---:|---:|
| I3 | +1.27% [−2.60%, +3.99%] | .0263 |
| A2×R+ | +3.00% [−2.38%, +9.52%] | .0442 |
| A3 | +4.32% [+1.06%, +7.41%] | .0522 |

The A3 residual is material and is preserved. It could mix remaining path/minimum
error, reference q32 error and finite-cohort fluctuations; this experiment
does not assign it uniquely to one mechanism. **Tiny harmonic defects do not
validate the complete general sampler.** Exact special-cone references should
continue to supply the main control geometries. Generic correspondence claims
at the level of a few-percent metric change remain unsupported.

## Genuinely paired observation-grid checks

All56 new fine paths were thinned at q8/q32/q128 while retaining the same
underlying polygonal realization and junction. The three reused fine cases
also have one preserved full path each, checked against their saved minima
and thinned without resimulation. Those one-path checks do not estimate
population uncertainty. At n2048, among the seven eight-parent cohorts:

| Uniform observations/cell | Edge Jaccard versus retained full path | Largest absolute cohort mean-distance change |
|---|---:|---:|
| q8 | .8266–.8337 | .781% |
| q32 | .9340–.9394 | .495% |
| q128 | .9808–.9826 | .351% |

These coupled results show that observation grids can still change local
edges when the metric mean barely changes. They do not control the Brownian
paths omitted inside sub-balls, the terminal tail, approximate harmonic or
initial angular law. The full q32 paired feature bootstrap retains roots and
all matrix subsets within each physical cluster. It is explicitly different
from comparing independently seeded coarse/finer WOS ensembles.

## Region decision and prospective monitoring

**C019 is unsafe for family interpretation.** Its new degree6 cohort has
largest absolute signed leg defect .05460 and largest sum of absolute leg
defects .27094; degree8 failed positivity certification. The old coarse
figures .15234/.37555 are not repaired by the finer setting. Retain C019 data
and failure, and do not use a low geometry score to override the warning.
The previously diagnosed C023 and failed C022 are outside this ten-point gate
and retain their Phase3B exclusion/warning statuses.

Other new degree6 maxima (absolute signed sum / absolute-term sum) are:
I3 ~1e−11/~3e−11; A2 ~1e−12/~2e−12; A3 .0000345/.000215;
equi+.3 .00614/.01081; birect+.5 ~1e−10/~2e−10; C016 .00682/.01435.
Reused C028 gives .00567/.02213. These support monitoring, not a global pass.

The coordinator accepted **provisional broad discovery** subject to warning
gates |signed leg log-defect sum|>.03 **or** sum of absolute terms>.10.
These thresholds correspond to normalizer contributions at roughly3%/10%
likelihood scales per leg, while omitting unknown endpoint, angular and
Brownian path terms. They are warning thresholds, never true law-distance
bounds. Monitor every parent: C019's first degree6 pilot was below these
thresholds, while the eight-parent maximum exceeded them.

Retain all parents/failures and flag the **whole configuration** if a warning
occurs. Discarding individual bad paths and replacing their seeds would
condition the ensemble and is not an allowed accuracy repair. No uniform
covariance neighborhood is certified by these ten points. All other regions
remain provisional; observable-specific sensitivity and MC uncertainty must
be carried into family claims. Failure to detect a difference at this sample
size is not an equivalence conclusion.

## Reproduction and completed checks

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3c_reliability plan
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3c_reliability pilot
# Coordinator review preceded the actual next invocation.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3c_reliability generate
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3c_reliability analyze
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_reliability_phase3c.py
```

Generation reuses exact existing checkpoints and never replaces a saved
failure. Completed analysis is refused rather than silently overwritten;
initial analysis before copula/reference additions remains preserved in its
own subdirectory. For a full independent reproduction use a fresh `--output`.
Six new tests cover covariance-label alignment, exact coarsening of saved
minima, truly nested polygonal observations, analytical Wasserstein values,
explicit mean normalization and rank-copula invariance. Final source/data
manifests and archive-integrity checks accompany the completed gate.
