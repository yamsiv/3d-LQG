# Linked LFPP/GMC families for Phase 3C

This document owns only the Euclidean field campaign. Its role is to make a
shared collection of intrinsic metric-measure observations available across
the broad covariance search. It neither identifies a covariance-to-field map
nor treats correlated observations as additional independent realizations.
Previous code, field arrays, metrics, calibration and reports remain unchanged.

## Mathematical conventions

The original local papers consulted for these conventions are
Dello Schiavo–Herry–Kopfer–Sturm,
[2302.02963v1](https://arxiv.org/abs/2302.02963v1), Lemma1.4 and §§3.1,4.1;
Contreras Hip–Zhuang,
[2504.09141v1](https://arxiv.org/abs/2504.09141v1), equations1.1–1.3,
Definition1.7 and Proposition1.8; and
[2512.06292v1](https://arxiv.org/abs/2512.06292v1), equation1.5 and
Corollary1.9. The prior complete source audit is preserved in
`docs/COMPARISON_PROTOCOL_PHASE3A.md` and
`docs/PARAMETER_CALIBRATION_PHASE3B.md`.

The centered unit-torus field has covariance operator
`2*pi^2*(-Delta)^(-3/2)`, spectrum `S(k)=1/(4*pi*|k|^3)` for nonzero integer
modes, and sharp cutoff `|k|<N/3`. Its separated-point covariance has unit
logarithmic coefficient. The zero mode is absent. NumPy FFT normalization
and exact deterministic point variance are handled by the unchanged
`src/log_field_3d.py`. This is a continuum Fourier cutoff, rather than an
inverse nearest-neighbor lattice Laplacian.

For the same h, GMC cell masses are
`N^-3 exp(gamma*h-gamma^2*Var(h)/2)`. Save their total M and normalized
probabilities separately. In unit-log dimension3, `|gamma|<sqrt(6)` is the
subcritical measure range, and a positive q-moment requires `q<6/gamma^2`.
Consequently high uncertainty probes around gamma1.74 need not have a finite
continuum second mass moment. Finite-grid moments exist, but a few dozen
fields cannot validate their continuum rare tails. This campaign reuses the
previous validated Gaussian/GMC implementations rather than declaring a new
mass-expectation validation from48 realizations.

The LFPP target is the path length integral of `exp(xi*h)`. For intrinsic
torus comparisons the finite metric uses the existing26-direction lattice
graph, integrating the trilinear interpolant of the exponential weight along
each segment. At xi0 this is a fixed polyhedral norm, not an exact Euclidean
metric. Refining N alone does not remove directional anisotropy. Paper-faithful
axis **open-cube crossings**, used for exponent calibration, are a different
observable and are not recomputed in this campaign.

The theoretical identities are

```
lambda_cross=1-xi*Q,
Q=3/gamma+gamma/2,
gamma=6/(Q+sqrt(Q^2-6)),   Q>sqrt(6),
d_gamma=gamma/xi.
```

The numerical gamma(xi) comes from the existing five-resolution Phase3B
crossing calibration. It remains a finite-window proxy; interpolation does
not supply a new exponent measurement. The central map, MC uncertainty,
invalid-model flags and sensitivity probes are saved under
`results/phase3c/calibration/`. No gamma/xi Cartesian optimization is presented
as a theoretical LQG family. The flat endpoint xi=gamma=0 is separate.

One inherited interpolation detail is explicit: the old API interpolates
central lambda and then inverts, but interpolates the gamma-envelope endpoints
directly. At an intermediate xi, its nominal low probe can therefore exceed
central gamma slightly. The labels **low/high mean inherited endpoint-model
sensitivity probes**, not a guaranteed pointwise bracket. The central probe
is always retained in robust scoring. The main campaign preserves this
convention to permit exact cache reuse; a saved-data-only modelwise-lambda
interpolation audit reports its size separately. Invalid models are not
silently turned into real gamma values.

## Physical families, randomness and reuse

The immutable registry `results/phase3c/fields/registry.json` contains48
independent physical Gaussian field families:

- IDs0–23: all24 saved Phase3A confirmation families, each originally coupled
  at N24/32/48.
- IDs24–35: all12 saved Phase3B holdout families, each originally coupled
  at N32/48. These now count as **screening data**, not fresh holdout evidence.
- IDs36–47: twelve new common-Fourier N32/48 families. Field seeds are
  `SeedSequence([202611123100,index,1])`; their root streams use final key2.

The registry records actual field seeds, original complete requested grid
tuples, root seeds, source-file hashes and separate N32/N48 array hashes.
All48 seeds and each resolution's48 arrays are distinct. New seeds are checked
against recorded previous field archives and calibration seeds. The independent
confirmation base202611123900 is reserved and unused.

Changing maximum FFT grid with the same seed does not preserve common low
modes. Cached arrays are loaded directly, never regenerated under a changed
tuple. Within each new physical family, both requested sizes come from one
common-Fourier call. The source tuples differ between cohorts, but their
individual marginal field laws at a fixed N are the same.

Common iid root uniforms are retained across xi and gamma within each
physical field. Phase3A fields retain the actual Phase3B root stream
`seed_for(202611113701,cohort_index,2)`; the cached Phase3B fields retain
`seed_for(202611113861,cohort_index,2)`. These historical producers used the
same streams at N32 and N48. Reuse additionally checks actual sampled point
arrays rather than relying on a seed declaration. Common inverse-CDF uniforms
are not a claim that sampled locations are geometrically close across gamma
or grids.

There are seven reusable current-xi16-point26-direction N32 records for each
of the first twelve Phase3A fields:84 records. There are nine compatible N48
records for each:108 records. Original Phase3A metrics with8 points and
Phase3B holdout metrics with98 directions cannot be relabeled as these new
16-point26-direction observations. Completed Phase3C pilot records are also
reused. Every cache match requires source-field hash, N, xi, stencil,
boundary, gamma probe labels/values and all16 sampled point indices to agree.
Copied numerical arrays remain exactly equal to the source arrays.

## Checkpoints and resource plan

The code is `src/phase3c_fields.py` and `experiments/phase3c_fields.py`.
The registry, batch plans, frozen source snapshots, pilot validation and per
metric NPZ checkpoints are under `results/phase3c/fields/`. Output format is
compatible with the previous `phase3b_search.field_groups` loader, with
globally unique physical IDs0–47. Each checkpoint contains gamma-probe
16×16 matrices, all-target weighted root CDFs, root means, sampled indices,
bin widths, unnormalized total masses, collision probabilities and observed
duplicate fractions. Duplicate iid measure samples are retained.

One Dijkstra calculation handles the union of distinct roots for a fixed
field/xi. Identical gamma probes and repeated roots are not recomputed.
Different xi values generally change shortest paths and require a new metric.
Full distance arrays are not retained for every parameter; raw16-point and
all-target root-CDF summaries suffice for the planned intrinsic statistics.

The pilot used fields0/36, N32/48 and xi .125,.225,.2875. Its twelve records
took **19.81s** with peakRSS473MB; two were exact old-cache matches. New
N32 records cost .69–.89s, and N48 records2.62–3.32s. CRC, source-array
equality, CDF monotonicity/endpoint and16-point metric axioms passed before
the full campaign was approved by the coordinator.

The approved initial campaign is:

- N32:48 physical fields, fourteen xi values
  `0,.1,.125,.15,.175,.2,.2125,.225,.2375,.25,.2625,.275,.2875,.3`.
  Four sequential batches of12 fields;672 checkpoints in total.
- N48: the first twelve physical fields on the same fourteen xi values;
  168 checkpoints, initially about58 new solves after old/pilot reuse.

The pilot forecast was6–8minutes for N32 and about3minutes for the N48
subset, each sequential process below0.5GB. These are specific field/metric
budgets; no all48×14 N48 run, selected refinement or independent holdout is
implicitly authorized. Scientific questions, expected costs/reuse and source
hashes are written before each substantial batch in `batch_plans.json`.

To reproduce the first batch without changing any completed result:

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3c_fields inventory
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3c_fields run --output results/phase3c/fields/screen --field-ids 0 1 2 3 4 5 6 7 8 9 10 11 --sizes 32 --label screen_00_11
```

The source/configuration hashes are frozen on first use. Per-metric atomic
checkpoints support resumption. A completed batch's label is immutable; use a
new invocation label when checking already completed records. Neither source
recomputations nor cache reuse count as new independent fields.

## Downstream analysis conventions

These raw metrics carry no fitted ensemble normalization. The coordinating
analysis prespecifies ensemble **mean** pair distance as primary scale, median
as sensitivity, grounded gauge as primary and unit-mass Weyl gauge as a fixed
parallel comparison. Unit-mass gauge multiplies each field metric by
`M^(-xi/gamma)` without changing normalized GMC or conditioning on M=1.
All roots, probes, xi values, grid sizes and gauges share the same physical
field cluster for bootstraps. They are not independent ensemble members.

Matching distributions or graph dimensions does not establish a common limit,
topology, canonical graph embedding, scalar covariance parameterization or
3D LQG identification. A graph GFF is not introduced or relabeled as the
Euclidean unit-log field.

## Completed initial campaign

All approved batches completed:

| Batch | Independent field IDs | N | Metric records | Exact cache copies | New solves | Runtime | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen00–11 | 0–11 | 32 | 168 | 86 | 82 | 69.38s | 312MB |
| screen12–23 | 12–23 | 32 | 168 | 0 | 168 | 117.01s | 341MB |
| screen24–35 | 24–35 | 32 | 168 | 0 | 168 | 117.96s | 345MB |
| screen36–47 | 36–47 | 32 | 168 | 3 | 165 | 112.98s | 424MB |
| resolution00–11 | 0–11 | 48 | 168 | 110 | 58 | 171.69s | 487MB |

The five batches total **589.04s**, with840 saved records. There are48
independent physical fields in total; the twelve N48 families are paired
reobservations, not an increase to60. The pilot's19.81s and twelve preserved
records are separate; pilot matches are never counted as new families.
The final fresh batch reused three pilot records instead of the initial
two-record estimate. This reduces the main solve count by one.

All840 records passed CRC, source-identity,16-point symmetry/diagonal/triangle
inequality, CDF monotonicity/endpoint and collision checks. All199 cache copies
are exactly equal to their source numerical arrays; previous files were not
modified. Maximum matrix asymmetry is6.66e−16. Every source/configuration hash
matches its frozen producer snapshot. All48 source archives passed CRC and
shared N32/N48 Fourier coefficients agree within2e−15. The existing field,
LFPP and parameter-conversion suites passed19 relevant tests.

The screening loader finds40 parameter/probe groups at N32 (one flat group
plus13×3 gamma-probe groups), each with exactly48 physical realization IDs.
N48 has the same groups with twelve paired IDs each. The coordinating analysis
can load `results/phase3c/fields/screen` using the previous
`experiments.phase3b_search.field_groups` interface.

## Saved-only interpolation audit and recommended sensitivity

The requested modelwise-lambda audit recomputes no field, path, crossing,
regression fit or Dijkstra metric. It interpolates each of the five existing
prespecified correction models' lambda curves before taking the real
subcritical inversion, then compares that finite envelope to the inherited
endpoint probes used in the main campaign.

Nominal-low ordering violations occur at xi .2125,.2375,.2625,.2875. The
largest discrepancy is at **xi=.2875**:

| Quantity | gamma |
| --- | ---: |
| Central finite proxy | 1.33967272 |
| Inherited nominal low | 1.35515184 |
| Inherited high | 1.62995763 |
| Modelwise-lambda valid-model lower | 1.33967272 |
| Modelwise-lambda valid-model upper | 1.78068778 |

Thus the inherited low is.01548 above central, and the inherited high is.15073
below the modelwise upper endpoint. These differences reflect interpolation
and model-branch sensitivity, not a different field sampler. At the original
calibration knots .225,.25,.3 the conventions agree. At .3 an invalid
correction model is still excluded and explicitly flagged; no finite envelope
there is a complete error bound.

If a proposed family relies materially on xi=.2875, use genuine modelwise
probes in a **separate selected sensitivity setting**, preserving the main
campaign and its declared cache-compatible convention. The existing raw data
remain valid for their recorded gamma values. A new parameter setting should
not be relabeled as a completed simulation or used to retune a frozen holdout.

Artifacts:

- [Registry](../results/phase3c/fields/registry.json),
  [batch plans](../results/phase3c/fields/batch_plans.json),
  [source coupling audit](../results/phase3c/fields/coupling_audit.json),
  [integrity audit](../results/phase3c/fields/integrity_audit.json).
- [Linked parameter grid](../results/phase3c/calibration/parameter_grid.json),
  [modelwise interpolation audit](../results/phase3c/calibration/probe_interpolation_audit.json)
  and [CSV](../results/phase3c/calibration/probe_interpolation_audit.csv).
- [Probe comparison figure](../results/phase3c/calibration/parameter_probe_audit.png)
  and [PDF](../results/phase3c/calibration/parameter_probe_audit.pdf).
- Reproducible saved-only audits:
  `python -m results.phase3c.fields.check_integrity` and
  `python -m results.phase3c.calibration.probe_interpolation_audit`, using fresh
  output names if already completed. Both reject overwriting their results.

No field simulation or metric computation remains running. Selected-family
refinement and independent confirmation have not been launched by this
workstream; those require the next coordinated stage plan.

## All five primary-source conventions checked

| Original source/version | Convention checked for this comparison |
| --- | --- |
| Budd–Castro2207.05355v1, §§3.3–3.4, equations32–43 | The graph target is the paper-biased cone excursion. The original/unbiased cone bridge is a different finite law. Cone alpha1 and susceptibility are not GMC gamma. Its Doob harmonic function is not the Euclidean Gaussian field. |
| Contreras Hip–Zhuang2504.09141v1, equations1.1–1.3, Definition1.7, Proposition1.8 | Physical epsilon is already in lattice site-sum lengths; lambda_cross has the sign used here. The unique subcritical gamma/xi parameterization is implicit through the limiting exponent. |
| Contreras Hip–Zhuang2512.06292v1, equation1.5, Corollary1.9 | Q=3/gamma+gamma/2 and xi=gamma/d_gamma are weak-metric subcritical relations. A real finite proxy does not establish that an approximation reached the limiting exponent. |
| Ding–Gwynne–Zhuang2310.03996v5, equations1.7–1.10, Proposition1.1, Theorem1.2 | Their lambda_n is a median normalization sequence, with lambda_n=2^(-(1-xi*Q)n+o(n)); it is not lambda_cross. Subcritical tightness and Euclidean topology of subsequential limits do not establish uniqueness or identify finite torus/stencil laws. |
| Dello Schiavo–Herry–Kopfer–Sturm2302.02963v1, Lemma1.4, §§3.1,4.1 | Grounding removes the zero Fourier mode. The unit-log3D spectrum is1/(4*pi*\|k\|^3), using a3=1/(2*pi²). Continuous Fourier cutoff and discrete fractional-Laplacian schemes have different finite approximations and theorem ranges. |

None of these references proves a Budd–Castro/LFPP/GMC correspondence or a
canonical Euclidean embedding for the graph. Fixed global distance rescalings,
including each realization's unit-mass Weyl gauge, do not change its Hausdorff
dimension; a finite-feature optimum therefore cannot resolve a dimension
discrepancy merely by changing that gauge.

## Prepared selected-family stage: not executed

The separate driver `experiments/phase3c_selected_fields.py` leaves the entire
completed screen and its producer unchanged. It uses genuine modelwise-lambda
gamma probes and also interpolates each **paired bootstrap lambda curve**
before inversion for MC metadata. The central gamma conversion remains
unchanged. All five model values, invalid real branches, conditional-bootstrap
fractions and finite ansatzes violating known theoretical lambda bounds are
recorded. The probe envelope remains a finite-model sensitivity, not an error
certificate. The old-knot .225/.25/.3 envelopes agree with the completed screen.

`results/phase3c/fields/selected_registry.json` is an immutable **plan** for100
discovery families: existing48 plus52 new global IDs48–99. The latter use
`SeedSequence([202611123110,global_id,1])` for fields and key2 for roots. Their
seeds were checked against recorded old field seeds and the prior calibration.
They are explicitly marked `planned_not_generated`; **none of these52 new
fields has been generated**. Each future field is coupled at N32/48 in one
call. Upon an authorized run, independent per-field sidecars record realized
archive/array hashes without modifying the seed plan or old48 registry.
The wholly new24-family confirmation base202611123900 remains unused.

The selected driver can copy individual matching gamma-probe rows from old
checkpoints, even when a new high sensitivity probe requires additional roots.
It builds a weighted graph only if a probe is missing, and computes the union
of missing roots once. It never relabels different gamma values, root sets,
stencils or boundaries as an exact cache match. All per-probe provenance is
retained. The checkpoint schema remains compatible with `field_groups`.

The proposed selected scope is100families at N32 and the existing48families
at N48, at approximately six **coordinator-selected xi values**. That list
has not been frozen. The template
`results/phase3c/fields/selected_template.json` therefore refuses metric jobs.
The driver requires a configuration with `status='frozen_by_coordinator'`,
`metric_jobs_authorized=true`, an explicit xi list and
`gamma_rule='modelwise_lambda'`. This is a scientific staging guard, not an
additional request for user permission.

Conservative no-cache bounds from the completed pilot are .95s per N32 record
and3.6s per N48 record. For six xi values this is570s for100N32families and
1,037s for48N48families; exact/per-probe reuse should reduce those totals.
At existing calibration knots the old48 N32 records can often be copied in
full. At intermediate nodes central rows remain reusable and only genuinely
different sensitivity probes need paths. A batch of12fields×6xi is bounded
by roughly68s at N32 or259s at N48, with measured working memory below0.5GB.
The driver enforces at most12 fields per invocation and saves a batch plan
before any Gaussian-field materialization or metric calculation.

Planning command already completed without sampling:

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3c_selected_fields plan
```

Once the coordinator freezes xi, `estimate --config <frozen.json>` generates
the exact scope/budget/conversion table without sampling; a subsequent
`run --config <frozen.json> --field-ids ... --sizes32 --label ...` performs
only that explicit batch. No selected metric job or new holdout was launched
when this preparation was saved.
