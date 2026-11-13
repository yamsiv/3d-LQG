# Independent field confirmation protocol for Phase3C

Prepared before observing any new holdout field or metric. This document and
the producer do not freeze a scientific parameter choice; the coordinator's
testing plan must do so first. The original broad fields and selected100-family
prototype are preserved unchanged. The52 planned discovery extensions remain
ungenerated; they are not substituted for independent confirmation.

## Independent physical identities

`experiments/phase3c_confirmation_fields.py plan` creates the immutable
`results/phase3c/fields/confirmation_registry.json` for24 wholly new physical
field families, global IDs1000–1023. Seeds are
`SeedSequence([202611123900,global_ID,1])` for the Gaussian field and final key2
for iid root uniforms. The plan checks both streams against previous recorded
field/root seeds, including the unused52 discovery reservations, and records
prior field archive/array hashes. There is no hidden reseeding or retry.

All24 registry entries remain **planned_not_generated** until an authorized
frozen-plan run. Actual source generation saves both h32 and h48 in one
common-Fourier call, checks their archive/array hashes against all prior field
data and other new parents, and saves an immutable per-parent sidecar. The
original archive is reused for every xi, measure probe, control and grid.
Coupled reobservations do not add independent parents.

## Frozen metric-measure experiments

The main log-field model uses exactly the existing unit-log Fourier spectrum,
zero-mode removal, deterministic variance subtraction and torus boundary.
The gamma conversion interpolates each of the five prespecified correction
models' **lambda** curves before subcritical inversion. The central conversion
is unchanged. Invalid branches, the paired-bootstrap validity fraction and
known-theorem bound violations of finite ansatzes remain visible.

Primary records retain the three labels central/low/high. A frozen option
`diagnostic_model_controls=['log']` additionally saves one named probe for
each valid model, such as `model_all_plain` or `model_all_Ninv`. Equal gamma
values alias the same points/distances/CDFs; eight labels need at most five
distinct gamma calculations. Headers include `model_gammas`, `probe_models`,
`primary_probe_labels` and `diagnostic_probe_labels`.

The named model is **one global calibration curve across all xi and graph
covariances**. It is not an independently optimized gamma per graph. Diagnostic
model curves must not silently enter a previously defined primary objective.
In particular, a robust maximum over sensitivity probes is a **sufficient
uniform-model compatibility check**, not a necessary condition for any
correspondence. Failure of one high probe does not rule out the central curve
or another single, globally consistent calibration model.
The finite probe maximum is not a bound over unsampled gamma values or other
systematic errors; named five-model diagnostics explicitly test those five
prespecified alternatives.

For each field, all parameter/measure settings share one stream of16 iid
uniforms. Inverse-CDF sampling preserves collisions and couples the sampled
points without claiming they are spatially close across gamma. Save16×16
distance matrices, weighted all-target root CDFs, root means, total mass,
collision probability and duplicate fractions. Each field remains one cluster
for uncertainty, including all roots, matrix subsets and model probes.

## Fixed negative controls and gauges

Both controls use the same parent IDs and root-uniform stream as the main
log-field model:

- Flat: xi=gamma=0, uniform volume, deterministic zero field. The98-direction
  metric remains a polyhedral finite-stencil norm, not exact Euclidean distance.
- Smooth nonlog control: retain Fourier modes0<|k|≤2 and multiply by the
  square root of the ratio of the original deterministic point variance to
  the retained variance. The smooth field drives both its exponential metric
  and its normalized exponential measure. This is a finite, variance-matched
  specificity control, not a log-correlated GMC continuum construction.

The frozen plan states which smooth xi values and which paired parent subset
are used. The intended economical option is the first8 parents at2–3 fixed xi,
with central gamma only; flat uses all24 parents. No parameters are selected
from holdout scores. Raw grounded metrics retain unnormalized total mass.
Grounded and unit-mass Weyl gauges are derived later on these same records;
the latter rescales distance byM^(-xi/gamma), without changing normalized
measure or conditioning onM=1. Mean ensemble normalization remains primary,
with median a fixed sensitivity.

Output models live in separate `log/`, `smooth/` and `flat/` directories so
the existing `field_groups` loader does not combine controls or double-count
their physical IDs. Compare paired controls on their actual shared parent
subset, not an artificially enlarged ensemble.

## Validated finite-rule acceleration

`src/lfpp_periodic_3c.py` replaces repeated interpolation geometry with
positive periodic shift kernels. Trilinear interpolation of exp(xi*h) is
linear in its grid values, so the same fixed ray/trapezoid quadrature is a
finite positive weighted sum of shifted arrays. Each undirected ray is
computed once and assigned identically to its reverse. Cached sorted CSR
topology is shared across fields. Cube boundaries and interpolation of the
field itself explicitly fall back to the original builder.

Nine deterministic unit tests compare full CSR structure, weights, symmetry
and shortest paths for26/98directions, xi0/.2/.3 and nondefault quadrature,
plus both fallback cases. The N32/N48 deterministic benchmark has maximum
relative edge discrepancy8.72e−16 and path discrepancy4.45e−16 against the
unchanged source. N48/98 construction fell from8.31s to.332s cold/.161s warm;
N32/98 from1.97s to.083s cold/.042s warm. This is a change of positive
floating-point summation order, **not** a new finite stencil or continuum
discretization. Finite-stencil anisotropy and path-quadrature limitations remain.
The benchmark's510MB peak retained both builders simultaneously; production
memory is separately budgeted before actual holdout runs.

The new numerical engine `src/field_metrics_3c.py` computes the union of
distinct gamma roots once, and reuses duplicate labels. Tests also verify
smooth-mode projection/variance, gamma aliases and global-model labels.

## Scientific freeze and resource discipline

The template `results/phase3c/fields/confirmation_template.json` is deliberately
unfrozen. A metric run requires stage independent_confirmation, status
frozen_by_coordinator, an explicit nonempty hypothesis list and xi list,
modelwise gamma rule, fixed control xi/parent sets, and an authorized scientific
testing plan. These are coordination/staging guards, not extra user permission.

Every invocation accepts at most6 field IDs. Prefer separate model/grid
invocations: for example six parents for main N48, then main N32, followed
by the declared control subsets. Each field/model/grid/xi is checkpointed
atomically. Source code, calibration/bootstrap, original hypothesis/config
hashes and original-data disjointness references are frozen. Reused partial
confirmation batches must retain the exact same plan and sources.

Before any real holdout job, the coordinator will freeze hypotheses and a
small counting pilot. Development so far generated **zero holdout fields or
metrics**. The forecast combines saved deterministic kernel costs and archived
reference-builder costs; an actual frozen pilot must replace these estimates.

## Optional N64 extension: prepared, not executed

`src/extend_log_field_3c.py` offers a separate sharp-cutoff extension. It
overwrites a new independent proposal's retained active Fourier coefficients
with the original h48 coefficients and keeps the independent high modes.
Matching low-mode variances and Hermitian symmetry give the correct N64
Gaussian marginal and genuine low-mode coupling. Heat filters are rejected
because their retained coefficients have different scale-dependent variances.
The caller must use a separate high-mode RNG stream; the proposed reservation
is final key3 on the same physical parent seed tuple.

Deterministic8→12 tests verify retained and new coefficients, realness and the
new variance formula. **No Gaussian N64 field has been generated.** N64
metrics remain disabled in the unfrozen template; first8 parents at3
predeclared xi need an explicitly frozen scope and measured memory/time pilot.
Those remain reobservations of the same physical parents.

### Updated N64 preparation checkpoint

The confirmation producer now supports that optional scope behind an explicit
frozen `enable_n64` setting. The recommended template is
`confirmation_template_v2.json`; the original template remains unchanged.
Only IDs1000–1007 and exactly three predeclared main xi values are eligible.
N64 defaults to the central gamma probe for a focused resolution diagnostic;
the main N32/N48 data retain the primary and global-model diagnostics.

Each extension is a **separate** `confirmation_extensions/field_ID_n64.npz`.
Headers retain the physical parentID, parent archive andh48array hashes,
independent high-mode seed, actual shared Fourier coefficient error, and
resolution-specific extension archive hash. The base h32/h48 archive is never
overwritten. Audits must permit source-file hashes to differ across grids,
while checking that actual h48 Fourier coefficients survive in h64. Pair
N64 with N48 on the **same first8 parents**, not24versus8.

A deterministic analytic N64 metric pilot, with98directions and3 distinct
gamma query sets, took **10.18s** and peaked at **658MB**, below the declared
1.5GB budget. Its309MB CSR and all temporary arrays are included. It generated
zero Gaussian or held-out fields. The frozen producer checks this resource
record before enabling N64; absence or a failed memory gate disables that
scope. An actual held-out counting pilot is still required after hypotheses
are frozen. A conservative full-stress forecast for8parents×3xi is about
6minutes, despite the default central-only setting being cheaper.

## Prepared handoff and resource estimates

The completed selected-field refinement is separate from this holdout:
`results/phase3c/fields/selected_fast/` contains576records,48 existing parents,
bothN32/N48 and six frozen xi values. It generated no new Gaussian fields and
did not materialize the52 unused discovery reservations. Counting pilot plus
four batches took347.07s, with maximum measuredRSS358MB. All576 CRC/matrix/
CDF/source checks pass;803 reused gamma rows equal their source arrays
exactly. Those selected records remain training/refinement data.

For confirmation, the saved-only forecast including allfive named global
calibration curves gives approximately **15–23minutes** for3–5 main xi values
at N32/N48 on24 new parents, smooth first8parents at3 fixed xi and flat24.
This uses measured fast construction and root-search costs plus conservative
CDF/timing allowances; it is not a runtime guarantee. Split by model and
grid, with at most6 parents per invocation. Optional N64 adds its separately
declared forecast and1.5GB cap. Update estimates from the first frozen counting
pilot before the rest of the confirmation campaign.

Useful entry points:

- `experiments.phase3c_confirmation_fields.plan`: seed/independence plan only.
- `modelwise_conversion` in the preserved selected prototype: one coherent
  finite lambda-model curve at every xi, with real-branch/model flags.
- `tasks`: central/low/high plus optional named global-model diagnostics.
- `materialize_n64`: separate parent-linked extension; it is never called by
  planning/estimation.
- `forecast` or CLI `estimate`: saved-data-only cost calculation.
- CLI `run`: requires the frozen configuration, explicit model/grid/IDs and
  unique batch label; checkpoints every completed metric record.

For example, **after** the coordinator freezes the actual hypothesis/config:

```
python -m experiments.phase3c_confirmation_fields estimate --config <frozen.json> --output <holdout-directory>
python -m experiments.phase3c_confirmation_fields run --config <frozen.json> --output <holdout-directory> --models log --sizes 48 --field-ids 1000 1001 1002 1003 1004 1005 --label log_n48_1000_1005
```

No actual confirmation Gaussian field, extension or metric had been generated
at this preparation checkpoint. Deterministic numerical tests and saved-data
audits do not count as holdout realizations.
