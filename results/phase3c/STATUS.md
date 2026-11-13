# Phase 3C status — complete; stopped for review

**Latest checkpoint: Phase 3C is complete.** All frozen physical simulations,
analyses and tests are saved. No further covariance sweep or precision
extension is planned. Earlier entries below are historical. Preserve all
paths and Git exclusions. No Phase 3D is authorized.

12 November2026. See [technical plan](../../docs/TECHNICAL_PLAN_PHASE3C.md)
and [task specification](../../docs/REQUEST_PHASE3C.md).

Completed at startup:

- Recovered priorreports,code,registries,calibration andsealed3Bhandoff.
- Assigned independent sampler/field/statistics workstreams;old source/data
  remain unchanged.
- Declared ensemblemean normalization primary,median sensitivity,both fixed
  gauges,linkedgamma(xi) andwhole-realization analysis before3Cdata scoring.
- Preparedinitial109pointdesign target andresourcebudget;~12adaptive slots
  reserved forfamily discovery rather than globalargmin concentration.

Current work:

- Stage I: representative sampler gate,existingdatareuse andsmallpilots.
- Field stage: cachedfield inventory,compatible16pointmetric reuse andpilot.
- Statistics: old-data-only tolerance calibration andfamilyanalysis code.
- Root: covariance design andcheckpointedgraph campaign setup.

Pending: finishStageIdecision before >100covarianceattempts; broadgraph/field
screen; low-Jcomponents/localrank; adaptive additions; familyrefinement;
frozenindependentconfirmation; finalreport/tests/manifests.

Exact experiment commands,batch completion and updated resource estimates
will be appended as stages complete. No Phase3D work has begun.

## Checkpoint: reliability gate and field screen completed

- StageI is complete:56 newfine paths plus reusedcoarse/fine/exact controls;
  one difficultD8certificate failure retained. See
  [gate summary](calibration/sampler/analysis/summary.json) and
  [coordinator decision](calibration/sampler/coordinator_decision.json).
  Genericfew-percent geometryerror remains unresolved. C019/C023 areunsafe;
  newlygenerated settings arealsoexcludedif anyparent has signed accumulated
  logdefect>.03inabsolutevalue or absolute-termsum>.10. Theseareexclusionflags,
  not true path-law bounds. No parent is selectively discarded/replaced.
- Covariancedesign is frozen:109totalincludingold33;20birectangular,
  15equicorrelated,25two-correlation,50generic,withI3shared. All4,130previously
  sealedartifacts matched atstartup. See
  [design](covariance_design/initial_design.json) and[recovery](recovery.json).
- Initial field screen iscomplete:48families×14xi atN32 and12ofthesamefamilies
  ×14xi atN48,all16point/26direction.840metricrecords,199exactcachecopies;
  source/coefficient/CRC/metric checks pass. Fivebatches589.04s;pilot19.81s.
  See[registry](fields/registry.json),[audit](fields/integrity_audit.json).
- Calibration interpolationaudit found inheritedhighprobes under-cover some
  modelwise-lambda brackets (largestgammadifference .15073 atxi.2875).
  Broadresults remainunchanged andflagged; selectedfieldproducer willusea
  separatecorrectmodelwise rule. See
  [audit](calibration/probe_interpolation_audit.json).
- Familytolerances werefrozen usingolddataonly,before3C scoring. Grounded/mean
  primaryepsilon .26041420isaloose smaller-cohortnoise reference;unitmass/mean
  .20661248. Componentwiseandsigned residualchecks remainnecessary. See
  [predeclaration](families/statistics/predeclaration.json).
- Graphscreen isrunningin19checkpointedbatches. Existing31six-pathcohorts
  reused;1110newpathsrequestedto reach12parentsatsuccessfulsettings.
  Batchplans/reusehashes:[campaign](screen/campaign.json).
  First13batchescomplete;laterbatchCOMPLETEfilesarethe authoritative progress.
  C058/C065 alreadyhavenumericalmodel-certificate failures,retainedexplicitly.
- Saved-datascoringpilot passed:6configs,48fields,14xi,4weights,bothgaugesand
  normalizations,4feature-bootstrapdraws,68.5s. Mainanalysiswilluse24draws,
  withcheckpointsevery8;estimated3–6minutes. Pilotnotusedasfinalfamilyevidence.

Graphcommandsinprogress:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3c_graph_campaign run --first 12 --last 19
```

After all19batches complete, runfull saved-data comparison into
`screen/analysis/` and observed-family analysis. Adaptive12points,representative
familyrefinement and independentconfirmation remainpending. No holdout data
havebeenexamined. No Phase3D work hasbegun.

## Resumption checkpoint — broad coverage complete

Recovered after the usage interruption; the last graph batches completed
successfully. **Do not rerun the broad campaign.**

- All19batchmarkers exist. The109attemptedcovariances include106successful
  cohorts with12independentpaths each:1,272parents and2,544paired-size records.
  1,086parents are new and186were reused. C022/C058/C065 have explicit
  numericalpositivity-certificate failures; no missingbatch or partialarchive.
- The completed saved-data graph audit is
  [screen/audit/summary.json](screen/audit/summary.json). It labels1exactI3,
  2exact-grid references,81provisional configurations and22unsafe sampled
  settings. Allparents andfailedattempts remainrecorded; no badparent is
  selectively discarded. All graph/CDF/matrix/pairing checks pass.
- Recovery CRC checks passed for3,607NPZarchives. All4,130artifacts from the
  sealed3Bhandoff remain unchanged. The
  [recoveryinventory](resumption_20261112/completion_inventory.json) records
  completed,provisional,pending andoptional work. Fullsuite:193tests pass.
- Fullbroad saved-data comparison is now running into `screen/analysis/`:
  24whole-parent feature-bootstrapdraws, bothgauges andnormalizations,4weights.
  Its [budget](resumption_20261112/analysis_budget.json) is3–7minutes and~700MB.
  It checkpoints pointfeatures andevery8bootstrapdraws. This is analysis,
  not another graph/field sweep.
- After `screen/analysis/scores.json` exists, run the prepared family driver
  with both bootstrap NPZ paths and `--propose-adaptive --first-new-id109`.
  It can reuse saved graph/field feature draws for localrank/levelset diagnostics.
- The100-family selectedregistry is **a plan only**:52additionalGaussian
  fields have not been generated. Currentresourcepriority is selected N48
  refinement andindependentconfirmation, rather than precision-only N32
  expansion when sampler systematics dominate. Freeze the actual scope after
  broad-family interpretation.
- Confirmation producer preparation is assigned; no confirmationfields,
  metrics orgraphs have been generated or examined. Reservedseeds remainunused.

Remaining mandatorywork: fullfamilyresponseanalysis; targetedadaptive additions;
representatives spread across observedcandidate regions; feasible refinement
andfrozenindependentconfirmation; complete3Creport,log,tests andfinalmanifest.
No Phase3D work hasstarted.

## Completion checkpoint — broad search, refinement and holdout saved

- Combined coverage is121distinct canonicalSPD attempts,117successful coarse
  cohorts,24unsafe sampled configurations and4positivity-certificate failures
  (C022/C058/C065/C116). Broad/adaptive screening has1,404physical parents and
  2,808paired-size records; no completed broad dataset was regenerated.
- The complete response surface and its24whole-parent feature-bootstrap draws
  are saved at `screen/analysis/` and `adaptive/analysis/`. Observed components,
  signed profiles, adjacency sensitivity,localrank and direct graph-levelsets
  are at `families/{screen,adaptive}/`. Twelve adaptive additions were allocated
  to coverage/observed edges/uncertainty/disagreeing components,notglobalargmin.
- Selected graph refinement iscomplete:12configurations,24parents each at
  paired2048/8192,including40reused exact/exact-grid parents. The24separate
  finest general-kernel anchor controls are not pooled as exact references.
  Paths:`families/refinement/{reused_graphs,new_graphs,new_a3,analysis}`.
- Selected field refinement iscomplete:48discoveryparents,576records atpaired
  N32/N48,sixxi,98directions,true modelwise-lambda probes. No52-field optional
  precision extension was generated. Source hashes and low-mode checks are saved.
- Independent confirmation was frozen before data generation at
  `confirmation/frozen_plan.json`:11representatives,16newgraphparents each
  (176total),all `confirmation/graphs/batch_00` through `batch_05/COMPLETE.json`.
- All frozen field outputs are present:24newfamilies,120logrecords perN32/N48,
  24flat and24smoothrecords pergrid (smoothuses8parents×3xi),24N64records on
  the same8parents×3xi,and24matched26-direction records. See
  `fields/confirmation/{log,flat,smooth}` and
  `fields/confirmation_stencil_control/log`. The scheduler completed its
  boundedsubprocesses despite the specialist later hitting its usage limit.
- Principal report draft exists at `../RESEARCH_REPORT_PHASE_3C.md`.
  Frozen-hypothesis confirmation analysis isrunning via
  `experiments/phase3c_confirm_analysis.py`,with physical-parent bootstrap
  caches at `confirmation/analysis/`. Pairedchecks and finaltables follow.
- Audit discovered the legacy energy-U bootstrap percentile range is
  upwardbiased by repeated physical panels. Preserve old outputs,do not use
  those intervals as null tests. New analysis reports descriptive ensemble
  energy plus a separately labeled parent-shape permutation diagnostic.
- Remaining:complete saved-data confirmation/pairedchecks,figures/tables,
  fulltest suite,rawcheckpoint/sourcehash audit,finalreport/log/manifest.
  No further physical simulations and no Phase3D are planned.

## Final handoff — all required feasible stages closed

- **Complete:** 121 covariance attempts, 117 successful coarse cohorts;
  broad response surfaces, 12 adaptive additions, representative refinement,
  frozen independent confirmation, both gauges, global calibration models,
  negative controls, paired N64/stencil checks, tables, figures and report.
- **Still provisional scientifically:** 90 permitted numerical-kernel coarse
  configurations, including analytic-harmonic paths whose continuous joint
  minima are approximate; all arbitrary-covariance family interpretations.
  Twenty-four unsafe sampled configurations and four certificate failures
  remain excluded from family claims and fully preserved.
- **Independent evidence:** negative equicorrelations −.30/−.225 and unequal
  C110 reproduce nominal ξ=.225 compatibility in both gauges. The whole
  proposed birectangular ξ=.25 level segment fails componentwise confirmation.
  No unique Φ(C), continuum correspondence or matching-set dimension is found.
- **Inventory:** 1,724 newly generated physical Brownian paths, after seed/C
  deduplication and excluding prior Phase 3B paths. Gaussian families: 48
  discovery +24 independent holdout =72 used; 36 new and36 reused. Repeated
  sizes/roots/probes/stencils are not additional independent realizations.
- **Tests:** 214 pass, 0 failures/errors/skips,10.27s, `final_tests.xml`.
  Read-only audit:5,717 NPZ CRC checks, no partial archives; prior4,128
  immutable finalization artifacts and120 old source files unchanged.
  See `final_integrity.json` and final SHA256 `final_manifest.json`.
- **Final analysis:** `confirmation/analysis/COMPLETE.json`,
  `confirmation/paired_checks/COMPLETE.json`, and `tables_final/`.
  A diagnostic dictionary error was repaired using cached bootstrap features;
  `confirmation/analysis/diagnostic_resume.json` records zero regenerated
  physical samples or expensive feature bootstraps.
- **Principal output:** [final report](../RESEARCH_REPORT_PHASE_3C.md), outside
  the excluded phase directory. All figures and full machine-readable matrices
  are linked there. Archive paths and Git exclusions are unchanged.
- **Optional, omitted:** extra52 discovery fields, N96, larger generic graphs
  beyond the measured retained-path cap, and another covariance sweep.
  These would not resolve the current systematic/identifiability obstacles.
- **Pending required simulations: none.** Mathematical identification of a
  continuum family remains unresolved, as the report explicitly concludes.
  Stop here for review. Do not begin Phase 3D without new authorization.
