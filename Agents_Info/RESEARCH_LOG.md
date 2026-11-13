# Research log

## 2026-11-10 — Task 1: Budd–Castro graph construction and intrinsic geometry

Read repository instructions, the research context, and Budd–Castro
arXiv:2207.05355v1 directly. Four agents handled coordination, construction,
metrics and independent validation. The current user scope supersedes the
context document's later field/LFPP/GMC proposals; none of that code was begun.

Implemented the full three-correlation parameter interface, reference and
optimized graph construction, retained multiedge provenance, sparse metrics,
diameter bounds, balls, cluster-bootstrap scaling, angular FEM, walk/resistance
diagnostics and reproducible experiment runners. Exact independent-coordinate
biased/unbiased sampling includes exact continuous segment minima. A2 Weyl
controls provide exact correlated grid values with documented polygonal minima.
The general SPD generator is explicitly a finite-grid MCMC approximation.

Executed 568 independent biased I3 graphs over nine sizes 2,048–524,288 with
eight uniform roots each, plus 288 biased I2 and 288 unbiased I3 graphs over
six sizes, 192 independent Weyl paths over six sizes with coupled resolutions,
204 small general-covariance/mixing diagnostic graphs, distribution tests and
secondary-observable studies. The combined I3 records concatenate the two
independent original runs and must not be counted as additional simulations.

The I3 naive inverse slope is 4.190 (MC 95% interval 4.141–4.239).
An additive-offset mean fit gives 4.833 (4.527–5.133), whereas shifted histogram
collapse gives 5.070 (4.254–6.008). All fitting windows are retained. These
are correction-sensitive finite-size diagnostics, compatible with the paper's
4.83±0.06 but not a precision replication or a proven dimension. Two-tree and
correlated A2 controls are broadly compatible within larger uncertainty.
Ball slopes drift with size; short-time walk slopes around 2.2–2.3 do not
establish a spectral dimension.

**Validation checkpoint:** exact independent graphs pass; the A2 metric pilot
passes a cautious resolution check (mean distances stable within about 0.7%,
individual edges less stable). The general-covariance dimension gate remains
closed. Coarse positive-grid conditioning changes distances, and rho=-0.49
chains still drift after 128 sweeps. No reliable d_H(rho) curve, near-corner
dimension minimum, general biased h-transform implementation, or parameter
correspondence was claimed. This is the explicit unresolved sampling obstacle.

The final test suite has 54 passing tests. At n=524,288 a fresh-process
benchmark used about 799 MB peak RSS; sampling/graph/metrics took about
0.27/5.90/1.55 seconds with four metric roots. Machine-specific numbers are
saved, not portable guarantees.

Artifacts:

- [Detailed phase report](../results/RESEARCH_REPORT_PHASES_1_2.md)
- [Combined raw graph records](../results/data/i3_biased_combined.jsonl),
  [provenance](../results/data/i3_biased_combined.metadata.json),
  [summary table](../results/tables/metric_summary.csv)
- [Main geometry figure](../results/figures/i3_biased_combined_geometry.png)
- [Independent validation and methods](../docs/VALIDATION.md)
- [General covariance diagnosis](../docs/GENERAL_COVARIANCE_DIAGNOSTICS.md)
- [Reproduction commands](../README.md)

Next graph priorities: higher independent large-size counts and correction
stability; exact/controlled correlated minima; then the paper's general
Doob/walk-on-balls sampler validated against both exact special cases. Stop
after this graph-phase handoff for user review before any field/GMC/LFPP work.

## 2026-11-11 — Task 2B: audit, analytic Doob sampler and general-cone obstacle

Resumed after a usage interruption by inspecting Git status, instructions,
sources, tests and saved results. Recovered completed work rather than
restarting it. Earlier data and figures were preserved. The original exact
I3 law/minima, angle conventions, adjacency, metrics, seeded independence and
graph-bootstrap design passed an independent audit. Original estimates were
recomputed as 4.189503 (naive), 4.833297 (offset mean), 5.070283 (collapse).
The offset model is not the paper's final equation (45).

Eighty additional exact biased I3 graphs, forty at n=262,144 and 524,288,
finished before interruption after a cost estimate (~9 minutes, old measured
peak ~0.8 GB). Actual compute time was ~471 seconds. There are now 648 unique
graph and metric seeds, with 64 graphs at each largest size. Updated naive
inverse slope is 4.215 [4.171,4.245] MC95; offset fit 4.910 [4.643,5.159];
shifted histogram collapse 4.916 [4.541,5.440]. A reference-normalized Eq. (45)
fit with delta=p gives 4.856 [4.650,5.123]. Other mean correction models give
central values about 4.09–5.40, and size-window sensitivity persists. These
are ansatz-dependent finite-size diagnostics, not a precise asymptotic result.
Cached bootstrap draws were reused for further checks; re-estimating variance
weights changes the offset interval little compared with model choice.

Implemented a separate paper-biased Doob/interior-ball sampler with analytic
harmonics for all 2D wedges and 3D birectangular cones/permutations. Records
distinguish bounded-walk spatial/time errors, relative step size, cutoff/tail
approximation and polygonal joint minima. No silent retry of failed paths or
unbiased/biased ensemble substitution is allowed. The old finite-grid MCMC
sampler remains explicitly approximate and separate.

The original P1 angular FEM has three open interior zero patches near corner
rays. Centroid-only enrichment fails; conforming interior-edge splits remove
zero support but retain severe relative corner and harmonic mean-value errors
(e.g. I3 refinement5 mean-value defect about −15.4% near a corner). The Doob
API now rejects unresolved zero-support meshes before sampling. Enriched FEM
kernels remain provisional because locally normalizing nonharmonic weights
does not produce the intended exact h-transform.

Validation includes 1,840 independent physical control paths across main,
wedge/permutation, one-factor, independent-repeat and Weyl-reference checks;
coupled size/resolution rows are not extra independent paths. Main Doob
controls have zero failures. Individual/joint minimum CDF errors remain too
imprecise for tight numerical control. A fine-I3 mean-distance offset repeats:
pooled fine mean 4.2720 versus exact 4.1903, difference MC95 about [.0282,.1351].
A low cutoff-only result did not replicate; all runs are retained. Mean ball
fractions are cumulative pair-distance laws, not independent validation data.

**Stopping decision:** the general correlated exponent gate remains closed.
No large three-parameter sweep or new unvalidated covariance dimensions were
generated. Exact I3 remains validated; A2 continuum-minimum and analytic Doob
controls remain provisional; current general FEM/MCMC paths are unreliable
for controlled exponents. No field, Gaussian-measure, GMC or LFPP comparison
stage was started. Final suite: **83 tests passed**.

Artifacts:

- [Phase 2B report](../results/RESEARCH_REPORT_PHASE_2B.md)
- [Updated fitting analysis](../results/phase2b/updated_independent_scaling_audited.json)
- [New exact large-graph records](../results/phase2b/data/i3_high_extension.jsonl)
- [Independent audit](../docs/AUDIT_PHASE_2B.md)
- [Correlated validation and error budget](../docs/CORRELATED_VALIDATION_PHASE2B.md)
- [FEM repair diagnostics](../results/phase2b/fem_enriched_audit/audit.json)
- [Final tests](../results/phase2b/final_tests.xml)

Next graph priorities: boundary-singularity-aware harmonic approximation,
whole-leg control of kernel defects, coupled joint-minimum/walk-clock
refinement, and restricted analytic-cone studies before general covariance
exploration. Stop for user review at this documented obstacle.

## 2026-11-11 — Task 2C: A3, corner-aware harmonics and complete-path stopping gate

Recovered the interrupted work from source, saved checkpoints and completed
specialist contributions. All declared path records parsed and matched the
experiment inventory; all five saved NPZ archives passed CRC checks. The
interruption left final integration unfinished, rather than an experiment
requiring restart. No completed numerical experiment was repeated. Preservation
checks compare prior source and datasets against their archived SHA-256 hashes.

Verified the A3 covariance (a,b,c)=(-.5,0,-.5), its degree-six harmonic
u=x1*x2*x3*(x1+x2)*(x2+x3)*(x1+x2+x3), spectral eigenvalue 42,
alpha1=13/2, and spherical normalization integral U²=pi/180180. Exact rational
tests check harmonicity and Gaussian moments. New 4×4 Hermitian bridge
samplers give independently accessible original and paper-biased excursion
grid laws. Their joint continuous gap minima remain approximate. Angular
validation used 8,000 GUE and 8,000 independent rejection draws; 2,400 original
paths and biased conditional moments support the derived grid-law identities.

Implemented two genuinely different angular strategies: fractional Ferrers
MPS with local corner continuation, and positive corner-factor Galerkin/Ritz.
The latter has exact face zeros, the correct leading corner shape and exact
rational positivity certificates for stored polynomial coefficients. Generic
degree 10 cross-method interior/corner discrepancies were .0490%/.387% for
(-.2,.1,.35), and .00210%/.0298% for equicorrelation .2. Observed mean-value
defects decrease, but these grids are not uniform relative-error certificates.
Global spectral cancellation, matching errors and high-degree conditioning
are documented rather than hidden. No generic graph paths were generated.

Derived and tested the exit-chain likelihood identity
dQ/dP=e(xN)/e(x0)*product(1+R_j)^(-1). Six saved analytic paths, twelve legs
and 5,467 balls validate the numerical diagnostic; they do not bound generic
full-path error. A local polynomial box bound greatly improves an avoidably
loose global rejection bound, without certifying the complete generic process.

Isolated a real old-walk clock error: at ell/r=.1 its mean unit-ball exit
time was about 6.06% too large; Gaussian grid increments alone gave6.63%.
A new Brownian sub-sphere skeleton uses exact-law sub-ball clocks, with
measured inverse-CDF and terminal projection errors recorded separately.
Its 4,096-clock mean .331967±.003220 SE agrees with 1/3.

Executed 192 coupled exact-grid controls (64 each I3/A2/A3), 192 complete
new-Doob controls, and a focused 48-path finer/interpolation check, all saved
before interruption. Coupled sizes/resolutions are not extra independent
graphs. At n=256, q=256 grid-reference distance changes are below .03% but
edge Jaccards remain around .964–.966. Full new-Doob mean differences are
+1.30%,−.21%,+1.27% across the three cones, with broad independent-graph
intervals; joint-minimum comparisons do not establish the .02 goal. Finer
I3 differs by +3.91% [+.61%,+7.21%] MC95, an unresolved discrepancy. No
small-size Hausdorff exponent was fitted. Main runs took roughly13,74,178
and 69 seconds after pilots; none approached15 minutes.

**Decision: Outcome C for a complete general-covariance sampler.** Angular
and clock advances are substantial, but the full path/minimum gate does not
pass. Existing exact I3 remains validated; A2/A3 provide exact grid laws with
explicit continuum-minimum error. Stop generic sweeps. Recommend the next
task temporarily build independent Euclidean 3D log-correlated fields, GMC
and LFPP references, retaining these qualified graph controls. No field/GMC/
LFPP implementation or large covariance sweep was started. Final verification:
**121 tests passed**; the resumed check and preservation audit are archived.

Artifacts:

- [Phase 2C report](../results/RESEARCH_REPORT_PHASE_2C.md)
- [A3 mathematical derivation](../docs/WEYL_A3_PHASE2C.md)
- [Spectral corner methods](../docs/CORNER_METHODS_PHASE2C.md) and
  [positive-factor Ritz method](../docs/CORNER_RITZ_PHASE2C.md)
- [Likelihood analysis](../docs/LIKELIHOOD_PHASE2C.md) and
  [path/minimum error budget](../docs/PATH_ACCURACY_PHASE2C.md)
- [Checkpoint inventory](../results/phase2c/paths/completed_experiment_inventory.json),
  [resumption audit](../results/phase2c/resumption_audit.json) and
  [final manifest](../results/phase2c/final_manifest.json)

Task 2C stops here for review. A different next phase requires the next user task.

## 2026-11-11 — Task 3A: executed Euclidean LFPP/GMC and graph comparisons

The user's new task explicitly authorized the field/GMC/LFPP phase. Read the
five original local sources and audited all parameter/normalization conventions.
Four workstreams handled coordination/comparison, field/GMC, LFPP and independent
mathematical/statistical review. Prior graph implementations, source versions,
reports and raw datasets were preserved; the general excursion sampler was
not repaired or used for a new covariance sweep.

Implemented unit-log torus Fourier covariance1/(4*pi*|k|³), zero-mode removal,
common Fourier randomness, sharp/heat filters and fixed-field resampling.
GMC retains unnormalized mass and normalized probabilities separately using
the exact deterministic point variance. The256-field N16/24/32/48 validation
has variance-growth slope 1.00426. Exact finite-grid second moments expose
rare-tail undercoverage: gamma2,N48 empirical mass SE .0568 versus true .2113;
mean .890 is not evidence of a renormalization bug. Coupled high-gamma cutoff
effects are large and explicitly retained.

Implemented proper LFPP edge metrics on torus/open cube, exact CHZ lattice
vertex-sum crossings via endpoint costs, and 6/26/98-direction stencils with
path quadrature/interpolation checks. Xi0 verifies L1 for 6 directions;98 rays
retain measured maximum flat angular error4.93% at N32. These finite stencils
are not mislabeled as exact Euclidean eikonal solvers.

Executed24 discovery fields at N16/24/32,30 gamma–xi pairs andaxis/26 paths.
Frozen choices: I3(.3,1.5), A2/A3(.2,0), the latter unmatched to the theoretical
gamma–xi curve. Executed24 independent confirmation fields at N24/32/48 with
26/98 paths, the selected/flat pairs and two nonzero-GMC brackets. Each physical
field/grid was checkpointed; coupled parameters are not extra independent
samples. Main discovery/confirmation metric checkpoints total202s/960s; the
98-ray confirmation exceeded its500s extrapolation and took about 16 minutes.
Eight-field mesh/filter/boundary controls and a small smooth non-log probe
were saved separately, without counting reused fields again.

Graph benchmarks have48 realizations/model/size at n2048/8192, with eight iid
measure points, root CDFs and full distance matrices. I3 regenerates96 archived
graphs only to add missing matrices, verifying previous root histograms.
A2/A3 contribute192 new exact-grid special-cone paths with q32 polygonal minima;
first-four q64→32 paired checks show small mean shifts but about 7% edge changes.
All648 existing large I3 records are reused for finite-size pair/ball analysis.
Actual selection/confirmation split is24/24 per model/size, superseding an
early design note proposing a different I3 split. Pilots are not extra samples.

Atn8192/N48, selected pairW1 values are .0353/.0328/.0334 and joint descriptive
objectives .0855/.0910/.0762, versus flat objectives .1287/.1430/.1221.
I3's paired objective improvement interval includes zero; A2/A3 improvements
over this particular flat control are clearer. Most improvement is root-ball
heterogeneity, not uniquely matched global pair/triangle shapes. Signed triangle
and ball-dispersion residuals remain; raw fields/matrices and cluster-bootstrap
intervals are retained rather than treating pair distances as independent.

Atn524288 (64 archived exact I3 graphs), selected pairW1 .0140 improves on flat
.0299; paired improvement MC95 [.0052,.0186]. Ball-dispersion discrepancy also
improves, while half-radius root mass SD .0185 remains below the field's .0274.
The grounded-field global CV differs; a separate unit-mass Weyl-gauge sensitivity
reduces field CV to ~.0391 versus the large graph's~.0406. This gauge rescales
distance by M^(-xi/gamma) and is neither conditioning onM=1 nor per-realization
median normalization.

Confirmation axis site-crossing slopes over N24/32/48 give finite proxies
lambda(.2)=.0673 [.0462,.1266], lambda(.3)=.1785 [.1278,.2123] MC95.
The latter implies gamma proxy 1.514 [1.341,1.680] and dimension proxy 5.048
[4.470,5.600], compatible with prior correction-sensitive I3 estimates. Adjacent
window dimension proxies 4.30/5.82 prevent a stable asymptotic conversion.
At xi.2 the proxy dimension 3.48 is in tension with the source A2 graph value 4.67;
a distributional optimum alone does not identify the theoretical family.

An auxiliary median-atom diagnostic initially inspected the next integer bin
because bisection's median lay a few ulps above an integer. Saved-data correction
rounds to the support point in `diagnostics_audited/`; original outputs remain.
Generators and pair/ball scores were unaffected. Final suite: **147 tests pass**.

**Decision: Outcome B, partial or ambiguous correspondence.** Recommend a
targeted Phase 3B with wider axis-crossing calibration, larger-graph distance
matrices, independently refined metric meshes, volume-gauge and smooth-field
controls. No common scaling limit, topology, unique parameter map, or 3D LQG
identification is claimed. No new major phase was launched.

Artifacts:

- [Phase 3A report](../results/RESEARCH_REPORT_PHASE_3A.md)
- [Theory/parameter protocol](../docs/COMPARISON_PROTOCOL_PHASE3A.md),
  [field/GMC methods](../docs/FIELD_GMC_PHASE3A.md),
  [LFPP methods](../docs/LFPP_PHASE3A.md), and
  [statistics/power](../docs/STATISTICS_PHASE3A.md)
- [Frozen selection](../results/phase3a/selection/selection.json),
  [confirmation comparisons](../results/phase3a/comparison/comparisons.csv),
  [leading geometry](../results/phase3a/tables/leading_geometry.csv),
  [crossing analysis](../results/phase3a/crossing_analysis/confirmation.csv)
- [Audited diagnostics](../results/phase3a/diagnostics_audited/diagnostics.json),
  [large-graph reuse](../results/phase3a/large_graph_comparison/summary.json),
  [paired improvement](../results/phase3a/paired_improvement/summary.json)
- [Final tests](../results/phase3a/final_tests.xml) and
  [source/artifact preservation manifest](../results/phase3a/final_manifest.json)

Task 3A is closed for user review. Phase 3B awaits a new instruction.

## 2026-11-11 — Task 3B resumed and completed

Recovered the interrupted state before computing: only the parameter-conversion
module/test persisted; no completed3B dataset or simulation checkpoint existed.
Recorded their hashes in the resumption audit, recovered the workstreams and
reused previous field/graph data. No completed3B experiment was restarted.
Previous generators, reports and raw datasets remain unchanged.

Completed high-resolution linked calibration with20 common-Fourier field
families,N32/48/64/96/128,11xi values and1,100 main crossings,plus206
controls/pilot records. Main runtime322.4s,peakRSS1.26GB. Two thousand paired
whole-field bootstrap draws quantify MC uncertainty; window/correction,
filter,infrared and endpoint sensitivities are separate. Finite central
conversion gives gamma(.225)=.85001,gamma(.25)=1.00771,gamma(.3)=1.49840.
Xi.325 permits inversion in only44.15% of bootstrap draws and is excluded
from primary ranking; .35/.375 have no central subcritical inversion.
The nonzero asymptotic map remains unvalidated.

Added a separately labeled positive-Ritz/rotated-sphere provisional sampler.
The pilot saved288 full paths,168 conditional special-cone bridges and
4,096 isolated angular samples. Conditional bridges do not validate the
junction/time ensemble or count as extra independent paths. I3/A2/A3
controls identify unresolved joint-minimum and local-shape error; no1% law
accuracy is claimed. Full diagnostics, including the low-lifetime A3 cohort
and its independent repeat, are retained.

Explored33 distinct SPD covariance configurations,32 successfully sampled:
initial29 over equicorrelated,birectangular,symmetric-two-correlation and
asymmetric families,three GP-guided proposals andone rho=-.8 boundary probe.
C022's numerical positivity certificate failed despite SPD covariance; no
clipping,imputation or hidden retries. C019/C023 have substantial coarse
accumulated harmonic defects and carry explicit reliability warnings.

The main search saved632 graph records from348 stage-qualified paths,
including18 cross-generator anchor groups sharing seed values,never pooled
as independent. All graph/CDF/metric checks pass. Ten intermediate models
have8 paths each at pairedn2048/8192. Four frozen holdout models each have16
newn8192 graphs; twelve independent new fields haveN32/48,98-direction
metrics andgamma uncertainty probes. All16-point matrices androot CDFs are
saved. Calibration/search/holdout identities remain separated.

Grounded robust47-feature scores for A3/C030/C029/I3 are
.1355/.1381/.1632/.1945,with broadMC95 ranges;the frozenlinkedxi values are
.225/.25/.25/.25. All improve overflat andthe testedvariance-matched smooth
control. Triangle/quartet aspects,half-radius ball dispersion andsomeglobal
fluctuations retain residuals. Unit-mass gauge andmean/median sensitivity
change scores;no unique optimum. Independent code review corrected future
selection bootstraps to usemax-gamma-probe criterion,matching ranking.
The original central-only screenbootstrap remains preserved andlabeled.

Reused all648 archived exactI3 records. At n524288,N48 groundedpairW1 is
.01590 forfrozenxi.25,.01295 forhistoricalxi.3 and.02989 forflat. Both
improvements overflat havepositive pairedMC intervals;theirrelative
ordering changes underunit-mass gauge. Frozenxi.25 impliesdimensionproxy
4.0308,below commonlyusedcorrection-conditional graphestimates4.85–4.91;
historicalxi.3 implies4.9947. This tensionremains material,withoutforcing
any fit. Budd–CastroTable2 containsno exactA3 row; neighboringrowsare not
relabeled asA3.

**Decision: partial or ambiguous correspondence.** Candidate region nearA3
andnearby asymmetricnegative covariances,linkedfinite xi.225–.25,gamma.85–1.01.
Neitheradistinguishedcovariancecurve nor many-to-onecontinuumlevelsets are
established. Recommend targetedlarger-matrix/special-cone andcalibrated
fieldresolution confirmation beforefurtherbroadsearch. Phase3C notstarted.

Completedfullsuite: **173 passingtests**. Finalaudit verifiesprevious
hashes,newcheckpoints,source snapshots,seedseparationandcachedfieldreuse.

Artifacts:

- [Phase3B report](../results/RESEARCH_REPORT_PHASE_3B.md)
- [Completion checkpoint](../results/phase3b/STATUS.md),
  [actual covariance registry](../results/phase3b/covariance_registry.json),
  [resumption audit](../results/phase3b/resumption_audit.json)
- [Calibration](../docs/PARAMETER_CALIBRATION_PHASE3B.md),
  [provisional sampler](../docs/PROVISIONAL_SAMPLER_PHASE3B.md),
  [quality audit](../results/phase3b/quality/README.md),
  [structure review](../docs/STRUCTURE_PHASE3B.md)
- [Frozen plan](../results/phase3b/final_plan.json),
  [heldout comparisons](../results/phase3b/final_analysis/summary.json),
  [largeI3 reuse](../results/phase3b/final_analysis/archived_i3.json)
- [Test XML](../results/phase3b/final_tests.xml),
  [final preservation manifest](../results/phase3b/final_manifest.json)

Task3B isclosed foruserreview; noactiveorinterruptedprimarysimulationsremain.

## 2026-11-12 — Phase 3B final consolidation after the second interruption

Recovered the completed report and sealed experiment checkpoint. All 4,126
artifacts in the original Phase 3B manifest and its recorded source hashes
matched before edits. CRC checks passed for all 1,962 NPZ archives. No corrupt
outputs or incomplete required simulation batches were found. All subagent
derivations, validation and independent raw-score reviews were already saved.

**No new simulation, parameter optimization or metric computation was run.**
Exported the existing best-tested linked xi/gamma for every attempted covariance
into a 33-row table, clarified the provisional cases, and documented the
completion inventory. Corrected only resource accounting: graph headers store
cumulative elapsed time across paired sizes, so the latest per-path header
must be used rather than selecting an arbitrary size. Raw data, scientific
scores and previous source modules are unchanged. The original report and
manifest remain preserved alongside the revised handoff.

The final report now explicitly compares Phase 3B with Phase 3A and gives three
focused Phase 3C recommendations with resource estimates. The conclusion
remains **partial or ambiguous correspondence**: A3 and nearby asymmetric
negative covariances are candidate regions near linked finite xi=.225–.25,
gamma=.85–1.01. Higher-point and ball residuals, generic-sampler bias and
finite-size dimension tension remain. Neither a distinguished covariance
curve nor a many-to-one continuum relationship has been established. The
historical I3 xi=.3/gamma≈1.4984 candidate remains relevant to larger-graph
and dimension checks.

Reran the existing full suite: **173 tests passed in 9.55 seconds**. The new
verification record preserves the earlier XML. Finalization checks distinguish
permitted report/log/README/checkpoint edits from unchanged source/raw data.

- [Final report](../results/RESEARCH_REPORT_PHASE_3B.md)
- [Completion inventory](../results/phase3b/finalization_20261112/COMPLETION_INVENTORY.md)
- [Full tested parameter table](../results/phase3b/finalization_20261112/covariance_parameter_map.csv)
- [New test XML](../results/phase3b/finalization_20261112/tests.xml)
- [Finalization manifest](../results/phase3b/finalization_20261112/final_manifest.json)

Phase 3B is complete and stops for review. No Phase 3C work was started.

## 2026-11-12 — Phase 3C broad family search and interruption recovery

The user explicitly authorized Phase 3C after the completed Phase 3B handoff.
Prepared the technical/resource plan and predeclared ensemble-mean distance
normalization as primary, median sensitivity, both fixed volume gauges,
linked gamma(xi), four feature weightings and whole-parent inference units.
Preserved all previous source/data/reports. No Phase 3D is authorized.

Stage I completed a ten-configuration sampler gate, adding only 56 missing
fine paths and retaining a degree-8 positivity failure. Existing coarse,
fine and special-cone controls were reused. The fine general pipeline's A3
control has a +4.32% mean-distance residual, with MC95 [1.06%,7.41%]; ordinal
minimum shifts remain. Small harmonic defects therefore do not validate path
or continuous joint-minimum accuracy. General samples remain provisional.
Prospective accumulated-defect warning gates mask unsafe settings from family
claims without discarding or replacing individual bad paths.

The broad design contains 109 distinct permutation-canonical SPD matrices:
20 birectangular, 15 equicorrelated, 25 two-correlation and 50 generic, with I3
shared between the first two families. All 19 graph batches completed during
the interruption: 106 successful configurations each have 12 physical paths
at paired n512/2048, giving 1,272 parents and 2,544 records. There are 1,086
new parents and 186 reused parents. C022/C058/C065 have recorded numerical
positivity-certificate failures. The completed audit identifies 22 unsafe
sampled settings; their low scores cannot establish correspondence families.

The initial field campaign is complete: 48 independent Gaussian families at
N32 across 14 xi values, and 12 paired families at N48. All 840 metric records
use 16 measure points and 26 directions; 199 are verified exact cache copies.
Five batches took 589.04 seconds, with measured peak RSS 487MB. Fourier-mode
coupling, source hashes, metric/CDF checks and archive integrity passed.

A saved-data parameter audit found that directly interpolating the inherited
gamma-envelope endpoints is not the same as interpolating each model's lambda
and then inverting. At xi=.2875 the high gamma probe undercovers the modelwise
envelope by .15073. Existing data remain unchanged and explicitly labeled;
the separate selected/confirmation producers use genuine modelwise probes.
No new crossing calibration was required.

Family tolerances were frozen from old same-model splits before Phase 3C
scores. The grounded/mean primary epsilon=.26041420 is deliberately loose
and is not an equality criterion. Signed residuals, componentwise checks,
numerical masks and independent confirmation remain essential.

On resumption, all 3,607 NPZ archives passed CRC checks; no partial archives or
incomplete primary batches were found. All 4,130 sealed Phase 3B artifacts
remain unchanged. The full recovered suite passed 193 tests in 10.99 seconds.
The interrupted full response-surface analysis had not started, so it is now
running from these saved data with 24 feature-bootstrap draws and checkpoints.
Adaptive family interpretation, selected refinement and independent holdout
remain pending; the planned extra 52 selected fields are not yet generated.

- [Status and exact checkpoint paths](../results/phase3c/STATUS.md)
- [Recovery inventory](../results/phase3c/resumption_20261112/completion_inventory.json)
- [Graph registry and audit](../results/phase3c/screen/audit/configuration_registry.json)
- [Sampler reliability](../docs/SAMPLER_RELIABILITY_PHASE3C.md)
- [Field methods](../docs/FIELDS_PHASE3C.md)
- [Family protocol](../docs/FAMILY_PROTOCOL_PHASE3C.md)
- [Technical plan](../docs/TECHNICAL_PLAN_PHASE3C.md)

Continue from these checkpoints; do not restart completed broad simulations.

## 12 November 2026 — Phase 3C consolidation checkpoint

Broad and adaptive coverage is now121distinct covariance attempts,117successful
cohorts,24unsafe sampled settings and4recorded numerical model failures.
The combinedscreen contains1,404physicalparents atpaired512/2048. All full
response surfaces,24feature-bootstrapdraws,component profiles,local-rank and
direct graph-levelset diagnostics are saved. Twelve adaptive probes expanded
neighbor coverage rather than repeatedly chasing the globalminimum.

Selected refinement iscomplete at12configurations,24parents each,paired
2048/8192. The48discoveryfieldparents were reused for57698-direction N32/N48
metric records. Finecontrols still show systematic residuals: approximateA3
mean distance is+2.11% versus24exact-gridparents,MC95[.62%,5.01%]. Generic
samples remainprovisional. Same-C broadxi support preventsidentifyingPhi(C);
extended low-J regions mustnotbe advertisedascontinuum equivalence families.

The complete independent confirmation simulations were frozen before viewing
their metric statistics. All176newgraphparents at11spread-outconfigurations
are saved,alongwith24newGaussianfamilies andallfrozenlog/flat/smooth/N64/stencil
records. A specialist hit its usage limit afterits sequentialscheduler saved
the plannedoutputs; no completedphysicalsimulation isbeingrestarted.
The coordinatingagent isfinishing saved-dataanalysis andthe principalreport.

The energy-Ustatistic's oldnaive bootstrappercentiles areupwardbiased by
repeatedresampledpanels. Oldoutputs remainunchanged;newconfirmation uses
descriptiveensemblescaledenergy andaseparatewholeparent-permutation test
ofparent-normalized matrixshape. This doesnotalterthe47featureprimaryJ.

Theuser'slatestinstruction prioritizescompletion andpreservingallgenerated
paths andGitexclusions. No100-familyprecisionextension,newbroadsearch,cleanup,
Gitexclusionchange,commit/push orPhase3D isplanned. See
[status](../results/phase3c/STATUS.md) and
[draftreport](../results/RESEARCH_REPORT_PHASE_3C.md).

## 12 November 2026 — Phase 3C completed and stopped for review

Finalized [RESEARCH_REPORT_PHASE_3C.md](../results/RESEARCH_REPORT_PHASE_3C.md)
with the complete 121-matrix registry, all failed/unsafe settings, broad and
adaptive response surfaces, local-rank/level-set diagnostics, twelve-point
refinement and independent confirmation. There are 117 successful coarse
configurations; 24 sampled settings remain unsafe for family claims. General
samplers are provisional even where accumulated harmonic defects are small.

The final audit counts 1,724 newly generated physical Brownian paths across
the gate, screen/adaptation, refinement, controls, pilots and confirmation,
excluding reused Phase 3B paths and duplicate graph sizes. Field comparison
uses 72 distinct Gaussian families:48 discovery plus24 independent holdout,
36 newly generated in Phase 3C and36 reused. All frozen N32/N48/N64,98/26
direction, flat/smooth and calibration-model tests completed. No optional
52-field extension or additional broad sweep was generated.

The strongest surviving provisional region is negative equicorrelation at
rho=−.30 and−.225, nominal xi=.225, central linked gamma=.850 (model probes
.850–.905; conditional conversion MC95 .837–.889). These representatives and
asymmetric C110=(−.53072,−.39863,.24028) pass frozen feature-family diagnostics
in both gauges. Nominal independent grounded J values are .0958,.0748,.1025.
Their broad overlapping xi support does not identify a unique scalar map.
The proposed shared-xi=.25 birectangular segment does not survive as a whole:
several representatives fail ball/mesoscopic/higher-point component checks
despite low scalar J. I3 retains corrected-dimension tension; A3's favorable
grounded profile does not imply compatibility in the other volume gauge.

Log-field geometry fits all11holdout graph settings better than flat by about
.24–.30 J units, with shared-parent conditional intervals excluding zero.
Matched smooth-field and paired stencil/N64 tests provide additional finite
evidence, not continuum certification. Pair W1 is partly forced by normalized
graph-hop discreteness:nearest-lattice costs are around .022–.025, compared
with the best observed W1 around .0285. Full signed vectors, ball distributions
and distance-matrix diagnostics are saved. Naive bootstrap intervals of the
old energy U-form are not used as null tests; parent-shape permutation results
are separate secondary sensitivities with explicit power/multiplicity limits.

A mathematical next question is compact-ensemble matching: the paper-biased
excursion law tilts the original law by I_alpha/C_alpha, preserving a.s.
exponents under equivalence but not global normalized matrix laws. Grounded,
Weyl-shifted and volume-conditioned field ensembles also differ. No tilt was
chosen to improve Phase 3C fits. No Euclidean graph embedding or common limit
has been established.

Final tests:214 passed in10.27s, no failures/errors/skips. Integrity checks:
5,717 NPZ archives pass CRC, no partial archives; graph/CDF/matrix/probe/source
checks pass. All4,128 immutable artifacts of the latest Phase3Bfinalization
and120 old source files remain unchanged. Low-mode errors are at roundoff.
The final manifest preserves source versions and artifact hashes. A late
dictionary-key error affected only diagnostic output; cached primary features
and bootstrap draws were reused, with no physical resimulation or retuning.

[Final status](../results/phase3c/STATUS.md),
[all actual covariance matrices](../results/phase3c/tables_final/all_covariances.json),
[independent confirmation](../results/phase3c/confirmation/analysis/COMPLETE.json),
[paired checks](../results/phase3c/confirmation/paired_checks/summary.json),
[integrity audit](../results/phase3c/final_integrity.json),
[final preservation manifest](../results/phase3c/final_manifest.json).

Assessment: extended finite-resolution compatibility regions, but no convincing
identified continuum correspondence family. Prioritize compact-ensemble
definitions and discriminating exact/special-cone scaling/ball/matrix tests
before another wide search. All result paths and Git exclusions were preserved;
no datasets were cleaned, moved or reorganized and nothing was committed/pushed.
Phase3C is complete. No Phase3D work has begun; stop for user review.
