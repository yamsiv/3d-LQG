# Phase 3C technical plan and initial budget

Prepared before new simulations,12 November2026. The complete operational
requirements are in [REQUEST_PHASE3C.md](REQUEST_PHASE3C.md).

## Scientific question and fixed comparison conventions

Does the measured low-discrepancy set in (canonicalC,xi) extend over a
nontrivial xi interval,with neighboring simulated covariances and coherent
signed higher-point/ball residuals? A single argmin is not the objective.
Use all47Phase3Bfeatures,paired graph-size observations and whole-realization
sampling units. Main scale is the **ensemble mean pair distance**,declared
before Phase3C scores to reduce the previous median-hop instability; median
is a sensitivity. Grounded gauge remains primary and unit-mass Weyl gauge
is a fixed parallel ensemble. Retain four previously declared feature weights.

Gamma(xi) reuses the five-resolution/20-field Phase3Bcalibration; interpolate
finite lambda only within valid intervals, retain MC and model-window probes,
and flag branch failure. No gamma/xi Cartesian optimization is claimed as LQG.
No new graph GFF, Euclidean graph embedding or topology assumption is introduced.

## Stages and budgets

These are initial estimates from recorded3Bcosts. Each producer must record
its own pilot and batch-specific plan before launching a significant batch.
Counts may decrease if observed errors exceed the intended geometry signal.

| Stage | Question | Initial scope and reuse | Estimated cost and working memory |
|---|---|---|---|
| Recovery | What can be reused unchanged? | Verify sealed3Boutputs/source hashes;read prior exact/control/pilot records | Seconds,hashing~.84GB;no simulations |
| I: reliability gate | Are path/minimum and harmonic errors smaller than the screening response? | 10representativeCs,two levels;reuse oldcoarse/fine/special controls. Approximately56missing fine paths,plus<=30s pilot;coupled thinning on the same retained path | Roughly2–4minutes initially;~250MB per sampler process. Complete this gate before >100Cs are actually attempted |
| II: broad graph design | Which covariance regions exhibit coherent metric variation? |109initialCs INCLUDINGold33:20birectangular,15equicorrelated,25two-correlation,50generic (I3 shared byfirsttwo). Target12parents/C,n512/2048,16points;reuse31existing six-path coarse cohorts,augment withnewparents. Known failedpolynomials remainrecorded | About1,100new paths x.8–2s =15–37CPUminutes in batches of~6–10Cs;several hundredMB/process. Never run a monolithic uncheckpointed campaign |
| III: field screening | Is the linked LFPP response continuous and distinguishable acrossxi? |48physicalfields:24cached3Aconfirmation+12cached3Bholdout+12new. Xi=0,.1,.125,.15,.175,.2,.2125,.225,.2375,.25,.2625,.275,.2875,.3,N32/26directions,16points,central/low/highgamma. Reusecompatible16pointmetrics. InitiallyN48on12families,notall48 | PriorN32~.7s/xi;48x14 roughly8minutes beforecache reuse. N48subset12x14~8minutes. Chunkby<=12fields;~0.7–1.4GB per process |
| IV–V: family screen and adaptation | Curves,levelsets,multiplebranches or unstable coincidences? |Predeclareepsilon from oldsame-modeldisjointsplits;computeallCs versus sharedfieldfeatures once. Observedadjacency/localrank anddirectgraph-levelsetchecks. Reserve~12newCs spread acrossxi/branches/uncertainty | Saved-data scoring initially1–3minutes;~2–5minutes newadaptivepaths. GP/acquisition uncertainty isheuristic,notMC/systematicerror |
| VI: selected-family refinement | Do several representatives survive size/solver changes? |Choose~8–12Cs spreadthroughcandidatexi/components,notonlybest3;24graphparents/C,n2048/8192,finerD6path. FieldN48atselectedxis,expand toward72discoveryfamilies if worthwhile | Graphs~10–20CPUminutes,chunkedby3–4Cs. Selectedfieldwork~15–25minutes,chunkedby6–12fields;refinebudgetaftermeasuredpilot |
| Independent confirmation | Does a frozen family prediction reproduce? |Freezeallconfigs/xi/statistics;newgraph/fieldseeds. Initially24fieldfamilies atselectedxi,N32/48,98directions as cost allows;flat andsmoothcontrols. Noholdoutretuning | Potentially20–35CPUminutes fieldmetrics plusgraphs;chunk6fields(~5–8minutes). Benchmarkmemory before any N64/96 extension |
| Handoff | What is evidenceversusconjecture? |ReportallCs/reliability,observedcomponents,modelgaps,dimensiontension;source/seedmanifest,tests,CRC checks andstatus |No new scientificsimulationafterthe decision/report |

Total is a staged research budget,not approval to spend time on every optional
refinement. Early measurements determine where work is informative. The gate,
graph screen and field screen have distinct owners; limit simultaneous
memory-heavy field jobs. Any individual invocation approaching15minutes must
be split or given an explicit reason and updated budget.

## Checkpoints, independence and decisions

Results are organized under results/phase3c/{calibration,covariance_design,
screen,fields,adaptive,families,confirmation}. Olddata are read-only references.
Every physical graph seed can supply several n values;that does not increase
parents. Every field is one cluster acrossxi,gamma,gauges,stencils andgrid sizes.
Source hashes and originalroot streams determine whether cached metric reuse
is valid;an8point or98direction record is not silently relabeled as16point/26.

Stage I supplies per-region/per-observable reliability labels. A low J does
not override a large harmonic defect or unstable minima. Observed numerical
shifts,graphMC,fieldMC,calibration andfinite-size effects remain separate.
Thresholds are calibrated from oldcontrol splits before inspecting3Cscores;
smaller-cohort splitnoise is explicitly conservative andnot a formal equality
test. Family continuity requires neighboring actual samples;surrogate paths
are proposals only. Initial low-J membership does not establish a limit law.

New seeds: gatebase202611123001,discoveryfieldsbase202611123100,
screen graphbase202611123200,adaptivebase202611123500,
familyrefinementbase202611123700. Reserve202611123900+ for independent
confirmation. Derived PCG64 seeds,not just these bases,are recorded perpath.

No Phase3D work is authorized by this plan.
