# Phase 3C task specification (12 November 2026)

This is an operational summary of the user's complete Phase 3C instructions
in the conversation, including the continuation supplied after the first
message was truncated. The conversation is authoritative.

Search for **families**, not the global best point, in the permutation quotient
of SPD unit-diagonal C(a,b,c). Investigate a matching curve, many-to-one scalar
level sets, disconnected branches, isolated finite-size coincidences or no
identifiable relationship. Do not concentrate the majority of effort on I3,
A3 or C030. Those are controls. Preserve all previous source/data/reports.

Mathematics: paper-biased versus original excursions remain distinct. Covariance
parameters/cone alpha1/susceptibility are distinct from LFPP xi and GMC gamma.
Only use gamma(xi) from Q=(1-lambda_cross)/xi and subcritical
gamma=6/(Q+sqrt(Q^2-6)); dproxy=gamma/xi. These use a limiting exponent in theory;
numerical finite-window conversions are provisional. Invalid branches are
flagged, not forced. Gamma/xi may be independently varied only as diagnostic
controls. Read relevant instructions/context/log, Phases2B/2C/3A/3B reports and
the five local primary PDFs before using their constructions.

Stage I: before expanding beyond100 covariance configurations, assess8–12
representatives at two angular/path/clock resolutions. Reuse existing data and
exactI3/A2/A3 references. Include analytic birectangular, generic asymmetric
and moderately difficult cones. Measure marginal/joint minima, pair andball
laws, triangles/quartets/root dependence, coupled refinement, accumulated
harmonic defects, angularESS/failures. Classify reliability by region and
observable empirically. Do not restart exhaustive sampler development; exclude
regions whose error dominates the geometric response from family claims.

Stage II: target100–150 total distinct covariances. Suggested coverage is
20birectangular,15equicorrelated,25two-correlation,40–60generic asymmetric,
then10–30adaptive additions. UseSPD/condition/minimum-eigenvalue safeguards,
permutation canonicalization and structured/maximin designs far from winners.
Main screen n512/2048,12–24independent paths where feasible; pair graph sizes
on a physical path. Refine promising **families**, not only three points,
at n2048/8192 and larger sizes only when informative.

Stage III: build on3B crossing calibration, initially xi~.15–.30 at
.01–.025spacing, extending only when theory and evidence permit. Carry model
gamma probes, branchflags andMC uncertainty. Target40–60screening Gaussian
families, moving toward100selected where feasible, with independent holdout.
Reuse fields/metrics acrossCs andgammaprobes. UseN32/48broadly andricher/larger
discretizations selectively. Changing maximumFFTgrid withsame seed does not
preserve low modes: verify coupling. Keepcube/torus andgrounded/unitmass
Weylgauges separate andprespecified.

Stages IV–V: compare full normalizedpairlaws/Wasserstein, rootballquantiles
andfluctuations, mesoscopicgrowth, triangles/quartets/higherpointmatrices,
shared/disjointdistance dependence, globalfluctuations, two-sizegrowth and
dimensioncompatibility. Wholephysicalrealizations arebootstrapunits. Calibrate
J(C,xi) tolerance fromsamemodelsplits/numericalerrors BEFOREseeingan attractive
curve; retain multipleprespecifiedweights andsignedresidualvectors. Different
MC/systematicerrors neednotfitoneCI. Traceobserved low-J sets/components and
xi extent; do not identify interpolation as evidenceofcontinuity. Localfeature
rank mustaccountfornoise, discretization andnonlinearcurvature. Directlycompare
graphgeometries onproposedxi levelsets beforecallingthemequivalent. Adaptive
acquisition shouldcoverbranches/uncertainty/disagreeingobservables,notjustargmin.

Stage VI: select5–10representatives spreadalongpossiblefamilies, increase
parents/graphsizes/pathprecision/fieldcounts andmetricstencils asbenchmarks
allow,retainI3/A3/analyticanchors,andtest whether signedobservables vary
coherently. Freezehypotheses,configs,xi,statistics before independent graph
andfieldholdout; includeflat and/orsmoothnegativegeometry. Separateclaims of
better-than-controlfit, internalcoherence, expectedmetricdimension anddecreasing
refinementresiduals. Preservecontradictoryholdoutdata withoutretuning.

Resource policy: planbeforeeachsignificantbatch,with scientificquestion,
runtime/memoryestimate,reusedinputs,seeds/settings/sourceversions andexact
checkpointpaths. Noenormousuncheckpointedrun. Pilotfirst,checkpointbyC/field,
preservecompletedoutputs,andprioritizebreadthoverprecisionifneeded. Substantial
taskmayspanmultipleCodexsessions. Do notstopafteronlyplanning.

Required outputs: results/RESEARCH_REPORT_PHASE_3C.md in10parts (mathematics,
methods,samplerreliability,broadregistry,LFPPsearch,candidatefamilies [central],
metricmeasurecomparisons,independentconfirmation,interpretation,nextquestions),
statusdocument,rawresults,figures,tables,sourcehashes,reproducibilitychecks and
updatedRESEARCH_LOG.md. ReporteveryattemptedC,includingfailures. Runrelevant
tests/checkpointchecks. Honestwidesttrustworthyrestrictedfamilyresults are
acceptable ifgenericbiasprecludesidentification. **StopafterPhase3C;noPhase3D.**
