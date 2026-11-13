# Phase 3B: adaptive covariance search and linked LFPP/GMC comparison

Experiments completed 11 November 2026; final handoff checked 12 November 2026.
New results, configurations,
source versions, figures and checkpoints are under [phase3b/](phase3b/).

## Executive summary and research decision

**Partial or ambiguous correspondence.** The experiment now covers **33
distinct SPD covariance configurations attempted, 32 successfully sampled**,
rather than the three special models of Phase 3A. A coarse-to-fine search and
new held-out realizations identify a candidate region near the A3 Weyl cone
and two nearby asymmetric, negatively correlated matrices. The associated
finite-resolution linked parameters are approximately
`xi=.225–.25, gamma=.85–1.01`. These are candidate regions, not a discovered
continuum correspondence or an identified covariance curve.

All four frozen candidates, including the independent I3 baseline, improve
substantially on the flat control and the particular smooth Gaussian control
tested. Nevertheless, triangle shapes, quartet aspects, ball fluctuations and
some global fluctuations retain systematic-looking residuals at the measured
sizes. Graph quantization and unbounded path/minimum errors are material.
Different volume gauges and graph sizes change the relative merits of the
candidates. No unique optimum, distinguished one-parameter covariance
subfamily, or many-to-one continuum map is established.

The historical I3 candidate near `xi=.3,gamma=1.5` remains useful: it is more
compatible with commonly used corrected graph-dimension estimates and the
large archived I3 pair/ball laws than the new small-graph feature optimum is
in some comparisons. It was tested separately, without retuning the frozen
held-out candidates. Phase 3C should resolve this finite-size/parameter tension
through targeted special-cone and larger-matrix comparisons. A larger generic
Cartesian sweep is not justified.

Relative to Phase 3A, Phase 3B **strengthens the evidence that the selected
log-field models fit better than the tested flat and smooth controls**, but
**complicates the proposed parameter correspondence**. A3 and nearby asymmetric
covariances compete with I3, and the best finite-feature parameters are not
automatically compatible with the independently estimated metric dimension.
Additional geometric parameters remain a possibility, not an established
consequence of the observed higher-point residuals.

## 1. Recovery, scope and preservation

At resumption, the interrupted 3B work had left only
`src/parameter_conversion_3b.py` and its test file. No Phase 3B dataset, report
or completed simulation checkpoint existed. The two files were hashed before
continuing; the prior subagent attempts had ended at the usage limit. See
[resumption audit](phase3b/resumption_audit.json). No completed 3B experiment
was restarted. Existing Phase 3A field families, the 648 archived exact I3
graphs, and prior analytic-cone references were reused where their recorded
observables sufficed.

Work was divided among a coordinating implementation/report stream, a
parameter-calibration and mathematical-audit stream, a provisional-sampler
and validation stream, and a statistical/structure review stream. All their
saved contributions were integrated. Previous generators, reports and raw
datasets remain unchanged; only new modules were added and the user-facing
README, research log and questions were appended. The final
[manifest](phase3b/final_manifest.json) verifies previous source/data hashes,
new checkpoint integrity, source snapshots and stage counts. There are no
unfinished simulations or incomplete primary ensembles at this handoff.

This phase implements the revised permission to use **explicitly provisional**
general-covariance samples for exploratory selection. It does not retroactively
validate them, and does not replace the Phase 2C stopping assessment.

The recovery above describes the first interruption. At the final resumption
on 12 November, the full report, all primary batches and the original final
manifest already existed. All **4,126 listed files and all recorded source
hashes matched** before any handoff edits. ZIP CRC checks passed for **1,962
NPZ archives**. No incomplete or corrupt primary simulation output was found;
the latest resumption generated **no new fields, paths, graphs or optimizer
evaluations**. It adds only saved-data exports, essential verification and
report clarification. The
[completion inventory](phase3b/finalization_20261112/COMPLETION_INVENTORY.md)
distinguishes validated components, provisional results, unresolved targets
and optional omitted experiments. The original manifest and pre-consolidation
report are preserved; the
[finalization manifest](phase3b/finalization_20261112/final_manifest.json)
seals the revised handoff documents.

## 2. Mathematical comparison and linked parameters

### Distinct parameters and ensembles

The Brownian covariance is

\[
C(a,b,c)=\begin{pmatrix}1&a&b\\a&1&c\\b&c&1\end{pmatrix}>0.
\]

Write `C=RR^T`, let `W={z:Rz>0}`, and let `m` be the principal Dirichlet
spherical eigenfunction on `W intersect S^2`. If its eigenvalue is `lambda_1`,
then `alpha_1=sqrt(lambda_1+1/4)` and `p=alpha_1-1/2`. The regular harmonic
is `u(z)=|z|^p m(z/|z|)`; the inward harmonic is
`h(z)=|z|^(-p-1)m(z/|z|)`. The spherical interior angles, as an unordered
triple, are `acos(-a),acos(-b),acos(-c)`. Their stored vertex order is fixed by
the existing covariance module. The susceptibility parameter `gamma_s=1-alpha_1`
is distinct from the GMC parameter `gamma`.

All principal graph comparisons target Budd–Castro's **biased** excursion law:
draw a unit-radius direction with density `m^2`, generate two independent
inward Doob legs, reverse the first, concatenate and rescale to unit duration.
With the true harmonic, each leg duration is `InverseGamma(alpha_1,1/2)`,
independent of the initial direction. This is not the original unbiased
fixed-duration cone bridge. The distinction is retained even for matrix-bridge
controls. The source is Budd–Castro, arXiv:2207.05355v1, §§3.3–3.4 and §4.1,
[local PDF](../papers/Budd_Castro.pdf).

The field remains the validated mean-zero, unit-log polyharmonic torus field
from Phase 3A, with Fourier covariance `1/(4*pi*|k|^3)`, zero mode removed and
sharp spherical cutoff `|k|<N/3`. Its GMC is
`exp(gamma*h - gamma^2*Var(h)/2) dx`, with subcritical range
`|gamma|<sqrt(6)`. Both raw total mass `M` and probability measure `mu/M`
are saved. The same field generates LFPP weights `exp(xi*h)` and GMC roots.

For physical nearest-neighbor **cube site-sum** crossings, the conventions are

\[
\operatorname{med}D_N\approx N^{-\lambda(\xi)},\quad
Q(\xi)=\frac{1-\lambda(\xi)}{\xi},\quad
\gamma=\frac{6}{Q+\sqrt{Q^2-6}},\quad d_\gamma=\gamma/\xi.
\]

The last two identities use the subcritical branch `Q>sqrt(6)` and the
hypotheses of the high-dimensional metric results. See Contreras Hip–Zhuang,
arXiv:2504.09141v1, (1.1)–(1.3), Definition 1.7 and Proposition 1.8, and
arXiv:2512.06292v1, (1.5), Corollary 1.9; the
[parameter document](../docs/PARAMETER_CALIBRATION_PHASE3B.md) records the
source checks. Replacing the unknown limiting exponent by a small-resolution
slope gives a **finite proxy**, not a theorem about these numerical ensembles.
At `xi=0` the separate flat endpoint is `gamma=0,d=3`; no finite Q is invented.

Primary search varies `xi`, using this calibrated gamma and its model probes.
It does not independently optimize gamma and xi or identify either with a
Brownian correlation. No asymptotic dimension is inferred from the two graph
sizes used in the screen.

### Two fixed volume gauges

The primary grounded ensemble keeps the field zero mode zero and samples
points from `mu/M`. The separately specified unit-mass Weyl gauge rescales
distances by `M^(-xi/gamma)` while retaining the same probability measure.
This corresponds to shifting `h` by `-log(M)/gamma`; it is not conditioning
on `M=1`, nor normalization by each realization's distance median. Both gauges
are reported throughout. A global Weyl rescaling cannot repair a disagreement
in limiting Hausdorff dimension.

## 3. Crossing-exponent calibration

Twenty independent common-Fourier field families were generated at
`N=32,48,64,96,128`, with xi values
`0,.1,.15,.2,.225,.25,.275,.3,.325,.35,.375`. This gives **1,100 main
crossings**, plus 200 paired filter/infrared/orientation controls and six pilot
records. Shared resolutions and xi are not additional independent fields.
The primary fit uses log ensemble medians over all five sizes; 2,000
whole-field paired bootstrap draws propagate sampling uncertainty.

| xi | lambda proxy | gamma proxy | Gamma MC95 | Valid-model/window point envelope | Dimension proxy |
|---:|---:|---:|---|---|---:|
| .15 | .05588 | .49619 | [.48917,.50413] | [.48098,.49794] | 3.3079 |
| .20 | .09283 | .71827 | [.70839,.73158] | [.71827,.76411] | 3.5913 |
| .225 | .11027 | .85001 | [.83689,.88929] | [.85001,.90544] | 3.7778 |
| .25 | .12977 | 1.00771 | [.99296,1.07561] | [1.00771,1.11605] | 4.0308 |
| .275 | .15262 | 1.21191 | [1.19328,1.31586] | [1.21191,1.51861] | 4.4069 |
| .30 | .17460 | 1.49840 | [1.44956,1.69732] | [1.49840,1.74131], plus invalid model | 4.9947 |
| .325 | .20039 | 2.22965 | [1.87258,2.36890], conditional | Only central model valid | 6.8604 |
| .35 | .22799 | unavailable | No valid inversion | No valid model | unavailable |
| .375 | .25237 | unavailable | No valid inversion | No valid model | unavailable |

At xi=.325 only **44.15%** of bootstrap draws allow a real subcritical
inversion. It was retained as a stress diagnostic but excluded from primary
ranking. The missing branch at .35/.375 does not locate a continuum critical
parameter. At .3 a fixed `N^(-1/2)` correction to log distance leaves the
subcritical branch, whereas a `1/N` correction gives gamma1.741. All-size,
drop32 and largest-three pure-power estimates give gamma1.498,1.529,1.601.
The envelope only covers models admitting inversion; it is neither a full
systematic bound nor an additional confidence interval. The implementation
flags invalid models rather than silently dropping this qualification.

Heat versus sharp filtering changes N128 crossing levels by +2.22% at xi.2
and +4.11% at xi.3, with slope shifts -.0104 and -.0180. Removing the lowest
nonzero modes changes levels by +11.4%/+16.9%, while the measured slope shifts
are smaller and uncertain. Reinterpreting the same N nodes as a closed cube
with spacing `1/(N-1)` shifts lambda by about +.0151/+.0137. These effects
are comparable to sampling errors. The periodic Fourier field, finite open
cube and sharp cutoff also differ from the source's whole-space/mollifier
conventions. The calibration is useful for **linked exploratory candidates**,
not a validated asymptotic gamma(xi).

![Calibration with branch qualifications](phase3b/calibration/annotated_figures/calibration.png)

Raw and derived tables:
[crossings](phase3b/calibration/main/),
[conversion CSV](phase3b/calibration/analysis/conversion.csv),
[all fit models](phase3b/calibration/analysis/models.csv),
[paired controls](phase3b/calibration/controls_summary/controls.json).

## 4. Generic sampler: implemented approximation and validation status

The new [provisional module](../src/provisional_excursions_3b.py) uses the
positive corner-factor Ritz formula from Phase 2C, with an exact-rational
Bernstein positivity certificate for the stored polynomial. It avoids the
previous impractical rejection envelope using:

1. Continuous uniform spherical-triangle proposals and finite importance
   resampling with weights `m_tilde^2` for the initial direction.
2. Independently Haar-rotated Gauss sphere quadrature in each interior ball,
   with positive exit weights proportional to the numerical inward harmonic.
3. The Phase 2C Brownian sub-ball clock and stopping skeleton, retaining all
   stopping vertices before computing polygonal joint minima.

Finite importance resampling is not exact rejection. Local normalization with
a nonharmonic function is not an exact Doob transform. Random rotation removes
a fixed directional support but does not remove random-normalizer bias. The
code saves signed and absolute accumulated log mean-value defects, angular
importance ESS, maximum retained clock intervals, lifetimes and quadrature
comparisons. It never clips a failed polynomial, retries until convenient
results appear, or applies unjustified independent minimum corrections to
correlated coordinates. A straight vertex tail and unobserved bridges inside
sub-balls remain approximations. Full derivations and limitations are in
[the sampler document](../docs/PROVISIONAL_SAMPLER_PHASE3B.md).

The targeted pilot completed 288 full excursions, 168 conditional special-cone
bridge comparisons and 4,096 isolated angular samples. Conditional bridges
share saved junction/time variables and cannot validate their marginal law or
be counted as extra independent excursions. At n256, fine independent cohorts
have mean-distance differences of approximately -.32% (I3), -1.11% (A2),
and +.52% (A3), with intervals several percent wide. Joint-minimum CDF
differences of order .06–.165 remain inadequately resolved. One A3 cohort
had a low lifetime mean; a single independent repeat and conditional bridge
audit explain substantial cohort imbalance without proving full-law accuracy.
Fine conditional I3 graph distance was +2.80% with MC95 [-.26%,+5.85%].
This prevents a blanket 1% sampler-accuracy claim.

The n2048 anchor check compares six provisional paths/model to 48 independently
seeded Phase 3A references, using eight roots and 8x8 matrices on both sides:

| Model | Relative mean-distance difference, provisional minus reference | MC95 |
|---|---:|---|
| I3 | +4.44% | [-2.42%,+11.87%] |
| A2 times R+ | -1.75% | [-4.04%,+1.20%] |
| A3 | +.38% | [-1.71%,+1.72%] |

Mean agreement is insufficient: A2's small/large triangle ratio differs by
+.0312, pointwise MC95 [.0058,.0562], and its half-radius ball SD by -.00446,
MC95 [-.00765,-.00145]. These selected intervals are not simultaneous tests
over all diagnostics. They preserve evidence of unresolved local law error.

### Error budget and reliability across the actual screen

| Issue | Measured information | What remains uncontrolled |
|---|---|---|
| Independent-realization MC | Whole graph/field bootstrap; conditional root noise retained | Small ensembles, especially six-path screening |
| Angular positivity | Stored polynomial certificate; no clipping | Positivity does not establish eigenfunction accuracy |
| Nonharmonic transition normalization | Coarse C019/C023 largest signed leg sums .152/.194; absolute sums .376/.356 | No uniform likelihood/path-law bound |
| Selected finer angular kernels | Worst intermediate absolute leg sum .0221; holdout maximum .0001002 | Small observed defects are not a proof of exact h or path law |
| Angular finite importance resampling | K512/2048 moment audit within 2.11 MC SE; leading moment-bias diagnostic <=1.28e-4 | No total-variation bound |
| Temporal/spatial path resolution | Maximum retained clock interval/cell duration 3.12 screen,3.06 intermediate,.605 holdout | Unseen Brownian segments and joint minima |
| Analytic matrix references | Exact finite-time grid values; increasing oversampling | Continuous correlated minima still polygonal |
| LFPP mesh/field regularization | Paired 26/98 stencil, N32/48, filters and IR controls | Continuum path/field limit and topology |
| Parameter conversion | 2,000 paired bootstrap draws and fit windows/models | Unknown asymptotic lambda and branch ambiguity |

C019/C023 are particularly unreliable for close-fit interpretation. **C022
generated no graph**: its SPD matrix passed covariance checks, but the
numerical polynomial positivity certificate failed. This is neither covariance
inadmissibility nor proof that the true mode changes sign.

The authoritative [quality audit](phase3b/quality/audited/) contains 94
configuration/stage/size accuracy rows, a classification for every attempted
matrix and all signed local/matrix residuals. Every saved graph passes
connectedness, Hamiltonian cycle, coordinate triangulation edge counts, CDF
mass and matrix checks. The metric is the unweighted underlying simple graph;
parallel-edge labels remain stored, and parallel unit edges do not alter
shortest-path distances.

## 5. Actual covariance design and adaptive workflow

All triples are canonically sorted to quotient coordinate permutations. In
three dimensions every permutation of the three correlations arises from a
coordinate relabeling. A3 is stored as `(-.5,-.5,0)`, coordinate-permutation
equivalent to `(-.5,0,-.5)` in the original request. The special reference
sampler's original coordinate order is recorded in metadata. This is harmless
for the union graph and its intrinsic observables; unpermuted coordinate-path
statistics must not be compared as if the labels agreed.

The initial 29 configurations contain six equicorrelated points, six nonzero
birectangular points (with I3 the shared zero point), four symmetric
two-correlation points, the unequal control `(-.2,.1,.35)` and 12 constrained
Latin-hypercube/maximin asymmetric points. The screen enforces minimum
eigenvalue at least .15 and condition number at most12; it avoids the
near-singular equicorrelation boundary. This is a design on the permutation
quotient, not uniform sampling of covariance space.

Three constrained Gaussian-process lower-confidence-bound proposals were
then measured. The GP's fixed .04 score-noise floor and predictive SD guide
selection only; they do not model all shared-field dependence or sampler bias
and have no inferential confidence interpretation. A separate negative
birectangular endpoint extension to rho=-.8 was motivated by the coarse
family's best tested point at its rho=-.7 boundary. These additions produce
33 attempts, 32 with saved graphs.

| Stage | Physical graph paths | Coupled graph sizes | Field comparison | Purpose |
|---|---:|---|---|---|
| Initial screen | 28x6=168 | 512,2048 | 12 reused fields,N32,26 directions | Broad screening; C022 failed |
| Three adaptive coarse points | 3x6=18 | 512,2048 | Same cached fields | Test GP proposals |
| Intermediate | 9x8=72 | 2048,8192 | Same12 fields,N48,26 directions | Finer angular/path settings and extra xi |
| Birectangular extension | 1x8=8 | 2048,8192 | Same intermediate fields | Test rho=-.8 |
| Frozen held-out finalists | 4x16=64 | 8192 | 12 new fields,N32/48,98 directions | Independent confirmation |
| Provisional anchor control | 3x6=18 | 512,2048 | Graph-only independent reference comparison | Diagnose approximate law |

There are **632 graph records from 348 stage-qualified physical paths**, not
632 independent samples. Eighteen exact/provisional anchor groups deliberately
reuse RNG seed values across different generators; these paths are not pooled
as independent and the quality audit identifies them explicitly. All main
graph-stage seed sets and the held-out field/source identities are disjoint.

Coarse numerical paths use Ritz degree4, exit order8, K512, relative sub-ball
step .2 and base spatial step .01. Intermediate uses degree6/order12/K1024,
relative .1/base .005; holdout uses degree6/order12/K2048, relative .05/base
.0025. The base spatial step is further scaled by
`sqrt(2.5/(alpha_1-1))` to roughly equalize expected normalized clock resolution;
this heuristic is recorded, not asserted to control error. The cutoff is .001.
Reference matrix oversampling is q64 in the screen, q64 at the largest
intermediate graph size, and q128 in the holdout. Thus at n2048 the coupled
intermediate A3 path has **effective q256**, versus screen q64; their difference
mixes grid refinement and independent-cohort variation. I3 minima remain exact.

The complete actual triples, full matrices, cone angles, spectral estimates,
condition numbers, reliability and per-stage counts are available in the
[JSON registry](phase3b/covariance_registry.json),
[CSV registry](phase3b/covariance_registry.csv), and
[readable 33-row table](phase3b/covariance_registry.md).
These include equicorrelations -.3,-.15,0,.15,.3,.45;
birectangular rho=-.8,-.7,-.5,-.25,.25,.5,.7; symmetric
two-correlation rho=-.5,-.3,.2,.4; all tested asymmetric matrices and the failure.

### Actual covariance-to-parameter discovery table

The table below shows each matrix and its best **tested**, linked parameter
at its finest available discovery stage, before the independent holdout.
`screen` means n2048/N32 with six graph parents; `intermediate` means
n8192/N48 with eight graph parents. Both use twelve field parents and the
grounded robust J criterion. These stages have different fidelity and samples;
their scores are not interchangeable accuracy estimates. C019/C023 retain
their large-harmonic-defect warnings, all non-reference graph laws remain
provisional, and every entry is subject to selection among the tested xi.
Only the frozen finalists have the independent confirmation intervals in §7.
The full covariance is obtained from the displayed triple using C(a,b,c) in
§2; full-precision matrices are saved in the registry and
[machine-readable map](phase3b/finalization_20261112/covariance_parameter_map.json).

| ID | Canonical (a,b,c) | Discovery stage | Best tested xi | Linked gamma | Robust J |
|---|---|---|---:|---:|---:|
| C000 | (-0.3, -0.3, -0.3) | intermediate | 0.25 | 1.00771 | 0.1611 |
| C001 | (-0.15, -0.15, -0.15) | screen | 0.25 | 1.00771 | 0.1498 |
| C002 | (0, 0, 0) | intermediate | 0.25 | 1.00771 | 0.1459 |
| C003 | (0.15, 0.15, 0.15) | screen | 0.3 | 1.49840 | 0.2547 |
| C004 | (0.3, 0.3, 0.3) | screen | 0.3 | 1.49840 | 0.2657 |
| C005 | (0.45, 0.45, 0.45) | screen | 0.25 | 1.00771 | 0.3543 |
| C006 | (-0.7, 0, 0) | intermediate | 0.25 | 1.00771 | 0.1640 |
| C007 | (-0.5, 0, 0) | screen | 0.225 | 0.85001 | 0.1935 |
| C008 | (-0.25, 0, 0) | screen | 0.225 | 0.85001 | 0.1911 |
| C009 | (0, 0, 0.25) | screen | 0.3 | 1.49840 | 0.3290 |
| C010 | (0, 0, 0.5) | screen | 0.3 | 1.49840 | 0.2569 |
| C011 | (0, 0, 0.7) | screen | 0.25 | 1.00771 | 0.2562 |
| C012 | (-0.5, -0.5, 0) | intermediate | 0.225 | 0.85001 | 0.1365 |
| C013 | (-0.3, -0.3, 0) | intermediate | 0.275 | 1.21191 | 0.1806 |
| C014 | (0, 0.2, 0.2) | screen | 0.25 | 1.00771 | 0.1994 |
| C015 | (0, 0.4, 0.4) | screen | 0.25 | 1.00771 | 0.2865 |
| C016 | (-0.2, 0.1, 0.35) | screen | 0.25 | 1.00771 | 0.1893 |
| C017 | (-0.54618, -0.4553, 0.64423) | screen | 0.25 | 1.00771 | 0.2108 |
| C018 | (-0.48957, -0.0049562, 0.63912) | screen | 0.25 | 1.00771 | 0.3151 |
| C019 | (-0.52775, 0.29975, 0.35604) | screen | 0.25 | 1.00771 | 0.1707 |
| C020 | (-0.52475, -0.30232, 0.31179) | screen | 0.25 | 1.00771 | 0.2033 |
| C021 | (-0.28435, -0.25217, 0.53882) | screen | 0.25 | 1.00771 | 0.2461 |
| C022 | (0.097354, 0.61509, 0.63176) | model_certificate_failure | — | — | — |
| C023 | (-0.20924, 0.29008, 0.62334) | screen | 0.3 | 1.49840 | 0.2722 |
| C024 | (-0.33822, -0.10388, 0.24128) | screen | 0.25 | 1.00771 | 0.1967 |
| C025 | (0.2307, 0.3326, 0.55493) | screen | 0.25 | 1.00771 | 0.3360 |
| C026 | (-0.23535, 0.027241, 0.59158) | screen | 0.25 | 1.00771 | 0.2580 |
| C027 | (-0.54926, -0.18949, -0.1613) | screen | 0.25 | 1.00771 | 0.2004 |
| C028 | (-0.53549, -0.21041, 0.53225) | intermediate | 0.275 | 1.21191 | 0.1785 |
| C029 | (-0.52584, -0.50199, -0.22268) | intermediate | 0.25 | 1.00771 | 0.1438 |
| C030 | (-0.49698, -0.48862, -0.08034) | intermediate | 0.25 | 1.00771 | 0.1422 |
| C031 | (-0.39908, -0.37315, -0.20202) | intermediate | 0.25 | 1.00771 | 0.1491 |
| C032 | (-0.8, 0, 0) | intermediate | 0.25 | 1.00771 | 0.1671 |

Within the initial same-fidelity screen, negative equicorrelations and the
negative birectangular direction tend to fit better than their positive
counterparts. Positive equicorrelations .15/.3 favor xi=.3, while .45 returns
to .25 with a substantially worse score: this is not a monotone calibrated
covariance-to-gamma law. Refinement moves C013 and C028 from xi=.225/.25 to
.275, whereas A3 remains at .225 and the I3 baseline at .25. Such shifts,
including those across asymmetric matrices, are evidence of unresolved
finite-size/fidelity dependence rather than established scalar level sets.

![Covariance coverage and candidate scores](phase3b/final_analysis/covariance_candidate_map.png)

### Search decisions before holdout

The primary criterion uses the grounded gauge and takes the largest discrepancy
over the central/low/high **valid model gamma probes** at each xi, then selects
xi. This sensitivity criterion is called the robust score below. It is not
a worst-case bound over unmeasured errors. At some xi the low probe coincides
with the central value; this causes no new independent realization.

Initial eligible central xi values were .15,.2,.225,.25,.275,.3;
xi.325 was a non-primary stress point and xi0 the flat control. Intermediate
also measures .2125,.2375 by interpolation of the calibrated lambda table.
No extrapolation or separate gamma optimization is used. The coarse screen
favors A3 at xi.225 (robust score .1131), followed by the symmetric
`(-.3,-.3,0)` (.1364), equicorrelation -.3 (.1438), and a mixed-sign
asymmetric matrix C028 (.1445). I3's best coarse robust score is .1565 at .25.

After the ten-configuration intermediate comparison, the frozen finalists are:

| ID | Canonical (a,b,c) | Condition | alpha_1 | Frozen xi | Central gamma | Intermediate robust score |
|---|---|---:|---:|---:|---:|---:|
| C012, A3 | (-.5,-.5,0) | 5.83 | 6.5, analytic | .225 | .85001 | .13649 |
| C030 | (-.496980,-.488615,-.080340) | 6.33 | 6.92887, numerical | .25 | 1.00771 | .14219 |
| C029 | (-.525839,-.501995,-.222681) | 10.60 | 9.15522, numerical | .25 | 1.00771 | .14382 |
| C002, I3 baseline | (0,0,0) | 1.00 | 3.5, analytic | .25 | 1.00771 | .14586 |

A2 C007 has alpha_1=4.5. The new birectangular rho=-.8 point has analytic
alpha_1=6.38203 and remained a useful restricted-family comparison, rather
than displacing the finalists. The [final plan](phase3b/final_plan.json)
predates the held-out samples and appends I3 as a reference if absent from
the top three. No held-out retuning is incorporated into its primary scores.

Independent code review found that the original **screen selection bootstrap**
had used central scores while candidate ranking used max-over-probe scores.
The old files are preserved and explicitly labeled central-score sensitivity;
they are not robust-selection probabilities. Intermediate and final bootstrap
code was corrected to use the same robust criterion as ranking. Both source
versions are preserved. This did not change raw simulations or point rankings.

## 6. Intrinsic statistics and inference units

Every new comparison uses 16 iid measure points, 16 root-to-all CDFs and a
16x16 distance matrix per realization. There are 120 dependent matrix pairs,
560 dependent triples and 1,820 dependent quartets, **not** extra independent
replicates. Old eight-point records are truncated/matched when used for
validation; their point-count-sensitive quantiles are not mixed into the
principal 16-point objective.

The 47 features cover pair quantiles/mean, centered root-ball mass quantiles
and SD at normalized radii .35,.5,.75,1,1.5, triangle ratios, quartet shapes,
shared/disjoint pair correlations, global mean-distance CV, finite-radius
growth and atom/eligibility diagnostics. Quartet coordinates use ordered
opposite-edge sums `s1<=s2<=s3`: `(s3-s2)/s3` and `(s2-s1)/s3`. The first
is normalized four-point delta and vanishes for a tree metric. Ball means
are not counted again as independent of the pair CDF. Missing estimates carry
validity masks rather than being treated as measured zeros.

Within each family, discrepancies are RMS differences divided by declared
natural scales; family weights are pair1, balls1.5, triangles1, quartets1.5,
matrix-dependence1, global-scale.5, mesoscopic1, atoms.25. The weighted family
average is **J**, a descriptive objective, not a likelihood or chi-squared
test. Equal-family, higher-point-emphasis and no-mesoscopic sensitivities are
reported. The structure study's fixed-noise-unit scores and the old Phase 3A
objective use different units and must not be numerically compared with J.

The deterministic normalizer is the ensemble median measure-pair distance,
recomputed in every bootstrap. Ensemble-mean normalization is a separate
sensitivity. Whole graphs and whole fields are resampled independently;
field indices remain shared across xi, gamma probes and controls. The final
intervals use 64 draws, intentionally sufficient for coarse uncertainty only.
They exclude systematic sampler, field, finite-size and calibration errors.
They are not multiplicity-adjusted simultaneous intervals. Conditional argmin
frequencies are finite-ensemble diagnostics, not posterior probabilities.

## 7. Independent held-out results

Each of four graph models has 16 new n8192 paths. Twelve new Gaussian field
families provide coupled N32/48 grids and 98-direction torus metrics at
xi0,.225,.25. Their central/low/high gamma probes share field and LFPP
distance data. The primary N48 results below retain the frozen xi for each
graph. Source seeds and exact matrices are in per-checkpoint headers.

| Model | Frozen xi,gamma | Grounded J, central | Grounded robust J, MC95 | Flat J | Flat minus robust J, MC95 | Unit-mass robust J, MC95 |
|---|---|---:|---|---:|---|---|
| A3 C012 | .225,.85001 | .1337 | .1355 [.0829,.1779] | .4076 | .2721 [.2245,.2868] | .1531 [.1025,.1869] |
| C030 | .25,1.00771 | .1360 | .1381 [.1135,.1698] | .3912 | .2531 [.1740,.2674] | .1262 [.0935,.1672] |
| C029 | .25,1.00771 | .1621 | .1632 [.1442,.2225] | .3783 | .2151 [.1230,.2517] | .1444 [.1278,.1959] |
| I3 C002 | .25,1.00771 | .1830 | .1945 [.1349,.2560] | .5216 | .3270 [.2605,.3500] | .2435 [.1700,.2855] |

The candidate region survives independent seeds and a finer LFPP stencil,
with A3/C030 having similar primary discrepancy and broad overlapping
uncertainty. C029 worsens somewhat. The unit-mass gauge favors C030 over A3
at the frozen parameters; the gauge was not selected after this comparison.
The bootstrap intervals cannot establish equality of models, and overlapping
marginal intervals alone are not a pairwise ranking test.

Random disjoint same-model half splits give grounded J medians
.133/.138/.066/.150 for A3/C030/C029/I3 graphs (8 versus8) and approximately
.075–.095 for fields (6 versus6). These smaller-ensemble floors help assess
the magnitude of .002–.03 score gaps, but are not 16-versus12 acceptance
bands or formal goodness-of-fit thresholds. See
[same-model splits](phase3b/final_analysis/same_model_half_split.json).
Reoptimizing only as a separately labeled conditional diagnostic on the
holdout often selects either A3 or C030 at .225 or .25; it does not define
a new confirmed optimum.

![Held-out discrepancy and MC intervals](phase3b/final_analysis/heldout_scores.png)
![Normalized pair CDFs](phase3b/final_analysis/pair_cdfs.png)
![Root-ball mass dispersion](phase3b/final_analysis/ball_mass_dispersion.png)
![Quartet shape laws](phase3b/final_analysis/quartet_shapes.png)

### What agrees and what remains different

The grounded signed residuals below are graph minus field at central gamma:

| Model | Small/large triangle ratio difference, MC95 | Quartet delta difference, MC95 | Half-radius ball SD: graph/field |
|---|---|---|---|
| A3 | -.0176 [-.0367,.0035] | -.00716 [-.01296,-.00208] | .02157/.01170 |
| C030 | -.0290 [-.0428,-.0119] | +.00101 [-.00435,.00507] | .01983/.01186 |
| C029 | -.0415 [-.0613,-.0254] | -.00296 [-.00823,.00020] | .01781/.01186 |
| I3 | -.0287 [-.0494,-.0103] | -.01399 [-.01892,-.00977] | .02428/.01186 |

C030's delta agreement does not mean the full four-point law agrees: its
quartet aspect differs by +.0182, MC95 [.0025,.0274]. Its grounded global
root-mean CV is .0272 versus field .0544; I3 is .0541 versus .0544. These
CVs contain finite-root noise; raw variance-subtracted diagnostics are saved
without clipping negative estimates into a spurious absence of randomness.
All four half-radius ball SD residual intervals exclude zero under MC-only
resampling. Thus pair-law resemblance and delta agreement leave other
observables capable of distinguishing the finite ensembles.

Mean normalization reduces grounded robust J to approximately
.0822/.1072/.1487/.1309 (A3/C030/C029/I3). Integer graph medians near 11
hops and maximum CDF atoms around .12–.14 cause material normalized-radius
effects; the continuous-CDF KS atom floor alone is about .061–.069. This is
a known finite-graph obstruction to demanding tiny KS discrepancy. The new
score is not dominated solely by a KS statistic, but ball/growth features
remain quantization-sensitive. Weight variants change magnitudes and some
rankings; the tested negative-correlation region remains interesting without
being uniquely identified.

All point scores, signed feature differences, masks, gamma probes and
bootstrap intervals are in
[summary](phase3b/final_analysis/summary.json),
[candidate CSV](phase3b/final_analysis/heldout_candidates.csv), and
[weight/gauge/normalization sensitivity](phase3b/final_analysis/sensitivity.json).

## 8. Controls, refinement and historical I3 candidate

### Matched LFPP path discretization

Four identical held-out field/root families at N48 compare 26 to98 directions.
Mean distances with26 directions are +5.79% at xi0, +3.26% at .225 and
+3.10% at .25. Ensemble-scale-normalized feature shifts are much smaller:
J=.00453,.00475,.00536. Thus the finer stencil substantially changes raw
distance scale while leaving these measured shape summaries relatively
stable. Neither stencil is exactly Euclidean at xi0; finite directional
anisotropy is retained. The former six-neighbor L1 metric is not used as an
unqualified isotropic continuum surrogate. Results are in
[stencil controls](phase3b/final_analysis/stencil_controls.json).

### Smooth-field specificity control

Eight of the same held-out N32 fields are projected onto `|k|<=2`, then
rescaled to the unchanged deterministic point variance. Gamma/xi and root
uniforms stay fixed. This deliberately smooth exponential measure is not
called a log-field GMC approximation. Grounded smooth-minus-log J is:

| Model | Smooth minus log J | Paired MC95 |
|---|---:|---|
| A3 | +.1815 | [.0707,.2721] |
| C030 | +.2605 | [.0399,.3546] |
| C029 | +.2324 | [.0265,.3076] |
| I3 | +.1889 | [.0438,.2424] |

Unit-mass differences remain positive in these intervals. Improvement over
both flat and this smooth control is useful specificity evidence, rather
than merely matching a mean distance. It does not establish that logarithmic
covariance is uniquely required, because one smooth control is not the space
of competing random geometries. See
[smooth controls](phase3b/final_analysis/smooth_controls.json).

### Coupled resolutions and graph-size sensitivity

At the frozen parameters and fixed holdout N48 fields, intermediate graph
cohorts change central J from n2048 to8192 as follows: A3 .1700→.1276,
C030 .1814→.1595, C029 .1896→.1629, I3 .2118→.1365. The paired graph sizes
share a physical path. The independent n8192 held-out cohort then gives
.1337,.1360,.1621,.1830; settings and seeds both change for provisional
graphs. I3's exact generator also has appreciable cohort variation, so these
shifts must not all be attributed to sampler bias.

For the final graphs, coupled field N32→48 central J changes
A3 .1561→.1337, C030 .1336→.1360, C029 .1584→.1621, I3 .1917→.1830.
The data do not show monotone convergence of every statistic. These are
finite-size sensitivity diagnostics, not continuum extrapolations. The
complete [table](phase3b/final_analysis/finite_size.json) retains both gauges.

### Larger archived I3 graphs and the earlier xi=.3 candidate

The 648 exact archived I3 graph records are reused without regeneration.
At n524288 there are64 independent graphs. Their missing higher-point
matrices are not invented; this check is restricted to pair/ball/global
summaries. Twelve held-out fields additionally evaluate the historical
xi=.3 with the current linked gamma=1.49840.

| Grounded N48 model | Pair W1 at graph n524288 | Ball-dispersion L1 | SD of half-radius ball mass |
|---|---:|---:|---:|
| Frozen .25,1.00771 | .01590 | .00987 | .01186 |
| Historical .3,1.49840 | .01295 | .00592 | .02422 |
| Flat | .02989 | .05489 | approximately0 |
| Exact large I3 graph | — | — | .01849 |

Both nonzero candidates improve the pair W1 over flat: paired MC95
improvement [.00827,.01671] for .25 and [.00449,.01924] for .3. These are
not intervals for the difference between the two candidates. The historical
candidate has the better point pair/ball discrepancy in the grounded gauge;
unit-mass pair W1 is .01947 for .25 and .02370 for .3, so the ordering changes.
At n8192 its richer grounded J=.1986 is close to the frozen .25 central
J=.1830, with better quartet but poorer mesoscopic/global components.
There is no single robust winner across observables, sizes and gauges.

The theoretical tension is consequential: .25 implies dimension proxy4.0308,
MC95[3.9718,4.3024], with core valid-model point envelope[4.0308,4.4642].
Published I3 is4.83±.06; our prior additive-offset fit is4.910[4.643,5.159]
and the Eq45 adaptation4.856[4.650,5.123], conditional on their correction
models. Historical .3 gives4.9947[4.8319,5.6577], compatible with those
central graph fits. Broader graph corrections can give lower, very uncertain
values (e.g.4.093[3.416,5.128]); this is diagnostic tension, not a continuum
disproof or a license to choose correction models to force agreement.

For published correlated controls, Budd–Castro Table2 has exact A2
`(alpha,beta)=(pi/3,pi/2)` at4.67±.04. It has **no exact A3 row**
`(pi/2,pi/3)`; nearby beta=.95/1.11 give4.42±.06/4.52±.09 and must not be
misreported as an A3 measurement. No new precise correlated Hausdorff exponent
is claimed here.

![Large-graph I3 reuse and parameter sensitivity](phase3b/final_analysis/archived_i3_finite_size.png)

Detailed [archive comparison](phase3b/final_analysis/archived_i3.json) and
[historical 47-feature diagnostic](phase3b/final_analysis/historical_candidate.json)
preserve these separately from the frozen selection.

## 9. Covariance structure, identifiability and interpretation

The screen genuinely probes different covariance directions. Its first three
natural-scale feature PCs explain approximately62.4%,25.9%,4.11% of initial
between-configuration variation; conditional whole-path noise ratios are
2.05,3.05,1.86. Several linear PCs do **not** refute a curved one-dimensional
family; unknown covariance-dependent sampling errors are also excluded from
that noise comparison. Conversely a dominant PC does not establish scalar
level sets. No intrinsic continuum rank has been estimated.

Two selected examples explain why dimension alone is inadequate:

- Equicorrelation -.15 (C001) and mixed-sign C028
  `(-.535493,-.210409,.532253)` have canonical separation .786 but a
  fixed-noise-unit rich-feature discrepancy .356, comparable with same-unit
  conditional reference medians .50–.75. These are possible finite
  near-degeneracies, selected among many comparisons, not equal laws.
- A3 and C026 have two-size growth-proxy difference .0114, MC95
  [-.0056,.0298], yet quartet-delta mean difference .02024, SE .00244.
  The growth proxy is `log(mean_distance_2048/mean_distance_512)/log(4)`,
  not a Hausdorff estimate. Similar finite growth can hide different
  higher-point geometry, including possible sampler artifacts.

The [structure report](../docs/STRUCTURE_PHASE3B.md) and
[datasets](phase3b/structure/) save all pair contrasts and weight analyses,
not only selected examples. Its first analysis covers the initial28 successful
points; the adaptive addendum covers31. The subsequently added C032 is
included in this report's final registry/map and ten-point intermediate
ranking. These counts describe different checkpoints, not missing data.

The evidence presently supports **neither** a distinguished matching covariance
curve nor established many-to-one covariance-to-gamma level sets. A3 and C030
provide a reproducible candidate region; some far-apart covariances are
indistinguishable at screen precision; other similar-growth configurations
differ in shape. Sampling bias, finite graph hops, and finite torus geometry
prevent assigning these patterns a continuum interpretation. A candidate
match must eventually satisfy the implicit dimension relation and multiple
joint laws, rather than choosing whichever observable gives the closest fit.

Intrinsic pair/ball/matrix laws require no Euclidean graph embedding. They
compare probability metric spaces meaningfully, while leaving their topology
and relation to Euclidean field coordinates unresolved. No graph GFF was
introduced as a presumed 3D log-correlated field. Matching Hausdorff dimension
or these finite statistics would not identify a field law, common scaling
limit, manifold topology or three-dimensional LQG.

## 10. Cost, reproducibility, checks and Phase 3C handoff

Substantial computations were preceded by persisted budgets and pilots:
[screen budget](phase3b/screening_budget.json),
[intermediate plan](phase3b/intermediate_plan.json),
[holdout plan](phase3b/final_plan.json), and
[control plan](phase3b/holdout_control_plan.json). Each covariance/path or
field/xi/grid result is checkpointed independently. No single main run was
expanded into an unbenchmarked multi-hour sweep.

The main calibration took322.4s and measured peak RSS1.26GB. Main graph
producer totals, deduplicating paired sizes, are133.5s initial screen,33.5s
adaptive coarse,117.2s intermediate,15.8s boundary extension and132.1s
holdout. Field metrics took65.3s initial pilot/screen,295.7s intermediate
and403.4s holdout; matched controls took23.0s stencil,36.5s smooth and98.4s
historical candidate. These sums exclude setup/analysis and are not elapsed
wall time when stages run concurrently. Holdout memory was budgeted around
1.4GB; unlike the calibration RSS this is an estimate. Artifacts occupied
about0.84GB before the final source snapshot/manifest. The
[audited graph timings](phase3b/finalization_20261112/producer_timings_audited.json)
use the largest cumulative checkpoint time for each physical path. The older
[resource summary](phase3b/resource_summary.json) selected a size-dependent
header and differs slightly; this accounting correction does not alter samples
or scientific scores. Both versions are retained.

Saved configurations record PCG64 seeds, covariance matrices, numerical
settings, source hashes, software versions and input field hashes. Calibration
uses seeds202611113510–202611113529. Main graph-stage base seeds are
202611113621/202611113752/202611113751/202611113753/202611113851 for screen/adaptive/intermediate/extension/
holdout; held-out fields use base202611113861. Full derived seeds reside in
headers. Discovery uses12 cached Phase3A field families; holdout uses12 new
families. Parameter probes, resolutions and controls reuse those families.
Only the first two complete root-distance arrays per field stage are cached
to limit storage; all root CDFs, mean distances and sampled matrices are saved.
Other root distances reproduce from the complete grid tuple, source and seed.

The completed full suite has **173 passing tests**. The final audit checks
all primary counts, graph invariants, CDF/metric consistency, NPZ CRCs,
JSON records, disjoint main-stage seeds, shared cached-field identities and
unchanged prior source/artifact hashes. Source snapshots preserve both the
screen bootstrap version and its corrected successor. Primary reproducible
entry points, with `--help`, are:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3b_crossings --help
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3b_calibrate --help
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3b_graphs --help
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3b_fields --help
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3b_search --help
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_results --output results/reproduction/phase3b_final --bootstrap 64
```

Use new output directories for analysis; completed outputs refuse replacement.
Simulation producers skip complete checkpoints and check configuration;
the field producer also checks its calibration hash. Source hashes are saved,
but not every resume entry point enforces source equality automatically: use
the exact recorded producer snapshots when resuming. Registry and final manifest exporters use the
fixed completed phase directory and should not replace their saved outputs.
The [test XML](phase3b/final_tests.xml),
[completion checkpoint](phase3b/STATUS.md) and
[original preservation manifest](phase3b/final_manifest.json) anchor the
experiment checkpoint. The final resumed suite again passed **173 tests**;
its [new XML](phase3b/finalization_20261112/tests.xml) preserves the earlier
test record. The finalization manifest verifies unchanged raw data and source
files while recording the permitted report/log/README/checkpoint revisions.

Recommended **Phase 3C**, requiring a new instruction, consists of three
targeted experiments. These are resource estimates from present timings,
not new benchmarks, promised run times or calculations already performed.

| Priority experiment | Question and evidence it would provide | Initial resource scope |
|---|---|---|
| Larger exact/special-cone graph matrices, paired with the existing field candidates | Retain I3 at linked xi=.25 and historical .3, and A3 near .225–.25. Increase graph size and matrix point count together with a minimum-resolution control. This tests whether triangle/quartet/ball residuals and dimension compatibility improve together, without generic-sampler ambiguity. | Start with 8–16 parents at n32768,16–32 roots/points,then consider n65536. A3 n8192/q128 presently costs about2 seconds/path; linear grid-cost scaling suggests roughly10–30 minutes for a modest two-size/two-q batch, with per-path checkpoints. Memory must be piloted before retaining larger matrix-bridge paths. |
| Focused coupled field and crossing refinement | Hold xi near .225,.25,.3 and both volume gauges fixed. N64/96 metric statistics and wider crossing windows test whether the apparent correspondence and gamma conversion drift consistently. They separate finite-resolution compatibility from an asymptotic identification. | N48/98-direction metrics cost about9 seconds per field/xi. N64 extrapolation is roughly25 seconds and N96 roughly80–100 seconds per field/xi. Pilot N64 first; a modest N96 ensemble can take tens of minutes. Dominant grid storage grows with N^3: scaling the present conservative1.4GB budget gives about3.3GB atN64 and11GB atN96, subject to measured implementation peaks. Do not commit a large batch before that pilot. |
| Paired near-A3 sampler sensitivity | Refine C030 with matched A3/I3 anchors and a short predeclared list of signed higher-point/minimum observables. This tests whether the inferred covariance response exceeds numerical path-law bias, and whether C030 can be interpreted as a genuine nearby geometry rather than an optimizer artifact. | Begin at n2048/8192 with roughly8 parents/settings. Current fine paths cost a few seconds each, so an initial comparison should take minutes; higher angular degree must be benchmarked separately. Couple only refinements for which a valid coupling is available; thinning one polygonal path does not bound the missing Brownian bridges. |

Use ensemble mean/median and both gauges as prespecified sensitivities, and
increase independent parents rather than counting more matrix subsets as
replication. Extending the crossing maximum also requires care: this project's
Fourier sampler couples resolutions through the complete requested grid tuple.
Changing its maximum with the same seed does not preserve the earlier fields.
Use explicit low-mode extension or a verified mode-indexed coupling before
claiming a paired extension beyond the present N128 calibration. No broad
generic covariance sweep or forced gamma conversion is recommended.

The next informative question is whether the A3/near-A3 region and the I3
historical parameter survive **simultaneous** larger-size matrix, ball and
dimension checks. This phase stops here for review; no Phase 3C experiment
has been launched.
