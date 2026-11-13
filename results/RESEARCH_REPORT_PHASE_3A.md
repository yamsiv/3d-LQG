# Phase 3A: intrinsic comparisons of three-tree graphs and 3D LFPP/GMC

Research date: 11 November 2026. **Outcome B: partial or ambiguous
correspondence.** The implemented field/GMC/LFPP pipeline produces meaningful
finite-ensemble comparisons. I3 has interesting pair/ball compatibility near
xi=.3,gamma=1.5, including the existing largest graphs, but higher-point
statistics, global-scale conventions and finite-resolution drift prevent a
model identification. A2/A3 also show partial resemblance, without an
established parameter match. This is neither a positive identification nor
a robust continuum mismatch.

This phase implements and executes a Euclidean field-first pipeline. It does
not repair the general Brownian excursion sampler, assign a graph GFF, or
assert a common continuum limit. Earlier code, datasets and reports remain
preserved. Four workstreams covered coordination/comparison, field/GMC,
LFPP numerics, and independent mathematical/statistical review.

## 1. Mathematical definitions and parameter conventions

All five requested original sources were consulted in their local versions:
Budd–Castro [2207.05355v1](https://arxiv.org/abs/2207.05355v1),
Contreras Hip–Zhuang [2504.09141v1](https://arxiv.org/abs/2504.09141v1),
Contreras Hip–Zhuang [2512.06292v1](https://arxiv.org/abs/2512.06292v1),
Ding–Gwynne–Zhuang [2310.03996v5](https://arxiv.org/abs/2310.03996v5),
and Dello Schiavo–Herry–Kopfer–Sturm
[2302.02963v1](https://arxiv.org/abs/2302.02963v1).
Equation/theorem references, hypotheses and source hashes are recorded in
[the comparison protocol](../docs/COMPARISON_PROTOCOL_PHASE3A.md) and
[source index](phase3a/theory/source_index.json).

| Parameter | Definition used here |
| --- | --- |
| `(a,b,c)` | Brownian correlations `(C12,C13,C23)`; not field couplings |
| `alpha1`, `gamma_s` | Cone spectral index `sqrt(lambda1+1/4)` and susceptibility `1-alpha1` |
| `gamma` | GMC exponential parameter for a unit-log Gaussian field |
| `xi` | LFPP path-length exponential parameter, distinct from gamma |
| `lambda_cross` | Crossing exponent with physical distances proportional to `epsilon^lambda_cross` |
| `Q(xi)` | LFPP background-charge exponent, satisfying `lambda_cross=1-xi*Q` in the relevant framework |
| `d_gamma` | Subcritical weak-metric Hausdorff dimension, implicitly parameterized through the LFPP exponent |

CHZ's lattice path cost already includes the physical factor epsilon:
`epsilon*sum exp(xi*h_epsilon(v))`, including endpoint vertices. DGZ's
`lambda_n` denotes a median normalization sequence, not `lambda_cross`.
Regressing physical crossing distances against N=1/epsilon therefore gives
slope **minus** lambda_cross. A fit to an unscaled vertex sum would have a
different exponent. CHZ equations (1.1)–(1.3), DGZ equations (1.7)–(1.8).

In the subcritical weak-metric parameterization,

```
Q(xi)=3/gamma+gamma/2,       xi=gamma/d_gamma.
```

The relations are implicit, rather than a prescription xi=gamma. They use
the limiting exponent and the appropriate subcritical metric framework;
they do not automatically hold for a finite torus regression. Weak-metric
existence/tightness and dimension results do not establish uniqueness of the
3D limiting metric or any identification with three-tree graphs.

The theory bounds safely place xi=.1/.2 within a sufficient subcritical
range. Xi=.3 is not settled by those bounds. Under the same limiting-exponent
conventions, the CHZ lower bound rules out xi=.4/.6 from the subcritical
gamma parameterization; these remain legitimate **finite exploratory weighted
metrics**, not purported subcritical LQG metrics. The protocol provides the
inequalities and thresholds explicitly.

Our primary metric-measure object is a finite unit-mass random probability
space. Graphs use normalized vertex counting measure. LFPP uses the same h
as its GMC probability measure, while retaining the separate unnormalized
total mass. Both use the **deterministic ensemble pair median** as distance
scale. Per-realization median normalization is a separate diagnostic because
it removes global random scale variation. Torus pair laws and open-cube
crossings are distinct experiments.

## 2. Log-correlated field construction and validation

On the unit three-torus with Fourier basis exp(2*pi*i*k.x), the source
polyharmonic normalization has `a3=1/(2*pi²)`. Consequently

```
Covariance = 2*pi²*(-Delta)^(-3/2),
E|h_hat(k)|² = 1/(4*pi*|k|³), k != 0.
```

The shell integral is dr/r, giving logarithmic coefficient one. The zero
mode is removed explicitly. A real Gaussian FFT on the largest requested
grid supplies common Hermitian Fourier randomness at every resolution.
NumPy's inverse-FFT factor is compensated by multiplying coefficients by
N³. A conjugate mode pair has independent real/imaginary variance S(k)/2;
there is no additional factor of two.

The default sharp spherical cutoff is `0<|k|<N/3`, avoiding Nyquist modes.
The point variance used for GMC is the deterministic spectral sum, not a
sample's spatial variance. A heat-filter alternative and a fixed-bandlimit
resampling function separate field regularization from mesh quadrature.
This is a cutoff of the **continuum torus spectrum**, not an inverse
nearest-neighbor lattice Laplacian.

Validation used 256 independent common-field families at N16/24/32/48,
after a pilot. It checked realness, Gaussian mode moments/independence,
exact common low modes, covariance, variance growth, directional symmetry,
regularization sensitivity and fixed-field resampling. Exact finite-spectrum
variance versus log N has slope **1.00426**. Declared finite separation-window
covariance slopes range **.957–1.023**, with cutoff and infrared corrections.
These are checked against the exact spectral covariance, rather than forced
to be one over every coarse window.

The 256-field validation took 12.68 seconds after an eight-field .43-second
pilot. Its traced pilot peak was about 17.6 MB. Scientific metric ensembles
also preserve full field arrays and seed/configuration information. Common
randomness is guaranteed within the requested sizes tuple; changing its
largest size with the same seed is not a cross-call coupling contract.

Implementation: [log_field_3d.py](../src/log_field_3d.py).
Details: [field/GMC methods](../docs/FIELD_GMC_PHASE3A.md).
Data: [field validation](phase3a/field_validation/summary.json).

## 3. GMC implementation and results

For the identical field used by LFPP, cell masses and probability weights are

```
m_i=N^-3*exp(gamma*h_i-gamma²*V_N/2),
M_N=sum_i m_i,              p_i=m_i/M_N.
```

Log-sum-exp keeps both quantities stable. E[M_N]=1 exactly at finite cutoff;
the expectation was validated before normalization. At gamma=0 the masses
are uniform. No samplewise variance is substituted for V_N.

In unit-log dimension three, the subcritical GMC range is |gamma|<sqrt(6).
Positive continuum q moments require `q<6/gamma²`; the Euclidean set-mass
exponent is `zeta(q)=3q-gamma²*q*(q-1)/2`. A partition sum instead has exponent
zeta(q)-3. These are Euclidean multifractal quantities, not LFPP metric-ball
dimensions. At gamma=2 a continuum second moment does not exist; finite-grid
second moments nevertheless remain finite and can be computed exactly.

At N48, the independent 256-field sample gives:

| gamma | Mean unnormalized mass | Empirical SE |
| ---: | ---: | ---: |
| .5 | .99929 | .00253 |
| 1 | .99405 | .01099 |
| 1.5 | .97114 | .02859 |
| 2 | .89003 | .05678 |

For a stationary finite grid, Gaussian integration gives independently
`E[M_N²]=mean_j exp(gamma²*C_N(j/N))`. At gamma=2,N48 this is about 12.428,
so the **exact finite-grid SE** of the 256-sample mean is **.2113**, much
larger than the empirical .0568. The observed .890 mean therefore does not
establish failed normalization. Rare large masses are under-sampled. At
gamma=1 the exact SE is .01176; at gamma=1.5 it is .03717.

The paired coarse 4³ probability-mass L1 change from cutoff N32 to48 is
**.0583±.00067 SE** at gamma=1 and **.2906±.00681** at gamma=2. For the
same fixed N24 bandlimit evaluated on N48, the gamma=2 total-mass quadrature
ratio changes on average only .225%±.433% SE, with SD3.47%. Cutoff and
quadrature are therefore empirically different error sources. Intermittency,
coarse mass distributions and admissible moment-scaling diagnostics are
saved; short Euclidean scaling windows are not treated as asymptotic proofs.

Probability normalization preserves a finite-grid collision probability
`sum p_i²`. All sampled zero distances and repeated matrix points are retained.
At high gamma this atom is a material discretization diagnostic, not a
statistical nuisance to remove.

![GMC mass uncertainty with exact finite-grid SE](phase3a/field_validation/mass_uncertainty_audit_plotted.png)

Implementation: [gmc_3d.py](../src/gmc_3d.py). The exact-moment audit and
fixed-field controls are linked in [the methods](../docs/FIELD_GMC_PHASE3A.md).

## 4. LFPP implementation, discretization and crossing conventions

The proper finite edge metric uses positive path-integrated weights without
GMC variance subtraction. The axis default is
`cost(i,j)=(exp(xi*h_i)+exp(xi*h_j))/(2N)`. An auxiliary source and target
half-site costs reproduce the CHZ vertex-sum crossing convention exactly on
this open grid. At xi=0, the proper face distance is (N−1)/N, while the
vertex-sum face cost is exactly one. A periodic graph is explicitly rejected
by the cube-crossing helper.

Continuum-oriented comparisons use 26 and 98 (`primitive2`) directions,
integrating along every long edge rather than skipping its interior field.
The default interpolates exp(xi*h); an independent sensitivity integrates
exp(xi*interpolated h). Composite path quadrature is refined separately.
Neither a fixed six- nor a fixed rich-direction stencil becomes exactly
isotropic merely by increasing N.

At N32, errors relative to exact Euclidean torus distances are:

| Directions | Mean relative excess | Maximum relative excess |
| ---: | ---: | ---: |
| 6 | 54.49% | 73.21% |
| 26 | 8.08% | 12.80% |
| 98 | 2.04% | 4.93% |

The six-direction metric is precisely L1 at xi=0. The others are explicitly
polyhedral approximations. Exact flat-torus Euclidean distances are also
computed analytically as a separate control; no path solver is needed there.

Four independent log fields at N16/24 and xi=.2/.4/.6 give 96 paired stencil
rows. Raw98-direction distances are about 1.9–3.2% below 26-direction values;
quadrature refinement changes averages by .06–.16%. At N24, switching the
interpolation target changes mean distances by .38%,1.45%,3.35% for the
three xi values, with smaller mean-normalized distribution-shape changes.
These are measured discretization effects, not zero-error declarations.

Measured build-plus-40-root search times for the 26-direction implementation
were .079,.260,.650,2.643 seconds at N16/24/32/48. The N48 CSR and distance
array occupied about 35 MB each; process peak after several builds was about
335 MB. The full confirmation's repeated98-direction builds cost more than
the 26-direction pilot; actual totals will be reported with the final inventory.

![Directional and paired LFPP controls](phase3a/lfpp_validation/paired_discretization_controls.png)

Implementation and detailed checks: [LFPP methods](../docs/LFPP_PHASE3A.md).
No finite-grid crossing slope is asserted to be a limiting exponent.

## 5. Graph benchmarks and their reliability

All graphs target the paper-biased finite-volume excursion law, whose bias
differs from the original excursion at finite volume. The comparison is not
silently switched to the original law. The underlying simple graph with unit
edges supplies distances; normalized vertex counting measure supplies mass.

| Model | Covariance entries (a,b,c) | Sampler qualification |
| --- | --- | --- |
| G0 / I3 | (0,0,0) | Validated exact excursion and continuous cell minima |
| G1 / A2×coordinate | (−.5,0,0) | Exact matrix-bridge finite-time laws; polygonal joint minima |
| G2 / A3 | (−.5,0,−.5) | Exact original/biased matrix-grid construction; polygonal joint minima |

For each model,48 realizations at n2048 and 8192 have eight iid counting-measure
points, root-to-all distance/ball laws and their full8×8 distance matrix.
The first24 are discovery graphs and the next24 independent confirmation
graphs. I3 regenerates archived seeds to add the missing matrices, checking
the old root histograms exactly; these are not counted as newly independent
graphs. A2/A3 add192 new special correlated graphs in total.

A2/A3 use32 grid segments per graph cell. For the first four paths per model
and size, a64-segment source is thinned to32 on the **same** realization;
distance and local-edge changes are retained separately. The previous Phase2C
controls remain available; no independent coordinate-minimum correction is
applied to correlated paths. The general angular/MCMC samplers are not used.

The existing648 unique I3 graph records up to524,288 vertices are reused for
large-size pair/ball comparisons and correction-sensitive dimension evidence.
No new large graph dimension campaign is needed. The prior offset estimate
4.910 [4.643,5.159] MC95 remains ansatz-dependent; other correction choices
gave central estimates roughly4.09–5.40. Two new sizes for A2/A3 are not
enough to assert asymptotic Hausdorff dimensions.

## 6. Quantitative metric-measure comparisons

The primary normalization is the median of the equal-realization-weight
root-to-all pair mixture. It is one deterministic ensemble estimate, applied
to every realization in that ensemble. Each realization has probability mass
one. Fields with larger unnormalized GMC mass are not given extra mixture
weight. The same type of normalization is used for graphs and LFPP/GMC.

Eight iid measure points supply their full distance matrix. Weighted
root-to-all CDFs use every target, preserving rootwise probability balls;
the same points are matrix roots. LFPP CDFs retain the zero atom and 1,024
positive distance bins; graph CDFs use exact integer support. The distance
displacement from binning is bounded by one bin, rather than an assumed
uniform bound on CDF error. Matrices retain coincident measure samples.

The descriptive joint objective is

```
J = W1(pair laws)
  + mean over declared radii |SD_root(ball mass)_graph - SD_root(ball mass)_field|
  + |mean(smallest/largest triangle distance)_graph - corresponding field mean|
  + |mean(middle/largest triangle distance)_graph - corresponding field mean|.
```

Radii are .25,.5,.75,1,1.5,2 in ensemble-median units. Mean ball mass already
equals the cumulative pair law, so its dispersion provides the extra
observable. Triangle means first average valid triangles within a field,
then weight eligible fields equally; all-zero triangles are reported
separately. The plotted valid-triangle CDF has its conditioning stated in
the statistics document. Pair W1 integrates through the known finite support,
with tail mass beyond normalized radius4 recorded separately.

Whole-realization bootstrap intervals recompute the ensemble normalizer and
retain all roots together. They cover finite-ensemble Monte Carlo variability,
not field, path, topology or continuum-extrapolation systematics. Percentile
intervals for a nonnegative discrepancy are not null-hypothesis tests; a
positive lower endpoint is not a rejection of equal laws. Split-ensemble
discrepancies and signed-feature differences provide additional context.

### Independent confirmation

The following uses24 independent confirmation graphs per model at n8192
and 24 independent confirmation fields at N48 with 98-direction paths. All
intervals are whole-realization MC95 percentile intervals (160 draws).

| Graph | Frozen (xi,gamma) | Pair W1, MC95 | Joint J, MC95 | Flat-control J |
| --- | --- | --- | --- | ---: |
| I3 | (.3,1.5) | .0353 [.0287,.0543] | .0855 [.0562,.1511] | .1287 |
| A2 | (.2,0) | .0328 [.0279,.0611] | .0910 [.0680,.1268] | .1430 |
| A3 | (.2,0) | .0334 [.0263,.0445] | .0762 [.0583,.1097] | .1221 |

The flat-control pair W1 values themselves are .0357,.0323,.0289. Thus
the selected models do **not** substantially improve these small-graph pair
laws alone. Their main advantage is nonzero root-ball heterogeneity. A
separate paired holdout bootstrap gives flat-minus-selected J differences
.0432 [−.0163,.0793], .0520 [.0368,.0680], .0459 [.0316,.0603]. The I3
joint improvement is not statistically resolved by that interval; the A2/A3
improvement over this specific flat control is clearer. These are not
multiplicity-adjusted tests identifying logarithmic correlations.

![Normalized pair laws](phase3a/comparison/pair_cdfs.png)

Root-ball and triangle statistics expose remaining differences:

| Graph/model pair | SD of mass at radius .5, graph / field | Smallest/largest triangle mean, graph / field | Middle/largest mean, graph / field |
| --- | --- | --- | --- |
| I3 / selected | .02745 / .02739 | .62876 / .64635 | .85736 / .88082 |
| A2 / selected | .02326 / .03338 | .65357 / .62299 | .86416 / .84731 |
| A3 / selected | .02008 / .03338 | .64265 / .62299 | .86449 / .84731 |

I3 matches the half-median ball-mass SD especially well, but its middle
triangle ratio differs by −.0235, MC95[−.0364,−.0048]. A2/A3 selected fields
overproduce half-radius ball dispersion: graph-minus-field differences are
−.0101 [−.0146,−.0051] and−.0133 [−.0177,−.0092]. Their triangle differences
also remain. These are finite-family, unadjusted component intervals; graph
size and field discretization effects are separate. No passing equality test
or simultaneous fit is claimed.

![Rootwise ball-mass distributions](phase3a/comparison/ball_mass_laws.png)

![Triangle shapes from eight-point matrices](phase3a/comparison/triangle_shapes.png)

Eight-point matrices also give shared-root distance correlations of
.184/.117 for I3/selected, .139/.118 for A2/selected, and .052/.118 for
A3/selected. Disjoint-pair correlations and complete matrices are saved.
These noisy estimates demonstrate an available higher-point observable;
they are not precision estimates of an extra continuum parameter.

Finite-radius rootwise mass-growth slopes over normalized radii
.25,.5,.75 have median 3.70 for I3 versus 3.06 for its selected field;
A2/A3 medians are3.75/3.72 versus 3.27 for their selected field. The flat
control gives2.986. These are explicitly finite-radius diagnostics. They
should not be substituted for the asymptotic graph dimension or for the
crossing-derived dimension proxy below.

Maximum root-to-all distances provide lower diameter proxies, not exact
diameters. Median normalized proxies are1.91/1.62 for I3/selected,
1.82/1.94 for A2/selected and 1.77/1.94 for A3/selected. Their interpretation
depends on the root measure and small number of roots.

Full tables, quantiles, CDFs, root distributions and signed component
intervals: [comparisons.csv](phase3a/comparison/comparisons.csv),
[comparisons.json](phase3a/comparison/comparisons.json),
[distribution arrays](phase3a/comparison/summary_distributions.json), and
[audited auxiliary diagnostics](phase3a/diagnostics_audited/diagnostics.json).

### Reuse of the largest exact graphs

The648 archived I3 graphs give a particularly useful size check without new
large simulations. At n524,288 (64 graphs), pair W1 to the selected N48 field
is **.0140**, versus **.0299** to the flat control. The paired MC95 interval
for the flat-minus-selected W1 difference is [.0052,.0186]. Ball-dispersion
discrepancy is .00956 versus .05489, with improvement interval
[.0354,.0517]. This is actual compatibility evidence in two intrinsic
distributional diagnostics, beyond the n8192 discovery plot.

It is not complete stability. At this largest graph size, half-radius
ball-mass SD is .01849, below the selected field's .02739, and root-averaged
global distance CV is .04057 versus .08034. Large eight-point matrices are
not in the archived data, so the small-graph triangle residual cannot be
checked there without a future targeted matrix calculation.

![Archived I3 finite-size checks](phase3a/diagnostics_audited/large_graph_finite_size.png)

See [large-graph comparisons](phase3a/large_graph_comparison/summary.json).

## 7. Parameter scan and possible correspondences

The discovery ensemble has 24 independent fields at N16/24/32, all30 Cartesian
pairs from xi=0,.1,.2,.3,.4,.6 and gamma=0,.5,1,1.5,2, with axis and 26-direction
paths. All parameters and resolutions share the same field randomness within
a realization. This is30 parameter combinations, not30 independent ensembles.

The declared objective combines pair W1, root-ball dispersion and two
triangle-shape mean differences. Average ball mass duplicates the pair CDF
and is not double-counted. At graph n8192 versus field N32/26 directions,
the frozen selections are I3:(xi=.3,gamma=1.5) and A2/A3:(xi=.2,gamma=0).
The latter are explicitly unmatched metric/measure parameters.

Independent confirmation uses24 new fields at N24/32/48 and 26/98 directions.
Besides the selected pairs and flat control, gamma=.5/1 at xi=.2 bracket the
rough discovery exponent-based gamma proxy. They were declared before
confirmation analysis, rather than selected after inspecting confirmation.
See [frozen selection](phase3a/selection/selection.json) and
[confirmation plan](phase3a/confirmation_plan.json).

![Discovery parameter scan](phase3a/selection/parameter_scan.png)

The selection is not unique: I3's next-best discovery J is .0747 at(.2,0),
only .0048 above its winner. The additional independent xi=.2 probes give
J=.0649/.0737 for A2 at gamma=.5/1, and .0648/.0725 for A3. These can be
closer than the frozen gamma0 choice on the new seeds. They are reported as
predeclared bracketing checks, not retrospectively certified new winners.
The ordering is unstable enough to preclude a unique covariance-to-parameter
map. Only three graph covariances were tested; no continuous birectangular
or general-covariance scan was needed for this first comparison.

### Crossing exponents and an implicit conversion attempt

Opposite-face cube crossings use the exact axis vertex-sum convention on
the torus field restricted to the open cube. Their physical scale factor
is retained. With common field identities across N,500 paired bootstrap
draws give the following full-window confirmation proxies:

| xi | Finite crossing lambda, MC95 | Q proxy | Implicit gamma proxy, MC95 | gamma/xi proxy, MC95 |
| ---: | --- | ---: | --- | --- |
| .2 | .0673 [.0462,.1266] | 4.6636 | .695 [.677,.752] | 3.475 [3.386,3.758] |
| .3 | .1785 [.1278,.2123] | 2.7382 | 1.514 [1.341,1.680] | 5.048 [4.470,5.600] |

The inversion uses the subcritical branch
`gamma=6/(Q+sqrt(Q²-6))` only when Q>sqrt(6). At xi0, the site-crossing
slope is zero to rounding error; the uncorrected edge crossing instead
has a spurious slope from the known shrinking face gap, which is why
endpoint and domain conventions matter.

The xi=.3 full-window gamma proxy is close to the selected1.5, and its
dimension proxy overlaps the prior correction-dependent I3 estimate near 4.9.
But the same confirmation data give gamma proxies **1.290 and 1.746** in
adjacent N24→32 and 32→48 windows, corresponding to dimensions4.30 and 5.82.
Discovery-window results also drift. This prevents a reliable limiting
conversion. The reported intervals cover Monte Carlo variation, not that
extrapolation error or the torus/whole-space distinction.

The xi=.2 proxy instead suggests gamma near .70, not gamma0. Its dimension
proxy near 3.48 is also below the published A2 graph estimate 4.67±.04
(Budd–Castro Table 2, angles alpha=pi/3,beta=pi/2). Thus
the small-size distributional optimum does not simultaneously establish
the required one-parameter dimension relation. This is a concrete tension
for that candidate, not a disproof of another parameter choice or covariance.

![Crossing and window sensitivity](phase3a/crossing_analysis/confirmation.png)

[Discovery crossing table](phase3a/crossing_analysis/discovery.csv),
[confirmation crossing table](phase3a/crossing_analysis/confirmation.csv).

## 8. Finite-size effects and uncertainty

There are several different uncertainties; a single bootstrap cannot combine
them honestly.

| Source | Observed scale / treatment |
| --- | --- |
| Independent physical sampling |24 confirmation fields and 24 graphs/model/size; cluster bootstrap, root counts not inflated |
| Root and matrix sampling | Eight points/realization; triangle scalar 80%-power MDE about .015–.046 in a discovery normal approximation |
| Graph integer-distance atoms | Median-distance jumps about .126–.132 in discovery n8192 graphs; continuous-CDF KS has an atom floor |
| A2/A3 polygonal minima | q32→64 four-path paired checks; mean changes small but roughly7% changing local edges |
| Field cutoff and GMC | Strong near gamma2; coarse probability-mass changes persist under coupled cutoff refinement |
| Path directions |98 versus 26 raw changes about 2–3%; fixed-stencil anisotropy remains measured |
| Distance CDF binning | Maximum bin displacement about .002 ensemble-median units in the principal N48 comparisons |
| Exponent extrapolation | Adjacent-window dimension proxies at xi.3 drift4.30→5.82 |
| Finite-volume ensemble | Torus versus cube changes the intrinsic law; it is not merely numerical error |

Reported root-averaged CVs and between-realization ball variability still
contain finite eight-root sampling noise. The within/between decomposition
is descriptive, rather than a deconvolution of pure geometric fluctuations.

At n8192 the paired q32-minus-q64 mean-distance changes are
−.309%±.421% SE for A2 and−.189%±.141% for A3, with only four paired
physical paths each. Edge Jaccards are about .926–.931. This supports using
these graphs exploratorily, not claiming exact continuum correlated metrics.
At n2048 the analogous means are −.217% and−.084%, with larger uncertainty.

Small-graph finite-size shifts are substantial. The n2048→8192 normalized
graph joint discrepancies are about .147,.127,.164 across I3/A2/A3, larger
than several selected cross-model discrepancies. Independent graph split
noise floors are .042,.035,.028; I3's field split J at N32 is about .058.
These are descriptive scale comparisons, not subtraction of a universal
noise bias. In particular, a lower bound of a nonnegative W1 bootstrap
interval is not a formal inequality-of-laws test.

Integer atoms explain much of the apparent KS residual near normalized
distance one. Large I3 atoms shrink to about .053 at n524,288. The auxiliary
diagnostic initially used searchsorted on a floating-point median a few
ulps above its integer support, occasionally inspecting the next bin. The
audited diagnostic rounds to the integer support. Original outputs remain
preserved; the report uses `diagnostics_audited/`. Graph generation and the
pair/ball comparison scores were unaffected by that reporting correction.

![Cutoff and stencil dependence](phase3a/comparison/finite_resolution.png)

Eight-field same-noise controls separate several effects. At the selected
I3 pair, switching sharp to heat regularization changes the normalized joint
summary by about .0255; doubling the mesh at fixed N24 bandlimit changes it
by .0830 while changing raw mean distance by only−.079%. The latter joint
shift includes changed finite point sampling, not a pure path-error bound:
inverse-CDF coupling of points across meshes is not a spatial transport
coupling. Even its flat control has Jshift .039. Consequently stable mean
distance is informative but insufficient for claiming stable full matrix law.

Open-cube versus torus raw means are21% higher at the selected I3 pair and
about 38% higher at(.2,0); normalized J shifts are about .150 and .224.
These are different finite geometric ensembles. They cannot be interpreted
as evidence against an embedding-free continuum comparison while ignoring
boundary conditions. An additional exact Euclidean torus control confirms
that the flat-control conclusion is not an artifact of the remaining98-ray
angular error.

### Global scale and measure normalization

Realizationwise median normalization changes I3's selected pair CDF by
W1≈.0146, compared with≈.0020 at the selected A2/A3 pair. It removes genuine
global fluctuation information, so it is not the primary comparison.

There is also a distinct, mathematically motivated unit-mass gauge. For
gamma>0, shifting h by `-log(M_N)/gamma` preserves the normalized measure
and multiplies every LFPP distance by `M_N^(-xi/gamma)`. This is **not**
conditioning on M_N=1 and is not arbitrary per-realization median scaling.
For the I3 selected pair, its global-distance CV drops from .0803 to about
.0391, close to the largest graph's .0406, while its n8192 joint J worsens
from .0855 to .1036. Thus volume/gauge conventions can move different
observables in different directions. A global-CV mismatch alone does not
yet identify an additional continuum parameter.

### A specificity check

A paired eight-field probe replaces h by its |k|≤2 projection and rescales
back to the same point variance, keeping xi=.3,gamma=1.5. This is a smooth
Gaussian control, not a log-field/GMC limit. Its joint J to I3 is .156,
versus .106 for the corresponding eight log fields; the paired smooth-minus-log
interval is [−.061,.121]. It is too small to distinguish the correlation
mechanisms. Some heterogeneity-sensitive improvement is evident, but its
specificity to log correlations is not demonstrated.

See [statistics methods/power audit](../docs/STATISTICS_PHASE3A.md),
[paired improvement](phase3a/paired_improvement/summary.json) and
[audited systematics](phase3a/diagnostics_audited/diagnostics.json).

## 9. Mathematical interpretation and possible research directions

The result is scientifically more informative than a dimension-only match.
Normalized global pair shapes can resemble even a flat torus. Random
root-ball heterogeneity separates the graph geometry from that homogeneous
control, while triangle ratios and fluctuations reveal residual structure.
The observed improvement is therefore not a unique fingerprint of an LQG
field; its concentration in the ball-dispersion term should temper any
correspondence claim.

I3 near(.3,1.5) is a useful candidate for further study. Its largest archived
pair and ball diagnostics outperform the flat control, the full-window
crossing-derived dimension overlaps the graph estimate, and richer stencils
preserve the broad finite-resolution comparison. Against that evidence are
the triangle residual, smaller large-graph root dispersion, gauge-sensitive
global scale and non-stabilized exponent conversion. These point to specific
next tests rather than a conclusion of equality.

A2/A3 have finite intrinsic laws close enough that their apparent best field
parameters are not reliably separated by this experiment. Gamma=.5/1
bracketing probes can outperform their frozen gamma0 winner, showing
parameter-selection uncertainty. The necessary dimension relation is not
simultaneously satisfied by the xi=.2 optimum. This may reflect finite-size
effects, the objective's weighting, different volume ensembles, or a real
mismatch of the proposed one-parameter family. Three graph covariances
cannot discriminate those explanations or exclude another covariance locus.

A graph GFF would be a new conditional Gaussian model with covariance given
by a graph Laplacian inverse. It would not automatically be a Euclidean 3D
log field: an ordinary 3D GFF has a 1/r rather than logarithmic singularity.
No graph field or canonical embedding was introduced here. Pair laws, ball
masses and sampled distance matrices are nevertheless intrinsic probability
metric-space invariants and can be compared without an embedding. Matching
them at finite resolution does not establish Gromov–Hausdorff–Prokhorov
convergence, topological three-manifold structure, or a 3D LQG identification.

## 10. Recommendation and reproducibility handoff

**Outcome B: partial or ambiguous correspondence.** The finite numerical
comparison is informative, with qualified compatibility near I3/(.3,1.5),
clear discrimination from a homogeneous control in ball heterogeneity, and
specific residual discrepancies. It is not a validated parameter relationship.
Neither Outcome A nor a robust continuum Outcome C is justified. The
validated field/GMC and LFPP components do not require calling the whole
comparison untrustworthy; their finite-resolution limits are explicit.

Recommended Phase 3B, **not launched**:

1. Calibrate crossing slopes separately using inexpensive axis site-sum
   crossings over a wider range, e.g. N32–128, with coupled fields and
   regularizer/infrared checks. Keep adjacent-window and correction-model
   sensitivity visible before asserting a gamma–xi conversion.
2. Concentrate on xi near .3,gamma≈1.4–1.6 for I3. Use more measure points
   on a modest independent ensemble, refine the metric mesh independently
   of field cutoff, and compare the grounded-field and unit-mass Weyl-gauge
   ensembles. Establish isotropy through a richer solver/stencil study rather
   than simply raising N with fixed directions.
3. Regenerate a targeted subset of existing large I3 seeds to add32-point
   matrices, resolving the triangle/higher-point question without another
   broad graph campaign. Extend A2/A3 sizes only with paired minimum checks.
   Treat xi≈.2,gamma≈.7 as a distributional bracketing test with explicit
   dimension tension, not an established A2/A3 correspondence.
4. Increase power for log-field versus smooth/short-range Gaussian controls,
   and study smaller normalized balls above both graph-atom and field-cell
   scales. This tests specificity and distinguishes scale effects from a
   possible extra geometric parameter.

Actual computational scale: field validation12.7s; discovery metric checkpoints
sum202s; graph checkpoints sum66s; independent confirmation checkpoints sum960s
(about 16min), versus a preliminary estimate 500s. Repeated98-direction N48
builds were more expensive than extrapolating the 26-direction pilot. Every
physical/grid checkpoint was saved; no checkpoint required a long monolithic
run. The three eight-field sensitivities took42s,12s,11s. No production
64³ metric scan, general covariance sweep or extra Brownian repair was needed.
The new output tree is about 267 MB before final source snapshots/manifest.
The26-direction memory benchmark is measured; a production98-direction
peak-RSS bound was not instrumented and is not claimed.

Final validation: **147 tests pass**, covering existing graph controls, field
normalization/coupling, GMC, path-cost conventions, geometric controls and
independent comparison/crossing checks. Raw NPZs preserve fields, seeds,
CDFs, full sampled matrices, crossings, masses, configurations and source
hashes. Main source versions were frozen or recovered with exact hash checks.

Reproduction entry points:

- `experiments.phase3a_field_validation` and `phase3a_field_second_moments`.
- `experiments.phase3a_lfpp_validation` for geometry/cost controls.
- `experiments.phase3a_ensembles` for checkpointed field/graph ensembles.
- `experiments.phase3a_compare` for frozen selection and confirmation.
- `experiments.phase3a_crossing_analysis`, `phase3a_diagnostics`,
  `phase3a_paired_improvement`, and `phase3a_large_graph_comparison` reuse
  saved data. `diagnostics_audited/` is the corrected auxiliary analysis.

Use new output directories or the exact saved configuration when resuming;
completed outputs are not replaced. The initial graph pilot exposed an edge
array/set conversion error after saving its I3 checkpoints; it was fixed and
only the missing special cases resumed. The original experimental design's
proposed I3 split was superseded by the actual48-per-size,24/24 split recorded
in the data. Pilots and paired sensitivities are not extra independent samples.

[Final tests](phase3a/final_tests.xml),
[final preservation/artifact manifest](phase3a/final_manifest.json),
[checkpoint handoff](phase3a/STATUS.md), and
[research log](../Agents_Info/RESEARCH_LOG.md). Task 3A stops for review here;
the next major phase requires a new user instruction.
