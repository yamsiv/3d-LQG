# Unit-log torus field and GMC: Phase 3A methods and validation

This work introduces Euclidean fields independently of the Brownian graph
encoding. No graph GFF or covariance-to-LFPP parameter map is assumed.
The new implementation is `src/log_field_3d.py` and `src/gmc_3d.py`.

## Source convention and Fourier derivation

The local source is Dello Schiavo–Herry–Kopfer–Sturm,
[arXiv:2302.02963v1](https://arxiv.org/abs/2302.02963v1), Introduction,
§1.1 and Lemma 1.4; it was read directly from `papers/PolyharmonicFields.pdf`.
Their constant in dimension n is

```
a_n = 2/[Gamma(n/2)*(4*pi)^(n/2)],
Cov(h)=a_n^(-1)*(-Delta)^(-n/2)
```

on the mean-zero unit torus. Thus a_3=1/(2*pi²). For complex Fourier modes
exp(2*pi*i*k·x), the Laplacian eigenvalues are (2*pi*|k|)², and

```
S(k)=E|h_hat(k)|²=2*pi²/(2*pi*|k|)^3=1/(4*pi*|k|³), k!=0.
```

The shell integral is 4*pi*(1/(4*pi))*integral dr/r, giving coefficient
one for the logarithmic singularity. This agrees with Lemma 1.4. The zero
mode is removed, fixing the infrared normalization; adding a random constant
would multiply both metric and unnormalized GMC by random global factors.

The implemented spectrum is a regularization of the **continuum** torus
polyharmonic field. It is not the discrete nearest-neighbor Laplacian spectrum
`[4N² sum sin²(pi*k_j/N)]^(-3/2)`. The paper distinguishes raw lattice,
Fourier-projected and reduced constructions and different convergence ranges.
Its reduced-measure full subcritical theorem is not claimed automatically for
this finite quadrature scheme.

Default sharp regularization retains integer modes 0<|k|<N/3. This spherical
cutoff avoids Nyquist ambiguity and leaves three grid points per shortest
wavelength. A heat-multiplier alternative has covariance multiplier
exp[-(|k|/(N/3))²], with a strict |k_j|<N/2 cap. That cap and the regularizer
change finite-grid results and are recorded explicitly. They are not silently
identified with the white-noise fields used in the LFPP theorems.

## FFT and coupling

For requested sizes, a real iid N(0,1) array on the largest grid is transformed
by FFT/sqrt(Nmax³). The resulting complex coefficients satisfy
z_-k=conj(z_k), E|z_k|²=1, and independent nonconjugate component variance 1/2.
At each smaller grid the same integer coefficients are selected. The field is

```
h_N = ifft_N(N³ sqrt(S_N) z),
C_N(j/N) = ifft_N(N³ S_N),
v_N = sum_k S_N(k).
```

Realness and stationarity therefore follow analytically, up to floating-point
roundoff. S_N and C_N are cached immutable arrays. Shared low modes agree to
roundoff even at non-nested sizes 16,24,32. Reproducibility requires recording
the complete requested sizes tuple: the same seed with a different largest
grid does not promise identical Gaussian mode values.

`resample_field(field,Nfine)` evaluates the identical band-limited Fourier
polynomial on a finer spatial grid, preserving its deterministic variance.
This separates quadrature/path mesh error from adding high-frequency field
randomness. All coupled LFPP and GMC quantities must use the same returned h.

## GMC definitions and moment qualifications

For unit-volume grid cells, the unnormalized quadrature masses are

```
w_j = N^-3 exp(gamma*h_j - gamma²*v_N/2),
M_N=sum_j w_j,       p_j=w_j/M_N.
```

The implementation uses log-sum-exp and keeps both M_N and p. It subtracts
the deterministic ensemble variance, never a sample's spatial variance. In
particular E M_N=1 exactly on every finite grid. Gamma0 gives uniform mass.
No samplewise normalization is performed before checking that expectation.

In unit-log dimension3, the continuum subcritical range is |gamma|<sqrt6.
For a fixed nonempty-volume set, positive q moments have threshold
q<6/gamma²; gamma0 is deterministic. The Euclidean set-mass exponent is
zeta(q)=3q−gamma²*q*(q−1)/2 in its admissible range. A dyadic partition sum
has exponent zeta(q)−3, rather than zeta(q). These exponents concern Euclidean
balls/cubes, not LFPP metric balls or a metric Hausdorff dimension.

Every fixed finite grid has finite moments even when their continuum limits
do not. At gamma2, the threshold is1.5, so a continuum second-moment/variance
claim would be false. Heavy tails can invalidate an apparently precise
empirical variance before their divergence becomes visible.

An exact finite-grid uncertainty check is available independently of Monte
Carlo:

```
E[M_N²] = N^-3 sum_j exp(gamma²*C_N(j/N)).
```

This follows from the two-point Gaussian exponential identity and translation
invariance. Its square root yields the exact standard error of an independent
sample mean. It does not guarantee a Gaussian approximation at small sample
counts. `experiments/phase3a_field_second_moments.py` evaluates this identity
from the saved covariance and sample count, without new simulations.

## Executed experiments and budget

Eight coupled field/GMC realizations cost .427 seconds with17.6 MB Python-
traced peak allocation; the linear256-realization estimate was13.7 seconds.
The actual256 independent common-field realizations took12.68 seconds, at
N=16,24,32,48 and gamma=0,.5,1,1.5,2. Each realization is checkpointed as one
JSONL row. Its four resolutions and spatial sites are not additional
independent replicates. The first two complete coupled fields are saved as
compressed NPZ; all others have reproducible seeds and configurations.

The first64 realizations also compare the heat multiplier using identical
Gaussian Fourier modes, and compare fixed-field GMC quadrature N16→32 and
N24→48. These are paired systematic-error diagnostics. Coarse 1³,2³,4³,8³
block masses, normalized effective cell fraction, maximum cell mass, empirical
covariances and three complex mode components are retained. Numerical normal
tests are diagnostics, not simultaneous proofs or acceptance certificates.

Reproduction commands, with new output paths when outputs already exist:

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3a_field_validation --output results/reproduction/phase3a_field_validation
.venv/bin/python -m experiments.phase3a_field_second_moments
.venv/bin/python -m pytest -q tests/test_log_field_3d.py tests/test_gmc_3d.py
```

## Main numerical findings

The exact point variances at N16/24/32/48 are
1.96920,2.37715,2.66848,3.07507. Over N16 through128, the deterministic
variance slope against logN is1.00426. Axial covariance fits on declared
finite windows give coefficients .957–1.023. Window and cutoff oscillations
are visible; the data were not rescaled to force a coefficient one.
The empirical spatial covariances agree with exact spectral sums within
realization-level uncertainty. Tested mode component variances .434–.538
are consistent with1/2 at256 independent samples; their common values across
N verify the intended coupling, rather than four independent observations.
At equal integer radius3, the exact axial-minus-oblique covariance discrepancy
is .00997 at N16 and −.000870 at N48. Cubic and periodic artifacts remain,
despite the isotropic leading singularity.

At N48 the total-mass results are:

| gamma | Mean mass | Empirical mean SE | Exact finite-grid mean SE,256 samples | Mean effective cell fraction |
| --- | ---: | ---: | ---: | ---: |
|0|1|0|0|1|
|.5|.99929|.00253|.00262|.4731|
|1|.99405|.01099|.01176|.06986|
|1.5|.97114|.02859|.03717|.00882|
|2|.89003|.05678|.21128|.00176|

Gamma2 demonstrates rare-tail undercoverage: the true finite-grid second
moment is12.428, much larger than suggested by its empirical mass variance.
The observed mean .890 is not convincing evidence of broken variance
renormalization; the exact mean is1, and its exact sampling SE is .211.
High-gamma confidence should not be based on the empirical SE alone.

For normalized coarse4³ block probabilities, paired N32→48 L1 differences
are .0583±.00067 SE at gamma1 and .2906±.00681 at gamma2. Thus high-gamma
metric-measure conclusions are materially cutoff sensitive at these sizes.
In contrast, at fixed N24 field cutoff, doubling quadrature resolution gives
gamma2 mean relative total-mass change .225%±.433% SE across64 fields,
though its realization SD is3.47%. Mean mass alone hides local errors; this
quadrature check does not bound all coarse masses or metric-ball masses.

Three-scale Euclidean block-moment slopes are saved, with admissibility flags.
The block sizes are close to the field cutoff and infrared scale, so they are
finite-scale diagnostics rather than asymptotic multifractal measurements.
Near-critical GMC is not validated by merely obtaining positive weights.

## Artifacts and remaining scope

- `results/phase3a/field_validation/metadata.json`: exact seed/configuration and
  source/software hashes at the experiment checkpoint.
- `records.jsonl`:256 independent realization rows with all coupled statistics.
- `fields_000.npz`, `fields_001.npz`: complete shared field samples.
- `summary.json`, `exact_spectral_diagnostics.json`,
  `mass_uncertainty_audit.json`: independent uncertainty and spectral checks.
- `mass_uncertainty_audit_plotted.json` and `.png`/`.pdf`: the same exact
  mass-variance audit with figures using exact finite-grid SE whiskers.
  These whiskers are not asserted Gaussian confidence intervals.
- `field_gmc_validation.png` and `.pdf`: readable standalone figure.

Eight dedicated tests pass. The implementation makes controlled finite-grid
field/GMC ensembles accessible; it does not establish LFPP universality or
a correspondence with the graph model. Confirmation should prioritize
gamma≤1.5, while retaining gamma2 as a labeled intermittency/cutoff stress
test. Metric discretization and gamma–xi matching remain separate questions.
