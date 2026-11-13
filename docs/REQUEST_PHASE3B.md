# Research Task 3B: Adaptive Parameter Search for a 3D Mated-CRT / LFPP / GMC Correspondence

## Primary objective

Phase 3A found preliminary similarities between the independent three-tree Budd–Castro model and 3D LFPP/GMC near (xi,gamma)=(0.3,1.5).

However, that phase investigated only three isolated covariance matrices in the full three-parameter Budd–Castro family, and it used a sparse independent grid of gamma and xi values.

This is insufficient for discovering a general parameter correspondence.

I want Phase 3B to conduct a substantially more flexible, adaptive numerical investigation of the covariance and LFPP/GMC parameter spaces.

**Our priority is exploratory scientific discovery rather than rigorous convergence certification.** Reasonable numerical approximations are acceptable if their limitations are quantified, monitored, and disclosed.

Do not restrict the investigation to the independent three-tree model or assume that (xi,gamma)=(0.3,1.5) is the correct correspondence.

## Part 1: Correct the mathematical parameter framework

Review the project reports, especially Phases 2C and 3A, and the papers by Contreras Hip–Zhuang on higher-dimensional LFPP and weak exponential metrics.

In the subcritical metric framework, gamma and xi are not independent parameters.

For dimension three, the theoretical relationship is

Q(xi)=3/gamma+gamma/2,

xi=gamma/d_gamma,

lambda_cross(xi)=1-xi*Q(xi).

The map between gamma and xi is unique in the relevant regime but not explicitly known.

Therefore:

1. Treat the theoretically coupled family as the main candidate LFPP/GMC family.
2. Numerically estimate the gamma(xi) relationship using crossing exponents.
3. Account for Monte Carlo uncertainty and finite-resolution extrapolation error.
4. Interpolate the estimated parameter relationship, preserving uncertainty bands.
5. Retain independently varied gamma and xi values as diagnostic controls, not as theoretically valid correspondences.

Do not treat gamma=0 with xi>0 as an ordinary subcritical LQG metric-measure pairing.

Check all normalization and sign conventions against the original mathematical papers.

When estimated Q values do not permit a subcritical gamma or are unreliable, flag those regions instead of forcing a conversion.

## Part 2: Improve the LFPP exponent estimates

The Phase 3A crossing-derived gamma(xi) estimates were unstable across adjacent grid sizes.

Develop a more systematic crossing study using coupled Gaussian fields at several resolutions, extending toward N=64, 96, or 128 where computationally feasible.

Use the inexpensive nearest-neighbor, paper-faithful crossing metric for this exponent-calibration stage.

Separately assess sensitivity to field regularization, boundary conditions, and finite-size correction models.

Increase the resolution in xi beyond the earlier grid.

Initially prioritize xi near 0.2–0.35, but examine the larger accessible range and adaptively extend it if numerical evidence suggests a better candidate.

Use interpolation or regression with suitable uncertainty quantification to estimate a continuous relationship gamma(xi).

Do not identify narrow bootstrap intervals with control of extrapolation bias.

Output a reusable numerical parameter-conversion module and a table containing xi, estimated gamma, d_gamma, crossing exponent, Monte Carlo error and systematic uncertainty.

## Part 3: Enable exploratory general-covariance graph sampling

The Budd–Castro model uses

C(a,b,c)=[[1,a,b],[a,1,c],[b,c,1]].

Positive definiteness is necessary.

Phase 2C developed promising corner-aware Ritz approximations for generic cones but did not certify complete path sampling.

For this exploratory task, investigate whether the positive Ritz harmonic approximation and improved Brownian stopping skeleton can be combined into a provisional general-covariance generator.

Do not claim it is exact or fully validated.

First run the approximate pipeline against I3, A2 and A3, for which independent reference samplers exist.

Measure errors in path statistics, joint minima, mean graph distances, root-ball laws, and relevant higher-point observables.

Then perform small pilot tests on moderate-condition-number generic covariance matrices.

Use multiple angular approximation degrees and Brownian path resolutions for selected pilot matrices to assess stability.

If a general approximate generator can produce stable enough exploratory results, make it available for a broader coarse parameter survey.

If it cannot, do not invent results or quietly substitute an incorrect law. Instead, conduct the broadest feasible search in analytically tractable continuous covariance families and document the remaining restrictions.

The goal is to distinguish plausible geometric trends from large numerical artifacts, not to achieve arbitrary numerical precision.

## Part 4: Broad covariance parameter search

Investigate a much larger sample of the admissible covariance domain than Phase 3A.

Include:

- Equicorrelated matrices (rho,rho,rho).
- Birectangular matrices (rho,0,0).
- Symmetric two-correlation matrices (rho,0,rho).
- Genuinely generic asymmetric triples.

Avoid singular and extremely ill-conditioned matrices in the initial scan.

Exploit coordinate-permutation symmetry to avoid redundant simulations.

Use a structured space-filling design, such as Latin hypercube or another appropriate method adapted to the positive-definite domain.

Aim initially for approximately 20–40 distinct covariance configurations if the implementation and cost permit. Treat this as a target, not a reason to run unvalidated or prohibitively expensive simulations.

For the broad screening stage, use relatively small graphs and modest independent sample counts.

Do not claim precise Hausdorff dimensions from such graphs.

Instead, compare normalized distance distributions, rootwise ball-mass fluctuations, triangle ratios, and other inexpensive intrinsic geometric summaries.

Reuse existing validated graph data wherever possible.

Maintain explicit sampler accuracy and resolution indicators for every configuration.

## Part 5: Adaptive optimization

After the initial parameter scan, identify promising regions of the joint parameter space.

Do not simply select the smallest observed discrepancy.

Account for:

- Monte Carlo uncertainty.
- Brownian sampler and grid-resolution sensitivity.
- LFPP path discretization.
- Field-cutoff sensitivity.
- Changes in graph size.
- Statistical uncertainty from parameter optimization.

Use a multistage adaptive search.

For example:

1. Fit a surrogate or interpolation model to the preliminary discrepancy data.
2. Identify regions where agreement appears promising or the model is uncertain.
3. Sample additional covariance matrices near those regions.
4. Refine xi and its theoretically linked gamma simultaneously.
5. Increase numerical precision and graph sizes for the most interesting cases.
6. Retain independent candidate configurations for later confirmation.

Consider approximately 5–10 promising covariance matrices for intermediate refinement, followed by 2–4 finalists, subject to observed computational cost.

Avoid presenting a unique optimal parameter when several possibilities are indistinguishable within numerical uncertainty.

Use held-out seeds to avoid repeatedly selecting and evaluating candidates on the same samples.

## Part 6: Investigate the structure of the correspondence

Do not assume a unique covariance matrix for each gamma.

Consider two possibilities:

A. There exists a distinguished one-parameter curve of covariance matrices whose metric-measure limits correspond to the gamma-family of 3D LFPP/GMC metrics.

B. Several different covariance matrices produce the same intrinsic limiting geometry, so gamma may be a scalar function of the covariance triple, with nontrivial level sets.

Explore whether the measured geometry varies significantly in independent covariance directions.

For example, find covariance matrices with similar apparent metric dimensions and test whether their normalized higher-point distance distributions are also similar.

Determine whether changes in the covariance parameters can be reproduced by changing gamma, or whether some geometric differences persist after optimizing gamma.

Do not infer a continuum parameter relationship from a single small-graph fit.

A meaningful positive outcome would be a reproducible geometric pattern across several covariance matrices, not merely one isolated numerical coincidence.

## Part 7: Broader and more informative metric-measure comparisons

Use the existing Phase 3A comparison code as a starting point.

Retain:

- Pairwise distance distributions.
- Rootwise metric-ball mass distributions.
- Triangle side-length ratios.
- Higher-point distance-matrix distributions.
- Global distance fluctuations.
- Finite-radius volume growth.

Give higher-point and multiscale observables substantial weight.

Calibrate discrepancies against same-model independent sample splits.

Investigate whether the conclusion depends strongly on arbitrary weights assigned to different components of the joint objective.

Compare at multiple graph and field resolutions whenever feasible.

Keep the field's original zero-mode normalization and unit-mass Weyl gauge as separately defined ensembles. Do not select the gauge after inspecting which gives the best fit.

Include flat and smooth Gaussian controls for the strongest candidate matches.

Distinguish large-scale finite-domain geometry from mesoscopic statistics.

Do not automatically compare an unknown graph continuum topology to a torus as though their global geometries must be identical.

## Part 8: Computational efficiency and autonomy

Use specialist agents where useful for mathematical parameter calibration, general-covariance graph sampling, adaptive optimization and independent statistical validation.

Before expensive simulations, estimate runtime and memory from pilot experiments.

Use cached fields, graph realizations, roots, distance arrays and parameter-independent intermediate results.

Vary GMC gamma values without regenerating the underlying Gaussian fields.

Reuse previously computed LFPP distance structures when possible, but recompute them when xi changes the weighted path metric.

Save each completed configuration with seeds, full parameters, approximation settings and source hashes.

Design experiments to survive usage interruptions without redoing completed results.

Do not run a large combinatorial Cartesian product of covariance matrices, gamma, xi, graph sizes and resolutions.

Use an efficient coarse-to-fine adaptive design.

The task should produce actual numerical results, not merely a plan or new software architecture.

## Part 9: Outputs and conclusions

Produce:

`results/RESEARCH_REPORT_PHASE_3B.md`

and reproducible datasets and figures under `results/phase3b/`.

Include:

1. The calibrated gamma(xi) relationship and its uncertainties.
2. The number and distribution of covariance matrices actually tested.
3. Validation and approximation status for each graph sampler.
4. Initial broad parameter-search results.
5. Adaptive refinement and optimization results.
6. The best candidate correspondence regions and associated uncertainties.
7. Finite-size and numerical-discretization checks.
8. Evidence for or against a one-parameter covariance subfamily.
9. Evidence for or against many-to-one covariance-to-gamma relationships.
10. Recommendations for Phase 3C.

Include plots showing the covariance parameter domain, the sampled configurations, promising regions and the corresponding LFPP/GMC parameters.

Clearly distinguish numerical fit quality from theoretical compatibility.

If the general sampler is still too unreliable for a broad search, give the strongest possible restricted-family results and identify which remaining mathematical obstacle limits the scientific conclusions.

Do not claim that matching one or several finite-dimensional observables proves a common continuum limit.

**The primary deliverable is a map of candidate parameter correspondences and their reliability, not a declaration that a unique correspondence has been discovered.**

Finish the feasible experiments, update `RESEARCH_LOG.md`, and stop for review before launching a major Phase 3C confirmation campaign.