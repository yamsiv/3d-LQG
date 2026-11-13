# 3D LQG Analog — Research Context and Codex Working Brief

**Status:** Working research context, prepared November 2026.  
**Repository intended for:** `3d-LQG` (copy this file into the repository root or `research-notes/`).  
**Primary immediate assignment:** Implement and validate the Budd–Castro higher-dimensional mated-CRT **graphs**, study their intrinsic metric and candidate field observables, and compare the evidence to dimension-three Liouville first-passage percolation (LFPP) and Gaussian multiplicative chaos (GMC).  
**Document role:** Persistent context, literature map, definitions, research hypotheses, execution instructions, validation criteria, and open problems for coding/research agents. This document is not itself proof of any conjecture.

> **Agent read-first instruction.** Read this entire file, then the original Budd–Castro paper, especially Sections 2.5, 3, 4 and Tables 1–2, **before** implementing a sampler. The 3D graph is made from an **octant-conditioned Brownian excursion**, not merely three unrestricted Gaussian random walks. Do not state that the graphs have a proven 3-manifold scaling limit, a canonical 3D GFF, a GMC law, or an identified LFPP coupling. Record model definitions, approximations, caveats, and reproducible evidence separately.

---

## 0. Mission, research taste, and decision history

The overarching goal is to investigate whether there is a mathematically natural **three-dimensional analogue of Liouville quantum gravity** (3D LQG), potentially emerging as a universal scaling limit of discrete random geometries, with a tractable intrinsic random metric, volume measure, probabilistic encoding, and natural diffusion/growth processes. Also ask what survives or changes in dimensions `d >= 3`.

The user's research background is rigorous probability, random geometry, GFF/GMC, mated-CRT maps, SLE/LQG, random planar maps, and statistical mechanics. Work should be mathematically precise, research-level, explicit about which statements are rigorous, and open to unexpected results. Avoid assuming the hoped-for correspondence.

**Evolution of the proposal (important priorities):**

1. Initial desiderata: a 3D conformally natural log-correlated field/GMC volume, conformal covariance, an intrinsic LQG-type metric, a random-graph or triangulation scaling limit, and perhaps an induced 2D LQG boundary measure.
2. Subsequent conclusion: a *mandatory* 2D-LQG boundary theory is **too restrictive** and should **not** be a gate for the project. A boundary theory and bulk–boundary gluing can be pursued as secondary possibilities.
3. Current focus: **higher-dimensional mating of trees, intrinsic graph metrics, LFPP/GMC comparisons, and possibly QLE-, Eden-, or DLA-like growth/diffusion**. Do not insist the 3D model is a conformal scalar field from the outset.
4. Main computational project: code the **Budd–Castro 3D mated-CRT graph**, analyze the **metric and possible fields associated with it**, and scan its three-parameter Brownian correlation family to see whether any lower-dimensional locus might be compatible with one-parameter 3D LFPP/GMC.
5. After a rigorous baseline and simulation summary, let results guide whether to pursue a matching-parameter conjecture, a metric universality result, an emergent field construction, or a different class of geometry.

**Priority order:** correct model/sampler → strong 2D validation → 3D distances/volume/spectral observables → define and test candidate fields → independent LFPP/GMC benchmarks → parameter fitting with uncertainty → interpret, falsify, or refine conjectures. QLE/DLA and physical quantum-gravity interpretations are downstream.

**Critical distinctions:**

- Brownian **encoding dimension 3** does not establish a **topological 3-manifold** or a **Hausdorff dimension of 3**.
- A numerical graph Hausdorff dimension `d_H(C)` is *potentially comparable* to 3D LFPP metric dimension `d_{γ,3}`, **not directly** to its crossing exponent `λ(3,ξ)`.
- One matching exponent is far from a metric, measure, or field universality theorem.
- The 3D covariance family has **three independent pairwise correlations**, not only the two angles plotted in the Budd–Castro paper (which restricts to an isosceles slice). Notation `γ_angle` in their paper is unrelated to GMC `γ`.
- The graph does **not** come with a canonically defined three-dimensional log-correlated field or a canonical Euclidean embedding. Finding an appropriate field is itself an open problem.

---

## 1. Core models at a glance

| Object | Input | Natural discrete/continuum output | Status (as of Nov. 2026) |
|---|---|---|---|
| Standard 3D GFF | `(-Δ)^{-1}` | Covariance singularity ~ `|x-y|^{-1}` | Rigorous, **not** log-correlated |
| Critical 3D log-correlated field | `(-Δ)^{-3/2}` with normalization | Distribution with covariance `-log|x-y| + O(1)` | Rigorous on suitable backgrounds |
| 3D GMC | log field `h`, `0<γ<sqrt(6)` | Random measure `μ_h^γ` | Rigorous subcritical theory |
| Higher-dimensional LFPP | regularized log field `h_ε`, exponent `ξ` | Renormalized random path metric | Tightness, weak-metric theory, dimension bounds; general uniqueness open |
| Budd–Castro 3-tree graph | positive-octant excursion from 3D correlated Brownian motion `C` | finite nonplanar graph `G_n^C`, graph metric, counting measure | Construction/numerics; continuum metric/topology unknown |
| Random tree-decorated 3D triangulations | random tetrahedra and tree decorations | random simplicial complexes | Physics Monte Carlo/numerical phases, no LFPP identification |
| 3D odd-dimensional Liouville CFT | fractional conformal operator, exponential interaction | formal interacting scalar field theory | Physics predictions; full rigorous counterpart incomplete |
| Eden/FPP growth | graph + positive edge weights | metric-ball growth under FPP distance | Canonical finite-graph process; continuum 3D-tree limit open |
| DLA / dielectric breakdown | graph + harmonic measure | random cluster growth | Canonical finite-graph processes; 3-tree continuum limit open |
| 2D QLE | LQG + SLE/Loewner interface | random growth and 2D LQG metric | Deep rigorous 2D theory; **not** a ready-made 3D SPDE |

### 1.1 Two main candidate notions of 3D LQG

**Approach F (field-first).** Sample the critical log-correlated Gaussian field on a fixed Euclidean 3D domain; use GMC for volume and renormalized LFPP for distances. This is a random metric-measure geometry on a fixed topological 3-space. It has rigorous mathematical infrastructure, but it is not yet proved universal for random 3D triangulations.

**Approach G (graph-first).** Construct Budd–Castro graphs intrinsically from three correlated Brownian excursions. Analyze graph distance, counting measure, random walk, potential theory, and growth; only afterwards determine whether a Gaussian-field description exists. The graph limit could be new and unrelated to the field-first model.

**Working conjecture, NOT an established connection:** For some one-parameter curve `C(γ)` in the three-parameter family, `G_n^{C(γ)}` with intrinsic distance and normalized counting measure might approach a compact 3D LFPP/GMC-type metric-measure space. Both the existence of that curve and the common scaling limit remain open.

---

## 2. Budd–Castro: precise mathematical and simulation setup

**Primary reference:** T. Budd and A. Castro, *Scale-invariant random geometry from mating of trees: A numerical study*, Phys. Rev. D **107** (2023), 026010, [arXiv:2207.05355](https://arxiv.org/abs/2207.05355). Author-hosted [PDF](https://hef.ru.nl/~tbudd/docs/treemating.pdf). Important locations: §2.5 polygon triangulation and 2D validation; §3.1 multigraph; §3.2 cone eigenvalues; §§3.3–3.4 biased excursion/Doob transform; §4 Monte Carlo; Table 1 (2D) and Table 2 (3D). This is the **implementation authority**; confirm exact endpoint conventions from equations/figures.

### 2.1 Correlation parameter space: three parameters, not two

Let `X(t)=(X_1(t),X_2(t),X_3(t))` be 3D Brownian motion with covariance

```math
\operatorname{Cov}(X_i(s),X_j(t)) = C_{ij}\min(s,t), \qquad
C=\begin{pmatrix}1&a&c\\a&1&b\\c&b&1\end{pmatrix}.
```

Normalize `C_ii=1` without loss for the graph law: positive coordinate rescalings preserve the order/minimum comparisons. There are **three** pairwise correlations `(a,b,c)` satisfying `C` positive definite. A convenient validation condition is `|a|,|b|,|c|<1` and

```math
\det C = 1 + 2abc-a^2-b^2-c^2 >0.
```

Budd–Castro express correlations by **three spherical angles**:

```math
C_{12}=-\cos\alpha,\quad C_{23}=-\cos\beta,\quad
C_{13}=-\cos\theta.
```

The original paper calls the third angle `γ`; in this project use **`θ` / `theta_angle`** to avoid confusing it with **GMC coupling `γ`**. Angles must describe a valid spherical triangle (positive-definiteness is the robust numerical test). The paper's numerical phase diagram mainly uses the **isosceles restriction** `β=θ` (so `C_23=C_13`), leaving two angular parameters `(α,β)`. The **equilateral/equicorrelated** subfamily `α=β=θ` has

```math
C_\rho=(1-\rho)I_3+\rho\mathbf1\mathbf1^T,\qquad -1/2<\rho<1,
\quad \rho=-\cos\alpha.
```

At `ρ -> -1/2` the covariance becomes singular of rank 2, not rank 1; `X_1+X_2+X_3` becomes constrained. **Do not sample exactly on the singular boundary by Cholesky decomposition.** Study `ρ=-0.5+δ` with controlled `δ>0`.

Other initial subfamilies: `C=I_3` (three independent excursion coordinates); `C_12=a,C_13=C_23=0`; isosceles `C_12=-cos α, C_13=C_23=-cos β`; and general positive-definite matrices generated by normalized random SPD matrices (label parameter sampling distribution explicitly).

### 2.2 Excursion law and cone geometry

For the **finite-volume** graph, use a Brownian excursion `X:[0,1]→R^3_{>=0}` starting and ending at 0 and remaining strictly in the positive octant at interior times, with covariance `C`. This is a conditioned **vector** process: sampling an unconstrained correlated Brownian bridge and independently clipping/reflecting coordinates is wrong. Independent positive 1D excursions are exact only at `C=I_3`.

Let `C=RR^T` via Cholesky, `W=R^{-1}R^3_{>=0}`. Then standard Brownian motion in the cone `W`, conditioned to make a cone excursion, maps through `R` to correlated motion in the octant. `W∩S^2` is a spherical triangle with angles `(α,β,θ)`.

The principal Dirichlet spherical eigenpair is

```math
-\Delta_{S^2}m_1=\lambda_1m_1\quad \text{on }W\cap S^2,
\qquad m_1|_{\partial W}=0.
```

Budd–Castro connect it to the **string susceptibility** exponent:

```math
\gamma_s(C)=1-\sqrt{\lambda_1+\tfrac14}
\qquad(d=3).
```

Here `γ_s` is a *string susceptibility*, **not** the LQG/GMC parameter `γ`. A 3D exact benchmark is the birectangular case `β=θ=π/2`:

```math
\lambda_1=(1+\pi/\alpha)(2+\pi/\alpha),\qquad
\gamma_s=-\tfrac12-\pi/\alpha.
```

This supplies an independent validation observable, separate from the graph distance dimension.

### 2.3 Finite mated-CRT graph: horizontal chords / triangulated polygons

Partition `[0,1]` into `n` equal time cells `I_i=[(i-1)/n,i/n]`, `i=1,...,n`. Each scalar excursion coordinate `X_k` gives a triangulation of a convex `n`-gon with vertices `1,...,n`: include the polygon boundary cycle and, for non-neighboring `i<j`, draw an interior chord if there exist `s∈I_i,t∈I_j` with `X_k(s)=X_k(t)` and `X_k(u)>=X_k(s)` for all `u∈[s,t]`. Equivalently, use the minimum-over-interval criterion from §2.5 (carefully resolve strict/weak inequalities and cyclic endpoint handling).

**Graph `G_n^C`:** glue the three polygon triangulations **along the same Hamiltonian boundary cycle**. Take their union as a multigraph if retaining parallel edges/multiplicity, or reduce to a simple graph when measuring unweighted shortest-path distances. Record which choice is used; edge multiplicity can matter for random walks, conductances, and any Laplacian-defined field even when it does not change unweighted graph distance.

Every scalar coordinate's chord set must give a noncrossing triangulation of the `n`-gon, with **exactly `n-3` internal diagonals** for a generic resolved excursion, and the full `n`-edge boundary cycle. This is an invaluable unit/integration test. The union of two independent coordinate triangulations must yield the known 2D mated-CRT triangulation; the union of three need not be planar or a simplicial 3-complex.

**Implementation pitfall:** Comparing *only the sampled values at cell endpoints* may miss important within-cell minima and generate incorrect chords. For a piecewise-linear approximation, compute minima over all interpolated segments (not only endpoints); refine time resolution well below `1/n`. Optionally develop a coupled Brownian-bridge extrema refinement. Do not sample minima independently across coordinates if that destroys their cross-correlation. Compare graphs at successively finer resolutions with the **same underlying refined path**.

**Algorithm design:** Begin with an `O(n^2)` or `O(n^2 log m)` *reference* adjacency algorithm for small `n`, checked against a direct geometric horizontal-segment routine. Then build an optimized monotone-stack or interval-minimum/triangulation method. Each component creates `O(n)` edges; the resulting graph should be sparse. Never allocate a dense `n×n` adjacency matrix for production sizes.

### 2.4 Sampling: unbiased vs biased excursions

For general `C`, sampling the correctly conditioned cone excursion is the hardest part of faithful replication. Budd–Castro §3.3 introduce a **biased excursion** `\widehat X` with law weighted by

```math
\frac{\int_0^1 |X(t)|^{2\alpha_1-2}\,dt}{\mathcal C_{\alpha_1}},
\qquad \alpha_1=\sqrt{\lambda_1+(d/2-1)^2},
```

and use a Doob `h`-transform in the cone to simulate trajectories conditioned to hit the vertex at the origin. Here `h(x)` is a homogeneous harmonic function proportional to

```math
|x|^{-\alpha_1-d/2+1}m_1(x/|x|).
```

The bias changes the finite-volume ensemble but is absolutely continuous relative to the unbiased excursion, and is argued to preserve appropriate critical/local exponents. **For reproducibility, track and label the biased ensemble `hat_G` separately from unbiased `G`; do not claim they have identical finite-`n` distributions.** Follow §§3.3–4.1 for exact path concatenation and time rescaling, including inverse-gamma hitting times and the angular starting density `m_1^2`.

Implementation progression, in order of confidence:

1. `C=I_3`: independent standard 1D Brownian excursions for each coordinate (can sample finite-time marginals via independent 3D Brownian-bridge norms/Bessel-3 bridges), then refine to control chord errors. This is a **correct independent-coordinate baseline**.
2. `d=2`: any correlation via the exact planar prescription and/or paper's cone algorithm; use known 2D dimensions for calibration.
3. General `d=3,C`: implement and test the cone/Doob-`h` biased-excursion sampler from §4.1. If a simpler approximation is used while developing, label **`approximate / not Budd–Castro law`** prominently and do not use it for universal claims.
4. Cross-validate in overlapping cases (`C=I_3`, 2D known examples), and quantify sampler discretization bias separately from finite-graph-size bias.

### 2.5 Established numerical benchmarks to reproduce

From Budd–Castro Table 2 (their **biased** 3D excursion graphs, angle convention `(α,β,θ=β)`):

| Angles `(α,β)` | Reported 3D `d_H` estimate | Note |
|---|---:|---|
| `(π/3+0.005, π/3+0.005)` | `4.08 ± 0.04` | near degenerate equilateral corner |
| `(π/3, π/3+0.04)` | `4.36 ± 0.11` | near corner, isosceles |
| `(π/3, π/2)` | `4.67 ± 0.04` | isosceles |
| `(π/2, π/2)` | `4.83 ± 0.06` | `C=I_3` |
| `(2π/3, π/2)` | `5.15 ± 0.09` | isosceles |
| `(9π/64, 1.14)` | `5.61 ± 1.16` | large uncertainty |

These are **numerical finite-size fits**, not exact dimensions or rigorous scaling limits. Near-degenerate-corner behavior `d_H≈4.1` is tentative and notably does **not** tend toward the classical dimension `3` in these simulations. The paper investigated `n=2^11,...,2^19` in the finite-size analysis; reproduce a credible smaller pilot first, scale only after validation. The authors used histogram collapses with additive finite-size shifts and correction-to-scaling terms; report fit protocols rather than treating a one-line log–log regression as definitive.

**2D calibration:** For `d=2`, Brownian correlation `ρ=-cos(π γ_2^2/4)` with **2D LQG** `γ_2∈(0,2)`. Key checks: `γ_2=sqrt(2)` (independent coordinates, reported dimension around `3.59`); `γ_2=sqrt(8/3)` (Brownian-map/LQG dimension exactly `4`, Budd–Castro report `4.07±0.14`). In the `d=2` case do not equate full finite mated-CRT graph metric convergence with the proved continuum SLE/LQG mating identity; these are different statements.

### 2.6 Graph observables and geometric structure

For each graph sample and parameter:

- Degree distribution, multi-edge multiplicities, edge types by contributing tree, graph sparsity, connectivity and Hamiltonian-cycle check.
- Distances `d_G(U,V)` for uniform independent vertices; root-to-all BFS; diameter approximations with safeguards; radius/quantiles.
- Empirical pair-distance histogram `ρ_n(r)`; rescale `r/n^{1/d_H}` and test full distribution collapse, not just the mean.
- Intrinsic ball volumes `V_G(v,r)=#{u:d_G(u,v)≤r}` and ball mass `V_G/n`; finite-volume/boundary/time-order biases.
- Graph bottlenecks, cut vertices and two-vertex separators, degree-dependent shortcuts; candidate minbu counts and their exponent `γ_s(C)`.
- Effective resistance and random-walk return probabilities (later, after the graph is correct). Note multigraph vs simple-graph conductance changes these results.
- If asking about topology, recognize `G` alone is just a graph. Any purported 3D complex must be *defined* and checked (e.g., links of vertices, local homology, Betti numbers) before calling it a 3-manifold.

---

## 3. Three-dimensional LFPP and the log-correlated Gaussian field

**Key references:**

- Jian Ding, Ewain Gwynne, Zijie Zhuang, *Tightness of exponential metrics for log-correlated Gaussian fields in arbitrary dimension*, [arXiv:2310.03996](https://arxiv.org/abs/2310.03996).
- Andrés A. Contreras Hip, Zijie Zhuang, *Bounds on the distance exponent for higher-dimensional Liouville first passage percolation*, [arXiv:2504.09141](https://arxiv.org/abs/2504.09141).
- Andrés A. Contreras Hip, Zijie Zhuang, *Weak exponential metrics for high-dimensional log-correlated Gaussian fields*, [arXiv:2512.06292](https://arxiv.org/abs/2512.06292).
- Related work of Contreras Hip and Gwynne on metric regularity and Hausdorff dimension; review the cited papers and later versions when implementing.

### 3.1 Correct Gaussian field (not the ordinary 3D GFF)

The **ordinary** 3D massless GFF has covariance with singularity comparable to `|x-y|^{-1}` and is **not** the 3D log-correlated field. The correct log-correlated analog is a critical fractional/polyharmonic Gaussian field with formal covariance

```math
\operatorname{Cov}(h)\propto(-\Delta)^{-3/2},
\qquad\mathbb E[h(x)h(y)] = \log(1/|x-y|)+O(1)
```

for a unit-log-covariance normalization. On a compact torus remove the zero mode; on `R^3` define modulo additive constants/test-function restrictions or through a white-noise decomposition, and specify IR regularization. A Fourier construction on a periodic mesh has spectral amplitudes proportional to `|k|^{-3/2}` **for the field itself** (variance ∝ `|k|^{-3}`), with zero mode removed. Verify the chosen FFT conventions and covariance coefficient empirically; a missing factor changes `γ` and `ξ`.

A suitable white-noise scale decomposition is

```math
h_\epsilon(x)=\int_\epsilon^{\text{IR cutoff}}
\int_{\mathbb R^3}K((x-y)/t)t^{-2}\,W(dy,dt),
```

with normalization chosen to achieve coefficient one for `log`. Different regularizations need not agree at finite scale, and comparison of metric limits is nontrivial.

### 3.2 GMC volume

The (subcritical) 3D Gaussian multiplicative chaos is the weak limit

```math
\mu_h^\gamma(dx)=\lim_{\epsilon\downarrow0}
\exp\!\left(\gamma h_\epsilon(x)-\frac{\gamma^2}{2}
\mathbb E[h_\epsilon(x)^2]\right)\,d^3x,
\quad 0<\gamma<\sqrt6.
```

Normalizations and domain/zero-mode conventions matter. Critical coupling is `sqrt(2d)=sqrt6` in 3D under unit-log covariance. The Euclidean-ball moment exponent (for admissible `q`, away from boundaries and with a specified IR normalization) is

```math
\zeta_\gamma(q)=3q-\frac{\gamma^2}{2}q(q-1).
```

A typical GMC point has Euclidean local dimension `3-γ²/2` (distinct from Hausdorff dimension in the **random metric**). The GMC expectation at a fixed test set is just Lebesgue volume, due to variance renormalization. Do not confuse its Euclidean multifractal spectrum with metric-ball mass scaling.

Dello Schiavo–Herry–Kopfer–Sturm, *Polyharmonic fields and Liouville quantum gravity measures on tori of arbitrary dimension: from discrete to continuous*, [arXiv:2302.02963](https://arxiv.org/abs/2302.02963), prove rigorous discrete-to-continuum field/GMC results; consult precise variants and allowed `γ` ranges for raw vs reduced measures. This is an independent computational benchmark. **Do not describe basic discrete polyharmonic GMC convergence as an entirely open project.**

### 3.3 LFPP definition and normalization

For regularized `h_ε`, define the continuum regularized exponential metric

```math
D_\epsilon^{\xi}(x,y)=\inf_{P:x\to y}
\int_P e^{\xi h_\epsilon(z)}|dz|.
```

The lattice formulation weights an `ε Z^3` path by `ε∑_v e^{ξ h_ε(v)}` (specify edge/vertex convention). Distances need a nontrivial deterministic normalization, e.g., typical left-to-right unit-cube crossing distance. The field-first mathematical theory obtains tightness/subsequential continuous metrics in its subcritical metric regime; the higher-dimensional full metric uniqueness and random-graph identification are not established. Do **not** substitute raw Euclidean distances or simply use GMC weights as graph edge lengths without defining the precise model.

Define the high-probability LFPP crossing exponent by

```math
D^{\epsilon,\xi}(\partial_L Q,\partial_R Q)=
\epsilon^{\lambda(3,\xi)+o(1)}
```

for a fixed cube `Q`, with the paper's exact normalization and probabilistic definition. `λ(3,0)=0`. The associated 3D LQG Hausdorff dimension parameter `d_{γ,3}` is tied implicitly to `λ` by

```math
\xi=\frac{\gamma}{d_{\gamma,3}},\qquad
Q_3=\frac3\gamma+\frac\gamma2,\qquad
\lambda(3,\xi)=1-\xi Q_3,
```

hence

```math
d_{\gamma,3} =
\frac{3+\gamma^2/2}{1-\lambda(3,\gamma/d_{\gamma,3})}.
```

Here `Q_3` is the background-charge/coordinate-change exponent in the log-field convention, **not** the matrix `C` and not `γ_s` (string susceptibility). In the field-first theory, the weak exponential metric has Hausdorff dimension `d_{γ,3}`; this is not the ordinary topological dimension `3` nor the Euclidean local dimension of GMC.

### 3.4 Rigorous bounds to use in numerical comparison

Contreras Hip–Zhuang prove for `d>=3`:

```math
\underline\rho(\xi)\le\lambda(d,\xi)
\le\xi\sqrt{2d-2},
```

and dimension monotonicity across ambient dimensions. In dimension 3, `λ(3,ξ)≤2ξ`. Corollary 1.9 gives (for `0<γ<sqrt6`)

```math
\max\left\{3+\tfrac12\gamma^2,
\frac6{\sqrt{15}+4}\left(3+\tfrac12\gamma^2
+(\sqrt{5/2}-1/\sqrt6)\gamma\right)\right\}
\le d_{\gamma,3}\le 3+\tfrac12\gamma^2+2\gamma.
```

The exact `d_{γ,3}` for nonzero `γ` is not known. Numerical examples (bounds, not inferred values):

| GMC `γ` | rigorous `d_{γ,3}` lower | upper |
|---:|---:|---:|
| `0` (limit) | `3.000` | `3.000` |
| `0.50` | `3.125` | `4.125` |
| `1.00` | `3.561` | `5.500` |
| `1.25` | `3.999` | `6.281` |
| `1.50` | `4.484` | `7.125` |
| `2.00` | `5.598` | `9.000` |

**Important scientific caution:** The intervals are wide. A Budd–Castro `d_H` near 4.8 is compatible with many `γ` values. Matching a bound interval is not evidence for a unique correspondence. Conversely, the equicorrelation near-degenerate corner appears to approach `d_H≈4.1`, whereas LFPP `d_{γ,3}→3` for `γ→0`. This is a **conditional obstruction only if one independently justifies mapping that corner to the LFPP semiclassical limit**; no such map is established.

**Numerical LFPP obligations:** Independently simulate appropriate 3D log fields and LFPP shortest-path/crossing distances at multiple mesh sizes, using consistent normalization; estimate `λ(3,ξ)`, `d_{γ,3}` via the implicit relation (with propagated errors), and GMC moment exponents. Benchmark at `ξ=0` / `γ=0`, compare with rigorous inequalities, and use a range of mollifiers when affordable. Exact equality in law of the two models is **not** established by finite-resolution fit.

---
## 4. The first task's central ambiguity: what is the *field associated with* a 3-tree graph?

This must be decided carefully. Budd–Castro gives a Brownian encoding, a graph and counting measure, **not** an immediately defined scalar GFF or GMC field in three spatial coordinates. We must not invent an embedding, assume a graph GFF equals a 3D log field, or infer GMC purely from a fractal graph-distance exponent.

### 4.1 Several testable candidates, all with different meanings

**(A) Encoding-coordinate fields.** The functions `X_1(t),X_2(t),X_3(t)` are 1D-in-time Brownian excursion heights. They are genuine known inputs, not putative 3D log fields. Study correlations/excursions, minima, cone observables and dependence of local graph statistics on `X_i(t)`. An apparent logarithmic covariance of a transformed *time-indexed* statistic is not a `-log|x-y|` covariance on 3D Euclidean space.

**(B) Graph (massless) GFF.** On each finite connected graph, choose a conductance convention and normalized/un-normalized Laplacian `L`, and sample the centered Gaussian random vector `φ` with covariance `L^+` (Moore–Penrose pseudoinverse) on constants' orthogonal complement. This is a perfectly well-defined *derived* random field; it is **extra Gaussian randomness conditional on the graph**, not a field canonically encoded by the Brownian data. Its logarithmic or nonlogarithmic behavior depends on resistance/spectral properties. In a spectral-dimension-3-like regime, ordinary graph GFF is **not** expected to be log-correlated.

**(C) Graph spectral fractional Gaussian fields.** Define `φ_s` conditionally on `G` with covariance `(L|_{1^⊥})^{-s}` for a documented spectral power `s`, plus appropriate mass and normalization. On a regular Euclidean `d`-lattice, critical log correlation is achieved by `s=d/2`; on a fractal random graph, **spectral dimension** rather than topological dimension enters the criticality heuristic. Varying `s` until a log is observed is *model engineering*, not evidence that the original graph spontaneously produces an LQG field. Document this distinction.

**(D) Coarse-grained intrinsic volume fluctuation.** Define for vertices/centers `v`

```math
H_r(v)=\log\mu_n(B_{d_G}(v,r))
-\mathbb E[\log\mu_n(B_{d_G}(V,r))],
```

or normalized variants and joint-scale increments. These quantities are canonically determined by the graph, but `H_r` is **not** automatically Gaussian or log-correlated. Study multiscale covariance, higher cumulants, dependence on graph distance, and finite-volume effects. Avoid calling it the GMC logarithm without a field-reconstruction theorem.

**(E) Log density under a *proposed* embedding.** If an independently justified embedding `ι_n:V_n→D⊂R^3` exists, define smoothed vertex-counting density `ρ_{n,ε}(x)` and compare `log ρ` to a 3D log field. An arbitrary force-directed or random projection plot cannot establish this. The 1D Hamiltonian time order is not a 3D volume-preserving embedding. The 2D Tutte embedding is not immediately generalizable to arbitrary 3D nonplanar graphs.

**(F) Field recovered from an independently identified continuum measure.** In standard GMC settings, the underlying log field can be recovered from the measure under suitable assumptions. This could be useful *after* a candidate 3D embedding and measure convergence have been established, not as a justification to label finite graph counts GMC already.

### 4.2 What to implement first

Baseline exploratory field analysis should calculate **(A)** and **(D)** for every sample because they require no extra random field or embedding. Implement **(B)** for small/medium graphs as a distinct derived observable, and estimate the covariance versus graph distance and low eigenvalue spectral density. Add **(C)** only with an explicitly stated research question about which spectral covariance is log-correlated. Preserve all definitions and outputs so they can be compared across seeds and sample sizes.

Primary diagnostics:

- Empirical covariance of field/coarse-volume observables vs `log(dist)` or `log(time separation)` (clearly identify which distance), and compare to alternative power laws.
- Normality of finite-dimensional marginals *after centering/scaling*: skewness, kurtosis, QQ plots and stability with size; joint Gaussianity needs more than a 1-point histogram.
- Estimated variance growth vs increasing resolution; check scaling `Var(h_ε)≈log(1/ε)` in the correct **Euclidean** regularization for actual 3D LFPP/GMC reference fields.
- Spectral dimension from `p_{2t}(v,v)` and Laplacian eigenvalue density. A log-correlated **ordinary graph GFF** is naturally associated with spectral dimension near 2 (and suitable resistance estimates), which must not be conflated with intrinsic Hausdorff dimension 3, 4, or 5.
- Joint statistics of graph distances, ball masses, coarse fields and original Brownian minima. Look for robust coupling rather than only matching marginal distributions.

**A decisive negative result is valuable:** If there is no canonical 3D log field under any geometrically justified construction, the three-tree graphs may be a genuinely different random-geometry universality class. Do not conceal this outcome.

---

## 5. What exactly should be compared between Budd–Castro and 3D LFPP/GMC?

### 5.1 Comparison hierarchy, from weakest to strongest

1. **Well-definedness and regularity:** each finite model is correctly sampled and normalized; known 2D/`γ=0` cases pass.
2. **Metric exponent:** `d_H(C)` vs `d_{γ,3}`. Necessary if a common metric limit exists, but not sufficient. Use LFPP **bounds** as bounds, not as data points.
3. **Distance-distribution shape:** compare distribution of appropriately normalized distances between two random volume points, across sizes; metric samples need compatible finite-volume geometry and normalizations (e.g., disk/sphere vs cube are not automatically comparable).
4. **Ball volume:** intrinsic `μ(B_D(v,r))` distribution and scaling. This compares graph counting measure to LFPP metric plus GMC volume at a fixed total volume/radius convention.
5. **Multifractality:** only apply the **Euclidean** GMC moment law `ζ(q)` when a justified Euclidean embedding exists; otherwise define and compare intrinsic moment exponents separately.
6. **Spectral/transport geometry:** resistance, return probability and heat-kernel scaling, diffusion dimension, escape/exploration statistics.
7. **Joint coupling:** compare metric/volume/field joint laws (distance-weight correlations, thick points, conditional local volume) rather than separate histograms.
8. **Universality or identity theorem:** identify a common continuum metric-measure-space law, and eventually a canonical shared field/embedding. This is a conjectural endpoint.

### 5.2 Parameter correspondence hypotheses

**H0 (null):** No `C ↔ γ` correspondence; tree graphs form distinct scale-invariant universality classes, possibly not even 3-manifolds.

**H1 (equicorrelated one-parameter family):** `C=C_ρ` with `ρ=-cos α`. Search for `ρ(γ)` that matches multiple intrinsic observables, not just dimensions. This is the simplest symmetry-preserving candidate. Do not impose the 2D formula `ρ=-cos(πγ²/4)` in 3D as a theorem.

**H2 (isosceles curve):** Within `C_12=-cos α, C_13=C_23=-cos β`, fit a one-dimensional curve `(α(γ),β(γ))` selected by at least one *additional structural condition* (symmetry, exponent identity, exploration law, etc.). One dimension matching constraint in a two-parameter slice leaves families of possible fits.

**H3 (general three-parameter surface/curve):** Search the full SPD three-correlation space; use permutation-invariant features (`sum C_ij`, `sum C_ij²`, determinant, principal eigenvalues, cone `λ_1`, etc.). Symmetry may help reduce redundancy. Matching one target scalar in a 3D parameter space generically produces a 2D level set, not a canonical curve.

**H4 (universality only after reweighting/decoration):** Perhaps a decorated graph, FPP-weighted graph, or metric derived via diffusion—not the **unweighted** graph metric—could match LFPP. Treat each as a new model with separately reported statistics.

**H5 (different topology / no match):** The observed three-tree limit could fail to be homeomorphic to `R^3`; if so it cannot equal a metric inducing Euclidean 3-space topology. Numerically probe possible intrinsic obstructions rather than presupposing topology.

**Semiclassical diagnostic:** The 2D formula sends `γ_2→0` to perfect anticorrelation `ρ→-1`. For three equicorrelated coordinates the covariance degenerates at `ρ→-1/2`; this *may* be a tempting 3D analog, but Budd–Castro report `d_H≈4.1` there whereas field-first `d_{γ,3}→3`. Thus equating those endpoints would be **inconsistent if the numerics and limits are correct**. No reason is yet known that forces that identification; record this as a conditional test, not a contradiction or disproof of 3D LQG.

### 5.3 Fitting protocol and avoiding false discoveries

**Do not** optimize `(α,β,θ)` and `γ` simultaneously to match one exponent and call it a correspondence. The problem is underidentified. Instead:

- Estimate `d_H(C)` with statistical **and** systematic error bars, from *independent graph realizations*, several `n` and several sampling discretizations.
- Estimate independent dimension/metric exponents for LFPP, and overlay rigorous upper/lower bounds. Propagate covariance and normalization uncertainty.
- Define a multiobservable loss, e.g. `L=w_1 z_d² + w_2 W_1(hist_distances) + w_3 W_1(ball_mass_laws) + w_4 z_spectral²`, but report weights, parameter-sensitivity and distinct model volume domains. Do not combine noncomparable observables in the same loss.
- Use training vs **held-out** graph seeds and parameter points; examine predictive ability for new observables not used in fitting.
- Plot compatibility **regions** in `(a,b,c,γ)` and uncertainty bands rather than only a best-fit curve.
- Compare against null models: random regular/sparse graphs with similar degree distributions, ordinary Euclidean lattice graphs, 2D mated-CRT graphs, and graph-rewired controls preserving degrees. Otherwise a simple shared fractal exponent may look falsely special.
- For theoretical interpretation, test whether the fitted curve has an independent characterization (e.g., cone eigenvalue, symmetry or Markovian exploration) that could be formulated rigorously.

---

## 6. First executable experiment: deliverables and implementation specification for Codex

**Experiment ID:** `E001_budd_castro_graph_vs_3d_lfpp`  
**Scientific objective:** Faithfully reproduce Budd–Castro graph-distance measurements on baseline families, quantify what is measurable about an associated field and volume, and compare to independently simulated 3D log-field/GMC/LFPP benchmarks without overclaiming identification.

### 6.1 Repo structure (suggested, not a restriction on existing code)

```text
3d-LQG/
  RESEARCH_CONTEXT.md                # this document
  AGENTS.md                          # optional short instruction to read context
  papers/                            # original papers and citation notes; respect licenses
  src/lqg3d/
    covariance.py                    # SPD validation, parameter/angle transforms
    excursions.py                    # reference samplers + provenance labels
    triangulations.py                # scalar excursion -> polygon triangulation
    mated_graph.py                   # union of k triangulations and metrics
    graph_fields.py                  # clearly labeled derived field constructions
    lfpp_field.py                    # unit-log 3D log field generator
    lfpp_metric.py                   # crossing/point-to-point shortest paths
    gmc.py                           # chaos weights and measure statistics
    observables.py                   # distances, balls, walks, spectra, separators
    fitting.py                       # size-collapse and uncertainty estimation
    plotting.py                      # publication-ready plots, no invented data
  tests/
    test_covariance.py
    test_triangulation.py
    test_excursions.py
    test_graph_invariants.py
    test_2d_baselines.py
    test_lfpp_gmc.py
  experiments/
    e001_config.yaml
    e001_pilot.py
    e001_sweep.py
  data/                          # usually gitignored; record seed and manifest
  results/
    e001/
      figures/
      tables/
      logs/
      summary.md
      manifest.json
  research-notes/
    questions.md
    literature-status.md
```

**Tooling:** Python with `numpy`, `scipy` (`sparse`, `csgraph`, `fft`, optimization), `networkx` only for small-validation cases, `matplotlib`/`pandas`; speed-critical components can later use compiled extensions, Numba, Rust or C++, after agreeing on tests. Avoid an unnecessary heavy infrastructure build. Specify environment lockfile, Python version, CLI examples, and seed protocol.

### 6.2 Implementation work packages

**WP1 — Covariance family and scalar 2D baselines (must pass).**

- Implement a `Correlation3D` object with SPD checks (`eigmin>tol`, determinant, condition number), angle⇄correlation conversion, equicorrelated and isosceles constructors, permutation handling.
- Validate empirical covariance of *unconditioned* Brownian increments as a unit test—this does not validate the excursion distribution.
- Implement scalar Brownian excursions, polygon triangulation from continuous/piecewise-linear approximations, graph union. Independent excursions for `C=I` are a strong exact-case baseline.
- Verify each scalar polygon has `n` boundary edges and `n−3` diagonals, no crossing diagonals, Hamiltonian cycle, and the expected simple/multigraph edge accounting.
- Reproduce 2D graph-distance trends at `γ_2=sqrt2` and `sqrt(8/3)` before making any 3D claims.

**WP2 — Budd–Castro general 3D excursion sampler (highest scientific risk).**

- Implement **paper §3.3/§3.4 biased cone excursion** faithfully, including numerical angular eigenpair for general cone if required, `m_1²` angular sampling, `h`-transform, interior-ball steps, stopping threshold, path concatenation, and correct time normalization.
- Independently confirm any quoted discretization and algorithmic conditions directly in the paper; maintain a methodology note citing sections/equations.
- Compare in overlapping exactly tractable cases; quantify invariance under coordinate rescaling and tree-coordinate permutation.
- At near-singular `C`, monitor condition numbers and use a separate precision/sampler convergence study. No hidden clipping, ad-hoc rejection or coordinate-wise reflection without recording a deliberate alternative model.

**WP3 — Graph metric and finite-size scaling.**

- Use sparse adjacency lists/CSR and BFS for unweighted distances; keep original multi-edge labels in a parallel representation for conductances.
- Sample independent graphs and several uniform source vertices per graph; cluster/bootstrap **by graph** rather than treating all pairs from one graph as independent replicates.
- Suggested pilot sizes `n=2^8,2^9,...,2^13` (adjust to runtime); production sizes as far as possible toward paper's `2^11`–`2^19`. Record deviations in precision and graph ensemble.
- Metrics: means/quantiles of uniform-to-uniform distance, distance histograms, ball profiles, leading correction exponent, cut-point/minbu distribution.
- Fit `E[d_G(U,V)]≈A n^{1/d_H}(1+B n^{-ω})` and compare against paper's histogram-collapse-with-shift procedure. Use bootstrap confidence bands and systematic checks (fit-window and path resolution).
- First 3D anchor: `C=I_3` should be *numerically compatible* with Budd–Castro `d_H=4.83±0.06` under comparable ensemble and sufficient resolution/size. Do not promise exact reproduction at pilot sizes.

**WP4 — Associated field / graph potential theory.**

- Implement field variants **A/B/D** of §4, each with distinct names, metadata and plots.
- For derived graph GFF, solve sparse Poisson systems or sample using an eigendecomposition only on small graphs; ensure zero-mode removal. Distinguish vertex-degree and conductance normalizations.
- Compute covariance vs intrinsic distance, coordinate/time-index covariance, low-frequency spectrum, resistance proxy, random-walk return probabilities and multiscale log-ball-mass statistics.
- Investigate Gaussianity and log-correlations; note that a field produced *by adding Gaussian noise* does not demonstrate emergent Gaussianity from the original graph.
- No self-fulfilling definitions: do not define `h=log μ/γ` and report a GMC discovery without testing appropriate spatial regularity, renormalization, and an embedding.

**WP5 — Independent 3D LFPP/GMC reference simulation.**

- Implement unit-log 3D Gaussian field on a periodic `N³` lattice by spectral method; specify Fourier/zero-mode convention and independently verify covariance slope ≈1 against `−log spatial distance` over a finite scaling window. Periodicity and finite `N` alter covariance at large separations.
- Generate GMC weights `w_x=N^{-3}exp(γh_x−(γ²/2)Var(h_x))`; check `E total mass≈1` before samplewise normalization (which changes some statistics). Evaluate moment scaling and admissible `q`.
- Implement LFPP on a matching cubic lattice with `ξ≥0`, weights `N^{-1}exp(ξh_x)`, Dijkstra/shortest paths, unit-cube crossing and matched random-pair distance experiments. Verify `ξ=0` crossings O(1) and `λ≈0`.
- **Start with modest `N`** because 3D path search and ensemble averaging are expensive. Check the effect of torus vs box, correlation kernel, mass normalization, and mollifier before comparing to literature exponents.
- Estimate `λ(3,ξ)` from slope of log crossing length vs log `ε` with finite-size corrections; convert to a candidate `d_{γ,3}` only when solving the **implicit** `ξ=γ/d_{γ,3}` relation, or present `λ(ξ)` directly. Compare to rigorous bounds, not an imagined exact formula.

**WP6 — Parameter scan and cross-model comparison.**

1. Anchor parameters: `C=I_3`, equicorr `ρ` on a safe grid spanning negative, zero and positive correlations, isosceles slice, and a sparse sample of full SPD space.
2. Report sampler fidelity and metric dimension estimate first for each `C`.
3. For LFPP, select independent fixed grids in `γ` and `ξ` with documented implicit relation, compare graph `d_H` against the rigorous dimension-interval feasibility region.
4. If dimension fits are robust, compare *held-out* ball-volume laws, normalized distance histograms and random-walk/spectral observables where meaningful.
5. Fit/explore curves `C(γ)` only after independent evidence is available; display uncertainty and possible non-uniqueness.
6. Treat the symmetry line `C=C_ρ` as the natural first hypothesis, not an obligatory truth; the two-parameter isosceles slice and full 3D covariance space are crucial fallbacks.

### 6.3 Initial grid suggestion (modifiable after pilot)

| Family | Example parameters | Purpose |
|---|---|---|
| 2D known | `γ_2=√2`, `γ_2=√(8/3)` | Validate graph and distance pipeline |
| Independent 3-tree | `(α,β,θ)=(π/2,π/2,π/2)` | reproduce `d_H≈4.83` |
| Equicorr | `ρ∈{-0.45,-0.35,-0.2,0,0.2,0.5}` | test symmetric curve |
| Near singular | `ρ=-0.5+δ`, `δ∈{0.05,0.02,0.01,0.005}` | study tentative `d_H≈4.1`, strong sampler warnings |
| Isosceles | `α∈{π/3,π/2,2π/3}`, `β` on an SPD-feasible mesh | compare Budd–Castro Table 2 |
| Full SPD | Latin hypercube/random SPD normalized to unit diagonal | test whether symmetric slice is special |
| LFPP reference | `γ∈{0,0.5,1,1.25,1.5,2}` | bound curves/GMC tests; choose `ξ` consistently |

All proposed values are **experimental settings**, not inferred maps from `C` to `γ`. Automatically skip ill-conditioned/SPD-invalid points, and report the skipped region. Keep a pilot small enough to debug before a costly sweep.

### 6.4 Quantities and metadata every run must save

Minimum `manifest.json` fields:

```json
{
  "experiment_id": "E001_budd_castro_graph_vs_3d_lfpp",
  "git_commit": "RECORD_AT_RUNTIME",
  "date_utc": "RECORD_AT_RUNTIME",
  "python_environment": "RECORD_AT_RUNTIME",
  "model": "biased_BC_excursion_or_exact_case_or_explicit_approximation",
  "n_vertices": 4096,
  "C": [[1,0,0],[0,1,0],[0,0,1]],
  "angles_rad": [1.5707963267948966,1.5707963267948966,1.5707963267948966],
  "positive_definite_eigenmin": 1.0,
  "seed": 12345,
  "n_independent_graphs": "RECORD_AT_RUNTIME",
  "sources_per_graph": "RECORD_AT_RUNTIME",
  "brownian_path_resolution": "RECORD_AT_RUNTIME",
  "graph_edge_policy": "simple_for_distances_multigraph_retained_for_walks",
  "distance_normalization": "RECORD_AT_RUNTIME",
  "finite_size_fit_model": "RECORD_AT_RUNTIME",
  "notes": "record approximations and validation status"
}
```

The JSON is an illustrative manifest schema, **not** actual experimental data. For the LFPP run record `N`, `ε`, covariance kernel, field variance, `ξ`, `γ`, simulation seed, path-weight convention, boundary conditions, and mass normalization.

### 6.5 Figure and table checklist

- Panel A: example 2D graphs and 3D graph adjacency/degree visualizations (do **not** mistake a force-directed layout for a 3D embedding).
- Panel B: 2D calibration dimension vs known LQG dimension, with uncertainty.
- Panel C: raw 3D distance histograms `ρ_n(r)` and rescaled collapse for `C=I` plus equicorr cases.
- Panel D: `d_H(C)` heatmap/line for equicorr and isosceles, with Monte Carlo and finite-size uncertainty.
- Panel E: 3D LFPP rigorous `d_{γ,3}` upper/lower curves overlaid with **compatible bands** for tree graph dimensions, **not** a fabricated `γ(C)` correspondence.
- Panel F: canonical graph-derived coarse field covariances; ordinary vs fractional graph GFF plots separately.
- Panel G: ball-mass distribution comparison and, where justified, GMC Euclidean `ζ(q)` fit.
- Panel H: spectral/return probability, random walk, minbu exponent, and possible cone-eigenvalue validation.

Include numerical data tables (`.csv`/`.parquet`), confidence intervals, methods, and scripts for regeneration. All claims in plots must state sample sizes, fitted scaling window, seed/method, finite-size limitations, and whether the model is the *original*, *biased*, or *approximate* ensemble.

### 6.6 Go / no-go gates

- **Gate A (correct graphs):** triangulation invariants + exact/known baselines + sampler validation pass. Otherwise stop interpretive fitting.
- **Gate B (credible 3D distance):** fit stable across sample sizes, path resolution, bootstrap choices and sampling method. Otherwise report finite-size diagnostics only.
- **Gate C (field):** candidate field is explicitly defined; tests show whether Gaussian/log structure is supported. Otherwise report failure/inconclusive, not a GMC claim.
- **Gate D (LFPP):** Gaussian covariance and `γ=0` control tests pass; exponents respect rigorous bounds within estimation error. Otherwise fix the LFPP implementation.
- **Gate E (correspondence):** multiple independent observables fit out of sample. Only then formulate a precise parameter-curve conjecture; still **do not claim a theorem**.

**First agent deliverable:** runnable code/tests, a `results/e001/summary.md` with source citations, exact definitions, parameter grid, 2D validation, graph distance fits, field diagnostics, independent LFPP/GMC baselines, a table of compatible/ruled-out parameter ranges, limitations, and prioritized new questions. Do not claim to have run simulations until logs/data actually exist.

---

## 7. Longer-term proposed proof programs

### Program P1 — Intrinsic 3-tree geometry

**Conjecture P1.1:** For nondegenerate positive-definite `C`, there is `d_H(C)>0` and an appropriate finite-volume scaling for which

```math
(V(G_n^C),\,n^{-1/d_H(C)}d_{G_n^C},\,n^{-1}\sum_v\delta_v)
\Rightarrow (\mathcal X_C,D_C,\mu_C)
```

in a suitable Gromov–Hausdorff–Prokhorov metric-measure topology. Such a limit and `d_H` value are **not proved** for the general 3-tree model.

**Intermediate possible papers:**

- Sharp/coarse polynomial bounds on typical pairwise graph distance, graph diameter, or ball growth; bounds that depend explicitly on `C`.
- Degree-tail/multiscale bottleneck bounds from the Brownian-minimum encoding.
- Finite-size shape/tightness criteria via covering estimates, not just one-point distance distributions.
- Nondegeneracy/topology of the metric limit, including whether it is a 3-manifold at all.
- Exact or bounded `d_H(ρ)` at an equicorrelated solvable point or near `ρ→-1/2`.

**Proof tools:** strong Markov property and excursion decomposition; cone Brownian motion and spectral Dirichlet eigenvalue; comparison with 2D mated-CRT maps; multiscale graph decompositions; resistance/graph-energy estimates; metric covering numbers. Independent exponent information from minbu-size tails may help but does **not** alone control distance.

### Program P2 — Compare to high-dimensional weak exponential metrics

**Conjecture P2.1 (metric universality):** There exists a nontrivial locus `C(γ)` with both `d_H(C(γ))=d_{γ,3}` and a common or quantitatively comparable scaling-limit metric. Graph and LFPP processes are not naturally coupled today; inventing a *canonical* coupling is a central challenge.

**Potential intermediate results:**

- Establish LFPP crossing-exponent stability under a **particular discrete fractional-field regularization** (the 2025 LFPP bounds paper identifies discrete regularization universality as an open issue).
- Construct graph–Euclidean correspondences / embeddings or compare abstract metric-ball growth distributions.
- Build a coupling with coarse polynomial comparisons of distances (`n^{±o(1)}` after appropriate matching), and then improve to tight multiplicative constants if possible.
- Identify Weyl scaling and field measurability for the continuum metric; uniqueness of weak metrics in `d≥3` remains a major goal.
- Prove (or refute) compatibility of the GMC volume with the LFPP metric and with the graph counting measure.

**Major obstruction:** `G_n^C` might not be a 3-manifold. LFPP weak metrics induce the Euclidean topology in their established regime. Agreement of `d_H` is far too weak to settle topology or the metric law.

### Program P3 — Diffusion and growth: Eden first, DLA/QLE later

Define iid exponential edge passage times `τ_e`. On any finite graph the first-passage metric

```math
d_{\mathrm{FPP}}(u,v)=\inf_{P:u\to v}\sum_{e\in P}\tau_e
```

produces an **Eden-type cluster** `A_t={v:d_FPP(o,v)≤t}`. Study whether normalized weighted FPP metrics converge along with ordinary graph metrics, and whether the continuum FPP balls define a nontrivial canonical growth process. Equality of weighted and unweighted scaling metrics up to a factor must be **proved**, not assumed.

For DLA, harmonic measure of random walk, rather than shortest-path distance, controls attachment. On finite graphs with an outer source, define precise killed-walk/hitting distribution and update a cluster. A dielectric-breakdown variant has attachment weights proportional to `ω_A(e)^η` for interface edges, with `η=1` DLA and `η=0` uniform-interface-edge Eden-type growth. Note that the 2D QLE notation/parameter conventions are not automatically identical to this discrete `η`.

**Intermediate papers:** random walk recurrence/transience and spectral dimension of `G_C`; heat-kernel/effective-resistance estimates; convergence of discrete Dirichlet forms/diffusions; capacity and harmonic-measure convergence for evolving clusters; a Markovian exploration of the Brownian encoding; continuum Eden/FPP; continuum DLA if robust boundary estimates are available.

**QLE warning:** 2D QLE uses planar conformal uniformization, SLE and Loewner evolution, not just an SPDE on a boundary. In 3D there is no general conformal map of a domain to a ball. A natural 3D analog should probably first be characterized via intrinsic Eden or harmonic-measure growth and a Markov description of the unexplored random geometry. Ang–Mithal (2026), [arXiv:2605.03385](https://arxiv.org/abs/2605.03385), is relevant recent 2D work on quantum-natural-time QLE, growth phases, and cut-out quantum disk laws.

**Smooth-diffusion obstruction:** For a conformal metric `g=e^{2σ}g_Eucl` in dimension `d`,

```math
\Delta_g f=e^{-2\sigma}\left(\Delta f+(d-2)\nabla\sigma\cdot\nabla f\right).
```

In `d=2`, the gradient drift vanishes (conformal invariance of Brownian motion up to time change); in `d=3`, it remains and becomes singular as `σ` approaches a log field. Do not identify Euclidean Brownian motion time-changed by GMC with the canonical diffusion of a limiting 3D LFPP metric without a proof.

### Program P4 — Rigorous odd-dimensional Liouville CFT

Kislev–Levy–Oz, *Odd dimensional nonlocal Liouville conformal field theories*, [arXiv:2206.10884](https://arxiv.org/abs/2206.10884), develop physics-level odd-dimensional Liouville theory with nonlocal kinetic term, critical conformal operator, `Q`-curvature background, exponential potential, and predicted higher-dimensional three-point identities. Important rigorous analogs in **even** dimensions exist: Dello Schiavo–Herry–Kopfer–Sturm [arXiv:2105.13925](https://arxiv.org/abs/2105.13925). Basic arbitrary-dimensional discrete field/GMC convergence also exists [arXiv:2302.02963](https://arxiv.org/abs/2302.02963).

Candidate mathematical objectives: interacting finite-volume Liouville weight built from 3D GMC, careful zero mode/background charge, vertex insertion/Seiberg-type integrability conditions, Möbius covariance on `S³`, and ultimately rigorous correlation identities. **An `exp(-μM_γ)` tilt of a Gaussian reference field is easy compared with full LCFT**; do not market the elementary tilt as proving the full CFT. A 3D DOZZ-type formula is ambitious because Virasoro/BPZ techniques specific to 2D are not available in the same way in 3D.

Relevant conformal notions in field-first 3D geometry: for unit-log normalization `Q_3=3/γ+γ/2`; under a Möbius coordinate change with local length scale `λ_f(x)` the field transforms formally as `h↦h∘f+Q_3 log λ_f`, yielding GMC conformal covariance after consistent changes of variables/renormalization. This uses the very restricted Möbius conformal group in dimension three; it is **not** a general coordinate-invariant random 3D metric theory. A generic 3D Riemannian metric is not conformally flat (Cotton tensor obstruction), so one scalar conformal factor cannot parametrize all 3D metrics.

### Program P5 — Optional bulk–boundary field and measure theory

The earlier proposal to **require** a 2D LQG boundary was dropped. Still, a log-correlated whole-space field restricted to a flat codimension-1 plane has a two-dimensional log covariance (after appropriate normalization). This suggests *optional* joint bulk/surface chaos and trace/Markov questions. Distinguish fixed planar restriction, arbitrary curved boundaries and intrinsic boundary metrics. A fixed smooth surface carries zero *bulk 3D GMC* mass; separately renormalized boundary chaos is needed. Surface GMC coupling and conformal charge compatibility depend on boundary conditions and normalization. No automatic relation between induced ambient metric with shortcuts and an intrinsic 2D LQG metric.

Possible theorems: joint discrete-to-continuum bulk/surface field and GMC, cross-covariance estimates and mixed moments, conditional laws under gluing, and boundary conditions for nonlocal conformal operators. This remains secondary to `E001`.

### Program P6 — Other natural discrete 3D geometry models

Budd–Németh, *Phases of tree-decorated dynamical triangulations in 3D*, [arXiv:2507.01604](https://arxiv.org/abs/2507.01604), study 3D dynamical triangulations decorated with spanning trees and observe a new triple-tree phase numerically. This model is **different** from the Budd–Castro three-Brownian mated-CRT graph. Do not conflate the ensembles, graph degrees, volume counts, topological properties, or scaling exponents.

A future project could compare their phase transitions, bulk graph distances, branching exponents and diffusion with Budd–Castro/LFPP; a true 3D triangulation has a built-in simplicial topology, but the continuum metric scaling limit may degenerate. No known theorem identifies it with fractional-GFF/GMC geometry.

Other physical 3D quantum gravity constructions—Ponzano–Regge, Turaev–Viro, Chern–Simons formulations of 2+1 gravity and 3D causal dynamical triangulations—are potentially useful conceptual comparisons but **do not** come with an established common law equal to 3D GMC/LFPP. Distinguish 3 spatial Euclidean dimensions from `AdS_3` **2+1-dimensional spacetime** and its 2D conformal boundary (AdS3/CFT2) when discussing holographic Liouville theories.

---

## 8. Connections to the user's other probability/random-geometry reading

These are **adjacent mathematical tools and prospective projects**, not assumptions required for E001.

1. **2D SLE/LQG mating, Tutte/harmonic embeddings and walk convergence.** The original 2D peanosphere relation uses correlated Brownian coordinates with `Corr(L,R)=-cos(πγ²/4)`, and the Tutte embedding of mated-CRT maps converges to LQG in its appropriate embedding/measure sense. This supplies strong 2D simulation benchmarks, not a ready-made 3D embedding.
2. **Spectral convergence for FK/UST-decorated random planar maps and Liouville Brownian motion.** These are examples of proving joint geometry-plus-random-walk results using discrete structure and embeddings. Their 2D conclusions do not automatically give Tutte embedding convergence or 3D diffusion convergence for a new graph.
3. **GFF, Neumann GFF, GMC, Liouville metric and conformal welding.** Good tools for continuum field and measure program P2/P4/P5, but 3D fractional-GFF nonlocality loses the ordinary spatial Markov property of 2D Dirichlet GFF.
4. **SLE/CLE4 nesting and related conformal invariance of dimers/loops.** Illustrate special 2D topological and conformal structures; can motivate a search for intrinsic 3D exploration laws but do not furnish 3D SLE directly.
5. **Large deviations/random matrix/combinatorial analytic methods.** May help with Gaussian tails, graph rare-event estimates, multiscale optimization, cone spectra, finite-size scaling or model selection.
6. **Imaginary Liouville CFT/bootstrap and AdS3/CFT2.** Secondary field-theoretic opportunities, different in aim from the graph-to-LFPP numerical matching. Do not silently expand E001 into an AdS/CFT proof project.

### Available uploaded/retrieved research documents

Some materials referred to in earlier Project conversations may not be automatically present in a local VS Code/GitHub checkout or accessible to Codex. Previously retrievable user Library research PDFs include:

- `MatingOfTrees_MainSurvey.pdf` — Gwynne–Holden–Sun survey; 2D discrete mating, SLE/LQG and random-map continuum methods.
- `2510.16431v1.pdf` — 2D LQG/random planar-map survey material; check the PDF title and bibliography before citing a particular claim.
- `spectral-convergence-critical-fk-ising-planar-maps.pdf` — background for spectral/diffusion convergence if available.
- `linear-clock-random-walk-tree-weighted-planar-maps.pdf` — possible random-walk techniques.
- `LDP-SDP-bipolar-maps.pdf`, `LQG_book.pdf` and other related planar-map references, if locally present.

**Important provenance limitation:** During preparation, the directly searchable file surface did **not** expose a copy of the Budd–Castro paper or the Contreras Hip–Zhuang papers as Project-attached files, so their primary texts were verified through publicly accessible papers linked in this document. Do not claim these PDFs are already downloaded to the repository; acquire the originals under `papers/` when appropriate. ChatGPT Project chats and uploads are **not automatically visible** inside a Codex VS Code workspace; this checked-in context file is the bridge.

---

## 9. Mathematical conjectures, testable hypotheses, and key unresolved questions

Maintain a living table; update each item with strongest supporting/refuting **evidence** and date.

| ID | Question / conjecture | Present status | Best next step |
|---|---|---|---|
| Q01 | Does `G_n^C` have a nondegenerate metric-measure scaling limit? | Open | Prove bounds; stable distance law across `n` |
| Q02 | Is its topology that of a 3-manifold? | Open; graph is nonplanar and has no canonical 3-complex | Define topology/complex first |
| Q03 | Does `d_H(C)` exist for generic SPD `C`? | Numerical evidence only | Distance growth and finite-size stability |
| Q04 | Does equicorr `ρ→-1/2` yield `d_H≈4.1`? | Tentative numeric observation | Fine-resolution near-boundary tests, rigorous bounds |
| Q05 | Is there a canonical curve `C(γ)` matching 3D LFPP? | No known formula | Multiple observables + structural criterion |
| Q06 | Can metric and counting measure jointly match LFPP/GMC? | Unknown | Intrinsic ball-mass and distance distribution checks |
| Q07 | What field is canonically measurable from the Brownian graph? | No accepted 3D analog | Compare encodings, volume log fluctuations, graph spectral fields |
| Q08 | Is the emerging canonical field log-correlated and Gaussian in 3D? | Unproved and possibly false | Define spatial metric/embedding and Gaussianity tests |
| Q09 | Is the graph spectral dimension 2, 3, or parameter dependent? | Open | Return probability, resistance and eigenvalue numerics |
| Q10 | Can one prove FPP/Eden limit with the same or related graph metric? | Open | Edge-weighted vs ordinary distance exponents |
| Q11 | Can graph random walk/diffusion converge on a metric limit? | Open | Effective resistance, heat kernel, Dirichlet forms |
| Q12 | Can DLA have a nontrivial continuum limit and Markov characterization? | Highly open | Diffusion/harmonic-measure estimates first |
| Q13 | Does a 3D QLE-type equation exist without planar conformal uniformization? | Conceptually open | Derive growth from intrinsic graph/Brownian exploration |
| Q14 | Is full high-dimensional LFPP metric uniqueness true? | Not established generally | Axiomatic uniqueness / regularization comparison |
| Q15 | Do discrete fractional-field LFPP exponents equal continuum regularization exponents? | Open for many schemes | Direct comparison, rigorous comparison techniques |
| Q16 | Can 3D interacting Liouville CFT correlations be constructed rigorously? | Formal physics theory; partial mathematical foundations | Zero modes, GMC insertions, conformal covariance |
| Q17 | Is a 2D boundary theory required? | **No; abandoned as mandatory axiom** | Optional trace/gluing research |
| Q18 | Can random 3D triangulations produce the same graph/GMC metric? | Open, different discrete ensembles | Compare discrete phase and intrinsic statistics |

### Candidate theorem statements for rigorous follow-up

- **Distance exponent:** Under specified `C`, prove `log d_G(U_n,V_n)/log n → 1/d_H(C)` in probability and quantify the exponent.
- **Metric tightness:** Prove tightness of rescaled `G_n^C` in a metric-measure topology using explicit covering/bottleneck estimates.
- **Spectral dimension:** Establish `p_{2m}^G(v,v)=m^{-d_s(C)/2+o(1)}` with proper root/infinite-volume conventions.
- **Graph FPP:** Establish exponential weighted distances of order `n^{1/d_FPP(C)}` and compare `d_FPP` with ordinary `d_H`.
- **LFPP discrete universality:** For precisely matched 3D log-field approximation schemes, compare normalized crossing exponents and prove at least bounds invariant under replacement of the field regularization.
- **Metric-measure universality:** Construct a coupling/embedding and show joint convergence `(G_n^C,d_n,μ_n)` to `(D_h,μ_h^γ)` at some candidate relation `C=C(γ)`.

A numerical paper should label all of these as *conjectures or goals* until they are actually proved.

---

## 10. Literature map with stable links (check latest versions before new claims)

### A. Main papers for immediate assignment

1. **Budd–Castro (2023)**, *Scale-invariant random geometry from mating of trees: A numerical study*. [arXiv:2207.05355](https://arxiv.org/abs/2207.05355); [full author PDF](https://hef.ru.nl/~tbudd/docs/treemating.pdf). Precise graph and cone excursion sampler; original simulated 3D dimension estimates; Table 2.
2. **Contreras Hip–Zhuang (2025)**, *Bounds on the distance exponent for higher-dimensional Liouville first passage percolation*. [arXiv:2504.09141](https://arxiv.org/abs/2504.09141), [HTML](https://arxiv.org/html/2504.09141). Bounds on `λ(d,ξ)`, derivative and dimension, Corollary 1.9.
3. **Ding–Gwynne–Zhuang (2023; revised 2026)**, *Tightness of exponential metrics for log-correlated Gaussian fields in arbitrary dimension*. [arXiv:2310.03996](https://arxiv.org/abs/2310.03996). Higher-dimensional exponential metric tightness/topology, subcritical parameter regime.
4. **Contreras Hip–Zhuang (2025)**, *Weak exponential metrics for high-dimensional log-correlated Gaussian fields*. [arXiv:2512.06292](https://arxiv.org/abs/2512.06292), [HTML](https://arxiv.org/html/2512.06292). Field-measurable weak metrics, Weyl scaling, moment/Hölder/KPZ, Hausdorff dimension.
5. **Dello Schiavo–Herry–Kopfer–Sturm (2025 journal)**, *Polyharmonic fields and Liouville quantum gravity measures on tori of arbitrary dimension: from discrete to continuous*. [arXiv:2302.02963](https://arxiv.org/abs/2302.02963). Discrete spectral fractional-field and GMC convergence.

### B. 2D techniques and benchmarks

6. **Duplantier–Miller–Sheffield**, *Liouville quantum gravity as a mating of trees*, [arXiv:1409.7055](https://arxiv.org/abs/1409.7055). Foundational 2D peanosphere/continuum law.
7. **Gwynne–Holden–Sun**, *Mating of trees for random planar maps and Liouville quantum gravity: a survey*, consult `MatingOfTrees_MainSurvey.pdf` and its bibliographic links. 2D discrete models, adjacency and continuum couplings.
8. **Gwynne–Miller–Sheffield**, *The Tutte embedding of the mated-CRT map converges to Liouville quantum gravity*. [arXiv:1705.11161](https://arxiv.org/abs/1705.11161); primary established fact: 2D Tutte-embedded mated-CRT convergence of measure/curve and random walk.
9. **Miller–Sheffield**, *Quantum Loewner Evolution*, [arXiv:1312.5745](https://arxiv.org/abs/1312.5745); and QLE/Brownian-map metric papers. Distinguish QLE metric growth vs DLA scaling-limit conjectures.
10. **Ang–Mithal (2026)**, *Quantum Loewner evolution in quantum natural time: phases and Markov properties*. [arXiv:2605.03385](https://arxiv.org/abs/2605.03385). New 2D constructions, not a 3D method ready to use.

### C. Adjacent discrete/field/gravity research

11. **Budd–Németh (2025)**, *Phases of tree-decorated dynamical triangulations in 3D*. [arXiv:2507.01604](https://arxiv.org/abs/2507.01604). Distinct triangulation model.
12. **Kislev–Levy–Oz (2022)**, *Odd dimensional nonlocal Liouville conformal field theories*. [arXiv:2206.10884](https://arxiv.org/abs/2206.10884). Formal odd-dimensional LCFT, vertex observables, DOZZ analog.
13. **Dello Schiavo–Herry–Kopfer–Sturm (2024)**, *Conformally invariant random fields, Liouville quantum gravity measures, and random Paneitz operators on Riemannian manifolds of even dimension*. [arXiv:2105.13925](https://arxiv.org/abs/2105.13925). Rigorous conformal-field/GMC/Polyakov–Liouville construction in even dimension.
14. **Lodhia–Sheffield–Sun–Watson**, *Fractional Gaussian fields: a survey*, [arXiv:1407.5598](https://arxiv.org/abs/1407.5598). Fractional fields, critical log correlation and operator constructions.
15. **Duplantier–Rhodes–Sheffield–Vargas**, GMC review, [arXiv:1407.5605](https://arxiv.org/abs/1407.5605). GMC background and subcritical/critical caveats.
16. **Ding–Gwynne–Zhuang (2026 publication)**, *Percolation of thick points of the log-correlated Gaussian field in high dimensions*, journal Advances in Mathematics (2026). Higher-dimensional phenomena; sufficiently-high-`d` conclusions should not be transferred uncritically to `d=3`.

**Bibliography maintenance:** Record title, authors, version, publication date, DOI/arXiv, theorem/proposition number and any differences between updated versions. Distinguish sources independently read by an agent from citations inherited from this context. Verify specific theorems and version histories before citing them in a research write-up.

---

## 11. How agents should reason, write, and report

1. **Source fidelity:** Before changing the model, quote the mathematical definition in your own words and refer to its equation/section. A running approximation that is simpler than the original model must be a separate named ensemble.
2. **Four-way claim labels:** `THEOREM (source, hypotheses)`, `PHYSICS PREDICTION (source)`, `NUMERICAL EVIDENCE (configuration, CI)`, `CONJECTURE/HEURISTIC (testable implications)`. Do not silently elevate any category.
3. **Parameter/notation hygiene:** `d_enc=3` number of Brownian coordinates; `d_space=3` spatial dimension for LFPP; `d_H(C)` graph metric exponent; `γ_GMC` scalar coupling; `γ_s` string susceptibility; `theta_angle` third Brownian angle; `ξ_LFPP`; `λ(d,ξ)` crossing exponent; `Q_3=3/γ_GMC+γ_GMC/2`.
4. **Computational reproducibility:** Log actual environment, commit, configuration and seeds. Include unit tests, error bars, bootstrap by independent realizations, finite-size and Brownian path resolution extrapolations, and negative controls. Never invent numerical measurements or claim code has run when it has not.
5. **Definitions before plots:** Every chart must state whether it uses ordinary graph distance, FPP distance, Euclidean distance, Brownian parameter distance, or an assumed embedding. Likewise distinguish vertex counting measure, GMC, and arbitrary weighted vertex measures.
6. **Falsifiability:** Prefer attempting to rule out bad parameter mappings over overfitting matching curves. An inconsistent exponent or topological obstruction is a scientific result.
7. **Research notes:** Maintain `results/e001/summary.md` and `research-notes/questions.md` with exact obstacles, changed hypotheses and links to supporting plots/tests. At end of each task give concise next experiments and mathematical proof ideas.
8. **Autonomy and scope:** Agents may write code, run simulations, generate figures and tables, and write LaTeX summaries in the authorized local workspace. Avoid destructive repo operations, unapproved external services, and uncontrolled expensive parameter sweeps. Do not change historical experimental data to force expected fits.
9. **Intellectual humility:** A 3D mating of trees might be a new universality class rather than 3D LQG. It is acceptable—and valuable—to conclude the models are different or that the evidence is currently inconclusive.

### Suggested opening task prompt for the coding agent

> Read `RESEARCH_CONTEXT.md` and the primary Budd–Castro paper. Implement a tested, reproducible Python package for `d=2` and `d=3` finite mated-CRT graphs, correctly distinguishing unbiased correlated Brownian excursions, the paper's biased cone-excursion sampler, and any provisional approximation. First validate polygon triangulation and the 2D LQG benchmark values. Then simulate 3D graphs at `C=I_3`, the equicorrelated family, and selected isosceles pairs. Measure distance histograms, finite-size Hausdorff exponents, ball masses, and simple graph-derived field observables. Independently implement a unit-log 3D fractional Gaussian field, GMC and LFPP baselines. Compare measured dimensions to rigorous Contreras Hip–Zhuang bounds and compare independent observables, with uncertainty and explicit parameter non-identifiability. Write fully reproducible code, tests, figures, raw result tables and `results/e001/summary.md`. Do not claim a 3D LQG correspondence or a canonical graph GFF until independently justified.

---

## 12. Compact glossary

- **mated-CRT map (2D):** planar random triangulation arising from the mating of two Brownian-encoded CRTs; connected to 2D SLE-decorated LQG.
- **mated-CRT graph (3D encoding):** generally nonplanar graph obtained by gluing **three** scalar-excursion polygon triangulations along their common Hamiltonian cycle. It is not inherently a 3D simplicial manifold.
- **Brownian covariance `C`:** SPD 3×3 unit-diagonal correlation matrix; three independent off-diagonal parameters.
- **`α,β,theta_angle`:** spherical triangle angles satisfying `C_ij=-cos(angle)`; paper's third angle `γ` is **not** GMC `γ`.
- **`γ_GMC`:** coupling of log-correlated Gaussian field to GMC, `0<γ<sqrt6` in dimension 3 under unit-log normalization.
- **`ξ_LFPP`:** path-length exponential parameter; tied to `γ_GMC` through `ξ=γ/d_{γ,3}` only under the corresponding quantum-geometry parameterization.
- **`λ(3,ξ)`:** **LFPP Euclidean-unit-cube crossing exponent**, not Hausdorff dimension.
- **`d_{γ,3}`:** dimension of a 3D weak exponential metric; known only within rigorous bounds for nonzero coupling.
- **`d_H(C)`:** conjectural 3-tree continuum Hausdorff dimension, currently estimated via finite-size graph distances.
- **`γ_s(C)`:** string susceptibility from Brownian-cone spectral exponent, unrelated by any known map to GMC `γ`.
- **GMC:** renormalized exponential of a log-correlated Gaussian distribution; requires an ambient domain and reference measure.
- **Graph GFF:** Gaussian field with covariance graph Laplacian pseudoinverse; a derived object, not automatically the field encoding the graph.
- **LFPP:** first-passage metric from exponential weights of a regularized log field, with a nontrivial renormalization.
- **FPP/Eden:** independent exponential edge passage-time growth; distinct from DLA harmonic-measure growth.
- **DLA:** diffusion-limited aggregation, attachment according to harmonic measure of random walk/diffusion.
- **QLE:** a 2D Loewner/SLE/LQG-based family of growth processes; higher-dimensional intrinsic analogues are conjectural.
- **GHP:** Gromov–Hausdorff–Prokhorov topology for compact metric-measure-space convergence.
- **Critical fractional field in 3D:** covariance given by normalized `(-Δ)^{-3/2}` rather than ordinary 3D GFF `(-Δ)^{-1}`.

**End of current research context. Update this document when hypotheses, literature, implementation conventions or numerical evidence change; preserve exact distinctions between proved, numerical and conjectural claims.**
