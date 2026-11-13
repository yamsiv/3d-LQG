# A correlated tractable benchmark: Weyl A2 times a half-line

`src/weyl_excursions.py` samples exact excursion **grid values** for
`C12=-1/2, C13=C23=0` in dimension three, with `C12=-1/2` alone in
dimension two. It provides another correlated validation case independently of
the general-covariance MCMC approximation. Its cell minima remain approximate.

The Brownian-cone/graph definitions and biased-law construction remain those
of Budd–Castro, [arXiv:2207.05355v1](https://arxiv.org/abs/2207.05355v1),
§§2.5, 3.1–3.4. The random-matrix ingredient uses the equivalence of Hermitian
matrix eigenvalue diffusion, noncolliding Brownian motion, and the Vandermonde
Doob transform in a Weyl chamber; see Katori–Tanemura,
[*Noncolliding Brownian Motion and Determinantal Processes*](https://arxiv.org/abs/0705.2460).
The explicit normalization and bridge specialization below are our analytical
implementation derivation, checked independently by the validation workstream.

## Eigenvalue gaps and covariance

Let H be 3x3 Hermitian Brownian motion: diagonal real entries have variance dt,
and each independent real/imaginary offdiagonal component has variance dt/2.
Subtract the trace and write its ordered eigenvalues as `lambda1<lambda2<lambda3`.
Set `X1=(lambda2-lambda1)/sqrt(2)` and
`X2=(lambda3-lambda2)/sqrt(2)`. The eigenvalue martingale covariance is the
identity before trace subtraction. Consequently the gap covariance has unit
diagonal and correlation -1/2. The trace coordinate cancels from the gaps.

The whitened radial identity is

`(X1,X2)^T C_pair^-1 (X1,X2) = lambda1^2+lambda2^2+lambda3^2`

for traceless eigenvalues. Ordered noncollision corresponds to the wedge of
opening pi/3. Its positive harmonic polynomial is the Vandermonde product,
of degree p=3. The independent positive third coordinate contributes another
degree-one factor. Thus:

| Dimension | Cone | p | alpha1 | lambda_sphere | gamma_s |
| --- | --- | --- | --- | --- | --- |
| 2 | A2 chamber | 3 | 3 | 9 | -2 |
| 3 | A2 chamber times R_+ | 4 | 4.5 | 20 | -3.5 |

These agree with the paper's birectangular formula (29) at alpha=pi/3.

## Unbiased and biased bridges

For the unbiased excursion, take eigenvalues of a Hermitian Gaussian matrix
bridge from zero to zero. The eigenvalue process is the noncolliding bridge;
the Doob Vandermonde factors cancel when conditioning on the endpoints.
In dimension three append an independent Bessel(3) bridge from zero to zero.

For the biased excursion, generate a traceless GUE matrix with density
proportional to `exp(-Tr(H^2)/2)` and take its eigenvalues. Their density in
the traceless eigenvalue plane is Gaussian times Vandermonde squared. In three
dimensions, independently generate `y=norm(N(0,I3))`. Divide eigenvalues and y
by `sqrt(sum(lambda_i^2)+y^2)` (omit y in dimension two). In radial/angular
coordinates this removes the chi_(d+2p) radial variable and yields the desired
`m1(S)^2` spherical density.

Draw the paper's `T1,T2~InvGamma(alpha1,1/2)`, set `U=T1/(T1+T2)`, and
divide the normalized eigenvalues and height by `sqrt(T1+T2)`. Sample independent
Hermitian bridges zero to the resulting diagonal eigenvalue matrix over [0,U],
and back to zero over [U,1]. A fixed matrix endpoint is valid here: starting at
zero makes the endpoint eigenvectors Haar and independent of the entire
eigenvalue path by conjugation invariance. Conditioning those eigenvectors to
the identity therefore does not change that path law conditional on its
endpoint eigenvalues. The positive third coordinate uses the corresponding two
Bessel(3) bridges. Insert U into the regular grid before sampling.

## What is exact and what requires refinement

All retained grid values have the exact continuous excursion law, unbiased or
paper-biased as requested, up to numerical eigensolver/roundoff error. There is
no MCMC or burn-in. There is currently no exact joint eigenvalue-gap minima
sampler here. `segment_minima=None` causes the graph constructor to use
polygonal minima; symbolic cell-boundary ties are counted in graph metadata.
Distance studies require increasing oversampling and comparing the changes
with independent-graph statistical errors. This sampler must not inherit the
independent-coordinate branch's exact-minima claim.

For a coupled grid sensitivity experiment, generate one finest-resolution
sample, then call `thin_weyl_sample(fine,n,oversample)` for nested coarser grids.
It preserves the exact random junction and yields **the same realization** at
different path resolutions. This pairing isolates discretization changes; the
coarse/fine pair is not two independent graph samples.

```python
import numpy as np
from src.weyl_excursions import sample_weyl_excursion, thin_weyl_sample
from src.mated_crt import graph_from_excursion

fine = sample_weyl_excursion(1024, 3, np.random.default_rng(20261110),
                              law="biased", oversample=128)
coarse = thin_weyl_sample(fine, 1024, 32)
fine_graph = graph_from_excursion(fine, 1024)
coarse_graph = graph_from_excursion(coarse, 1024)
```

Time and storage scale linearly with the number of retained path samples;
the eigensolvers are fixed 3x3 size. Whole-array complex matrices are currently
constructed, so path oversampling dominates peak memory. Use moderate graph
sizes for refinement tests before scaling those factors to larger graphs.
