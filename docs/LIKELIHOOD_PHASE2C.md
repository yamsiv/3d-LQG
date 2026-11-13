# Interior-ball likelihood audit, Task 2C

This separates an exact change-of-kernel identity from numerical observations.
It does not certify arbitrary-covariance excursions or Brownian joint minima.
The relevant original construction is Budd–Castro,
[arXiv:2207.05355v1, §3.4, equations41–43](https://arxiv.org/abs/2207.05355).

## Exact telescoping identity

Let h>0 be the true inward cone harmonic, and let h_tilde>0 be another
function. The radius r(x) is a deterministic choice giving a strict interior
ball. Write mu_x for ordinary Brownian exit measure on its sphere; because
the motion starts at its center, this is uniform sphere measure. Define

```
P(x,dy) = h(y)/h(x) mu_x(dy),
Z(x) = integral h_tilde(y) mu_x(dy),
Q(x,dy) = h_tilde(y)/Z(x) mu_x(dy),
R(x) = Z(x)/h_tilde(x)-1,
e(x) = h_tilde(x)/h(x).
```

The first expression is a probability kernel by harmonicity of h. The
locally normalized second expression is a probability kernel even when
h_tilde is not harmonic. For a common finite chain x0,...,xN,

```
dQ_path/dP_path = e(xN)/e(x0) * product[j=0,...,N-1] (1+R(xj))^(-1),
log(dQ_path/dP_path) = log e(xN)-log e(x0)-sum log(1+R(xj)).
```

The intermediate relative errors telescope. Multiplying a separate pointwise
error factor at every step is therefore unnecessarily pessimistic. However,
the accumulated mean-value defects remain. The same identity applies to
common stopping rules when the stopped path laws are well defined and the
likelihood is integrable; a finite deterministic cap avoids optional-stopping
ambiguities. A stopped chain must not be silently conditioned on successful
completion of a computational budget.

If independent analysis establishes |e−1|≤epsilon<1 at both endpoints and
sum|log(1+R)|≤eta **uniformly over the relevant paths**, then

```
abs(log(dQ_path/dP_path)) <= log((1+epsilon)/(1-epsilon)) + eta.
```

`src/harmonic_likelihood.py` implements this conditional algebra and stable
log-sphere quadrature. It never infers a uniform bound from sampled centers.
There are four tests, including direct multiplication in an independent
finite Markov chain and exact harmonic/nonharmonic mean-value identities.

## Further terms for a complete excursion

The exit-chain ratio alone omits the initial angular density, which in the
paper is m1² divided by its spherical integral. Approximate angular modes
change that normalized distribution. Two conditional legs contribute two
chain likelihoods. Terminal path/timing approximations and Brownian paths
inside balls need their own analysis. If those conditional inside-ball laws
were exact and identical, endpoint reweighting would account for the whole
stopped path; bounded walks, approximate clocks, omitted tails and polygonal
minima introduce separate errors.

The true h in dimension3 is r^(−p−1)m, p=alpha1−1/2. A spectral error
Delta p changes the relative radial factor by exp(−Delta p log r).
Consequently even excellent angular error at unit radius does not give a
uniform endpoint bound near the origin: the cutoff contributes a factor
involving |Delta p| |log cutoff|. A possible large-radius excursion must
also be controlled; conditioning on a hard radius cap changes the ensemble.
An inverse-gamma tail time derived for an exact harmonic must not be assumed
exact for an arbitrary locally normalized numerical kernel.

## Independent audit of the positive-factor Ritz construction

The stored factor has homogeneous degree beta=sum_j(pi/omega_j)−3. Its
angular derivative is nu*cot(nu*theta)−cot(theta)+cot(omega−theta), with the
small-angle difference evaluated by its convergent expansion. Multiplication
by a degreeD homogeneous polynomial gives total degree beta+D, so subtracting
(beta+D)z from the log-gradient gives the correct spherical tangent gradient.
The Duffy quadrature's radial-projection Jacobian |det R^-1|/|R^-1 t|³ is
consistent. These identities were checked independently against the source.

The rational Bernstein certificate proves that the polynomial represented by
the **stored binary floating-point coefficients**, interpreted as exact
rationals, is positive on the entire closed coordinate simplex. Together
with the positive analytic factor, this proves positivity of that real-valued
formula in the cone interior. It does not certify every rounded evaluation,
the exact eigenfunction's error, quadrature/assembly error, harmonicity, or a
path likelihood bound. The new certificate properly states this scope.

## Observed-chain diagnostics

The experiment reuses the first two successful recorded analytic-h paths in
each of I3, A2×R+, A3, with both legs: six paths, twelve legs, 5,467 balls.
It does not rerun any Brownian simulation. The stored full paths are still
provisional continuous-path approximations; the outer exit kernels use the
analytic harmonics. The candidate Ritz modes are degree0 for I3/A2 and
degree1 for A3, with quadrature48 during variational fitting. Those basis
choices contain the analytic controls.

Sphere quadrature orders8 and16 evaluate log(1+R) at every recorded ball.
At order16 the largest observed one-step defect is 3.20e-14, the largest
absolute accumulated sum is 1.76e-12, and the likelihood log ratios lie
approximately between −1.39e-13 and +5.60e-14. Order8 versus16 changes one
leg's accumulated result by up to 2.45e-7, demonstrating that quadrature
resolution must be checked even for exact harmonics. These tiny order16
values validate the diagnostic numerics on analytic controls; they do not
bound rare paths, the generic kernel, Brownian clocks or graph minima.

## Generic spatial checks and rejected cost of a naive chain

Four candidate fits use rho=.2 and unequal(a,b,c)=(-.2,.1,.35), degrees6/10,
quadrature80. Diagnostics cover one interior direction, all three edge and
corner rays at epsilon=1e-2,1e-4,1e-6,1e-8, ball fractions .25,.5,.8, and
sphere quadrature8/16 with an independent order32 supplement. This is an
observed finite spatial grid, not a supremum over the cone.

At sphere order32 the maximum observed |R| at ball fraction .5 decreases
from 2.32e-5 to 4.94e-6 for rho=.2 (degree6 to10), and from 1.04e-4 to
4.64e-5 for the unequal case. At fraction .8, degree10 maxima are 1.26e-5
and 1.19e-4. Order16 versus32 changes individual log means by up to 1.66e-7
at the largest fraction, versus 1.48e-10 at .25 and 1.04e-10 at .5. These
observations do not certify the quadrature remainder; larger balls are more
demanding even with good angular functions. Tiny |R| at these selected balls
does not give a uniform accumulated sum over a complete random leg.

A rigorous *analytic-formula* rejection envelope can be constructed using
the Bernstein coefficient bounds Bmin,Bmax. Put eta=r/min d_i(x),
phi_j=sin(nu_j theta)/(sin theta sin(omega_j−theta)), and
phi_max≤nu/min(sin omega,sin(omega/2)). The product-factor ratio is bounded
by (1+eta)^3 times the radial corner bounds
(1+eta)^max(nu−2,0)(1−eta)^min(nu−2,0) and phi_max/phi_j(x).
The polynomial ratio is at most (1+eta)^D Bmax/Bmin. Finally
|y|^(−E)/|x|^(−E)≤(1−r/|x|)^(−E), with E=p+1+beta+D.
Floating evaluation of this envelope would require outward rounding for a
machine-verified rejection implementation.

At the interior test point, eta=.5 gives conservative envelope ratios
4.42e6 and 2.29e9 for rho=.2 degrees6/10, and 4.47e6 and 1.19e9 for unequal
degrees6/10. With observed normalizers close to h_tilde(x), that envelope
would require millions to billions of proposals for one exit, before
hundreds of exits per leg. No generic chain is generated using this bound;
an empirical envelope is not substituted. This is a limitation of this
conservative bound, not proof that efficient rejection is impossible.

Useful next steps include local Bernstein bounds on the ball's coordinate
box/simplex, interval bounds for the geometric factors, or a separately
validated angular inversion. These are specific algorithmic improvements,
distinct from an uninformative global refinement campaign. Even a fast
exactly normalized numerical kernel would still need the likelihood and
path/minimum validations above.

## Reproduction and decision

Outputs are in `results/phase2c/likelihood/`: twelve analytic leg checkpoints,
four generic grids, an order32 supplement and provenance. Existing checkpoints
are preserved. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase2c_likelihood --output results/reproduction/phase2c_likelihood
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_harmonic_likelihood.py
```

The conditional likelihood identity is verified. The relative angular and
harmonicity evidence improves substantially over P1. A uniform accumulated
error budget for general correlated excursions is still missing; the generic
sampler remains provisional, and no parameter sweep is justified by these
diagnostics alone.
