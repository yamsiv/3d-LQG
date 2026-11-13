# Intrinsic metric and statistical conventions

The reference is Budd–Castro, arXiv:2207.05355v1, §§3.1 and 4.2. All observables
here concern the finite graph's unweighted shortest-path metric and vertex
counting measure. Parallel unit edges have the same shortest-path distances as
the underlying simple graph. Multiplicities remain in the generated graph for
observables that need conductances, but the reported degree statistics concern
the simple graph.

## Uniform pairs and graph realizations

For each independent graph, sample `k` roots independently and uniformly with
replacement. Run a complete shortest-path search from each root, retaining
distances to all `n` vertices, including the root. This samples the conditional
law of two independent uniform vertices without a boundary-root or degree bias.
The atom at distance zero is exactly `1/n`. A distance histogram is stored for
every root; a graph record also retains its pooled histogram and root identities.

Normalize each graph histogram before averaging graphs at a fixed size. This
gives equal graph weights even if root counts differ. The mean, median, 25th and
75th percentiles are statistics of the resulting uniform-pair distance law.
They are not averages of graphwise medians. Roots within one graph and all their
targets are correlated. Only independent graph realizations are resampled in
the uncertainty calculations.

For graph-averaged distances, the sample variance of root means estimates the
within-graph root variability. Its value divided by `k` estimates the variance
from finite root sampling in that graph's average. The observed variance across
graph averages also contains this root-sampling contribution. Subtracting its
mean, and truncating at zero, gives a diagnostic estimate of the between-graph
variance of the true conditional mean. The raw quantities are saved separately.
The bootstrap resamples entire graph records and therefore includes both sources
of Monte Carlo variability actually present in the experiment.

## Finite-size slopes

`bootstrap_scaling` fits `log(observable) = intercept + q log(n)` by ordinary
least squares across sizes, and reports the effective dimension `1/q`. Within
each size it independently resamples graph records with replacement and repeats
the complete fit. The 2.5th and 97.5th percentiles form a Monte Carlo interval.
These intervals do not cover bias from a finite size range, the excursion
approximation, model misspecification, or an invalid asymptotic scaling ansatz.
The fit is repeated after omitting successively smaller graph sizes, always
retaining at least three sizes. Quantile fits are more sensitive to integer
distance resolution than mean fits.

An additive lattice-distance correction can dominate this regression over the
available range. `fixed_offset_sensitivity` reports fits of `log(mean+s)` for a
preselected range of offsets, without choosing an offset for agreement with a
published exponent. `finite_size_corrections.py` provides a separate fitted
offset diagnostic. Neither is an asymptotic theorem or the paper's complete
correction fit.

## Paper-style shifted histogram collapse

For each smaller size, fit the scale `k_n` and shift `s_n` to minimize the
integrated squared difference between the largest-size reference histogram and

`rho_n((x+s_n)/k_n-s_n)/k_n`.

Histograms are linearly interpolated. Average the first-pass shifts, then refit
each scale using this common shift, following the two passes in §4.2. Regress
`log(k_n)` on `log(n)` including the reference scale `k=1` to obtain `-1/q`.
The reference histogram is resampled in each graph-cluster bootstrap, so its
uncertainty is propagated. Independent fit-window diagnostics test sensitivity
to removing smaller sizes. The final nonlinear finite-size correction ansatz in
the paper's Eq. (45) is **not fitted**: this preliminary study has much fewer
graphs than the source's production study, and additional correction parameters
would not have reliable identifiability. Therefore the collapse result is a
diagnostic effective dimension, with Monte Carlo uncertainty and residual
finite-size error, not a precision replication.

The source text describes 80,000 graphs at `n=2^11,...,2^19`, whereas its Figure 12
caption describes 10,000 independent graphs and ten roots per graph. This study
records its actual sample counts explicitly rather than resolving that mismatch
by assumption. It uses substantially fewer realizations.

## Diameter and intrinsic balls

For a pivot `p`, its eccentricity `e(p)=max_v d(p,v)` is a lower bound on the
diameter. For several pivots, a certified upper bound is

`max_v min_p [d(p,v)+e(p)]`.

The implementation reports these bounds, includes farthest-point sweeps, and
labels the result exact only when both bounds coincide (or all vertex searches
have been performed). Mean lower and upper bounds can be plotted versus size,
but slopes of these bounds do not themselves bound the limiting Hausdorff
dimension. Small graphs can have exact all-pairs diameters.

For each root, cumulative distance histograms give the intrinsic vertex ball
volume `V(r)`. A diagnostic log slope uses radii `r>=2` with mean ball volume
between 2% and 25% of total vertices, requiring at least three usable radii.
This window avoids immediate-neighbour effects and most volume saturation but
does not guarantee an asymptotic power-law regime. Its endpoints and failures
are saved. Finite compact graphs can have no usable intermediate window, and
small-radius apparent ball slopes need not agree with finite-size pair slopes.

## Complexity and checks

Pure Python breadth-first search takes `O(n+m)` time and `O(n)` auxiliary memory.
For sparse matrices the implementation uses SciPy's compiled unweighted
Dijkstra search; its general complexity is `O((n+m) log n)` per root. Construction
of CSR is performed once per metric sample. With a fixed number of roots and
pivots, metric sampling therefore avoids all-pairs distance storage.

Tests check hand-computed cycle and complete-graph distances, connectedness,
exact and bounded diameters against an independent Floyd–Warshall calculation,
CSR/list equivalence, equal graph weights, known synthetic power-law regression,
and a known histogram-collapse scaling law.

The general-covariance finite-grid MCMC pilot is explicitly held behind a closed
validation gate. It records finite-graph metrics to diagnose the approximation;
it does not fit or report a covariance-dependent continuum Hausdorff dimension.
