"""Independent-panel diagnostics; preserve the primary ensemble normalization.

The old plug-in bootstrap of the energy U statistic is upward biased when a
resampled physical panel is repeated. Its percentile interval is not a null
confidence interval. Here the ensemble-scaled energy is descriptive only.
A separate parent-normalized shape statistic admits an ordinary label
permutation test: its normalization is a fixed function of each parent.
This sensitivity removes random global size and never replaces primary J.
"""
import numpy as np
from src.family_search_3c import sampled_matrix_panels, signed_matrix_energy


def matrix_diagnostic(left, right, rng, *, permutations=99, gauge='grounded'):
    a=sampled_matrix_panels(left,gauge=gauge)['vectors']
    b=sampled_matrix_panels(right,gauge=gauge)['vectors']
    raw=signed_matrix_energy(a,b)
    # Population point-label exchangeability is not an assumption of independent
    # pair entries. An entire 120-coordinate panel is one statistical unit.
    a=a/a.mean(axis=1,keepdims=True);b=b/b.mean(axis=1,keepdims=True)
    observed=signed_matrix_energy(a,b)['energy_u']
    pool=np.vstack([a,b]);null=[]
    for _ in range(permutations):
        ix=rng.permutation(len(pool));null.append(signed_matrix_energy(pool[ix[:len(a)]],pool[ix[len(a):]])['energy_u'])
    return dict(ensemble_scaled_energy=raw['energy_u'],parent_shape_energy=observed,
        shape_permutation_p=(1+np.count_nonzero(np.asarray(null)>=observed-1e-12))/(1+permutations),
        permutation_energies=null,permutations=permutations,parent_counts=[len(a),len(b)],
        note='Whole-parent permutation of separately parent-normalized panels; unadjusted secondary shape sensitivity. Ensemble energy has no percentile-bootstrap CI.')
