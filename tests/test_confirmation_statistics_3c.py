import numpy as np
from src.confirmation_statistics_3c import matrix_diagnostic


def records(scales):
    m=np.abs(np.arange(16)[:,None]-np.arange(16)[None,:])+0.
    return [dict(matrix=m*s,root_means=np.ones(16)*s*5.5)for s in scales]


def test_parent_shape_removes_global_size_without_changing_ensemble_diagnostic():
    r=matrix_diagnostic(records([1,2,4,8]),records([2,2,2,2]),np.random.default_rng(2),permutations=19)
    assert abs(r['parent_shape_energy'])<1e-12
    assert r['shape_permutation_p']==1.
    assert r['ensemble_scaled_energy']>0


def test_identity_null_keeps_signed_value_and_cluster_count():
    r=matrix_diagnostic(records([1,2,3]),records([1,2,3]),np.random.default_rng(3),permutations=9)
    assert r['parent_counts']==[3,3]
    assert r['ensemble_scaled_energy']<0
    assert len(r['permutation_energies'])==9
