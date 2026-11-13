from itertools import permutations
import numpy as np
from src.covariance_design_3b import canonical_triple,design_covariances


def test_canonical_labels_represent_vertex_permutations():
    triple=(-.2,.1,.35)
    assert len({canonical_triple(x)for x in permutations(triple)})==1


def test_space_filling_design_is_spd_nonredundant_and_reproducible():
    a=design_covariances();b=design_covariances()
    assert a==b and len(a)>=25
    assert len({tuple(x['triple'])for x in a})==len(a)
    assert all(np.linalg.eigvalsh(x['covariance']).min()>=.15 for x in a)
    assert all(x['condition']<=12 for x in a)
    assert any('asymmetric'in x['family']for x in a)
