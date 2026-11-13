import numpy as np
import pytest

from src.lfpp_3d import build_lfpp_graph,root_distances,stencil_offsets
from src.lfpp_periodic_3c import build_periodic_lfpp_graph,segment_kernel


@pytest.mark.parametrize('stencil',['26','primitive2'])
@pytest.mark.parametrize('xi',[0.,.2,.3])
def test_periodic_kernel_same_edges_and_shortest_paths(stencil,xi):
    n=7
    x,y,z=np.meshgrid(np.arange(n)/n,np.arange(n)/n,np.arange(n)/n,indexing='ij')
    h=2*np.sin(2*np.pi*x)+np.cos(4*np.pi*y)-.7*np.sin(2*np.pi*(x+y+z))
    reference,_=build_lfpp_graph(h,xi,stencil=stencil)
    fast,metadata=build_periodic_lfpp_graph(h,xi,stencil=stencil)
    np.testing.assert_array_equal(fast.indptr,reference.indptr)
    np.testing.assert_array_equal(fast.indices,reference.indices)
    np.testing.assert_allclose(fast.data,reference.data,rtol=3e-13,atol=0)
    assert (fast-fast.T).nnz==0
    roots=[0,3,17,n**3-1]
    np.testing.assert_allclose(root_distances(fast,roots),root_distances(reference,roots),rtol=3e-13,atol=2e-14)
    assert metadata['kernel_coefficient_sum_max_defect']<2e-15


@pytest.mark.parametrize('boundary,interpolation',[('cube','weight'),('torus','field')])
def test_unsupported_cases_use_exact_reference_fallback(boundary,interpolation):
    h=np.arange(6**3).reshape((6,)*3)/100.
    reference,_=build_lfpp_graph(h,.2,boundary=boundary,interpolation=interpolation)
    actual,metadata=build_periodic_lfpp_graph(h,.2,boundary=boundary,interpolation=interpolation)
    np.testing.assert_array_equal(actual.indptr,reference.indptr)
    np.testing.assert_array_equal(actual.indices,reference.indices)
    np.testing.assert_array_equal(actual.data,reference.data)
    assert 'periodic_kernel_fallback'in metadata


def test_positive_kernel_and_nondefault_quadrature():
    for offset in stencil_offsets('primitive2'):
        for resolution in[1.,2.,3.]:
            coefficients=[c for _,c in segment_kernel(tuple(offset),resolution)]
            assert min(coefficients)>0
            assert sum(coefficients)==pytest.approx(1.,abs=2e-15)
    h=np.sin(np.arange(6**3).reshape((6,)*3))
    original,_=build_lfpp_graph(h,.3,stencil='primitive2',quadrature_per_grid=3)
    fast,_=build_periodic_lfpp_graph(h,.3,stencil='primitive2',quadrature_per_grid=3)
    np.testing.assert_array_equal(original.indices,fast.indices)
    np.testing.assert_allclose(original.data,fast.data,rtol=3e-13,atol=0)
