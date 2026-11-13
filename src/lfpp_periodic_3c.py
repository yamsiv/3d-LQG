"""Periodic LFPP builder preserving the Phase3A finite quadrature rule.

Trilinear interpolation of exp(xi*h) is linear in its grid values. For each
fixed ray and trapezoid rule, integrate once symbolically into positive
shift coefficients, then evaluate by periodic rolls. This changes summation
roundoff, not the finite path stencil or quadrature. Cube boundaries and
field interpolation explicitly fall back to the original implementation.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import time

import numpy as np
from scipy.sparse import csr_matrix

from .lfpp_3d import build_lfpp_graph,stencil_offsets


@lru_cache(maxsize=512)
def segment_kernel(offset,quadrature_per_grid=2.):
    offset=tuple(int(x)for x in offset)
    length=float(np.linalg.norm(offset))
    if length==1:
        return (((0,0,0),.5),(offset,.5))
    subdivisions=max(1,int(np.ceil(quadrature_per_grid*length)))
    result={}
    direction=np.asarray(offset,float)
    for step in range(subdivisions+1):
        point=(step/subdivisions)*direction
        lower=np.floor(point).astype(int);fraction=point-lower
        factor=(.5 if step in[0,subdivisions]else 1.)/subdivisions
        for bits in product((0,1),repeat=3):
            coefficient=float(factor*np.prod(np.where(bits,fraction,1-fraction)))
            if coefficient:
                shift=tuple(lower+bits)
                result[shift]=result.get(shift,0.)+coefficient
    return tuple(result.items())


@lru_cache(maxsize=2)
def _topology(n,stencil):
    offsets=stencil_offsets(stencil)
    count=n**3;directions=len(offsets)
    dtype=np.int32 if count*directions<np.iinfo(np.int32).max else np.int64
    vertices=np.arange(count,dtype=dtype).reshape((n,)*3)
    neighbors=np.empty((count,directions),dtype=dtype)
    for column,offset in enumerate(offsets):
        neighbors[:,column]=np.roll(vertices,tuple(-offset),axis=(0,1,2)).ravel()
    order=np.argsort(neighbors,axis=1).astype(np.uint8)
    indices=np.take_along_axis(neighbors,order,axis=1).ravel()
    slots=np.argsort(order,axis=1).astype(np.uint8)
    indptr=np.arange(count+1,dtype=dtype)*directions
    for value in[offsets,indices,slots,indptr]:value.setflags(write=False)
    return offsets,indices,slots,indptr


def build_periodic_lfpp_graph(h,xi,*,stencil='26',boundary='torus',
                              quadrature_per_grid=2,interpolation='weight'):
    """Same sparse finite metric, with a cached positive convolution kernel."""
    if boundary!='torus'or interpolation!='weight':
        graph,metadata=build_lfpp_graph(h,xi,stencil=stencil,boundary=boundary,
            quadrature_per_grid=quadrature_per_grid,interpolation=interpolation)
        metadata['periodic_kernel_fallback']='only periodic weight interpolation is accelerated'
        return graph,metadata
    started=time.perf_counter();h=np.asarray(h,dtype=float)
    if h.ndim!=3 or len(set(h.shape))!=1 or h.shape[0]<5:
        raise ValueError('h must be a cubic grid of side N>=5')
    if not np.all(np.isfinite(h))or not np.isfinite(xi):
        raise ValueError('Field and xi must be finite')
    if not np.isfinite(quadrature_per_grid)or quadrature_per_grid<=0:
        raise ValueError('quadrature_per_grid must be positive')
    with np.errstate(over='raise',under='raise',invalid='raise'):weight=np.exp(float(xi)*h)
    n=h.shape[0];offsets,indices,slots,indptr=_topology(n,stencil)
    data=np.empty((n**3,len(offsets)),dtype=float)
    vertices=np.arange(n**3)
    reverse={tuple(offset):i for i,offset in enumerate(offsets)}
    maximum_kernel_sum_defect=0.
    for column,offset in enumerate(offsets):
        if offset[np.flatnonzero(offset)[0]]<0:continue
        kernel=segment_kernel(tuple(offset),float(quadrature_per_grid))
        integral=np.zeros_like(weight)
        for shift,coefficient in kernel:
            integral+=coefficient*np.roll(weight,tuple(-np.asarray(shift)),axis=(0,1,2))
        maximum_kernel_sum_defect=max(maximum_kernel_sum_defect,abs(sum(c for _,c in kernel)-1))
        cost=float(np.linalg.norm(offset))/n*integral
        data[vertices,slots[:,column]]=cost.ravel()
        # Compute each undirected segment once, exactly matching the original
        # builder's positive-orientation/reverse assignment convention.
        back=reverse[tuple(-offset)]
        data[vertices,slots[:,back]]=np.roll(cost,tuple(offset),axis=(0,1,2)).ravel()
    graph=csr_matrix((data.ravel(),indices,indptr),shape=(n**3,n**3),copy=False)
    graph.has_sorted_indices=True
    graph.lfpp_boundary='torus';graph.lfpp_grid_side=n
    metadata={'grid_side':n,'n_vertices':n**3,'xi':float(xi),'boundary':'torus',
        'stencil':stencil,'directions':offsets.tolist(),'directed_edges':graph.nnz,
        'mesh':1/n,'cube_span':(n-1)/n,
        'edge_rule':'same composite-trapezoid integral of trilinear exp(xi*h) interpolant as Phase3A',
        'interpolation':'weight','quadrature_per_grid':float(quadrature_per_grid),
        'axis_rule':'endpoint trapezoid','gmc_variance_subtraction':False,
        'xi_zero_metric':'L1'if stencil=='axis'else'polyhedral directional-stencil norm',
        'construction_seconds':time.perf_counter()-started,
        'csr_bytes':int(graph.data.nbytes+graph.indices.nbytes+graph.indptr.nbytes),
        'builder':'phase3c_periodic_positive_kernel',
        'kernel_coefficient_sum_max_defect':maximum_kernel_sum_defect,
        'numerical_difference':'reordered positive floating-point sums only; same finite ray/quadrature metric',
        'limitations':['Fixed stencil anisotropy persists under spatial refinement',
                       'Finite-volume torus is different from an open cube',
                       'No claim of an exact isotropic continuum metric']}
    return graph,metadata
