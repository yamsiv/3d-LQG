"""A tractable correlated cone: A2 Weyl chamber, optionally times R_+.

Ordered eigenvalue gaps of Hermitian 3x3 Gaussian matrix bridges provide exact
grid values for covariance C12=-1/2, C13=C23=0 (paper alpha=pi/3,
beta=theta=pi/2). Cell minima are polygonal approximations, unlike the exact
independent-coordinate sampler. Oversampling sensitivity must be measured.
"""
from __future__ import annotations
import numpy as np
from .brownian_excursions import ExcursionSample, _vector_bridge, _bessel_bridge


def weyl_covariance(dimension=3):
    if dimension not in (2,3):
        raise ValueError("Weyl sampler supports dimension two or three")
    c=np.eye(dimension)
    c[0,1]=c[1,0]=-.5
    return c


def _hermitian_from_coordinates(coordinates):
    """Nine independent standard normals -> isotropic Hermitian Gaussian.

    Diagonal entries have variance one; offdiagonal real/imaginary parts have
    variance one half, giving density proportional exp(-Tr(H^2)/2).
    """
    z=np.asarray(coordinates)
    matrix=np.zeros(z.shape[:-1]+(3,3),dtype=complex)
    matrix[...,np.arange(3),np.arange(3)]=z[...,:3]
    for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
        entry=(z[...,3+2*k]+1j*z[...,4+2*k])/np.sqrt(2)
        matrix[...,i,j]=entry
        matrix[...,j,i]=np.conjugate(entry)
    return matrix


def _traceless_eigenvalues(matrices):
    values=np.linalg.eigvalsh(matrices)
    return values-values.mean(axis=-1,keepdims=True)


def _eigenvalue_bridge(times,start,end,rng):
    start_z=np.r_[start,np.zeros(6)]
    end_z=np.r_[end,np.zeros(6)]
    coordinates=_vector_bridge(times,start_z,end_z,rng)
    return _traceless_eigenvalues(_hermitian_from_coordinates(coordinates))


def weyl_junction_direction(dimension,rng):
    """Draw S~m1² on the whitened sphere, returned as eigenvalues and height.

    Traceless GUE eigenvalue density is Vandermonde² times a 2D Gaussian.
    The additional positive coordinate has chi3 law (its m1 factor is y).
    Normalization removes the independent radial chi_(d+2p) variable.
    """
    if dimension not in (2,3):
        raise ValueError("dimension must be two or three")
    eigenvalues=_traceless_eigenvalues(_hermitian_from_coordinates(rng.normal(size=9)))
    height=np.linalg.norm(rng.normal(size=3)) if dimension==3 else 0.
    radius=np.sqrt(np.sum(eigenvalues**2)+height**2)
    return eigenvalues/radius,float(height/radius)


def sample_weyl_excursion(n_steps,dimension=3,rng=None,law="biased",oversample=32):
    """Exact Weyl-cone excursion values, polygonal minima approximation.

    d=2: gaps of three noncolliding bridges, wedge angle pi/3, alpha1=3.
    d=3: that cone times an independent positive coordinate, alpha1=4.5.
    `law` is unbiased or paper-biased. Return is compatible with
    `mated_crt.graph_from_excursion`; segment_minima=None explicitly records
    the absence of an exact joint gap-minima sampler.
    """
    if int(n_steps)!=n_steps or n_steps<3 or int(oversample)!=oversample or oversample<1:
        raise ValueError("n_steps>=3 and oversample>=1 must be integers")
    if law not in {"biased","unbiased"}:
        raise ValueError("law must be biased or unbiased")
    c=weyl_covariance(dimension)
    rng=np.random.default_rng() if rng is None else rng
    times=np.linspace(0,1,n_steps*oversample+1)
    p=3 if dimension==2 else 4
    alpha1=p+dimension/2-1
    meta={"sampler":"weyl_a2_hermitian_bridge_polygonal_minima",
          "law":law,"covariance":c.tolist(),"oversample":int(oversample),
          "grid_values_exact":True,"cell_minima_exact":False,"mcmc":False,
          "alpha1":float(alpha1),"homogeneity_p":p,"gamma_s":float(1-alpha1),
          "lambda1":float(p*(p+dimension-2)),
          "limitations":["Exact grid values, polygonal gap minima only",
                         "Refine oversampling and inspect symbolic boundary ties"]}
    zero=np.zeros(3)
    if law=="biased":
        eigen_direction,height_direction=weyl_junction_direction(dimension,rng)
        hitting=.5/rng.gamma(alpha1,size=2)
        junction=hitting[0]/hitting.sum()
        times=np.unique(np.r_[times,junction])
        mid=int(np.searchsorted(times,junction))
        endpoint=eigen_direction/np.sqrt(hitting.sum())
        eigenvalues=np.empty((len(times),3))
        eigenvalues[:mid+1]=_eigenvalue_bridge(times[:mid+1],zero,endpoint,rng)
        eigenvalues[mid:]=_eigenvalue_bridge(times[mid:],endpoint,zero,rng)
        if dimension==3:
            endpoint_height=height_direction/np.sqrt(hitting.sum())
            height=np.empty(len(times))
            height[:mid+1]=_bessel_bridge(times[:mid+1],0,endpoint_height,rng)
            height[mid:]=_bessel_bridge(times[mid:],endpoint_height,0,rng)
        gap_direction=np.diff(eigen_direction)/np.sqrt(2)
        meta.update({"hitting_times":hitting.tolist(),"junction_time":float(junction),
                     "junction_eigenvalues":eigen_direction.tolist(),
                     "junction_extra_height":height_direction,
                     "junction_direction_correlated":np.r_[gap_direction,height_direction][:dimension].tolist()})
    else:
        eigenvalues=_eigenvalue_bridge(times,zero,zero,rng)
        if dimension==3:
            height=_bessel_bridge(times,0,0,rng)
    values=np.diff(eigenvalues,axis=1)/np.sqrt(2)
    if dimension==3:
        values=np.column_stack([values,height])
    values[[0,-1]]=0
    return ExcursionSample(times,values,None,meta)


def thin_weyl_sample(sample,n_steps,oversample):
    """Retain a nested coarser grid and the biased junction, for paired tests.

    The requested regular grid must be a subset of the existing one. Returns
    exact coarser grid values with fresh polygonal interpolation/minima, not a
    new independent excursion; retain realization identifiers in experiments.
    """
    requested=np.linspace(0,1,n_steps*oversample+1)
    if "junction_time" in sample.metadata:
        requested=np.unique(np.r_[requested,sample.metadata["junction_time"]])
    indices=np.searchsorted(sample.times,requested)
    indices=np.minimum(indices,len(sample.times)-1)
    previous=np.maximum(indices-1,0)
    use_previous=np.abs(sample.times[previous]-requested)<np.abs(sample.times[indices]-requested)
    indices=np.where(use_previous,previous,indices)
    if not np.allclose(sample.times[indices],requested,rtol=0,atol=1e-13):
        raise ValueError("Coarse grid is not nested in source grid")
    metadata=dict(sample.metadata)
    metadata.update({"oversample":int(oversample),
                     "coupled_source_oversample":sample.metadata["oversample"]})
    return ExcursionSample(sample.times[indices].copy(),sample.values[indices].copy(),None,metadata)
