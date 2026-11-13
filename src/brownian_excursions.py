"""Excursion samplers with explicit law and approximation metadata.

Independent orthants: exact values AND segment minima, including paper's biased
law, by Bessel(3) bridges. General SPD: multiscale elliptical-slice MCMC for a
Gaussian bridge conditioned positive at a finite grid. The latter is a distinct
explicit approximation without a certified numerical error budget, not a
claimed implementation of the general h-law.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .covariance import validate_covariance, orthant_spectral_data


@dataclass
class ExcursionSample:
    times: np.ndarray
    values: np.ndarray
    segment_minima: np.ndarray | None
    metadata: dict


def _vector_bridge(times, start, end, rng):
    times = np.asarray(times)
    duration = times[-1]-times[0]
    dt = np.diff(times)
    noise = np.vstack([np.zeros(len(start)), np.cumsum(rng.normal(size=(len(dt), len(start)))*np.sqrt(dt[:, None]), axis=0)])
    fraction = ((times-times[0])/duration)[:, None]
    return start + fraction*(end-start) + noise-fraction*noise[-1]


def _bessel_bridge(times, start, end, rng):
    """Only used when at least one endpoint is zero: norm of 3D bridge is exact."""
    if start != 0 and end != 0:
        raise ValueError("This construction requires one zero endpoint")
    return np.linalg.norm(_vector_bridge(times, np.array([start,0.,0.]), np.array([end,0.,0.]), rng), axis=1)


def positive_bridge_minima(x, y, dt, rng):
    """Exact minima for independent positive scalar Brownian bridges.

    Survival above z is [1-exp(-2(x-z)(y-z)/dt)]/[1-exp(-2xy/dt)].
    A Bessel(3) bridge conditioned on its positive endpoints has this law.
    """
    x, y, dt = np.broadcast_arrays(np.asarray(x,float), np.asarray(y,float), np.asarray(dt,float))
    if np.any(x < 0) or np.any(y < 0) or np.any(dt <= 0):
        raise ValueError("Need nonnegative endpoints and positive durations")
    survival0 = -np.expm1(-2*x*y/dt)
    log_term = np.log1p(-rng.random(x.shape)*survival0)
    # Stable smaller root of z²-(x+y)z+xy+(dt/2)log_term=0.
    discriminant = np.sqrt((x-y)**2-2*dt*log_term)
    numerator = 2*x*y+dt*log_term
    result = np.divide(numerator, x+y+discriminant, out=np.zeros_like(x), where=(x+y+discriminant)>0)
    return np.clip(result, 0, np.minimum(x,y))


def sample_independent_excursion(n_steps, dimension=3, rng=None, law="biased", oversample=1):
    """Exact continuous-law values/minima at requested grid for C=I.

    For biased law, m1(s)^2 gives S_i²~Dirichlet(3/2,...,3/2),
    T_i~InvGamma(alpha1,1/2), and conditional scalar paths are Bessel3 bridges
    joined at U=T1/(T1+T2), height S/sqrt(T1+T2) (paper eqs.32--40).
    """
    if int(n_steps) != n_steps or n_steps < 3 or int(oversample) != oversample or oversample < 1:
        raise ValueError("n_steps>=3 and oversample>=1 must be integers")
    if int(dimension) != dimension or dimension < 2:
        raise ValueError("dimension must be an integer at least two")
    if law not in {"biased", "unbiased"}:
        raise ValueError("law must be biased or unbiased")
    rng = np.random.default_rng() if rng is None else rng
    times = np.linspace(0,1,n_steps*oversample+1)
    metadata = {"sampler":"independent_bessel3_exact_minima", "law":law,
                "covariance":np.eye(dimension).tolist(), "oversample":int(oversample),
                "grid_values_exact":True, "cell_minima_exact":True,
                "mcmc":False, **orthant_spectral_data(dimension)}
    if law == "biased":
        alpha1 = metadata["alpha1"]
        hitting = 0.5/rng.gamma(alpha1, size=2)
        junction = hitting[0]/hitting.sum()
        direction = np.sqrt(rng.dirichlet(np.full(dimension,1.5)))
        height = direction/np.sqrt(hitting.sum())
        times = np.unique(np.r_[times,junction])
        mid = int(np.searchsorted(times,junction))
        values = np.empty((len(times),dimension))
        for k in range(dimension):
            values[:mid+1,k] = _bessel_bridge(times[:mid+1],0,height[k],rng)
            values[mid:,k] = _bessel_bridge(times[mid:],height[k],0,rng)
        metadata.update({"hitting_times":hitting.tolist(),"junction_time":float(junction),
                         "junction_direction":direction.tolist()})
    else:
        values = np.column_stack([_bessel_bridge(times,0.,0.,rng) for _ in range(dimension)])
    minima = positive_bridge_minima(values[:-1],values[1:],np.diff(times)[:,None],rng)
    return ExcursionSample(times,values,minima,metadata)


def _conditional_bridge_noise(length, cholesky, dt, rng):
    increments = rng.normal(size=(length,cholesky.shape[0])) @ cholesky.T * np.sqrt(dt)
    walk = np.vstack([np.zeros(cholesky.shape[0]),np.cumsum(increments,axis=0)])
    return walk[1:-1] - np.arange(1,length)[:,None]/length*walk[-1]


def sample_conditioned_bridge(n_steps, covariance, rng=None, oversample=4, sweeps=64, initial=None, initial_scale=1.0):
    """Finite-grid orthant bridge, targeted by exact ESS Markov transitions.

    Stationary density is the Gaussian bridge density times positivity at grid
    points. It differs from continuous excursion law through missed crossings;
    finite burn-in adds a separate mixing error. No general cone eigenfunction
    is assumed. Use independent chains and grid/sweep sensitivity before fitting.
    """
    c = validate_covariance(covariance)
    if not np.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("initial_scale must be finite and positive")
    if int(n_steps)!=n_steps or n_steps<3 or int(oversample)!=oversample or oversample<1 or int(sweeps)!=sweeps or sweeps<1:
        raise ValueError("n_steps>=3, oversample>=1, sweeps>=1 must be integers")
    rng = np.random.default_rng() if rng is None else rng
    m = n_steps*oversample
    d = len(c)
    times = np.linspace(0,1,m+1)
    if initial is None:
        x = sample_independent_excursion(m,d,rng,law="unbiased").values
    else:
        x = np.asarray(initial,float).copy()
        if x.shape != (m+1,d) or np.any(x[1:-1] <= 0) or np.any(x[[0,-1]] != 0):
            raise ValueError("Initial state must be positive interior bridge on the requested grid")
    x *= initial_scale
    chol = np.linalg.cholesky(c)
    levels = [2**p for p in range(1,int(np.ceil(np.log2(m)))+1)]
    levels = sorted(set([min(m,v) for v in levels]))
    updates = trials = 0
    trace = []
    for sweep in range(sweeps):
        # Random shift avoids permanently pinning dyadic block boundary values.
        for length in rng.permutation(levels):
            shift = int(rng.integers(0,length))
            endpoints = np.unique(np.r_[0,np.arange(shift,m,length),m]).astype(int)
            for lo,hi in zip(endpoints[:-1],endpoints[1:]):
                size = hi-lo
                if size < 2:
                    continue
                frac = np.arange(1,size)[:,None]/size
                mean = x[lo]+frac*(x[hi]-x[lo])
                current = x[lo+1:hi]-mean
                noise = _conditional_bridge_noise(size,chol,1/m,rng)
                theta = rng.uniform(0,2*np.pi)
                lower,upper = theta-2*np.pi,theta
                for trial in range(10000):
                    candidate = mean+current*np.cos(theta)+noise*np.sin(theta)
                    trials += 1
                    if np.all(candidate > 0):
                        x[lo+1:hi] = candidate
                        break
                    if theta < 0:
                        lower = theta
                    else:
                        upper = theta
                    theta = rng.uniform(lower,upper)
                else:
                    raise RuntimeError("Elliptical slice bracket failed to find current support")
                updates += 1
        trace.append([float(np.mean(x[1:-1])),float(np.linalg.norm(np.linalg.solve(chol,x[m//2])))])
    meta = {"sampler":"finite_grid_orthant_bridge_multiscale_ess", "law":"unbiased_finite_grid_approximation",
            "covariance":c.tolist(),"oversample":int(oversample),"sweeps":int(sweeps),"mcmc":True,
            "updates":updates,"likelihood_evaluations":trials,"trace_area_midpoint_radius":trace,
            "grid_values_exact":False,"cell_minima_exact":False,
            "initialization":"independent_unbiased_excursions" if initial is None else "user_supplied",
            "initial_scale":float(initial_scale),
            "limitations":["finite burn-in, no certified mixing", "positivity only at grid points",
                           "piecewise-linear minima and symbolic ties; refine oversampling"]}
    return ExcursionSample(times,x,None,meta)


def sample_excursion(n_steps, covariance, rng=None, law="biased", oversample=1, sweeps=64, initial_scale=1.0):
    c = validate_covariance(covariance)
    if np.array_equal(c,np.eye(len(c))):
        return sample_independent_excursion(n_steps,len(c),rng,law,oversample)
    if law != "unbiased":
        raise ValueError("General covariance currently supports only law='unbiased' finite-grid MCMC approximation; biased h-law needs a validated cone eigenfunction")
    return sample_conditioned_bridge(n_steps,c,rng,oversample,sweeps,initial_scale=initial_scale)
