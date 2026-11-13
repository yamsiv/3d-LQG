"""Coupled path/minimum and Brownian-ball diagnostics for Task 2C.

No generic excursion sampler is asserted here. Nested exact finite-grid
controls isolate missing continuous minima, and nested Gaussian stopped
skeletons isolate the clock error of polygonal sphere exits.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache

from .brownian_excursions import ExcursionSample


def thin_polygonal(sample, n, oversample):
    """Keep a nested regular grid and junction, dropping exact minima.

    This is the same physical realization, never an independent sample.
    A source with exact scalar segment minima can be retained separately as
    the continuum reference for *all* its nested graph sizes.
    """
    if int(n) != n or n < 3 or int(oversample) != oversample or oversample < 1:
        raise ValueError("n>=3 and oversample>=1 must be integers")
    requested = np.linspace(0, 1, int(n*oversample)+1)
    if "junction_time" in sample.metadata:
        requested = np.unique(np.r_[requested, sample.metadata["junction_time"]])
    right = np.minimum(np.searchsorted(sample.times, requested), len(sample.times)-1)
    left = np.maximum(0, right-1)
    indices = np.where(abs(sample.times[left]-requested) < abs(sample.times[right]-requested), left, right)
    if not np.allclose(sample.times[indices], requested, rtol=0, atol=1e-13):
        raise ValueError("Requested grid must be nested in the source")
    metadata = dict(sample.metadata, cell_minima_exact=False,
                    coupled_source_oversample=sample.metadata.get("oversample"),
                    oversample=int(oversample),
                    minimum_method="polygonal, no coordinatewise corrections")
    return ExcursionSample(sample.times[indices].copy(), sample.values[indices].copy(), None, metadata)


def ball_exit_laplace(s, radius=1.):
    """Exact E[exp(-s*tau)] for standard three-dimensional BM from center.

    Solve (Delta/2) f=s f, f(radius)=1, regular at zero: z/sinh(z).
    Consequently E tau=r²/3 and Var tau=2r⁴/45.
    """
    s = np.asarray(s, float)
    if np.any(s < 0) or radius <= 0:
        raise ValueError("Need s>=0 and radius>0")
    z = radius*np.sqrt(2*s)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        result = np.where(z == 0, 1., z/np.sinh(z))
    return float(result) if result.ndim == 0 else result


def _clip_first_exit(points, dt, radius):
    """First *polygonal* sphere exit, not the continuous Brownian exit."""
    exits = np.flatnonzero(np.sum(points*points, axis=1) >= radius*radius)
    if not len(exits):
        raise ValueError("Skeleton does not exit")
    index = int(exits[0])
    if index == 0:
        raise ValueError("Skeleton must start at the ball center")
    previous = points[index-1]
    delta = points[index]-previous
    dot = float(previous @ delta)
    size2 = float(delta @ delta)
    fraction = (-dot+np.sqrt(dot*dot+size2*(radius*radius-float(previous@previous))))/size2
    fraction = float(np.clip(fraction, 0., 1.))
    path = points[:index+1].copy()
    path[-1] = previous+fraction*delta
    times = np.arange(index+1)*dt
    times[-1] = (index-1+fraction)*dt
    return path, times, {"dt":float(dt), "steps":index,
                         "final_step_fraction":fraction,
                         "continuous_exit_exact":False}


def coupled_gaussian_ball_walks(radius, dimension, rng, *, relative_steps=(.2,.1,.05,.025), max_steps=2_000_000):
    """Exact nested Brownian grid values with coupled polygonal exits.

    Finest Gaussian increments aggregate into every coarser step. Generate
    until the coarsest skeleton exits; each finer skeleton exits no later
    because it contains all coarser grid values. For ell=relative_step*r,
    dt=ell²/d. Grid sampling is exact; stopping time and between-grid paths
    remain approximate. This is not a Brownian-inside-ball exact sampler.
    """
    scales = np.asarray(relative_steps, float)
    if radius <= 0 or int(dimension) != dimension or dimension < 1:
        raise ValueError("Need positive radius and integer dimension")
    if scales.ndim != 1 or len(scales) == 0 or np.any(scales <= 0):
        raise ValueError("Need a nonempty positive scale vector")
    dt = (radius*scales.min())**2/dimension
    ratios_float = (scales/scales.min())**2
    ratios = np.rint(ratios_float).astype(int)
    if not np.allclose(ratios_float, ratios, rtol=0, atol=1e-10):
        raise ValueError("Squared scale ratios must be integers")
    coarsest = int(ratios.max())
    if max_steps < coarsest:
        raise ValueError("Step budget smaller than one coarse interval")
    chunks = [np.zeros((1,dimension))]
    position = np.zeros(dimension)
    count = 0
    # Complete coarse blocks are necessary for honest nested stopping.
    batch = max(coarsest, int(np.ceil((2*dimension/scales.min()**2)/coarsest))*coarsest)
    while count+coarsest <= max_steps:
        size = min(batch, ((max_steps-count)//coarsest)*coarsest)
        points = position+np.cumsum(rng.normal(size=(size,dimension))*np.sqrt(dt), axis=0)
        coarse = points[coarsest-1::coarsest]
        exits = np.flatnonzero(np.sum(coarse*coarse, axis=1) >= radius*radius)
        if len(exits):
            chunks.append(points[:(int(exits[0])+1)*coarsest])
            full = np.vstack(chunks)
            return {float(scale): _clip_first_exit(full[::ratio], dt*ratio, radius)
                    for scale,ratio in zip(scales,ratios)}
        chunks.append(points)
        position = points[-1]
        count += size
    raise RuntimeError("Nested Gaussian ball paths exceeded step budget")


def bridge_ball_missed_exit_bound(start, end, radius, duration):
    """Conservative conditional upper bound for a missed ball exit.

    If the straight segment's margin is delta=r-max(|start|,|end|), an
    excursion of the bridge outside the ball requires some component of
    its fluctuation to exceed delta/sqrt(d). The one-sided Brownian bridge
    crossing probability is exp(-2*a²/dt); union over d axes and both signs.
    The bound degenerates to one near the sphere and cannot certify a full
    stopped path without adaptive boundary treatment.
    """
    start,end=np.asarray(start,float),np.asarray(end,float)
    if start.ndim != 1 or start.shape != end.shape or duration <= 0 or radius <= 0:
        raise ValueError("Need matching vectors, radius>0, duration>0")
    margin=radius-max(np.linalg.norm(start),np.linalg.norm(end))
    if margin <= 0:
        return 1.
    return float(min(1.,2*len(start)*np.exp(-2*margin*margin/(len(start)*duration))))


def ball_exit_cdf(times):
    """CDF of the unit-radius 3D Brownian center-to-sphere exit time.

    Jacobi's imaginary transformation avoids cancellation for short times.
    Twelve terms in either representation suffice to floating-point error
    at the switching point t=.3. This law belongs to generator Delta/2.
    """
    original=np.asarray(times,float)
    t=np.atleast_1d(original)
    answer=np.zeros_like(t)
    short=(t>0)&(t<=.3)
    long=t>.3
    if np.any(short):
        odd=np.arange(1,24,2,dtype=float)
        answer[short]=2*np.sqrt(2/(np.pi*t[short]))*np.exp(-odd[:,None]**2/(2*t[short][None,:])).sum(axis=0)
    if np.any(long):
        k=np.arange(1,13,dtype=float)
        survival=2*((-1)**(k+1))[:,None]*np.exp(-k[:,None]**2*np.pi**2*t[long][None,:]/2)
        answer[long]=1-survival.sum(axis=0)
    return float(answer[0]) if original.ndim==0 else answer


def _ball_exit_quantile_bisection(uniforms):
    u=np.asarray(uniforms,float)
    lo=np.zeros_like(u);hi=np.full_like(u,24.)
    for _ in range(58):
        mid=(lo+hi)/2
        lower=ball_exit_cdf(mid)<u
        lo=np.where(lower,mid,lo);hi=np.where(lower,hi,mid)
    return (lo+hi)/2


@lru_cache(maxsize=1)
def _ball_exit_quantile_table():
    # A logit grid retains tail accuracy. Deterministic interpolation defects
    # are measured against the independently evaluated CDF in experiments.
    logit=np.linspace(-32.,32.,16385)
    u=1/(1+np.exp(-logit))
    return u,_ball_exit_quantile_bisection(u)


def ball_exit_quantile(uniforms):
    """Numerical inverse CDF; exact mathematical target, tabulated numerical law.

    Monotone linear interpolation, verified with explicit CDF residuals. Tail
    draws outside the table are inverted directly, never clipped/rejected.
    """
    original=np.asarray(uniforms,float)
    u=np.atleast_1d(original)
    if np.any((u<=0)|(u>=1)):
        raise ValueError("Uniform variates must be strictly between zero and one")
    grid,quantile=_ball_exit_quantile_table()
    result=np.interp(u,grid,quantile)
    outside=(u<grid[0])|(u>grid[-1])
    if np.any(outside):
        result[outside]=_ball_exit_quantile_bisection(u[outside])
    return float(result[0]) if original.ndim==0 else result


def brownian_sphere_skeleton(radius, dimension, rng, *, step_size,
                            boundary_tolerance=1e-7, max_steps=2_000_000):
    """Brownian walk-on-sub-spheres with independent exact-law 3D clocks.

    At each sub-ball center the exit time is independent of its uniform exit
    direction. The sampled positions and clocks are genuine Brownian stopping
    skeleton values up to numerical inverse-CDF error. Stop within delta of
    the outer sphere and project the final point; omit a residual time whose
    conditional mean <=2*r*delta/3. Unknown paths between skeleton points are
    confined in the sub-balls: their polygonal interpolation admits spatial
    error <=2*step_size on the same underlying Brownian realization before
    the final boundary layer. Joint minima are still approximated, not exact.
    """
    if dimension!=3:
        raise ValueError("Exact ball clock implemented for dimension three only")
    if radius<=0 or step_size<=0 or not 0<boundary_tolerance<.1:
        raise ValueError("Need positive radii and boundary tolerance in (0,.1)")
    point=np.zeros(3);points=[point.copy()];times=[0.]
    elapsed=0.;largest=0.;step=0
    while step<max_steps:
        gap=radius-np.linalg.norm(point)
        if gap<=radius*boundary_tolerance:
            # The genuine remaining Brownian path may wander; projection is
            # explicitly approximate, with mean-time / endpoint bounds below.
            if np.linalg.norm(point)==0:
                raise RuntimeError("Degenerate boundary termination")
            points[-1]=point*radius/np.linalg.norm(point)
            return np.asarray(points),np.asarray(times),{
                "steps":step,"dt":float(max(np.diff(times),default=0.)),
                "max_subball_radius":float(largest),"omitted_boundary_gap":float(gap),
                "omitted_time_mean_bound":float(2*radius*gap/3),
                "unprojected_endpoint_mean_squared_error_bound":float(2*radius*gap),
                "clock":"3D center-to-sphere inverse CDF, numerical interpolation",
                "continuous_exit_exact":False,"joint_minima_exact":False}
        if gap>=step_size/.9:
            # Exact constant-radius sub-ball skeleton until the first center
            # approaches the outer boundary. Oversimulated increments are
            # discarded; all retained sub-balls lie within the outer ball.
            batch=min(max_steps-step,max(32,min(4096,int(2*(radius/step_size)**2))))
            directions=rng.normal(size=(batch,3))
            directions/=np.linalg.norm(directions,axis=1)[:,None]
            candidates=point+np.cumsum(step_size*directions,axis=0)
            near=np.flatnonzero(np.linalg.norm(candidates,axis=1)>=radius-step_size/.9)
            used=int(near[0])+1 if len(near) else batch
            candidates=candidates[:used]
            uniforms=np.maximum(rng.random(used),np.nextafter(0.,1.))
            clocks=step_size**2*ball_exit_quantile(uniforms)
            clock_values=elapsed+np.cumsum(clocks)
            points.extend(candidates);times.extend(clock_values)
            elapsed=float(clock_values[-1]);point=candidates[-1]
            step+=used;largest=max(largest,step_size)
            continue
        subradius=min(step_size,.9*gap)
        largest=max(largest,subradius)
        direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        u=rng.random()
        if u==0:
            # Generator endpoint event: draw from the open uniform interval.
            u=np.nextafter(0.,1.)
        elapsed+=subradius*subradius*ball_exit_quantile(u)
        point=point+subradius*direction
        points.append(point.copy());times.append(elapsed)
        step+=1
    raise RuntimeError("Brownian sphere skeleton exceeded step budget")


def _wos_conditioned_leg(model,start,rng,*,step_size,relative_step,ball_fraction,
                         boundary_tolerance,origin_cutoff,max_steps,max_balls,trace_balls):
    from .doob_excursions import sample_sphere_exit,_rotate_to_exit
    point=start.copy();chunks=[point[None,:]];clocks=[np.zeros(1)]
    elapsed=0.;steps=0;bounds=0.;max_dt=0.;traces=[]
    for ball in range(max_balls):
        norm=np.linalg.norm(point)
        if norm<=origin_cutoff:
            tail_time=norm**2*.5/rng.gamma(model.alpha1)
            chunks.append(np.zeros((1,3)));clocks.append(np.array([elapsed+tail_time]))
            values=np.vstack(chunks);times=np.concatenate(clocks)
            # Adding local clocks to a much larger leg clock can produce
            # duplicate floating-point times near sub-ball boundaries.
            keep=np.r_[np.diff(times)>0,True]
            return values[keep],times[keep],{
                "ball_count":ball,"step_count":steps,"max_physical_dt":max_dt,
                "boundary_omitted_time_conditional_mean_bound_sum":bounds,
                "tail_radius":float(norm),"tail_time":float(tail_time),
                "collapsed_roundoff_times":int(np.sum(~keep)),"ball_trace":traces}
        radius=ball_fraction*float(np.min(model.R@point))
        if radius<=0:
            raise RuntimeError("Conditioned skeleton escaped cone")
        endpoint,_=sample_sphere_exit(model,point,radius,rng)
        ell=min(step_size,relative_step*radius)
        path,times,info=brownian_sphere_skeleton(radius,3,rng,step_size=ell,
            boundary_tolerance=boundary_tolerance,max_steps=max_steps-steps)
        steps+=info["steps"];bounds+=info["omitted_time_mean_bound"]
        max_dt=max(max_dt,info["dt"])
        if trace_balls:
            traces.append({"center":point.tolist(),"radius":float(radius),
                           "exit":endpoint.tolist(),"log_h_ratio":float(model.log_h(endpoint)-model.log_h(point))})
        path=point+_rotate_to_exit(path,endpoint-point);path[-1]=endpoint
        chunks.append(path[1:]);clocks.append(elapsed+times[1:])
        elapsed+=times[-1];point=endpoint
    raise RuntimeError("Conditioned WOS leg reached ball budget")


def sample_wos_doob(n_steps,covariance=None,rng=None,*,model=None,step_size=.01,
                    relative_step=.2,ball_fraction=.5,boundary_tolerance=1e-7,
                    origin_cutoff=.001,max_steps=2_000_000,max_balls=20000,
                    trace_balls=False):
    """Paper-biased analytic-h excursion with exact-law sub-ball clocks.

    New provisional path approximation; old Doob code remains unchanged.
    Numerical angular modes are deliberately refused until separately audited.
    The outer sphere exit direction is h-weighted, and each Brownian interior
    path is rotated to it exactly as in Budd--Castro Section 3.4.
    """
    from .doob_excursions import analytic_cone
    if int(n_steps)!=n_steps or n_steps<3 or step_size<=0 or not 0<relative_step<1:
        raise ValueError("Need n_steps>=3, positive step_size, relative_step in (0,1)")
    if not 0<ball_fraction<1 or not 0<origin_cutoff<1:
        raise ValueError("Need ball_fraction and origin_cutoff in (0,1)")
    rng=np.random.default_rng() if rng is None else rng
    model=analytic_cone(covariance) if model is None else model
    if model.dimension!=3 or not model.analytic:
        raise ValueError("This control API requires a validated three-dimensional analytic h")
    direction=model.sample_direction(rng)
    options=dict(step_size=step_size,relative_step=relative_step,ball_fraction=ball_fraction,
        boundary_tolerance=boundary_tolerance,origin_cutoff=origin_cutoff,max_steps=max_steps,
        max_balls=max_balls,trace_balls=trace_balls)
    first,tfirst,ifirst=_wos_conditioned_leg(model,direction,rng,**options)
    second,tsecond,isecond=_wos_conditioned_leg(model,direction,rng,**options)
    total=tfirst[-1]+tsecond[-1]
    raw_times=np.r_[tfirst[-1]-tfirst[::-1],tfirst[-1]+tsecond[1:]]/total
    raw_values=np.vstack([first[::-1],second[1:]])@model.R.T/np.sqrt(total)
    keep=np.r_[np.diff(raw_times)>0,True]
    raw_times=raw_times[keep];raw_values=raw_values[keep]
    raw_values[[0,-1]]=0
    if np.min(raw_values)<-1e-10:
        raise RuntimeError("WOS path violates positivity")
    raw_values=np.maximum(raw_values,0.)
    times=np.unique(np.r_[raw_times,np.linspace(0,1,n_steps+1)])
    values=np.column_stack([np.interp(times,raw_times,raw_values[:,i]) for i in range(3)])
    metadata={"sampler":"phase2c_analytic_doob_exact_subball_clocks",
        "law":"biased","covariance":model.covariance.tolist(),"alpha1":float(model.alpha1),
        "grid_values_exact":False,"cell_minima_exact":False,"mcmc":False,
        "angular_mode":"analytic","hitting_times":[float(tfirst[-1]),float(tsecond[-1])],
        "junction_time":float(tfirst[-1]/total),"junction_direction":direction.tolist(),
        "junction_direction_correlated":(model.R@direction).tolist(),
        "step_size":step_size,"relative_step":relative_step,"ball_fraction":ball_fraction,
        "boundary_tolerance":boundary_tolerance,"origin_cutoff":origin_cutoff,
        "leg_diagnostics":[ifirst,isecond],"max_normalized_dt":max(ifirst["max_physical_dt"],isecond["max_physical_dt"])/total,
        "limitations":["numerical 3D sub-ball exit inverse CDF",
            "outer boundary projection and omitted residual clock explicitly bounded",
            "origin tail is straight with exact marginal lifetime",
            "polygonal joint minima; no coordinatewise correlated corrections"]}
    return ExcursionSample(times,values,None,metadata)
