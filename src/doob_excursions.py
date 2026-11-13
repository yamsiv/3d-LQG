"""Budd--Castro biased excursions by h-weighted interior-ball paths.

Mathematical target: equations (32), (33), (41) and Sections 3.4/4.1 of
arXiv:2207.05355v1. Analytical cone harmonics give exact sphere-exit laws;
bounded random walks approximate Brownian paths inside the balls. General
spherical FEM modes define a separately marked, unvalidated approximation.
No finite-grid positivity MCMC is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .brownian_excursions import ExcursionSample
from .covariance import validate_covariance


def _unit_vectors(rng, count, dimension):
    vectors = rng.normal(size=(count, dimension))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


@dataclass
class AnalyticCone:
    """A wedge, or a wedge times a perpendicular positive half-line.

    pair=(i,j), extra=k: rows i,j of R are wedge inward unit normals;
    row k is its perpendicular axis. A rotation of whitening is harmless.
    """
    covariance: np.ndarray
    R: np.ndarray
    pair: tuple
    extra: int | None

    def __post_init__(self):
        self.dimension = len(self.covariance)
        i, j = self.pair
        rho = self.covariance[i, j]
        self.opening_angle = float(np.arccos(-rho))
        self.wedge_power = np.pi / self.opening_angle
        self.p = self.wedge_power + int(self.extra is not None)
        self.alpha1 = self.p + self.dimension / 2 - 1
        self.eigenvalue = self.p * (self.p + self.dimension - 2)
        self.e0 = (self.R[i] - rho*self.R[j]) / np.sqrt(1-rho*rho)
        self.e1 = self.R[j]
        self.ey = self.R[self.extra] if self.extra is not None else None
        self.analytic = True
        self.refinement = None
        self.homogeneity = -self.p-self.dimension+2

    def log_u(self, points):
        x = np.asarray(points, float)
        scalar = x.ndim == 1
        x = np.atleast_2d(x)
        a, b = x @ self.e0, x @ self.e1
        angle = np.arctan2(b, a)
        wedge_radius = np.hypot(a, b)
        inside = np.all(x @ self.R.T > 0, axis=1)
        inside &= (angle > 0) & (angle < self.opening_angle)
        answer = np.full(len(x), -np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            answer[inside] = self.wedge_power*np.log(wedge_radius[inside]) + np.log(
                np.sin(self.wedge_power*angle[inside]))
            if self.ey is not None:
                answer[inside] += np.log(x[inside] @ self.ey)
        return float(answer[0]) if scalar else answer

    def log_h(self, points):
        x = np.asarray(points, float)
        return self.log_u(x) - (2*self.p+self.dimension-2)*np.log(np.linalg.norm(x, axis=-1))

    def angular_mode(self, points):
        x = np.asarray(points, float)
        return np.exp(self.log_u(x)-self.p*np.log(np.linalg.norm(x, axis=-1)))

    def sample_direction(self, rng):
        # Inverse CDF is unnecessary: rejection costs two theta proposals on
        # average. y²~Beta(3/2,k+1) follows from sphere area dtheta*dy.
        for _ in range(10000):
            theta = rng.uniform(0, self.opening_angle)
            if rng.random() < np.sin(self.wedge_power*theta)**2:
                break
        else:
            raise RuntimeError("Angular rejection failed")
        planar = np.cos(theta)*self.e0 + np.sin(theta)*self.e1
        if self.ey is None:
            return planar
        y = np.sqrt(rng.beta(1.5, self.wedge_power+1))
        return np.sqrt(1-y*y)*planar + y*self.ey

    def log_exit_bound(self, point, radius):
        distance = np.min(self.R @ point)
        eta = radius/distance
        if not 0 < eta < 1:
            raise ValueError("A strict interior ball is required")
        # Positive harmonic-function Harnack bound in the larger ball of
        # radius distance: sup h(x+r*u)/h(x) <= (1+eta)/(1-eta)^(d-1).
        return float(self.log_h(point)+np.log1p(eta)-(self.dimension-1)*np.log1p(-eta))


@dataclass
class FEMCone:
    covariance: np.ndarray
    R: np.ndarray
    eigenpair: object

    def __post_init__(self):
        self.dimension = 3
        expected_corners = np.linalg.solve(self.R,np.eye(3)).T
        expected_corners /= np.linalg.norm(expected_corners,axis=1)[:,None]
        vertices = np.asarray(self.eigenpair.vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices)<3 or not np.allclose(
                vertices[:3],expected_corners,rtol=1e-10,atol=1e-10):
            raise ValueError("The supplied eigenpair cone does not match the covariance whitening")
        self.eigenvalue = float(self.eigenpair.eigenvalue)
        self.alpha1 = np.sqrt(self.eigenvalue+.25)
        self.p = self.alpha1-.5
        self.homogeneity = -self.alpha1-.5
        self.refinement = self.eigenpair.refinement
        self.analytic = False
        values = np.asarray(self.eigenpair.values)
        if np.any(~np.isfinite(values)) or np.any(values < 0) or values.max() <= 0:
            raise ValueError("FEM eigenmode must be finite, nonnegative and nonzero")
        self.maximum_mode = float(values.max())
        faces = np.asarray(self.eigenpair.faces)
        unresolved = np.max(values[faces],axis=1) <= 1e-14*self.maximum_mode
        if np.any(unresolved):
            raise ValueError("Unresolved interior FEM zero-support facets; use a connected corner-enriched mesh and audit its errors")

    def angular_mode(self, points):
        return self.eigenpair.angular_mode(points)

    def log_h(self, points):
        x = np.asarray(points, float)
        with np.errstate(divide="ignore"):
            return self.homogeneity*np.log(np.linalg.norm(x, axis=-1))+np.log(self.angular_mode(x))

    def sample_direction(self, rng):
        # Global nodal maximum bounds a positive barycentric interpolant.
        for _ in range(100000):
            directions = _unit_vectors(rng, 64, 3)
            inside = np.all(directions @ self.R.T > 0, axis=1)
            modes = np.zeros(64)
            if np.any(inside):
                modes[inside] = self.angular_mode(directions[inside])
            if np.any(modes > self.maximum_mode*(1+1e-10)):
                raise RuntimeError("FEM angular envelope violated")
            accepted = np.flatnonzero(rng.random(64) < (modes/self.maximum_mode)**2)
            if len(accepted):
                return directions[accepted[0]]
        raise RuntimeError("FEM angular rejection exhausted; cone unresolved or too narrow")

    def log_exit_bound(self, point, radius):
        # Harnack cannot be invoked: an interpolated approximate FEM mode is
        # not harmonic. Use a rigorous global bound for this interpolant.
        minimum_radius = np.linalg.norm(point)-radius
        return float(np.log(self.maximum_mode)+self.homogeneity*np.log(minimum_radius))


def analytic_cone(covariance):
    c = validate_covariance(covariance)
    dimension = len(c)
    if dimension == 2:
        return AnalyticCone(c, np.linalg.cholesky(c), (0,1), None)
    if dimension != 3:
        raise ValueError("Analytical Doob sampler supports dimension two or three")
    for extra in range(3):
        pair = tuple(i for i in range(3) if i != extra)
        if np.all(np.abs(c[extra, list(pair)]) < 1e-14):
            return AnalyticCone(c, np.linalg.cholesky(c), pair, extra)
    raise ValueError("No birectangular analytical cone; supply a numerical eigenpair explicitly")


def make_cone_model(covariance, eigenpair=None):
    c = validate_covariance(covariance)
    if eigenpair is not None:
        if c.shape != (3,3):
            raise ValueError("Numerical angular modes apply to 3D cones only")
        return FEMCone(c, np.linalg.cholesky(c), eigenpair)
    return analytic_cone(c)


def sample_sphere_exit(model, point, radius, rng, *, max_proposals=1_000_000):
    """Exit density proportional to h(x+r*u), relative to uniform sphere area.

    Exact h gives normalizer h(x). FEM uses the interpolant's own normalizer,
    which is generally not h(x): the resulting chain is provisional.
    """
    distances = model.R @ point
    if not np.isfinite(radius) or radius <= 0 or radius >= np.min(distances):
        raise ValueError("Sphere exit requires a strictly interior positive-radius ball")
    if not np.isfinite(model.log_h(point)):
        raise ValueError("h is zero/nonfinite at the ball center; an interior FEM zero patch is inadmissible")
    upper = model.log_exit_bound(point, radius)
    evaluated = 0
    used = 0
    zero_proposals = 0
    while evaluated < max_proposals:
        directions = _unit_vectors(rng, 32, model.dimension)
        candidates = point + radius*directions
        log_values = model.log_h(candidates)
        zero_proposals += int(np.count_nonzero(~np.isfinite(log_values)))
        evaluated += len(candidates)
        if np.any(log_values > upper+1e-9):
            raise RuntimeError("h rejection envelope violated")
        accept = np.flatnonzero(np.log(rng.random(len(candidates))) < log_values-upper)
        if len(accept):
            used += int(accept[0])+1
            return candidates[accept[0]], {"proposals_used":used,"evaluated":evaluated,
                                          "zero_h_candidates":zero_proposals}
        used += len(candidates)
    raise RuntimeError("Sphere rejection exhausted; FEM/global envelope may be impractical")


def isotropic_ball_walk(radius, dimension, rng, *, step_size, relative_step=.2, max_steps=2_000_000):
    """Approximate BM from a ball center to its sphere, with physical clock.

    Fixed length ell has covariance ell²/d per step, so dt=ell²/d. Clip the
    final segment geometrically and use its linear time fraction. This is an
    explicit boundary-time approximation, not an exact Brownian ball walk.
    """
    ell = min(float(step_size), float(relative_step)*radius)
    dt = ell*ell/dimension
    if not np.isfinite(ell) or ell <= 0:
        raise ValueError("Positive finite walk step required")
    chunks = []
    position = np.zeros(dimension)
    count = 0
    batch = min(8192, max(32, int(2*(radius/ell)**2)))
    while count < max_steps:
        number = min(batch, max_steps-count)
        increments = ell*_unit_vectors(rng, number, dimension)
        path = position+np.cumsum(increments, axis=0)
        exits = np.flatnonzero(np.sum(path*path,axis=1) >= radius*radius)
        if not len(exits):
            chunks.append(path)
            position = path[-1]
            count += number
            continue
        index = int(exits[0])
        previous = position if index == 0 else path[index-1]
        delta = increments[index]
        dot = float(previous @ delta)
        fraction = (-dot+np.sqrt(dot*dot+ell*ell*(radius*radius-float(previous@previous))))/(ell*ell)
        fraction = float(np.clip(fraction,0,1))
        path[index] = previous+fraction*delta
        chunks.append(path[:index+1])
        points = np.vstack([np.zeros(dimension),*chunks])
        times = np.arange(len(points))*dt
        times[-1] = (len(points)-2+fraction)*dt
        return points, times, {"steps":len(points)-1,"ell":ell,
                               "dt":dt,"final_step_fraction":fraction}
    raise RuntimeError("Ball random walk reached step budget")


def _rotate_to_exit(points, target):
    u = points[-1]/np.linalg.norm(points[-1])
    v = target/np.linalg.norm(target)
    delta = u-v
    norm = np.linalg.norm(delta)
    if norm < 1e-14:
        return points.copy()
    # A Householder map is orthogonal and maps u to v. Reflection is valid:
    # Brownian/random-walk laws are invariant under all orthogonal maps.
    direction = delta/norm
    return points-2*(points@direction)[:,None]*direction


def _conditioned_leg(model, start, rng, *, ball_fraction, step_size,
                     relative_step, origin_cutoff, max_steps, max_balls,
                     max_exit_proposals):
    point = start.copy()
    chunks = [point[None,:]]
    time_chunks = [np.zeros(1)]
    elapsed = 0.
    steps = proposals = evaluated = zero_candidates = 0
    max_dt = 0.
    for ball in range(max_balls):
        norm = np.linalg.norm(point)
        if norm <= origin_cutoff:
            # The remaining *time* law is known radially for any exact cone h.
            # Its path is replaced by a straight segment only inside cutoff.
            tail_time = norm*norm*.5/rng.gamma(model.alpha1)
            elapsed += tail_time
            chunks.append(np.zeros((1,model.dimension)))
            time_chunks.append(np.array([elapsed]))
            return np.vstack(chunks), np.concatenate(time_chunks), {
                "ball_count":ball,"step_count":steps,"rejection_proposals":proposals,
                "evaluated_proposals":evaluated,"max_physical_dt":max_dt,
                "zero_h_candidates":zero_candidates,
                "tail_radius":float(norm),"tail_time":float(tail_time),
                "tail_sup_exceeds_twice_cutoff_bound":float((norm/(2*origin_cutoff))**(2*model.alpha1))
                    if model.analytic else None}
        distances = model.R @ point
        if np.min(distances) <= 0:
            raise RuntimeError("Doob path escaped its cone")
        radius = ball_fraction*float(np.min(distances))
        endpoint, exit_info = sample_sphere_exit(model, point, radius, rng,
                                                max_proposals=max_exit_proposals)
        proposals += exit_info["proposals_used"]
        evaluated += exit_info["evaluated"]
        zero_candidates += exit_info["zero_h_candidates"]
        path, times, walk_info = isotropic_ball_walk(radius,model.dimension,rng,
            step_size=step_size,relative_step=relative_step,max_steps=max_steps-steps)
        steps += walk_info["steps"]
        max_dt = max(max_dt,walk_info["dt"])
        path = point+_rotate_to_exit(path,endpoint-point)
        path[-1] = endpoint
        chunks.append(path[1:])
        time_chunks.append(elapsed+times[1:])
        elapsed += times[-1]
        point = endpoint
        if steps >= max_steps:
            raise RuntimeError("Doob leg reached total step budget")
    raise RuntimeError("Doob leg reached ball budget")


def sample_doob_excursion(n_steps, covariance, rng=None, *, ball_fraction=.5,
                         step_size=.01, relative_step=.2, origin_cutoff=.01,
                         max_steps=2_000_000, max_balls=20000, eigenpair=None,
                         oversample=1, max_exit_proposals=1_000_000):
    """Approximate the paper-biased excursion with explicit numerical layers.

    `step_size` and `origin_cutoff` are measured before duration normalization
    in the whitened cone. All raw polygonal vertices are retained, then graph
    cell boundaries are inserted by interpolation. Inserting those boundaries
    cannot recover missing Brownian minima; refine the walks themselves.
    """
    if int(n_steps)!=n_steps or n_steps<3 or int(oversample)!=oversample or oversample<1:
        raise ValueError("n_steps>=3 and oversample>=1 must be integers")
    if not 0 < ball_fraction < 1 or not 0 < relative_step < 1:
        raise ValueError("ball_fraction and relative_step must lie in (0,1)")
    if not np.isfinite(step_size) or step_size <= 0 or not 0 < origin_cutoff < 1:
        raise ValueError("Positive step_size and origin_cutoff in (0,1) required")
    rng = np.random.default_rng() if rng is None else rng
    model = make_cone_model(covariance,eigenpair)
    direction = model.sample_direction(rng)
    options = dict(ball_fraction=ball_fraction,step_size=step_size,
                   relative_step=relative_step,origin_cutoff=origin_cutoff,
                   max_steps=max_steps,max_balls=max_balls,
                   max_exit_proposals=max_exit_proposals)
    first, tfirst, info_first = _conditioned_leg(model,direction,rng,**options)
    second, tsecond, info_second = _conditioned_leg(model,direction,rng,**options)
    total = tfirst[-1]+tsecond[-1]
    raw_times = np.r_[tfirst[-1]-tfirst[::-1],tfirst[-1]+tsecond[1:]]/total
    raw_values = np.vstack([first[::-1],second[1:]]) @ model.R.T/np.sqrt(total)
    raw_values[[0,-1]] = 0
    if np.any(raw_values < -1e-10):
        raise RuntimeError("Rotated polygonal path violates positivity")
    negative_roundoff_count = int(np.count_nonzero(raw_values < 0))
    negative_roundoff_magnitude = float(max(0.,-np.min(raw_values)))
    raw_values = np.maximum(raw_values,0)
    times = np.unique(np.r_[raw_times,np.linspace(0,1,n_steps*oversample+1)])
    values = np.column_stack([np.interp(times,raw_times,raw_values[:,k]) for k in range(model.dimension)])
    metadata = {"sampler":"doob_walk_on_interior_balls","law":"biased",
        "covariance":model.covariance.tolist(),"mcmc":False,
        "angular_mode":"analytical" if model.analytic else "surface_FEM_interpolant_PROVISIONAL",
        "grid_values_exact":False,"cell_minima_exact":False,
        "dimension":model.dimension,"alpha1":float(model.alpha1),
        "lambda1":float(model.eigenvalue),"gamma_s":float(1-model.alpha1),
        "fem_refinement":model.refinement,"ball_fraction":ball_fraction,
        "max_exit_proposals":max_exit_proposals,
        "step_size":step_size,"relative_step":relative_step,"origin_cutoff":origin_cutoff,
        "brownian_clock":"fixed_length_ell_has_dt_ell_squared_over_dimension",
        "final_ball_step_clock":"linear_fraction_of_dt",
        "hitting_times":[float(tfirst[-1]),float(tsecond[-1])],
        "junction_time":float(tfirst[-1]/total),
        "junction_direction":direction.tolist(),
        "junction_direction_correlated":(model.R@direction).tolist(),
        "leg_diagnostics":[info_first,info_second],"raw_vertices":len(raw_times),
        "negative_roundoff_clamp_count":negative_roundoff_count,
        "negative_roundoff_max_magnitude":negative_roundoff_magnitude,
        "max_normalized_dt":float(max(info_first["max_physical_dt"],info_second["max_physical_dt"])/total),
        "oversample":int(oversample),
        "limitations":["bounded isotropic walk instead of continuous Brownian motion",
                       "clipped final ball-step time approximation",
                       "straight path tail inside origin_cutoff with radial residual-time law",
                       "polygonal joint cell minima; inserted grids alone do not refine them"]}
    if not model.analytic:
        metadata["limitations"].extend(["FEM angular error and boundary relative error",
            "interpolated h is not harmonic; locally normalized exit kernel is not an exact Doob transform"])
    return ExcursionSample(times,values,None,metadata)
