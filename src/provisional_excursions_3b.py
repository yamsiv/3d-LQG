"""Explicitly provisional positive-Ritz, angular-quadrature cone excursions.

The target is Budd--Castro's *biased* excursion, not a positive-grid bridge.
Both angular kernels are approximations: finite importance resampling for the
initial m^2 direction, and randomly rotated Gauss sphere quadrature for exits.
No harmonic Harnack bound is used for the numerical Ritz function. Previous
samplers are unchanged. See docs/PROVISIONAL_SAMPLER_PHASE3B.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp

from .brownian_excursions import ExcursionSample
from .corner_ritz import fit_corner_ritz
from .covariance import validate_covariance
from .harmonic_likelihood import sphere_rule
from .phase2c_paths import brownian_sphere_skeleton
from .doob_excursions import _rotate_to_exit


def _uniform_sphere(rng, count):
    x = rng.normal(size=(count, 3))
    return x/np.linalg.norm(x, axis=1)[:, None]


def _rotation(rng):
    """Haar orthogonal frame; reflection is harmless for uniform sphere area."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q*np.sign(np.diag(r))[None, :]


@lru_cache(maxsize=12)
def _rule(order):
    return sphere_rule(order)


@dataclass
class ProvisionalRitzCone:
    mode: object
    initial_candidates: int = 512

    def __post_init__(self):
        self.covariance = self.mode.covariance
        self.R = self.mode.factor.R
        self.dimension = 3
        self.alpha1 = float(self.mode.alpha1)
        self.p = self.alpha1-.5
        self.analytic = False
        self.certificate = self.mode.positivity_certificate()
        if not self.certificate["certified_positive"]:
            raise ValueError("Ritz polynomial positivity certificate failed; no clipping allowed")
        if self.initial_candidates < 8:
            raise ValueError("At least eight initial importance candidates required")

    def angular_mode(self, points):
        """Same stored Ritz formula as mode.angular_mode, without gradients."""
        z = np.asarray(points, float)
        scalar = z.ndim == 1
        z = np.atleast_2d(z)
        z = z/np.linalg.norm(z, axis=1)[:, None]
        d = z@self.R.T
        if np.any(d <= 0):
            raise ValueError("Numerical harmonic evaluation requires interior directions")
        factor = self.mode.factor
        logf = np.log(d).sum(axis=1)-factor.log_normalization
        for (opening, nu, e0, e1), (j, k, rho) in zip(factor.charts, factor.face_pairs):
            if abs(nu-2) < 1e-12:
                logf += np.log(2.)
                continue
            dj, dk = d[:, j], d[:, k]
            a = (dj-rho*dk)/np.sqrt(1-rho*rho)
            r2 = a*a+dk*dk
            near, far = np.minimum(dj, dk), np.maximum(dj, dk)
            delta = np.arctan2(near, (far-rho*near)/np.sqrt(1-rho*rho))
            logf += ((nu-2)*np.log(r2)/2+np.log(nu)
                     +np.log(np.sinc(nu*delta/np.pi))-np.log(np.sinc(delta/np.pi))
                     -np.log(np.sin(opening-delta)))
        # Normalize polynomial arguments before evaluating its homogeneous
        # expansion. This also avoids avoidable underflow near the vertex.
        scale = d.sum(axis=1)
        t = d/scale[:, None]
        powers = np.prod(t[:, None, :]**self.mode.exponents[None, :, :], axis=2)
        polynomial = powers@self.mode.coefficients
        if np.any(polynomial <= 0):
            raise FloatingPointError("Stored positive polynomial evaluated nonpositive")
        result = np.exp(logf+self.mode.degree*np.log(scale)+np.log(polynomial))
        return float(result[0]) if scalar else result

    def log_h(self, points):
        z = np.asarray(points, float)
        return (-self.alpha1-.5)*np.log(np.linalg.norm(z, axis=-1))+np.log(self.angular_mode(z))

    def sample_direction(self, rng, *, return_diagnostics=False):
        """Continuous finite-SIR approximation to m^2 spherical area law.

        Uniform-sphere points conditioned on cone membership are iid uniform
        spherical triangle proposals. Selecting among K by m^2 weights is
        self-normalized importance resampling, *not exact rejection*.
        """
        pieces = []
        total = 0
        proposals = 0
        for _ in range(2000):
            raw = _uniform_sphere(rng, max(256, 4*(self.initial_candidates-total)))
            inside = raw[np.all(raw@self.R.T > 0, axis=1)]
            proposals += len(raw)
            pieces.append(inside)
            total += len(inside)
            if total >= self.initial_candidates:
                break
        else:
            raise RuntimeError("Initial direction cone rejection budget exceeded")
        points = np.vstack(pieces)[:self.initial_candidates]
        weights = self.angular_mode(points)**2
        weights /= weights.sum()
        result = points[rng.choice(len(points), p=weights)]
        info = {"method": "finite_SIR_uniform_spherical_triangle",
                "candidates": self.initial_candidates, "uniform_proposals": proposals,
                "importance_ess": float(1/np.sum(weights*weights)),
                "maximum_probability": float(weights.max())}
        return (result, info) if return_diagnostics else result


def make_provisional_model(covariance, *, degree=4, quadrature_order=48,
                           initial_candidates=512):
    covariance = validate_covariance(covariance)
    if covariance.shape != (3, 3):
        raise ValueError("This provisional API supports three-dimensional cones")
    return ProvisionalRitzCone(fit_corner_ritz(covariance, degree, quadrature_order),
                               initial_candidates)


def quadrature_sphere_exit(model, point, radius, rng, *, order=8,
                           audit_order=None):
    """Randomly rotated finite sphere-quadrature exit kernel.

    Given a Haar frame, choose a Gauss node with probability proportional to
    its sphere weight times numerical h. Integration over random frames makes
    this continuous but does not remove finite-quadrature normalization bias.
    The returned mean-value defect is diagnostic, never a certified bound.
    """
    point = np.asarray(point, float)
    if not 0 < radius < np.min(model.R@point):
        raise ValueError("Strict interior positive-radius ball required")
    omega, weights = _rule(order)
    frame = _rotation(rng)
    candidates = point+radius*(omega@frame)
    logs = model.log_h(candidates)+np.log(weights)
    normalizer = float(logsumexp(logs))
    if not np.isfinite(normalizer):
        raise FloatingPointError("Nonfinite angular quadrature normalizer")
    index = rng.choice(len(candidates), p=np.exp(logs-normalizer))
    central = float(model.log_h(point))
    info = {"log_mean_value_ratio": normalizer-central, "quadrature_order": order,
            "quadrature_nodes": len(candidates)}
    if audit_order is not None:
        audit_nodes, audit_weights = _rule(audit_order)
        audit_logs = model.log_h(point+radius*(audit_nodes@frame))+np.log(audit_weights)
        audit_normalizer = float(logsumexp(audit_logs))
        info["audit_log_mean_value_ratio"] = audit_normalizer-central
        info["quadrature_log_normalizer_difference"] = audit_normalizer-normalizer
    return candidates[index], info


def _conditioned_leg(model, start, rng, *, step_size, relative_step, ball_fraction,
                     boundary_tolerance, origin_cutoff, max_steps, max_balls,
                     exit_order, audit_order, audit_stride, trace_balls):
    point = start.copy()
    chunks, clocks = [point[None, :]], [np.zeros(1)]
    elapsed = 0.
    steps = 0
    max_dt = 0.
    trace = []
    log_defects = []
    audit_differences = []
    omitted = 0.
    for ball in range(max_balls):
        norm = np.linalg.norm(point)
        if norm <= origin_cutoff:
            # Exact marginal tail lifetime only for the true harmonic model;
            # for an approximate h this is another explicit approximation.
            tail = norm*norm*.5/rng.gamma(model.alpha1)
            chunks.append(np.zeros((1, 3)))
            clocks.append(np.array([elapsed+tail]))
            values, times = np.vstack(chunks), np.concatenate(clocks)
            keep = np.r_[np.diff(times) > 0, True]
            return values[keep], times[keep], {
                "ball_count": ball, "step_count": steps, "max_physical_dt": max_dt,
                "tail_radius": float(norm), "tail_time": float(tail),
                "boundary_omitted_time_conditional_mean_bound_sum": omitted,
                "sum_log_mean_value_ratio": float(np.sum(log_defects)),
                "sum_abs_log_mean_value_ratio": float(np.sum(np.abs(log_defects))),
                "maximum_abs_log_mean_value_ratio": float(np.max(np.abs(log_defects), initial=0.)),
                "maximum_audited_quadrature_log_difference": float(np.max(np.abs(audit_differences), initial=0.)),
                "ball_trace": trace}
        radius = ball_fraction*float(np.min(model.R@point))
        if radius <= 0:
            raise RuntimeError("Provisional chain escaped cone")
        endpoint, exit_info = quadrature_sphere_exit(model, point, radius, rng,
            order=exit_order, audit_order=audit_order if ball % audit_stride == 0 else None)
        log_defects.append(exit_info["log_mean_value_ratio"])
        if "quadrature_log_normalizer_difference" in exit_info:
            audit_differences.append(exit_info["quadrature_log_normalizer_difference"])
        path, times, info = brownian_sphere_skeleton(radius, 3, rng,
            step_size=min(step_size, relative_step*radius),
            boundary_tolerance=boundary_tolerance, max_steps=max_steps-steps)
        steps += info["steps"]
        max_dt = max(max_dt, info["dt"])
        omitted += info["omitted_time_mean_bound"]
        if trace_balls:
            trace.append({"center": point.tolist(), "radius": float(radius),
                          "exit": endpoint.tolist(), **exit_info})
        path = point+_rotate_to_exit(path, endpoint-point)
        path[-1] = endpoint
        chunks.append(path[1:])
        clocks.append(elapsed+times[1:])
        elapsed += times[-1]
        point = endpoint
    raise RuntimeError("Provisional conditioned leg exceeded ball budget")


def sample_provisional_excursion(n_steps, covariance=None, rng=None, *, model=None,
                                degree=4, quadrature_order=48, initial_candidates=512,
                                step_size=.005, relative_step=.15, ball_fraction=.5,
                                boundary_tolerance=1e-7, origin_cutoff=.001,
                                max_steps=2_000_000, max_balls=20000, exit_order=8,
                                audit_order=16, audit_stride=16, trace_balls=False):
    """Approximate biased excursion for arbitrary SPD C, with explicit flags.

    n_steps inserts the desired graph grid, retaining *all* stopping skeleton
    points so no extra coarse time-grid minimum loss is introduced. Independent
    calls use independent randomness; there is no MCMC or hidden retry.
    """
    if int(n_steps) != n_steps or n_steps < 3 or step_size <= 0:
        raise ValueError("n_steps>=3 integer and step_size>0 required")
    if not 0 < relative_step < 1 or not 0 < ball_fraction < 1 or not 0 < origin_cutoff < 1:
        raise ValueError("relative_step, ball_fraction, origin_cutoff must lie in (0,1)")
    if audit_stride < 1:
        raise ValueError("Positive audit stride required")
    rng = np.random.default_rng() if rng is None else rng
    model = make_provisional_model(covariance, degree=degree,
        quadrature_order=quadrature_order, initial_candidates=initial_candidates) if model is None else model
    direction, direction_info = model.sample_direction(rng, return_diagnostics=True)
    options = dict(step_size=step_size, relative_step=relative_step,
        ball_fraction=ball_fraction, boundary_tolerance=boundary_tolerance,
        origin_cutoff=origin_cutoff, max_steps=max_steps, max_balls=max_balls,
        exit_order=exit_order, audit_order=audit_order, audit_stride=audit_stride,
        trace_balls=trace_balls)
    first, tfirst, ifirst = _conditioned_leg(model, direction, rng, **options)
    second, tsecond, isecond = _conditioned_leg(model, direction, rng, **options)
    total = tfirst[-1]+tsecond[-1]
    raw_times = np.r_[tfirst[-1]-tfirst[::-1], tfirst[-1]+tsecond[1:]]/total
    raw_values = np.vstack([first[::-1], second[1:]])@model.R.T/np.sqrt(total)
    keep = np.r_[np.diff(raw_times) > 0, True]
    raw_times, raw_values = raw_times[keep], raw_values[keep]
    raw_values[[0, -1]] = 0.
    if np.min(raw_values) < -1e-10:
        raise RuntimeError("Provisional excursion violates positivity")
    raw_values = np.maximum(raw_values, 0.)
    times = np.unique(np.r_[raw_times, np.linspace(0., 1., n_steps+1)])
    values = np.column_stack([np.interp(times, raw_times, raw_values[:, j]) for j in range(3)])
    metadata = {"sampler": "phase3b_provisional_positive_ritz_rotated_sphere_quadrature",
        "law": "biased_approximation", "target_law": "biased", "mcmc": False,
        "validation_status": "provisional", "covariance": model.covariance.tolist(),
        "grid_values_exact": False, "cell_minima_exact": False,
        "alpha1": model.alpha1, "eigenvalue": float(model.mode.eigenvalue),
        "ritz_degree": model.mode.degree, "ritz_quadrature_order": model.mode.quadrature_order,
        "ritz_positivity_certificate": model.certificate,
        "initial_direction": direction_info, "exit_order": exit_order,
        "exit_audit_order": audit_order, "exit_audit_stride": audit_stride,
        "step_size": step_size, "relative_step": relative_step,
        "ball_fraction": ball_fraction, "origin_cutoff": origin_cutoff,
        "boundary_tolerance": boundary_tolerance,
        "hitting_times": [float(tfirst[-1]), float(tsecond[-1])],
        "junction_time": float(tfirst[-1]/total), "junction_direction": direction.tolist(),
        "leg_diagnostics": [ifirst, isecond],
        "max_normalized_dt": max(ifirst["max_physical_dt"], isecond["max_physical_dt"])/total,
        "limitations": ["finite initial-direction importance resampling",
            "randomly rotated finite sphere-quadrature transition kernel",
            "locally normalized approximate nonharmonic Ritz function",
            "no uniform accumulated likelihood bound",
            "numerical ball clocks, omitted boundary layer, approximate origin tail",
            "polygonal continuous joint minima; no coordinatewise correction"]}
    return ExcursionSample(times, values, None, metadata)
