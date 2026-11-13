"""Corner-aware spherical method of particular solutions (Task 2C).

This research solver returns a Helmholtz solution with a *numerical* boundary
residual. It is not a certified positive Dirichlet mode or a sampling kernel.
No boundary clipping, positive floor, or FEM interpolation is applied.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1
from numpy.polynomial.legendre import leggauss

from .covariance import validate_covariance


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def triangle_geometry(covariance):
    """Cholesky-whitened rays, vertex tangent frames and interior angles."""
    R = np.linalg.cholesky(validate_covariance(covariance))
    if R.shape != (3, 3):
        raise ValueError("Three-dimensional covariance required")
    vertices = _normalize(np.linalg.solve(R, np.eye(3)).T)
    frames, angles = [], []
    for j in range(3):
        q = vertices[j]
        e = _normalize(vertices[(j+1) % 3] - (vertices[(j+1) % 3] @ q)*q)
        t = _normalize(vertices[(j+2) % 3] - (vertices[(j+2) % 3] @ q)*q)
        f = np.cross(q, e)
        if f @ t < 0:
            f = -f
        frames.append(np.array([q, e, f]))
        angles.append(np.arctan2(t @ f, t @ e))
    return R, vertices, np.asarray(frames), np.asarray(angles)


def collocation_points(vertices, boundary_count=48, interior_count=120, seed=9274):
    """Chebyshev edge points plus log-spaced corner probes, independent interior."""
    t = (1-np.cos(np.pi*(np.arange(boundary_count)+.5)/boundary_count))/2
    t = np.unique(np.r_[t, np.geomspace(1e-7, .01, 8),
                         1-np.geomspace(1e-7, .01, 8)])
    boundary = np.concatenate([_normalize((1-t[:, None])*vertices[j]
                                +t[:, None]*vertices[(j+1) % 3]) for j in range(3)])
    rng = np.random.default_rng(seed)
    interior = _normalize(rng.dirichlet([1.2]*3, size=interior_count) @ vertices)
    return boundary, interior


def _radial(degree, order, cosine, sine=None):
    # P_degree^(-order)(cos r), with Gamma(1+order) removed by column scaling.
    # sin(r/2)^2=(1-cos r)/2; this representation supports fractional orders.
    if sine is None:
        z = np.clip((1-cosine)/2, 0, 1-1e-14)
    else:
        tangent_half = sine/(1+cosine)
        z = tangent_half**2/(1+tangent_half**2)
    return (z/(1-z))**(order/2)*hyp2f1(-degree, degree+1, 1+order, z)


def basis_matrix(points, degree, frames, angles, order, include_interior=True):
    points = _normalize(np.atleast_2d(np.asarray(points, float)))
    columns = []
    for frame, angle in zip(frames, angles):
        cosine = np.clip(points @ frame[0], -1, 1)
        a, b = points @ frame[1], points @ frame[2]
        theta = np.arctan2(b, a)
        for k in range(1, order+1):
            mu = k*np.pi/angle
            columns.append(_radial(degree, mu, cosine, np.hypot(a, b))*np.sin(mu*theta))
    if include_interior:
        q = _normalize(frames[:, 0].sum(axis=0))
        e = _normalize(frames[0, 0]-(frames[0, 0] @ q)*q)
        f = np.cross(q, e)
        cosine = np.clip(points @ q, -1, 1)
        a, b = points @ e, points @ f
        theta = np.arctan2(b, a)
        sine = np.hypot(a, b)
        columns.append(_radial(degree, 0, cosine, sine))
        for k in range(1, order+1):
            radial = _radial(degree, k, cosine, sine)
            columns.extend([radial*np.cos(k*theta), radial*np.sin(k*theta)])
    return np.asarray(columns).T


@dataclass
class CornerSpectralMode:
    covariance: np.ndarray
    degree: float
    order: int
    coefficients: np.ndarray
    frames: np.ndarray
    angles: np.ndarray
    boundary_tension: float
    condition_number: float
    include_interior: bool = True

    @property
    def eigenvalue(self):
        return self.degree*(self.degree+1)

    @property
    def alpha1(self):
        return self.degree+.5

    def angular_mode(self, points):
        scalar = np.asarray(points).ndim == 1
        values = basis_matrix(points, self.degree, self.frames, self.angles,
                              self.order, self.include_interior) @ self.coefficients
        return float(values[0]) if scalar else values

    def log_h(self, points):
        points = np.asarray(points, float)
        value = self.angular_mode(points)
        if np.any(value <= 0):
            raise ValueError("MPS mode is nonpositive: no clipping is permitted")
        return np.log(value)-(self.degree+1)*np.log(np.linalg.norm(points, axis=-1))

    def local_corner_expansion(self, vertex, radius=.12, terms=12, quadrature=160):
        """Stable local continuation from a geodesic arc inside one corner.

        Every retained term satisfies Helmholtz and both adjacent Dirichlet
        faces exactly. This is a local representation ONLY: replacing the
        global mode across a patch interface requires a separate matching
        error analysis. The trace is projected rather than forced positive.
        """
        frame, angle = self.frames[vertex], self.angles[vertex]
        nodes, weights = leggauss(quadrature)
        theta = (nodes+1)*angle/2
        points = (np.cos(radius)*frame[0]+np.sin(radius)*(
            np.cos(theta)[:, None]*frame[1]+np.sin(theta)[:, None]*frame[2]))
        trace = self.angular_mode(points)
        mus = np.arange(1, terms+1)*np.pi/angle
        # 2/angle times arc integral, hence the Gauss weights have no factor.
        amplitudes = np.sin(theta[:, None]*mus).T @ (weights*trace)
        radial_at_arc = np.array([_radial(self.degree, mu, np.cos(radius)) for mu in mus])
        return LocalCornerExpansion(self.degree, frame, angle, radius, mus,
                                    amplitudes/radial_at_arc)


@dataclass
class LocalCornerExpansion:
    degree: float
    frame: np.ndarray
    angle: float
    radius: float
    orders: np.ndarray
    coefficients: np.ndarray

    def angular_mode(self, points):
        scalar = np.asarray(points).ndim == 1
        points = _normalize(np.atleast_2d(points))
        a, b = points @ self.frame[1], points @ self.frame[2]
        # atan2 tangential norm retains r~1e-9; arccos(point dot pole)
        # loses all such distances in double precision.
        r = np.arctan2(np.hypot(a, b), points @ self.frame[0])
        theta = np.arctan2(b, a)
        if np.any(r > self.radius*(1+1e-10)):
            raise ValueError("Point outside the local corner patch")
        z = np.sin(r/2)**2
        values = np.zeros(len(points))
        for mu, coefficient in zip(self.orders, self.coefficients):
            radial = (z/(1-z))**(mu/2)*hyp2f1(-self.degree, self.degree+1, 1+mu, z)
            values += coefficient*radial*np.sin(mu*theta)
        return float(values[0]) if scalar else values


def fit_corner_mode(covariance, order=6, degree_bracket=None, degree=None,
                    boundary_count=48, interior_count=120, seed=9274,
                    include_interior=True):
    """Fit the first mode near an externally supplied spectral bracket.

    Stacked boundary/interior samples are column-scaled, rank-truncated by SVD,
    and orthonormalized. Minimize the smallest boundary singular value. Rank
    truncation avoids spurious vectors representing numerical zero everywhere.
    The bracket is NOT a certified eigenvalue enclosure. Fixed ``degree`` is
    useful for independently testing a known exact eigenspace.
    """
    R, vertices, frames, angles = triangle_geometry(covariance)
    boundary, interior = collocation_points(vertices, boundary_count, interior_count, seed)
    points = np.vstack([boundary, interior])
    def solve(nu, get_coefficients=False):
        A = basis_matrix(points, nu, frames, angles, order, include_interior)
        scales = np.linalg.norm(A, axis=0)
        A = A/scales
        U, singular, Vh = svd(A, full_matrices=False, check_finite=False)
        keep = singular > singular[0]*2e-13
        Ub = U[:len(boundary), keep]
        _, sigma, wh = svd(Ub, full_matrices=False, check_finite=False)
        if not get_coefficients:
            return float(sigma[-1])
        coefficients = (Vh[keep].T @ (wh[-1]/singular[keep]))/scales
        value = basis_matrix(_normalize(vertices.sum(axis=0)), nu, frames, angles,
                              order, include_interior) @ coefficients
        coefficients /= value[0]
        return coefficients, float(sigma[-1]), float(singular[0]/singular[keep][-1])
    if degree is None:
        if degree_bracket is None:
            raise ValueError("Supply an independently motivated degree bracket")
        opt = minimize_scalar(solve, bounds=degree_bracket, method="bounded",
                              options={"xatol": 2e-11, "maxiter": 80})
        if not opt.success:
            raise RuntimeError(opt.message)
        degree = float(opt.x)
        # scipy's bounded stopping rule contains sqrt(eps)*abs(x). Recenter
        # once, otherwise an order-one degree imposes an avoidable ~1e-8
        # floor even with a much smaller requested xatol.
        center = degree
        fine = minimize_scalar(lambda z: solve(center+1e-5*z), bounds=(-1, 1),
                               method="bounded", options={"xatol": 2e-9})
        if fine.success:
            degree = float(center+1e-5*fine.x)
    coefficients, tension, condition = solve(degree, True)
    return CornerSpectralMode(np.asarray(covariance), degree, order, coefficients,
                              frames, angles, tension, condition, include_interior)
