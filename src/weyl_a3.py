"""A3 chamber excursions: exact finite-time laws, approximate joint minima.

The four-particle Hermitian bridge is independent of the angular FEM and
walk-on-balls implementations. No continuous cell-minimum oracle is claimed.
See docs/WEYL_A3_PHASE2C.md for normalizations and the biased-law derivation.
"""
from __future__ import annotations

import numpy as np
from .brownian_excursions import ExcursionSample, _vector_bridge


def a3_covariance():
    return np.array([[1., -.5, 0.], [-.5, 1., -.5], [0., -.5, 1.]])


def a3_polynomial(x):
    """Positive degree-six harmonic polynomial in correlated gap coordinates."""
    x = np.asarray(x, dtype=float)
    a, b, c = np.moveaxis(x, -1, 0)
    return a*b*c*(a+b)*(b+c)*(a+b+c)


def a3_log_u_white(z):
    """log u(Rz) in W=R^-1 R_+^3; -inf outside the cone."""
    z = np.asarray(z, dtype=float)
    x = z @ np.linalg.cholesky(a3_covariance()).T
    factors = np.concatenate([x, (x[..., 0]+x[..., 1])[..., None],
                              (x[..., 1]+x[..., 2])[..., None],
                              x.sum(axis=-1)[..., None]], axis=-1)
    interior = np.all(x > 0, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(factors).sum(axis=-1)
    return np.where(interior, result, -np.inf)


def a3_angular_mode_white(s):
    """L2-normalized spherical eigenfunction on unit white directions.

    Integral_WcapS2 u(Rs)^2 d_sigma(s)=pi/180180. The caller supplies unit s.
    """
    return np.sqrt(180180/np.pi)*np.exp(a3_log_u_white(s))


class A3AnalyticCone:
    """Analytic A3 oracle implementing the existing interior-ball interface.

    The normalized angular mode is independent of radial length. The
    inward h is U(z)/|z|^13; normalization cancels in exit weights.
    """
    def __init__(self):
        self.covariance = a3_covariance()
        self.R = np.linalg.cholesky(self.covariance)
        self.dimension = 3
        self.p = 6.
        self.alpha1 = 6.5
        self.eigenvalue = 42.
        self.homogeneity = -7.
        self.analytic = True
        self.refinement = None

    def log_u(self, points):
        return a3_log_u_white(points)

    def log_h(self, points):
        points = np.asarray(points, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.log_u(points)-13*np.log(np.linalg.norm(points, axis=-1))

    def angular_mode(self, points):
        points = np.asarray(points, dtype=float)
        radius = np.linalg.norm(points, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(180180/np.pi)*np.exp(self.log_u(points)-6*np.log(radius))

    def sample_direction(self, rng):
        eigenvalues = a3_junction_direction(rng)
        correlated = np.diff(eigenvalues)/np.sqrt(2)
        return np.linalg.solve(self.R, correlated)

    def log_exit_bound(self, point, radius):
        distance = np.min(self.R @ np.asarray(point, dtype=float))
        eta = radius/distance
        if not 0 < eta < 1:
            raise ValueError("A strict interior ball is required")
        # Harnack on the maximal interior ball; no approximate-h normalization.
        return float(self.log_h(point)+np.log1p(eta)-2*np.log1p(-eta))


def a3_hermitian(coordinates):
    """16 orthonormal real coordinates for Hermitian 4x4 matrices.

    With iid N(0,1) coordinates, density is exp(-Tr(H^2)/2). Increments
    scaled by sqrt(dt) therefore have the standard Dyson beta=2 clock.
    """
    z = np.asarray(coordinates)
    if z.shape[-1] != 16:
        raise ValueError("A Hermitian 4x4 matrix needs 16 real coordinates")
    matrix = np.zeros(z.shape[:-1]+(4, 4), dtype=complex)
    matrix[..., np.arange(4), np.arange(4)] = z[..., :4]
    for k, (i, j) in enumerate([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]):
        value = (z[..., 4+2*k]+1j*z[..., 5+2*k])/np.sqrt(2)
        matrix[..., i, j] = value
        matrix[..., j, i] = value.conjugate()
    return matrix


def _centered_eigenvalues(coordinates):
    eigenvalues = np.linalg.eigvalsh(a3_hermitian(coordinates))
    return eigenvalues-eigenvalues.mean(axis=-1, keepdims=True)


def a3_junction_direction(rng):
    """Ordered centered eigenvalues of norm one; exact m1^2 angle law."""
    eigenvalues = _centered_eigenvalues(rng.normal(size=16))
    return eigenvalues/np.linalg.norm(eigenvalues)


def _matrix_bridge_eigenvalues(times, start, end, rng):
    start_z = np.r_[start, np.zeros(12)]
    end_z = np.r_[end, np.zeros(12)]
    return _centered_eigenvalues(_vector_bridge(times, start_z, end_z, rng))


def sample_a3_excursion(n_steps, rng=None, law="biased", oversample=32):
    """Sample the original or paper-biased A3 cone excursion on a time grid.

    Four centered eigenvalues give three positive gaps /sqrt(2), with
    covariance (a,b,c)=(-1/2,0,-1/2). The finite-dimensional law is exact
    up to Gaussian/eigensolver roundoff. Polygonal cell minima are not exact.
    """
    if int(n_steps) != n_steps or n_steps < 3:
        raise ValueError("n_steps must be an integer at least three")
    if int(oversample) != oversample or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    if law not in {"biased", "unbiased"}:
        raise ValueError("law must be biased or unbiased")
    rng = np.random.default_rng() if rng is None else rng
    times = np.linspace(0., 1., n_steps*oversample+1)
    zero = np.zeros(4)
    metadata = {"sampler": "weyl_a3_hermitian_bridge_polygonal_minima",
                "law": law, "covariance": a3_covariance().tolist(),
                "oversample": int(oversample), "grid_values_exact": True,
                "cell_minima_exact": False, "mcmc": False,
                "homogeneity_p": 6, "alpha1": 6.5, "lambda1": 42.,
                "gamma_s": -5.5, "radial_dimension": 15,
                "limitations": ["Exact grid values; polygonal joint gap minima only",
                                "Resolution rows from one path are paired, not independent"]}
    if law == "unbiased":
        eigenvalues = _matrix_bridge_eigenvalues(times, zero, zero, rng)
    else:
        direction = a3_junction_direction(rng)
        hitting = .5/rng.gamma(6.5, size=2)
        junction = hitting[0]/hitting.sum()
        times = np.unique(np.r_[times, junction])
        middle = int(np.searchsorted(times, junction))
        endpoint = direction/np.sqrt(hitting.sum())
        eigenvalues = np.empty((len(times), 4))
        eigenvalues[:middle+1] = _matrix_bridge_eigenvalues(times[:middle+1], zero, endpoint, rng)
        eigenvalues[middle:] = _matrix_bridge_eigenvalues(times[middle:], endpoint, zero, rng)
        metadata.update({"hitting_times": hitting.tolist(), "junction_time": float(junction),
                         "junction_eigenvalues": direction.tolist(),
                         "junction_direction_correlated": (np.diff(direction)/np.sqrt(2)).tolist()})
    values = np.diff(eigenvalues, axis=1)/np.sqrt(2)
    values[[0, -1]] = 0.
    return ExcursionSample(times, values, None, metadata)


def thin_a3_sample(sample, n_steps, oversample):
    """Nested retained grid from the same path, preserving the exact junction."""
    if sample.metadata.get("sampler") != "weyl_a3_hermitian_bridge_polygonal_minima":
        raise ValueError("Expected an A3 matrix-bridge sample")
    if int(n_steps) != n_steps or n_steps < 3 or int(oversample) != oversample or oversample < 1:
        raise ValueError("n_steps>=3 and oversample>=1 must be integers")
    requested = np.linspace(0., 1., n_steps*oversample+1)
    if "junction_time" in sample.metadata:
        requested = np.unique(np.r_[requested, sample.metadata["junction_time"]])
    right = np.minimum(np.searchsorted(sample.times, requested), len(sample.times)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(abs(sample.times[left]-requested) < abs(sample.times[right]-requested), left, right)
    if not np.allclose(sample.times[indices], requested, atol=1e-13, rtol=0):
        raise ValueError("Requested grid is not nested in source grid")
    metadata = dict(sample.metadata)
    metadata.update({"oversample": int(oversample),
                     "coupled_source_oversample": sample.metadata["oversample"]})
    return ExcursionSample(sample.times[indices].copy(), sample.values[indices].copy(), None, metadata)
