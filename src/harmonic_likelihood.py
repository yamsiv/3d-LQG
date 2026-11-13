"""Interior-ball kernel likelihood diagnostics, independent of path clocks.

Quadrature observations here are NOT uniform bounds on transition kernels.
See docs/LIKELIHOOD_PHASE2C.md for the exact telescoping identity and limits.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp


def sphere_rule(order=16):
    if not isinstance(order, int) or order < 2:
        raise ValueError("Quadrature order must be an integer >=2")
    z, weights = leggauss(order)
    phi = 2*np.pi*(np.arange(2*order)+.37)/(2*order)
    points = np.column_stack((np.repeat(np.sqrt(1-z*z), 2*order)*np.tile(np.cos(phi), order),
                             np.repeat(np.sqrt(1-z*z), 2*order)*np.tile(np.sin(phi), order),
                             np.repeat(z, 2*order)))
    return points, np.repeat(weights, 2*order)/(4*order)


def log_mean_value_ratios(log_h, centers, radii, order=16, batch_size=48):
    """Return quadrature log(average_sphere h/h(center)), without flooring."""
    centers = np.asarray(centers, float); radii = np.asarray(radii, float)
    if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
        raise ValueError("Expected N three-dimensional centers and N radii")
    if np.any(radii <= 0): raise ValueError("Radii must be positive")
    omega, weights = sphere_rule(order); result = []
    for start in range(0, len(centers), batch_size):
        points = centers[start:start+batch_size]
        balls = points[:, None, :]+radii[start:start+batch_size, None, None]*omega
        central = np.asarray(log_h(points))
        values = np.asarray(log_h(balls.reshape(-1, 3))).reshape(len(points), len(omega))
        if np.any(~np.isfinite(central)) or np.any(~np.isfinite(values)):
            raise ValueError("Positive finite h required on every center and sphere")
        result.extend(logsumexp(values-central[:, None]+np.log(weights), axis=1))
    return np.asarray(result)


def chain_log_likelihood(log_relative_start, log_relative_end, log_mean_ratios):
    """Exact algebra if the supplied log sphere means are exact integrals."""
    return float(log_relative_end-log_relative_start-np.sum(log_mean_ratios))


def conditional_log_likelihood_bound(relative_error, cumulative_log_defect_bound):
    """Symmetric bound under independently established UNIFORM hypotheses.

    If |h_tilde/h-1|<=epsilon at both endpoints and sum |log(Z/h_tilde)|<=eta,
    then |log(dQ/dP)|<=log((1+epsilon)/(1-epsilon))+eta. This function does
    not establish either hypothesis from numerical samples.
    """
    if not 0 <= relative_error < 1 or cumulative_log_defect_bound < 0:
        raise ValueError("Need 0<=relative_error<1 and nonnegative defect bound")
    return float(np.log1p(relative_error)-np.log1p(-relative_error)+cumulative_log_defect_bound)
