"""Variance-renormalized GMC quadrature and separate probability measures.

All finite grids have finite moments. The continuum positive q-moment
threshold is q<6/gamma^2 for a nontrivial fixed-volume set in unit-log d=3.
This is distinct from the metric-ball exponent of an LFPP/GMC geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class GMCMeasure:
    probabilities: np.ndarray
    total_mass: float
    log_total_mass: float
    log_cell_masses: np.ndarray
    gamma: float


def gmc_weights(h, gamma, variance):
    """Return unnormalized unit-volume cell masses and normalized probabilities.

    variance is the deterministic ensemble point variance of exactly this
    regularization. Replacing it by a realization's spatial variance changes
    the measure. No samplewise normalization is used in total_mass.
    """
    h = np.asarray(h, dtype=float)
    if h.ndim != 3 or min(h.shape) < 1 or not np.all(np.isfinite(h)):
        raise ValueError("h must be a finite three-dimensional array")
    if not np.isfinite(gamma) or not np.isfinite(variance) or variance < 0:
        raise ValueError("Require finite gamma and nonnegative variance")
    log_cells = gamma*h-.5*gamma**2*variance-np.log(h.size)
    log_total = float(logsumexp(log_cells))
    probabilities = np.exp(log_cells-log_total)
    total = float(np.exp(log_total)) if log_total < np.log(np.finfo(float).max) else float("inf")
    return GMCMeasure(probabilities, total, log_total, log_cells, float(gamma))


def moment_threshold(gamma):
    """Positive continuum q moment threshold; gamma=0 has all moments."""
    return float("inf") if gamma == 0 else float(6/gamma**2)


def euclidean_mass_exponent(q, gamma):
    """zeta(q)=3q-gamma² q(q-1)/2 for admissible Euclidean GMC moments."""
    return 3*q-.5*gamma**2*q*(q-1)


def coarse_cell_masses(measure, blocks_per_axis):
    """Unnormalized GMC masses in aligned equally sized cubic blocks.

    The number of blocks must divide every grid dimension. Large log total
    masses may overflow; the caller should instead analyze log weights there.
    """
    p = measure.probabilities
    b = int(blocks_per_axis)
    if b != blocks_per_axis or b < 1 or any(n % b for n in p.shape):
        raise ValueError("Number of blocks must divide every axis")
    s = tuple(n//b for n in p.shape)
    return p.reshape(b, s[0], b, s[1], b, s[2]).sum(axis=(1, 3, 5))*measure.total_mass
