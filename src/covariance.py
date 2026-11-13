"""Correlation conventions for Budd--Castro (2207.05355), equation (28).

User coordinates are a=C12, b=C13, c=C23. Paper cone angles are
alpha=acos(-a), beta=acos(-c), theta=acos(-b); theta replaces paper gamma.
"""
from __future__ import annotations
import numpy as np


def validate_covariance(matrix):
    c = np.asarray(matrix, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        raise ValueError("Covariance must be square with dimension at least two")
    if not np.all(np.isfinite(c)) or not np.allclose(c, c.T, rtol=0, atol=1e-12):
        raise ValueError("Covariance must be finite and symmetric")
    if not np.allclose(np.diag(c), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Use unit diagonal (positive coordinate scales do not change graph law)")
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance must be strictly positive definite") from exc
    return c.copy()


def from_correlations(a, b, c):
    return validate_covariance([[1, a, b], [a, 1, c], [b, c, 1]])


def from_angles(alpha, beta, theta_angle):
    angles = np.asarray([alpha, beta, theta_angle], dtype=float)
    if np.any(~np.isfinite(angles)) or np.any(angles <= 0) or np.any(angles >= np.pi):
        raise ValueError("Cone angles must lie strictly between zero and pi")
    return from_correlations(-np.cos(alpha), -np.cos(theta_angle), -np.cos(beta))


def to_angles(matrix):
    c = validate_covariance(matrix)
    if c.shape != (3, 3):
        raise ValueError("Three cone angles require a 3 by 3 covariance")
    return tuple(np.arccos(-np.array([c[0, 1], c[1, 2], c[0, 2]])))


def equicorrelated(rho, dimension=3):
    if not -1.0 / (dimension - 1) < rho < 1:
        raise ValueError("rho is outside the strictly positive-definite interval")
    return validate_covariance((1-rho)*np.eye(dimension) + rho*np.ones((dimension, dimension)))


def cone_geometry(matrix):
    """C=RR^T, W={z:Rz>0}; rows of R are inward unit face normals."""
    c = validate_covariance(matrix)
    r = np.linalg.cholesky(c)
    return {"cholesky": r, "inverse_cholesky": np.linalg.inv(r),
            "face_normals": r, "eigenvalues": np.linalg.eigvalsh(c),
            "condition_number": float(np.linalg.cond(c))}


def orthant_spectral_data(dimension=3):
    """Analytical independent orthant: m1 proportional to product coordinates."""
    p = dimension
    alpha1 = p + dimension / 2 - 1
    return {"homogeneity_p": p, "lambda1": p*(p+dimension-2),
            "alpha1": alpha1, "gamma_s": 1-alpha1,
            "h_homogeneity": -alpha1-dimension/2+1}


def birectangular_spectral_data(alpha):
    if not 0 < alpha < np.pi:
        raise ValueError("alpha must lie in (0,pi)")
    lam = (1+np.pi/alpha)*(2+np.pi/alpha)
    alpha1 = np.sqrt(lam+0.25)
    return {"lambda1": float(lam), "alpha1": float(alpha1), "gamma_s": float(1-alpha1)}
