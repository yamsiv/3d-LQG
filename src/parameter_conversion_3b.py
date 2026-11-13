"""Finite-resolution LFPP/GMC conversion with explicit uncertainty and domain flags.

The identities are theoretical; their use with finite cube-crossing slopes is
an exploratory proxy, never a certified asymptotic parameter identification.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def implicit_conversion(xi, crossing_exponent):
    """Return Q, subcritical gamma and d, keeping invalid branches as NaN.

    xi=0 is the separately defined flat endpoint gamma=0,d=3,Q=+infinity.
    For positive xi require strictly Q>sqrt(6); no supercritical inversion.
    """
    xi, exponent = np.broadcast_arrays(np.asarray(xi, float),
                                       np.asarray(crossing_exponent, float))
    if np.any(xi < 0) or np.any(~np.isfinite(xi)) or np.any(~np.isfinite(exponent)):
        raise ValueError("Nonnegative finite xi and finite exponent required")
    q = np.full(xi.shape, np.inf)
    np.divide(1-exponent, xi, out=q, where=xi > 0)
    valid = (xi > 0) & (q > np.sqrt(6))
    gamma = np.full(xi.shape, np.nan)
    gamma[xi == 0] = 0.
    gamma[valid] = 6/(q[valid]+np.sqrt(q[valid]**2-6))
    dimension = np.full(xi.shape, np.nan)
    dimension[xi == 0] = 3.
    np.divide(gamma, xi, out=dimension, where=xi > 0)
    return q, gamma, dimension


def fit_crossing_exponents(crossings, sizes, *, indices=None, correction_power=None):
    """Fit log ensemble medians, resampling whole field rows outside this call.

    Shape is (independent fields, resolutions, parameters). The model is
    log median D_N=A-lambda*log N+B*N**(-p), with B absent if p=None.
    Fixed p is a sensitivity ansatz, not an inferred correction exponent.
    """
    values = np.asarray(crossings, float)
    sizes = np.asarray(sizes, float)
    if values.ndim != 3 or values.shape[1] != len(sizes):
        raise ValueError("Expected field,resolution,xi crossing array")
    if len(set(sizes)) != len(sizes) or np.any(sizes <= 0):
        raise ValueError("Positive distinct sizes required")
    if np.any(values <= 0) or np.any(~np.isfinite(values)):
        raise ValueError("Positive finite physical distances required")
    indices = np.arange(len(sizes)) if indices is None else np.asarray(indices, int)
    selected = sizes[indices]
    columns = [np.ones(len(selected)), -np.log(selected)]
    if correction_power is not None:
        if correction_power <= 0:
            raise ValueError("Correction power must be positive")
        columns.append(selected**(-correction_power))
    design = np.column_stack(columns)
    if len(selected) < len(columns) or np.linalg.matrix_rank(design) != len(columns):
        raise ValueError("Insufficient independent sizes for this correction model")
    medians = np.median(values[:, indices, :], axis=0)
    coefficients = np.linalg.lstsq(design, np.log(medians), rcond=None)[0]
    return coefficients[1]


@dataclass(frozen=True)
class ParameterCalibration:
    """Table interpolator; no extrapolation or monotonicity imposed on data."""
    rows: tuple[dict, ...]
    metadata: dict

    def evaluate(self, xi):
        xi = float(xi)
        if xi == 0:
            return {"xi": 0., "gamma": 0., "dimension": 3., "Q": None,
                    "valid": True, "status": "exact_flat_endpoint",
                    "gamma_mc95": [0., 0.], "gamma_systematic": [0., 0.]}
        rows = sorted((r for r in self.rows if r["xi"] > 0), key=lambda r:r["xi"])
        grid = np.array([r["xi"] for r in rows])
        if not rows or xi < grid[0] or xi > grid[-1]:
            return {"xi": xi, "valid": False, "status": "outside_calibration_domain",
                    "gamma": None, "dimension": None, "Q": None}
        def interpolate(key):
            values = np.array([np.nan if r.get(key) is None else r[key] for r in rows])
            nearest = np.flatnonzero(np.isclose(grid, xi, rtol=0, atol=1e-12))
            if len(nearest):
                value = values[nearest[0]]
            else:
                right = np.searchsorted(grid, xi)
                if not np.all(np.isfinite(values[right-1:right+1])):
                    return None
                # Local linear interpolation of slopes avoids bridging invalid
                # subcritical nodes or inventing extrema in uncertainty bands.
                value = np.interp(xi, grid[right-1:right+1], values[right-1:right+1])
            return float(value) if np.isfinite(value) else None
        exponent = interpolate("lambda")
        if exponent is None:
            return {"xi": xi, "valid": False, "status": "invalid_crossing_proxy",
                    "gamma": None, "dimension": None, "Q": None}
        q, gamma, dimension = implicit_conversion(xi, exponent)
        if not np.isfinite(gamma):
            return {"xi": xi, "valid": False, "status": "Q_not_subcritical",
                    "gamma": None, "dimension": None, "Q": float(q)}
        mc = [interpolate("gamma_mc95_low"), interpolate("gamma_mc95_high")]
        systematic = [interpolate("gamma_systematic_low"),
                      interpolate("gamma_systematic_high")]
        fraction = interpolate("bootstrap_subcritical_fraction")
        all_models = interpolate("all_sensitivity_models_subcritical")
        stable = (fraction is not None and fraction >= .975 and all_models == 1.)
        return {"xi": xi, "lambda": exponent, "Q": float(q), "gamma": float(gamma),
                "dimension": float(dimension), "gamma_mc95": mc,
                "gamma_systematic": systematic, "valid": True,
                "bootstrap_subcritical_fraction": fraction,
                "gamma_mc95_is_conditional": fraction is not None and fraction < 1.,
                "gamma_systematic_has_invalid_models": all_models != 1.,
                "status": "provisional_finite_resolution" if stable else
                          "provisional_branch_or_model_sensitive",
                "asymptotic_map_validated": False}


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    return ParameterCalibration(tuple(data["conversion_rows"]), data.get("metadata", {}))
