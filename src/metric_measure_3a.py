"""Intrinsic probability metric-space summaries and realization bootstrap.

Every row is one physical field or graph. Root averages are conditional
Monte Carlo estimates; they are never counted as independent realizations.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


RADII = np.array([.25, .5, .75, 1., 1.5, 2.])
EVAL_GRID = np.linspace(0., 4., 1025)


def weighted_root_cdfs(distances, probabilities, *, bins=1024, discrete=False):
    """Weighted all-target CDFs; separate root atom retained at grid[0]=0.

    Continuous distances have histogram-bracketing error <=max_distance/bins.
    Discrete graph CDFs use exact integer support and right continuity.
    """
    distances = np.asarray(distances, float)
    p = np.asarray(probabilities, float).ravel()
    if distances.ndim != 2 or distances.shape[1] != len(p):
        raise ValueError("Expected root-by-target distances and target weights")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Connected finite metric required")
    if np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Probability weights must sum to one")
    top = float(distances.max())
    grid = np.arange(int(top)+1, dtype=float) if discrete else np.linspace(0., top, bins+1)
    curves = []
    for d in distances:
        order = np.argsort(d, kind="stable")
        cumulative = np.r_[0., np.cumsum(p[order])]
        curves.append(cumulative[np.searchsorted(d[order], grid, side="right")])
    return {"grid": grid, "cdfs": np.array(curves),
            "root_means": distances @ p,
            "distance_bin_width": 0. if discrete else top/bins}


def evaluate_cdf(record, points):
    grid, curves = np.asarray(record["grid"]), np.asarray(record["cdfs"])
    points = np.asarray(points)
    if record.get("discrete", False):
        indices = np.searchsorted(grid, points, side="right")-1
        return np.where(indices[None, :] >= 0,
                        curves[:, np.clip(indices, 0, len(grid)-1)], 0.)
    return np.array([np.interp(points, grid, c, left=0., right=1.) for c in curves])


def ensemble_median(records):
    """Median of equal-realization-weight root/all-target pair mixtures."""
    upper = max(float(np.asarray(r["grid"])[-1]) for r in records)
    lo, hi = 0., upper
    for _ in range(50):
        mid = (lo+hi)/2
        cdf = np.mean([evaluate_cdf(r, [mid]).mean() for r in records])
        if cdf >= .5: hi = mid
        else: lo = mid
    if hi <= 1e-12:
        raise ValueError("Ensemble pair median is zero: atomic measure dominates")
    return hi


def matrix_features(matrix):
    """Scale-free triangle shapes, keeping collisions in separate observables."""
    d = np.asarray(matrix, float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not np.allclose(d, d.T):
        raise ValueError("Symmetric square distance matrix required")
    triangles = np.array([sorted([d[i,j], d[i,k], d[j,k]])
                          for i,j,k in combinations(range(len(d)), 3)])
    positive = triangles[:, -1] > 0
    ratios = triangles[positive, :2]/triangles[positive, -1, None]
    # Matrix law includes repeated measure samples; never silently de-duplicate.
    return {"small_over_large": ratios[:, 0], "middle_over_large": ratios[:, 1],
            "degenerate_triangle_fraction": float(1-positive.mean()),
            "zero_pair_fraction": float(np.mean(d[np.triu_indices(len(d), 1)] == 0))}


def summarize(records, *, scale=None, per_realization=False):
    """Comparable finite-ensemble features and raw root/shape distributions."""
    if not records: raise ValueError("No realizations")
    scale = ensemble_median(records) if scale is None else float(scale)
    curves, balls, means, smallest, middle, atoms, degenerates = [], [], [], [], [], [], []
    small_means, middle_means, matrix_rows = [], [], []
    support=max(float(np.asarray(r["grid"])[-1])/(ensemble_median([r]) if per_realization else scale) for r in records)
    integration_grid=EVAL_GRID if support<=4 else np.unique(np.r_[EVAL_GRID,np.geomspace(4.,support,513)])
    integration_curves=[]
    for record in records:
        local_scale = ensemble_median([record]) if per_realization else scale
        curves.append(evaluate_cdf(record, EVAL_GRID*local_scale).mean(axis=0))
        integration_curves.append(evaluate_cdf(record, integration_grid*local_scale).mean(axis=0))
        balls.append(evaluate_cdf(record, RADII*local_scale))
        means.append(np.mean(record["root_means"])/local_scale)
        atoms.append(float(np.asarray(record["cdfs"])[:, 0].mean()))
        if record.get("matrix") is not None:
            features = matrix_features(record["matrix"])
            smallest.append(features["small_over_large"])
            middle.append(features["middle_over_large"])
            degenerates.append(features["degenerate_triangle_fraction"])
            if len(features["small_over_large"]):
                small_means.append(features["small_over_large"].mean())
                middle_means.append(features["middle_over_large"].mean())
            matrix_rows.append(np.asarray(record["matrix"])/local_scale)
    balls = np.asarray(balls)
    small = np.concatenate(smallest) if smallest else np.array([])
    mid = np.concatenate(middle) if middle else np.array([])
    return {"scale": scale, "cdf": np.mean(curves, axis=0),
            "integration_grid":integration_grid,"integration_cdf":np.mean(integration_curves,axis=0),
            "normalized_max_support":support,"tail_mass_at4":float(1-np.mean(curves,axis=0)[-1]),
            "realization_cdfs": np.array(curves), "balls": balls,
            "ball_sd": np.std(balls.reshape(-1,len(RADII)), axis=0),
            "ball_within_root_variance":np.var(balls,axis=1).mean(axis=0),
            "ball_between_realization_variance":np.var(balls.mean(axis=1),axis=0),
            "ball_quantiles": np.quantile(balls.reshape(-1,len(RADII)), [.1,.5,.9], axis=0),
            "normalized_realization_means": np.asarray(means),
            "scale_cv": float(np.std(means, ddof=1)/np.mean(means)) if len(means)>1 else 0.,
            "triangle_small": small, "triangle_middle": mid,
            "triangle_small_mean":float(np.mean(small_means)) if small_means else None,
            "triangle_middle_mean":float(np.mean(middle_means)) if middle_means else None,
            "triangle_eligible_realizations":len(small_means),"normalized_matrices":matrix_rows,
            "zero_distance_atom": float(np.mean(atoms)),
            "degenerate_triangles": float(np.mean(degenerates)) if degenerates else None,
            "realizations": len(records)}


def discrepancy(left, right):
    """Declared exploratory objective: pair law + root dispersion + triangles.

    Ball means duplicate pair CDFs and are deliberately not double-counted.
    Weights are descriptive choices, not a likelihood or a formal GOF test.
    """
    grid=np.unique(np.r_[left["integration_grid"],right["integration_grid"]])
    delta=np.interp(grid,left["integration_grid"],left["integration_cdf"],right=1.)-np.interp(grid,right["integration_grid"],right["integration_cdf"],right=1.)
    pair_w1 = float(np.trapezoid(abs(delta), grid))
    cvm = float(np.trapezoid(delta*delta, grid))
    ball_dispersion = float(np.mean(abs(left["ball_sd"]-right["ball_sd"])))
    triangle = None
    if left["triangle_small_mean"] is not None and right["triangle_small_mean"] is not None:
        triangle = abs(left["triangle_small_mean"]-right["triangle_small_mean"]) + abs(left["triangle_middle_mean"]-right["triangle_middle_mean"])
    return {"pair_w1": pair_w1, "pair_cvm_integral": cvm, "pair_ks": float(abs(delta).max()),
            "ball_dispersion_l1": ball_dispersion, "triangle_mean_l1": triangle,
            "global_scale_cv_difference": abs(left["scale_cv"]-right["scale_cv"]),
            "ball_sd_half_signed":float(left["ball_sd"][1]-right["ball_sd"][1]),
            "triangle_small_signed":None if triangle is None else left["triangle_small_mean"]-right["triangle_small_mean"],
            "triangle_middle_signed":None if triangle is None else left["triangle_middle_mean"]-right["triangle_middle_mean"],
            "global_scale_cv_signed":left["scale_cv"]-right["scale_cv"],
            "objective": pair_w1+ball_dispersion+(triangle or 0.)}


def bootstrap_comparison(left_records, right_records, rng, *, draws=200):
    """Independent whole-realization bootstrap, recomputing ensemble medians."""
    point = discrepancy(summarize(left_records), summarize(right_records))
    samples = {k: [] for k,v in point.items() if v is not None}
    for _ in range(draws):
        left = [left_records[i] for i in rng.integers(len(left_records), size=len(left_records))]
        right = [right_records[i] for i in rng.integers(len(right_records), size=len(right_records))]
        score = discrepancy(summarize(left), summarize(right))
        for key in samples: samples[key].append(score[key])
    return {"point": point, "mc95": {key: np.quantile(value,[.025,.975]).tolist()
                                     for key,value in samples.items()},
            "draws": draws, "independence_unit": "physical field or graph, all roots retained"}
