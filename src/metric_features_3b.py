"""Intrinsic, cluster-aware feature summaries for the Phase 3B search.

These are finite metric-probability-space diagnostics, not sufficient statistics
for a continuum law. Every input record is ONE independent graph/field. The
points in its matrix and its root CDFs are dependent conditional observations.
No feature count or number of matrix subsets increases the realization count.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

from src.metric_measure_3a import ensemble_median, evaluate_cdf


BALL_RADII = np.array([.35, .5, .75, 1., 1.5])
PAIR_QUANTILES = np.array([.1, .25, .75, .9, .95])
SHAPE_QUANTILES = np.array([.1, .5, .9])
DEFAULT_WEIGHTS = {
    "pair": 1., "balls": 1.5, "triangles": 1., "quartets": 1.5,
    "matrix_dependence": 1., "global_scale": .5, "mesoscopic": 1.,
    "atoms": .25,
}


def _rescale_record(record, factor):
    result = dict(record)
    for name in ("grid", "root_means", "matrix"):
        if record.get(name) is not None:
            result[name] = np.asarray(record[name], float) * factor
    for name in ("bin_width", "distance_bin_width"):
        if name in result:
            result[name] *= factor
    return result


def _gauge_records(records, gauge, already_transformed):
    if gauge not in ("grounded", "unit_mass"):
        raise ValueError("gauge must be grounded or unit_mass")
    if gauge == "grounded" or already_transformed:
        return records
    result = []
    for record in records:
        # Graphs have unit counting mass and no field shift to apply.
        if record.get("discrete", False) or record.get("gauge") == "unit_mass":
            result.append(record)
            continue
        header = record.get("header", {})
        xi, gamma = record.get("xi"), record.get("gamma")
        if xi is None or gamma is None:
            pairs = header.get("pairs", [])
            pair_index = record.get("pair_index")
            if pair_index is None and len(pairs) == 1:
                pair_index = 0
            if pair_index is None:
                raise ValueError("unit_mass requires record xi/gamma (or pair_index)")
            xi, gamma = pairs[pair_index]
        xi, gamma = float(xi), float(gamma)
        if gamma == 0:
            if xi != 0:
                raise ValueError("unit-mass Weyl shift undefined for gamma=0, xi>0")
            result.append(record)
            continue
        log_mass = record.get("log_total_mass")
        if log_mass is None:
            masses = header.get("gmc", {})
            entry = masses.get(str(gamma), masses.get(f"{gamma:g}"))
            if entry is None:
                raise ValueError("unit_mass requires log_total_mass metadata")
            log_mass = entry.get("log_total_mass")
            if log_mass is None and "total_mass" in entry:
                log_mass = np.log(entry["total_mass"])
        if log_mass is None or not np.isfinite(log_mass):
            raise ValueError("finite log_total_mass required")
        result.append(_rescale_record(record, float(np.exp(-xi * log_mass / gamma))))
    return result


@lru_cache(maxsize=16)
def _subsets(size):
    return (np.asarray(list(combinations(range(size), 3)), int).reshape(-1, 3),
            np.asarray(list(combinations(range(size), 4)), int).reshape(-1, 4))


def matrix_shape_samples(matrix):
    """Permutation-invariant triangle and quartet shapes, preserving collisions.

    For a quartet sort its three opposite-edge sums s1<=s2<=s3. The two
    reported coordinates are (s3-s2)/s3 and (s2-s1)/s3. The first is twice
    Gromov's four-point delta divided by s3; it is zero for tree metrics.
    These finite samples are not a claim about hyperbolicity of the full space.
    """
    d = np.asarray(matrix, float)
    if (d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 4
            or not np.all(np.isfinite(d)) or np.any(d < 0)
            or not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0)):
        raise ValueError("Finite symmetric nonnegative distance matrix, >=4 points required")
    triples, quartets = _subsets(len(d))
    i, j, k = triples.T
    sides = np.sort(np.column_stack((d[i, j], d[i, k], d[j, k])), axis=1)
    positive_t = sides[:, 2] > 0
    triangle = sides[positive_t, :2] / sides[positive_t, 2, None]
    i, j, k, ell = quartets.T
    sums = np.sort(np.column_stack((d[i, j] + d[k, ell],
                                   d[i, k] + d[j, ell],
                                   d[i, ell] + d[j, k])), axis=1)
    positive_q = sums[:, 2] > 0
    quartet = np.column_stack((sums[positive_q, 2] - sums[positive_q, 1],
                               sums[positive_q, 1] - sums[positive_q, 0]))
    quartet /= sums[positive_q, 2, None]
    # All exchangeable shared/disjoint pair products, not independent draws.
    i, j, k = triples.T
    shared = np.mean((d[i, j] * d[i, k] + d[i, j] * d[j, k]
                      + d[i, k] * d[j, k]) / 3)
    i, j, k, ell = quartets.T
    disjoint = np.mean((d[i, j] * d[k, ell] + d[i, k] * d[j, ell]
                        + d[i, ell] * d[j, k]) / 3)
    pairs = d[np.triu_indices(len(d), 1)]
    return {"triangles": triangle, "quartets": quartet,
            "triangle_degenerate": float(1 - positive_t.mean()),
            "quartet_degenerate": float(1 - positive_q.mean()),
            "pair_zero": float(np.mean(pairs == 0)),
            "pair_mean": float(pairs.mean()), "pair_second": float(np.mean(pairs**2)),
            "shared_product": float(shared), "disjoint_product": float(disjoint)}


def _pair_quantiles(records, probabilities, scale):
    # The union of saved grids preserves all graph atoms. Interpolation for
    # continuous records retains their previously declared distance-bin error.
    grid = np.unique(np.concatenate([np.asarray(r["grid"], float) for r in records]))
    cdf = np.mean([evaluate_cdf(r, grid).mean(axis=0) for r in records], axis=0)
    if all(r.get("discrete", False) for r in records):
        values = grid[np.minimum(np.searchsorted(cdf, probabilities, side="left"), len(grid)-1)]
    else:
        # A positive zero atom gives exact zero quantiles, not negative lengths.
        indices = np.minimum(np.searchsorted(cdf, probabilities, side="left"), len(grid)-1)
        previous = np.maximum(indices-1, 0)
        increment = cdf[indices]-cdf[previous]
        fraction = np.divide(probabilities-cdf[previous], increment,
                             out=np.zeros(len(probabilities)), where=increment > 0)
        values = grid[previous] + fraction*(grid[indices]-grid[previous])
        values[indices == 0] = 0.
    return values / scale


def feature_vector(records, *, gauge="grounded", already_transformed=False, scale=None):
    """Return a stable vector plus names, groups, natural scales and diagnostics.

    Ball quantiles are centered by the corresponding pair-CDF mean; ball means
    themselves are deliberately excluded because they duplicate the pair law.
    Matrix quantiles are first computed per realization, then averaged, so a
    field with many nondegenerate subsets receives no extra ensemble weight.
    Missing shape/slope estimates have value zero and valid=False; callers must
    retain this mask. No missing value is interpreted as a geometric zero.

    Unit-mass gauge rescales each field distance by M^(-xi/gamma), without
    changing normalized measure or conditioning the distribution on M=1.
    Supply xi,gamma,log_total_mass in each field record; already_transformed
    permits the caller's explicitly pretransformed record ensemble.
    """
    records = list(records)
    if not records:
        raise ValueError("No independent realizations")
    records = _gauge_records(records, gauge, already_transformed)
    norm = ensemble_median(records) if scale is None else float(scale)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Positive finite ensemble distance normalization required")
    names, groups, values, scales, valid = [], [], [], [], []

    def add(name, family, value, natural_scale):
        good = np.isfinite(value)
        names.append(name); groups.append(family); scales.append(natural_scale)
        values.append(float(value) if good else 0.); valid.append(bool(good))

    for q, value in zip(PAIR_QUANTILES, _pair_quantiles(records, PAIR_QUANTILES, norm)):
        add(f"pair_q{q:g}", "pair", value, 1.)
    add("pair_mean_over_median", "pair",
        np.mean([np.mean(r["root_means"]) for r in records]) / norm, 1.)

    # Equal realization weights, including when records use different root counts.
    balls = [evaluate_cdf(r, BALL_RADII * norm) for r in records]
    means = np.mean([b.mean(axis=0) for b in balls], axis=0)
    second = np.mean([np.mean(b*b, axis=0) for b in balls], axis=0)
    ball_sd = np.sqrt(np.maximum(second - means**2, 0.))
    for index, radius in enumerate(BALL_RADII):
        add(f"ball_sd_r{radius:g}", "balls", ball_sd[index], .1)
        # Quantiles of the equal-realization root mixture (uniform roots within
        # each realization), rather than a root-count-weighted pooled sample.
        samples = np.concatenate([b[:, index] for b in balls])
        weights = np.concatenate([np.full(len(b), 1 / len(b) / len(balls)) for b in balls])
        order = np.argsort(samples, kind="stable")
        quantiles = np.interp([.1, .9], np.cumsum(weights[order]), samples[order])
        for q, value in zip((.1, .9), quantiles):
            add(f"ball_centered_q{q:g}_r{radius:g}", "balls", value-means[index], .1)

    shape_rows = [matrix_shape_samples(np.asarray(r["matrix"]) / norm)
                  for r in records if r.get("matrix") is not None]
    for key, family, labels in (("triangles", "triangles", ("small", "middle")),
                                ("quartets", "quartets", ("delta", "aspect"))):
        eligible = [row[key] for row in shape_rows if len(row[key])]
        for coordinate, label in enumerate(labels):
            mean = np.mean([a[:, coordinate].mean() for a in eligible]) if eligible else np.nan
            add(f"{family}_{label}_mean", family, mean, .25)
            quantiles = (np.mean([np.quantile(a[:, coordinate], SHAPE_QUANTILES)
                                  for a in eligible], axis=0) if eligible else [np.nan]*3)
            for q, value in zip(SHAPE_QUANTILES, quantiles):
                add(f"{family}_{label}_q{q:g}", family, value, .25)
    correlation_valid = False
    if shape_rows:
        mu = np.mean([a["pair_mean"] for a in shape_rows])
        variance = np.mean([a["pair_second"] for a in shape_rows]) - mu**2
        correlation_valid = variance > 1e-14
        for key in ("shared_product", "disjoint_product"):
            numerator = np.mean([a[key] for a in shape_rows]) - mu**2
            add(f"matrix_{key}_correlation", "matrix_dependence",
                numerator/variance if correlation_valid else np.nan, .25)
    else:
        for key in ("shared_product", "disjoint_product"):
            add(f"matrix_{key}_correlation", "matrix_dependence", np.nan, .25)
    realization_means = np.array([np.mean(r["root_means"]) / norm for r in records])
    add("root_mean_distance_cv", "global_scale",
        np.std(realization_means, ddof=1) / np.mean(realization_means)
        if len(records) > 1 else np.nan, .1)

    # Finite-radius growth, not Hausdorff dimension. The zero atom remains in B.
    # Flag contamination instead of silently correcting correlated minima/masses.
    slope_rows = []
    slope_eligible = []
    for r in records:
        b = evaluate_cdf(r, np.array([.35, .5, .75]) * norm)
        atom = np.asarray(r["cdfs"])[:, 0]
        eligible = (b[:, 0] > 0) & (b[:, 2] < .8) & (atom < .1*b[:, 0])
        slope_eligible.append(float(np.mean(eligible)))
        if np.any(eligible):
            slope_rows.append(np.log(b[eligible, 2]/b[eligible, 0]) / np.log(.75/.35))
    for q in (.1, .5, .9):
        add(f"finite_radius_growth_q{q:g}", "mesoscopic",
            np.mean([np.quantile(row, q) for row in slope_rows]) if slope_rows else np.nan, 1.)
    add("finite_radius_growth_eligible", "mesoscopic", np.mean(slope_eligible), 1.)
    zero_atom = np.mean([np.asarray(r["cdfs"])[:, 0].mean() for r in records])
    add("zero_distance_atom", "atoms", zero_atom, .05)
    for key in ("triangle_degenerate", "quartet_degenerate"):
        add(key, "atoms", np.mean([a[key] for a in shape_rows]) if shape_rows else np.nan, .05)
    # Atom-induced KS floor is exact for continuous comparison CDFs.
    graph_jumps = [np.max(np.diff(np.asarray(r["cdfs"]).mean(axis=0), prepend=0.))
                   for r in records if r.get("discrete", False)]
    # This average of per-realization largest jumps is descriptive; compute the
    # exact *ensemble* maximum jump on the union support when every row is graph.
    max_jump = None
    if len(graph_jumps) == len(records):
        grid = np.unique(np.concatenate([np.asarray(r["grid"]) for r in records]))
        cdf = np.mean([evaluate_cdf(r, grid).mean(axis=0) for r in records], axis=0)
        max_jump = float(np.max(np.diff(cdf, prepend=0.)))
    diagnostics = {
        "realizations": len(records), "root_counts": [len(r["cdfs"]) for r in records],
        "matrix_point_counts": [len(r["matrix"]) for r in records if r.get("matrix") is not None],
        "gauge": gauge, "normalization": norm, "shape_realizations": len(shape_rows),
        "triangle_eligible_realizations": sum(bool(len(a["triangles"])) for a in shape_rows),
        "quartet_eligible_realizations": sum(bool(len(a["quartets"])) for a in shape_rows),
        "matrix_correlations_valid": bool(correlation_valid),
        "finite_radius_growth_eligible_fraction": float(np.mean(slope_eligible)),
        "largest_ensemble_graph_atom": max_jump,
        "continuous_cdf_ks_atom_floor": max_jump / 2 if max_jump is not None else None,
        "max_normalized_distance_bin": max(float(r.get("bin_width", r.get("distance_bin_width", 0.)))
                                           for r in records) / norm,
        "cv_contains_finite_root_noise": True,
        "correlations_are_finite_U_statistic_estimates": True,
        "ball_means_not_independent_of_pair_cdf": True,
    }
    # Independent iid roots permit this variance subtraction *in expectation*.
    # The raw estimate may be negative; do not hide that by calling clipped zero
    # a demonstrated absence of field/graph global randomness.
    root_noise = np.mean([np.var(np.asarray(r["root_means"])/norm, ddof=1)
                          / len(r["root_means"]) for r in records
                          if len(r["root_means"]) > 1])
    between = np.var(realization_means, ddof=1) if len(records) > 1 else np.nan
    diagnostics.update(root_mean_sampling_variance=float(root_noise),
                       root_mean_between_variance=float(between),
                       root_mean_deconvolved_variance_raw=float(between-root_noise),
                       deconvolution_assumes_iid_measure_sampled_roots=True)
    return {"vector": np.asarray(values), "names": names, "families": groups,
            "scales": np.asarray(scales), "valid": np.asarray(valid), "diagnostics": diagnostics}


def grouped_feature_distance(left, right, *, noise_scales=None, weights=None):
    """Weighted mean of within-family RMS discrepancies; not a likelihood/test.

    Natural scales are used unless externally calibrated same-model split
    scales are supplied. Group averaging prevents the number of overlapping
    features within a family from silently determining its weight.
    """
    if left["names"] != right["names"] or left["families"] != right["families"]:
        raise ValueError("Feature layouts differ")
    scale = np.asarray(left["scales"] if noise_scales is None else noise_scales, float)
    if scale.shape != np.asarray(left["vector"]).shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Positive finite scale for every feature required")
    selected = np.asarray(left["valid"]) & np.asarray(right["valid"])
    delta = (np.asarray(left["vector"]) - np.asarray(right["vector"])) / scale
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Feature-family weights must be finite and nonnegative")
    groups = np.asarray(left["families"])
    components = {family: float(np.sqrt(np.mean(delta[(groups == family) & selected]**2)))
                  for family in dict.fromkeys(groups)
                  if np.any((groups == family) & selected)}
    denominator = sum(weights.get(family, 0.) for family in components)
    if denominator <= 0:
        raise ValueError("At least one valid positively weighted feature family required")
    total = sum(weights.get(family, 0.) * value for family, value in components.items()) / denominator
    return {"total": float(total), "components": components, "weights": weights,
            "valid_features": int(selected.sum()), "normalization": "family RMS, weighted family mean"}


def split_noise_scales(records, rng, *, draws=32, split_size=None, relative_floor=.05,
                       gauge="grounded", already_transformed=False):
    """RMS feature differences over balanced disjoint physical-record splits.

    Repeated splits reuse the same dataset: they calibrate a conditional noise
    scale, not a collection of independent experiments or a null distribution.
    A natural-scale floor prevents division by accidentally zero empirical noise.
    """
    records = list(records)
    size = len(records)//2 if split_size is None else int(split_size)
    if size < 2 or 2*size > len(records) or draws < 2 or relative_floor <= 0:
        raise ValueError("Need >=2 draws and two disjoint groups of >=2 realizations")
    differences, masks = [], []
    layout = feature_vector(records, gauge=gauge, already_transformed=already_transformed)
    for _ in range(draws):
        order = rng.permutation(len(records))[:2*size]
        a = feature_vector([records[i] for i in order[:size]], gauge=gauge,
                           already_transformed=already_transformed)
        b = feature_vector([records[i] for i in order[size:]], gauge=gauge,
                           already_transformed=already_transformed)
        differences.append(a["vector"] - b["vector"])
        masks.append(a["valid"] & b["valid"])
    delta = np.asarray(differences)
    mask = np.asarray(masks)
    rms = np.sqrt(np.sum(np.where(mask, delta**2, 0.), axis=0) / np.maximum(mask.sum(axis=0), 1))
    floor = relative_floor * np.asarray(layout["scales"])
    return {"scales": np.maximum(rms, floor), "raw_rms": rms, "floor": floor,
            "names": layout["names"], "differences": delta, "valid": mask,
            "draws": draws, "split_size": size,
            "note": "dependent re-splits of one dataset; physical realization is the unit"}


def bootstrap_feature_difference(left_records, right_records, rng, *, draws=100,
                                 paired=False, gauge="grounded", already_transformed=False):
    """Signed feature differences with whole-realization percentile intervals.

    paired=True requires matching physical realization IDs in the same order;
    useful for stencils/gauges/refinements. Independent graph versus field
    ensembles use paired=False. Normalization is recomputed in each replicate.
    """
    left_records, right_records = list(left_records), list(right_records)
    if paired and len(left_records) != len(right_records):
        raise ValueError("Paired ensembles must have equal lengths")
    if paired:
        for a, b in zip(left_records, right_records):
            if a.get("realization") is None or a.get("realization") != b.get("realization"):
                raise ValueError("Paired samples require matching physical realization IDs")
            for key in ("field_seed", "seed"):
                left_seed = a.get("header", {}).get(key)
                right_seed = b.get("header", {}).get(key)
                if (left_seed is not None or right_seed is not None) and left_seed != right_seed:
                    raise ValueError("Paired samples require matching physical realization seeds")
    kwargs = dict(gauge=gauge, already_transformed=already_transformed)
    a, b = feature_vector(left_records, **kwargs), feature_vector(right_records, **kwargs)
    differences = []
    for _ in range(draws):
        il = rng.integers(len(left_records), size=len(left_records))
        ir = il if paired else rng.integers(len(right_records), size=len(right_records))
        fl = feature_vector([left_records[i] for i in il], **kwargs)
        fr = feature_vector([right_records[i] for i in ir], **kwargs)
        delta = fl["vector"] - fr["vector"]
        delta[~(fl["valid"] & fr["valid"])] = np.nan
        differences.append(delta)
    differences = np.asarray(differences)
    interval = np.full((2, len(a["vector"])), np.nan)
    any_valid = np.any(np.isfinite(differences), axis=0)
    interval[:, any_valid] = np.nanquantile(differences[:, any_valid], [.025, .975], axis=0)
    return {"names": a["names"], "point": a["vector"]-b["vector"],
            "valid": a["valid"] & b["valid"],
            "mc95": interval,
            "draws": draws, "paired": paired,
            "independence_unit": "physical graph/field, all roots and matrix subsets retained"}
