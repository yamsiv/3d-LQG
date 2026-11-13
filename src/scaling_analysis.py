"""Graph-cluster bootstrap and transparent finite-size scaling diagnostics.

Naive log--log slopes are finite-size effective exponents, not estimates with
controlled asymptotic bias. Confidence intervals account for Monte Carlo graph
variability only. Shifted histogram collapse follows Budd--Castro §4.2 up to the
final nonlinear correction ansatz, which is not identifiable in tiny studies.
"""
from __future__ import annotations

import numpy as np


def pooled_histogram(records):
    """Average graph-normalized histograms (equal graph weights)."""
    width = max(len(record["distance_histogram"]) for record in records)
    out = np.zeros(width)
    for record in records:
        hist = np.asarray(record["distance_histogram"], dtype=float)
        out[:len(hist)] += hist / hist.sum()
    return out / len(records)


def histogram_statistic(hist, statistic="mean"):
    hist = np.asarray(hist, dtype=float)
    hist = hist / hist.sum()
    if statistic == "mean":
        return float(np.dot(np.arange(len(hist)), hist))
    quantiles = {"median": .5, "q25": .25, "q75": .75, "q90": .9}
    if statistic not in quantiles:
        raise ValueError(f"unknown statistic {statistic}")
    return float(np.searchsorted(np.cumsum(hist), quantiles[statistic]))


def _log_fit(sizes, values):
    if len(sizes) < 3 or np.any(np.asarray(values) <= 0):
        raise ValueError("need at least three sizes with positive observable")
    logn, logy = np.log(sizes), np.log(values)
    slope, intercept = np.polyfit(logn, logy, 1)
    residual = logy - intercept - slope * logn
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual**2)))


def bootstrap_scaling(records, *, statistic="mean", bootstrap=500, seed=0,
                      min_n=None, observable="distance"):
    """Resample independent graphs *within each size*, then refit log slopes.

    Quantiles refer to the pooled pair-distance law, not the mean of graphwise
    quantiles. For diameter_lower/upper use mean graphwise bound as observable.
    Diameter bound exponents do not themselves bound the limiting exponent.
    """
    groups = {}
    for record in records:
        if min_n is None or record["n"] >= min_n:
            groups.setdefault(int(record["n"]), []).append(record)
    sizes = np.asarray(sorted(groups), dtype=int)

    def evaluate(group):
        if observable == "distance":
            return histogram_statistic(pooled_histogram(group), statistic)
        if observable in ("diameter_lower", "diameter_upper"):
            return float(np.mean([r["diameter"][observable.split("_")[1]] for r in group]))
        raise ValueError("unknown observable")

    values = np.asarray([evaluate(groups[n]) for n in sizes])
    slope, intercept, rms = _log_fit(sizes, values)
    rng = np.random.default_rng(seed)
    slopes = []
    boot_values = []
    for _ in range(bootstrap):
        sample_values = []
        for n in sizes:
            group = groups[n]
            sample_values.append(evaluate([group[i] for i in rng.integers(len(group), size=len(group))]))
        slopes.append(_log_fit(sizes, sample_values)[0])
        boot_values.append(sample_values)
    slopes = np.asarray(slopes)
    positive = slopes > 0
    dimensions = 1 / slopes[positive]
    value_ci = np.quantile(boot_values, [.025, .975], axis=0).T
    return {"statistic": statistic, "observable": observable, "sizes": sizes.tolist(),
            "graphs_per_size": [len(groups[n]) for n in sizes], "values": values.tolist(),
            "values_ci95": value_ci.tolist(), "slope": slope, "intercept": intercept,
            "effective_dimension": float(1 / slope) if slope > 0 else None,
            "slope_ci95": np.quantile(slopes, [.025, .975]).tolist(),
            "dimension_ci95": np.quantile(dimensions, [.025, .975]).tolist() if len(dimensions) else None,
            "nonpositive_bootstrap_slopes": int((~positive).sum()), "log_residual_rms": rms,
            "bootstrap": bootstrap, "bootstrap_seed": seed,
            "uncertainty": "independent_graph_cluster_bootstrap_MC_only_no_asymptotic_bias"}


def window_stability(records, *, statistic="mean", bootstrap=500, seed=0,
                     observable="distance"):
    sizes = sorted({r["n"] for r in records})
    return [bootstrap_scaling(records, statistic=statistic, bootstrap=bootstrap,
                              seed=seed + i, min_n=n, observable=observable)
            for i, n in enumerate(sizes[:-2])]


def fixed_offset_sensitivity(records, offsets=(-2., -1., 0., 1., 2., 3.)):
    """Expose how a lattice-scale offset changes the apparent mean exponent.

    This is a sensitivity analysis, not a fit selecting an offset for agreement
    with a target dimension. All offsets are reported without preference.
    """
    sizes = sorted({r["n"] for r in records})
    out = []
    for minimum in sizes[:-2]:
        use = [n for n in sizes if n >= minimum]
        means = [histogram_statistic(pooled_histogram([r for r in records if r["n"] == n])) for n in use]
        for offset in offsets:
            if np.any(np.asarray(means) + offset <= 0):
                continue
            slope, intercept, rms = _log_fit(use, np.asarray(means) + offset)
            out.append({"min_n": minimum, "offset": offset,
                        "effective_dimension": float(1 / slope) if slope > 0 else None,
                        "log_residual_rms": rms})
    return out


def collapse_histograms(records):
    """Two-stage shift/scale least-squares collapse, Eq.(44) and §4.2.

    This diagnostic fits k_n and s_n, averages the first-pass shifts, then refits
    k_n with common shift. A pure log slope of k_n is returned. The paper's final
    Eq.(45) finite-size correction fit requires larger data and is not attempted.
    Finite graph statistics can make this nonlinear optimization ill-conditioned.
    """
    from scipy.optimize import least_squares
    groups = {}
    for record in records:
        groups.setdefault(record["n"], []).append(record)
    sizes = sorted(groups)
    reference = pooled_histogram(groups[sizes[-1]])
    x = np.linspace(0, len(reference) + 2, max(256, len(reference) * 8))
    target = np.interp(x, np.arange(len(reference)), reference, left=0, right=0)
    first = []
    for n in sizes[:-1]:
        hist = pooled_histogram(groups[n])
        mean_n = histogram_statistic(hist)
        mean_ref = histogram_statistic(reference)

        def residual(parameters):
            k, shift = parameters
            return np.interp((x + shift) / k - shift, np.arange(len(hist)), hist,
                             left=0, right=0) / k - target

        fit = least_squares(residual, [mean_ref / mean_n, 1.],
                            bounds=([.25, -5.], [20., 20.]), max_nfev=500)
        first.append({"n": n, "scale": float(fit.x[0]), "shift": float(fit.x[1]),
                      "rms": float(np.sqrt(np.mean(fit.fun**2))), "success": bool(fit.success)})
    shift = float(np.mean([f["shift"] for f in first]))
    second = []
    for n, previous in zip(sizes[:-1], first):
        hist = pooled_histogram(groups[n])

        def residual(k):
            return np.interp((x + shift) / k[0] - shift, np.arange(len(hist)), hist,
                             left=0, right=0) / k[0] - target

        fit = least_squares(residual, [previous["scale"]], bounds=([.25], [20.]))
        second.append({"n": n, "scale": float(fit.x[0]),
                       "rms": float(np.sqrt(np.mean(fit.fun**2))), "success": bool(fit.success)})
    fit_sizes = np.asarray([f["n"] for f in second] + [sizes[-1]])
    scales = np.asarray([f["scale"] for f in second] + [1.])
    slope, intercept, rms = _log_fit(fit_sizes, scales)
    return {"reference_n": sizes[-1], "common_shift": shift, "first_pass": first,
            "second_pass": second, "scale_log_slope": slope,
            "effective_dimension": float(-1 / slope) if slope < 0 else None,
            "scale_log_rms": rms, "final_correction_ansatz_45_fitted": False,
            "warning": "diagnostic_MC_and_finite_size_uncertainty_not_controlled"}


def bootstrap_collapse(records, *, bootstrap=100, seed=0):
    """Graph-cluster Monte Carlo uncertainty and fit-window collapse sensitivity.

    The same two-stage histogram optimization is repeated for each resampled
    collection of graphs; the largest-size reference is resampled as well.
    This captures reference-curve uncertainty. It does not remove finite-size
    bias or fit the paper's final correction ansatz (45).
    """
    groups = {}
    for record in records:
        groups.setdefault(record["n"], []).append(record)
    sizes = sorted(groups)
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(bootstrap):
        resampled = []
        for n in sizes:
            group = groups[n]
            resampled.extend(group[i] for i in rng.integers(len(group), size=len(group)))
        fits.append(collapse_histograms(resampled))
    valid = [f for f in fits if f["effective_dimension"] is not None]
    window = []
    for n in sizes[:-2]:
        fit = collapse_histograms([r for r in records if r["n"] >= n])
        window.append({"minimum_n": n, **fit})
    central = collapse_histograms(records)
    return {"central": central, "window_sensitivity": window,
            "bootstrap": bootstrap, "bootstrap_seed": seed,
            "valid_bootstrap_fits": len(valid),
            "dimension_ci95": np.quantile([f["effective_dimension"] for f in valid], [.025, .975]).tolist() if valid else None,
            "common_shift_ci95": np.quantile([f["common_shift"] for f in fits], [.025, .975]).tolist(),
            "bootstrap_dimensions": [f["effective_dimension"] for f in fits],
            "uncertainty": "graph_cluster_bootstrap_MC_only_no_asymptotic_bias",
            "final_correction_ansatz_45_fitted": False}
