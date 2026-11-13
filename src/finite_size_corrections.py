"""Additive-shift sensitivity for mean distances; no asymptotic certification.

Fit mean(n)=A*n**p+B by profiling over p and linear least squares for A,B.
This is a diagnostic distinct from Budd--Castro's histogram correction Eq.(45).
"""
import numpy as np
from scipy.optimize import minimize_scalar


def shifted_mean_fit(records, *, min_n=0, bootstrap=500, seed=0):
    sizes = np.asarray(sorted({r["n"] for r in records if r["n"] >= min_n}))
    if len(sizes) < 4:
        raise ValueError("At least four sizes are required for a three-parameter diagnostic")
    groups = [np.array([r["mean_distance"] for r in records if r["n"] == n]) for n in sizes]
    mean = np.array([g.mean() for g in groups])
    se = np.array([g.std(ddof=1)/np.sqrt(len(g)) for g in groups])
    if np.any(se <= 0):
        raise ValueError("At least two variable independent graph observations per size are required")
    nratio = sizes / sizes.max()

    def fit(y):
        def profile(p, full=False):
            design = np.column_stack((nratio**p, np.ones(len(sizes))))
            coefficients = np.linalg.lstsq(design/se[:, None], y/se, rcond=None)[0]
            residual = (design@coefficients-y)/se
            return (float(residual@residual), coefficients) if full else float(residual@residual)
        optimum = minimize_scalar(profile, bounds=(.03, .8), method="bounded",
                                   options={"xatol": 1e-9})
        chi2, coefficients = profile(optimum.x, True)
        return optimum.x, coefficients, chi2

    p, (A, B), chi2 = fit(mean)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        y = np.array([rng.choice(g, len(g), replace=True).mean() for g in groups])
        b_p, (b_A, b_B), b_chi2 = fit(y)
        draws.append([b_p, 1/b_p, b_A, b_B])
    draws = np.asarray(draws)
    return {"sizes": sizes.tolist(), "graphs_per_size": [len(g) for g in groups],
            "model": "E[d]=A*(n/nmax)^p+B; weighted least squares",
            "p": float(p), "effective_dimension": float(1/p),
            "A_at_nmax": float(A), "B": float(B), "chi2": float(chi2),
            "degrees_of_freedom": len(sizes)-3,
            "p_ci95": np.quantile(draws[:, 0], [.025, .975]).tolist(),
            "dimension_ci95": np.quantile(draws[:, 1], [.025, .975]).tolist(),
            "B_ci95": np.quantile(draws[:, 3], [.025, .975]).tolist(),
            "bootstrap_boundary_fraction": float(np.mean((draws[:, 0] < .0301) | (draws[:, 0] > .7999))),
            "bootstrap": bootstrap, "seed": seed,
            "limitation": "Additive-shift model sensitivity only; bootstrap MC uncertainty omits systematic corrections"}
