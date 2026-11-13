"""Convergence checks for angular eigenvalues, not a graph parameter sweep."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cone_geometry import cone_eigenpair
from src.provenance import environment_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cone_diagnostics")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in [np.pi/3, np.pi/2, 2*np.pi/3]:
        C = np.eye(3)
        C[0, 1] = C[1, 0] = -np.cos(alpha)
        exact = (1 + np.pi/alpha)*(2 + np.pi/alpha)
        for level in [3, 4, 5, 6]:
            start = time.perf_counter()
            mode = cone_eigenpair(C, level)
            rows.append(dict(C=C.tolist(), alpha_angle=alpha, refinement=level,
                             vertices=len(mode.vertices), faces=len(mode.faces),
                             eigenvalue=mode.eigenvalue, analytical=exact,
                             relative_error=abs(mode.eigenvalue/exact-1),
                             seconds=time.perf_counter()-start))
    equicorr = []
    for rho in [-.49, -.45, -.3, 0., .3, .6]:
        C = (1-rho)*np.eye(3) + rho*np.ones((3, 3))
        for level in [4, 5]:
            mode = cone_eigenpair(C, level)
            equicorr.append(dict(rho=rho, C=C.tolist(), refinement=level,
                                 eigenvalue=mode.eigenvalue, alpha1=mode.alpha1,
                                 gamma_s=mode.string_susceptibility,
                                 condition_number=float(np.linalg.cond(C))))
    (output/"data.json").write_text(json.dumps({"analytic_checks": rows,
        "equicorrelated": equicorr, "provenance": environment_manifest(),
        "status": "surface_FEM_geometry_only_not_sampler_validation"}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for alpha in [np.pi/3, np.pi/2, 2*np.pi/3]:
        group = [r for r in rows if r["alpha_angle"] == alpha]
        axes[0].loglog([r["faces"] for r in group], [r["relative_error"] for r in group],
                       "o-", label=f"alpha/pi={alpha/np.pi:.3g}")
    axes[0].set(xlabel="Number of triangular facets", ylabel="Relative eigenvalue error",
                title="Birectangular analytic checks")
    axes[0].legend()
    for level in [4, 5]:
        group = [r for r in equicorr if r["refinement"] == level]
        axes[1].plot([r["rho"] for r in group], [r["gamma_s"] for r in group], "o-",
                     label=f"refinement {level}")
    axes[1].set(xlabel="Equicorrelation rho", ylabel="String susceptibility gamma_s",
                title="Cone exponent (distinct from graph dimension)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output/"cone_eigenvalues.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
