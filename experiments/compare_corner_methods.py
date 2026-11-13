"""Independent MPS/Ritz mode comparisons, no sampler or exact generic claim.

Estimated cost: <1 minute, <200 MB (8 small variational fits, degree<=10).
Checkpoints are per covariance/degree; old outputs are never replaced.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.corner_ritz import fit_corner_ritz
from src.corner_spectral import fit_corner_mode, triangle_geometry
from experiments.validate_corner_solver import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase2c/angular/comparison"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    epsilon = np.logspace(-2, -10, 9)
    for name in ["equicorrelated_02", "unequal"]:
        C = CASES[name][0]; R, vertices, _, angles = triangle_geometry(C)
        center = vertices.sum(axis=0); center /= np.linalg.norm(center)
        mps = fit_corner_mode(C, 20, degree_bracket=(2., 3.2), boundary_count=90, interior_count=200)
        rng = np.random.default_rng(114732)
        regular = rng.dirichlet([2.]*3, size=1000) @ vertices
        corners = []
        for j in range(3):
            gap = np.tile(epsilon[:, None], (1, 3)); gap[:, j] = 1
            points = np.linalg.solve(R, gap.T).T
            corners.append((points, mps.local_corner_expansion(j, terms=18)))
        for degree in [4, 6, 8, 10]:
            path = args.output/f"{name}_degree{degree}.json"
            if path.exists(): print("preserved", path, flush=True); continue
            start = time.perf_counter()
            try:
                ritz = fit_corner_ritz(C, degree=degree, quadrature_order=80)
            except ValueError as error:
                path.write_text(json.dumps(dict(case=name, degree=degree, error=str(error)), indent=2)+"\n")
                continue
            norm = ritz.angular_mode(center)
            regular_ratio = ritz.angular_mode(regular)/norm/mps.angular_mode(regular)
            corner_rows, edge_rows = [], []
            for j, (points, local) in enumerate(corners):
                local_values = local.angular_mode(points)
                ritz_values = ritz.angular_mode(points)/norm
                corner_rows.append(dict(vertex=j, expected_exponent=float(np.pi/angles[j]),
                    ritz_value=ritz_values.tolist(), local_mps_value=local_values.tolist(),
                    global_mps_value=mps.angular_mode(points).tolist(),
                    ritz_over_local_mps=(ritz_values/local_values).tolist()))
                gap = np.ones((len(epsilon), 3)); gap[:, j] = epsilon
                points = np.linalg.solve(R, gap.T).T
                edge_rows.append(dict(face=j, ritz_value=(ritz.angular_mode(points)/norm).tolist(),
                    global_mps_value=mps.angular_mode(points).tolist(),
                    ritz_over_global_mps=(ritz.angular_mode(points)/norm/mps.angular_mode(points)).tolist()))
            result = dict(case=name, covariance=C.tolist(), degree=degree,
                quadrature_order=80, mps_order=20, mps_degree=mps.degree,
                mps_eigenvalue=mps.eigenvalue, ritz_eigenvalue=ritz.eigenvalue,
                ritz_mass_condition_number=ritz.mass_condition_number,
                positivity=ritz.positivity_certificate(), epsilon=epsilon.tolist(),
                regular_max_relative=float(np.max(abs(regular_ratio-1))),
                regular_rms_relative=float(np.sqrt(np.mean((regular_ratio-1)**2))),
                regular_quantiles=np.quantile(regular_ratio-1, [0, .05, .5, .95, 1]).tolist(),
                corners=corner_rows, edges=edge_rows, elapsed_seconds=time.perf_counter()-start,
                seed=114732)
            path.write_text(json.dumps(result, indent=2)+"\n")
            print(name, degree, result["ritz_eigenvalue"], result["regular_max_relative"],
                  result["elapsed_seconds"], flush=True)


if __name__ == "__main__": main()
