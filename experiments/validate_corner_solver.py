"""Task 2C small MPS audit; checkpoint each case/order, never replace results.

Cost: ~1 minute, <100 MB with one BLAS thread, no Brownian paths generated.
Run OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m
experiments.validate_corner_solver --output results/phase2c/angular
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from src.corner_spectral import fit_corner_mode, triangle_geometry, collocation_points
from src.doob_excursions import analytic_cone


CASES = {
    "orthant": (np.eye(3), 3.),
    "a2": (np.array([[1., -.5, 0], [-.5, 1, 0], [0, 0, 1]]), 4.),
    "positive_birectangular": (np.array([[1., .5, 0], [.5, 1, 0], [0, 0, 1]]), 2.5),
    "a3": (np.array([[1., -.5, 0], [-.5, 1, -.5], [0, -.5, 1]]), 6.),
    "equicorrelated_02": (np.full((3, 3), .2)+np.eye(3)*.8, None),
    "unequal": (np.array([[1., -.2, .1], [-.2, 1, .35], [.1, .35, 1]]), None),
}


def sphere_quadrature(n=30):
    z, weights = leggauss(n)
    phi = 2*np.pi*(np.arange(2*n)+.37)/(2*n)
    directions = np.column_stack([np.repeat(np.sqrt(1-z*z), 2*n)*np.tile(np.cos(phi), n),
                                  np.repeat(np.sqrt(1-z*z), 2*n)*np.tile(np.sin(phi), n),
                                  np.repeat(z, 2*n)])
    return directions, np.repeat(weights, 2*n)/(4*n)


def run_case(name, C, exact_degree, order):
    started = time.perf_counter()
    R, vertices, frames, angles = triangle_geometry(C)
    center = vertices.sum(axis=0)
    center /= np.linalg.norm(center)
    # Broad brackets are informed by the separately computed P1 eigenvalue,
    # solely to identify the first mode; the P1 eigenfunction is not used.
    if exact_degree is None:
        from src.cone_geometry import cone_eigenpair
        p0 = np.sqrt(cone_eigenpair(C, 4).eigenvalue+.25)-.5
    else:
        p0 = exact_degree
    searched = fit_corner_mode(C, order, degree_bracket=(p0-.3, p0+.3),
                              boundary_count=90, interior_count=200)
    mode = searched if exact_degree is None else fit_corner_mode(
        C, order, degree=exact_degree, boundary_count=90, interior_count=200)
    locals_ = [mode.local_corner_expansion(j, terms=18) for j in range(3)]
    def truth(points):
        points = np.atleast_2d(points)
        if name == "a3":
            gaps = points @ R.T
            a, b, c = gaps.T
            return a*b*c*(a+b)*(b+c)*(a+b+c)/np.linalg.norm(points, axis=1)**6
        return analytic_cone(C).angular_mode(points)
    norm = truth(center)[0] if exact_degree is not None else None
    epsilon = np.logspace(-2, -9, 8)
    corners, edges, interfaces = [], [], []
    for j in range(3):
        x = np.tile(epsilon[:, None], (1, 3)); x[:, j] = 1
        points = np.linalg.solve(R, x.T).T
        values = mode.angular_mode(points)
        local_values = locals_[j].angular_mode(points)
        target = truth(points)/norm if norm is not None else None
        slopes = np.diff(np.log(np.abs(values)))/np.diff(np.log(epsilon))
        local_slopes = np.diff(np.log(np.abs(local_values)))/np.diff(np.log(epsilon))
        corners.append(dict(vertex=j, expected_exponent=float(np.pi/angles[j]),
            global_value=values.tolist(), local_value=local_values.tolist(),
            exact_value=target.tolist() if target is not None else None, global_slopes=slopes.tolist(),
            local_slopes=local_slopes.tolist()))
        x = np.ones((len(epsilon), 3)); x[:, j] = epsilon
        edge_points = np.linalg.solve(R, x.T).T
        edges.append(dict(face=j, global_value=mode.angular_mode(edge_points).tolist(),
                          exact_value=(truth(edge_points)/norm).tolist() if norm else None))
        theta = np.r_[np.geomspace(1e-7, .45, 60),
                      1-np.geomspace(1e-7, .45, 60)]*angles[j]
        frame = frames[j]; r = locals_[j].radius
        points = np.cos(r)*frame[0]+np.sin(r)*(np.cos(theta)[:, None]*frame[1]
                                                 +np.sin(theta)[:, None]*frame[2])
        g, l = mode.angular_mode(points), locals_[j].angular_mode(points)
        interfaces.append(dict(vertex=j, max_absolute_over_peak=float(np.max(np.abs(g-l))/np.max(np.abs(g))),
                               max_relative=float(np.max(np.abs(g-l)/np.maximum(np.abs(g), 1e-300)))))
    boundary, interior = collocation_points(vertices, 251, 1000, 77291)
    probe = mode.angular_mode(interior)
    boundary_residual = mode.angular_mode(boundary)
    omega, weights = sphere_quadrature()
    mean_value = []
    for e in [1., .01, 1e-4, 1e-6]:
        point = np.linalg.solve(R, [e, e, 1.]); point /= np.linalg.norm(point)
        radius = .5*np.min(R @ point)
        points = point+radius*omega
        def mean_defect(evaluator):
            value = np.asarray(evaluator(point)).item()/(np.linalg.norm(point)**(mode.degree+1))
            sphere = evaluator(points)/np.linalg.norm(points, axis=1)**(mode.degree+1)
            return float(weights @ sphere/value-1)
        record = dict(epsilon=e, radius=float(radius), global_defect=mean_defect(mode.angular_mode))
        if e < .01:
            record["local_defect"] = mean_defect(locals_[2].angular_mode)
        if exact_degree is not None:
            record["analytic_defect"] = mean_defect(lambda x: truth(x)/norm)
        mean_value.append(record)
    return dict(case=name, covariance=C.tolist(), order=order,
        degree=float(mode.degree), searched_degree=float(searched.degree),
        exact_degree=exact_degree, eigenvalue=float(mode.eigenvalue),
        searched_tension=searched.boundary_tension, tension=mode.boundary_tension,
        condition_number=mode.condition_number, epsilon=epsilon.tolist(),
        corners=corners, edges=edges, interfaces=interfaces, mean_value=mean_value,
        min_random_interior=float(probe.min()), negative_random_interior=int(np.sum(probe<=0)),
        max_boundary_residual=float(np.max(np.abs(boundary_residual))),
        elapsed_seconds=time.perf_counter()-started)


def extended_harmonicity(output):
    """Separate resolution/radius/edge checks, append-only checkpoint records.

    Estimated cost <2 minutes and ~150 MB. The one-off pilot main audit took
    10 seconds, so this ~5x evaluator workload remains comfortably bounded.
    Negative centers/values are retained and not interpreted as a Doob law.
    """
    path = output/"harmonicity_grid.jsonl"
    done = set()
    if path.exists():
        done = {tuple(json.loads(line)[k] for k in ["case", "order", "location", "fraction", "quadrature"])
                for line in path.read_text().splitlines()}
    locations = {"interior": [1., 1., 1.], "edge_01": [.01, 1., 1.],
                 "edge_0001": [1e-4, 1., 1.], "corner_01": [.01, .01, 1.],
                 "corner_0001": [1e-4, 1e-4, 1.]}
    for name in ["orthant", "positive_birectangular", "a3", "equicorrelated_02", "unequal"]:
        C, exact = CASES[name]; R = np.linalg.cholesky(C)
        for order in [8, 20]:
            row = json.loads((output/f"{name}_order{order}.json").read_text())
            mode = fit_corner_mode(C, order, degree=row["degree"], boundary_count=90, interior_count=200)
            for location, gaps in locations.items():
                center = np.linalg.solve(R, gaps); center /= np.linalg.norm(center)
                central = mode.angular_mode(center)
                for fraction in [.25, .5, .8]:
                    radius = fraction*np.min(R @ center)
                    for quadrature in [16, 32]:
                        key = (name, order, location, fraction, quadrature)
                        if key in done: continue
                        directions, weights = sphere_quadrature(quadrature)
                        points = center+radius*directions
                        values = mode.angular_mode(points)/np.linalg.norm(points, axis=1)**(mode.degree+1)
                        result = dict(case=name, order=order, location=location, fraction=fraction,
                            quadrature=quadrature, center_value=float(central),
                            min_sphere_value=float(values.min()), mean_value_defect=float(weights @ values/central-1))
                        with path.open("a") as stream: stream.write(json.dumps(result)+"\n")
            print("harmonicity", name, order, flush=True)


def interface_derivatives(output):
    """Finite-difference matching diagnostic; not an interval error bound."""
    path = output/"interface_derivatives.json"
    if path.exists(): return
    results = []
    for name, (C, _) in CASES.items():
        saved = json.loads((output/f"{name}_order20.json").read_text())
        mode = fit_corner_mode(C, 20, degree=saved["degree"], boundary_count=90, interior_count=200)
        for j in range(3):
            local = mode.local_corner_expansion(j, terms=18)
            frame = mode.frames[j]; angle = mode.angles[j]
            theta = np.r_[np.geomspace(1e-7, .45, 60), 1-np.geomspace(1e-7, .45, 60)]*angle
            tangent = np.cos(theta)[:, None]*frame[1]+np.sin(theta)[:, None]*frame[2]
            def evaluate(evaluator, r):
                return evaluator(np.cos(r)*frame[0]+np.sin(r)*tangent)
            r = local.radius
            for step in [1e-4, 3e-5, 1e-5]:
                dg = (evaluate(mode.angular_mode, r+step)-evaluate(mode.angular_mode, r-step))/(2*step)
                dl = (3*evaluate(local.angular_mode, r)-4*evaluate(local.angular_mode, r-step)
                      +evaluate(local.angular_mode, r-2*step))/(2*step)
                results.append(dict(case=name, vertex=j, step=step,
                    max_absolute_over_derivative_peak=float(np.max(np.abs(dg-dl))/np.max(np.abs(dg))),
                    max_relative=float(np.max(np.abs(dg-dl)/np.maximum(np.abs(dg), 1e-300)))))
    path.write_text(json.dumps(results, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase2c/angular"))
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 8, 12, 16, 20])
    parser.add_argument("--extra-mean-value", action="store_true")
    parser.add_argument("--extra-interface", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (C, p) in CASES.items():
        for order in args.orders:
            path = args.output/f"{name}_order{order}.json"
            if path.exists():
                print(f"preserved {path}", flush=True); continue
            row = run_case(name, C, p, order)
            path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
            print(name, order, row["degree"], row["searched_tension"], row["elapsed_seconds"], flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["orthant", "a2", "a3"]:
        row = json.loads((args.output/f"{name}_order{max(args.orders)}.json").read_text())
        c = row["corners"][0]; target = np.asarray(c["exact_value"])
        for key, style in [("global_value", "-"), ("local_value", "--")]:
            axes[0].loglog(row["epsilon"], np.maximum(np.abs(np.asarray(c[key])/target-1), 1e-16),
                           style, label=f"{name}: {'global' if style=='-' else 'local arc'}")
    for name in ["unequal", "equicorrelated_02"]:
        rows = [json.loads((args.output/f"{name}_order{o}.json").read_text()) for o in args.orders]
        axes[1].semilogy(args.orders, [r["searched_tension"] for r in rows], "o-", label=name)
    axes[0].set(xlabel="Correlated small gaps epsilon", ylabel="Relative corner error", title="Exact controls; vertex 0")
    axes[1].set(xlabel="Modes per expansion", ylabel="Boundary tension", title="Generic numerical convergence")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    figure_path = args.output/"mps_diagnostics.png"
    if not figure_path.exists(): fig.savefig(figure_path, dpi=170)
    if args.extra_mean_value: extended_harmonicity(args.output)
    if args.extra_interface: interface_derivatives(args.output)


if __name__ == "__main__":
    main()
