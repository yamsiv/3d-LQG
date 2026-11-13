"""Small interior-ball likelihood audit, no Brownian sampling or graph sweeps.

Cost estimate: <2 minutes, <150 MB. Analyze at most two existing paths/case;
save per-case/leg checkpoints. Generic candidates get spatial diagnostics
and conservative rejection-envelope cost estimates, not uncertified chains.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.corner_ritz import fit_corner_ritz
from src.doob_excursions import analytic_cone
from src.harmonic_likelihood import log_mean_value_ratios, chain_log_likelihood
from src.weyl_a3 import A3AnalyticCone, a3_covariance


def bound_ratio(mode, point, eta=.5):
    """Conservative analytic envelope h_tilde(y)/h_tilde(x) on one ball.

    Requires the stored polynomial's exact-rational Bernstein certificate.
    Floating arithmetic in evaluating the bound is not interval certified.
    """
    cert = mode.positivity_certificate()
    if not cert["certified_positive"]: raise ValueError("Positive polynomial required")
    factor = mode.factor; radius = eta*np.min(factor.R @ point)
    logbound = (3+mode.degree)*np.log1p(eta)
    for opening, nu, e0, e1 in factor.charts:
        if abs(nu-2) < 1e-12: continue
        theta = np.arctan2(point @ e1, point @ e0)
        phi = np.sin(nu*theta)/(np.sin(theta)*np.sin(opening-theta))
        phimax = nu/min(np.sin(opening), np.sin(opening/2))
        logbound += max(nu-2, 0)*np.log1p(eta)+min(nu-2, 0)*np.log1p(-eta)
        logbound += np.log(phimax/phi)
    logbound += np.log(cert["polynomial_simplex_upper"]/cert["polynomial_simplex_lower"])
    exponent = mode.p+1+factor.homogeneity+mode.degree
    logbound -= exponent*np.log1p(-radius/np.linalg.norm(point))
    return float(np.exp(logbound))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("results/phase2c/paths/wos_controls/records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2c/likelihood"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    selected = {name: [] for name in ["i3", "a2", "a3"]}
    for line in args.trace.open():
        record = json.loads(line); name = record["case"]
        if name in selected and len(selected[name]) < 2 and record.get("status") == "success":
            if record["sampler_metadata"]["leg_diagnostics"][0].get("ball_trace"):
                selected[name].append(record)
        if all(len(rows) >= 2 for rows in selected.values()): break
    for name, rows in selected.items():
        C = np.eye(3) if name == "i3" else (a3_covariance() if name == "a3" else np.array([[1., -.5, 0], [-.5, 1, 0], [0, 0, 1]]))
        exact = A3AnalyticCone() if name == "a3" else analytic_cone(C)
        mode = fit_corner_ritz(C, degree=int(name == "a3"), quadrature_order=48)
        for row in rows:
            for leg, diagnostic in enumerate(row["sampler_metadata"]["leg_diagnostics"]):
                path = args.output/f"analytic_{name}_seed{row['seed']}_leg{leg}.json"
                if path.exists(): continue
                trace = diagnostic["ball_trace"]
                centers = np.array([t["center"] for t in trace]); radii = np.array([t["radius"] for t in trace])
                endpoint = np.array(trace[-1]["exit"])
                relative_start = float(mode.log_h(centers[0])-exact.log_h(centers[0]))
                relative_end = float(mode.log_h(endpoint)-exact.log_h(endpoint))
                results = []
                for order in [8, 16]:
                    logs = log_mean_value_ratios(mode.log_h, centers, radii, order)
                    results.append(dict(quadrature=order, log_mean_ratios=logs.tolist(),
                        signed_log_normalizer_sum=float(logs.sum()), absolute_log_normalizer_sum=float(np.abs(logs).sum()),
                        max_mean_value_defect=float(np.max(abs(np.expm1(logs)))),
                        log_likelihood=chain_log_likelihood(relative_start, relative_end, logs)))
                output = dict(case=name, seed=row["seed"], leg=leg, balls=len(trace),
                    degree=mode.degree, eigenvalue=mode.eigenvalue, relative_start_log=relative_start,
                    relative_end_log=relative_end, diagnostics=results,
                    qualification="Point quadrature on observed analytic-control exit chains, not a uniform path-law bound")
                path.write_text(json.dumps(output, indent=2)+"\n")
                print(name, row["seed"], leg, len(trace), results[-1]["log_likelihood"], flush=True)
    for name, C in [("equicorrelated_02", np.full((3,3), .2)+np.eye(3)*.8),
                    ("unequal", np.array([[1., -.2, .1], [-.2, 1, .35], [.1, .35, 1]]))]:
        R = np.linalg.cholesky(C)
        gap_points = [[1., 1., 1.]]; labels = ["interior"]
        for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
            for j in range(3):
                gap = np.full(3, eps); gap[j] = 1.; gap_points.append(gap); labels.append(f"corner{j}_eps{eps}")
                gap = np.ones(3); gap[j] = eps; gap_points.append(gap); labels.append(f"edge{j}_eps{eps}")
        centers = np.linalg.solve(R, np.asarray(gap_points).T).T
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        for degree in [6, 10]:
            path = args.output/f"generic_{name}_degree{degree}.json"
            if path.exists(): continue
            mode = fit_corner_ritz(C, degree, 80)
            diagnostic = []
            for fraction in [.25, .5, .8]:
                radii = fraction*np.min(centers @ R.T, axis=1)
                for order in [8, 16]:
                    logs = log_mean_value_ratios(mode.log_h, centers, radii, order)
                    diagnostic.append(dict(fraction=fraction, quadrature=order,
                        log_mean_ratios=logs.tolist(), max_mean_value_defect=float(np.max(abs(np.expm1(logs))))))
            envelope = bound_ratio(mode, centers[0], .5)
            output = dict(case=name, covariance=C.tolist(), degree=degree, quadrature_order=80,
                labels=labels, eigenvalue=mode.eigenvalue, diagnostics=diagnostic,
                positivity=mode.positivity_certificate(), center_rejection_envelope_ratio=envelope,
                generic_chain_run=False,
                reason="Conservative Bernstein envelope implies excessive rejection cost; no empirical envelope or uncertified chain substituted",
                qualification="Finite spatial grid and sphere quadrature; no uniform defect bound, no generic Brownian paths or graphs")
            path.write_text(json.dumps(output, indent=2)+"\n")
            print(name, degree, envelope, diagnostic[-1]["max_mean_value_defect"], flush=True)
    # Independent finer quadrature preserves the previous checkpoint data.
    supplement = args.output/"generic_quadrature32.json"
    if not supplement.exists():
        refined = []
        for saved in sorted(args.output.glob("generic_*_degree*.json")):
            row = json.loads(saved.read_text()); C = np.asarray(row["covariance"])
            mode = fit_corner_ritz(C, row["degree"], 80); R = np.linalg.cholesky(C)
            gaps = [[1., 1., 1.]]
            for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
                for j in range(3):
                    gap = np.full(3, eps); gap[j] = 1.; gaps.append(gap)
                    gap = np.ones(3); gap[j] = eps; gaps.append(gap)
            centers = np.linalg.solve(R, np.asarray(gaps).T).T
            centers /= np.linalg.norm(centers, axis=1)[:, None]
            for fraction in [.25, .5, .8]:
                radii = fraction*np.min(centers @ R.T, axis=1)
                logs = log_mean_value_ratios(mode.log_h, centers, radii, 32)
                previous = next(d for d in row["diagnostics"] if d["fraction"] == fraction and d["quadrature"] == 16)
                refined.append(dict(case=row["case"], degree=row["degree"], fraction=fraction,
                    quadrature=32, log_mean_ratios=logs.tolist(),
                    max_mean_value_defect=float(np.max(abs(np.expm1(logs)))),
                    max_quadrature16_32_difference=float(np.max(abs(logs-previous["log_mean_ratios"])))))
        supplement.write_text(json.dumps(refined, indent=2)+"\n")
    figure = args.output/"mean_value_defects.png"
    if not figure.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = json.loads(supplement.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
        for ax, name in zip(axes, ["equicorrelated_02", "unequal"]):
            for degree in [6, 10]:
                selected = [r for r in rows if r["case"] == name and r["degree"] == degree]
                ax.semilogy([r["fraction"] for r in selected], [r["max_mean_value_defect"] for r in selected],
                            "o-", label=f"Ritz degree {degree}")
            ax.set(xlabel="Ball radius / nearest face distance", title=name)
            ax.grid(alpha=.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("Maximum observed mean-value defect")
        fig.tight_layout(); fig.savefig(figure, dpi=170)


if __name__ == "__main__": main()
