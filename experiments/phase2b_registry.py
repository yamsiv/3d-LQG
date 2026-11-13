"""Requested moderate covariance registry and cone spectral diagnostics only.

This script samples no paths and estimates no graph dimension.
"""
import json
from pathlib import Path
import numpy as np
from src.covariance import from_correlations,equicorrelated,to_angles
from src.doob_excursions import analytic_cone
from src.cone_sampling import enriched_cone_eigenpair
from src.provenance import environment_manifest


def main():
    output=Path("results/phase2b/covariance_registry.json")
    if output.exists():raise FileExistsError(output)
    points=[("independent",np.eye(3)),("A2_halfline",from_correlations(-.5,0,0)),
            ("positive_pair",from_correlations(.5,0,0))]
    points += [(f"equicorr_{r:+.1f}",equicorrelated(r)) for r in [-.2,.2,.4]]
    points += [("unequal",from_correlations(-.2,.1,.35))]
    rows=[]
    for name,C in points:
        row=dict(name=name,C=C.tolist(),angles_rad=list(to_angles(C)),
                 eigenvalues=np.linalg.eigvalsh(C).tolist(),condition_number=float(np.linalg.cond(C)))
        try:
            analytic=analytic_cone(C)
            row.update(spectral_source="analytic_birectangular",lambda1=analytic.eigenvalue,
                       alpha1=analytic.alpha1,gamma_s=1-analytic.alpha1)
        except ValueError:
            estimates=[]
            for level in [4,5,6]:
                mode=enriched_cone_eigenpair(C,level)
                estimates.append(dict(refinement=level,lambda1=mode.eigenvalue,
                                     alpha1=mode.alpha1,gamma_s=mode.string_susceptibility))
            row.update(spectral_source="corner_enriched_FEM_provisional",mesh_estimates=estimates,
                       lambda1=estimates[-1]["lambda1"],alpha1=estimates[-1]["alpha1"],
                       gamma_s=estimates[-1]["gamma_s"])
        rows.append(row)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(provenance=environment_manifest(),configurations=rows,
        interpretation="Cone spectral information, not a covariance-dependent graph study or sampler validation"),indent=2))
    print(output)


if __name__=="__main__":main()
