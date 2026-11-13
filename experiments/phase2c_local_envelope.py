"""Small exact-polynomial local-envelope feasibility probe; no paths/graphs."""
import json
from pathlib import Path
import time
import numpy as np
from src.corner_ritz import fit_corner_ritz
from src.local_polynomial_bounds import polynomial_box_bound
from src.provenance import environment_manifest


def main():
    output=Path("results/phase2c/local_envelope.json")
    if output.exists():raise FileExistsError(output)
    cases=[("unequal",np.array([[1.,-.2,.1],[-.2,1,.35],[.1,.35,1.]]))]
    equi=np.full((3,3),.2);np.fill_diagonal(equi,1.);cases.append(("equicorr_02",equi))
    rows=[]
    for name,C in cases:
        R=np.linalg.cholesky(C)
        for degree in [6,10]:
            mode=fit_corner_ritz(C,degree,80);cert=mode.positivity_certificate()
            for coordinates in [[1.,1.,1.],[.02,1.,1.],[.02,.02,1.]]:
                point=np.linalg.solve(R,coordinates);point/=np.linalg.norm(point)
                d=R@point
                for fraction in [.25,.5]:
                    radius=fraction*d.min();begun=time.perf_counter()
                    bound=polynomial_box_bound(mode,d,radius)
                    old=(1+fraction)**degree*cert["polynomial_simplex_upper"]/cert["polynomial_simplex_lower"]
                    rows.append(dict(case=name,degree=degree,coordinates=coordinates,ball_fraction=fraction,
                        old_global_polynomial_ratio=old,new_local_polynomial_ratio=bound["upper_ratio"],
                        seconds=time.perf_counter()-begun,polynomial_bound=bound))
    output.write_text(json.dumps(dict(provenance=environment_manifest(),rows=rows,
        conclusion="Local polynomial envelopes can improve proposal costs; no generic path or graph validation follows."),indent=2))
    print(output)


if __name__=="__main__":main()
