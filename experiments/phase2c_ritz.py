"""Corner-factor Galerkin validation with preserved per-fit checkpoints."""
import json
import os
from pathlib import Path
import time
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/lqg-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from src.corner_ritz import fit_corner_ritz
from src.doob_excursions import analytic_cone
from src.weyl_a3 import a3_covariance,A3AnalyticCone
from src.provenance import environment_manifest


def sphere_rule(order):
    z,w=leggauss(order);phi=np.arange(2*order)*np.pi/order
    points=np.column_stack((np.repeat(np.sqrt(1-z*z),2*order)*np.tile(np.cos(phi),order),
                            np.repeat(np.sqrt(1-z*z),2*order)*np.tile(np.sin(phi),order),
                            np.repeat(z,2*order)))
    return points,np.repeat(w/(4*order),2*order)


def main():
    output=Path("results/phase2c/ritz");output.mkdir(parents=True,exist_ok=True)
    metadata=output/"metadata.json"
    if not metadata.exists():metadata.write_text(json.dumps(environment_manifest(),indent=2))
    pair=np.eye(3);pair[0,1]=pair[1,0]=-.5
    positive=np.eye(3);positive[0,1]=positive[1,0]=.5
    generic=np.array([[1.,-.2,.1],[-.2,1.,.35],[.1,.35,1.]])
    equicorr=np.full((3,3),.2);np.fill_diagonal(equicorr,1.)
    cases=[("i3",np.eye(3),0),("pair_negative",pair,0),("pair_positive",positive,0),
           ("a3",a3_covariance(),1),("unequal",generic,None),("equicorr_02",equicorr,None)]
    directions,weights=sphere_rule(32)
    allrows=[]
    for name,C,exact_degree in cases:
        R=np.linalg.cholesky(C)
        oracle=A3AnalyticCone() if name=="a3" else (analytic_cone(C) if exact_degree is not None else None)
        for degree in ([exact_degree,exact_degree+2] if exact_degree is not None else [0,2,4,6,8,10,12]):
            for quadrature in ([48,80] if degree<=8 else [80,120]):
                path=output/f"{name}_d{degree}_q{quadrature}.json"
                if path.exists():allrows.append(json.loads(path.read_text()));continue
                start=time.perf_counter()
                try:mode=fit_corner_ritz(C,degree,quadrature)
                except Exception as exc:
                    row=dict(name=name,C=C.tolist(),degree=degree,quadrature=quadrature,error=repr(exc))
                    path.write_text(json.dumps(row,indent=2));allrows.append(row);continue
                center=np.linalg.solve(R,np.ones(3));center/=np.linalg.norm(center)
                scale=oracle.angular_mode(center)/mode.angular_mode(center) if oracle is not None else 1.
                boundary=[];harmonic=[]
                for location in ["face","corner"]:
                    for index in range(3):
                        for eps in [1e-2,1e-4,1e-6,1e-8,1e-10]:
                            x=np.ones(3) if location=="face" else np.full(3,eps)
                            x[index]=eps if location=="face" else 1.
                            point=np.linalg.solve(R,x);point/=np.linalg.norm(point)
                            m=mode.angular_mode(point)
                            row=dict(location=location,index=index,epsilon=eps,mode=m)
                            if oracle is not None:
                                truth=oracle.angular_mode(point)
                                row.update(exact=truth,relative_error=float(scale*m/truth-1))
                            boundary.append(row)
                            if index==0 and eps in [1e-2,1e-4,1e-6]:
                                for fraction in [.25,.5,.75]:
                                    radius=fraction*np.min(R@point)
                                    probe=point+radius*directions
                                    relative=np.exp(mode.log_h(probe)-mode.log_h(point))
                                    defect=float(weights@relative-1)
                                    harmonic.append(dict(location=location,epsilon=eps,ball_fraction=fraction,
                                                         relative_mean_value_defect=defect))
                row=dict(name=name,C=C.tolist(),degree=degree,quadrature=quadrature,
                    eigenvalue=mode.eigenvalue,p=mode.p,alpha1=mode.alpha1,
                    mass_condition=mode.mass_condition_number,positivity=mode.positivity_certificate(),
                    boundary=boundary,harmonic=harmonic,seconds=time.perf_counter()-start,
                    status="angular diagnostics only; not a validated excursion sampler")
                path.write_text(json.dumps(row,indent=2));allrows.append(row)
        print(json.dumps({"case":name,"checkpoints":len([r for r in allrows if r["name"]==name])}),flush=True)
    (output/"summary.json").write_text(json.dumps(dict(rows=allrows,scope="corner-aware Ritz angular diagnostics"),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name in ["unequal","equicorr_02"]:
        rows=[r for r in allrows if r["name"]==name and r.get("quadrature")==80 and "eigenvalue" in r]
        reference=10.360059121480615 if name=="unequal" else 8.150124848028183
        axes[0].semilogy([r["degree"] for r in rows],
                        [abs(r["eigenvalue"]-reference) for r in rows],"o-",label=name)
        values=[max(abs(h["relative_mean_value_defect"]) for h in r["harmonic"]) for r in rows]
        axes[1].semilogy([r["degree"] for r in rows],values,"o-",label=name)
    axes[0].set(xlabel="Polynomial trial degree",ylabel="Eigenvalue difference from independent MPS",
                title="Different formulations agree; no eigenfunction-error theorem")
    axes[1].set(xlabel="Polynomial trial degree",ylabel="Maximum probed |mean-value defect|",
                title="Face/corner probes; ball fractions .25,.5,.75")
    axes[0].legend();axes[1].legend();fig.tight_layout()
    fig.savefig(output/"ritz_convergence.png",dpi=180);plt.close(fig)


if __name__=="__main__":main()
