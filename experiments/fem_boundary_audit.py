"""Audit angular FEM near faces/corners and its harmonic mean-value defect."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lqg-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cone_geometry import cone_eigenpair
from src.cone_sampling import FastFEMMode, enriched_cone_eigenpair
from src.provenance import environment_manifest


def analytic_angular(points, alpha):
    points = np.atleast_2d(np.asarray(points,dtype=float))
    points = points/np.linalg.norm(points,axis=1)[:,None]
    radial_xy = np.linalg.norm(points[:,:2],axis=1)
    theta = np.arctan2(points[:,1],points[:,0])-(np.pi/2-alpha)
    value = radial_xy**(np.pi/alpha)*np.sin(np.pi*theta/alpha)*points[:,2]
    return np.maximum(value,0)


def sphere_quadrature(count=32768):
    index=np.arange(count)
    z=1-2*(index+.5)/count
    phi=index*np.pi*(3-np.sqrt(5))
    return np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/phase2b/fem_audit")
    parser.add_argument("--enriched",action="store_true")
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    path=output/"audit.json"
    if path.exists():
        raise FileExistsError(path)
    start=time.perf_counter()
    sphere=sphere_quadrature()
    rows=[]
    boundary_rows=[]
    for alpha in [np.pi/3,np.pi/2,2*np.pi/3]:
        C=np.eye(3); C[0,1]=C[1,0]=-np.cos(alpha)
        R=np.linalg.cholesky(C)
        reference=np.linalg.solve(R,np.ones(3)); reference/=np.linalg.norm(reference)
        for level in [3,4,5,6]:
            mode=enriched_cone_eigenpair(C,level) if args.enriched else cone_eigenpair(C,level)
            fast=FastFEMMode(mode,C)
            scale=analytic_angular(reference,alpha)[0]/fast.angular_mode(reference)
            zero_faces=np.all(mode.boundary[mode.faces],axis=1)
            for eps in [1e-1,1e-2,1e-3,1e-4,1e-6]:
                for label,correlated in [("face",[eps,1,1]),("corner",[eps,eps,1])]:
                    point=np.linalg.solve(R,correlated); point/=np.linalg.norm(point)
                    numerical=scale*fast.angular_mode(point)
                    truth=analytic_angular(point,alpha)[0]
                    boundary_rows.append(dict(alpha=alpha,refinement=level,location=label,
                        epsilon=eps,exact_mode=truth,numerical_mode=numerical,
                        relative_error=float(numerical/truth-1),interior_zero=bool(numerical==0)))
            for label,correlated in [("center",[1,1,1]),("near_face",[.02,1,1]),
                                     ("near_corner",[.02,.02,1])]:
                point=np.linalg.solve(R,correlated);point/=np.linalg.norm(point)
                radius=.5*np.min(R@point)
                ball=point+radius*sphere
                h_center=np.exp(fast.log_h(point))
                h_ball=np.exp(fast.log_h(ball))
                if h_center>0:
                    defect=float(h_ball.mean()/h_center-1)
                else:
                    defect=None
                ptotal=1+np.pi/alpha
                exponent=-(ptotal+.5)-.5
                # Analytic h=r^(-alpha1-1/2)m(s), alpha1=ptotal+1/2.
                h_exact=analytic_angular(ball,alpha)*np.linalg.norm(ball,axis=1)**exponent
                exact_center=analytic_angular(point,alpha)[0]
                rows.append(dict(alpha=alpha,refinement=level,location=label,
                    eigenvalue=mode.eigenvalue,analytical_eigenvalue=ptotal*(ptotal+1),
                    interior_nonpositive_nodes=int(np.sum(mode.values[~mode.boundary]<=0)),
                    all_boundary_zero_faces=int(zero_faces.sum()),faces=len(mode.faces),
                    mean_value_relative_defect=defect,
                    zero_h_at_center=bool(h_center==0),
                    analytic_quadrature_defect=float(h_exact.mean()/exact_center-1)))
    general=[]
    for triple in [(-.2,-.2,-.2),(.2,.2,.2),(.4,.4,.4),(-.2,.1,.35)]:
        a,b,c=triple; C=np.array([[1,a,b],[a,1,c],[b,c,1.]])
        for level in [4,5,6]:
            mode=enriched_cone_eigenpair(C,level) if args.enriched else cone_eigenpair(C,level)
            general.append(dict(correlations=triple,refinement=level,eigenvalue=mode.eigenvalue,
                interior_negative_nodes=int(np.sum(mode.values[~mode.boundary]<0)),
                all_boundary_zero_faces=int(np.all(mode.boundary[mode.faces],axis=1).sum())))
    path.write_text(json.dumps(dict(provenance=environment_manifest(),analytic_checks=rows,
        boundary_checks=boundary_rows,general_mesh_checks=general,
        elapsed_seconds=time.perf_counter()-start,enriched=args.enriched,
        conclusion="Corner enrichment removes open zero patches but does not certify relative boundary accuracy or harmonicity." if args.enriched else
        "P1 angular mode has open interior corner zero patches; no controlled general h sampler follows from eigenvalue convergence."),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for level in [3,4,5,6]:
        group=[r for r in boundary_rows if r["alpha"]==np.pi/2 and r["refinement"]==level and r["location"]=="corner"]
        axes[0].semilogx([r["epsilon"] for r in group], [r["relative_error"] for r in group],"o-",label=f"refinement {level}")
        group=[r for r in rows if r["alpha"]==np.pi/2 and r["refinement"]==level]
        axes[1].plot([r["location"] for r in group],
                     [r["mean_value_relative_defect"] if r["mean_value_relative_defect"] is not None else np.nan for r in group],"o-",label=f"refinement {level}")
    axes[0].set(xlabel="Positive correlated-coordinate epsilon near cone ray",
        ylabel="Relative angular-mode error",title="Orthant: true mode positive, FEM corner patch zero")
    axes[1].set(ylabel="Mean sphere h / center h - 1",title="Harmonic mean-value defect, ball fraction 1/2")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout();fig.savefig(output/"fem_boundary.png",dpi=180);plt.close(fig)
    print(json.dumps({"output":str(path),"elapsed":time.perf_counter()-start}),flush=True)


if __name__=="__main__":
    main()
