"""Select exploratory parameter pairs, then analyze independent confirmation.

Raw ensembles are immutable NPZ checkpoints; analysis outputs refuse replacing
completed selections. All bootstrap units are whole graphs or whole fields.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.metric_measure_3a import (summarize,discrepancy,bootstrap_comparison,
    EVAL_GRID,RADII,ensemble_median)


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(jsonable(value),indent=2)+"\n")


def field_groups(directory,*,n=None,stencil=None):
    groups={}
    for path in sorted(Path(directory).glob("metric_*.npz")):
        with np.load(path) as a:
            h=json.loads(str(a["header"]))
            if n is not None and h["sampling_n"]!=n:continue
            if stencil is not None and h["stencil"]!=stencil:continue
            arrays={key:a[key] for key in ["grid","cdfs","root_means","matrices","bin_width"]}
            for j,(xi,gamma) in enumerate(h["pairs"]):
                key=(h["sampling_n"],h["stencil"],xi,gamma)
                groups.setdefault(key,[]).append({"grid":arrays["grid"][j],"cdfs":arrays["cdfs"][j],
                    "root_means":arrays["root_means"][j],"matrix":arrays["matrices"][j],
                    "discrete":False,"realization":h["realization"],"header":h,
                    "bin_width":float(arrays["bin_width"][j])})
    return groups


def graph_groups(directory,*,split=None):
    groups={}
    for path in sorted(Path(directory).glob("graph_*.npz")):
        with np.load(path) as a:
            h=json.loads(str(a["header"]))
            if split=="discovery" and h["realization"]>=24:continue
            if split=="confirmation" and h["realization"]<24:continue
            data={key:a[key] for key in ["grid","cdfs","root_means","matrix"]}
            groups.setdefault((h["case"],h["n"]),[]).append(dict(data,discrete=True,header=h,
                realization=h["realization"]))
    return groups


def selection(args):
    graphs=graph_groups(args.graphs,split="discovery")
    fields=field_groups(args.discovery,n=32,stencil="26")
    rows=[];choices={}
    for case in ["i3","a2","a3"]:
        target=summarize(graphs[(case,8192)])
        case_rows=[]
        for (_,_,xi,gamma),records in fields.items():
            summary=summarize(records);score=discrepancy(target,summary)
            row=dict(case=case,xi=xi,gamma=gamma,**score,
                graph_realizations=len(graphs[(case,8192)]),field_realizations=len(records),
                zero_atom=summary["zero_distance_atom"],tail_mass_at4=summary["tail_mass_at4"])
            rows.append(row);case_rows.append(row)
        choices[case]=sorted(case_rows,key=lambda r:r["objective"])[0]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    result={"selection":"frozen before independent field confirmation",
        "parameter_combinations":len(fields),"graph_parameter_relationships":len(rows),
        "reference":"discovery graph n8192 versus field N32,26-directions",
        "objective":"W1 integral + mean absolute rootballSD difference at declared radii + two triangle ratio mean differences",
        "ball_means_not_double_counted":True,"choices":choices,
        "rows":rows,"confirmation_seed":202611113201,
        "confirmation_pairs":sorted(set([(0.,0.)]+[(r["xi"],r["gamma"]) for r in choices.values()]))}
    save_json(output/"selection.json",result)
    with (output/"scan.csv").open("x") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    xis=sorted({r["xi"] for r in rows});gammas=sorted({r["gamma"] for r in rows})
    for ax,case in zip(axes,["i3","a2","a3"]):
        table=np.array([[next(r["objective"] for r in rows if r["case"]==case and r["xi"]==x and r["gamma"]==g) for g in gammas]for x in xis])
        im=ax.imshow(table,origin="lower",aspect="auto",cmap="viridis_r")
        ax.set(xticks=range(len(gammas)),xticklabels=gammas,yticks=range(len(xis)),yticklabels=xis,xlabel="GMC gamma",ylabel="LFPP xi",title=f"{case.upper()}: discovery objective")
        choice=choices[case];ax.plot(gammas.index(choice["gamma"]),xis.index(choice["xi"]),"rx",ms=12,mew=2)
        fig.colorbar(im,ax=ax,label="Smaller = closer finite ensembles")
    fig.savefig(output/"parameter_scan.png",dpi=170);fig.savefig(output/"parameter_scan.pdf");plt.close(fig)
    print(json.dumps({"choices":choices,"confirmation_pairs":result["confirmation_pairs"]}),flush=True)


def final_analysis(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    select=json.loads(Path(args.selection).read_text())
    graphs=graph_groups(args.graphs,split="confirmation")
    fields=field_groups(args.confirmation)
    if any(len(records)!=24 for records in fields.values()) or len(fields)<30:
        raise ValueError("Confirmation incomplete: require 24 physical fields at every requested size/stencil/pair")
    discovery=field_groups(args.discovery)
    rng=np.random.default_rng(202611113900)
    rows=[];comparison={};summaries={}
    for case in ["i3","a2","a3"]:
        choice=select["choices"][case];pair=(choice["xi"],choice["gamma"])
        target=graphs[(case,8192)]
        for n in [24,32,48]:
            for stencil in ["26","primitive2"]:
                for label,p in [("selected",pair),("flat",(0.,0.))]:
                    key=(n,stencil,*p)
                    if key not in fields:continue
                    data=fields[key]
                    # Bootstrap only the principal N48 comparisons; resolution
                    # series keep paired physical seeds and report point shifts.
                    name=f"{case}_N{n}_{stencil}_{label}"
                    if n==48 and stencil=="primitive2":
                        checkpoint=output/(name+"_bootstrap.json")
                        if checkpoint.exists():
                            fit=json.loads(checkpoint.read_text())
                            if fit["draws"]!=args.bootstrap:raise ValueError("Cached bootstrap draw count differs")
                        else:
                            local_rng=np.random.default_rng(np.random.SeedSequence([202611113900,["i3","a2","a3"].index(case),int(label=="flat")]))
                            fit=bootstrap_comparison(target,data,local_rng,draws=args.bootstrap)
                            save_json(checkpoint,fit)
                    else:
                        fit={"point":discrepancy(summarize(target),summarize(data))}
                    comparison[name]=fit
                    s=summarize(data)
                    rows.append(dict(case=case,n=n,stencil=stencil,kind=label,xi=p[0],gamma=p[1],
                        **fit["point"],field_realizations=len(data),graph_realizations=len(target),
                        gmc_zero_atom=s["zero_distance_atom"],median_scale=s["scale"],tail_mass_at4=s["tail_mass_at4"],
                        ball_sd_half=s["ball_sd"][1],triangle_small=s["triangle_small_mean"],triangle_middle=s["triangle_middle_mean"],
                        cv_global_scale=s["scale_cv"],normalized_histogram_bin_bound=max(r["bin_width"] for r in data)/s["scale"]))
                    if n==48 and stencil=="primitive2":
                        summaries[name]=s
        # Same graph family at two finite sizes, and split-half noise floors.
        comparison[f"{case}_graph_size_shift"]={"point":discrepancy(summarize(graphs[(case,2048)]),summarize(target))}
        old_graph=graph_groups(args.graphs,split="discovery")[(case,8192)]
        comparison[f"{case}_graph_split_floor"]={"point":discrepancy(summarize(old_graph),summarize(target))}
        for n in [24,32]:
            comparison[f"{case}_field_split_floor_N{n}"]={"point":discrepancy(summarize(discovery[(n,"26",*pair)]),summarize(fields[(n,"26",*pair)]))}
        summaries[f"{case}_graph"]=summarize(target)
    save_json(output/"comparisons.json",{"comparisons":comparison,"rows":rows,
        "bootstrap_note":"MC uncertainty only; independent graph and field realizations; ensemble normalizer recomputed; finite-grid discretization/systematics excluded"})
    with (output/"comparisons.csv").open("x") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    slim={key:{k:v for k,v in value.items() if k not in ["normalized_matrices"]}for key,value in summaries.items()}
    save_json(output/"summary_distributions.json",slim)
    plot_final(output,summaries,rows)
    print(json.dumps({"output":str(output),"comparisons":len(comparison),"rows":len(rows)}))


def plot_final(output,summaries,rows):
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,case in zip(axes,["i3","a2","a3"]):
        selected=next(r for r in rows if r["case"]==case and r["kind"]=="selected")
        parameter_label=f"N48, xi={selected['xi']}, gamma={selected['gamma']}"
        for kind,color,label in [("graph","black",f"{case.upper()} graph n8192"),("N48_primitive2_selected","tab:blue",parameter_label),("N48_primitive2_flat","tab:orange","Flat torus control")]:
            s=summaries[f"{case}_{kind}"]
            ax.plot(EVAL_GRID,s["cdf"],color=color,label=label)
        ax.set(xlim=(0,2.5),ylim=(0,1),xlabel="Distance / ensemble pair median",ylabel="Pair CDF",title=case.upper())
        ax.legend(fontsize=7)
    fig.savefig(output/"pair_cdfs.png",dpi=180);fig.savefig(output/"pair_cdfs.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,case in zip(axes,["i3","a2","a3"]):
        for kind,color,label in [("graph","black","Graph"),("N48_primitive2_selected","tab:blue","Selected LFPP/GMC"),("N48_primitive2_flat","tab:orange","Flat control")]:
            s=summaries[f"{case}_{kind}"]
            q=s["ball_quantiles"];ax.plot(RADII,q[1],color=color,label=label)
            ax.fill_between(RADII,q[0],q[2],color=color,alpha=.14)
        ax.set(xlabel="Radius / ensemble pair median",ylabel="Ball probability mass",title=f"{case.upper()}: root-law 10–90% band",ylim=(0,1))
        ax.legend(fontsize=8)
    fig.savefig(output/"ball_mass_laws.png",dpi=180);fig.savefig(output/"ball_mass_laws.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,case in zip(axes,["i3","a2","a3"]):
        for kind,color,label in [("graph","black","Graph"),("N48_primitive2_selected","tab:blue","Selected LFPP/GMC"),("N48_primitive2_flat","tab:orange","Flat control")]:
            x=np.sort(summaries[f"{case}_{kind}"]["triangle_small"])
            ax.plot(x,np.arange(1,len(x)+1)/max(1,len(x)),color=color,label=label)
        ax.set(xlabel="Smallest / largest triangle distance",ylabel="Sampled triangle CDF",title=case.upper(),xlim=(0,1));ax.legend(fontsize=8)
    fig.savefig(output/"triangle_shapes.png",dpi=180);fig.savefig(output/"triangle_shapes.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,case in zip(axes,["i3","a2","a3"]):
        for stencil,marker in [("26","o"),("primitive2","s")]:
            for kind,style in [("selected","-"),("flat","--")]:
                r=sorted([r for r in rows if r["case"]==case and r["stencil"]==stencil and r["kind"]==kind],key=lambda r:r["n"])
                ax.plot([v["n"] for v in r],[v["objective"] for v in r],marker+style,label=f"{kind}, {stencil}")
        ax.set(xlabel="Field grid N",ylabel="Joint descriptive discrepancy",title=case.upper());ax.legend(fontsize=7)
    fig.savefig(output/"finite_resolution.png",dpi=180);fig.savefig(output/"finite_resolution.pdf");plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("stage",choices=["select","final"])
    p.add_argument("--graphs",default="results/phase3a/graphs");p.add_argument("--discovery",default="results/phase3a/discovery")
    p.add_argument("--confirmation",default="results/phase3a/confirmation")
    p.add_argument("--selection",default="results/phase3a/selection/selection.json")
    p.add_argument("--output",required=True);p.add_argument("--bootstrap",type=int,default=160)
    args=p.parse_args()
    if args.stage=="select":selection(args)
    else:final_analysis(args)


if __name__=="__main__":main()
