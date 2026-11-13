"""Analyze checkpointed crossings and freeze a provisional continuous xi map.

Sampling uncertainty uses paired bootstrap resampling of complete physical
field families. Window and fixed correction ansatz sensitivity is saved
separately; its envelope is not a confidence interval for asymptotic bias.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from src.parameter_conversion_3b import fit_crossing_exponents, implicit_conversion


def load_crossings(directory, axis=0):
    directory=Path(directory)
    config=json.loads((directory/"config.json").read_text())
    sizes=np.array(config["sizes"])
    xis=np.array(config["xis"])
    values=np.empty((config["fields"],len(sizes),len(xis)))
    hashes={}
    for f in range(config["fields"]):
        for j,n in enumerate(sizes):
            for k,xi in enumerate(xis):
                path=directory/f"cross_f{f:03d}_n{n}_xi{xi:g}_a{axis}.json"
                record=json.loads(path.read_text())
                if record["field_seed"]!=config["seed"]+f:
                    raise ValueError("Inconsistent physical field seed")
                values[f,j,k]=record["site_crossing"]
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return config,sizes,xis,values,hashes


def clean_number(value):
    value=float(value)
    return value if np.isfinite(value) else None


def analyze(args):
    config,sizes,xis,values,hashes=load_crossings(args.input)
    output=Path(args.output)
    if output.exists():
        raise FileExistsError("Analysis is immutable; choose a fresh output directory")
    output.mkdir(parents=True)
    models=[("all_plain",list(range(len(sizes))),None),
            ("drop_smallest",list(range(1,len(sizes))),None),
            ("largest_three",list(range(len(sizes)-3,len(sizes))),None),
            ("all_Ninv_half",list(range(len(sizes))),.5),
            ("all_Ninv",list(range(len(sizes))),1.)]
    for j in range(len(sizes)-1):
        models.append((f"adjacent_{sizes[j]}_{sizes[j+1]}",[j,j+1],None))
    rng=np.random.default_rng(args.seed)
    draws=rng.integers(len(values),size=(args.bootstrap,len(values)))
    point=np.stack([fit_crossing_exponents(values,sizes,indices=indices,
                       correction_power=power) for _,indices,power in models])
    boot=np.empty((args.bootstrap,len(models),len(xis)))
    for b,draw in enumerate(draws):
        for j,(_,indices,power) in enumerate(models):
            boot[b,j]=fit_crossing_exponents(values[draw],sizes,indices=indices,
                                           correction_power=power)
    rows=[]
    for j,(name,indices,power) in enumerate(models):
        for k,xi in enumerate(xis):
            exponents=boot[:,j,k]
            q,gamma,dim=implicit_conversion(xi,point[j,k])
            qs,gs,ds=implicit_conversion(xi,exponents)
            valid=np.isfinite(gs)
            def bounds(array):
                valid_values=np.asarray(array)[np.isfinite(array)]
                return [None,None] if not len(valid_values) else list(map(float,
                    np.quantile(valid_values,[.025,.975])))
            lamlo,lamhi=bounds(exponents)
            qlo,qhi=bounds(qs)
            glo,ghi=bounds(gs)
            dlo,dhi=bounds(ds)
            rows.append({"model":name,"sizes":"/".join(map(str,sizes[indices])),
                "correction_power":power,"xi":float(xi),"fields":len(values),
                "lambda":float(point[j,k]),"lambda_mc95_low":lamlo,
                "lambda_mc95_high":lamhi,"Q":clean_number(q),
                "Q_mc95_low":qlo,"Q_mc95_high":qhi,
                "gamma":clean_number(gamma),"gamma_mc95_low":glo,
                "gamma_mc95_high":ghi,"dimension":clean_number(dim),
                "dimension_mc95_low":dlo,"dimension_mc95_high":dhi,
                "bootstrap_subcritical_fraction":float(valid.mean())})
    conversion=[]
    for k,xi in enumerate(xis):
        row=dict(rows[k])
        # Exclude highly noisy two-point slopes from the core envelope; retain
        # them, including bootstrap intervals, in the complete model table.
        sensitivity=[r for r in rows if r["xi"]==xi and not r["model"].startswith("adjacent")]
        for variable in ["lambda","Q","gamma","dimension"]:
            numbers=[r[variable] for r in sensitivity if r[variable] is not None]
            row[variable+"_systematic_low"]=min(numbers) if numbers else None
            row[variable+"_systematic_high"]=max(numbers) if numbers else None
        row["all_sensitivity_models_subcritical"]=float(all(r["gamma"] is not None for r in sensitivity))
        row["status"]="exact_flat_endpoint" if xi==0 else "provisional_finite_resolution"
        if row["gamma"] is None:
            row["status"]="invalid_subcritical_conversion"
        elif row["bootstrap_subcritical_fraction"]<.975 or not row["all_sensitivity_models_subcritical"]:
            row["status"]="provisional_branch_or_model_sensitive"
        conversion.append(row)
    metadata={"source_config":config,"bootstrap":args.bootstrap,
        "bootstrap_seed":args.seed,"resampling_unit":"complete independent field across sizes and xi",
        "primary_model":"all_plain; log median physical site crossing vs log N",
        "systematic_envelope":"point-estimate envelope of full/drop-smallest/largest-three/plain and fixed N^-1/2,N^-1 correction ansatzes; not a bias bound",
        "interpolation":"piecewise linear crossing slopes and uncertainty endpoints; no extrapolation",
        "regularization":"finite torus spectral field restricted to open cube; not whole-space box-average field",
        "asymptotic_map_validated":False,"checkpoint_sha256":hashes}
    with (output/"calibration.json").open("x") as stream:
        json.dump({"metadata":metadata,"conversion_rows":conversion,"model_rows":rows},stream,indent=2,allow_nan=False)
    for name,data in [("conversion.csv",conversion),("models.csv",rows)]:
        with (output/name).open("x") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    with (output/"bootstrap.npz").open("xb") as stream:
        np.savez_compressed(stream,field_crossings=values,sizes=sizes,xis=xis,
                            models=np.array([m[0] for m in models]),point_lambda=point,
                            bootstrap_lambda=boot,field_resample_indices=draws)
    for source in ["src/parameter_conversion_3b.py","experiments/phase3b_calibrate.py"]:
        shutil.copyfile(source,output/Path(source).name)
    plot(output,sizes,xis,values,conversion,rows)
    print(json.dumps({"fields":len(values),"sizes":sizes.tolist(),"rows":[
        {key:r[key] for key in ["xi","lambda","gamma","gamma_mc95_low","gamma_mc95_high",
                               "gamma_systematic_low","gamma_systematic_high","status"]}
        for r in conversion]},indent=2))


def plot(output,sizes,xis,values,conversion,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(11.5,8.5),constrained_layout=True)
    selected=[0,.1,.2,.25,.3,.35,.375]
    medians=np.median(values,axis=0)
    for k,xi in enumerate(xis):
        if xi in selected:
            axes[0,0].plot(sizes,medians[:,k],"o-",label=f"xi={xi:g}")
    axes[0,0].set(xscale="log",yscale="log",xlabel="Resolution N",ylabel="Median physical site crossing")
    axes[0,0].legend(fontsize=8)
    for model,style in [("all_plain","o-"),("drop_smallest","s--"),("largest_three","^--"),
                        ("all_Ninv_half",":"),("all_Ninv","-.")]:
        chosen=[r for r in rows if r["model"]==model]
        axes[0,1].plot(xis,[r["lambda"] for r in chosen],style,label=model)
    axes[0,1].fill_between(xis,[r["lambda_mc95_low"] for r in conversion],
                              [r["lambda_mc95_high"] for r in conversion],alpha=.18)
    axes[0,1].set(xlabel="LFPP xi",ylabel="Finite crossing exponent proxy lambda")
    axes[0,1].legend(fontsize=7)
    for ax,var,label in [(axes[1,0],"gamma","Implicit GMC gamma"),(axes[1,1],"dimension","Implicit dimension gamma/xi")]:
        ax.plot(xis,[np.nan if r[var] is None else r[var] for r in conversion],"o-",label="Full window")
        ax.fill_between(xis,[np.nan if r[var+"_mc95_low"] is None else r[var+"_mc95_low"] for r in conversion],
                             [np.nan if r[var+"_mc95_high"] is None else r[var+"_mc95_high"] for r in conversion],alpha=.25,label="Paired field MC95")
        ax.fill_between(xis,[np.nan if r[var+"_systematic_low"] is None else r[var+"_systematic_low"] for r in conversion],
                             [np.nan if r[var+"_systematic_high"] is None else r[var+"_systematic_high"] for r in conversion],alpha=.13,color="tab:orange",label="Model/window envelope")
        ax.set(xlabel="LFPP xi",ylabel=label)
        ax.legend(fontsize=8)
        ax.set_xlim(0,max(xis)+.005)
        invalid=[r["xi"] for r in conversion if r[var] is None]
        if invalid:
            ax.axvspan(min(invalid),max(xis)+.005,color=".8",alpha=.3)
            ax.text(.97,.05,"Gray: no real subcritical conversion",ha="right",
                    transform=ax.transAxes,fontsize=7)
        ambiguous=[r for r in conversion if r["xi"]>0 and r[var] is not None
                   and r["bootstrap_subcritical_fraction"]<.975]
        for row in ambiguous:
            ax.annotate(f"{100*row['bootstrap_subcritical_fraction']:.1f}% valid draws;\nMC band conditional",
                        xy=(row["xi"],row[var]),xytext=(-100,10),textcoords="offset points",
                        arrowprops={"arrowstyle":"->","lw":.6},fontsize=7)
    figure.suptitle("Phase 3B: finite-resolution conversion; bands do not certify asymptotic bias")
    figure.savefig(output/"calibration.png",dpi=180)
    figure.savefig(output/"calibration.pdf")
    plt.close(figure)


def analyze_controls(args):
    """Compare paired regularizer, infrared and orientation diagnostics.

    Four physical fields are a sensitivity probe, not a replacement estimate
    of the twenty-field calibration. Retain the full paired raw records.
    """
    config,sizes,xis,main,_=load_crossings(args.input)
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows=[]
    raw={}
    rng=np.random.default_rng(args.seed)
    for directory in args.controls:
        c=json.loads((Path(directory)/"config.json").read_text())
        if c["sizes"]!=config["sizes"] or c["seed"]!=config["seed"]:
            raise ValueError("Controls must use the identical seed and complete size tuple")
        for axis in c["axes"]:
            _,_,cx,control,_=load_crossings(directory,axis=axis)
            selected=[int(np.flatnonzero(np.isclose(xis,x,rtol=0,atol=1e-12))[0]) for x in cx]
            base=main[:len(control),:,selected]
            label=Path(directory).name+f"_axis{axis}"
            raw[label]=control
            draws=rng.integers(len(control),size=(args.bootstrap,len(control)))
            delta_lambda=fit_crossing_exponents(control,sizes)-fit_crossing_exponents(base,sizes)
            boot=np.stack([fit_crossing_exponents(control[d],sizes)-fit_crossing_exponents(base[d],sizes) for d in draws])
            relative=control/base-1
            for k,x in enumerate(cx):
                lo,hi=np.quantile(boot[:,k],[.025,.975])
                rows.append({"control":label,"xi":float(x),"fields":len(control),
                    "delta_lambda":float(delta_lambda[k]),"paired_mc95_low":float(lo),
                    "paired_mc95_high":float(hi),
                    "mean_relative_crossing_change":float(relative[:,:,k].mean()),
                    "N128_mean_relative_crossing_change":float(relative[:,-1,k].mean()),
                    "max_absolute_relative_crossing_change":float(abs(relative[:,:,k]).max())})
    # Deterministic endpoint convention: same measured crossings mapped to the
    # closed unit cube with grid step1/(N-1), rather than the truncated N-node
    # cube with step1/N. Record it without generating another field or path.
    closed=main*(sizes/(sizes-1))[None,:,None]
    delta=fit_crossing_exponents(closed,sizes-1)-fit_crossing_exponents(main,sizes)
    for k,x in enumerate(xis):
        rows.append({"control":"closed_unit_cube_coordinate_rescaling","xi":float(x),
                     "fields":len(main),"delta_lambda":float(delta[k]),
                     "paired_mc95_low":None,"paired_mc95_high":None,
                     "mean_relative_crossing_change":float(np.mean(1/(sizes-1))),
                     "N128_mean_relative_crossing_change":float(1/(sizes[-1]-1)),
                     "max_absolute_relative_crossing_change":float(1/(sizes[0]-1))})
    payload={"metadata":{"main_config":config,"paired_fields_per_random_control":4,
               "bootstrap":args.bootstrap,"bootstrap_seed":args.seed,
               "interpretation":"paired finite-resolution sensitivity; no systematic error bound",
               "closed_cube_control":"coordinate/epsilon rescaling of identical field samples; not independent field or extrapolation"},
             "rows":rows}
    with (output/"controls.json").open("x") as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
    with (output/"controls.csv").open("x") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"raw_controls.npz").open("xb") as stream:
        np.savez_compressed(stream,main=main,sizes=sizes,xis=xis,**raw)
    shutil.copyfile(__file__,output/"phase3b_calibrate.py")
    print(json.dumps(payload,indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=202611113599)
    parser.add_argument("--controls",nargs="*")
    args=parser.parse_args()
    if args.controls:
        analyze_controls(args)
    else:
        analyze(args)


if __name__=="__main__":
    main()
