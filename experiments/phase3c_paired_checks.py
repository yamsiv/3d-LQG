"""Paired holdout resolution/stencil and log-versus-smooth diagnostics.

These are prespecified sensitivities, not another parameter optimization.
Only common physical parents enter paired differences. Graph-target features
are held fixed in these conditional field-only intervals.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.phase3a_compare import save_json
from experiments.phase3b_search import field_groups
from experiments.phase3c_compare import _array_score
from src.family_search_3c import ensemble_features,WEIGHTINGS

R=Path('results/phase3c')


def paired(left,right,rng,graph_features,*,gauge='grounded',draws=32):
    assert [r['realization']for r in left]==[r['realization']for r in right]
    a=ensemble_features(left,gauge=gauge);b=ensemble_features(right,gauge=gauge)
    def score(g,f):return _array_score(np.array(g['vector']),np.array(g['valid']),np.array(f['vector']),np.array(f['valid']),a['scales'],a['families'],WEIGHTINGS['primary'])
    delta=[];means=[];scores={i:[]for i in graph_features}
    for _ in range(draws):
        ix=rng.integers(len(left),size=len(left));x=ensemble_features([left[j]for j in ix],gauge=gauge);y=ensemble_features([right[j]for j in ix],gauge=gauge)
        delta.append(np.array(y['vector'])-x['vector']);means.append(y['diagnostics']['normalization']/x['diagnostics']['normalization']-1)
        for i,g in graph_features.items():scores[i].append(score(g,y)-score(g,x))
    return dict(parents=len(left),physical_ids=[r['realization']for r in left],gauge=gauge,
        relative_mean_distance=b['diagnostics']['normalization']/a['diagnostics']['normalization']-1,
        relative_mean_mc95=np.quantile(means,[.025,.975]).tolist(),
        signed_feature_change=np.array(b['vector'])-a['vector'],feature_change_mc95=np.quantile(delta,[.025,.975],axis=0),names=a['names'],
        target_score_changes={i:dict(left=score(g,a),right=score(g,b),change=score(g,b)-score(g,a),
            conditional_field_mc95=np.quantile(scores[i],[.025,.975]).tolist())for i,g in graph_features.items()},
        interpretation='Pairedfield-onlyconditionaluncertainty;graphfeaturesfixed,nosystematicbound')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--stencil',required=True);p.add_argument('--output',default=str(R/'confirmation/paired_checks'))
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise FileExistsError(out/'COMPLETE.json')
    plan=json.loads((R/'confirmation/frozen_plan.json').read_text());feature=json.loads((R/'confirmation/analysis/point_features.json').read_text())
    logdir=R/'fields/confirmation/log';log32=field_groups([logdir],32);log48=field_groups([logdir],48);log64=field_groups([logdir],64)
    stencil=field_groups([a.stencil],48);smooth=field_groups([R/'fields/confirmation/smooth'],48);rng=np.random.default_rng(202611123985)
    rows=[]
    for gauge in ['grounded','unit_mass']:
        graph=feature[f'{gauge}:mean']['graphs']
        tasks=[]
        for xi in [x for x in plan['xis']if x]:tasks.append(('N32_to_N48',xi,log32[(xi,'central')],log48[(xi,'central')]))
        for xi in plan['n64_xis']:
            ids=plan['n64_field_ids'];common=[r for r in log48[(xi,'central')]if r['realization']in ids]
            tasks.append(('N48_to_N64_same8',xi,common,log64[(xi,'central')]))
            tasks.append(('stencil26_to98_same8',xi,stencil[(xi,'central')],common))
        for xi in plan['smooth_xis']:
            ids=plan['smooth_field_ids'];common=[r for r in log48[(xi,'central')]if r['realization']in ids]
            tasks.append(('smooth_to_log_same8',xi,smooth[(xi,'central')],common))
        for name,xi,left,right in tasks:
            destination=out/f'{gauge}_{name}_{xi:g}.json'
            if destination.exists():row=json.loads(destination.read_text())
            else:
                row=dict(comparison=name,xi=xi,**paired(left,right,rng,graph,gauge=gauge));save_json(destination,row)
            rows.append(row);print(json.dumps({'paired_check':name,'xi':xi,'gauge':gauge,'parents':len(left)}),flush=True)
    save_json(out/'summary.json',dict(rows=rows,scope='Allfrozenpairedcontrols;notselectionofbestgaugeorresolution'))
    save_json(out/'COMPLETE.json',dict(comparisons=len(rows),physical_fields=24,extended_fields=8,no_new_simulations=True))


if __name__=='__main__':main()
