"""Adaptive covariance/linked-field search from immutable saved realizations."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from src.covariance_design_3b import configuration,canonical_triple
from src.metric_features_3b import feature_vector,grouped_feature_distance,DEFAULT_WEIGHTS
from src.adaptive_search_3b import ScoreSurrogate
from src.parameter_conversion_3b import load_calibration
from src.provenance import environment_manifest
from experiments.phase3a_compare import save_json,jsonable


WEIGHTS={
    'primary':dict(DEFAULT_WEIGHTS),
    'equal_families':{k:1. for k in DEFAULT_WEIGHTS},
    'higher_point':dict(DEFAULT_WEIGHTS,pair=.5,balls=1.,triangles=1.5,quartets=2.,matrix_dependence=1.5),
    'without_mesoscopic':dict(DEFAULT_WEIGHTS,mesoscopic=0.,atoms=0.),
}


def graph_groups(directories,n):
    groups={};seen=set()
    for directory in directories:
        for path in sorted(Path(directory).glob(f'C*/graph_*_n{n}.npz')):
            with np.load(path) as a:
                h=json.loads(str(a['header']));key=(h['id'],h['seed'],n)
                if key in seen:raise ValueError(f'Duplicate physical graph {key}')
                seen.add(key)
                row={k:a[k] for k in ['grid','cdfs','root_means','matrix']}
                row.update(header=h,discrete=True,realization=h['realization'],source=str(path))
                groups.setdefault(h['id'],[]).append(row)
    return groups


def field_groups(directories,n):
    groups={};seen=set()
    for directory in directories:
        for path in sorted(Path(directory).glob(f'xi*/metric_*_n{n}.npz')):
            with np.load(path) as a:
                h=json.loads(str(a['header']));arrays={k:a[k]for k in ['grid','cdfs','root_means','matrices','bin_width']}
                for i,probe in enumerate(h['probes']):
                    key=(h['xi'],probe['band']);physical=(h['realization'],key,n)
                    if physical in seen:raise ValueError(f'Duplicate physical field {physical}')
                    seen.add(physical)
                    row={'grid':arrays['grid'][i],'cdfs':arrays['cdfs'][i],'root_means':arrays['root_means'][i],
                        'matrix':arrays['matrices'][i],'bin_width':float(arrays['bin_width'][i]),
                        'xi':h['xi'],'gamma':probe['gamma'],'log_total_mass':probe['log_total_mass'],
                        'header':h,'discrete':False,'realization':h['realization'],'source':str(path)}
                    groups.setdefault(key,[]).append(row)
    for records in groups.values():records.sort(key=lambda r:r['realization'])
    return groups


def scores_for(graphs,fields,gauge):
    gf={k:feature_vector(v,gauge=gauge)for k,v in graphs.items()}
    ff={k:feature_vector(v,gauge=gauge)for k,v in fields.items()}
    return gf,ff


def calibration_eligible(records):
    h=records[0]['header'];c=h['conversion']
    return h['xi']>0 and c.get('bootstrap_subcritical_fraction',1)>=.975


def score_table(graphs,fields,gf,ff):
    rows=[]
    for identifier,feature in gf.items():
        spec=graphs[identifier][0]['header']['spec']
        for (xi,band),f in ff.items():
            score=grouped_feature_distance(feature,f)
            alternatives={name:grouped_feature_distance(feature,f,weights=weights)['total']for name,weights in WEIGHTS.items()}
            rows.append({'id':identifier,'xi':xi,'gamma':fields[(xi,band)][0]['gamma'],'band':band,
                'score':score['total'],'components':score['components'],'alternative_weights':alternatives,
                'eligible_primary':calibration_eligible(fields[(xi,band)]) and band=='central',
                'graph_realizations':len(graphs[identifier]),'field_realizations':len(fields[(xi,band)]),
                'graph_spec':spec,'valid_features':score['valid_features']})
    for row in rows:
        probes=[r['score']for r in rows if r['id']==row['id'] and r['xi']==row['xi']]
        row['calibration_score_envelope']=[min(probes),max(probes)]
        row['robust_score']=max(probes)
    return rows


def proposal_design(rows,specs,calibration):
    eligible=[r for r in rows if r['eligible_primary']]
    best={key:min((r for r in eligible if r['id']==key),key=lambda r:r['robust_score'])for key in {r['id']for r in eligible}}
    ordered=sorted(best.values(),key=lambda r:r['robust_score'])
    chosen=[];families=set()
    for row in ordered:
        if len(chosen)<3 or (row['graph_spec']['family'] not in families and len(chosen)<5):
            chosen.append(row['id']);families.add(row['graph_spec']['family'])
        if len(chosen)==5:break
    for row in ordered:
        if len(chosen)>=5:break
        if row['id']not in chosen:chosen.append(row['id'])
    # Include the validated I3 baseline as an intermediate control rather
    # than assume it is the best correspondence.
    i3=next((s['id']for s in specs if max(abs(np.array(s['triple'])))<1e-12),None)
    if i3 and i3 not in chosen:chosen[-1]=i3
    inputs=np.array([r['graph_spec']['triple']+[r['xi']]for r in eligible])
    model=ScoreSurrogate(inputs,[r['robust_score']for r in eligible],noise_floor=.04)
    rng=np.random.default_rng(202611113744)
    candidates=[];existing=np.array([s['triple']for s in specs])
    anchors=np.array([r['graph_spec']['triple']for r in ordered[:5]])
    for x in qmc.scale(qmc.LatinHypercube(3,seed=202611113743).random(120),[-.55]*3,[.65]*3):
        candidates.append(x)
    for _ in range(200):candidates.append(anchors[rng.integers(len(anchors))]+rng.normal(0,.09,3))
    valid=[]
    for x in candidates:
        triple=canonical_triple(x)
        try:s=configuration(triple,family='adaptive_GP')
        except ValueError:continue
        if s['condition']>12 or s['eigenvalues'][0]<.15:continue
        if np.linalg.norm(existing-triple,axis=1).min()<.04:continue
        valid.append(s)
    xis=np.linspace(.15,.3,13)
    queries=np.array([s['triple']+[float(x)]for s in valid for x in xis])
    means,sd=model.predict(queries);means=means.reshape(-1,len(xis));sd=sd.reshape(means.shape)
    acquisition=np.min(means-sd,axis=1)
    new=[]
    for k in range(3):
        i=int(np.argmin(acquisition))
        s=valid[i];s['id']=f'C{len(specs)+k:03d}'
        j=int(np.argmin(means[i]-sd[i]))
        s['acquisition']={'xi':float(xis[j]),'predicted_score':float(means[i,j]),'surrogate_sd':float(sd[i,j]),
            'qualification':'heuristic acquisition SD, not inferential confidence or sampler control'}
        new.append(s)
        distance=np.linalg.norm(np.array([s0['triple']for s0 in valid])-s['triple'],axis=1)
        acquisition[distance<.10]=np.inf
    adaptive_ids=chosen+[s['id']for s in new]
    anchors_xi=sorted({best[i]['xi']for i in chosen if i in best})
    extra=[]
    existing_xi=sorted({r['xi']for r in eligible})
    for x in anchors_xi:
        pos=existing_xi.index(x)
        neighbors=existing_xi[max(0,pos-1):min(len(existing_xi),pos+2)]
        for other in neighbors:
            mid=float((x+other)/2)
            if other!=x and mid not in existing_xi and calibration.evaluate(mid)['valid']:extra.append(mid)
    extra=list(dict.fromkeys(extra))[:2]
    return {'configurations':specs+new,'intermediate_ids':adaptive_ids,'additional_xi':extra,
        'ranking':ordered,'surrogate_length_scales':model.length_scales.tolist(),
        'acquisition_policy':'five existing diverse candidates including I3 baseline, three constrained LCB proposals; primary grounded gauge, max gamma-model probe score',
        'not_unique_optimum':True}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--stage',choices=['screen','intermediate','final'],required=True)
    p.add_argument('--graphs',nargs='+',required=True);p.add_argument('--fields',nargs='+',required=True)
    p.add_argument('--graph-n',type=int,required=True);p.add_argument('--field-n',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--design',default='results/phase3b/covariance_design.json')
    p.add_argument('--calibration',default='results/phase3b/calibration/analysis/calibration.json')
    p.add_argument('--bootstrap',type=int,default=32);p.add_argument('--seed',type=int,default=202611113745)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'scores.json').exists():raise FileExistsError(root/'scores.json')
    graphs=graph_groups(a.graphs,a.graph_n);fields=field_groups(a.fields,a.field_n)
    if any(len(r)<4 for r in graphs.values()) or any(len(r)<4 for r in fields.values()):raise ValueError('Incomplete or undersampled ensemble')
    data=json.loads(Path(a.design).read_text());specs=data['configurations']if isinstance(data,dict)else data
    grounded,ff=scores_for(graphs,fields,'grounded');unit,fu=scores_for(graphs,fields,'unit_mass')
    rows=score_table(graphs,fields,grounded,ff);unit_rows=score_table(graphs,fields,unit,fu)
    save_json(root/'scores.json',{'provenance':environment_manifest(),'configuration':vars(a),
        'primary_gauge':'grounded; unit-mass is separately reported, never fit-selected',
        'rows':rows,'unit_mass_rows':unit_rows,
        'graph_features':grounded,'field_features':{str(k):v for k,v in ff.items()}})
    with (root/'scores.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','xi','gamma','band','score','robust_score','eligible_primary','graph_realizations','field_realizations'])
        writer.writeheader();writer.writerows({k:r[k]for k in writer.fieldnames}for r in rows)
    if a.stage=='screen':
        design=proposal_design(rows,specs,load_calibration(a.calibration))
        save_json(root/'adaptive_design.json',design)
        print(json.dumps({'intermediate_ids':design['intermediate_ids'],'additional_xi':design['additional_xi']}),flush=True)
    # Conditional optimization uncertainty: whole graph/field resamples;
    # all xi share the same field resample and no matrix subsets add replicas.
    rng=np.random.default_rng(a.seed);choices=[];minima=[]
    eligible_xis={k[0]for k,r in fields.items()if k[1]=='central' and calibration_eligible(r)}
    eligible_keys=[k for k in fields if k[0]in eligible_xis]
    physical_ids=[r['realization']for r in fields[eligible_keys[0]]]
    if any([r['realization']for r in fields[k]]!=physical_ids for k in eligible_keys):raise ValueError('Xi fields must share physical identities')
    for draw in range(a.bootstrap):
        fi=rng.integers(len(physical_ids),size=len(physical_ids))
        boot_fields={k:feature_vector([v[i]for i in fi])for k,v in fields.items()if k in eligible_keys}
        ranked=[]
        for key,records in graphs.items():
            gi=rng.integers(len(records),size=len(records))
            feature=feature_vector([records[i]for i in gi])
            scored=[(max(grouped_feature_distance(feature,f)['total']for (x,_),f in boot_fields.items()if x==xi),xi)for xi in sorted(eligible_xis)]
            score,xi=min(scored);ranked.append((score,key,xi))
        score,key,xi=min(ranked);choices.append([key,xi]);minima.append(score)
        if draw%8==7:
            progress=root/f'bootstrap_through_{draw+1:03d}.json'
            save_json(progress,{'choices':choices,'minima':minima})
    save_json(root/'selection_uncertainty.json',{'choices':choices,'minimum_scores':minima,'draws':a.bootstrap,
        'criterion':'max over available gamma-model probes, same as robust candidate ranking',
        'qualification':'Conditional finite-ensemble argmin frequency, not posterior or continuum identification; omitted/invalid models and sampler errors separate'})
    if a.stage=='intermediate':
        best={i:min((r for r in rows if r['id']==i and r['eligible_primary']),key=lambda r:r['robust_score'])for i in graphs}
        ordered=sorted(best.values(),key=lambda r:r['robust_score'])
        finalists=ordered[:3]
        save_json(root/'finalists.json',{'candidates':finalists,'frozen_before_heldout_seeds':True,
            'selection':'primary grounded robust gamma-probe score; weight/gauge sensitivities reported separately'})
        print(json.dumps({'finalists':[(r['id'],r['xi'])for r in finalists]}),flush=True)
    print(json.dumps({'stage':a.stage,'graphs':len(graphs),'field_pairs':len(fields),'output':str(root)}),flush=True)


if __name__=='__main__':main()
