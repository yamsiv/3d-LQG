"""Append adaptive graph observations without repeating the field analysis.

The complete initial score/feature/field-bootstrap checkpoints are immutable
inputs. Only new physical graph feature draws are calculated. Shared field
bootstrap indices are inherited exactly, preserving dependence over C and xi.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.phase3a_compare import save_json
from experiments.phase3a_ensembles import write_npz
from experiments.phase3b_search import graph_groups
from experiments.phase3c_compare import graph_quality,_array_score
from src.family_search_3c import ensemble_features,compare_features,WEIGHTINGS
from src.provenance import environment_manifest


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--base',default='results/phase3c/screen/analysis')
    p.add_argument('--graphs',nargs='+',required=True);p.add_argument('--design',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=202611123501)
    a=p.parse_args();base=Path(a.base);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'scores.json').exists():raise FileExistsError(out/'scores.json')
    start=time.perf_counter();prior=json.loads((base/'scores.json').read_text())
    features=json.loads((base/'features.json').read_text());rows=copy.deepcopy(prior['rows'])
    n=prior['configuration']['graph_n'];graphs=graph_groups(a.graphs,n)
    if not graphs:raise ValueError('No new graph observations')
    old_ids=set(features['grounded:mean']['graphs'])
    if old_ids&set(graphs):raise ValueError('Adaptive graph IDs overlap the base; do not pool unidentified cohorts')
    specs=json.loads(Path(a.design).read_text())['configurations'];spec_by={s['id']:s for s in specs}
    tolerance=json.loads(Path('results/phase3c/families/statistics/tolerance_calibration.json').read_text())
    new_quality={i:graph_quality(r) for i,r in graphs.items()}
    for i,q in new_quality.items():
        if len(graphs[i])<12:q.update(sampler_allowed=False,status='incomplete_parent_count')
    # A safe exact reference supplies parameter/branch metadata only. All
    # graph-specific point features, signs, scores and eligibility are rebuilt.
    prototypes={(r['gauge'],r['normalization'],r['weighting'],r['xi']):r
                for r in prior['rows'] if r['id']=='C002'}
    for normalization in ['mean','median']:
        gf={i:ensemble_features(r,normalization=normalization) for i,r in graphs.items()}
        for gauge in ['grounded','unit_mass']:
            key=gauge+':'+normalization
            ff={ast.literal_eval(k):v for k,v in features[key]['fields'].items()}
            features[key]['graphs'].update(gf)
            for identifier,g in gf.items():
                for xi in sorted({k[0] for k in ff}):
                    fs={band:f for (x,band),f in ff.items() if x==xi}
                    for weighting in WEIGHTINGS:
                        proto=prototypes[(gauge,normalization,weighting,xi)]
                        comparisons={band:compare_features(g,f,tolerance,gauge=gauge,normalization=normalization,
                            weighting=weighting,sampler_allowed=new_quality[identifier]['sampler_allowed'],
                            calibration_allowed=bool(proto['eligible'])) for band,f in fs.items()}
                        central=comparisons['central'];robust=max(c['score'] for c in comparisons.values())
                        row={k:v for k,v in proto.items() if k not in ['robust_mc95','signed_residuals','valid',
                            'family_components','feature_flags','family_flags','probe_family_components','mc_scope']}
                        eligible=bool(proto['eligible'] and new_quality[identifier]['sampler_allowed'])
                        row.update(id=identifier,triple=spec_by[identifier]['triple'],score=central['score'],
                            robust_score=robust,eligible=eligible,candidate=bool(eligible and robust<=central['epsilon']),
                            componentwise_low=all(c['componentwise_low'] for c in comparisons.values()),
                            sampler_status=new_quality[identifier]['status'],graph_parents=len(graphs[identifier]),
                            probe_scores={band:c['score'] for band,c in comparisons.items()},
                            dimension_flag='not_resolved_by_small_graph_growth',adaptive_observation=True)
                        if weighting=='primary':
                            row.update(signed_residuals=central['signed_residuals'],valid=central['valid'],
                                family_components=central['family_components'],feature_flags=central['feature_flags'],
                                family_flags=central['family_flags'],probe_family_components={b:c['family_components'] for b,c in comparisons.items()})
                        rows.append(row)
    with np.load(base/'graph_bootstrap.npz') as z:
        bg={k:z[k] for k in z.files}
    with np.load(base/'field_bootstrap.npz') as z:
        bf={k:z[k] for k in z.files}
    graph_ids=sorted(graphs);field_keys=[ast.literal_eval(k) for k in bf['keys']]
    draws=len(bg['vectors']);rng=np.random.default_rng(a.seed);gv=[];gm=[]
    for draw in range(draws):
        v=[];valid=[]
        for identifier in graph_ids:
            r=graphs[identifier];indices=rng.integers(len(r),size=len(r));f=ensemble_features([r[i] for i in indices])
            v.append(f['vector']);valid.append(f['valid'])
        gv.append(v);gm.append(valid)
    gv=np.asarray(gv);gm=np.asarray(gm)
    for row in rows:
        if row['id'] not in graphs or row['gauge']!='grounded' or row['normalization']!='mean':continue
        gi=graph_ids.index(row['id']);choices=[j for j,k in enumerate(field_keys) if k[0]==row['xi']]
        values=[max(_array_score(gv[d,gi],gm[d,gi],bf['vectors'][d,j],bf['valid'][d,j],
                    bg['scales'],list(bg['families']),WEIGHTINGS[row['weighting']]) for j in choices) for d in range(draws)]
        row['robust_mc95']=np.nanquantile(values,[.025,.975]).tolist()
        row['mc_scope']='New independent whole-graph resamples; exact cached field-feature draws shared with base, no systematic bound'
    all_ids=list(bg['ids'])+graph_ids
    if len(set(all_ids))!=len(all_ids):raise ValueError('Duplicate graph bootstrap identities')
    point=features['grounded:mean']['graphs']
    write_npz(out/'graph_bootstrap.npz',ids=np.array(all_ids),vectors=np.concatenate([bg['vectors'],gv],axis=1),
        valid=np.concatenate([bg['valid'],gm],axis=1),point_vectors=np.array([point[i]['vector'] for i in all_ids]),
        point_valid=np.array([point[i]['valid'] for i in all_ids]),scales=bg['scales'],names=bg['names'],families=bg['families'])
    shutil.copyfile(base/'field_bootstrap.npz',out/'field_bootstrap.npz')
    if digest(base/'field_bootstrap.npz')!=digest(out/'field_bootstrap.npz'):raise ValueError('Field-bootstrap copy changed')
    quality=json.loads((base/'reliability.json').read_text());quality.update(new_quality)
    growth=json.loads((base/'growth.json').read_text());small=graph_groups(a.graphs,512)
    for identifier,r in graphs.items():
        if identifier not in small or len(small[identifier])!=len(r):continue
        lo={x['header']['seed']:np.mean(x['root_means']) for x in small[identifier]}
        hi={x['header']['seed']:np.mean(x['root_means']) for x in r}
        if lo.keys()!=hi.keys():raise ValueError('Unpaired adaptive sizes')
        keys=sorted(lo);u=np.array([lo[k] for k in keys]);v=np.array([hi[k] for k in keys]);beta=np.log(v.mean()/u.mean())/np.log(4)
        samples=[]
        for _ in range(64):
            ix=rng.integers(len(u),size=len(u));samples.append(np.log(v[ix].mean()/u[ix].mean())/np.log(4))
        growth.append(dict(id=identifier,beta_finite=beta,beta_mc95=np.quantile(samples,[.025,.975]).tolist(),
            inverse_growth_proxy=1/beta,warning='two-size finite proxy, not Hausdorff dimension',physical_parents=len(u)))
    save_json(out/'features.json',features);save_json(out/'reliability.json',quality);save_json(out/'growth.json',growth)
    save_json(out/'scores.json',dict(rows=rows,configuration=vars(a),provenance=environment_manifest(),
        initial_base=str(base),base_score_sha256=digest(base/'scores.json'),design=a.design,
        graph_configs=len(all_ids),physical_field_count=prior['physical_field_count'],bootstrap_draws=draws,
        raw_field_or_old_graph_recomputation=False,elapsed_seconds=time.perf_counter()-start,
        interpretation='Adaptive observations appended to immutable original response surface; no extra independent fields or old graphs are counted'))
    print(json.dumps({'output':str(out),'new_graph_configs':len(graph_ids),'total_graph_configs':len(all_ids),
                      'field_feature_recomputation':False,'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':main()
