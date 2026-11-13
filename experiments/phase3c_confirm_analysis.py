"""Frozen-hypothesis independent confirmation. No new physical simulation.

Shared physical-field resampling across xi/probes/gauges; primary robust J
uses ONLY central/low/high. Named calibration curves are kept as global
diagnostics, never optimized separately for individual covariance matrices.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from experiments.phase3a_compare import save_json
from experiments.phase3a_ensembles import write_npz
from experiments.phase3b_search import graph_groups,field_groups
from experiments.phase3c_compare import graph_quality,_array_score
from src.family_search_3c import ensemble_features,compare_features,WEIGHTINGS
from src.confirmation_statistics_3c import matrix_diagnostic

ROOT=Path('results/phase3c')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',default=str(ROOT/'confirmation/analysis'))
    p.add_argument('--draws',type=int,default=32);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise FileExistsError(out/'COMPLETE.json')
    start=time.perf_counter();plan=json.loads((ROOT/'confirmation/frozen_plan.json').read_text())
    gs=graph_groups(sorted((ROOT/'confirmation/graphs').glob('batch_*')),8192)
    allfields=field_groups([ROOT/'fields/confirmation/log'],48)
    fields={k:v for k,v in allfields.items()if k[1]in plan['primary_score_probe_labels']}
    if set(gs)!=set(plan['graph_ids'])or any(len(v)!=16 for v in gs.values()):raise ValueError('Incomplete graph confirmation')
    if len(fields)!=15 or any(len(v)!=24 for v in fields.values()):raise ValueError('Incomplete field confirmation')
    identities=[r['realization']for r in next(iter(fields.values()))]
    assert identities==plan['physical_holdout_ids']
    assert all([r['realization']for r in rs]==identities for rs in fields.values())
    tol=json.loads((ROOT/'families/statistics/tolerance_calibration.json').read_text());q={i:graph_quality(rs)for i,rs in gs.items()}
    inputs=sorted({r['source']for rs in list(gs.values())+list(allfields.values())for r in rs})
    fingerprint={'plan_sha256':hashlib.sha256((ROOT/'confirmation/frozen_plan.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{s:hashlib.sha256(Path(s).read_bytes()).hexdigest()for s in inputs},'draws':a.draws}
    fpath=out/'configuration.json'
    if fpath.exists():assert json.loads(fpath.read_text())==fingerprint
    else:save_json(fpath,fingerprint)
    if (out/'point_features.json').exists():features=json.loads((out/'point_features.json').read_text());rows=json.loads((out/'point_rows.json').read_text())
    else:
        features={};rows=[]
        for norm in ['mean','median']:
            gf={i:ensemble_features(rs,normalization=norm)for i,rs in gs.items()}
            for gauge in ['grounded','unit_mass']:
                ff={str(k):ensemble_features(rs,gauge=gauge,normalization=norm)for k,rs in allfields.items()}
                features[f'{gauge}:{norm}']={'graphs':gf,'fields':ff}
                for i,g in gf.items():
                    nominal=plan['profiles_frozen_from_discovery'][i]['nominal_xi']
                    for xi in [x for x in plan['xis']if x]:
                        for weighting in WEIGHTINGS:
                            cs={band:compare_features(g,ff[str((xi,band))],tol,gauge=gauge,normalization=norm,weighting=weighting,
                                sampler_allowed=q[i]['sampler_allowed'])for band in plan['primary_score_probe_labels']}
                            c=cs['central'];rows.append(dict(id=i,xi=xi,nominal=xi==nominal,gauge=gauge,normalization=norm,
                                weighting=weighting,central_score=c['score'],robust_score=max(c['score']for c in cs.values()),
                                epsilon=c['epsilon'],componentwise=all(c['componentwise_low']for c in cs.values()),
                                sampler_allowed=q[i]['sampler_allowed'],probe_scores={b:c['score']for b,c in cs.items()},
                                signed_residuals=c['signed_residuals']if weighting=='primary'else None,
                                family_components=c['family_components'],probe_components={b:c['family_components']for b,c in cs.items()}))
        save_json(out/'point_features.json',features);save_json(out/'point_rows.json',rows);save_json(out/'reliability.json',q)
    gids=sorted(gs);fkeys=list(fields);rng=np.random.default_rng(202611123980)
    # Draw indices are replayable independently of checkpoint/loop length.
    gindices={i:rng.integers(len(rs),size=(a.draws,len(rs)))for i,rs in gs.items()}
    findices=rng.integers(24,size=(a.draws,24));layout=features['grounded:mean']['graphs'][gids[0]]
    for gauge in ['grounded','unit_mass']:
        cache=out/f'bootstrap_{gauge}.npz'
        if cache.exists():
            with np.load(cache)as z:gb=z['g'];gv=z['gv'];fb=z['f'];fv=z['fv']
        else:
            gb=[];gv=[];fb=[];fv=[]
            for d in range(a.draws):
                gr=[ensemble_features([gs[i][j]for j in gindices[i][d]])for i in gids]
                fr=[ensemble_features([fields[k][j]for j in findices[d]],gauge=gauge)for k in fkeys]
                gb.append([r['vector']for r in gr]);gv.append([r['valid']for r in gr]);fb.append([r['vector']for r in fr]);fv.append([r['valid']for r in fr])
                if d%8==7:print(json.dumps({'gauge':gauge,'feature_draws':d+1}),flush=True)
            gb=np.array(gb);gv=np.array(gv);fb=np.array(fb);fv=np.array(fv)
            write_npz(cache,g=gb,gv=gv,f=fb,fv=fv,gids=np.array(gids),fkeys=np.array([str(k)for k in fkeys]),field_indices=findices,
                **{f'graph_indices_{i}':ix for i,ix in gindices.items()})
        for r in rows:
            if r['gauge']!=gauge or r['normalization']!='mean':continue
            gi=gids.index(r['id']);ks=[j for j,k in enumerate(fkeys)if k[0]==r['xi']];scores=[]
            for d in range(a.draws):scores.append(max(_array_score(gb[d,gi],gv[d,gi],fb[d,j],fv[d,j],layout['scales'],layout['families'],WEIGHTINGS[r['weighting']])for j in ks))
            r['robust_mc95']=np.quantile(scores,[.025,.975]).tolist()
    save_json(out/'scores.json',dict(rows=rows,draws=a.draws,scope='Conditional whole-parent MC only, not equivalence CI. Frozen scalar tolerance and component gates unchanged.'))
    globalrows=[]
    for i in gids:
        for gauge in ['grounded','unit_mass']:
            f=features[f'{gauge}:mean'];g=f['graphs'][i]
            for k,records in allfields.items():
                if not k[1].startswith('model_'):continue
                c=compare_features(g,f['fields'][str(k)],tol,gauge=gauge,sampler_allowed=q[i]['sampler_allowed'])
                globalrows.append(dict(id=i,xi=k[0],global_model=k[1],gamma=records[0]['gamma'],gauge=gauge,
                    score=c['score'],componentwise=c['componentwise_low'],components=c['family_components']))
    save_json(out/'global_calibration_models.json',dict(rows=globalrows,scope='Each named model is ONE coherent curve across all xi/C; no per-C gamma selection'))
    flat=field_groups([ROOT/'fields/confirmation/flat'],48);smooth=field_groups([ROOT/'fields/confirmation/smooth'],48)
    if len(next(iter(flat.values()),[]))!=24:raise ValueError('Flat control incomplete')
    diagnostics=[];matrix=[];control_differences=[]
    # Same underlying field IDs/root streams for flat and log. A single shared
    # graph draw and field draw enters both sides of each difference.
    for gauge in ['grounded','unit_mass']:
        with np.load(out/f'bootstrap_{gauge}.npz')as z:
            bg=z['g'];bgv=z['gv'];bf=z['f'];bfv=z['fv']
        baseline=flat[(0.,'central')]
        flat_point=ensemble_features(baseline,gauge=gauge)
        flat_draws=[ensemble_features([baseline[j]for j in ix],gauge=gauge)for ix in findices]
        for i in gids:
            xi=plan['profiles_frozen_from_discovery'][i]['nominal_xi'];gi=gids.index(i);fi=fkeys.index((xi,'central'))
            g=features[f'{gauge}:mean']['graphs'][i];f=features[f'{gauge}:mean']['fields'][str((xi,'central'))]
            def score(u,uv,v,vv):return _array_score(np.asarray(u),np.asarray(uv),np.asarray(v),np.asarray(vv),layout['scales'],layout['families'],WEIGHTINGS['primary'])
            central=score(g['vector'],g['valid'],f['vector'],f['valid']);flat_score=score(g['vector'],g['valid'],flat_point['vector'],flat_point['valid'])
            contrasts=[score(bg[d,gi],bgv[d,gi],bf[d,fi],bfv[d,fi])-score(bg[d,gi],bgv[d,gi],flat_draws[d]['vector'],flat_draws[d]['valid'])for d in range(a.draws)]
            control_differences.append(dict(id=i,xi=xi,gauge=gauge,central_log_score=central,flat_score=flat_score,
                log_minus_flat=central-flat_score,mc95=np.quantile(contrasts,[.025,.975]).tolist(),
                interpretation='Negative favorslog overflat;shared24fieldparents andgraphresamples;conditionalMC only'))
    for i in gids:
        xi=plan['profiles_frozen_from_discovery'][i]['nominal_xi']
        for gauge in ['grounded','unit_mass']:
            g=features[f'{gauge}:mean']['graphs'][i]
            controls={'log':fields[(xi,'central')],'flat':flat[(0.,'central')]}
            # Smooth is compared at predeclared xi, on the SAME eight field IDs.
            for sx in plan['smooth_xis']:
                if (sx,'central')not in smooth or len(smooth[(sx,'central')])!=8:raise ValueError('Smooth control incomplete')
            for model,rs in controls.items():
                f=ensemble_features(rs,gauge=gauge);c=compare_features(g,f,tol,gauge=gauge,sampler_allowed=q[i]['sampler_allowed'])
                diagnostics.append(dict(id=i,model=model,xi=xi if model=='log'else 0,parents=len(rs),**c))
            if gauge=='grounded':matrix.append(dict(id=i,xi=xi,**matrix_diagnostic(gs[i],fields[(xi,'central')],rng)))
    save_json(out/'negative_controls.json',diagnostics);save_json(out/'matrix_diagnostics.json',matrix)
    save_json(out/'log_minus_flat.json',control_differences)
    direct=[]
    for family in ['birectangular_level_segment','negative_equicorrelation_region']:
        ids=plan['hypotheses'][family]['ids']
        for k,i in enumerate(ids):
            for j in ids[k+1:]:direct.append(dict(family=family,left=i,right=j,**matrix_diagnostic(gs[i],gs[j],rng)))
    save_json(out/'direct_graph_family_matrices.json',direct)
    save_json(out/'COMPLETE.json',dict(seconds=time.perf_counter()-start,graph_parents=sum(map(len,gs.values())),
        field_parents=24,rows=len(rows),no_parameter_retuning=True,bootstrap_unit='whole physical parent'))
    print(json.dumps(json.loads((out/'COMPLETE.json').read_text())),flush=True)


if __name__=='__main__':main()
