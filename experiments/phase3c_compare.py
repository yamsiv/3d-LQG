"""Rich cross-model scores and cluster uncertainty from saved Phase3C data.

One feature bootstrap per graph/field is reused over the entire response
surface. This is an analysis Cartesian product, not a simulation Cartesian
product. Common field indices preserve xi/gamma dependence. Numerical errors
remain separately labeled rather than being hidden in a single interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.phase3a_compare import save_json
from experiments.phase3a_ensembles import write_npz
from experiments.phase3b_search import graph_groups,field_groups
from src.family_search_3c import ensemble_features,compare_features,WEIGHTINGS
from src.metric_features_3b import grouped_feature_distance
from src.provenance import environment_manifest

ROOT=Path('results/phase3c')


def graph_quality(records):
    identifier=records[0]['header']['id'];metas=[r['header']['sampler_metadata'] for r in records]
    diagnostics=[leg for m in metas for leg in m.get('leg_diagnostics',[])]
    signed=max((abs(d['sum_log_mean_value_ratio']) for d in diagnostics),default=0.)
    absolute=max((d['sum_abs_log_mean_value_ratio'] for d in diagnostics),default=0.)
    exact=all(m.get('cell_minima_exact',False) for m in metas)
    grid=all(m.get('grid_values_exact',False) for m in metas)
    unsafe=identifier in ['C019','C023'] or signed>.03 or absolute>.10
    return dict(sampler_allowed=not unsafe,status='unsafe_excluded' if unsafe else
        'exact_continuous_minima' if exact else 'exact_grid_polygonal_minima' if grid else 'provisional_screening',
        maximum_abs_signed_leg_sum=signed,maximum_leg_abs_sum=absolute,
        interpretation='Observed harmonic exclusion flag only; a small value does not validate the Brownian path or joint minima',
        alpha1=float(metas[0]['alpha1']),parents=len(records),
        unresolved_generic_error='Exact-harmonic StageI controls still show few-percent distance and ordinal-minimum shifts')


def _array_score(g,gv,f,fv,scales,families,weights):
    """Score independent bootstrap features without repeating raw-data work."""
    mask=gv&fv;parts=[];ws=[]
    delta=(g-f)/scales
    for name in dict.fromkeys(families):
        choose=mask&(np.asarray(families)==name)
        if choose.any() and weights.get(name,0)>0:
            parts.append(np.sqrt(np.mean(delta[choose]**2)));ws.append(weights[name])
    return float(np.average(parts,weights=ws)) if ws else np.nan


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--graphs',nargs='+');p.add_argument('--fields',default=str(ROOT/'fields/screen'))
    p.add_argument('--graph-n',type=int,default=2048);p.add_argument('--field-n',type=int,default=32)
    p.add_argument('--minimum-graph-parents',type=int,default=12);p.add_argument('--expect-fields',type=int,default=48)
    p.add_argument('--design',default=str(ROOT/'covariance_design/initial_design.json'))
    p.add_argument('--bootstrap',type=int,default=24);p.add_argument('--seed',type=int,default=202611123401)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'scores.json').exists():raise FileExistsError(out/'scores.json')
    start=time.perf_counter()
    dirs=a.graphs or ['results/phase3b/screen_graphs','results/phase3b/adaptive_coarse_graphs']+[
        str(d) for d in sorted((ROOT/'screen/batches').glob('batch_*'))]
    graphs=graph_groups(dirs,a.graph_n);fields=field_groups([a.fields],a.field_n)
    if not graphs or not fields:raise ValueError('No complete inputs')
    if any(len(r)!=a.expect_fields for r in fields.values()):raise ValueError('Incomplete field ensemble')
    ids=[r['realization'] for r in next(iter(fields.values()))]
    if any([r['realization'] for r in rows]!=ids for rows in fields.values()):raise ValueError('Field identities differ across parameters')
    specs=json.loads(Path(a.design).read_text())['configurations'];spec_by={s['id']:s for s in specs}
    inputs=sorted({r['source'] for records in list(graphs.values())+list(fields.values()) for r in records})
    fingerprint=dict(configuration=vars(a),graph_directories=dirs,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs})
    fingerprint_path=out/'analysis_configuration.json'
    if fingerprint_path.exists():
        if json.loads(fingerprint_path.read_text())!=fingerprint:raise ValueError('Analysis inputs/config/source differ from checkpoint')
    else:save_json(fingerprint_path,fingerprint)
    tolerance=json.loads((ROOT/'families/statistics/tolerance_calibration.json').read_text())
    quality={i:graph_quality(r) for i,r in graphs.items()}
    for i,q in quality.items():
        if q['parents']<a.minimum_graph_parents:q['sampler_allowed']=False;q['status']='incomplete_parent_count'
    if (out/'point_rows.json').exists():
        rows=json.loads((out/'point_rows.json').read_text())
        saved_features=json.loads((out/'features.json').read_text())
        primary_g=saved_features['grounded:mean']['graphs']
        primary_f={key:saved_features['grounded:mean']['fields'][str(key)] for key in fields}
    else:
        saved_features={};rows=[];primary_g={};primary_f={}
        for normalization in ['mean','median']:
            gf={i:ensemble_features(r,normalization=normalization) for i,r in graphs.items()}
            for gauge in ['grounded','unit_mass']:
                ff={key:ensemble_features(r,gauge=gauge,normalization=normalization) for key,r in fields.items()}
                if normalization=='mean' and gauge=='grounded':primary_g=gf;primary_f=ff
                saved_features[f'{gauge}:{normalization}']={'graphs':gf,'fields':{str(k):f for k,f in ff.items()}}
                for identifier,g in gf.items():
                    for xi in sorted({k[0] for k in fields}):
                        probes={band:feature for (x,band),feature in ff.items() if x==xi}
                        base=fields[(xi,'central')][0];conv=base['header']['conversion']
                        eligible=xi>0 and conv.get('bootstrap_subcritical_fraction',1)>=.975 and quality[identifier]['sampler_allowed']
                        for weighting in WEIGHTINGS:
                            comparisons={band:compare_features(g,f,tolerance,gauge=gauge,normalization=normalization,
                                weighting=weighting,sampler_allowed=quality[identifier]['sampler_allowed'],calibration_allowed=eligible)
                                for band,f in probes.items()}
                            central=comparisons['central'];robust=max(c['score'] for c in comparisons.values())
                            componentwise=all(c['componentwise_low'] for c in comparisons.values())
                            gamma=base['gamma'];dimension=gamma/xi if xi else 3.
                            special_tension=identifier in ['C002','C007'] and xi>0 and dimension<4.6
                            row=dict(id=identifier,triple=spec_by[identifier]['triple'],xi=xi,gamma=gamma,
                                band='central',gauge=gauge,normalization=normalization,weighting=weighting,
                                score=central['score'],robust_score=robust,epsilon=central['epsilon'],
                                eligible=bool(eligible),candidate=bool(eligible and robust<=central['epsilon']),
                                componentwise_low=componentwise,sampler_status=quality[identifier]['status'],
                                graph_parents=len(graphs[identifier]),field_parents=len(fields[(xi,'central')]),
                                graph_n=a.graph_n,field_n=a.field_n,
                                probe_scores={b:c['score'] for b,c in comparisons.items()},
                                dimension_proxy=dimension,dimension_flag='tension_with_common_corrected_or_published_special_graph_estimate' if special_tension else 'not_resolved_by_small_graph_growth',
                                calibration_scope='finite-window linked proxy; gamma probes not a full systematic bound')
                            if weighting=='primary':
                                row.update(signed_residuals=central['signed_residuals'],valid=central['valid'],
                                    family_components=central['family_components'],feature_flags=central['feature_flags'],
                                    family_flags=central['family_flags'],
                                    probe_family_components={b:c['family_components'] for b,c in comparisons.items()})
                            rows.append(row)
        save_json(out/'features.json',saved_features);save_json(out/'reliability.json',quality)
        save_json(out/'point_rows.json',rows)
    graph_ids=sorted(graphs);field_keys=list(fields);rng=np.random.default_rng(a.seed)
    boot_g=[];boot_g_valid=[];boot_f=[];boot_f_valid=[]
    checkpoints=sorted(out.glob('feature_bootstrap_through_*.npz'))
    if checkpoints:
        with np.load(checkpoints[-1]) as checkpoint:
            if list(checkpoint['graph_ids'])!=graph_ids or list(checkpoint['field_keys'])!=[str(k) for k in field_keys]:
                raise ValueError('Bootstrap checkpoint identities differ')
            boot_g=list(checkpoint['g']);boot_g_valid=list(checkpoint['gv'])
            boot_f=list(checkpoint['f']);boot_f_valid=list(checkpoint['fv'])
            rng.bit_generator.state=json.loads(str(checkpoint['rng_state']))
    for draw in range(len(boot_g),a.bootstrap):
        gf=[];gvalid=[]
        for identifier in graph_ids:
            rs=graphs[identifier];indices=rng.integers(len(rs),size=len(rs))
            f=ensemble_features([rs[i] for i in indices]);gf.append(f['vector']);gvalid.append(f['valid'])
        indices=rng.integers(len(ids),size=len(ids));ff=[];fvalid=[]
        for key in field_keys:
            f=ensemble_features([fields[key][i] for i in indices]);ff.append(f['vector']);fvalid.append(f['valid'])
        boot_g.append(gf);boot_g_valid.append(gvalid);boot_f.append(ff);boot_f_valid.append(fvalid)
        if draw%8==7 or draw+1==a.bootstrap:
            write_npz(out/f'feature_bootstrap_through_{draw+1:03d}.npz',g=np.asarray(boot_g),gv=np.asarray(boot_g_valid),
                f=np.asarray(boot_f),fv=np.asarray(boot_f_valid),graph_ids=np.array(graph_ids),
                field_keys=np.array([str(k) for k in field_keys]),rng_state=json.dumps(rng.bit_generator.state))
            print(json.dumps({'feature_bootstrap_complete':draw+1,'draws':a.bootstrap}),flush=True)
    layout=next(iter(primary_g.values()))
    boot_g=np.asarray(boot_g);boot_g_valid=np.asarray(boot_g_valid);boot_f=np.asarray(boot_f);boot_f_valid=np.asarray(boot_f_valid)
    write_npz(out/'graph_bootstrap.npz',ids=np.array(graph_ids),vectors=boot_g,valid=boot_g_valid,
        point_vectors=np.array([primary_g[i]['vector'] for i in graph_ids]),
        point_valid=np.array([primary_g[i]['valid'] for i in graph_ids]),
        scales=layout['scales'],names=np.array(layout['names']),families=np.array(layout['families']))
    write_npz(out/'field_bootstrap.npz',keys=np.array([str(k) for k in field_keys]),vectors=boot_f,valid=boot_f_valid,
        point_vectors=np.array([primary_f[k]['vector'] for k in field_keys]),
        point_valid=np.array([primary_f[k]['valid'] for k in field_keys]),
        scales=layout['scales'],names=np.array(layout['names']))
    for row in rows:
        if row['gauge']!='grounded' or row['normalization']!='mean':continue
        gi=graph_ids.index(row['id']);keys=[j for j,(xi,band) in enumerate(field_keys) if xi==row['xi']]
        values=[]
        for draw in range(a.bootstrap):
            values.append(max(_array_score(boot_g[draw,gi],boot_g_valid[draw,gi],boot_f[draw,j],boot_f_valid[draw,j],
                layout['scales'],layout['families'],WEIGHTINGS[row['weighting']]) for j in keys))
        row['robust_mc95']=np.nanquantile(values,[.025,.975]).tolist()
        row['mc_scope']=f'conditional{a.bootstrap}whole-parent feature bootstrap; common field indices across xi/probes; no systematic error bound'
    growth=[]
    if a.graph_n==2048:
        small=graph_groups(dirs,512)
        for i,records in graphs.items():
            if i not in small or len(records)!=len(small[i]):continue
            large_by_seed={r['header']['seed']:r for r in records};small_by_seed={r['header']['seed']:r for r in small[i]}
            if set(large_by_seed)!=set(small_by_seed):raise ValueError('Unpaired graph sizes')
            seeds=sorted(large_by_seed);u=np.array([np.mean(small_by_seed[s]['root_means']) for s in seeds]);v=np.array([np.mean(large_by_seed[s]['root_means']) for s in seeds])
            beta=np.log(v.mean()/u.mean())/np.log(4.);values=[]
            for _ in range(64):
                ix=rng.integers(len(u),size=len(u));values.append(np.log(v[ix].mean()/u[ix].mean())/np.log(4.))
            growth.append(dict(id=i,beta_finite=beta,beta_mc95=np.quantile(values,[.025,.975]).tolist(),
                inverse_growth_proxy=1/beta,warning='two-size finite proxy; not Hausdorff dimension',physical_parents=len(u)))
    save_json(out/'growth.json',growth)
    save_json(out/'scores.json',dict(rows=rows,configuration=vars(a),provenance=environment_manifest(),
        graph_sources=dirs,field_source=a.fields,
        failed_or_absent_configurations=[s for s in specs if s['id'] not in graphs],
        physical_field_count=len(ids),bootstrap_draws=a.bootstrap,elapsed_seconds=time.perf_counter()-start,
        score_units='weighted family RMS in declared natural scales; not a likelihood or equality test',
        interpretation='Primary mean/grounded response surface; median and other gauge prespecified; sampler errors separate; only eligible measured nodes may form provisional families'))
    print(json.dumps({'output':str(out),'graph_configs':len(graphs),'fields':len(ids),'rows':len(rows),
                      'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':main()
