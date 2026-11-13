"""Separate selected-family field campaign using genuine modelwise-lambda probes.

Planning and estimation never sample a field or run Dijkstra. A metric run
requires an explicit coordinator-frozen xi configuration. Existing3Cscreen
files and its producer are never modified.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from src.phase3c_fields import (seed_for,sha256_file,write_json_new,write_npz_new,
                               sampled_points,gamma_probes)
from src.parameter_conversion_3b import load_calibration,implicit_conversion
from src.log_field_3d import sample_coupled_fields,spectral_configuration
from src.gmc_3d import gmc_weights
from src.lfpp_3d import build_lfpp_graph,root_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest


MODELS=('all_plain','drop_smallest','largest_three','all_Ninv_half','all_Ninv')
DEFAULT_CALIBRATION='results/phase3b/calibration/analysis/calibration.json'


@lru_cache(maxsize=4)
def _calibration(path):
    path=Path(path);saved=json.loads(path.read_text())
    with np.load(path.parent/'bootstrap.npz')as archive:
        bootstrap=archive['bootstrap_lambda'].copy()
        xis=archive['xis'].copy()
        index=list(archive['models']).index('all_plain')
    return saved,load_calibration(path),xis,bootstrap[:,index,:]


def modelwise_conversion(xi,path=DEFAULT_CALIBRATION):
    """Interpolate each correction-model lambda, then invert valid branches.

    The central curve is unchanged. MC bands interpolate each paired bootstrap
    lambda curve before inversion; model/window sensitivity is separate.
    """
    saved,old_table,bootstrap_xis,bootstrap=_calibration(str(path))
    inherited=old_table.evaluate(xi)
    if not inherited['valid']:raise ValueError('No real central subcritical finite conversion')
    model_lambdas={};model_gammas={};invalid=[]
    for name in MODELS:
        rows=sorted([r for r in saved['model_rows']if r['model']==name],key=lambda r:r['xi'])
        grid=np.array([r['xi']for r in rows])
        if xi<grid[0]or xi>grid[-1]:raise ValueError('No calibration extrapolation')
        exponent=float(np.interp(xi,grid,[r['lambda']for r in rows]))
        _,gamma,_=implicit_conversion(xi,exponent)
        model_lambdas[name]=exponent
        model_gammas[name]=float(gamma)if np.isfinite(gamma)else None
        if not np.isfinite(gamma):invalid.append(name)
    values=[g for g in model_gammas.values()if g is not None]
    if not np.isclose(model_gammas['all_plain'],inherited['gamma'],rtol=0,atol=2e-14):
        raise ValueError('Central calibration changed unexpectedly')
    right=int(np.searchsorted(bootstrap_xis,xi,side='left'))
    if right<len(bootstrap_xis)and bootstrap_xis[right]==xi:
        samples=bootstrap[:,right]
    else:
        left=right-1;fraction=(xi-bootstrap_xis[left])/(bootstrap_xis[right]-bootstrap_xis[left])
        samples=(1-fraction)*bootstrap[:,left]+fraction*bootstrap[:,right]
    _,sample_gammas,_=implicit_conversion(xi,samples)
    valid=np.isfinite(sample_gammas)
    intervals=np.quantile(sample_gammas[valid],[.025,.975]).tolist()if valid.any()else[None,None]
    # Theoretical bounds are diagnostics on finite ansatzes, not constraints
    # imposed to force the uncertain fits into a plausible branch.
    lower=max((np.sqrt(2.5)-1/np.sqrt(6))*xi-(np.sqrt(15)-2)/6,0.)if xi<=1/np.sqrt(6)else max(.25-xi*xi/2,0.)
    bound_violations=[name for name,lam in model_lambdas.items()if lam<lower-1e-12 or lam>2*xi+1e-12]
    result=dict(inherited,inherited_gamma_systematic=inherited['gamma_systematic'],
        inherited_gamma_mc95=inherited['gamma_mc95'],gamma_systematic=[min(values),max(values)],
        gamma_mc95=intervals,bootstrap_subcritical_fraction=float(valid.mean()),
        gamma_mc95_is_conditional=bool(not valid.all()),
        gamma_systematic_has_invalid_models=bool(invalid),
        model_lambdas=model_lambdas,model_gammas=model_gammas,invalid_models=invalid,
        finite_ansatzes_violating_theoretical_lambda_bounds=bound_violations,
        gamma_probe_rule='interpolate each prespecified correction-model lambda, then invert real subcritical branches',
        gamma_mc_rule='interpolate each whole-field paired bootstrap lambda curve, then invert valid draws',
        asymptotic_map_validated=False)
    result['status']='exact_flat_endpoint'if xi==0 else('provisional_branch_or_model_sensitive'if invalid or valid.mean()<.975 else'provisional_finite_resolution')
    return result


def plan_registry(path,seed=202611123110):
    path=Path(path)
    if seed>=202611123900:raise ValueError('Selected discovery driver must not use the reserved confirmation seed bases')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['new_seed_base']!=seed:raise ValueError('Existing selected seed plan differs')
        return saved['records']
    old_path=Path('results/phase3c/fields/registry.json')
    old=json.loads(old_path.read_text())['records']
    records=[dict(r,materialization_status='previously_materialized')for r in old]
    old_seeds={r['field_seed']for r in old}
    for phase in[Path('results/phase3a'),Path('results/phase3b')]:
        for archive_path in phase.glob('*/field_*.npz'):
            with np.load(archive_path)as archive:
                if 'seed'in archive:old_seeds.add(int(archive['seed']))
                elif 'header'in archive:
                    metadata=json.loads(str(archive['header']))
                    if 'seed'in metadata:old_seeds.add(int(metadata['seed']))
    old_seeds.update(range(202611113510,202611113530))
    sources=path.parent/'selected_source_fields'
    for identifier in range(48,100):
        field_seed=seed_for(seed,identifier,1)
        if field_seed in old_seeds:raise ValueError('New discovery seed collision')
        old_seeds.add(field_seed)
        records.append({'realization':identifier,'cohort':'phase3c_selected_new_discovery',
            'cohort_index':identifier-48,'field_seed':field_seed,
            'root_seed':seed_for(seed,identifier,2),
            'source_field':str(sources/f'field_{identifier:03d}.npz'),
            'source_field_sha256':None,'array_sha256':None,
            'complete_source_resolution_tuple':[32,48],
            'materialization_status':'planned_not_generated',
            'independence':'one new complete physical Gaussian field family'})
    if len({r['field_seed']for r in records})!=100:raise ValueError('Duplicate physical fields')
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json_new(path,{'records':records,'counts':{'reused':48,'planned_new':52,'total_discovery':100},
        'previous_registry':str(old_path),'previous_registry_sha256':sha256_file(old_path),
        'new_seed_base':seed,'reserved_holdout_base_not_used':202611123900,
        'new_Gaussian_fields_generated_at_planning':0,
        'materialization_policy':'new arrays generated only by an explicitly frozen selected metric batch; sidecars carry realized hashes; this plan is immutable'})
    return records


def materialize(row):
    """Called only during an authorized metric batch, never by plan/estimate."""
    row=dict(row);path=Path(row['source_field'])
    if row['materialization_status']=='previously_materialized':
        if sha256_file(path)!=row['source_field_sha256']:raise ValueError('Existing field changed')
        return row
    path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=path.with_suffix('.json')
    if manifest_path.exists():
        actual=json.loads(manifest_path.read_text())
        if actual['field_seed']!=row['field_seed']or sha256_file(path)!=actual['source_field_sha256']:
            raise ValueError('Materialized field identity changed')
        return actual
    if not path.exists():
        draws=sample_coupled_fields(row['complete_source_resolution_tuple'],np.random.default_rng(row['field_seed']))
        write_npz_new(path,**{f'h{n}':f.h for n,f in draws.items()},
            header=json.dumps({'seed':row['field_seed'],'fields':{str(n):f.metadata for n,f in draws.items()}}))
    with np.load(path)as archive:
        metadata=json.loads(str(archive['header']))
        if metadata['seed']!=row['field_seed']:raise ValueError('Unexpected saved field seed')
        row['array_sha256']={str(n):hashlib.sha256(archive[f'h{n}'].tobytes()).hexdigest()for n in[32,48]}
    row['source_field_sha256']=sha256_file(path)
    row['materialization_status']='materialized_during_selected_batch'
    write_json_new(manifest_path,row)
    return row


def candidates(row,n,xi):
    identifier=row['realization']
    yield Path(f'results/phase3c/fields/screen/xi{xi:g}/metric_{identifier:03d}_n{n}.npz')
    yield Path(f'results/phase3c/fields/pilot/xi{xi:g}/metric_{identifier:03d}_n{n}.npz')
    if row['cohort'].startswith('phase3b'):
        index=row['cohort_index']
        for directory in['holdout_fields','holdout_stencil_control','historical_candidate_control']:
            yield Path(f'results/phase3b/{directory}/xi{xi:g}/metric_{index:03d}_n{n}.npz')


def cached_probes(row,n,xi,stencil,probes,points):
    """Reuse individual gamma rows only if actual law and point indices match."""
    found={}
    for path in candidates(row,n,xi):
        if not path.exists():continue
        with np.load(path)as archive:
            header=json.loads(str(archive['header']))
            if(header.get('source_field_sha256')!=row['source_field_sha256']or header['n']!=n
               or header['xi']!=xi or header['stencil']!=stencil or header.get('boundary','torus')!='torus'):
                continue
            for (_,gamma),roots in zip(probes,points):
                if gamma in found:continue
                for index,old in enumerate(header['probes']):
                    if old['gamma']==gamma and np.array_equal(archive['points'][index],roots):
                        found[gamma]={'grid':archive['grid'][index].copy(),'cdfs':archive['cdfs'][index].copy(),
                            'root_means':archive['root_means'][index].copy(),'matrix':archive['matrices'][index].copy(),
                            'distance_bin_width':float(archive['bin_width'][index]),
                            'source_metric':str(path),'source_probe_index':index,
                            'source_metric_sha256':sha256_file(path)}
                        break
    return found


def read_frozen_config(path):
    config=json.loads(Path(path).read_text())
    if config.get('status')!='frozen_by_coordinator'or not config.get('metric_jobs_authorized',False):
        raise ValueError('Selected xi list must be explicitly frozen by the coordinator before metrics')
    if not config.get('xis')or config.get('gamma_rule')!='modelwise_lambda':
        raise ValueError('A frozen modelwise-lambda xi list is required')
    if config.get('points',16)!=16:raise ValueError('This cache-preserving selected design uses16points')
    return config


def estimate(args):
    config=read_frozen_config(args.config)
    records=plan_registry(args.registry,args.seed)
    conversions=[modelwise_conversion(x,config.get('calibration',DEFAULT_CALIBRATION))for x in config['xis']]
    # Deliberately conservative empirical bounds from the completed3Cpilot.
    estimate_rows=[]
    for n,count in[(32,100),(48,48)]:
        estimates={'n':n,'families':count,'xi_count':len(config['xis']),
            'full_record_count':count*len(config['xis']),
            'no_cache_seconds_upper':count*len(config['xis'])*(.95 if n==32 else 3.6),
            'sequential_peak_memory_bytes_estimate':500_000_000,
            'cached_central_rows_available':min(count,48 if n==32 else 12)*len(config['xis']),
            'batch_families':12,'expected_batch_seconds_upper':12*len(config['xis'])*(.95 if n==32 else 3.6)}
        estimate_rows.append(estimates)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    write_json_new(output/'preflight_estimate.json',{'configuration':config,'registry_sha256':sha256_file(args.registry),
        'rows':estimate_rows,'conversions':conversions,'no_Gaussian_fields_or_metrics_generated':True,
        'cache_caveat':'available only where selected xi is on the existing14-point grid; genuine sensitivity rows can differ',
        'source_hashes':{p:sha256_file(p)for p in['experiments/phase3c_selected_fields.py','src/phase3c_fields.py']}})
    print(json.dumps(estimate_rows,indent=2))


def run(args):
    config=read_frozen_config(args.config)
    if not args.field_ids:raise ValueError('Explicit field batch IDs required')
    if len(args.field_ids)>12:raise ValueError('Split selected field work into batches of at most12families')
    records=plan_registry(args.registry,args.seed)
    if any(i not in range(100)for i in args.field_ids):raise ValueError('Discovery IDs are0..99')
    if 48 in args.sizes and any(i>=48 for i in args.field_ids):
        raise ValueError('Initial N48selected stage is restricted to discovery IDs0..47')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    source_paths=['experiments/phase3c_selected_fields.py','src/phase3c_fields.py','src/log_field_3d.py',
                  'src/gmc_3d.py','src/lfpp_3d.py','src/metric_measure_3a.py','src/parameter_conversion_3b.py']
    source_hashes={p:sha256_file(p)for p in source_paths}
    calibration_path=Path(config.get('calibration',DEFAULT_CALIBRATION))
    metadata={'configuration':config,'registry':args.registry,'registry_sha256':sha256_file(args.registry),
              'source_hashes':source_hashes,'calibration_sha256':sha256_file(calibration_path),
              'calibration_bootstrap_sha256':sha256_file(calibration_path.parent/'bootstrap.npz')}
    if(output/'metadata.json').exists():
        if json.loads((output/'metadata.json').read_text())!=metadata:
            raise ValueError('Frozen selected setting/source changed; use a new output directory')
    else:
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for path in source_paths:shutil.copyfile(path,snapshot/Path(path).name)
        write_json_new(output/'metadata.json',metadata)
        write_json_new(output/'environment.json',environment_manifest())
    plans=output/'batch_plans';plans.mkdir(exist_ok=True)
    plan={'field_ids':args.field_ids,'sizes':args.sizes,'xis':config['xis'],
        'scientific_question':config.get('scientific_question','Test selected-family feature stability under larger independent field ensembles and genuine modelwise probes'),
        'record_count':len(args.field_ids)*len(args.sizes)*len(config['xis']),
        'no_cache_seconds_upper':len(args.field_ids)*len(config['xis'])*sum(.95 if n==32 else 3.6 for n in args.sizes),
        'peak_memory_bytes_estimate':500_000_000,'source_hashes':source_hashes,
        'reuse':'matching per-probe16-point distributions from completed3Cscreen/3Bcontrols; new gamma rows use shared missing-root Dijkstra',
        'physical_seed_registry':args.registry,'registry_sha256':sha256_file(args.registry)}
    plan_path=plans/f'{args.label}.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing batch plan differs')
    else:write_json_new(plan_path,plan)
    conversions={x:modelwise_conversion(x,config.get('calibration',DEFAULT_CALIBRATION))for x in config['xis']}
    started=time.perf_counter();results=[]
    for identifier in args.field_ids:
        row=materialize(records[identifier])
        uniforms=np.random.default_rng(row['root_seed']).random(config.get('points',16))
        with np.load(row['source_field'])as archive:fields={n:archive[f'h{n}']for n in args.sizes}
        for n,h in fields.items():
            _,_,variance=spectral_configuration(n)
            for xi,conversion in conversions.items():
                folder=output/f'xi{xi:g}';folder.mkdir(exist_ok=True)
                destination=folder/f'metric_{identifier:03d}_n{n}.npz'
                if destination.exists():continue
                tick=time.perf_counter();probes=gamma_probes(conversion)
                measures={g:gmc_weights(h,g,variance)for _,g in probes}
                points=[sampled_points(measures[g].probabilities,uniforms)for _,g in probes]
                cached=cached_probes(row,n,xi,config['stencil'],probes,points)
                missing=[g for g in measures if g not in cached]
                numerical=dict(cached);metric_meta={'stencil':config['stencil'],'boundary':'torus'}
                if missing:
                    roots=np.unique(np.concatenate([points[next(i for i,(_,value)in enumerate(probes)if value==g)]for g in missing]))
                    graph,metric_meta=build_lfpp_graph(h,xi,stencil=config['stencil'],boundary='torus')
                    distances=root_distances(graph,roots)
                    for gamma in missing:
                        selected=points[next(i for i,(_,g)in enumerate(probes)if g==gamma)]
                        d=distances[np.searchsorted(roots,selected)]
                        numerical[gamma]={**weighted_root_cdfs(d,measures[gamma].probabilities.ravel(),bins=config.get('bins',1024)),
                            'matrix':d[:,selected],'source_metric':None,'source_probe_index':None,'source_metric_sha256':None}
                    del distances,graph
                probe_rows=[]
                for(band,gamma),selected in zip(probes,points):
                    measure=measures[gamma];value=numerical[gamma]
                    probe_rows.append({'band':band,'gamma':gamma,'log_total_mass':measure.log_total_mass,
                        'total_mass':measure.total_mass,'collision_probability':float(np.sum(measure.probabilities**2)),
                        'maximum_atom':float(measure.probabilities.max()),
                        'observed_duplicate_point_fraction':float(1-len(np.unique(selected))/len(selected)),
                        'source_metric':value['source_metric'],'source_probe_index':value['source_probe_index'],
                        'source_metric_sha256':value['source_metric_sha256']})
                header={**row,'n':n,'sampling_n':n,'xi':xi,'probes':probe_rows,'conversion':conversion,
                    'stencil':config['stencil'],'boundary':'torus','points':config.get('points',16),
                    'field_variance':variance,'metric_metadata':metric_meta,
                    'gamma_rule':'modelwise_lambda','control':'calibrated_log_field',
                    'gauge':'grounded raw metric; unit_mass derived later',
                    'seconds':time.perf_counter()-tick,'new_unique_gamma_rows':len(missing),
                    'reused_unique_gamma_rows':len(cached),
                    'sampling_count':'one physical field; all xi/probes/resolutions share its cluster'}
                write_npz_new(destination,grid=np.stack([numerical[g]['grid']for _,g in probes]),
                    cdfs=np.stack([numerical[g]['cdfs']for _,g in probes]).astype(np.float32),
                    root_means=np.stack([numerical[g]['root_means']for _,g in probes]),
                    matrices=np.stack([numerical[g]['matrix']for _,g in probes]),points=np.stack(points),
                    bin_width=np.array([numerical[g]['distance_bin_width']for _,g in probes]),header=json.dumps(header))
                results.append({'path':str(destination),'seconds':header['seconds'],'new_gamma_rows':len(missing),'reused_gamma_rows':len(cached)})
                print(json.dumps({'field':identifier,'n':n,'xi':xi,'seconds':header['seconds'],
                                  'new_gamma_rows':len(missing),'reused_gamma_rows':len(cached)}),flush=True)
    batches=output/'batches';batches.mkdir(exist_ok=True)
    write_json_new(batches/f'{args.label}.json',{'field_ids':args.field_ids,'sizes':args.sizes,
        'configuration':config,'seconds':time.perf_counter()-started,
        'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'records':results})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['plan','estimate','run'])
    parser.add_argument('--registry',default='results/phase3c/fields/selected_registry.json')
    parser.add_argument('--seed',type=int,default=202611123110)
    parser.add_argument('--config')
    parser.add_argument('--output',default='results/phase3c/fields/selected')
    parser.add_argument('--field-ids',type=int,nargs='+')
    parser.add_argument('--sizes',type=int,nargs='+',default=[32])
    parser.add_argument('--label',default='selected_batch')
    args=parser.parse_args()
    if args.command=='plan':print(json.dumps({'registry':args.registry,'planned_families':len(plan_registry(args.registry,args.seed)),'new_fields_generated':0}))
    elif args.config is None:parser.error('estimate/run requires a frozen --config')
    elif args.command=='estimate':estimate(args)
    else:run(args)


if __name__=='__main__':main()
