"""Wholly new confirmation fields; requires a frozen independent testing plan.

The plan and estimate commands never generate fields or metrics. Models use
the same underlying held-out Fourier field and root-uniform stream. Gauges
are applied later to raw saved metric-measure records, never fit-selected.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
import numpy as np

from experiments.phase3c_selected_fields import modelwise_conversion,DEFAULT_CALIBRATION
from src.field_metrics_3c import control_field,metric_arrays
from src.log_field_3d import sample_coupled_fields,spectral_configuration,TorusLogField
from src.extend_log_field_3c import extend_sharp_field
from src.phase3c_fields import seed_for,sha256_file,write_json_new,write_npz_new,gamma_probes
from src.provenance import environment_manifest


def prior_identities():
    seeds=set(range(202611113510,202611113530));root_seeds=set();paths=set();registries={}
    hashes={'32':set(),'48':set()}
    for name in['registry.json','selected_registry.json']:
        path=Path('results/phase3c/fields')/name
        if not path.exists():continue
        registries[str(path)]=sha256_file(path)
        for row in json.loads(path.read_text())['records']:
            seeds.add(row['field_seed']);root_seeds.add(row['root_seed'])
            source=Path(row['source_field'])
            if source.exists():paths.add(source)
    for phase in['phase3a','phase3b']:
        paths.update(Path('results',phase).glob('*/field_*.npz'))
    for n in[32,48]:paths.update(Path('results/phase3b/calibration/main').glob(f'field_f*_n{n}.npz'))
    sources={}
    for path in sorted(paths):
        sources[str(path)]=sha256_file(path)
        with np.load(path)as archive:
            if 'seed'in archive:seeds.add(int(archive['seed']))
            elif 'field_seed'in archive:seeds.add(int(archive['field_seed']))
            elif 'header'in archive:
                metadata=json.loads(str(archive['header']))
                if 'seed'in metadata:seeds.add(int(metadata['seed']))
            for n in[32,48]:
                key=f'h{n}'
                if key in archive:hashes[str(n)].add(hashlib.sha256(archive[key].tobytes()).hexdigest())
                elif 'h'in archive and archive['h'].shape==(n,n,n):
                    hashes[str(n)].add(hashlib.sha256(archive['h'].tobytes()).hexdigest())
    return {'field_seeds':sorted(seeds),'root_seeds':sorted(root_seeds),
        'array_hashes':{n:sorted(v)for n,v in hashes.items()},'source_sha256':sources,
        'registry_sha256':registries}


def plan(path,base=202611123900):
    path=Path(path)
    if base<202611123900:raise ValueError('Confirmation uses the reserved new seed range')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['seed_base']!=base:raise ValueError('Existing confirmation seed plan differs')
        return saved
    prior=prior_identities();used=set(prior['field_seeds'])|set(prior['root_seeds'])
    records=[]
    for identifier in range(1000,1024):
        field_seed=seed_for(base,identifier,1);root_seed=seed_for(base,identifier,2)
        if field_seed in used or root_seed in used or field_seed==root_seed:
            raise ValueError('Confirmation seed collision; no hidden reseeding')
        used.update([field_seed,root_seed])
        records.append({'realization':identifier,'cohort':'phase3c_wholly_new_confirmation',
            'field_seed':field_seed,'root_seed':root_seed,'complete_source_resolution_tuple':[32,48],
            'source_field':str(path.parent/'confirmation_sources'/f'field_{identifier:04d}.npz'),
            'materialization_status':'planned_not_generated'})
    saved={'records':records,'seed_base':base,'physical_field_count':24,'prior_identities':prior,
        'new_fields_generated_at_planning':0,'controls':'log,smooth andflat share these same field IDs;not extra realizations',
        'inference_unit':'whole physical field across grids,xi,gamma,controls andgauges'}
    path.parent.mkdir(parents=True,exist_ok=True);write_json_new(path,saved)
    return saved


def materialize(row,prior):
    row=dict(row);path=Path(row['source_field']);sidecar=path.with_suffix('.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if sidecar.exists():
        actual=json.loads(sidecar.read_text())
        if actual['field_seed']!=row['field_seed']or sha256_file(path)!=actual['source_field_sha256']:
            raise ValueError('Saved confirmation source changed')
        return actual
    if not path.exists():
        fields=sample_coupled_fields([32,48],np.random.default_rng(row['field_seed']))
        write_npz_new(path,**{f'h{n}':f.h for n,f in fields.items()},
            header=json.dumps({'seed':row['field_seed'],'fields':{str(n):f.metadata for n,f in fields.items()}}))
    actual_hash=sha256_file(path)
    if actual_hash in set(prior['source_sha256'].values()):raise ValueError('Confirmation archive duplicates an old field')
    with np.load(path)as archive:
        metadata=json.loads(str(archive['header']))
        if metadata['seed']!=row['field_seed']:raise ValueError('Unexpected confirmation seed')
        hashes={str(n):hashlib.sha256(archive[f'h{n}'].tobytes()).hexdigest()for n in[32,48]}
    if any(hashes[n]in prior['array_hashes'][n]for n in hashes):
        raise ValueError('Confirmation array duplicates previous field data')
    for previous in path.parent.glob('field_*.json'):
        old=json.loads(previous.read_text())
        if any(old['array_sha256'][n]==hashes[n]for n in hashes):
            raise ValueError('Duplicate new confirmation arrays')
    row.update(source_field_sha256=actual_hash,array_sha256=hashes,
               materialization_status='materialized_during_frozen_confirmation')
    write_json_new(sidecar,row);return row


def frozen_config(path):
    config=json.loads(Path(path).read_text())
    if(config.get('stage')!='independent_confirmation'or config.get('status')!='frozen_by_coordinator'
       or not config.get('metric_jobs_authorized',False)):
        raise ValueError('Freeze the independent testing plan before any holdout field or metric')
    if not config.get('hypotheses')or not config.get('xis')or config.get('gamma_rule')!='modelwise_lambda':
        raise ValueError('Nonempty fixed hypotheses andmodelwise xi settings required')
    if config.get('points',16)!=16:raise ValueError('Confirmation design uses16measure-sampled points')
    if not set(config.get('smooth_xis')or[]).issubset(config['xis']):
        raise ValueError('Smooth control xi must be frozen among the selected xi list')
    if not set(config.get('smooth_field_ids',[])).issubset(range(1000,1024)):
        raise ValueError('Smooth controls must share the same held-out physical field IDs')
    if config.get('enable_n64',False):
        if(not config.get('n64_field_ids')or not set(config.get('n64_field_ids',[])).issubset(range(1000,1008))
           or len(config.get('n64_xis')or[])!=3 or not set(config['n64_xis']).issubset(config['xis'])):
            raise ValueError('N64scope is first8physical parents and3frozen mainxi')
        pilot=Path(config.get('n64_deterministic_pilot',''))
        if not pilot.is_file():raise ValueError('N64requires a saved deterministic resourcepilot')
        checked=json.loads(pilot.read_text())
        if checked['peak_rss_native']>1_500_000_000:raise ValueError('N64deterministicpilot exceeds1.5GBbudget')
    return config


def tasks(config,models,xis=None):
    result=[]
    for model in models:
        if model not in['log','smooth','flat']:raise ValueError('Unknown model')
        parameters=[0.]if model=='flat'else(config['xis']if model=='log'else(config.get('smooth_xis')or[]))
        for xi in parameters:
            if xis is not None and xi not in xis:continue
            conversion=modelwise_conversion(xi,config.get('calibration',DEFAULT_CALIBRATION))
            allowed=['central']if model=='flat'else config.get(f'{model}_gamma_bands',['central','low','high']if model=='log'else['central'])
            probes=[p for p in gamma_probes(conversion)if p[0]in allowed]
            # Model-indexed diagnostics are coherent GLOBAL calibration curves,
            # never an independently optimized gamma for each graph covariance.
            if model in config.get('diagnostic_model_controls',[]):
                for name,gamma in conversion['model_gammas'].items():
                    if gamma is not None:probes.append((f'model_{name}',float(gamma)))
            if not probes:raise ValueError('At least one frozen gamma probe required')
            result.append((model,xi,conversion,probes))
    return result


def source_hashes():
    paths=['experiments/phase3c_confirmation_fields.py','experiments/phase3c_selected_fields.py',
        'src/field_metrics_3c.py','src/lfpp_periodic_3c.py','src/lfpp_3d.py',
        'src/phase3c_fields.py','src/log_field_3d.py','src/gmc_3d.py',
        'src/metric_measure_3a.py','src/parameter_conversion_3b.py','src/extend_log_field_3c.py']
    return {p:sha256_file(p)for p in paths}


def materialize_n64(parent,registry):
    """Separate extension archive;never replace the parent'sh32/h48file."""
    identifier=parent['realization']
    if identifier not in range(1000,1008):raise ValueError('N64onlyforfirst8confirmation parents')
    high_seed=seed_for(registry['seed_base'],identifier,3)
    forbidden=set(registry['prior_identities']['field_seeds'])|set(registry['prior_identities']['root_seeds'])
    forbidden.update(r[k]for r in registry['records']for k in['field_seed','root_seed'])
    high_seeds={seed_for(registry['seed_base'],i,3)for i in range(1000,1008)}
    if len(high_seeds)!=8 or high_seeds.intersection(forbidden):raise ValueError('Independenthighmode seed collision')
    directory=Path(parent['source_field']).parent.parent/'confirmation_extensions';directory.mkdir(exist_ok=True)
    path=directory/f'field_{identifier:04d}_n64.npz';sidecar=path.with_suffix('.json')
    if sidecar.exists():
        actual=json.loads(sidecar.read_text())
        if(actual['parent_source_field_sha256']!=parent['source_field_sha256']
           or actual['high_mode_seed']!=high_seed or sha256_file(path)!=actual['source_field_sha256']):
            raise ValueError('N64extension lineage orarchive changed')
        return actual
    with np.load(parent['source_field'])as archive:
        h48=archive['h48'];metadata=json.loads(str(archive['header']))['fields']['48']
    spectrum,covariance,variance=spectral_configuration(48)
    retained=TorusLogField(h48,variance,covariance,spectrum,metadata)
    if not path.exists():
        extended=extend_sharp_field(retained,64,np.random.default_rng(high_seed))
        write_npz_new(path,h64=extended.h,header=json.dumps({'physical_field_parent_id':identifier,
            'parent_source_field':parent['source_field'],'parent_source_field_sha256':parent['source_field_sha256'],
            'parent_h48_array_sha256':parent['array_sha256']['48'],'high_mode_seed':high_seed,
            'field_metadata':extended.metadata}))
    with np.load(path)as archive:
        h64=archive['h64'];extension_header=json.loads(str(archive['header']))
    if(extension_header['physical_field_parent_id']!=identifier
       or extension_header['high_mode_seed']!=high_seed
       or extension_header['parent_source_field_sha256']!=parent['source_field_sha256']
       or extension_header['parent_h48_array_sha256']!=parent['array_sha256']['48']):
        raise ValueError('Saved N64proposal identity differs from its reserved seed/parent')
    k=np.rint(np.fft.fftfreq(48)*48).astype(int);active=spectrum>0
    i,j,ell=np.nonzero(active);dest=(k[i]%64,k[j]%64,k[ell]%64)
    old_coefficients=np.fft.fftn(h48)/48**3;new_coefficients=np.fft.fftn(h64)/64**3
    error=float(abs(new_coefficients[dest]-old_coefficients[active]).max())
    if error>5e-14:raise ValueError('N64doesnotpreserve actualh48Fouriercoefficients')
    actual={**parent,'source_field':str(path),'source_field_sha256':sha256_file(path),
        'array_sha256':{'64':hashlib.sha256(h64.tobytes()).hexdigest()},
        'parent_source_field':parent['source_field'],'parent_source_field_sha256':parent['source_field_sha256'],
        'parent_array_sha256':parent['array_sha256'],'parent_h48_array_sha256':parent['array_sha256']['48'],
        'physical_field_parent_id':identifier,'high_mode_seed':high_seed,
        'coupled_resolution_scheme':{'base_joint_fft_sizes':[32,48],'extension_size':64,
            'method':'preserveactualh48activecoefficients;independentnewhighmodes'},
        'shared_Fourier_coefficient_max_abs_error':error,
        'coupling':'retainactiveh48modes;independenthighmodes;notanewphysicalparent'}
    write_json_new(sidecar,actual);return actual


def forecast(config):
    """Saved deterministic timings only;never draw a field or run a metric."""
    benchmark=json.loads(Path('results/phase3c/fields/periodic_kernel_validation/benchmark.json').read_text())['rows']
    rows=[]
    for model,xi,conversion,probes in tasks(config,['log','smooth','flat']):
        parents=len(config.get('smooth_field_ids',[]))if model=='smooth'else 24
        unique=len({g for _,g in probes})
        for n in[32,48]:
            stencil=config.get('stencils_by_n',{}).get(str(n),config.get('stencil','primitive2'))
            measured=next(r for r in benchmark if r['n']==n and r['stencil']==stencil)
            build=measured['warm_kernel_build_seconds']if config.get('builder','periodic_kernel')=='periodic_kernel'else measured['reference_build_seconds']
            per_record=1.3*(build+measured['three_root_dijkstra_seconds']/3*16*unique+(.08 if n==32 else .25)*unique)
            rows.append({'model':model,'xi':xi,'n':n,'parents':parents,'unique_gamma_queries':unique,
                'seconds_per_record_forecast':per_record,'total_seconds_forecast':parents*per_record})
    n64_forecast=None
    if config.get('enable_n64',False):
        pilot=json.loads(Path(config['n64_deterministic_pilot']).read_text())
        n64_forecast=0.
        for xi in config['n64_xis']:
            seconds=pilot['seconds']*1.5
            total=len(config['n64_field_ids'])*seconds;n64_forecast+=total
            rows.append({'model':'log','xi':xi,'n':64,'parents':len(config['n64_field_ids']),
                'unique_gamma_queries':len(config.get('n64_gamma_bands',['central'])),
                'seconds_per_record_forecast':seconds,'total_seconds_forecast':total,
                'note':'conservative full3-gamma deterministicpilot cost evenifcentralonly'})
    return {'rows':rows,'total_seconds_forecast':sum(r['total_seconds_forecast']for r in rows),
        'optional_n64_seconds_forecast':n64_forecast,
        'estimated_peak_memory_bytes':1_500_000_000 if config.get('enable_n64',False) else 750_000_000,'method':'measured warmbuilder plus3-root Dijkstra extrapolation, CDF allowance,and1.3timing factor;replace with frozen countingpilot',
        'new_fields_or_metrics_generated':False}


def estimate(args):
    config=frozen_config(args.config);result=forecast(config)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    write_json_new(root/'resource_estimate.json',{'configuration':config,'testing_plan_sha256':sha256_file(args.config),**result})
    print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))


def run(args):
    config=frozen_config(args.config)
    if not args.field_ids or len(args.field_ids)>6:raise ValueError('Use an explicit batch of at most6held-out families')
    if any(i not in range(1000,1024)for i in args.field_ids):raise ValueError('Confirmation IDs are1000..1023')
    if not set(args.sizes).issubset([32,48,64]):raise ValueError('Supported confirmation grids are32,48andoptional64')
    if 64 in args.sizes and not config.get('enable_n64',False):raise ValueError('N64isnotinthe frozen plan')
    if args.sizes==[64]and any(i not in config['n64_field_ids']for i in args.field_ids):
        raise ValueError('N64-onlybatchmustusethepredeclaredfirst8parentIDs')
    specs=tasks(config,args.models,args.xis)
    if not specs:raise ValueError('No frozen task selected')
    eligible_n64=[xi for model,xi,_,_ in specs if model=='log'and xi in(config.get('n64_xis')or[])]
    if 64 in args.sizes and not eligible_n64:raise ValueError('N64requestmustinclude predeclaredlogxi')
    registry=plan(args.registry,args.seed);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();calibration=Path(config.get('calibration',DEFAULT_CALIBRATION))
    metadata={'configuration':config,'testing_plan_sha256':sha256_file(args.config),
        'registry_sha256':sha256_file(args.registry),'source_hashes':hashes,
        'calibration_sha256':sha256_file(calibration),'bootstrap_sha256':sha256_file(calibration.parent/'bootstrap.npz')}
    if(root/'metadata.json').exists():
        if json.loads((root/'metadata.json').read_text())!=metadata:raise ValueError('Frozen confirmation setting/source changed')
    else:
        snapshot=root/'source_snapshot';snapshot.mkdir()
        for p in hashes:shutil.copyfile(p,snapshot/Path(p).name)
        write_json_new(root/'metadata.json',metadata);write_json_new(root/'environment.json',environment_manifest())
    for p,expected in registry['prior_identities']['source_sha256'].items():
        if sha256_file(p)!=expected:raise ValueError('Prior-data independence reference changed')
    plans=root/'batch_plans';plans.mkdir(exist_ok=True)
    batch={'field_ids':args.field_ids,'sizes':args.sizes,'models':args.models,'xi_subset':args.xis,
        'testing_plan_sha256':sha256_file(args.config),'source_hashes':hashes,
        'question':'Independent frozen family prediction andnegative controls;noholdouttuning',
        'resource_policy':'at most6families;prefer separate model/grid invocations;checkpoint each field/model/grid/xi',
        'estimated_memory_bytes':1_500_000_000 if 64 in args.sizes else 750_000_000,
        'estimated_seconds':sum(r['seconds_per_record_forecast']*sum(i in config.get('smooth_field_ids',[])if r['model']=='smooth'else True for i in args.field_ids)
            for r in forecast(config)['rows']if r['model']in args.models and r['n']in args.sizes and(args.xis is None or r['xi']in args.xis))}
    plan_path=plans/f'{args.label}.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=batch:raise ValueError('Existing confirmation batch differs')
    else:write_json_new(plan_path,batch)
    started=time.perf_counter();records=[]
    by_id={r['realization']:r for r in registry['records']}
    for identifier in args.field_ids:
        if not any(model!='smooth'or identifier in config.get('smooth_field_ids',[])for model,_,_,_ in specs):continue
        row=materialize(by_id[identifier],registry['prior_identities'])
        uniforms=np.random.default_rng(row['root_seed']).random(config.get('points',16))
        with np.load(row['source_field'])as archive:fields={n:archive[f'h{n}']for n in args.sizes if n in[32,48]}
        extended_row=None
        if 64 in args.sizes and identifier in config['n64_field_ids']:
            if 48 not in args.sizes:
                for xi in eligible_n64:
                    paired=root/'log'/f'xi{xi:g}'/f'metric_{identifier:04d}_n48.npz'
                    if not paired.exists():raise ValueError('GeneratepairedN48onthissameparentbeforeN64')
            extended_row=materialize_n64(row,registry)
            with np.load(extended_row['source_field'])as archive:fields[64]=archive['h64']
        for n,h in fields.items():
            for model,xi,conversion,probes in specs:
                if model=='smooth'and identifier not in config.get('smooth_field_ids',[]):continue
                if n==64:
                    if model!='log'or xi not in config['n64_xis']:continue
                    paired=root/'log'/f'xi{xi:g}'/f'metric_{identifier:04d}_n48.npz'
                    if not paired.exists():raise ValueError('PairedN48metriconthesamephysicalparentmustexistbeforeN64')
                    with np.load(paired)as archive:paired_header=json.loads(str(archive['header']))
                    if(paired_header['source_field_sha256']!=row['source_field_sha256']
                       or paired_header['testing_plan_sha256']!=sha256_file(args.config)):
                        raise ValueError('N64requires the sameparentN48andfrozen testingplan')
                    allowed=config.get('n64_gamma_bands',['central'])
                    probes=[p for p in gamma_probes(conversion)if p[0]in allowed]
                actual=extended_row if n==64 else row
                folder=root/model/f'xi{xi:g}';folder.mkdir(parents=True,exist_ok=True)
                destination=folder/f'metric_{identifier:04d}_n{n}.npz'
                if destination.exists():
                    with np.load(destination)as archive:
                        old=json.loads(str(archive['header']))
                        if old['source_field_sha256']!=actual['source_field_sha256']:raise ValueError('Checkpoint identity changed')
                    continue
                tick=time.perf_counter()
                transformed,variance,transform=control_field(h,model,smooth_radius=config.get('smooth_radius',2.))
                stencil=config.get('stencils_by_n',{}).get(str(n),config.get('stencil','primitive2'))
                arrays,masses,metric_meta,counts=metric_arrays(transformed,xi,variance,probes,uniforms,
                    stencil=stencil,bins=config.get('bins',1024),builder=config.get('builder','periodic_kernel'))
                header={**actual,'physical_field_parent_id':identifier,
                    'parent_source_field':row['source_field'],'parent_source_field_sha256':row['source_field_sha256'],
                    'parent_array_sha256':row['array_sha256'],
                    'n':n,'sampling_n':n,'xi':xi,'probes':masses,'conversion':conversion,
                    'points':config.get('points',16),'stencil':stencil,'boundary':'torus',
                    'field_variance':variance,'control':{'log':'calibrated_log_field','smooth':'smooth_nonlog','flat':'flat'}[model],
                    'field_transform':transform,'transformed_field_sha256':hashlib.sha256(transformed.tobytes()).hexdigest(),
                    'metric_metadata':metric_meta,'gamma_rule':'modelwise_lambda','independent_holdout':True,
                    'testing_plan_sha256':sha256_file(args.config),'seconds':time.perf_counter()-tick,
                    'source_metric':None,'gauges':'grounded rawmetric;unit_mass derived later on identical records',
                    'primary_probe_labels':[band for band,_ in probes if not band.startswith('model_')],
                    'diagnostic_probe_labels':[band for band,_ in probes if band.startswith('model_')],
                    'model_gammas':conversion['model_gammas'],'invalid_model_probes':conversion['invalid_models'],
                    'probe_models':{band:[name for name,g in conversion['model_gammas'].items()if g==gamma]
                                    for band,gamma in probes},
                    'global_calibration_models_required':True,
                    'sensitivity_interpretation':'robustmax tests uniform compatibility acrossqueried settings only;failure does notruleout central or another single global calibration curve',**counts}
                write_npz_new(destination,**arrays,header=json.dumps(header))
                records.append({'path':str(destination),'seconds':header['seconds']})
                print(json.dumps({'field':identifier,'n':n,'model':model,'xi':xi,'seconds':header['seconds']}),flush=True)
    batches=root/'batches';batches.mkdir(exist_ok=True)
    summary={'field_ids':args.field_ids,'sizes':args.sizes,'models':args.models,
        'seconds':time.perf_counter()-started,'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_records':len(records),'records':records,'no_holdout_parameter_tuning':True}
    write_json_new(batches/f'{args.label}.json',summary)
    print(json.dumps({k:v for k,v in summary.items()if k!='records'}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['plan','estimate','run'])
    parser.add_argument('--registry',default='results/phase3c/fields/confirmation_registry.json')
    parser.add_argument('--seed',type=int,default=202611123900)
    parser.add_argument('--config')
    parser.add_argument('--output',default='results/phase3c/fields/confirmation')
    parser.add_argument('--field-ids',type=int,nargs='+')
    parser.add_argument('--sizes',type=int,nargs='+',default=[32,48])
    parser.add_argument('--models',nargs='+',default=['log','smooth','flat'])
    parser.add_argument('--xis',nargs='+',type=float)
    parser.add_argument('--label')
    args=parser.parse_args()
    if args.command=='plan':
        saved=plan(args.registry,args.seed);print(json.dumps({'planned_fields':len(saved['records']),'fields_generated':0}))
    elif args.command=='estimate':
        if not args.config:parser.error('estimate requires a frozen --config')
        estimate(args)
    elif not args.config or not args.label:parser.error('run requires a frozen --config andunique --label')
    else:run(args)


if __name__=='__main__':main()
