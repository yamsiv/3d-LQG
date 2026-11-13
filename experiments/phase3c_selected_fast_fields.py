"""Existing48-field selected refinement with the validated periodic builder.

This separate producer preserves the original100-field prototype unchanged.
It cannot generate Gaussian fields or use planned discovery IDs48..99.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from experiments.phase3c_selected_fields import modelwise_conversion,cached_probes,read_frozen_config
from src.field_metrics_3c import metric_arrays
from src.gmc_3d import gmc_weights
from src.log_field_3d import spectral_configuration
from src.phase3c_fields import sha256_file,write_json_new,write_npz_new,gamma_probes,sampled_points
from src.provenance import environment_manifest


def run(args):
    config=read_frozen_config(args.config)
    if not args.field_ids or len(args.field_ids)>12 or any(i not in range(48)for i in args.field_ids):
        raise ValueError('Use a batch of at most12existing IDs0..47;new Gaussian fields are forbidden')
    if not set(args.sizes).issubset(config.get('field_sizes',[32,48])):raise ValueError('Grid not in frozen plan')
    xis=config['xis']if args.xis is None else args.xis
    if not set(xis).issubset(config['xis']):raise ValueError('Xi not in frozen plan')
    registry_path=Path(config['registry']);registry=json.loads(registry_path.read_text())['records']
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    paths=['experiments/phase3c_selected_fast_fields.py','experiments/phase3c_selected_fields.py',
        'src/field_metrics_3c.py','src/lfpp_periodic_3c.py','src/lfpp_3d.py','src/phase3c_fields.py',
        'src/log_field_3d.py','src/gmc_3d.py','src/metric_measure_3a.py','src/parameter_conversion_3b.py']
    hashes={p:sha256_file(p)for p in paths}
    calibration=Path(config['calibration'])
    metadata={'configuration':config,'source_hashes':hashes,'registry_sha256':sha256_file(registry_path),
        'testing_plan_sha256':sha256_file(args.config),'calibration_sha256':sha256_file(calibration),
        'bootstrap_sha256':sha256_file(calibration.parent/'bootstrap.npz')}
    if(root/'metadata.json').exists():
        if json.loads((root/'metadata.json').read_text())!=metadata:raise ValueError('Frozen selected source/settings changed')
    else:
        snapshot=root/'source_snapshot';snapshot.mkdir()
        for p in hashes:shutil.copyfile(p,snapshot/Path(p).name)
        write_json_new(root/'metadata.json',metadata);write_json_new(root/'environment.json',environment_manifest())
    plan={'field_ids':args.field_ids,'sizes':args.sizes,'xis':xis,'source_hashes':hashes,
        'scientific_question':config.get('question'),'new_Gaussian_fields':0,
        'resource_estimate_seconds':len(args.field_ids)*len(xis)*sum(.8 if n==32 else 2.2 for n in args.sizes),
        'estimated_peak_memory_bytes':750_000_000,'cache':'per-probe exact gamma/points/source/convention matches'}
    plans=root/'batch_plans';plans.mkdir(exist_ok=True);plan_path=plans/f'{args.label}.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing batch plan differs')
    else:write_json_new(plan_path,plan)
    conversions={x:modelwise_conversion(x,config['calibration'])for x in xis}
    started=time.perf_counter();records=[]
    for identifier in args.field_ids:
        row=dict(registry[identifier])
        if not Path(row['source_field']).exists()or sha256_file(row['source_field'])!=row['source_field_sha256']:
            raise ValueError('Existing Gaussian field changed or missing')
        uniforms=np.random.default_rng(row['root_seed']).random(16)
        with np.load(row['source_field'])as archive:fields={n:archive[f'h{n}']for n in args.sizes}
        for n,h in fields.items():
            _,_,variance=spectral_configuration(n)
            for xi,conversion in conversions.items():
                folder=root/f'xi{xi:g}';folder.mkdir(exist_ok=True)
                destination=folder/f'metric_{identifier:03d}_n{n}.npz'
                if destination.exists():continue
                tick=time.perf_counter();probes=gamma_probes(conversion)
                measures={g:gmc_weights(h,g,variance)for _,g in probes}
                points=[sampled_points(measures[g].probabilities,uniforms)for _,g in probes]
                cache=cached_probes(row,n,xi,config['stencil'],probes,points)
                arrays,masses,metric_meta,counts=metric_arrays(h,xi,variance,probes,uniforms,
                    stencil=config['stencil'],bins=config.get('bins',1024),builder='periodic_kernel',cached=cache)
                header={**row,'n':n,'sampling_n':n,'xi':xi,'probes':masses,'conversion':conversion,
                    'points':16,'stencil':config['stencil'],'boundary':'torus','field_variance':variance,
                    'metric_metadata':metric_meta,'control':'calibrated_log_field','gamma_rule':'modelwise_lambda',
                    'source_metric':None,'seconds':time.perf_counter()-tick,
                    'testing_plan_sha256':sha256_file(args.config),'new_Gaussian_fields':False,**counts}
                write_npz_new(destination,**arrays,header=json.dumps(header))
                record={'field':identifier,'n':n,'xi':xi,'seconds':header['seconds'],**counts}
                records.append(record);print(json.dumps(record),flush=True)
    batches=root/'batches';batches.mkdir(exist_ok=True)
    summary={'field_ids':args.field_ids,'sizes':args.sizes,'xis':xis,'new_records':len(records),
        'seconds':time.perf_counter()-started,'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_Gaussian_fields':0,'records':records}
    write_json_new(batches/f'{args.label}.json',summary)
    print(json.dumps({k:v for k,v in summary.items()if k!='records'}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--field-ids',type=int,nargs='+',required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[32,48])
    parser.add_argument('--xis',type=float,nargs='+');parser.add_argument('--label',required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
