"""LFPP/GMC along the calibrated family, reusing cached Gaussian fields.

Field randomness is shared across xi and uncertainty probes. The grounded
and unit-mass gauges are derived from the same records, never fit-selected.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.parameter_conversion_3b import load_calibration
from src.log_field_3d import sample_coupled_fields,spectral_configuration
from src.gmc_3d import gmc_weights
from src.lfpp_3d import build_lfpp_graph,root_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest
from experiments.phase3a_ensembles import seed_for,write_npz


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--calibration',required=True)
    p.add_argument('--xis',nargs='+',type=float,required=True);p.add_argument('--sizes',nargs='+',type=int,default=[32])
    p.add_argument('--samples',type=int,default=12);p.add_argument('--points',type=int,default=16)
    p.add_argument('--stencil',default='26');p.add_argument('--bands',nargs='+',default=['central'])
    p.add_argument('--source',choices=['phase3a','new'],default='phase3a');p.add_argument('--source-start',type=int,default=0)
    p.add_argument('--seed',type=int,default=202611113701);p.add_argument('--smooth',action='store_true')
    p.add_argument('--cache-distance-fields',type=int,default=2);p.add_argument('--resume',action='store_true')
    p.add_argument('--cached-field-directory')
    p.add_argument('--boundary',choices=['torus','cube'],default='torus')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    calibration=load_calibration(a.calibration);calibration_hash=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest()
    config={k:v for k,v in vars(a).items() if k!='resume'};meta=root/'metadata.json'
    if meta.exists():
        if not a.resume:raise FileExistsError(meta)
        old=json.loads(meta.read_text())
        if old['configuration']!=config or old['calibration_sha256']!=calibration_hash:raise ValueError('Resume config/calibration differs')
    else:
        meta.write_text(json.dumps({'configuration':config,'calibration_sha256':calibration_hash,
            'provenance':environment_manifest(),'primary_family':'gamma(xi) from finite crossing calibration; provisional systematic bands, not independent optimization',
            'gauges':'grounded primary; unit_mass Weyl sensitivity on identical candidates',
            'source':'cached Phase3A common field arrays' if a.source=='phase3a' else 'new independent held-out Gaussian fields',
            'points':'common iid uniforms across xi/gamma, inverse-CDF coupling not a spatial coupling'},indent=2)+'\n')
    parameters=[]
    for xi in a.xis:
        conversion=calibration.evaluate(xi)
        if not conversion['valid']:
            path=root/f'invalid_xi{xi:g}.json'
            if not path.exists():path.write_text(json.dumps(conversion,indent=2)+'\n')
            continue
        gamma=conversion['gamma'];probes=[('central',gamma)]
        if xi>0 and 'low'in a.bands:
            low=conversion.get('gamma_systematic',[None,None])[0]
            if low is not None and 0<low<np.sqrt(6):probes.append(('low',low))
        if xi>0 and 'high'in a.bands:
            high=conversion.get('gamma_systematic',[None,None])[1]
            if high is not None and 0<high<np.sqrt(6):probes.append(('high',high))
        parameters.append((xi,conversion,probes))
    for realization in range(a.samples):
        field_id=realization+a.source_start
        if a.source=='phase3a':
            source=Path(f'results/phase3a/confirmation/field_{field_id:03d}.npz')
            with np.load(source) as archive:
                fields={n:archive[f'h{n}'] for n in a.sizes}
            seed=None
        else:
            seed=seed_for(a.seed,field_id,1)
            source=Path(a.cached_field_directory)/f'field_{field_id:03d}.npz' if a.cached_field_directory else root/f'field_{field_id:03d}.npz'
            if source.exists():
                with np.load(source) as archive:fields={n:archive[f'h{n}']for n in a.sizes}
            else:
                if a.cached_field_directory:raise FileNotFoundError(source)
                draws=sample_coupled_fields(a.sizes,np.random.default_rng(seed))
                fields={n:f.h for n,f in draws.items()}
                write_npz(source,**{f'h{n}':h for n,h in fields.items()},seed=np.uint64(seed))
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        uniforms=np.random.default_rng(seed_for(a.seed,field_id,2)).random(a.points)
        for n,h in fields.items():
            spectrum,_,variance=spectral_configuration(n)
            if a.smooth:
                k=np.fft.fftfreq(n)*n
                mask=sum(x*x for x in np.meshgrid(k,k,k,indexing='ij'))<=4.
                low_variance=float(spectrum[mask].sum())
                h=np.fft.ifftn(np.fft.fftn(h)*mask).real*np.sqrt(variance/low_variance)
            for xi,conversion,probes in parameters:
                folder=root/f'xi{xi:g}';folder.mkdir(exist_ok=True)
                path=folder/f'metric_{field_id:03d}_n{n}.npz'
                if path.exists():continue
                started=time.perf_counter()
                measures=[gmc_weights(h,g,variance)for _,g in probes]
                points=[np.searchsorted(np.cumsum(m.probabilities.ravel()),uniforms,side='right') for m in measures]
                selected=np.unique(np.concatenate(points))
                graph,metric_meta=build_lfpp_graph(h,xi,stencil=a.stencil,boundary=a.boundary)
                distances=root_distances(graph,selected)
                rows=[];masses=[]
                for (band,gamma),measure,roots in zip(probes,measures,points):
                    d=distances[np.searchsorted(selected,roots)]
                    rows.append(dict(weighted_root_cdfs(d,measure.probabilities.ravel()),matrix=d[:,roots],points=roots))
                    masses.append({'gamma':gamma,'band':band,'log_total_mass':measure.log_total_mass,
                        'total_mass':measure.total_mass,'collision_probability':float(np.sum(measure.probabilities**2))})
                header={'realization':field_id,'field_seed':seed,'source_field':str(source),'source_field_sha256':source_hash,'n':n,'xi':xi,
                    'probes':masses,'conversion':conversion,'stencil':a.stencil,'boundary':a.boundary,'field_variance':variance,
                    'control':'smooth_nonlog' if a.smooth else 'calibrated_log_field','configuration':config,
                    'metric_metadata':metric_meta,'seconds':time.perf_counter()-started}
                arrays=dict(grid=np.stack([r['grid']for r in rows]),cdfs=np.stack([r['cdfs']for r in rows]).astype(np.float32),
                    root_means=np.stack([r['root_means']for r in rows]),matrices=np.stack([r['matrix']for r in rows]),
                    points=np.stack(points),bin_width=np.array([r['distance_bin_width']for r in rows]),header=json.dumps(header))
                if realization<a.cache_distance_fields:arrays.update(distance_roots=selected,distances=distances.astype(np.float32))
                write_npz(path,**arrays)
                print(json.dumps({'field':field_id,'n':n,'xi':xi,'seconds':header['seconds']}),flush=True)


if __name__=='__main__':main()
