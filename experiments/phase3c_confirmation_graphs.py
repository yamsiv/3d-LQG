"""Frozen confirmation graphs; exact controls and explicit provisional paths.

The algorithm is the preserved graph/excursion pipeline. New metadata retain
the paper's radial bias functional for future ensemble checks; it is NOT used
to reweight the primary Phase3C comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from scipy.special import gammaln,logsumexp

from experiments.phase3a_ensembles import seed_for,write_npz
from experiments.phase3a_compare import save_json
from experiments.phase3b_graphs import anchor_case
from src.brownian_excursions import sample_excursion
from src.weyl_excursions import sample_weyl_excursion
from src.weyl_a3 import sample_a3_excursion
from src.provisional_excursions_3b import make_provisional_model,sample_provisional_excursion
from src.mated_crt import graph_from_excursion,validate_graph,cell_minima
from src.graph_metrics import bfs_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest


def radial_bias_integral(times,values,covariance,alpha1):
    """Log trapezoidal integral of whitened radius^(2alpha1-2).

    This is a finite stopping/grid-path quadrature, not an exact functional of
    the unsampled Brownian path. C_alpha=2^(alpha-1)*Gamma(alpha)/alpha is the
    original-excursion expectation from Budd–Castro §3.3.
    """
    t=np.asarray(times,float);x=np.asarray(values,float);c=np.asarray(covariance,float)
    if len(t)!=len(x) or x.shape[1:]!=(3,) or alpha1<=1 or np.any(np.diff(t)<0):raise ValueError('Invalid radial functional input')
    dt=np.diff(t);weights=np.r_[dt,0.]/2+np.r_[0.,dt]/2
    inverse=np.linalg.inv(c);total=-np.inf
    for start in range(0,len(t),65536):
        xx=x[start:start+65536];ww=weights[start:start+65536]
        radius2=np.einsum('ni,ij,nj->n',xx,inverse,xx)
        good=(radius2>0)&(ww>0)
        if good.any():total=float(np.logaddexp(total,logsumexp(np.log(ww[good])+(alpha1-1)*np.log(radius2[good]))))
    if not np.isfinite(total):raise ValueError('Nonfinite radial functional')
    return dict(log_integral=total,log_original_expectation=float((alpha1-1)*np.log(2)+gammaln(alpha1)-np.log(alpha1)),
        exponent=2*alpha1-2,continuous_integral_exact=False,
        quadrature='trapezoid over all saved sample/stopping values in unit-duration whitened coordinates',
        scope='metadata only; primary law remains biased; no unbiasing or LFPP ensemble tilt applied')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ids',nargs='+',required=True);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if not a.pilot and not plan.get('frozen_before_confirmation',False):raise ValueError('Frozen hypotheses required')
    seed_base=plan['graph_seed_base'];samples=1 if a.pilot else plan['graph_parents']
    if a.pilot and seed_base>=202611123900:raise ValueError('Cost pilot cannot consume heldout seeds')
    if not a.pilot and seed_base<202611123900:raise ValueError('Independent confirmation requires reserved seed family')
    design=json.loads(Path(plan['design']).read_text())['configurations'];specs={s['id']:s for s in design}
    config={'plan':plan,'ids':a.ids,'pilot':a.pilot,'target_law':'paper_biased','algorithm':'existing sampler/graph implementation; extra metadata only'}
    meta=root/'metadata.json'
    if meta.exists():
        if json.loads(meta.read_text())['configuration']!=config:raise ValueError('Resume configuration differs')
    else:save_json(meta,dict(configuration=config,provenance=environment_manifest()))
    for identifier in a.ids:
        if identifier not in plan['graph_ids']:raise ValueError('Covariance not frozen in plan')
        spec=specs[identifier];folder=root/identifier;folder.mkdir(exist_ok=True);kind=anchor_case(spec['triple'])
        model=None
        if kind is None:
            try:model=make_provisional_model(np.asarray(spec['covariance']),degree=plan['ritz_degree'],initial_candidates=plan['initial_candidates'])
            except Exception as error:
                if not (folder/'model_failure.json').exists():save_json(folder/'model_failure.json',dict(spec=spec,error=repr(error),traceback=traceback.format_exc()))
                continue
        for i in range(samples):
            paths=[folder/f'graph_{i:03d}_n{n}.npz' for n in plan['graph_sizes']];failure=folder/f'failure_{i:03d}.json'
            if all(x.exists() for x in paths) or failure.exists():continue
            start=time.perf_counter();seed=seed_for(seed_base,int(identifier[1:]),i,11)
            try:
                rng=np.random.default_rng(seed);n=max(plan['graph_sizes'])
                if kind=='i3':sample=sample_excursion(n,np.eye(3),rng,law='biased')
                elif kind=='a2':sample=sample_weyl_excursion(n,rng=rng,law='biased',oversample=plan['reference_q'])
                elif kind=='a3':sample=sample_a3_excursion(n,rng=rng,law='biased',oversample=plan['reference_q'])
                else:
                    step=plan['base_step']*np.sqrt(2.5/(model.alpha1-1))
                    sample=sample_provisional_excursion(n,rng=rng,model=model,step_size=step,
                        relative_step=plan['relative_step'],origin_cutoff=.001,exit_order=plan['exit_order'],audit_order=24)
                covariance=sample.metadata.get('covariance',spec['covariance'])
                sample.metadata['radial_bias_integral']=radial_bias_integral(sample.times,sample.values,covariance,sample.metadata['alpha1'])
                uniforms=np.random.default_rng(seed_for(seed_base,int(identifier[1:]),i,12)).random(plan['points'])
                for nn,path in zip(plan['graph_sizes'],paths):
                    if path.exists():continue
                    graph=graph_from_excursion(sample,nn);csr=graph.to_sparse();points=(uniforms*nn).astype(int)
                    d=np.array([bfs_distances(csr,int(v)) for v in points]);data=weighted_root_cdfs(d,np.ones(nn)/nn,discrete=True)
                    h=dict(id=identifier,spec=spec,realization=i,seed=seed,n=nn,physical_path_id=f'{identifier}_{seed}',
                        points=plan['points'],sampler_status='provisional' if model else 'exact_grid_or_minimum_reference',
                        sampler_metadata=sample.metadata,validation=validate_graph(graph),configuration=config,
                        seconds=time.perf_counter()-start,retained_points=len(sample.times))
                    write_npz(path,**data,minima=cell_minima(sample,nn),points=points,matrix=d[:,points],header=json.dumps(h))
                if i==0 and not (folder/'first_path.npz').exists():
                    write_npz(folder/'first_path.npz',times=sample.times,values=sample.values,metadata=json.dumps(sample.metadata))
                print(json.dumps({'id':identifier,'parent':i,'seconds':time.perf_counter()-start,'retained_points':len(sample.times)}),flush=True)
            except Exception as error:
                save_json(failure,dict(id=identifier,seed=seed,realization=i,error=repr(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
                print(json.dumps({'id':identifier,'failure':repr(error)}),flush=True)
    done=root/'COMPLETE.json'
    if not done.exists():save_json(done,dict(ids=a.ids,parents_requested=samples,
        graph_records=len(list(root.glob('C*/graph_*_n*.npz'))),failures=[str(p) for p in root.glob('C*/*failure*.json')],
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        pilot=a.pilot,complete_attempts=True))


if __name__=='__main__':main()
