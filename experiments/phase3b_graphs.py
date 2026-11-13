"""Checkpointed covariance-screening graphs, with explicit sampler status.

One Brownian realization supplies several paired graph sizes. Failures are
saved, never silently replaced. Existing graph implementations are unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from src.provenance import environment_manifest
from src.metric_measure_3a import weighted_root_cdfs
from experiments.phase3a_ensembles import seed_for,write_npz


def anchor_case(triple):
    x=np.sort(triple)
    for name,ref in [('i3',[0,0,0]),('a2',[-.5,0,0]),('a3',[-.5,-.5,0])]:
        if np.allclose(x,ref,atol=1e-12,rtol=0):return name
    return None


def main():
    from src.provisional_excursions_3b import make_provisional_model,sample_provisional_excursion
    from src.brownian_excursions import sample_excursion
    from src.weyl_excursions import sample_weyl_excursion
    from src.weyl_a3 import sample_a3_excursion
    from src.mated_crt import graph_from_excursion,validate_graph,cell_minima
    from src.graph_metrics import bfs_distances
    p=argparse.ArgumentParser(__doc__);p.add_argument('--design',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ids',nargs='*');p.add_argument('--samples',type=int,default=6)
    p.add_argument('--sizes',nargs='+',type=int,default=[512,2048]);p.add_argument('--points',type=int,default=16)
    p.add_argument('--seed',type=int,default=202611113621);p.add_argument('--degree',type=int,default=4)
    p.add_argument('--step',type=float,default=.01);p.add_argument('--relative',type=float,default=.2)
    p.add_argument('--exit-order',type=int,default=8);p.add_argument('--initial',type=int,default=512)
    p.add_argument('--cutoff',type=float,default=.001);p.add_argument('--q',type=int,default=64)
    p.add_argument('--force-provisional',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k!='resume'}
    meta=root/'metadata.json'
    if meta.exists():
        if not a.resume:raise FileExistsError(meta)
        if json.loads(meta.read_text())['configuration']!=config:raise ValueError('Resume configuration differs')
    else:
        meta.write_text(json.dumps({'configuration':config,'provenance':environment_manifest(),
            'independence':'one physical path per config/seed; sizes paired; no retries',
            'target':'paper-biased excursion; provisional Ritz kernel explicitly approximate'},indent=2)+'\n')
    design=json.loads(Path(a.design).read_text())
    specs=design['configurations'] if isinstance(design,dict) else design
    for spec in specs:
        identifier=spec['id']
        if a.ids and identifier not in a.ids:continue
        folder=root/identifier;folder.mkdir(exist_ok=True)
        anchor=anchor_case(spec['triple'])
        provisional=a.force_provisional or anchor is None
        model=None
        if provisional:
            try:model=make_provisional_model(np.asarray(spec['covariance']),degree=a.degree,initial_candidates=a.initial)
            except Exception as error:
                failure=folder/'model_failure.json'
                if not failure.exists():failure.write_text(json.dumps({'spec':spec,'error':repr(error),'traceback':traceback.format_exc()},indent=2))
                continue
        for i in range(a.samples):
            files=[folder/f'graph_{i:03d}_n{n}.npz' for n in a.sizes]
            failure=folder/f'failure_{i:03d}.json'
            if all(x.exists() for x in files) or failure.exists():continue
            started=time.perf_counter();seed=seed_for(a.seed,int(identifier[1:]),i,11)
            try:
                rng=np.random.default_rng(seed)
                if provisional:
                    effective_step=a.step*np.sqrt(2.5/(model.alpha1-1))
                    sample=sample_provisional_excursion(max(a.sizes),rng=rng,model=model,
                        step_size=effective_step,relative_step=a.relative,origin_cutoff=a.cutoff,
                        exit_order=a.exit_order,audit_order=max(16,2*a.exit_order))
                elif anchor=='i3':sample=sample_excursion(max(a.sizes),np.eye(3),rng,law='biased')
                elif anchor=='a2':sample=sample_weyl_excursion(max(a.sizes),rng=rng,law='biased',oversample=a.q)
                else:sample=sample_a3_excursion(max(a.sizes),rng=rng,law='biased',oversample=a.q)
                # For the exact special controls, coordinate permutations do
                # not change the graph union. Record actual covariance too.
                uniforms=np.random.default_rng(seed_for(a.seed,int(identifier[1:]),i,12)).random(a.points)
                for n,path in zip(a.sizes,files):
                    if path.exists():continue
                    graph=graph_from_excursion(sample,n);csr=graph.to_sparse()
                    points=(uniforms*n).astype(int)
                    distances=np.array([bfs_distances(csr,int(v)) for v in points])
                    data=weighted_root_cdfs(distances,np.ones(n)/n,discrete=True)
                    minima=cell_minima(sample,n)
                    header={'id':identifier,'spec':spec,'realization':i,'seed':seed,'n':n,
                        'physical_path_id':f'{identifier}_{seed}','points':a.points,
                        'sampler_status':'provisional_numerical_Ritz_Doob' if provisional else 'exact_grid_or_exact_minimum_reference',
                        'sampler_metadata':sample.metadata,'validation':validate_graph(graph),
                        'configuration':config,'seconds':time.perf_counter()-started}
                    write_npz(path,**data,matrix=distances[:,points],points=points,
                        minima=minima,header=json.dumps(header))
                if i==0:
                    raw=folder/'first_path.npz'
                    if not raw.exists():write_npz(raw,times=sample.times,values=sample.values,metadata=json.dumps(sample.metadata))
                print(json.dumps({'id':identifier,'realization':i,'seconds':time.perf_counter()-started}),flush=True)
            except Exception as error:
                failure.write_text(json.dumps({'id':identifier,'realization':i,'seed':seed,'error':repr(error),
                    'traceback':traceback.format_exc(),'seconds':time.perf_counter()-started},indent=2)+'\n')
                print(json.dumps({'id':identifier,'failure':repr(error)}),flush=True)


if __name__=='__main__':main()
