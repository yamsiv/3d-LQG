"""Frozen representative-family refinement with exact-control data reuse.

No old path is regenerated. Exact I3 continuous minima and A3 fine-grid minima
can be grouped into coarser cells; new graph observations then come from the
same physical path, not a new sample. Generic old coarser D6 paths are retained
as sensitivity inputs rather than pooled with the new finest resolution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.phase3a_ensembles import write_npz
from experiments.phase3a_compare import save_json
from src.mated_crt import graph_from_minima,validate_graph
from src.graph_metrics import bfs_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest

ROOT=Path('results/phase3c/families/refinement')
DESIGN=Path('results/phase3c/adaptive/design.json')
REPRESENTATIVES=['C006','C036','C008','C043','C011','C000','C047','C049','C119','C110','C002','C012']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():raise FileExistsError(ROOT/'plan.json')
    specs=json.loads(DESIGN.read_text())['configurations'];byid={s['id']:s for s in specs}
    nonref=[i for i in REPRESENTATIVES if i not in ['C002','C012']]
    plan=dict(configurations=[byid[i] for i in REPRESENTATIVES],
        roles={'analytic_birectangular_sequence':['C006','C036','C008','C043','C011'],
               'negative_equicorrelation_sequence':['C000','C047','C049'],
               'asymmetric_and_new_coverage':['C119','C110'],
               'exact_or_exact_grid_anchors':['C002','C012']},
        purpose='Test extended observed regions and connectivity gaps across structured families and asymmetric representatives; positive birectangular endpoint is a deliberate poor-fit comparator, not a secretly replaced winner',
        target_parents_per_covariance=24,paired_sizes=[2048,8192],points=16,
        general_settings=dict(degree=6,base_step=.0025,relative_step=.05,exit_order=12,initial_candidates=2048,origin_cutoff=.001),
        matrix_reference_q=128,
        reuse={'C002':'8Phase3Bintermediate exact parents +16Phase3Bholdout exact parents; coarser observations derived from exact fine-cell minima when missing',
               'C012':'16Phase3Bholdout q128 parents; coarser observations derived from fine polygonal minima'},
        old_finer8_parent_C000_C006='kept separately: their .005/.1 path setting is not pooled into the new .0025/.05 setting',
        new_main_paths_requested=len(nonref)*24+8,
        additional_provisional_control_paths=24,
        seeds=dict(main=202611123700,provisional_anchors=202611123710),
        controls='Eight provisional finest paths each forI3/A2/A3, compared with existing independent references; not added as exact-law parent replicas',
        runtime_seconds_estimate=[500,1000],memory_mb_estimate=500,
        batch_policy='at most two24-parentCs per invocation; every physical path/n and failure checkpointed',
        selection_scope='Discovery-stage representatives frozen before refined observations; no independent confirmation data examined',
        provenance=environment_manifest())
    save_json(ROOT/'plan.json',plan)
    print(json.dumps({'representatives':len(REPRESENTATIVES),'new_paths':plan['new_main_paths_requested'],
                      'provisional_control_paths':24,'no_C030_concentration':True}))


def reuse_controls():
    manifest=[]
    for identifier in ['C002','C012']:
        sources=[]
        if identifier=='C002':sources+=sorted(Path('results/phase3b/intermediate_graphs/C002').glob('graph_*_n8192.npz'))
        sources+=sorted(Path(f'results/phase3b/holdout_graphs/{identifier}').glob('graph_*_n8192.npz'))
        if len(sources)!=(24 if identifier=='C002' else 16):raise ValueError('Control cohort incomplete')
        target=ROOT/'reused_graphs'/identifier;target.mkdir(parents=True,exist_ok=True)
        for index,source in enumerate(sources):
            with np.load(source) as z:
                h=json.loads(str(z['header']));fine=z['minima'].copy();points=z['points'].copy()
            destination=target/f'graph_{index:03d}_n8192.npz'
            if not destination.exists():shutil.copyfile(source,destination)
            if digest(destination)!=digest(source):raise ValueError('Exact control copy changed')
            coarse_destination=target/f'graph_{index:03d}_n2048.npz'
            old_coarse=source.with_name(source.name.replace('n8192','n2048'))
            if not coarse_destination.exists():
                if old_coarse.exists():shutil.copyfile(old_coarse,coarse_destination)
                else:
                    start=time.perf_counter();minima=fine.reshape(2048,4,3).min(axis=1)
                    graph=graph_from_minima(minima);csr=graph.to_sparse();roots=points//4
                    d=np.array([bfs_distances(csr,int(root)) for root in roots])
                    data=weighted_root_cdfs(d,np.ones(2048)/2048,discrete=True)
                    header=copy.deepcopy(h);header.update(n=2048,seconds=time.perf_counter()-start,
                        source_checkpoint=str(source),source_sha256=digest(source),
                        source_simulation_seconds=h['seconds'],
                        derivation='min over four consecutive fine-cell minima; roots=floor(oldroot/4), preserving common iid uniforms',
                        effective_reference_oversample_at_n=512 if identifier=='C012' else None,
                        validation=validate_graph(graph))
                    write_npz(coarse_destination,**data,minima=minima,points=roots,matrix=d[:,roots],header=json.dumps(header))
            manifest.append(dict(id=identifier,seed=h['seed'],source=str(source),source_sha256=digest(source),
                                 destinations=[str(destination),str(coarse_destination)],
                                 independent_new_paths=0,paired_observations=True))
    if not (ROOT/'reused_controls.json').exists():save_json(ROOT/'reused_controls.json',manifest)
    print(json.dumps({'reused_physical_controls':len(manifest),'new_Brownian_paths':0}))


def run(first,last,controls=False):
    plan=json.loads((ROOT/'plan.json').read_text())
    if controls:
        ids=['C002','C007','C012'];samples=8;folder=ROOT/'provisional_anchor_controls';seed=plan['seeds']['provisional_anchors']
    else:
        ids=[s['id'] for s in plan['configurations'] if s['id'] not in ['C002','C012']][first:last]
        samples=24;folder=ROOT/'new_graphs'/f'batch_{first:02d}_{last:02d}';seed=plan['seeds']['main']
    if not ids:return
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():print(json.dumps({'already_complete':str(folder)}));return
    command=[sys.executable,'-m','experiments.phase3b_graphs','--design',str(DESIGN),'--output',str(folder),'--ids',*ids,
        '--samples',str(samples),'--sizes','2048','8192','--points','16','--seed',str(seed),
        '--degree','6','--step','.0025','--relative','.05','--exit-order','12','--initial','2048','--cutoff','.001','--q','128']
    if controls:command+=['--force-provisional']
    if (folder/'metadata.json').exists():command+=['--resume']
    batch=dict(ids=ids,parents=samples,scientific_question=plan['purpose'],command=command,
               expected_seconds=[len(ids)*samples*2,len(ids)*samples*5],working_memory_mb_estimate=500)
    if not (folder/'batch_plan.json').exists():save_json(folder/'batch_plan.json',batch)
    start=time.perf_counter()
    with (folder/'producer.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(ids=ids,seconds=time.perf_counter()-start,n8192_records=len(list(folder.glob('C*/graph_*_n8192.npz'))),
        expected_parents=len(ids)*samples,model_failures=[str(p) for p in folder.glob('C*/model_failure.json')],
        path_failures=[str(p) for p in folder.glob('C*/failure_*.json')])
    save_json(folder/'COMPLETE.json',result);print(json.dumps(result),flush=True)


def new_a3():
    folder=ROOT/'new_a3';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():return
    command=[sys.executable,'-m','experiments.phase3b_graphs','--design',str(DESIGN),'--output',str(folder),'--ids','C012',
             '--samples','8','--sizes','2048','8192','--points','16','--seed','202611123700','--q','128']
    if (folder/'metadata.json').exists():command+=['--resume']
    start=time.perf_counter()
    with (folder/'producer.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    save_json(folder/'COMPLETE.json',dict(id='C012',parents=8,seconds=time.perf_counter()-start,law='exactfinitegridbiasedreference,q128atn8192;polygonalminima',
        expected_seconds=[12,30],working_memory_mb_estimate=500))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','reuse','run','controls','a3'])
    p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int,default=2)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='reuse':reuse_controls()
    elif a.stage=='a3':new_a3()
    else:run(a.first,a.last,a.stage=='controls')
