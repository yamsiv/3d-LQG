"""Bounded, checkpointed graph batches using the preserved Phase3B producer.

This coordinator does not change the Brownian algorithm. Existing equivalent
six-path cohorts are referenced once and augmented by independent seeds. New
configs receive12 paths. A recorded sampler decision gates the broad campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.provenance import environment_manifest


ROOT=Path('results/phase3c')
DESIGN=ROOT/'covariance_design/initial_design.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    specs=json.loads(DESIGN.read_text())['configurations']
    reuse=['results/phase3b/screen_graphs','results/phase3b/adaptive_coarse_graphs']
    parents={};sources=[]
    for d in reuse:
        for path in sorted(Path(d).glob('C*/graph_*_n2048.npz')):
            with np.load(path) as a:h=json.loads(str(a['header']))
            parents.setdefault(h['id'],set()).add(h['seed'])
            sources.append({'path':str(path),'sha256':sha(path),'id':h['id'],'seed':h['seed']})
    if len(parents)!=31 or any(len(x)!=6 for x in parents.values()):raise ValueError('Historical screen incomplete')
    six=[s['id'] for s in specs if s['id'] in parents]
    twelve=[s['id'] for s in specs if s['id'] not in parents and s['id']!='C022']
    batches=[]
    for samples,identifiers in [(6,six),(12,twelve)]:
        for first in range(0,len(identifiers),6):
            ids=identifiers[first:first+6];label=f'batch_{len(batches):03d}'
            path=ROOT/'screen/batches'/label
            command=[sys.executable,'-m','experiments.phase3b_graphs','--design',str(DESIGN),
                '--output',str(path),'--ids',*ids,'--samples',str(samples),'--sizes','512','2048',
                '--points','16','--seed','202611123200','--degree','4','--step','.01','--relative','.2',
                '--exit-order','8','--initial','512','--cutoff','.001','--q','64']
            batches.append(dict(label=label,ids=ids,parents_per_config=samples,output=str(path),command=command,
                scientific_question='Broad measured response across covariance families at matched biased-law screening settings; not global-minimum refinement',
                expected_seconds=[samples*len(ids)*.8,samples*len(ids)*2.],working_memory_mb_estimate=300,
                checkpoint='one NPZ per physical path and observed n; failures recorded without replacement'))
    data=dict(configuration_count=len(specs),target_parents=12,paired_sizes=[512,2048],points=16,
        reused_directories=reuse,reused_parents={k:len(v) for k,v in parents.items()},
        reused_input_hashes=sources,known_failure={'C022':'Prior identicalD4 positivity-certificate failure; do not rerun identical failed model'},
        batches=batches,provenance=environment_manifest(),
        producer_sha256=sha('experiments/phase3b_graphs.py'),
        initial_design_sha256=sha(DESIGN),
        interpretation='Reuse lower-level cohorts only when sampler options agree; C032 oldD6 not pooled intoD4. Independent sizes/roots are not extra parents.')
    save(ROOT/'screen/campaign.json',data)
    print(json.dumps({'batches':len(batches),'reused_physical_paths':len(sources),
                      'new_paths_requested':sum(b['parents_per_config']*len(b['ids']) for b in batches)}))


def run(first,last):
    gate=ROOT/'calibration/sampler/coordinator_decision.json'
    if not gate.exists() or not json.loads(gate.read_text()).get('allow_exploratory_broad_screen',False):
        raise ValueError('Sampler-reliability decision required before broad graph campaign')
    campaign=json.loads((ROOT/'screen/campaign.json').read_text())
    if sha('experiments/phase3b_graphs.py')!=campaign['producer_sha256'] or sha(DESIGN)!=campaign['initial_design_sha256']:
        raise ValueError('Producer/design version changed')
    batches=campaign['batches'][first:last]
    for b in batches:
        folder=Path(b['output']);folder.mkdir(parents=True,exist_ok=True)
        done=folder/'BATCH_COMPLETE.json'
        if done.exists():
            print(json.dumps({'batch':b['label'],'status':'already_complete'}),flush=True);continue
        plan=folder/'batch_plan.json'
        if not plan.exists():save(plan,b)
        command=b['command']+(['--resume'] if (folder/'metadata.json').exists() else [])
        start=time.perf_counter()
        with (folder/'producer.log').open('a') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
        if result.returncode:raise RuntimeError(f"Batch failed; inspect{folder}/producer.log")
        expected=len(b['ids'])*b['parents_per_config']*2
        actual=len(list(folder.glob('C*/graph_*_n*.npz')))
        failures=list(folder.glob('C*/model_failure.json'))+list(folder.glob('C*/failure_*.json'))
        if actual!=expected and not failures:raise ValueError((b['label'],actual,expected))
        saved=dict(batch=b['label'],seconds=time.perf_counter()-start,expected_size_records=expected,
            saved_size_records=actual,failures=[str(p) for p in failures],
            failure_note='BATCH_COMPLETE means every planned attempt finished or has an explicit failure, not every mathematical covariance was sampled',
            graph_n2048_hashes={str(p):sha(p) for p in folder.glob('C*/graph_*_n2048.npz')})
        save(done,saved);print(json.dumps({k:v for k,v in saved.items() if k!='graph_n2048_hashes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int)
    a=p.parse_args();prepare() if a.stage=='prepare' else run(a.first,a.last)
