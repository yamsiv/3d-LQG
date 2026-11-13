"""Export the actual sampled covariance registry and measured producer cost."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.phase3a_compare import save_json


def main():
    root=Path('results/phase3b')
    specs=json.loads((root/'extended_design.json').read_text())['configurations']
    reliability={r['id']:r for r in json.loads((root/'quality/audited/configuration_reliability.json').read_text())}
    rows=list(csv.DictReader((root/'quality/audited/sampler_accuracy_by_stage_configuration_size.csv').open()))
    registry=[];markdown=[]
    for s in specs:
        selected=[r for r in rows if r['id']==s['id'] and not r['failure']]
        priority={'holdout_graphs':5,'intermediate_graphs':4,'birect_extension_graphs':4,'adaptive_coarse_graphs':2,'screen_graphs':1,'anchor_controls':0}
        best=max(selected,key=lambda r:(priority[r['stage']],int(r['n']))) if selected else None
        counts={stage:max(int(r['independent_paths_within_stage']) for r in selected if r['stage']==stage)
                for stage in {r['stage'] for r in selected}}
        r=dict(id=s['id'],a=s['triple'][0],b=s['triple'][1],c=s['triple'][2],
            family=s['family'],condition=s['condition'],angles_radians=s['angles_radians'],
            covariance=s['covariance'],status=reliability[s['id']]['status'],
            finest_saved_alpha1=float(best['alpha1']) if best else None,
            finest_stage=best['stage'] if best else None,stage_physical_path_counts=counts,
            warning='Large observed coarse harmonic defect' if reliability[s['id']].get('large_observed_h_defect_flag',False) else '')
        registry.append(r)
        triple='('+', '.join(f'{x:.5f}'.rstrip('0').rstrip('.') if x else '0' for x in s['triple'])+')'
        status=('exact minima' if s['id']=='C002' else 'exact grid; polygonal minima' if s['id'] in ['C007','C012']
                else 'certificate failure' if not best else 'provisional; large coarse h defect' if r['warning'] else 'provisional')
        stages='/'.join(k.replace('_graphs','').replace('birect_extension','extension').replace('adaptive_coarse','adaptive')+':'+str(v)
                        for k,v in counts.items() if k!='anchor_controls') or 'none'
        markdown.append(f"| {s['id']} | {triple} | {s['family']} | {s['condition']:.2f} | {stages} | {status} |")
    save_json(root/'covariance_registry.json',registry)
    with (root/'covariance_registry.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(registry[0]));writer.writeheader();writer.writerows(registry)
    (root/'covariance_registry.md').write_text('| ID | Canonical (a,b,c) | Family | Condition | Paths by stage | Reliability |\n|---|---|---|---:|---|---|\n'+'\n'.join(markdown)+'\n')
    times={};counts={}
    for stage in ['screen_graphs','adaptive_coarse_graphs','intermediate_graphs','birect_extension_graphs','holdout_graphs','anchor_controls']:
        physical={}
        for path in (root/stage).glob('C*/graph_*.npz'):
            with np.load(path) as a:
                h=json.loads(str(a['header']));physical[(h['id'],h['seed'])]=h['seconds']
        times[stage]=sum(physical.values());counts[stage]=len(physical)
    for stage in ['field_pilot','screen_fields','intermediate_fields','holdout_fields','holdout_stencil_control','holdout_smooth_control','historical_candidate_control']:
        elapsed=0.;count=0
        for path in (root/stage).glob('xi*/metric_*.npz'):
            with np.load(path) as a:elapsed+=json.loads(str(a['header']))['seconds'];count+=1
        times[stage]=elapsed;counts[stage]=count
    save_json(root/'resource_summary.json',dict(producer_seconds=times,checkpoint_or_path_counts=counts,
        total_producer_seconds=sum(times.values()),
        note='Measured producer durations, deduplicating coupled graph sizes; excludes setup, calibration and sampler-pilot work and saved-data analysis. Concurrent stages mean summed durations are not elapsed wall time.'))
    print(json.dumps({'registry':len(registry),'producer_seconds':times}))


if __name__=='__main__':main()
