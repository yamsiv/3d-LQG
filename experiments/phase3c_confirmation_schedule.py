"""Execute only the already frozen remaining confirmation batches sequentially.

Each subprocess remains an explicit <=6-parent experiment (N64<=4), with its
own prior plan and metric checkpoints. Only resource/count summaries are read;
this scheduler never sees or tunes geometric observables.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
from src.phase3c_fields import write_json_new,sha256_file


def main():
    root=Path('results/phase3c/fields/confirmation')
    main_plan='results/phase3c/confirmation/frozen_plan.json'
    stencil_plan='results/phase3c/confirmation/frozen_stencil_control.json'
    positive=['.175','.2','.225','.25','.2625'];matched=['.2','.225','.25']
    jobs=[]
    def add(label,ids,sizes,models,xis,output=str(root),config=main_plan):
        jobs.append(dict(label=label,ids=ids,sizes=sizes,models=models,xis=xis,output=output,config=config))
    for start in[1012,1018]:
        add(f'log_n32_{start}_{start+5}',list(range(start,start+6)),[32],['log'],positive)
        # The previous6-parentN48process reached784MB. Restarting between
        # three-parent chunks limits allocator retention without changing
        # any scientific setting or physical realization.
        for small in[start,start+3]:
            add(f'log_n48_{small}_{small+2}',list(range(small,small+3)),[48],['log'],positive)
    for start in[1000,1006,1012,1018]:
        for n in[32,48]:add(f'flat_n{n}_{start}_{start+5}',list(range(start,start+6)),[n],['flat'],['0'])
    for ids in[list(range(1000,1006)),[1006,1007]]:
        for n in[32,48]:add(f'smooth_n{n}_{ids[0]}_{ids[-1]}',ids,[n],['smooth'],['.175','.225','.2625'])
    stencil_output='results/phase3c/fields/confirmation_stencil_control'
    for ids in[list(range(1000,1006)),[1006,1007]]:
        add(f'stencil26_n48_{ids[0]}_{ids[-1]}',ids,[48],['log'],matched,stencil_output,stencil_plan)
    add('n64_counting_pilot_1000',[1000],[64],['log'],['.225'])
    for start in[1000,1004]:add(f'log_n64_{start}_{start+3}',list(range(start,start+4)),[64],['log'],matched)
    plan_path=root/'remaining_schedule_plan.json'
    expected={'jobs':jobs,'main_plan_sha256':sha256_file(main_plan),
        'stencil_plan_sha256':sha256_file(stencil_plan),'no_scientific_choice_or_value_inspection':True}
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=expected:raise ValueError('Saved schedule differs')
    else:write_json_new(plan_path,expected)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='.mplconfig')
    for job in jobs:
        output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
        completion=output/'batches'/f"{job['label']}.json"
        if completion.exists():
            print(json.dumps({'label':job['label'],'existing_completed_batch_reused':True}),flush=True)
            continue
        cmd=[sys.executable,'-m','experiments.phase3c_confirmation_fields','run',
            '--config',job['config'],'--output',job['output'],'--models',*job['models'],
            '--sizes',*map(str,job['sizes']),'--field-ids',*map(str,job['ids']),
            '--xis',*job['xis'],'--label',job['label']]
        log=output/f"{job['label']}.log"
        # A resumed invocation appends its resource output;completed numerical
        # checkpoints are skipped by the producer,never overwritten.
        with log.open('a')as stream:
            completed=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
        if completed.returncode:
            print(json.dumps({'label':job['label'],'failed_exit_code':completed.returncode,'log':str(log)}),flush=True)
            raise SystemExit(completed.returncode)
        summary=json.loads(completion.read_text())
        print(json.dumps({'label':job['label'],**{k:summary[k]for k in
            ['field_ids','sizes','models','seconds','peak_rss_native','new_records']}}),flush=True)
        if 64 in job['sizes']and summary['peak_rss_native']>1_500_000_000:
            print(json.dumps({'N64_resource_fallback_triggered':True,'remaining_N64_jobs_stopped':True}),flush=True)
            raise SystemExit(3)
    print(json.dumps({'frozen_schedule_complete':True,'jobs':len(jobs)}),flush=True)


if __name__=='__main__':main()
