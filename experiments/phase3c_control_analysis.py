"""Independent finest-kernel controls against exact reference graph records.

Saved data only. A numerical sampler looking closer to LFPP than its exact
reference is not evidence for a physical correspondence. Approximate and
reference parents are independent; the field resamples are shared.
"""
from pathlib import Path
import ast
import json
import time
import numpy as np
from experiments.phase3a_compare import save_json
from experiments.phase3b_search import graph_groups
from experiments.phase3c_compare import _array_score
from src.family_search_3c import ensemble_features,WEIGHTINGS,matrix_energy_comparison


def main():
    start=time.perf_counter();root=Path('results/phase3c/families/refinement');out=root/'control_analysis';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out/'summary.json')
    approximate=graph_groups([root/'provisional_anchor_controls'],8192)
    reference=graph_groups([root/'reused_graphs',root/'new_a3'],8192)
    features=json.loads((root/'analysis/features.json').read_text())['grounded:mean']
    with np.load(root/'analysis/graph_bootstrap.npz') as z:gb={k:z[k] for k in z.files}
    with np.load(root/'analysis/field_bootstrap.npz') as z:fb={k:z[k] for k in z.files}
    fkeys=[ast.literal_eval(k) for k in fb['keys']];rng=np.random.default_rng(202611123720);rows=[]
    for identifier,xi in [('C002',.25),('C012',.225)]:
        g=ensemble_features(approximate[identifier]);truth=features['graphs'][identifier]
        choices=[i for i,k in enumerate(fkeys) if k[0]==xi];gi=list(gb['ids']).index(identifier)
        def distance(v,valid,j):
            return _array_score(np.asarray(v),np.asarray(valid),fb['point_vectors'][j],fb['point_valid'][j],gb['scales'],list(gb['families']),WEIGHTINGS['primary'])
        point_approx=max(distance(g['vector'],g['valid'],j) for j in choices)
        point_exact=max(distance(truth['vector'],truth['valid'],j) for j in choices)
        deltas=[];shifts=[];mean_changes=[]
        for d in range(len(gb['vectors'])):
            rs=approximate[identifier];ix=rng.integers(len(rs),size=len(rs));bg=ensemble_features([rs[i] for i in ix])
            score_a=max(_array_score(bg['vector'],bg['valid'],fb['vectors'][d,j],fb['valid'][d,j],gb['scales'],list(gb['families']),WEIGHTINGS['primary'])for j in choices)
            score_e=max(_array_score(gb['vectors'][d,gi],gb['valid'][d,gi],fb['vectors'][d,j],fb['valid'][d,j],gb['scales'],list(gb['families']),WEIGHTINGS['primary'])for j in choices)
            deltas.append(score_a-score_e);shifts.append(bg['vector']-gb['vectors'][d,gi])
            exact_rs=reference[identifier];ex=rng.integers(len(exact_rs),size=len(exact_rs))
            mean_changes.append(np.mean([np.mean(rs[i]['root_means'])for i in ix])/np.mean([np.mean(exact_rs[i]['root_means'])for i in ex])-1)
        raw_mean=np.mean([np.mean(r['root_means'])for r in approximate[identifier]])/np.mean([np.mean(r['root_means'])for r in reference[identifier]])-1
        row=dict(id=identifier,xi=xi,approximate_parents=len(approximate[identifier]),reference_parents=len(reference[identifier]),
            points_per_parent=16,mean_distance_relative=raw_mean,mean_relative_mc95=np.quantile(mean_changes,[.025,.975]).tolist(),
            approximate_field_score=point_approx,reference_field_score=point_exact,
            score_approx_minus_reference=point_approx-point_exact,score_difference_mc95=np.quantile(deltas,[.025,.975]).tolist(),
            signed_feature_shift=np.asarray(g['vector'])-np.asarray(truth['vector']),
            signed_feature_shift_mc95=np.quantile(shifts,[.025,.975],axis=0),names=g['names'],
            matrix_energy=matrix_energy_comparison(approximate[identifier],reference[identifier],rng,draws=32),
            scope='MC-only finite cohorts; a smaller approximate score cannot validate the sampler; referenceA3 minima remainpolygonal')
        rows.append(row)
    save_json(out/'summary.json',dict(rows=rows,seconds=time.perf_counter()-start,
        no_new_simulation=True,comparison='Finestgeneral.Doobkernel at8192 versusindependent exactI3andq128A3 reference;matched16points'))
    print(json.dumps([{k:r[k]for k in ['id','mean_distance_relative','approximate_field_score','reference_field_score','score_difference_mc95']}for r in rows]),flush=True)


if __name__=='__main__':main()
