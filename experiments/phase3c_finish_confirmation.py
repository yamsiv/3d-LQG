"""Finish diagnostics from cached confirmation features/bootstrap, no refit.

The main score job saved all point/32-draw results before a duplicate keyword
in a diagnostic dictionary stopped it. This resumes only the unsaved section;
the executed source version is retained in confirmation/source_snapshots.
"""
from pathlib import Path
import ast
import hashlib
import json
import time
import numpy as np
from experiments.phase3a_compare import save_json
from experiments.phase3b_search import graph_groups,field_groups
from experiments.phase3c_compare import _array_score
from src.family_search_3c import ensemble_features,compare_features,WEIGHTINGS
from src.confirmation_statistics_3c import matrix_diagnostic


def main():
    start=time.perf_counter();root=Path('results/phase3c');out=root/'confirmation/analysis'
    if(out/'COMPLETE.json').exists():raise FileExistsError(out/'COMPLETE.json')
    plan=json.loads((root/'confirmation/frozen_plan.json').read_text());features=json.loads((out/'point_features.json').read_text())
    gs=graph_groups(sorted((root/'confirmation/graphs').glob('batch_*')),8192);gids=sorted(gs)
    fields=field_groups([root/'fields/confirmation/log'],48);flat=field_groups([root/'fields/confirmation/flat'],48)
    tol=json.loads((root/'families/statistics/tolerance_calibration.json').read_text());rng=np.random.default_rng(202611123989)
    diagnostics=[];matrix=[];contrasts=[];layout=features['grounded:mean']['graphs'][gids[0]]
    for gauge in ['grounded','unit_mass']:
        with np.load(out/f'bootstrap_{gauge}.npz')as z:cache={k:z[k]for k in z.files}
        fkeys=[ast.literal_eval(k)for k in cache['fkeys']];draws=len(cache['g']);baseline=flat[(0.,'central')]
        flat_feature=ensemble_features(baseline,gauge=gauge)
        flat_draws=[ensemble_features([baseline[j]for j in ix],gauge=gauge)for ix in cache['field_indices']]
        def score(g,gv,f,fv):return _array_score(np.asarray(g),np.asarray(gv),np.asarray(f),np.asarray(fv),layout['scales'],layout['families'],WEIGHTINGS['primary'])
        for i in gids:
            gi=gids.index(i);xi=plan['profiles_frozen_from_discovery'][i]['nominal_xi'];fi=fkeys.index((xi,'central'))
            g=features[f'{gauge}:mean']['graphs'][i];f=features[f'{gauge}:mean']['fields'][str((xi,'central'))]
            central=score(g['vector'],g['valid'],f['vector'],f['valid']);flat_score=score(g['vector'],g['valid'],flat_feature['vector'],flat_feature['valid'])
            values=[score(cache['g'][d,gi],cache['gv'][d,gi],cache['f'][d,fi],cache['fv'][d,fi])-score(cache['g'][d,gi],cache['gv'][d,gi],flat_draws[d]['vector'],flat_draws[d]['valid'])for d in range(draws)]
            contrasts.append(dict(id=i,xi=xi,gauge=gauge,central_log_score=central,flat_score=flat_score,log_minus_flat=central-flat_score,
                mc95=np.quantile(values,[.025,.975]).tolist(),interpretation='Sharedphysicalfieldandgraphresamples;conditionalMConly'))
            for model,rs in [('log',fields[(xi,'central')]),('flat',baseline)]:
                c=compare_features(g,ensemble_features(rs,gauge=gauge),tol,gauge=gauge,sampler_allowed=True)
                diagnostics.append(dict(id=i,model=model,xi=xi if model=='log'else 0,parents=len(rs),**c))
            if gauge=='grounded':matrix.append(dict(id=i,xi=xi,**matrix_diagnostic(gs[i],fields[(xi,'central')],rng)))
    save_json(out/'negative_controls.json',diagnostics);save_json(out/'matrix_diagnostics.json',matrix);save_json(out/'log_minus_flat.json',contrasts)
    direct=[]
    for family in ['birectangular_level_segment','negative_equicorrelation_region']:
        ids=plan['hypotheses'][family]['ids']
        for k,i in enumerate(ids):
            for j in ids[k+1:]:direct.append(dict(family=family,left=i,right=j,**matrix_diagnostic(gs[i],gs[j],rng)))
    save_json(out/'direct_graph_family_matrices.json',direct)
    save_json(out/'diagnostic_resume.json',dict(reason='Duplicategaugekeywordindictionaryafterprimaryscorescompleted',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary_scores_sha256=hashlib.sha256((out/'scores.json').read_bytes()).hexdigest(),
        recomputed_physical_samples=0,recomputed_primary_feature_bootstraps=0,seconds=time.perf_counter()-start))
    save_json(out/'COMPLETE.json',dict(graph_parents=176,field_parents=24,feature_bootstrap_draws=32,primary_scores_unchanged=True,
        no_parameter_retuning=True,diagnostic_resume_seconds=time.perf_counter()-start))
    print(json.dumps(json.loads((out/'COMPLETE.json').read_text())),flush=True)


if __name__=='__main__':main()
