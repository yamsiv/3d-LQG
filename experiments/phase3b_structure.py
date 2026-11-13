"""Descriptive structure of the saved Phase 3B covariance screen.

No simulation is launched. Costs are measured from the feature implementation:
28 covariances x24 six-graph cluster bootstraps ~15 seconds, <150 MB. All
bootstrap observations are physical paths, keeping paired graph sizes intact.
The analysis does not estimate Hausdorff dimensions or certify level sets.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from experiments.phase3b_search import graph_groups
from experiments.phase3b_statistics_audit import _weight_variants
from src.metric_features_3b import feature_vector, grouped_feature_distance, matrix_shape_samples
from src.provenance import environment_manifest


SEED = 202611113841


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_jsonable(v) for v in value]
    return value


def _save(path, value):
    with Path(path).open('x') as stream:
        json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def _layout(raw):
    row = dict(raw)
    for name in ('vector','scales'): row[name] = np.asarray(row[name], float)
    row['valid'] = np.asarray(row['valid'], bool)
    return row


def _physical_key(record):
    header = record['header']
    return header.get('physical_path_id', (header['id'],header['seed']))


def _growth(small, large, rng, draws):
    left = {_physical_key(r): r for r in small}
    right = {_physical_key(r): r for r in large}
    if set(left) != set(right):
        raise ValueError('Both graph sizes must have identical physical path keys')
    keys = sorted(left)
    a = np.array([np.mean(left[key]['root_means']) for key in keys])
    b = np.array([np.mean(right[key]['root_means']) for key in keys])
    n1,n2 = small[0]['header']['n'],large[0]['header']['n']
    log_ratio = np.log(n2/n1)
    index = rng.integers(len(keys),size=(draws,len(keys)))
    boot = np.log(np.mean(b[index],axis=1)/np.mean(a[index],axis=1))/log_ratio
    return {'beta_finite':float(np.log(b.mean()/a.mean())/log_ratio),
            'mc95':np.quantile(boot,[.025,.975]),'bootstrap':boot,
            'small_mean_distance':float(a.mean()),'large_mean_distance':float(b.mean()),
            'physical_paths':len(keys),'sizes':[n1,n2],
            'interpretation':'two-size log mean-distance growth proxy, not an asymptotic exponent or Hausdorff dimension'}


def _shape_means(records):
    shapes = [matrix_shape_samples(r['matrix']) for r in records]
    return {name:np.array([r[key][:,index].mean() for r in shapes])
            for key,index,name in (('triangles',0,'triangle_small'),('triangles',1,'triangle_middle'),
                                   ('quartets',0,'quartet_delta'),('quartets',1,'quartet_aspect'))}


def _reweight(components,weights):
    denom = sum(weights.get(key,0.) for key in components)
    return sum(weights.get(key,0.)*value for key,value in components.items())/denom


def _sensitivity(data):
    result={}; choices={}; winners={}
    for gauge,key in [('grounded','rows'),('unit_mass','unit_mass_rows')]:
        rows=data[key]
        central=[r for r in rows if r['eligible_primary']]
        identifiers=sorted({r['id']for r in central})
        for variant,weights in _weight_variants().items():
            candidates=[]
            for row in central:
                probes=[r for r in rows if r['id']==row['id']and r['xi']==row['xi']]
                robust=max(_reweight(r['components'],weights)for r in probes)
                candidates.append(dict(id=row['id'],xi=row['xi'],gamma=row['gamma'],score=robust))
            best={identifier:min((r for r in candidates if r['id']==identifier),key=lambda r:r['score'])
                  for identifier in identifiers}
            ordered=sorted(best.values(),key=lambda r:r['score'])
            result[f'{gauge}_{variant}']=ordered
            choices[f'{gauge}_{variant}']={r['id']:r['xi']for r in ordered}
            winner=ordered[0]
            label=f"{winner['id']}:xi={winner['xi']:g}"
            winners.setdefault(gauge,Counter())[label]+=1
    reference=result['grounded_default']
    reference_rank={r['id']:i for i,r in enumerate(reference)}
    rank_sensitivity={}
    for name,ordered in result.items():
        rank={r['id']:i for i,r in enumerate(ordered)}
        correlation=float(spearmanr([reference_rank[i]for i in sorted(rank)],
                                    [rank[i]for i in sorted(rank)]).statistic)
        rank_sensitivity[name]={'spearman_vs_grounded_default':correlation,
                                'top5_overlap':len(set(r['id']for r in ordered[:5])&set(r['id']for r in reference[:5]))}
    return {'rankings':result,'xi_choices':choices,'global_winner_counts':winners,
            'rank_sensitivity':rank_sensitivity,'weight_scenarios':len(_weight_variants()),
            'optimization':'max all calibration-probe scores at each xi, then min eligible central xi',
            'warning':'post-selection weight/gauge sensitivities, not independent confirmation'}


def _reliability(records):
    headers=[r['header']for r in records]
    absolute=[];signed=[];dt=[]
    for h in headers:
        legs=h['sampler_metadata'].get('leg_diagnostics',[])
        absolute.append(sum(x.get('sum_abs_log_mean_value_ratio',0.)for x in legs))
        signed.append(sum(x.get('sum_log_mean_value_ratio',0.)for x in legs))
        dt.append(h['sampler_metadata'].get('max_normalized_dt',0.))
    sampler=headers[0]['sampler_metadata']['sampler']
    if 'independent_bessel3_exact_minima'in sampler: category='exact_continuous_minima'
    elif sampler.startswith(('weyl_a2','weyl_a3')): category='exact_matrix_grid_approximate_minima'
    else: category='provisional_Doob_and_minima'
    return {'category':category,'sampler':sampler,
            'path_abs_log_meanvalue_median':float(np.median(absolute)),
            'path_abs_log_meanvalue_max':float(np.max(absolute)),
            'path_signed_log_meanvalue_mean':float(np.mean(signed)),
            'max_normalized_dt':float(np.max(dt)),
            'warning':'mean-value diagnostics are path summaries, not total-law error bounds; zero for references means not applicable'}


def _intermediate(args, out):
    path=out/'intermediate_structure.json'
    if path.exists(): return
    meta=Path(args.intermediate)/'metadata.json'
    if not meta.exists():
        print('Intermediate metadata unavailable; retained completed screen analysis',flush=True)
        return
    config=json.loads(meta.read_text())['configuration']
    groups={n:graph_groups([args.intermediate],n)for n in config['sizes']}
    expected=config['ids']
    counts={str(n):{i:len(g.get(i,[]))for i in expected}for n,g in groups.items()}
    if any(counts[str(n)][i]!=config['samples']for n in groups for i in expected):
        print(json.dumps({'intermediate_incomplete':counts}),flush=True)
        return
    small,large=sorted(config['sizes'])
    old=graph_groups([args.graphs],small)
    rng=np.random.default_rng(SEED+999)
    rows=[]
    for identifier in expected:
        feature=feature_vector(groups[large][identifier])
        item={'id':identifier,'triple':groups[large][identifier][0]['header']['spec']['triple'],
              'growth':_growth(groups[small][identifier],groups[large][identifier],rng,args.growth_bootstrap),
              'features':feature,'reliability':_reliability(groups[large][identifier])}
        if identifier in old:
            original=feature_vector(old[identifier])
            refined=feature_vector(groups[small][identifier])
            item['same_n_coarse_to_refined']=grouped_feature_distance(original,refined)
            item['refinement_warning']='new physical seeds AND degree/path settings changed; this is not a coupled pure discretization estimate'
        rows.append(item)
    _save(path,{'rows':rows,'configuration':config,'counts':counts,
                'note':'independent intermediate seeds; no new simulation was launched by this script'})
    _growth_plot(out/'intermediate_growth_shapes',rows)


def _adaptive_augmentation(args,out):
    """Add the three completed coarse GP proposals without changing old files."""
    result=out/'adaptive_augmentation.json'
    if result.exists():return
    directory=Path(args.adaptive_graphs)
    meta=directory/'metadata.json'
    if not meta.exists():return
    config=json.loads(meta.read_text())['configuration']
    coarse={n:graph_groups([directory],n)for n in config['sizes']}
    if any(len(coarse[n].get(i,[]))!=config['samples']for n in coarse for i in config['ids']):
        print('Adaptive coarse proposals incomplete; preserving old analysis',flush=True);return
    data=json.loads((Path(args.screen_analysis)/'scores.json').read_text())
    original=json.loads((out/'covariance_table.json').read_text())
    with np.load(out/'feature_bootstrap.npz')as archive:noise=archive['noise_scales']
    fields={ast.literal_eval(key):_layout(value)for key,value in data['field_features'].items()}
    graph_features={i:_layout(value)for i,value in data['graph_features'].items()}
    templates={}
    for row in data['rows']:
        templates.setdefault((row['xi'],row['band']),row)
    small,large=sorted(config['sizes']);rng=np.random.default_rng(SEED+2222)
    additions=[];rows=[]
    intermediate=graph_groups([args.intermediate],large)
    for identifier in config['ids']:
        gf=feature_vector(coarse[large][identifier]);graph_features[identifier]=gf
        candidates=[]
        for key,ff in fields.items():
            xi,band=key;template=templates[key]
            score=grouped_feature_distance(gf,ff)
            candidates.append({'id':identifier,'xi':xi,'gamma':template['gamma'],'band':band,
                               'eligible_primary':template['eligible_primary'],'score':score['total']})
        for candidate in candidates:
            candidate['robust_score']=max(r['score']for r in candidates if r['xi']==candidate['xi'])
        eligible=[r for r in candidates if r['eligible_primary']]
        best=min(eligible,key=lambda r:r['robust_score'])
        spec=coarse[large][identifier][0]['header']['spec']
        entry={'id':identifier,'triple':spec['triple'],'condition':spec['condition'],'family':spec['family'],
               'best':{'id':identifier,'xi':best['xi'],'gamma':best['gamma'],'score':best['robust_score']},
               'growth':_growth(coarse[small][identifier],coarse[large][identifier],rng,args.growth_bootstrap),
               'reliability':_reliability(coarse[large][identifier]),'samples':config['samples']}
        if identifier in intermediate:
            refined=feature_vector(intermediate[identifier])
            entry['same_n_coarse_to_intermediate']=grouped_feature_distance(gf,refined)
            entry['refinement_warning']='new physical seeds and numerical parameters both change; not paired method bias'
        additions.append(entry);rows.extend(candidates)
    all_records=original+additions
    pair_distances=[]
    for addition in additions:
        a=addition['id']
        for other in all_records:
            b=other['id']
            if a==b:continue
            score=grouped_feature_distance(graph_features[a],graph_features[b],noise_scales=noise)
            pair_distances.append({'left':a,'right':b,
                                   'covariance_distance':float(np.linalg.norm(np.array(addition['triple'])-other['triple'])),
                                   'fixed_noise_feature_score':score['total']})
    pca_features=np.array([graph_features[r['id']]['vector']/graph_features[r['id']]['scales']for r in all_records])
    _,singular,_=np.linalg.svd(pca_features-pca_features.mean(axis=0),full_matrices=False)
    ratio=singular**2/(singular**2).sum()
    _save(result,{'successful_covariances':len(all_records),'attempted_covariances':len(all_records)+1,
                  'failed_id':'C022','new_configuration':config,'additions':additions,'coarse_field_scores':rows,
                  'all_covariance_table':all_records,'new_to_existing_feature_distances':pair_distances,
                  '31_configuration_PCA_explained_ratios':ratio,
                  'units':'fixed_noise_feature_score uses EXACT same noise_scales as original far-pair and control self-comparisons; primary scores remain natural-scale units',
                  'warning':'Adaptive proposals selected from earlier scores; not independent confirmations or new level-set evidence'})
    # Preserve the 28-point figures; produce a clearly separate31-point map.
    plot_folder=out/'adaptive31';plot_folder.mkdir(exist_ok=True)
    _covariance_plot(plot_folder,all_records)
    paths=[Path(__file__)]+list(directory.glob('C*/graph_*.npz'))
    _save(out/'adaptive_input_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})


def _adaptive_fidelity_addendum(args,out):
    """Explicit2048-vertex comparisons of new coarse proposals and refinement."""
    path=out/'adaptive_fidelity_addendum.json'
    if path.exists():return
    coarse=graph_groups([args.adaptive_graphs],2048)
    refined=graph_groups([args.intermediate],2048)
    if not coarse or any(i not in refined or len(refined[i])!=8 for i in coarse):return
    rows=[]
    for identifier,records in coarse.items():
        a=feature_vector(records);b=feature_vector(refined[identifier])
        rows.append({'id':identifier,'coarse_samples':len(records),'intermediate_samples':len(refined[identifier]),
                     'natural_feature_score':grouped_feature_distance(a,b),
                     'coarse_median':a['diagnostics']['normalization'],
                     'intermediate_median':b['diagnostics']['normalization'],
                     'warning':'physical seeds and numerical settings both differ; not a paired method-error estimate'})
    _save(path,{'rows':rows,'n':2048,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--screen-analysis',default='results/phase3b/screen_analysis')
    parser.add_argument('--graphs',default='results/phase3b/screen_graphs')
    parser.add_argument('--intermediate',default='results/phase3b/intermediate_graphs')
    parser.add_argument('--adaptive-graphs',default='results/phase3b/adaptive_coarse_graphs')
    parser.add_argument('--output',default='results/phase3b/structure')
    parser.add_argument('--bootstrap',type=int,default=24)
    parser.add_argument('--growth-bootstrap',type=int,default=64)
    parser.add_argument('--intermediate-only',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.intermediate_only:
        _intermediate(args,out);_adaptive_augmentation(args,out);_adaptive_fidelity_addendum(args,out);return
    if (out/'summary.json').exists():
        print('Completed coarse structure analysis exists; checking unfinished intermediate summary',flush=True)
        _intermediate(args,out);_adaptive_augmentation(args,out);_adaptive_fidelity_addendum(args,out);return
    start=time.perf_counter();rng=np.random.default_rng(SEED)
    _save(out/'configuration.json',{'arguments':vars(args),'seed':SEED,'expected_seconds':90,
                                   'expected_memory_mb':150,'no_simulations':True,
                                   'provenance':environment_manifest()})
    data=json.loads((Path(args.screen_analysis)/'scores.json').read_text())
    small=graph_groups([args.graphs],512);large=graph_groups([args.graphs],2048)
    identifiers=sorted(data['graph_features'])
    features={i:_layout(data['graph_features'][i])for i in identifiers}
    sensitivity=_sensitivity(data);_save(out/'sensitivity.json',sensitivity)
    best={row['id']:row for row in sensitivity['rankings']['grounded_default']}
    records=[]
    for identifier in identifiers:
        spec=large[identifier][0]['header']['spec']
        records.append({'id':identifier,'triple':spec['triple'],'family':spec['family'],
                        'condition':spec['condition'],'best':best[identifier],
                        'growth':_growth(small[identifier],large[identifier],rng,args.growth_bootstrap),
                        'reliability':_reliability(large[identifier]),'samples':len(large[identifier])})
    _save(out/'covariance_table.json',records)
    # Every bootstrap draw is a resample of whole six-graph ensembles. Projection
    # on the fixed empirical PC basis is descriptive; no de-noised manifold rank.
    valid=np.all([features[i]['valid']for i in identifiers],axis=0)
    scale=features[identifiers[0]]['scales']
    x=np.array([features[i]['vector']/scale for i in identifiers])[:,valid]
    center=x.mean(axis=0)
    _,singular,vectors=np.linalg.svd(x-center,full_matrices=False)
    scores=(x-center)@vectors.T
    between=singular**2/(len(identifiers)-1)
    variances=[];noise_vectors={};boot_features={}
    for identifier in identifiers:
        source=large[identifier];samples=[]
        for _ in range(args.bootstrap):
            index=rng.integers(len(source),size=len(source))
            samples.append(feature_vector([source[i]for i in index])['vector'])
        samples=np.asarray(samples);boot_features[identifier]=samples
        projected=(samples[:,valid]/scale[valid]-center)@vectors.T
        variances.append(np.var(projected,axis=0,ddof=1))
        noise_vectors[identifier]=np.std(samples,axis=0,ddof=1)
    within=np.mean(variances,axis=0)
    explained=between/between.sum()
    # Fixed analytic-control scale; no candidate gains an advantage by being
    # intrinsically noisy. It approximates a six-versus-six difference scale.
    controls=['C002','C007','C012']
    noise=np.sqrt(2*np.mean([noise_vectors[i]**2 for i in controls],axis=0))
    noise=np.maximum(noise,.05*scale)
    self_scores={}
    for identifier in controls:
        base=features[identifier];draws=boot_features[identifier]
        scores_same=[]
        for index in range(len(draws)//2):
            a=dict(base,vector=draws[2*index]);b=dict(base,vector=draws[2*index+1])
            scores_same.append(grouped_feature_distance(a,b,noise_scales=noise)['total'])
        self_scores[identifier]={'scores':scores_same,'median':float(np.median(scores_same)),
                                 'note':'conditional whole-record bootstrap self-comparison, not a null test'}
    triples=np.array([row['triple']for row in records])
    design=np.column_stack((np.ones(len(triples)),triples))
    coef=np.linalg.lstsq(design,x,rcond=None)[0]
    response_singular=np.linalg.svd(coef[1:],compute_uv=False)
    correlations={}
    for covariate,values in [('mean_correlation',triples.mean(axis=1)),
                            ('condition',[r['condition']for r in records]),
                            ('log_meanvalue_defect',np.log1p([r['reliability']['path_abs_log_meanvalue_median']for r in records]))]:
        correlations[covariate]=[float(spearmanr(values,scores[:,j]).statistic)for j in range(3)]
    pca={'ids':identifiers,'feature_names':[name for name,ok in zip(features[identifiers[0]]['names'],valid)if ok],
         'valid_all_features':valid,'natural_scales':scale,'scores':scores,'loadings':vectors,
         'between_variance':between,'explained_ratio':explained,'conditional_MC_projected_variance':within,
         'PCs_for_90pct_variance':int(np.searchsorted(np.cumsum(explained),.9)+1),
         'linear_covariance_response_singular_values':response_singular,
         'PC_spearman_correlations':correlations,
         'warning':'cross-covariance finite feature variation includes Monte Carlo and covariance-dependent sampler/quantization bias; rank is not continuum geometric dimension'}
    _save(out/'pca.json',pca)
    np.savez_compressed(out/'feature_bootstrap.npz',ids=identifiers,noise_scales=noise,
                        natural_scales=scale,**{i:boot_features[i]for i in identifiers})
    _save(out/'same_model_noise.json',self_scores)
    shape={i:_shape_means(large[i])for i in identifiers}
    pairs=[]
    for ia,ib in combinations(range(len(identifiers)),2):
        a,b=identifiers[ia],identifiers[ib]
        distance=float(np.linalg.norm(triples[ia]-triples[ib]))
        score=grouped_feature_distance(features[a],features[b],noise_scales=noise)
        delta=records[ia]['growth']['beta_finite']-records[ib]['growth']['beta_finite']
        item={'left':a,'right':b,'covariance_distance':distance,'feature_score':score['total'],
              'components':score['components'],'growth_beta_difference':float(delta),
              'growth_difference_mc95':np.quantile(records[ia]['growth']['bootstrap']-records[ib]['growth']['bootstrap'],[.025,.975])}
        for key in shape[a]:
            va,vb=shape[a][key],shape[b][key]
            se=float(np.sqrt(np.var(va,ddof=1)/len(va)+np.var(vb,ddof=1)/len(vb)))
            diff=float(np.mean(va)-np.mean(vb))
            item[key+'_difference']=diff;item[key+'_SE']=se
            item[key+'_signal_noise']=abs(diff)/se if se else 0.
        pairs.append(item)
    far=sorted([r for r in pairs if r['covariance_distance']>=.5],key=lambda r:r['feature_score'])[:12]
    similar_growth=sorted([r for r in pairs if abs(r['growth_beta_difference'])<.015 and r['covariance_distance']>=.35],
                          key=lambda r:max(r['quartet_delta_signal_noise'],r['triangle_small_signal_noise']),reverse=True)[:12]
    _save(out/'pair_structure.json',{'all_pairs':pairs,'far_covariance_similar_features':far,
                                    'similar_finite_growth_different_shapes':similar_growth,
                                    'multiple_selection_warning':'378 pairs examined; SNRs and selected examples are descriptive, not multiplicity-corrected tests or proof of level sets'})
    review={
        'files_reviewed':['src/adaptive_search_3b.py','experiments/phase3b_search.py','experiments/phase3b_fields.py','experiments/phase3b_graphs.py'],
        'findings':[
            {'severity':'interpretation','issue':'selection bootstrap minimizes central-probe score, whereas acquisition/finalists minimize maximum gamma-probe score',
             'resolution':'Label saved argmin frequencies central-score sensitivity or run an additional robust-criterion bootstrap; do not call them uncertainty of robust finalists.'},
            {'severity':'acquisition limitation','issue':'GP diagonal fixed noise ignores shared fields, covariance-dependent MC variance and sampler bias',
             'resolution':'Already documented as heuristic LCB acquisition SD, not inferential confidence; retain held-out and numerical refinements.'},
            {'severity':'completion guard','issue':'search loader permits four realizations, fewer than stage configurations request',
             'resolution':'Check expected IDs and actual stage counts before scoring a running ensemble.'},
            {'severity':'objective sensitivity','issue':'primary score uses natural scales rather than same-model noise scales',
             'resolution':'Valid prespecified descriptive objective; report fixed noise-scaled and weight sensitivities separately, without redefining frozen selection.'},
            {'severity':'calibration interpretation','issue':'calibration eligibility is based on MC subcritical fraction; finite-window systematics can be broader',
             'resolution':'Keep systematic gamma probes and invalid/near-critical regions explicit; max over a few probes is sensitivity, not a rigorous error enclosure.'},
            {'severity':'identity check','issue':'xi bootstrap checks realization numbers but not source hashes',
             'resolution':'Field writer uses common cached source fields and header hashes; verify source hash per realization across xi before calling samples physically coupled.'}],
        'no_generator_or_score_edits_made_by_reviewer':True}
    _save(out/'independent_code_review.json',review)
    _plots(out,records,pca,sensitivity,data)
    elapsed=time.perf_counter()-start
    summary={'screen_covariances':len(identifiers),'attempted_covariances':29,'failed_id':'C022',
             'failed_reason':'positive polynomial certificate unavailable; no graph results silently substituted',
             'sample_counts':{r['id']:r['samples']for r in records},
             'leading_PC_explained_ratios':explained[:5],
             'leading_PC_between_to_conditional_MC_ratio':between[:5]/np.maximum(within[:5],1e-20),
             'far_covariance_most_similar':far[:4],'similar_growth_shape_examples':similar_growth[:4],
             'same_model_conditional_noise_medians':{i:r['median']for i,r in self_scores.items()},
             'gauge_weight_global_winners':sensitivity['global_winner_counts'],
             'elapsed_seconds':elapsed,'no_simulations':True,
             'conclusion':'Candidate degeneracies and several finite feature directions are visible, but small graphs, six paths and covariance-dependent provisional bias prevent distinguishing a special continuum curve from level sets.'}
    inputs=[Path(__file__),Path('src/metric_features_3b.py'),Path(args.screen_analysis)/'scores.json']
    inputs+=list(Path(args.graphs).glob('C*/graph_*.npz'))
    _save(out/'input_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs})
    _save(out/'summary.json',summary)
    _intermediate(args,out)
    _adaptive_augmentation(args,out)
    _adaptive_fidelity_addendum(args,out)
    print(json.dumps(_jsonable(summary)),flush=True)


def _growth_plot(prefix,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for r in rows:
        f=r.get('features');g=r['growth']
        if f is None: continue
        names=f['names'];y=f['vector'][names.index('quartets_delta_mean')]
        interval=np.asarray(g['mc95']);x=g['beta_finite']
        ax.errorbar(x,y,xerr=[[max(x-interval[0],0.)],[max(interval[1]-x,0.)]],fmt='o',ms=4,color='tab:blue')
        ax.annotate(r['id'],(x,y),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.set(xlabel='Paired two-size log-distance growth proxy (not d_H)',ylabel='Mean four-point delta / largest pair sum',title='Intermediate: finite growth does not specify matrix shape')
    fig.savefig(str(prefix)+'.png',dpi=180);fig.savefig(str(prefix)+'.pdf');plt.close(fig)


def _covariance_plot(out,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    triples=np.array([r['triple']for r in records]);best=np.array([r['best']['gamma']for r in records])
    defect=np.array([r['reliability']['path_abs_log_meanvalue_median']for r in records])
    fig=plt.figure(figsize=(12,4.7),constrained_layout=True)
    for panel,values,title,label in [(1,best,'Best crossing-linked gamma (finite screen)','gamma proxy'),
                                     (2,np.log10(np.maximum(defect,1e-6)),'Sampler mean-value diagnostic','log10 median accumulated absolute defect')]:
        ax=fig.add_subplot(1,2,panel,projection='3d')
        points=ax.scatter(*triples.T,c=values,cmap='viridis',s=50,edgecolor='black',lw=.3)
        for index,r in enumerate(records):
            if r['reliability']['category']!='provisional_Doob_and_minima':
                ax.scatter(*triples[index],marker='s',s=95,facecolors='none',edgecolors='red',lw=1.2)
        for r in sorted(records,key=lambda r:r['best']['score'])[:5]:
            ax.text(*r['triple'],r['id'],fontsize=7)
        ax.set(xlabel='canonical a',ylabel='canonical b',zlabel='canonical c',title=title)
        fig.colorbar(points,ax=ax,shrink=.65,pad=.05,label=label)
        ax.view_init(elev=22,azim=-60)
    fig.savefig(out/'covariance_map.png',dpi=180);fig.savefig(out/'covariance_map.pdf');plt.close(fig)


def _plots(out,records,pca,sensitivity,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _covariance_plot(out,records)
    triples=np.array([r['triple']for r in records])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    scores=np.asarray(pca['scores']);values=np.mean(triples,axis=1)
    scatter=axes[0].scatter(scores[:,0],scores[:,1],c=values,cmap='coolwarm',s=40)
    for i,r in enumerate(records): axes[0].annotate(r['id'],scores[i,:2],fontsize=6,xytext=(2,2),textcoords='offset points')
    axes[0].set(xlabel='PC1 in declared natural feature units',ylabel='PC2',title='47 finite features across28 covariance configurations')
    fig.colorbar(scatter,ax=axes[0],label='mean Brownian correlation')
    k=np.arange(1,11)
    axes[1].bar(k-.15,np.asarray(pca['between_variance'])[:10],width=.3,label='Observed across covariances')
    axes[1].bar(k+.15,np.asarray(pca['conditional_MC_projected_variance'])[:10],width=.3,label='Conditional whole-path bootstrap noise')
    axes[1].set(xlabel='Principal component',ylabel='Projected variance',yscale='log',title='MC comparison excludes covariance-dependent sampler bias')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'feature_pca.png',dpi=180);fig.savefig(out/'feature_pca.pdf');plt.close(fig)
    ids=[r['id']for r in sensitivity['rankings']['grounded_default']]
    variants=['default','equal_families','mesoscopic_and_higher_point','without_global_scale','without_slope_and_atoms']
    columns=[f'{g}_{v}'for g in ('grounded','unit_mass')for v in variants]
    heat=np.array([[sensitivity['xi_choices'][col][identifier]for col in columns]for identifier in ids])
    fig,ax=plt.subplots(figsize=(11,7.2),constrained_layout=True)
    image=ax.imshow(heat,aspect='auto',cmap='viridis')
    ax.set(yticks=np.arange(len(ids)),yticklabels=ids,xticks=np.arange(len(columns)),
           xticklabels=[c.replace('grounded_','G:').replace('unit_mass_','U:')for c in columns],
           title='Optimal xi sensitivity: same linked family, prespecified gauge and weights')
    ax.tick_params(axis='x',labelrotation=60,labelsize=7);ax.tick_params(axis='y',labelsize=7)
    fig.colorbar(image,ax=ax,label='Selected finite crossing-linked xi')
    fig.savefig(out/'weight_gauge_sensitivity.png',dpi=180);fig.savefig(out/'weight_gauge_sensitivity.pdf');plt.close(fig)
    rows=[]
    for r in records: rows.append(dict(r,features=data['graph_features'][r['id']]))
    _growth_plot(out/'growth_shapes',rows)


if __name__=='__main__':main()
