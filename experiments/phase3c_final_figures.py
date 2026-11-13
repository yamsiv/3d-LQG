"""Final comparison figures/tables from frozen saved data; no optimization."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase3a_compare import save_json
from experiments.phase3b_search import graph_groups,field_groups
from src.metric_measure_3a import summarize,discrepancy,EVAL_GRID
from src.metric_features_3b import matrix_shape_samples

R=Path('results/phase3c')


def main():
    out=R/'tables_final';fig=out/'figures';plan=json.loads((R/'confirmation/frozen_plan.json').read_text())
    if(out/'metric_measure_comparisons.json').exists():raise FileExistsError(out/'metric_measure_comparisons.json')
    gs=graph_groups(sorted((R/'confirmation/graphs').glob('batch_*')),8192)
    fs=field_groups([R/'fields/confirmation/log'],48);flat=field_groups([R/'fields/confirmation/flat'],48)[(0.,'central')]
    rows=json.loads((R/'confirmation/analysis/scores.json').read_text())['rows']
    refined=json.loads((R/'families/refinement/analysis/scores.json').read_text())['rows']
    with np.load(R/'confirmation/analysis/bootstrap_grounded.npz')as z:boot={k:z[k]for k in z.files}
    physical=[];cached={}
    def summary(rs):return summarize(rs,scale=np.mean([np.mean(r['root_means'])for r in rs]))
    for i in plan['graph_ids']:
        xi=plan['profiles_frozen_from_discovery'][i]['nominal_xi'];g=summary(gs[i]);f=summary(fs[(xi,'central')]);point=discrepancy(g,f)
        cached[(i,xi)]=(g,f);draws=[]
        for a,b in zip(boot[f'graph_indices_{i}'],boot['field_indices']):draws.append(discrepancy(summary([gs[i][j]for j in a]),summary([fs[(xi,'central')][j]for j in b])))
        physical.append(dict(id=i,xi=xi,point=point,mc95={k:np.quantile([d[k]for d in draws],[.025,.975]).tolist()for k in point if point[k]is not None},
            graph_ball_between=g['ball_between_realization_variance'],field_ball_between=f['ball_between_realization_variance'],
            graph_ball_within=g['ball_within_root_variance'],field_ball_within=f['ball_within_root_variance'],
            scope='32wholeparentdraws,recomputedensemblemeans;MConly'))
    save_json(out/'metric_measure_comparisons.json',physical)
    # Frozen nominal points only; do not substitute holdout minimizers.
    ids=plan['graph_ids'];f,ax=plt.subplots(figsize=(11,5))
    for offset,source,label,color in [(-.1,refined,'Discovery fine (24G,48F)','C0'),(.1,rows,'Independent (16G,24F)','C1')]:
        chosen=[next(r for r in source if r['id']==i and r['xi']==plan['profiles_frozen_from_discovery'][i]['nominal_xi']and r['gauge']=='grounded'and r['normalization']=='mean'and r['weighting']=='primary')for i in ids]
        vals=np.array([r['robust_score']for r in chosen]);interval=np.array([r['robust_mc95']for r in chosen]);x=np.arange(len(ids))+offset
        # Intervals of nonregular scalar scores can fail to cover point estimates;
        # draw interval segments directly, do not alter/clamp saved estimates.
        ax.vlines(x,interval[:,0],interval[:,1],color=color,alpha=.7);ax.scatter(x,vals,label=label,color=color,s=24)
    ax.axhline(.26041420028,ls='--',color='gray',label='Frozen scalar ε (not equality threshold)');ax.set_xticks(range(len(ids)),ids,rotation=45)
    ax.set(ylabel='Robust J at frozen nominal ξ',title='Candidate-family confirmation; conditional MC95 ranges');ax.legend(fontsize=8);f.tight_layout();f.savefig(fig/'family_confirmation.png',dpi=180);plt.close(f)
    f,axs=plt.subplots(2,3,figsize=(13,8));fl=summary(flat)
    for j,i in enumerate(['C047','C008']):
        xi=plan['profiles_frozen_from_discovery'][i]['nominal_xi'];g,field=cached[(i,xi)]
        ax=axs[j,0]
        for s,label,c in [(g,f'{i} graph','C0'),(field,f'LFPP/GMC ξ={xi:g}','C1'),(fl,'Flat 98-direction torus','black')]:
            ax.plot(EVAL_GRID,s['cdf'],label=label,color=c)
            if c!='black':q=np.quantile(s['realization_cdfs'],[.1,.9],axis=0);ax.fill_between(EVAL_GRID,*q,alpha=.15,color=c)
        ax.set(xlim=(0,2.5),ylim=(0,1),xlabel='Distance / ensemble mean',ylabel='Pair CDF',title='Bands: 10–90% parent variability');ax.legend(fontsize=7)
        ax=axs[j,1]
        for s,label,c in [(g,'Graph','C0'),(field,'LFPP/GMC','C1')]:ax.hist(s['balls'][:,:,2].ravel(),bins=np.linspace(0,1,21),density=True,histtype='step',label=label,color=c)
        ax.set(xlabel='Mass of ball at r=.75 ensemble units',ylabel='Root density',title='Root distribution; parents are inference units');ax.legend(fontsize=8)
        ax=axs[j,2]
        for rs,label,c in [(gs[i],'Graph','C0'),(fs[(xi,'central')],'LFPP/GMC','C1'),(flat,'Flat','black')]:
            ts=[];qs=[]
            for r in rs:
                s=matrix_shape_samples(r['matrix']);ts.append(s['triangles'][:,0].mean());qs.append(s['quartets'][:,0].mean())
            ax.scatter(ts,qs,label=label,c=c,s=18,alpha=.7)
        ax.set(xlabel='Parent mean triangle smallest/largest',ylabel='Parent mean quartet delta ratio',title='One point per independent parent');ax.legend(fontsize=7)
    f.tight_layout();f.savefig(fig/'intrinsic_holdout_observables.png',dpi=180);plt.close(f)
    # Display all signed family residual magnitudes, not only a scalar ranking.
    chosen=[r for r in rows if r['nominal']and r['gauge']=='grounded'and r['normalization']=='mean'and r['weighting']=='primary'];families=list(chosen[0]['family_components'])
    image=np.array([[next(r for r in chosen if r['id']==i)['family_components'][k]for k in families]for i in ids])
    f,ax=plt.subplots(figsize=(10,5));im=ax.imshow(image,cmap='magma',vmin=0,vmax=.45,aspect='auto');ax.set_xticks(range(len(families)),families,rotation=35,ha='right');ax.set_yticks(range(len(ids)),ids);ax.set_title('Independent confirmation: feature-family RMS residuals');f.colorbar(im,ax=ax,label='Residual / declared natural scale');f.tight_layout();f.savefig(fig/'holdout_feature_families.png',dpi=180);plt.close(f)
    # Joint support across graph-size/kernel/field refinement, no surrogate.
    f,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,gauge in zip(axs,['grounded','unit_mass']):
        for i in ['C000','C047','C110','C008','C002']:
            rs=[r for r in rows if r['id']==i and r['gauge']==gauge and r['normalization']=='mean'and r['weighting']=='primary'];rs.sort(key=lambda r:r['xi'])
            ax.plot([r['xi']for r in rs],[r['robust_score']for r in rs],'-o',ms=3,label=i)
            for r in rs:
                if not r['componentwise']:ax.scatter(r['xi'],r['robust_score'],marker='x',c='black',s=35)
        ax.set(xlabel='Linked ξ',ylabel='Independent robust J',title=f'{gauge}; × = component warning');ax.legend(fontsize=7)
    f.tight_layout();f.savefig(fig/'holdout_parameter_profiles.png',dpi=180);plt.close(f)
    print(json.dumps({'comparisons':len(physical),'figures':4,'physical_simulations':0}),flush=True)


if __name__=='__main__':main()
