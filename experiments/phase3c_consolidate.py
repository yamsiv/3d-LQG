"""Tables/figures from completed saved checkpoints; no physical simulation.

Output is versioned by caller. Previously sealed simulation files are read-only.
"""
from pathlib import Path
import argparse
import collections
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase3a_compare import save_json
from experiments.phase3b_search import graph_groups
from experiments.phase3c_selected_fields import modelwise_conversion
from src.covariance_design_3b import canonical_triple

R=Path('results/phase3c')


def primary(rows):
    return [r for r in rows if r['gauge']=='grounded'and r['normalization']=='mean'and r['weighting']=='primary']


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='')as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v)if isinstance(v,(list,dict))else v for k,v in row.items()})


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',default=str(R/'tables'));a=p.parse_args();out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    specs=json.loads((R/'adaptive/design.json').read_text())['configurations']
    assert len({canonical_triple(s['triple'])for s in specs})==121
    for s in specs:
        assert np.linalg.eigvalsh(s['covariance'])[0]>.14999
    scores=primary(json.loads((R/'adaptive/analysis/scores.json').read_text())['rows'])
    q=json.loads((R/'adaptive/analysis/reliability.json').read_text())
    refine=primary(json.loads((R/'families/refinement/analysis/scores.json').read_text())['rows'])
    confirm_path=R/'confirmation/analysis/scores.json';confirm=primary(json.loads(confirm_path.read_text())['rows'])if confirm_path.exists()else[]
    registry=[]
    for s in specs:
        i=s['id'];rows=[r for r in scores if r['id']==i and r['xi']>0]
        best=min(rows,key=lambda r:r['robust_score'])if rows else None
        rr=[r for r in refine if r['id']==i];cc=[r for r in confirm if r['id']==i]
        registry.append({**s,'screen_status':q.get(i,{}).get('status','numerical_positivity_certificate_failure'),
            'screen_quality':q.get(i),'screen_parents':12 if rows else 0,'screen_sizes':[512,2048]if rows else[],
            'best_screen_xi_descriptive':best['xi']if best else None,'best_screen_J':best['robust_score']if best else None,
            'screen_low_xis':[r['xi']for r in rows if r['candidate']],
            'screen_componentwise_xis':[r['xi']for r in rows if r['candidate']and r['componentwise_low']],
            'refinement_parents':24 if rr else 0,'refinement_sizes':[2048,8192]if rr else[],
            'refinement_componentwise_xis':[r['xi']for r in rr if r['eligible']and r['robust_score']<=r['epsilon']and r['componentwise_low']],
            'confirmation_parents':16 if cc else 0,
            'confirmation_componentwise_xis':[r['xi']for r in cc if r['sampler_allowed']and r['robust_score']<=r['epsilon']and r['componentwise']]})
    save_json(out/'all_covariances.json',dict(configurations=registry,scope='All121attempts,notonlysuccessfulorwinningconfigurations',
        matrices='Canonical sorted edge-correlation labels; coordinate-equivalent copies not new configurations'))
    write_csv(out/'all_covariances.csv',registry)
    write_csv(out/'complete_response_surface.csv',scores)
    write_csv(out/'selected_refinement.csv',refine)
    if confirm:write_csv(out/'independent_confirmation.csv',confirm)
    calibration=[]
    for xi in sorted({r['xi']for r in scores}):
        c=modelwise_conversion(xi);calibration.append(dict(xi=xi,gamma=c['gamma'],gamma_systematic=c.get('gamma_systematic'),
            dimension_proxy=c['gamma']/xi if xi else 3,lambda_cross=c.get('lambda'),
            bootstrap_subcritical_fraction=c.get('bootstrap_subcritical_fraction'),model_gammas=c.get('model_gammas'),
            bootstrap_gamma_interval=c.get('gamma_mc95'),scope='Finite-resolutioncalibration;notlimitingtheoremparameter'))
    save_json(out/'linked_parameter_table.json',calibration);write_csv(out/'linked_parameter_table.csv',calibration)
    # Coverage: show all failures/masked points, rather than silently selecting.
    f=plt.figure(figsize=(9,7));ax=f.add_subplot(projection='3d');eligible=[s for s in registry if s['screen_quality']and s['screen_quality']['sampler_allowed']]
    pts=np.array([s['triple']for s in eligible]);colors=[s['best_screen_xi_descriptive']for s in eligible]
    sc=ax.scatter(*pts.T,c=colors,cmap='viridis',vmin=.1,vmax=.3,s=35)
    old=np.array([s['triple']for s in registry if int(s['id'][1:])<33]);ax.scatter(*old.T,facecolors='none',edgecolors='black',s=65,label='Earlier33')
    adapt=np.array([s['triple']for s in registry if int(s['id'][1:])>=109]);ax.scatter(*adapt.T,facecolors='none',edgecolors='red',s=80,label='12adaptive')
    for status,marker,color in [('unsafe_excluded','x','gray'),('numerical_positivity_certificate_failure','X','red')]:
        xs=np.array([s['triple']for s in registry if s['screen_status']==status])
        if len(xs):ax.scatter(*xs.T,marker=marker,c=color,s=55,label=status.replace('_',' '))
    ax.set(xlabel='a (canonical)',ylabel='b (canonical)',zlabel='c (canonical)',title='121 attempted covariances; color = finite-screen argmin only')
    f.colorbar(sc,ax=ax,label='ξ at lowest provisional J',shrink=.65);ax.legend(fontsize=7,loc='upper left');f.savefig(fig/'covariance_coverage.png',dpi=180);plt.close(f)
    # Structured-family profiles expose wide support, not an interpolated curve.
    f,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,family in zip(axs,['birectangular','equicorrelated']):
        ss=[s for s in registry if family in s.get('coverage_families',[])];ss.sort(key=lambda s:s['triple'][0]if family=='equicorrelated'else sum(s['triple']))
        xis=sorted({r['xi']for r in scores if r['xi']>0});image=np.full((len(xis),len(ss)),np.nan)
        for j,s in enumerate(ss):
            for k,xi in enumerate(xis):
                row=next((r for r in scores if r['id']==s['id']and r['xi']==xi),None)
                if row and row['eligible']:image[k,j]=row['robust_score']
        im=ax.imshow(image,origin='lower',aspect='auto',vmin=.07,vmax=.4,cmap='magma_r',interpolation='nearest')
        ax.set_xticks(np.arange(len(ss))[::2],[f"{sum(s['triple'])/(3 if family=='equicorrelated'else 1):.2f}"for s in ss][::2],rotation=60)
        ax.set_yticks(range(len(xis)),[f'{x:g}'for x in xis]);ax.set(xlabel='ρ',title=family,ylabel='linked ξ')
    f.colorbar(im,ax=axs,label='Robust J; gray/missing = excluded',shrink=.8);f.savefig(fig/'structured_profiles.png',dpi=180,bbox_inches='tight');plt.close(f)
    f,ax=plt.subplots(figsize=(7,4));xx=[r['xi']for r in calibration if r['xi']];yy=[r['gamma']for r in calibration if r['xi']]
    lo=[r['gamma_systematic'][0]for r in calibration if r['xi']];hi=[r['gamma_systematic'][1]for r in calibration if r['xi']]
    ax.fill_between(xx,lo,hi,alpha=.2,label='Modelwise finite-window probes');ax.plot(xx,yy,'o-',label='Central linked proxy');ax.axhline(np.sqrt(6),ls='--',color='gray',label='√6')
    ax.set(xlabel='LFPP ξ',ylabel='GMC γ',title='Crossing-calibrated parameters: systematic envelope, not a CI');ax.legend(fontsize=8);f.tight_layout();f.savefig(fig/'linked_parameters.png',dpi=180);plt.close(f)
    # Finite-growth uses paired physical seeds at both sizes; no correction fit.
    rd=R/'families/refinement';dirs=[rd/'reused_graphs',rd/'new_a3']+sorted((rd/'new_graphs').glob('batch_*'))
    small=graph_groups(dirs,2048);large=graph_groups(dirs,8192);rng=np.random.default_rng(202611123990);growth=[]
    for i,rs in small.items():
        aa={r['header']['seed']:r for r in rs};bb={r['header']['seed']:r for r in large[i]};assert set(aa)==set(bb)
        seeds=sorted(aa);u=np.array([np.mean(aa[s]['root_means'])for s in seeds]);v=np.array([np.mean(bb[s]['root_means'])for s in seeds])
        vals=[]
        for _ in range(128):
            ix=rng.integers(len(u),size=len(u));vals.append(np.log(v[ix].mean()/u[ix].mean())/np.log(4))
        beta=np.log(v.mean()/u.mean())/np.log(4)
        growth.append(dict(id=i,parents=len(rs),beta=beta,beta_mc95=np.quantile(vals,[.025,.975]).tolist(),inverse_growth_proxy=1/beta,
            small_mean=u.mean(),large_mean=v.mean(),scope='Paired2048to8192finitegrowth;notasymptoticdimension'))
    save_json(out/'refined_growth.json',growth);write_csv(out/'refined_growth.csv',growth)
    rank=json.loads((R/'families/adaptive/local_feature_rank.json').read_text())['rows']
    rank_count=collections.Counter(str(r['singular_directions_above_conditional_noise'])for r in rank if r['status']=='descriptive_local_fit')
    d=json.loads((R/'families/adaptive/direct_graph_levelsets.json').read_text())
    summary=dict(attempted=121,successful=len(q),unsafe=sum(not v['sampler_allowed']for v in q.values()),
        status_counts=dict(collections.Counter(s['screen_status']for s in registry)),
        rank_field_names=list(next(r for r in rank if r['status']=='descriptive_local_fit')),
        rank_counts=rank_count,levelset_rows=len(d['rows']),levelset_pairs=d['pairs'],
        levelset_beyond_noise=sum(not r['point_within_reference_noise']for r in d['rows']),
        field_parents_discovery=48,field_parents_independent_confirmation=24 if confirm else 0,
        adaptive_successes=11,adaptive_failures=['C116'],no_new_physical_simulation=True)
    save_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
