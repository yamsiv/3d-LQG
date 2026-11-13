from itertools import permutations, product

import numpy as np
from scipy.spatial.distance import cdist

from src.metric_measure_3a import weighted_root_cdfs
from src.family_search_3c import (
    ensemble_features, calibrate_split_tolerance, compare_features,
    observed_family_components, local_feature_rank,
    bootstrap_ensemble_features, component_residual_profiles,
    direct_graph_levelset_comparisons, propose_family_adaptation,
    sampled_matrix_panels, signed_matrix_energy, matrix_energy_comparison,
)


def records():
    positions=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.4,.2]])
    base=cdist(positions,positions)
    rows=[]
    for index in range(8):
        d=base*(1+.04*index)
        data=weighted_root_cdfs(d,np.full(5,.2))
        rows.append(dict(data,matrix=d,discrete=True,realization=index))
    return rows


def test_mean_primary_is_scale_invariant_and_median_remains_a_sensitivity():
    source=records()
    a=ensemble_features(source)
    modified=[]
    for r in source:
        copy=dict(r)
        for key in ['grid','matrix','root_means']:copy[key]=7*np.asarray(r[key])
        modified.append(copy)
    b=ensemble_features(modified)
    np.testing.assert_allclose(a['vector'],b['vector'],atol=1e-11)
    assert a['diagnostics']['normalization_rule']=='mean'
    assert a['vector'][a['names'].index('pair_mean_over_median')]==1.
    assert ensemble_features(source,normalization='median')['diagnostics']['normalization_rule']=='median'


def test_frozen_tolerance_is_max_cohort_quantile_and_gate_is_explicit():
    source=records()
    fit=calibrate_split_tolerance({'anchor':source},np.random.default_rng(42),draws=4)
    assert fit['sample_count_rescaling']is False
    entry=fit['results']['grounded:mean']
    values=entry['cohorts']['anchor']['scores']['primary']
    assert entry['epsilon']['primary']==np.quantile(values,.95)
    a=ensemble_features(source)
    unknown=compare_features(a,a,fit)
    assert unknown['point_low_without_gate']is True
    assert unknown['candidate']is False
    allowed=compare_features(a,a,fit,sampler_allowed=True)
    assert allowed['candidate']is True
    assert not any(allowed['family_flags'].values())


def test_observed_adjacency_does_not_jump_over_high_score_or_failed_nodes():
    nodes=[dict(id='C0',triple=[-.3,0,.2],xi=x,gamma=1.,score=j,eligible=True)
           for x,j in[(.2,.01),(.225,1.),(.25,.01)]]
    result=observed_family_components(nodes,epsilon=.1)
    assert len(result['components'])==2
    assert result['edges']==[]
    nodes[1]['score']=.01;nodes[1]['eligible']=False
    assert len(observed_family_components(nodes,epsilon=.1)['components'])==2


def test_finite_family_requires_observed_covariance_and_xi_extent():
    nodes=[]
    for i in range(4):
        for xi in [.15,.175,.2]:
            nodes.append(dict(id=f'C{i}',triple=[-.3+.03*i,0.,.2],xi=xi,
                              gamma=3*xi,score=.02,eligible=True))
    result=observed_family_components(nodes,epsilon=.1)
    assert len(result['components'])==1
    assert result['components'][0]['observed_family_support']is True
    singleton=observed_family_components(nodes[:3],epsilon=.1)
    assert singleton['components'][0]['observed_family_support']is False


def test_local_rank_uses_covariance_directions_and_checks_field_tangent():
    center=np.array([-.25,0.,.3])
    c=np.vstack([center,center+np.array(list(product([-.05,.05],repeat=3)))])
    tangent=np.array([1.,-2.,.5])
    scalar=c@np.array([1.,2.,3.])
    y=scalar[:,None]*tangent
    fit=local_feature_rank(c,y,0,np.ones(3),radius=.15,field_tangent=tangent,
                           bootstrap_vectors=np.repeat(y[None,:,:],4,axis=0))
    assert fit['status']=='descriptive_local_fit'
    assert fit['singular_values'][0]>1
    np.testing.assert_allclose(fit['singular_values'][1:],0,atol=1e-12)
    np.testing.assert_allclose(fit['transverse_to_field_tangent_singular_values'],0,atol=1e-12)
    assert fit['near_permutation_quotient_wall']is False
    assert fit['singular_directions_above_conditional_noise']==1
    collinear=np.column_stack((np.arange(7),np.zeros(7),np.zeros(7)))
    assert local_feature_rank(collinear,np.ones((7,3)),3,np.ones(3),radius=8)['status']=='insufficient_covariance_directions'


def test_whole_parent_bootstrap_accepts_common_indices_and_recomputes_scale():
    source=records()
    indices=np.vstack([np.zeros(len(source),int),np.full(len(source),7,int)])
    fit=bootstrap_ensemble_features(source,indices=indices)
    np.testing.assert_array_equal(fit['indices'],indices)
    # A repeated single scale has no between-parent scale CV in either draw.
    i=fit['point']['names'].index('root_mean_distance_cv')
    np.testing.assert_allclose(fit['vectors'][:,i],0,atol=1e-12)


def test_same_sign_flagged_residuals_are_not_rewarded_as_a_match():
    nodes=[dict(id=f'C{i}',signed_residuals=[.5],valid=[True],feature_flags=[True])for i in range(4)]
    profile=component_residual_profiles(nodes,[{'indices':list(range(4))}],['shape'],[1.])[0]
    assert profile['persistent_names']==['shape']
    assert 'oppose agreement'in profile['qualification']


def test_direct_graph_check_uses_graph_only_reference_noise():
    source=records()
    calibration=calibrate_split_tolerance({'graph_anchor':source},np.random.default_rng(1),draws=2)
    feature=ensemble_features(source)
    nodes=[dict(id='A',triple=[-.5,0,0],xi=.2,score=.01,eligible=True),
           dict(id='B',triple=[0,0,0],xi=.2,score=.01,eligible=True)]
    result=direct_graph_levelset_comparisons(nodes,{'A':feature,'B':feature},calibration,epsilon=.1)
    assert result['pairs']==1
    assert result['rows'][0]['score']==0
    assert result['rows'][0]['point_within_reference_noise']is True


def test_adaptive_design_covers_categories_and_never_claims_predicted_matches():
    existing=[];nodes=[]
    for i,a in enumerate([-.45,-.3,-.15,0.]):
        triple=[a,.05,.35]
        existing.append(dict(id=f'C{i:03d}',triple=triple))
        for xi in [.125,.15,.175,.2125,.225,.2375,.275,.2875,.3]:
            nodes.append(dict(id=f'C{i:03d}',triple=triple,xi=xi,score=.04,eligible=True,
                              family_flags={'quartets':i%2==0},signed_residuals=[i*.02]))
    result=propose_family_adaptation(nodes,existing,epsilon=.1,count=12,seed=43)
    assert len(result['configurations'])==12
    assert result['no_global_argmin_optimization']is True
    assert result['actual_category_counts']['maximin_coverage']>=2
    for r in result['configurations']:
        assert r['acquisition']['predicted_score']is None
        assert r['condition']<=12
        assert r['eigenvalues'][0]>=.15


def test_matrix_energy_symmetry_common_relabeling_and_negative_estimates():
    source=records()
    a=sampled_matrix_panels(source[:4],points=5)['vectors']
    b=sampled_matrix_panels(source[4:],points=5)['vectors']
    point=signed_matrix_energy(a,b)
    assert point['energy_u']==signed_matrix_energy(b,a)['energy_u']
    # Same empirical panels on both sides need not have nonnegative U-form.
    assert signed_matrix_energy(a,a)['energy_u']<0
    permutation=[3,1,4,0,2]
    permuted=[]
    for row in source:
        r=dict(row,matrix=row['matrix'][np.ix_(permutation,permutation)])
        permuted.append(r)
    pa=sampled_matrix_panels(permuted[:4],points=5)['vectors']
    pb=sampled_matrix_panels(permuted[4:],points=5)['vectors']
    np.testing.assert_allclose(signed_matrix_energy(pa,pb)['energy_u'],point['energy_u'],atol=1e-14)


def test_matrix_energy_recomputes_ensemble_scale_inside_parent_bootstrap():
    left,right=records()[:4],records()[4:]
    baseline=matrix_energy_comparison(left,right,np.random.default_rng(4),points=5,draws=4)
    changed=[]
    for row in left:
        changed.append(dict(row,matrix=11*row['matrix'],root_means=11*row['root_means']))
    scaled=matrix_energy_comparison(changed,right,np.random.default_rng(4),points=5,draws=4)
    np.testing.assert_allclose(baseline['energy_u'],scaled['energy_u'],atol=1e-14)
    np.testing.assert_allclose(baseline['bootstrap_energies'],scaled['bootstrap_energies'],atol=1e-14)


def test_complete_exchangeable_label_orbits_are_invariant_to_separate_relabeling():
    left=cdist(np.arange(4)[:,None],np.arange(4)[:,None])
    right=np.ones((4,4))-np.eye(4)
    indices=np.triu_indices(4,1)
    orbit=lambda matrix:np.array([matrix[np.ix_(p,p)][indices]for p in permutations(range(4))])
    baseline=signed_matrix_energy(orbit(left),orbit(right))['energy_u']
    p,q=[3,0,2,1],[2,1,3,0]
    changed=signed_matrix_energy(orbit(left[np.ix_(p,p)]),orbit(right[np.ix_(q,q)]))['energy_u']
    assert changed==baseline
