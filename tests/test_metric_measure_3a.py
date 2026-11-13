import numpy as np

from src.metric_measure_3a import (weighted_root_cdfs, evaluate_cdf,
    ensemble_median, summarize, discrepancy, bootstrap_comparison, matrix_features)


def record(scale=1.):
    distances=scale*np.array([[0,1,2],[1,0,1],[2,1,0]],float)
    data=weighted_root_cdfs(distances,np.ones(3)/3,discrete=False)
    return dict(data,matrix=distances,discrete=False)


def test_weighted_atoms_and_discrete_right_continuity():
    d=np.array([[0,1,2],[1,0,1]])
    data=weighted_root_cdfs(d,[.2,.3,.5],discrete=True)
    np.testing.assert_allclose(data['cdfs'][:,0],[.2,.3])
    np.testing.assert_allclose(evaluate_cdf(dict(data,discrete=True),[.5,1,2]),[[.2,.5,1],[.3,1,1]])


def test_deterministic_scale_invariance_and_realization_bootstrap():
    a=[record(),record(1.2)]
    b=[record(4),record(4.8)]
    np.testing.assert_allclose(summarize(a)['cdf'],summarize(b)['cdf'],atol=1e-12)
    assert discrepancy(summarize(a),summarize(b))['objective']<1e-12
    fit=bootstrap_comparison(a,b,np.random.default_rng(31),draws=8)
    assert fit['mc95']['pair_w1'][0]>=0
    assert abs(ensemble_median(b)/ensemble_median(a)-4)<1e-12


def test_distance_matrices_retain_coincident_samples():
    features=matrix_features(np.zeros((4,4)))
    assert features['degenerate_triangle_fraction']==1
    assert features['zero_pair_fraction']==1
    features=matrix_features(np.array([[0,1,2],[1,0,1],[2,1,0]]))
    np.testing.assert_allclose(features['small_over_large'],[.5])
