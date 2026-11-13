import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.metric_measure_3a import weighted_root_cdfs
from src.metric_features_3b import (
    feature_vector, grouped_feature_distance, matrix_shape_samples,
    split_noise_scales, bootstrap_feature_difference,
)


def record(scale=1., *, realization=0, discrete=False):
    positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [.4, .2]])
    d = cdist(positions, positions) * scale
    if discrete:
        d = np.array([[0, 1, 2, 3, 2], [1, 0, 1, 2, 1],
                      [2, 1, 0, 1, 2], [3, 2, 1, 0, 3], [2, 1, 2, 3, 0]], float) * scale
    data = weighted_root_cdfs(d, np.full(5, .2), discrete=False)
    if discrete:
        support = np.unique(d)
        data.update(grid=support, cdfs=np.array([(row[:, None] <= support).mean(axis=0) for row in d]))
    return dict(data, matrix=d, discrete=discrete, realization=realization)


def ensemble():
    return [record(1 + .05*i, realization=i) for i in range(8)]


def test_rich_features_are_scale_invariant_and_layout_stable():
    a = feature_vector(ensemble())
    b = feature_vector([record(7*(1+.05*i), realization=i) for i in range(8)])
    assert len(a['vector']) == 47
    assert a['names'] == b['names']
    np.testing.assert_array_equal(a['valid'], b['valid'])
    np.testing.assert_allclose(a['vector'], b['vector'], atol=2e-12)
    assert grouped_feature_distance(a, b)['total'] < 1e-11
    assert not any(name.startswith('ball_mean') for name in a['names'])


def test_four_point_tree_identity_and_euclidean_nonzero_delta():
    line = np.arange(4)[:, None]
    tree = matrix_shape_samples(cdist(line, line))
    np.testing.assert_allclose(tree['quartets'][:, 0], 0)
    square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    d = cdist(square, square)
    shape = matrix_shape_samples(d)
    np.testing.assert_allclose(shape['quartets'][0, 0], 1-1/np.sqrt(2))
    perm = [2, 0, 3, 1]
    permuted = matrix_shape_samples(d[np.ix_(perm, perm)])
    for name in ['quartets', 'triangles']:
        np.testing.assert_allclose(np.sort(shape[name], axis=0), np.sort(permuted[name], axis=0))
    for name in ['shared_product', 'disjoint_product']:
        assert permuted[name] == pytest.approx(shape[name])


def test_collisions_and_undefined_shapes_are_kept_explicit():
    shape = matrix_shape_samples(np.zeros((4, 4)))
    assert shape['triangle_degenerate'] == shape['quartet_degenerate'] == 1
    assert len(shape['triangles']) == len(shape['quartets']) == 0
    r = record()
    r['matrix'] = np.zeros((5, 5))
    result = feature_vector([r, r])
    for index, name in enumerate(result['names']):
        if name.startswith(('triangles_', 'quartets_', 'matrix_')):
            assert not result['valid'][index]
    assert result['vector'][result['names'].index('triangle_degenerate')] == 1


def test_unit_mass_weyl_gauge_is_an_ensemble_rescaling_not_conditioning():
    records = ensemble()
    for i, r in enumerate(records):
        r.update(xi=.3, gamma=1.5, log_total_mass=.1*i)
    transformed = []
    for r in records:
        factor = np.exp(-.3*r['log_total_mass']/1.5)
        t = dict(r)
        for name in ('grid', 'root_means', 'matrix'):
            t[name] = np.asarray(r[name])*factor
        transformed.append(t)
    expected = feature_vector(transformed, gauge='unit_mass', already_transformed=True)
    actual = feature_vector(records, gauge='unit_mass')
    np.testing.assert_allclose(actual['vector'], expected['vector'])
    assert actual['diagnostics']['gauge'] == 'unit_mass'
    bad = dict(records[0], gamma=0., xi=.2)
    with pytest.raises(ValueError, match='gamma=0'):
        feature_vector([bad], gauge='unit_mass')


def test_noise_calibration_is_positive_and_resamples_whole_realizations():
    records = ensemble()
    fit = split_noise_scales(records, np.random.default_rng(42), draws=4)
    assert fit['split_size'] == 4
    assert fit['differences'].shape == (4, 47)
    assert np.all(fit['scales'] >= fit['floor'])
    assert np.all(fit['scales'] > 0)
    paired = bootstrap_feature_difference(records, records, np.random.default_rng(21),
                                         draws=4, paired=True)
    np.testing.assert_allclose(paired['point'], 0)
    np.testing.assert_allclose(paired['mc95'][:, paired['valid']], 0)
    with pytest.raises(ValueError, match='matching physical'):
        bootstrap_feature_difference(records, records[::-1], np.random.default_rng(21),
                                     draws=4, paired=True)


def test_ensemble_graph_atom_floor_and_missing_layout_rejection():
    features = feature_vector([record(discrete=True), record(discrete=True)])
    # Each root has mass 1/5 at zero; the ensemble atom at distance 1 is 8/25.
    assert features['diagnostics']['largest_ensemble_graph_atom'] == pytest.approx(.32)
    assert features['diagnostics']['continuous_cdf_ks_atom_floor'] == pytest.approx(.16)
    malformed = dict(features, names=features['names'][::-1])
    with pytest.raises(ValueError, match='layouts'):
        grouped_feature_distance(features, malformed)
    with pytest.raises(ValueError, match='nonnegative'):
        grouped_feature_distance(features, features, weights={'pair': -1})
