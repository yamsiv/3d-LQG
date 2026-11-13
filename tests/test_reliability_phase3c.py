import numpy as np
import pytest

from experiments.phase3c_reliability import (aligned_minima, grid_thinning,
    pair_discrepancy, graph_record, metric_features, minima_features)
from src.brownian_excursions import sample_independent_excursion
from src.mated_crt import cell_minima


def test_minimum_labels_align_with_actual_covariance():
    source = np.array([[1, -.5, 0], [-.5, 1, -.5], [0, -.5, 1.]])
    perm = [1, 0, 2]
    target = source[np.ix_(perm, perm)]
    values = np.arange(24).reshape(8, 3)
    result = aligned_minima(values, source, target)
    np.testing.assert_array_equal(result, values[:, perm])
    with pytest.raises(ValueError, match="permutation"):
        aligned_minima(values, source, np.eye(3))


def test_saved_cell_minima_coarsen_without_new_Brownian_sampling():
    sample = sample_independent_excursion(32, 3, np.random.default_rng(9001), oversample=8)
    fine = cell_minima(sample, 32)
    np.testing.assert_allclose(fine.reshape(8, 4, 3).min(axis=1), cell_minima(sample, 8))
    u = np.random.default_rng(19).random(16)
    np.testing.assert_array_equal((u*2048).astype(int)//4, (u*512).astype(int))


def test_same_polygonal_path_thinning_has_nested_minima():
    sample = sample_independent_excursion(32, 3, np.random.default_rng(9002), oversample=64)
    coarse = grid_thinning(sample, 32, 2)
    finer = grid_thinning(sample, 32, 8)
    # Compare against the retained polygonal path, not its extra exact minima.
    sample.segment_minima = None
    assert np.all(cell_minima(coarse, 32) >= cell_minima(finer, 32)-1e-14)
    assert np.all(cell_minima(finer, 32) >= cell_minima(sample, 32)-1e-14)


def test_mean_normalized_Wasserstein_matches_independent_atom_calculation():
    a = {"grid": np.array([0., 1., 2.]), "cdfs": np.array([[0., .5, 1.]]),
         "root_means": np.array([1.5]), "discrete": True}
    b = {"grid": np.array([0., 1.]), "cdfs": np.array([[0., 1.]]),
         "root_means": np.array([1.]), "discrete": True}
    result = pair_discrepancy([a], [b])
    assert result["W1"] == pytest.approx(1/3)
    assert result["KS"] == pytest.approx(.5)


def test_mean_feature_normalization_is_explicit():
    sample = sample_independent_excursion(32, 3, np.random.default_rng(9003))
    record = graph_record(cell_minima(sample, 32), np.arange(16))
    result = metric_features([record, record])
    assert result["pair_mean_over_mean"] == pytest.approx(1.)
    assert "pair_mean_over_median" not in result


def test_rank_copula_is_invariant_under_coordinatewise_order_preserving_changes():
    minima = np.array([[0, 0, 0], [.1, .4, .2], [.3, .2, .6], [.2, .1, .5],
        [.4, .3, .1], [.5, .5, .4], [0, 0, 0]])
    a, b = minima_features(minima), minima_features(minima**3)
    for name in a:
        if name.startswith("copula_") or name.startswith("turning_fraction_"):
            assert a[name] == b[name]
