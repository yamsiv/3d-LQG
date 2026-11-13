"""Independent analytical checks for probability-measure sampling summaries."""
import numpy as np

from src.metric_measure_3a import (
    evaluate_cdf, ensemble_median, matrix_features, weighted_root_cdfs,
)


def test_ensemble_median_gives_realizations_equal_weight_with_unequal_roots():
    # One field's conditional pair distance is 1, another's is 3. Repeating
    # the roots of either field must not change the equal-field mixture.
    a = {"grid": np.array([0., 1., 2., 3.]),
         "cdfs": np.array([[0., 1., 1., 1.]]), "discrete": True}
    b = {"grid": np.array([0., 1., 2., 3.]),
         "cdfs": np.tile([0., 0., 0., 1.], (19, 1)), "discrete": True}
    assert abs(ensemble_median([a, b])-1) < 1e-12
    a["cdfs"] = np.repeat(a["cdfs"], 31, axis=0)
    assert abs(ensemble_median([a, b])-1) < 1e-12


def test_continuous_cdf_interpolation_is_bracketed_by_one_distance_bin():
    rng = np.random.default_rng(98017)
    distances = rng.exponential(size=(4, 37))
    distances[np.arange(4), np.arange(4)] = 0.
    probabilities = rng.dirichlet(np.ones(37))
    record = weighted_root_cdfs(distances, probabilities, bins=51)
    points = np.linspace(0., distances.max()+.1, 231)
    width = record["distance_bin_width"]
    estimates = evaluate_cdf(record, points)
    for root, distance in enumerate(distances):
        lower = (probabilities[None, :]*(distance[None, :] <= points[:, None]-width)).sum(axis=1)
        upper = (probabilities[None, :]*(distance[None, :] <= points[:, None]+width)).sum(axis=1)
        assert np.all(estimates[root] >= lower-2e-15)
        assert np.all(estimates[root] <= upper+2e-15)
    np.testing.assert_allclose(evaluate_cdf(record, [0.])[:, 0], probabilities[:4])


def test_triangle_sampling_retains_two_coincident_locations():
    # Samples indexed 0 and 1 coincide. This valid measure-sampled triangle
    # has side ratio (0,1,1), rather than being silently removed.
    matrix = np.array([[0., 0., 2.], [0., 0., 2.], [2., 2., 0.]])
    result = matrix_features(matrix)
    np.testing.assert_allclose(result["small_over_large"], [0.])
    np.testing.assert_allclose(result["middle_over_large"], [1.])
    assert result["degenerate_triangle_fraction"] == 0
    assert result["zero_pair_fraction"] == 1/3


def test_crossing_regression_sign_and_coupled_bootstrap_on_exact_power():
    from experiments.phase3a_crossing_analysis import fit_lambda
    sizes = np.array([16, 24, 32])
    amplitudes = np.array([.5, .7, 1., 2., 8.])
    powers = np.array([[0., .2, .35], [.3, .45, .6]])
    crossings = amplitudes[:, None, None, None]*sizes[None, :, None, None]**(-powers[None, None, :, :])
    np.testing.assert_allclose(fit_lambda(crossings, sizes, [0, 1, 2]), powers, atol=1e-14)
    # Resampling complete rows preserves a realization's random amplitude
    # across resolutions and thus exactly preserves these known exponents.
    np.testing.assert_allclose(fit_lambda(crossings[[4, 4, 2, 0, 1]], sizes, [0, 1, 2]), powers, atol=1e-14)


def test_implicit_gamma_conversion_checks_subcritical_discriminant():
    from experiments.phase3a_crossing_analysis import implicit_quantities
    gamma, xi = .7, .17
    q = 3/gamma+gamma/2
    result = implicit_quantities(1-xi*q, xi)
    np.testing.assert_allclose(result, [q, gamma, gamma/xi], rtol=1e-14)
    assert implicit_quantities(.5, .4)[1:] == (None, None)
    assert implicit_quantities(0., 0.) == (None, None, None)
