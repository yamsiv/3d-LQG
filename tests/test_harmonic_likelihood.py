import numpy as np
import pytest

from src.harmonic_likelihood import (sphere_rule, log_mean_value_ratios,
    chain_log_likelihood, conditional_log_likelihood_bound)


def test_quadrature_and_harmonic_cubic_mean_value():
    directions, weights = sphere_rule(8)
    assert abs(weights.sum()-1) < 1e-14
    assert np.max(abs(weights @ directions)) < 1e-14
    centers = np.array([[1., 2., 3.], [.01, .02, 1.]])
    logh = lambda x: np.log(np.prod(x, axis=1))
    values = log_mean_value_ratios(logh, centers, np.array([.4, .004]), 8)
    assert np.max(abs(values)) < 1e-13


def test_nonharmonic_quadratic_has_known_mean_value_defect():
    centers = np.array([[1., 2., 3.], [2., 1., 1.]])
    radii = np.array([.3, .4])
    logh = lambda x: np.log(1+np.sum(x*x, axis=1))
    values = log_mean_value_ratios(logh, centers, radii, 8)
    exact = np.log1p(radii*radii/(1+np.sum(centers*centers, axis=1)))
    assert np.max(abs(values-exact)) < 1e-14


def test_likelihood_telescopes_in_explicit_finite_markov_chain():
    # Ordinary exit laws on a synthetic three-point state space; h=1 is
    # harmonic. Direct multiplication must equal endpoint/normalizer formula.
    base = np.array([[.2, .5, .3], [.4, .1, .5], [.1, .3, .6]])
    htilde = np.array([.8, 1.3, 1.1]); normalizers = base @ htilde
    chain = [0, 1, 2, 1, 0, 2]
    direct = sum(np.log(htilde[y]/normalizers[x]) for x, y in zip(chain[:-1], chain[1:]))
    logs = np.log(normalizers[chain[:-1]]/htilde[chain[:-1]])
    result = chain_log_likelihood(np.log(htilde[chain[0]]), np.log(htilde[chain[-1]]), logs)
    assert abs(result-direct) < 1e-14
    bound = conditional_log_likelihood_bound(.3, np.sum(abs(logs)))
    assert abs(result) <= bound


def test_bounds_require_valid_hypotheses():
    with pytest.raises(ValueError): conditional_log_likelihood_bound(1., .1)
    with pytest.raises(ValueError): conditional_log_likelihood_bound(.1, -.1)
