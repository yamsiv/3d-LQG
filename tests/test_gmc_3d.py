import numpy as np
import pytest
from src.gmc_3d import gmc_weights, coarse_cell_masses, moment_threshold, euclidean_mass_exponent


def test_zero_gamma_uniform_mass_and_coarse_conservation():
    h = np.random.default_rng(11).normal(size=(8,)*3)
    measure = gmc_weights(h, 0., 4.)
    assert measure.total_mass == pytest.approx(1.)
    assert measure.probabilities == pytest.approx(np.full_like(h, 1/h.size))
    assert coarse_cell_masses(measure, 2) == pytest.approx(np.full((2,)*3, 1/8))


def test_constant_shift_and_stability_preserve_unnormalized_mass():
    h = np.arange(64).reshape(4, 4, 4)/10
    a = gmc_weights(h, 2., 3.)
    b = gmc_weights(h+100., 2., 3.)
    assert b.log_total_mass-a.log_total_mass == pytest.approx(200.)
    assert b.probabilities == pytest.approx(a.probabilities)
    assert a.probabilities.sum() == pytest.approx(1.)
    assert np.exp(a.log_cell_masses).sum() == pytest.approx(a.total_mass)


def test_gaussian_variance_renormalization_expectation_and_moment_conventions():
    # Gaussian exponential identity, independently of a Fourier implementation.
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    variance = 2.3
    value = sum(w*gmc_weights(np.full((1, 1, 1), np.sqrt(2*variance)*x), 1.5,
                              variance).total_mass for x, w in zip(nodes, weights))/np.sqrt(np.pi)
    assert value == pytest.approx(1., abs=1e-12)
    assert moment_threshold(2.) == 1.5
    assert np.isinf(moment_threshold(0.))
    assert euclidean_mass_exponent(1., 2.) == 3.
    assert euclidean_mass_exponent(2., 1.) == 5.
