import numpy as np
import pytest
from src.log_field_3d import sample_coupled_fields, sample_field, spectral_configuration, resample_field


def test_deterministic_covariance_fft_and_unit_log_normalization():
    n = 16
    spectrum, covariance, variance = spectral_configuration(n)
    assert spectrum[0, 0, 0] == 0
    assert spectrum[1, 0, 0] == pytest.approx(1/(4*np.pi))
    assert covariance[0, 0, 0] == pytest.approx(variance)
    assert abs(covariance.mean()) < 1e-16
    assert np.fft.fftn(covariance).real/n**3 == pytest.approx(spectrum, abs=1e-16)
    assert variance == pytest.approx(spectrum.sum())


def test_common_low_modes_at_non_nested_grid_sizes():
    fields = sample_coupled_fields([16, 24, 32], np.random.default_rng(9))
    coefficients = {n: np.fft.fftn(f.h)/n**3 for n, f in fields.items()}
    for k in [(1, 0, 0), (2, -2, 1), (-3, 1, 2)]:
        assert coefficients[16][tuple(t % 16 for t in k)] == pytest.approx(
            coefficients[24][tuple(t % 24 for t in k)], abs=5e-16)
    for f in fields.values():
        assert abs(f.h.mean()) < 1e-14
        assert f.metadata["imaginary_roundoff_max"] < 1e-14


def test_sample_covariance_matches_independent_fourier_sum():
    # Independent realizations are the units; no treating spatial sites as iid.
    rng = np.random.default_rng(123)
    values = np.array([sample_field(8, rng).h[[0, 1], 0, 0] for _ in range(1600)])
    _, covariance, variance = spectral_configuration(8)
    assert np.mean(values[:, 0]**2) == pytest.approx(variance, abs=.12)
    assert np.mean(values[:, 0]*values[:, 1]) == pytest.approx(covariance[1, 0, 0], abs=.10)


def test_heat_reality_reproducibility_and_invalid_settings():
    a = sample_field(12, np.random.default_rng(21), regularization="heat")
    b = sample_field(12, np.random.default_rng(21), regularization="heat")
    assert np.array_equal(a.h, b.h)
    assert a.metadata["imaginary_roundoff_max"] < 1e-14
    with pytest.raises(ValueError): sample_field(12, cutoff_fraction=.5)
    with pytest.raises(ValueError): sample_field(12, regularization="bad")


def test_same_bandlimited_field_quadrature_refinement():
    f = sample_field(16, np.random.default_rng(2))
    refined = resample_field(f, 32)
    assert refined.h[::2, ::2, ::2] == pytest.approx(f.h, abs=1e-14)
    assert refined.variance == f.variance
    assert refined.spectrum.sum() == pytest.approx(f.variance)
