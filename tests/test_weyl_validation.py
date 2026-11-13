"""Independent controls for the special A2 (angle pi/3) cone extension."""

import numpy as np
from scipy import stats
from src.weyl_excursions import sample_weyl_excursion, weyl_junction_direction, weyl_covariance


def wedge_angular_cdf(theta):
    width = np.pi/3
    theta = np.clip(theta,0,width)
    return theta/width-np.sin(2*np.pi*theta/width)/(2*np.pi)


def test_weyl_junction_radial_identity_and_angular_density():
    rng = np.random.default_rng(503)
    angles = []
    for _ in range(4000):
        eigenvalues, height = weyl_junction_direction(2,rng)
        gaps = np.diff(eigenvalues)/np.sqrt(2)
        whitened = np.linalg.solve(np.linalg.cholesky(weyl_covariance(2)),gaps)
        np.testing.assert_allclose(np.sum(whitened**2),1,atol=2e-14)
        angles.append(np.arctan2(whitened[1],whitened[0])-np.pi/6)
        assert height == 0
    # Density is proportional sin²(3 theta), independently derived in a wedge.
    assert stats.kstest(angles,wedge_angular_cdf).statistic < .035


def test_weyl_times_positive_axis_spherical_marginal():
    rng = np.random.default_rng(504)
    extra_squared = []
    for _ in range(4000):
        eigenvalues,height = weyl_junction_direction(3,rng)
        np.testing.assert_allclose(np.sum(eigenvalues**2)+height**2,1,atol=2e-14)
        extra_squared.append(height**2)
    # A2 contributes radial chi²8; the positive axis contributes chi²3.
    assert stats.kstest(extra_squared,"beta",args=(1.5,4)).statistic < .035


def test_unbiased_weyl_midpoint_radial_laws():
    for dimension,degrees,seed in [(2,8,505),(3,11,506)]:
        rng = np.random.default_rng(seed)
        inverse = np.linalg.inv(weyl_covariance(dimension))
        radius = []
        for _ in range(2500):
            sample = sample_weyl_excursion(4,dimension,rng,law="unbiased",oversample=1)
            midpoint = sample.values[2]
            radius.append(float(midpoint @ inverse @ midpoint/.25))
        assert stats.kstest(radius,"chi2",args=(degrees,)).statistic < .04


def test_weyl_quadratic_variation_correlations():
    rng = np.random.default_rng(507)
    sample = sample_weyl_excursion(16384,3,rng,law="unbiased",oversample=1)
    increments = np.diff(sample.values,axis=0)[3277:13107]
    observed = increments.T @ increments/(len(increments)/16384)
    np.testing.assert_allclose(observed,weyl_covariance(3),atol=.05)


def test_biased_weyl_junction_values_and_hitting_parameter():
    rng = np.random.default_rng(508)
    for dimension,alpha in [(2,3),(3,4.5)]:
        sample = sample_weyl_excursion(32,dimension,rng,law="biased",oversample=4)
        row = np.searchsorted(sample.times,sample.metadata["junction_time"])
        expected = np.array(sample.metadata["junction_direction_correlated"])
        expected /= np.sqrt(np.sum(sample.metadata["hitting_times"]))
        np.testing.assert_allclose(sample.values[row],expected,atol=2e-14)
        assert sample.metadata["alpha1"] == alpha
        assert sample.metadata["cell_minima_exact"] is False
        assert np.all(sample.values[1:-1] > 0)
