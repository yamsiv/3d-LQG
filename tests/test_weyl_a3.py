"""Independent algebra and probabilistic controls for the A3 benchmark."""
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import pytest

from src.covariance import to_angles
from src.mated_crt import graph_from_excursion, validate_graph
from src.weyl_a3 import (a3_covariance, a3_polynomial, a3_hermitian,
                         a3_junction_direction, a3_log_u_white,
                         sample_a3_excursion, thin_a3_sample, A3AnalyticCone)


def _multiply(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items():
            key = tuple(i+j for i, j in zip(a, b))
            out[key] = out.get(key, 0)+x*y
    return {k: v for k, v in out.items() if v}


def _polynomial():
    axes = [{tuple(int(i == j) for i in range(3)): Fraction(1)} for j in range(3)]
    factors = axes + [axes[0] | axes[1], axes[1] | axes[2], axes[0] | axes[1] | axes[2]]
    result = {(0,0,0): Fraction(1)}
    for factor in factors:
        result = _multiply(result, factor)
    return result


def test_covariance_generator_and_angles():
    # The gap map is derived directly from four independent particles.
    difference = np.diff(np.eye(4), axis=0)/np.sqrt(2)
    np.testing.assert_allclose(difference@difference.T, a3_covariance(), atol=3e-16)
    np.testing.assert_allclose(to_angles(a3_covariance()), [np.pi/3, np.pi/3, np.pi/2])
    assert np.linalg.eigvalsh(a3_covariance()).min() > 0


def test_exact_rational_harmonicity():
    polynomial = _polynomial()
    covariance = [[Fraction(1), Fraction(-1,2), Fraction(0)],
                  [Fraction(-1,2), Fraction(1), Fraction(-1,2)],
                  [Fraction(0), Fraction(-1,2), Fraction(1)]]
    generator = {}
    for powers, coefficient in polynomial.items():
        assert sum(powers) == 6
        for i in range(3):
            for j in range(3):
                factor = powers[i]*(powers[j]-int(i == j))
                if factor:
                    key = tuple(power-int(k == i)-int(k == j) for k, power in enumerate(powers))
                    generator[key] = generator.get(key, 0)+coefficient*factor*covariance[i][j]/2
    assert all(value == 0 for value in generator.values())
    assert 6*(6+3-2) == 42
    assert math.sqrt(42+.25) == 6.5


def test_exact_angular_normalization_by_wick_moments():
    # Independent rational Gaussian moment calculation, no angular solver.
    covariance = [[Fraction(str(v)) for v in row] for row in a3_covariance()]
    @lru_cache(None)
    def moment(powers):
        if not sum(powers):
            return Fraction(1)
        if sum(powers) % 2:
            return Fraction(0)
        i = next(k for k, p in enumerate(powers) if p)
        rest = list(powers)
        rest[i] -= 1
        value = Fraction(0)
        for j, count in enumerate(rest):
            if count:
                other = rest.copy()
                other[j] -= 1
                value += count*covariance[i][j]*moment(tuple(other))
        return value
    squared = _multiply(_polynomial(), _polynomial())
    gaussian = sum(coefficient*moment(power) for power, coefficient in squared.items())
    assert gaussian == Fraction(9, 2)
    # Weyl group has 24 chambers; radial density is r^14 exp(-r²/2).
    radial = 2**6.5*math.gamma(7.5)
    angular_integral = float(gaussian)*(2*np.pi)**1.5/(24*radial)
    assert angular_integral == pytest.approx(np.pi/180180, rel=2e-15)


def test_matrix_metric_vandermonde_and_angular_directions():
    rng = np.random.default_rng(21422)
    z = rng.normal(size=(100, 16))
    matrices = a3_hermitian(z)
    np.testing.assert_allclose(np.sum(abs(matrices)**2, axis=(1,2)), np.sum(z*z, axis=1))
    eigenvalues = np.linalg.eigvalsh(matrices)
    eigenvalues -= eigenvalues.mean(axis=1, keepdims=True)
    gaps = np.diff(eigenvalues, axis=1)/np.sqrt(2)
    radius2 = np.einsum('bi,ij,bj->b', gaps, np.linalg.inv(a3_covariance()), gaps)
    np.testing.assert_allclose(radius2, np.sum(eigenvalues**2, axis=1), rtol=2e-14)
    vandermonde = np.prod(np.stack([eigenvalues[:,j]-eigenvalues[:,i]
                                  for i in range(4) for j in range(i+1,4)]), axis=0)
    np.testing.assert_allclose(vandermonde, 8*a3_polynomial(gaps), rtol=2e-14)
    for _ in range(20):
        direction = a3_junction_direction(rng)
        assert abs(direction.sum()) < 1e-14
        assert abs(np.linalg.norm(direction)-1) < 1e-14
        assert np.all(np.diff(direction) > 0)


def test_corner_vanishing_orders():
    chol = np.linalg.cholesky(a3_covariance())
    for axis, expected in [(0, 3), (1, 2), (2, 3)]:
        logvalues = []
        for eps in [1e-6, 1e-7]:
            x = np.full(3, eps)
            x[axis] = 1.
            white = np.linalg.solve(chol, x)
            white /= np.linalg.norm(white)
            logvalues.append(a3_log_u_white(white))
        slope = (logvalues[1]-logvalues[0])/np.log(.1)
        assert abs(slope-expected) < 5e-6


def test_analytic_cone_interface_mean_value_and_harnack():
    model = A3AnalyticCone()
    assert model.analytic and model.refinement is None
    assert model.dimension == 3 and model.p == 6 and model.homogeneity == -7
    np.testing.assert_allclose(model.R@model.R.T, model.covariance)
    rng = np.random.default_rng(2014)
    for _ in range(30):
        direction = model.sample_direction(rng)
        assert np.linalg.norm(direction) == pytest.approx(1., abs=2e-15)
        assert np.all(model.R@direction > 0)
        assert model.angular_mode(3*direction) == pytest.approx(model.angular_mode(direction))
    # Product Gauss-Legendre / periodic trapezoidal quadrature on S².
    mu, weight = np.polynomial.legendre.leggauss(48)
    phi = np.arange(96)*2*np.pi/96
    directions = np.stack(np.broadcast_arrays(np.sqrt(1-mu[:,None]**2)*np.cos(phi),
                                              np.sqrt(1-mu[:,None]**2)*np.sin(phi),
                                              mu[:,None]), axis=-1).reshape(-1, 3)
    for correlated in [[1.,1.,1.], [.01,.01,1.], [1.,1e-5,1e-5], [1e-5,1.,1e-5]]:
        point = np.linalg.solve(model.R, correlated)
        radius = .5*np.min(correlated)
        ratios = np.exp(model.log_h(point+radius*directions)-model.log_h(point))
        mean = np.sum(ratios.reshape(48,96)*weight[:,None])/(2*96)
        assert mean == pytest.approx(1., abs=2e-9)
        assert np.max(model.log_h(point+radius*directions)) <= model.log_exit_bound(point, radius)
    with pytest.raises(ValueError, match='strict interior'):
        model.log_exit_bound(np.linalg.solve(model.R,[1.,1.,1.]), 1.)


@pytest.mark.parametrize('law', ['unbiased', 'biased'])
def test_graphs_reproducibility_and_paired_refinement(law):
    fine = sample_a3_excursion(31, np.random.default_rng(92), law, 8)
    duplicate = sample_a3_excursion(31, np.random.default_rng(92), law, 8)
    np.testing.assert_array_equal(fine.values, duplicate.values)
    assert fine.metadata == duplicate.metadata
    assert fine.metadata['grid_values_exact']
    assert not fine.metadata['cell_minima_exact']
    assert fine.segment_minima is None
    assert np.all(fine.values[1:-1] > 0)
    coarse = thin_a3_sample(fine, 31, 2)
    if law == 'biased':
        assert fine.metadata['junction_time'] in coarse.times
    graph = graph_from_excursion(coarse, 31)
    assert validate_graph(graph)['connected']
    assert [len(e) for e in graph.coordinate_edges] == [28]*3
    with pytest.raises(ValueError, match='not nested'):
        thin_a3_sample(fine, 31, 3)


def test_unbiased_radial_chi15_and_multitime_covariance():
    rng = np.random.default_rng(1167)
    chol = np.linalg.cholesky(a3_covariance())
    r2 = []
    for _ in range(1600):
        sample = sample_a3_excursion(4, rng, 'unbiased', 1)
        white = np.linalg.solve(chol, sample.values[1:3].T).T
        r2.append(np.sum(white**2, axis=1))
    r2 = np.asarray(r2)
    standardized = r2/np.array([.1875, .25])
    assert np.max(abs(standardized.mean(axis=0)-15)) < .6
    expected_cov = 30*(.25-.25*.5)**2
    assert abs(np.cov(r2.T)[0,1]-expected_cov) < .13


def test_biased_conditional_path_second_moment_and_times():
    rng = np.random.default_rng(54601)
    inv = np.linalg.inv(a3_covariance())
    residual, hitting, junction_radius, junction_time = [], [], [], []
    for _ in range(1600):
        sample = sample_a3_excursion(4, rng, 'biased', 1)
        tau = np.array(sample.metadata['hitting_times'])
        u = sample.metadata['junction_time']
        at_half = sample.values[np.argmin(abs(sample.times-.5))]
        # The traceless Hermitian bridge has 15 Gaussian coordinates.
        duration = u if .5 <= u else 1-u
        fraction = .5/duration
        expectation = fraction**2/tau.sum()+15*.5*(duration-.5)/duration
        residual.append(at_half@inv@at_half-expectation)
        hitting.extend(tau)
        junction_radius.append(1/tau.sum())
        junction_time.append(u)
    residual = np.array(residual)
    assert abs(residual.mean()) < 5*residual.std(ddof=1)/np.sqrt(len(residual))
    assert abs(np.mean(hitting)-1/11) < .004
    assert abs(np.mean(junction_radius)-2*6.5**2/14) < .19
    assert abs(np.var(junction_time)-1/56) < .002
