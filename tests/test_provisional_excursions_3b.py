import numpy as np
import pytest

from src.provisional_excursions_3b import (make_provisional_model,
    quadrature_sphere_exit, sample_provisional_excursion)
from src.covariance import from_correlations


@pytest.mark.parametrize("correlations,degree", [((0, 0, 0), 0),
    ((-.5, 0, 0), 0), ((-.5, 0, -.5), 1), ((-.2, .1, .35), 4)])
def test_fast_positive_formula_matches_existing_ritz(correlations, degree):
    model = make_provisional_model(from_correlations(*correlations), degree=degree)
    rng = np.random.default_rng(121)
    z = np.linalg.solve(model.R, rng.gamma(2., size=(3, 100))).T
    np.testing.assert_allclose(model.angular_mode(z), model.mode.angular_mode(z),
                               rtol=2e-13, atol=2e-14)
    assert model.certificate["certified_positive"]
    np.testing.assert_allclose(model.log_h(2*z)-model.log_h(z),
                               (-model.alpha1-.5)*np.log(2), atol=1e-13)


def test_initial_sir_inside_and_marked_approximate():
    model = make_provisional_model(np.eye(3), degree=0, initial_candidates=128)
    direction, info = model.sample_direction(np.random.default_rng(12), return_diagnostics=True)
    assert np.all(model.R@direction > 0)
    assert np.linalg.norm(direction) == pytest.approx(1.)
    assert info["method"].startswith("finite_SIR")
    assert 1 < info["importance_ess"] <= 128


def test_rotated_quadrature_harmonic_normalization_and_radius():
    model = make_provisional_model(np.eye(3), degree=0)
    point = np.ones(3)
    exit_point, info = quadrature_sphere_exit(model, point, .5, np.random.default_rng(3),
                                             order=10, audit_order=16)
    assert np.linalg.norm(exit_point-point) == pytest.approx(.5, abs=1e-14)
    assert abs(info["log_mean_value_ratio"]) < 2e-8
    assert abs(info["audit_log_mean_value_ratio"]) < 1e-12
    with pytest.raises(ValueError, match="interior"):
        quadrature_sphere_exit(model, point, 1., np.random.default_rng(3))


def test_pipeline_flags_and_graph_invariants():
    from src.mated_crt import graph_from_excursion, validate_graph
    sample = sample_provisional_excursion(32, np.eye(3), np.random.default_rng(24),
        degree=0, step_size=.03, relative_step=.25, origin_cutoff=.003)
    assert sample.metadata["law"] == "biased_approximation"
    assert sample.metadata["target_law"] == "biased"
    assert not sample.metadata["grid_values_exact"]
    assert not sample.metadata["cell_minima_exact"]
    assert np.all(np.diff(sample.times) > 0)
    assert np.all(sample.values >= 0)
    checks = validate_graph(graph_from_excursion(sample, 32))
    assert checks["connected"] and checks["hamiltonian_cycle"]
    assert checks["coordinate_diagonal_counts"] == [29]*3


def test_invalid_covariance_never_repaired():
    with pytest.raises(ValueError, match="positive definite"):
        make_provisional_model(from_correlations(-.9, -.9, -.9))


@pytest.mark.parametrize("case,correlations", [("i3", (0, 0, 0)),
    ("a2", (-.5, 0, 0)), ("a3", (-.5, 0, -.5))])
def test_conditional_reference_has_requested_junction(case, correlations):
    from experiments.phase3b_sampler_pilot import conditional_reference
    C = from_correlations(*correlations)
    R = np.linalg.cholesky(C)
    direction = np.linalg.solve(R, np.ones(3))
    direction /= np.linalg.norm(direction)
    metadata = {"covariance": C.tolist(), "hitting_times": [.1, .15],
        "junction_time": .4, "junction_direction": direction.tolist()}
    sample = conditional_reference(case, metadata, np.random.default_rng(81), oversample=2)
    at = np.searchsorted(sample.times, .4)
    np.testing.assert_allclose(sample.values[at], R@direction/.5, atol=1e-14)
    assert sample.metadata["law"] == "conditional_bridge_reference"
    assert np.all(sample.values >= 0)
    assert (sample.segment_minima is not None) == (case == "i3")
