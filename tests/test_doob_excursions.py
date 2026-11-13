import numpy as np
import pytest

from src.doob_excursions import (analytic_cone, make_cone_model,
    sample_sphere_exit, isotropic_ball_walk, sample_doob_excursion)
from src.mated_crt import graph_from_excursion, validate_graph
from src.cone_geometry import cone_eigenpair


def test_analytic_orthant_h_matches_polynomial_and_spectral_formula():
    cone = analytic_cone(np.eye(3))
    points = np.array([[1.,2,3],[.1,1,1],[2,2,2]])
    # Wedge sin(2theta)*s² is 2*x1*x2, an irrelevant normalization.
    target = np.log(2*points.prod(axis=1))-7*np.log(np.linalg.norm(points,axis=1))
    np.testing.assert_allclose(cone.log_h(points),target,atol=1e-12)
    assert cone.eigenvalue == 12
    assert cone.alpha1 == 3.5


@pytest.mark.parametrize("rho", [-.5,0,.5])
def test_wedge_and_birectangular_h_are_harmonic(rho):
    for dimension in [2,3]:
        C=np.eye(dimension)
        C[0,1]=C[1,0]=rho
        cone=analytic_cone(C)
        point=np.linalg.solve(cone.R,np.ones(dimension))
        epsilon=1e-4
        center=np.exp(cone.log_h(point))
        laplacian=0
        for coordinate in range(dimension):
            shift=epsilon*np.eye(dimension)[coordinate]
            laplacian+=(np.exp(cone.log_h(point+shift))+np.exp(cone.log_h(point-shift))-2*center)/epsilon**2
        assert abs(laplacian)/center<2e-5


def test_coordinate_permutations_preserve_analytical_h():
    C=np.array([[1,-.5,0],[-.5,1,0],[0,0,1.]])
    reference=analytic_cone(C)
    correlated=np.array([1.,2.,1.5])
    original=reference.log_h(np.linalg.solve(reference.R,correlated))
    for permutation in [[1,0,2],[2,1,0],[1,2,0]]:
        matrix=C[np.ix_(permutation,permutation)]
        cone=analytic_cone(matrix)
        transformed=cone.log_h(np.linalg.solve(cone.R,correlated[permutation]))
        assert abs(transformed-original)<1e-12


def test_analytic_angular_beta_moment_and_unit_sphere():
    C=np.array([[1,-.5,0],[-.5,1,0],[0,0,1.]])
    cone=analytic_cone(C)
    rng=np.random.default_rng(762)
    directions=np.array([cone.sample_direction(rng) for _ in range(5000)])
    np.testing.assert_allclose(np.sum(directions**2,axis=1),1,atol=1e-14)
    assert np.all(directions@cone.R.T>0)
    # For alpha=pi/3: third whitened squared coordinate Beta(1.5,4).
    assert abs(np.mean((directions@cone.ey)**2)-1.5/5.5)<.009


def test_sphere_exit_first_moment_independent_harmonic_oracle():
    cone=analytic_cone(np.eye(3))
    rng=np.random.default_rng(729)
    point=np.ones(3)
    radius=.4
    exits=np.array([sample_sphere_exit(cone,point,radius,rng)[0] for _ in range(2500)])
    gradient=1/point-7*point/np.dot(point,point)
    expected=radius**2/3*gradient
    np.testing.assert_allclose(np.mean(exits-point,axis=0),expected,atol=.014)


def test_ball_walk_clock_converges_to_radius_squared_over_dimension():
    rng=np.random.default_rng(825)
    durations=[]
    for _ in range(700):
        points,times,info=isotropic_ball_walk(1.,3,rng,step_size=.05,relative_step=.1)
        assert np.isclose(np.linalg.norm(points[-1]),1)
        assert np.all(np.linalg.norm(points[:-1],axis=1)<1)
        assert np.all(np.diff(times)>0)
        durations.append(times[-1])
    assert abs(np.mean(durations)-1/3)<.025


def test_doob_metadata_graph_and_reproducibility():
    options=dict(step_size=.04,relative_step=.3,origin_cutoff=.03)
    a=sample_doob_excursion(47,np.eye(3),np.random.default_rng(441),**options)
    b=sample_doob_excursion(47,np.eye(3),np.random.default_rng(441),**options)
    np.testing.assert_array_equal(a.times,b.times)
    np.testing.assert_array_equal(a.values,b.values)
    assert np.all(a.values[1:-1]>0)
    assert a.metadata["law"]=="biased"
    assert not a.metadata["cell_minima_exact"]
    assert not a.metadata["grid_values_exact"]
    assert all(info["zero_h_candidates"]==0 for info in a.metadata["leg_diagnostics"])
    graph=graph_from_excursion(a,47)
    assert validate_graph(graph)["connected"]
    assert validate_graph(graph)["hamiltonian_cycle"]
    assert graph.metadata["angular_mode"]=="analytical"


def test_fem_interior_zero_patch_fails_instead_of_flooring_mode():
    with pytest.raises(ValueError,match="interior FEM zero-support"):
        make_cone_model(np.eye(3),eigenpair=cone_eigenpair(np.eye(3),4))


def test_connected_enriched_mode_is_allowed_but_remains_provisional():
    from src.cone_sampling import FastFEMMode,enriched_cone_eigenpair
    pair=FastFEMMode(enriched_cone_eigenpair(np.eye(3),4),np.eye(3))
    cone=make_cone_model(np.eye(3),eigenpair=pair)
    assert not cone.analytic
    assert np.isfinite(cone.log_h([.001,.001,1.]))


def test_numerical_eigenpair_cannot_be_reused_with_wrong_covariance():
    mode=cone_eigenpair(np.eye(3),3)
    covariance=np.array([[1.,.2,0],[.2,1.,0],[0,0,1.]])
    with pytest.raises(ValueError,match="eigenpair cone does not match"):
        make_cone_model(covariance,eigenpair=mode)
