import numpy as np
import pytest

from src.cone_geometry import cone_eigenpair
from src.cone_sampling import FastFEMMode, enriched_cone_eigenpair


@pytest.mark.parametrize("C", [np.eye(3), np.array([[1,-.2,.1],[-.2,1,.35],[.1,.35,1]])])
def test_cached_fem_interpolation_matches_original(C):
    mode = cone_eigenpair(C, 4)
    cached = FastFEMMode(mode, C)
    R = np.linalg.cholesky(C)
    x = np.random.default_rng(51).exponential(size=(300, 3))
    directions = np.linalg.solve(R, x.T).T
    np.testing.assert_allclose(cached.angular_mode(directions), mode.angular_mode(directions),
                               atol=2e-13, rtol=2e-12)
    assert np.all(cached.angular_mode(directions) >= 0)
    assert np.all(cached.angular_mode(directions) <= cached.maximum*(1+1e-13))


def test_cached_fem_outside_and_boundary_zero():
    cached = FastFEMMode(cone_eigenpair(np.eye(3), 4), np.eye(3))
    assert cached.angular_mode([-1,1,1]) == 0
    assert abs(cached.angular_mode([0,1,1])) < 1e-12
    assert cached.angular_mode([1,1,1]) > 0
    assert np.isclose(cached.log_h([2,2,2])-cached.log_h([1,1,1]),
                      (-cached.alpha1-.5)*np.log(2))


def test_existing_fem_interior_corner_zero_patch_is_detected():
    mode = cone_eigenpair(np.eye(3), 5)
    cached = FastFEMMode(mode, np.eye(3))
    assert np.count_nonzero(np.all(mode.boundary[mode.faces], axis=1)) == 3
    # Strictly interior true orthant mode is positive, while this P1 facet is zero.
    assert cached.angular_mode([.001,.001,1]) == 0


def test_optional_corner_enrichment_removes_open_zero_patch():
    mode=enriched_cone_eigenpair(np.eye(3),5)
    cached=FastFEMMode(mode,np.eye(3))
    assert mode.inserted_corner_nodes==3
    assert np.count_nonzero(np.all(mode.boundary[mode.faces],axis=1))==0
    assert cached.angular_mode([.001,.001,1])>0
    assert np.all(mode.values[~mode.boundary]>0)


def test_centroid_only_enrichment_does_not_connect_corner_principal_mode():
    mode=enriched_cone_eigenpair(np.eye(3),5,strategy="centroid")
    # Added degrees of freedom connect only to prescribed zero boundary values.
    # Roundoff signs can vary; they are not positive first-mode corner values.
    assert np.max(np.abs(mode.values[-3:]))<1e-12


def test_mean_ball_curve_is_not_independent_of_pair_distance_cdf():
    from experiments.doob_error_budget import matrix
    row={"n":4,"distance_histogram":[2,4,2],"mean_ball_volumes":[1,3,4]}
    np.testing.assert_allclose(matrix([row],"distance_cdf"),matrix([row],"ball_fraction"))


def test_permuted_reference_junction_metadata_matches_actual_path():
    from experiments.validate_doob import reference_sample,cases
    C,kind=cases()["a2_permuted"]
    sample=reference_sample(16,C,kind,np.random.default_rng(7119))
    index=np.searchsorted(sample.times,sample.metadata["junction_time"])
    direction=np.asarray(sample.metadata["junction_direction_correlated"])
    np.testing.assert_allclose(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(C),direction)),1.,atol=2e-13)
    np.testing.assert_allclose(sample.values[index],direction/np.sqrt(sum(sample.metadata["hitting_times"])),atol=2e-13)
