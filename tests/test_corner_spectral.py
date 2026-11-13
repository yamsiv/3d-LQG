"""Mathematical controls for the exploratory Task 2C MPS method."""
import numpy as np
import pytest

from src.corner_spectral import fit_corner_mode, triangle_geometry
from src.doob_excursions import analytic_cone


A3 = np.array([[1., -.5, 0], [-.5, 1, -.5], [0, -.5, 1]])
GENERIC = np.array([[1., -.2, .1], [-.2, 1, .35], [.1, .35, 1]])


def test_spherical_vertex_angles_match_opposite_face_normals():
    R, vertices, frames, angles = triangle_geometry(GENERIC)
    assert np.allclose(R @ vertices.T, np.diag(np.diag(R @ vertices.T)))
    assert np.allclose(angles, [np.arccos(-GENERIC[1, 2]),
                               np.arccos(-GENERIC[2, 0]), np.arccos(-GENERIC[0, 1])])
    assert np.allclose(frames @ frames.transpose(0, 2, 1), np.eye(3))


@pytest.mark.parametrize("rho", [0., -.5, .5])
def test_recovers_known_birectangular_eigenspace(rho):
    C = np.eye(3); C[0, 1] = C[1, 0] = rho
    p = np.pi/np.arccos(-rho)+1
    mode = fit_corner_mode(C, order=8, degree_bracket=(p-.2, p+.2))
    assert abs(mode.degree-p) < 2e-9
    R, vertices, _, _ = triangle_geometry(C)
    center = vertices.sum(axis=0); center /= np.linalg.norm(center)
    points = np.random.default_rng(49).dirichlet([2.]*3, size=60) @ vertices
    true = analytic_cone(C)
    expected = true.angular_mode(points)/true.angular_mode(center)
    assert np.max(np.abs(mode.angular_mode(points)/expected-1)) < 2e-7


def test_a3_local_continuation_has_correct_high_order_corner_vanishing():
    mode = fit_corner_mode(A3, order=12, degree=6.)
    R, vertices, _, _ = triangle_geometry(A3)
    center = vertices.sum(axis=0); center /= np.linalg.norm(center)
    def exact(points):
        a, b, c = (points @ R.T).T
        return a*b*c*(a+b)*(b+c)*(a+b+c)/np.linalg.norm(points, axis=1)**6
    norm = exact(center[None, :])[0]
    for j in range(3):
        gap = np.full((6, 3), 1.); epsilon = np.logspace(-2, -7, 6)
        gap[:] = epsilon[:, None]; gap[:, j] = 1
        points = np.linalg.solve(R, gap.T).T
        local = mode.local_corner_expansion(j)
        assert np.max(np.abs(local.angular_mode(points)/(exact(points)/norm)-1)) < 2e-7


def test_generic_refinement_stabilizes_degree_and_relative_interior_mode():
    coarse = fit_corner_mode(GENERIC, order=8, degree_bracket=(2.6, 2.9))
    fine = fit_corner_mode(GENERIC, order=16, degree_bracket=(2.6, 2.9))
    assert abs(coarse.degree-fine.degree) < 2e-7
    assert fine.boundary_tension < coarse.boundary_tension*.01
    R, vertices, _, _ = triangle_geometry(GENERIC)
    points = np.random.default_rng(51).dirichlet([2.]*3, 80) @ vertices
    assert np.max(np.abs(coarse.angular_mode(points)/fine.angular_mode(points)-1)) < 1e-4
    corner = fine.local_corner_expansion(0)
    eps = np.array([1e-5, 1e-6, 1e-7])
    gaps = np.tile(eps[:, None], (1, 3)); gaps[:, 0] = 1
    values = corner.angular_mode(np.linalg.solve(R, gaps.T).T)
    slopes = np.diff(np.log(values))/np.diff(np.log(eps))
    assert np.max(np.abs(slopes-np.pi/fine.angles[0])) < 1e-4


def test_nonpositive_mode_is_never_silently_floored():
    mode = fit_corner_mode(np.eye(3), order=4, degree=3.)
    mode.coefficients *= -1
    with pytest.raises(ValueError, match="nonpositive"):
        mode.log_h([1., 1., 1.])


def test_independent_ritz_and_local_mps_agree_near_generic_corner():
    from src.corner_ritz import fit_corner_ritz
    mps = fit_corner_mode(GENERIC, 16, degree_bracket=(2.6, 2.9))
    ritz = fit_corner_ritz(GENERIC, degree=8, quadrature_order=64)
    R, vertices, _, _ = triangle_geometry(GENERIC)
    center = vertices.sum(axis=0); center /= np.linalg.norm(center)
    norm = ritz.angular_mode(center)
    for j in range(3):
        gap = np.full(3, 1e-5); gap[j] = 1
        point = np.linalg.solve(R, gap)
        ratio = ritz.angular_mode(point)/norm/mps.local_corner_expansion(j).angular_mode(point)
        assert abs(ratio-1) < .012
