import numpy as np
import pytest

from src.cone_geometry import cone_eigenpair


@pytest.mark.parametrize("angle", [np.pi/3, np.pi/2, 2*np.pi/3])
def test_angular_eigenpair_converges_to_birectangular_formula(angle):
    C = np.eye(3)
    C[0, 1] = C[1, 0] = -np.cos(angle)
    target = (1 + np.pi/angle) * (2 + np.pi/angle)
    coarse, fine = cone_eigenpair(C, 3), cone_eigenpair(C, 5)
    assert abs(fine.eigenvalue-target) < abs(coarse.eigenvalue-target)
    assert abs(fine.eigenvalue-target) / target < 0.012
    assert fine.values.min() >= 0
    assert np.all(fine.values[fine.boundary] == 0)


def test_orthant_harmonic_homogeneity_and_mode():
    mode = cone_eigenpair(np.eye(3), 5)
    point = np.ones(3)
    assert np.isclose(mode.log_h(2*point)-mode.log_h(point),
                      (-mode.alpha1-0.5)*np.log(2))
    assert abs(mode.string_susceptibility + 2.5) < .02
    points = np.array([[1, 1, 1], [2, 1, 1]], dtype=float)
    points /= np.linalg.norm(points, axis=1)[:, None]
    ratios = mode.angular_mode(points) / points.prod(axis=1)
    assert abs(ratios[0]/ratios[1]-1) < .03
