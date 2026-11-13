import numpy as np
import pytest
from src.corner_ritz import CornerFactor,spherical_quadrature,fit_corner_ritz
from src.doob_excursions import analytic_cone


@pytest.mark.parametrize("rho",[-.5,0,.5])
def test_corner_factor_reproduces_birectangular_modes(rho):
    C=np.eye(3);C[0,1]=C[1,0]=rho
    factor=CornerFactor(C);analytic=analytic_cone(C)
    x=np.linalg.solve(factor.R,np.random.default_rng(10).exponential(size=(50,3)).T).T
    x/=np.linalg.norm(x,axis=1)[:,None]
    ratio=factor.value(x)/analytic.angular_mode(x)
    np.testing.assert_allclose(ratio/ratio[0],1,rtol=2e-12)
    mode=fit_corner_ritz(C,degree=0,quadrature_order=64)
    assert abs(mode.eigenvalue-analytic.eigenvalue)<2e-5
    assert mode.positivity_certificate()["certified_positive"]


def test_a3_corner_evaluation_avoids_global_basis_cancellation():
    C=np.array([[1.,-.5,0],[-.5,1,-.5],[0,-.5,1.]])
    mode=fit_corner_ritz(C,degree=1,quadrature_order=64)
    R=np.linalg.cholesky(C)
    x=np.array([[1.,1.,1.],[1e-6,1e-6,1.],[1e-10,1e-10,1.],[1e-12,1e-12,1.]])
    z=np.linalg.solve(R,x.T).T;z/=np.linalg.norm(z,axis=1)[:,None]
    d=z@R.T
    truth=d.prod(axis=1)*(d[:,0]+d[:,1])*(d[:,1]+d[:,2])*d.sum(axis=1)
    ratios=mode.angular_mode(z)/truth
    np.testing.assert_allclose(ratios/ratios[0],1,rtol=1e-8)


def test_sphere_area_and_corner_factor_gradient():
    C=np.array([[1.,-.2,.1],[-.2,1,.35],[.1,.35,1.]])
    z,w=spherical_quadrature(C,64)
    expected=np.arccos(-C[0,1])+np.arccos(-C[0,2])+np.arccos(-C[1,2])-np.pi
    assert abs(w.sum()-expected)<1e-12
    factor=CornerFactor(C);point=np.linalg.solve(factor.R,[.2,.5,1.])
    value,gradient=factor.log_value_and_gradient(point)
    difference=np.array([(factor.log_value_and_gradient(point+1e-6*e)[0]-
                          factor.log_value_and_gradient(point-1e-6*e)[0])/2e-6 for e in np.eye(3)])
    np.testing.assert_allclose(gradient,difference,rtol=1e-7,atol=1e-7)
    assert abs(point@gradient-factor.homogeneity)<1e-12


def test_a3_harmonic_is_in_degree_one_trial_space():
    C=np.array([[1.,-.5,0],[-.5,1,-.5],[0,-.5,1.]])
    mode=fit_corner_ritz(C,degree=1,quadrature_order=64)
    assert abs(mode.eigenvalue-42)<1e-7
    x=np.random.default_rng(5).exponential(size=(50,3))
    z=np.linalg.solve(np.linalg.cholesky(C),x.T).T
    z/=np.linalg.norm(z,axis=1)[:,None];d=z@np.linalg.cholesky(C).T
    target=d.prod(axis=1)*(d[:,0]+d[:,1])*(d[:,1]+d[:,2])*d.sum(axis=1)
    ratio=mode.angular_mode(z)/target
    np.testing.assert_allclose(ratio/ratio[0],1,rtol=1e-9)
    assert mode.positivity_certificate()["certified_positive"]
