import numpy as np
from experiments.phase3c_confirmation_graphs import radial_bias_integral


def test_radial_functional_uses_covariance_norm_and_is_label_invariant():
    c=np.array([[1.,-.2,.1],[-.2,1.,.3],[.1,.3,1.]])
    t=np.linspace(0,1,101);x=np.sin(np.pi*t)[:,None]*np.array([[1.,2.,3.]])
    expected=radial_bias_integral(t,x,c,3.5)['log_integral']
    permutation=[2,0,1]
    got=radial_bias_integral(t,x[:,permutation],c[np.ix_(permutation,permutation)],3.5)['log_integral']
    assert np.isclose(expected,got,atol=2e-14)
    scaled=radial_bias_integral(t,2*x,c,3.5)['log_integral']
    assert np.isclose(scaled-expected,5*np.log(2))


def test_trapezoidal_functional_for_constant_radius_and_known_normalizer():
    t=np.array([0.,.1,.8,1.]);x=np.tile([1.,0.,0.],(4,1))
    result=radial_bias_integral(t,x,np.eye(3),2.)
    assert np.isclose(result['log_integral'],0)
    assert np.isclose(result['log_original_expectation'],0)
    assert result['continuous_integral_exact'] is False
