import numpy as np
from src.corner_ritz import fit_corner_ritz
from src.local_polynomial_bounds import polynomial_box_bound


def test_local_bound_covers_direct_values_and_shrinks_to_center():
    C=np.array([[1.,-.2,.1],[-.2,1,.35],[.1,.35,1.]])
    mode=fit_corner_ritz(C,6,48)
    center=np.array([.5,.6,.7]);width=.05
    bound=polynomial_box_bound(mode,center,width)
    points=center+np.random.default_rng(319).uniform(-width,width,size=(1000,3))
    values=np.prod(points[:,None,:]**mode.exponents[None,:,:],axis=2)@mode.coefficients
    assert values.max()<=bound["upper"]*(1+1e-12)
    assert values.min()>=bound["lower"]-1e-12
    tiny=polynomial_box_bound(mode,center,1e-10)
    assert tiny["upper_ratio"]<1+1e-6
