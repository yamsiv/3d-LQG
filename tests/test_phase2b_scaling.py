import numpy as np
from src.phase2b_scaling import fit_mean_model,fit_equation45


def test_equation45_recovers_synthetic_correction_and_preserves_reference():
    sizes=2.**np.arange(10,20)
    x=sizes/sizes[-1]
    p,b=.24,.018
    k=x**(-p)*(1-b+b*x**(-p))
    fit=fit_equation45(sizes,k)
    assert abs(fit["p"]-p)<1e-5
    assert abs(fit["b"]-b)<1e-5
    assert fit["prediction"][-1]==1


def test_mean_correction_models_can_recover_offset_and_curvature():
    sizes=2.**np.arange(10,20)
    x=sizes/sizes[-1]
    mean=30*x**.21-2+.2*x**(-.21)
    fit=fit_mean_model(sizes,mean,np.ones(len(x))*.01,"offset_inverse_power")
    assert abs(fit["p"]-.21)<1e-4
    assert fit["chi2"]<1e-5
