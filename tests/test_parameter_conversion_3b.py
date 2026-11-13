import numpy as np
import pytest

from src.parameter_conversion_3b import (ParameterCalibration,
    fit_crossing_exponents, implicit_conversion)


def test_inverse_branch_and_flat_endpoint():
    xi = np.array([0, .2, .3, .4])
    exponent = np.array([0., .05, .15, .2])
    q, gamma, d = implicit_conversion(xi, exponent)
    assert gamma[0] == 0 and d[0] == 3
    np.testing.assert_allclose(3/gamma[1:3]+gamma[1:3]/2, q[1:3])
    np.testing.assert_allclose(d[1:3]*xi[1:3], gamma[1:3])
    assert np.isnan(gamma[3])
    with pytest.raises(ValueError):
        implicit_conversion(-.1, 0.)


def test_known_power_law_and_correction():
    sizes = np.array([16, 24, 32, 48, 64])
    exponent = np.array([.12, .2])
    values = sizes[:, None]**(-exponent)*np.exp(2/sizes[:, None])
    array = np.broadcast_to(values, (8,)+values.shape)
    np.testing.assert_allclose(fit_crossing_exponents(array, sizes,
                                correction_power=1), exponent, atol=1e-12)
    pure = np.broadcast_to(sizes[:, None]**(-exponent), (8,)+values.shape)
    np.testing.assert_allclose(fit_crossing_exponents(pure, sizes), exponent)


def test_no_extrapolation_or_invalid_branch_bridge():
    rows = tuple({"xi": xi, "lambda": exponent,
                  "gamma_mc95_low": .5, "gamma_mc95_high": 1.5,
                  "gamma_systematic_low": .4,"gamma_systematic_high": 1.6,
                  "bootstrap_subcritical_fraction": 1.,
                  "all_sensitivity_models_subcritical": 1.}
                 for xi, exponent in [(.2,.1),(.3,.2),(.4,.3)])
    table = ParameterCalibration(rows,{})
    assert table.evaluate(.1)["status"] == "outside_calibration_domain"
    assert table.evaluate(.25)["valid"]
    assert not table.evaluate(.4)["valid"]
    assert not table.evaluate(.25)["asymptotic_map_validated"]
    assert table.evaluate(0)["gamma"] == 0


def test_branch_probability_is_not_hidden_by_conditional_interval():
    row={"xi":.325,"lambda":.2,"gamma_mc95_low":1.8,"gamma_mc95_high":2.4,
         "gamma_systematic_low":2.1,"gamma_systematic_high":2.3,
         "bootstrap_subcritical_fraction":.45,
         "all_sensitivity_models_subcritical":0.}
    result=ParameterCalibration((row,),{}).evaluate(.325)
    assert result["valid"]
    assert result["gamma_mc95_is_conditional"]
    assert result["gamma_systematic_has_invalid_models"]
    assert result["status"]=="provisional_branch_or_model_sensitive"
