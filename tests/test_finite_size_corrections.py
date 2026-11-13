import numpy as np

from src.finite_size_corrections import shifted_mean_fit


def test_shifted_fit_recovers_known_exponent_despite_additive_bias():
    rng = np.random.default_rng(112)
    records = [{"n": n, "mean_distance": 3*n**.25-2+float(rng.normal(scale=.03))}
               for n in [256, 512, 1024, 2048, 4096, 8192]
               for _ in range(24)]
    fit = shifted_mean_fit(records, bootstrap=40, seed=5)
    assert abs(fit["effective_dimension"]-4) < .08
    assert abs(fit["B"]+2) < .15
    assert fit["bootstrap_boundary_fraction"] == 0
