"""Independent archival and analytical checks added during the Phase 2B audit."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.stats import beta

from src.brownian_excursions import sample_independent_excursion
from src.covariance import from_correlations, from_angles, to_angles
from src.mated_crt import cell_minima, graph_from_excursion, graph_from_minima
from src.graph_metrics import metric_sample
from src.scaling_analysis import collapse_histograms


ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def old_records():
    path = ROOT / "results/data/i3_biased_combined.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_archived_samples_have_distinct_streams_and_consistent_histograms():
    records = old_records()
    assert len(records) == 568
    assert len({r["graph_seed"] for r in records}) == 568
    assert len({r["metric_seed"] for r in records}) == 568
    assert not ({r["graph_seed"] for r in records} & {r["metric_seed"] for r in records})
    for record in records:
        n, roots = record["n"], len(record["roots"])
        histogram = np.asarray(record["distance_histogram"])
        root_histograms = np.asarray(record["root_histograms"])
        assert histogram.sum() == roots*n
        assert histogram[0] == roots
        np.testing.assert_array_equal(root_histograms.sum(axis=0), histogram)
        np.testing.assert_array_equal(root_histograms.sum(axis=1), np.full(roots, n))
        assert record["mean_distance"] == np.dot(np.arange(len(histogram)), histogram)/histogram.sum()
        assert record["law"] == record["sampler_metadata"]["law"] == "biased"
        assert record["sampler_metadata"]["cell_minima_exact"]
        assert not record["sampler_metadata"]["mcmc"]


def test_archival_concatenation_preserves_inputs_without_counting_twice():
    path = ROOT / "results/data/i3_biased_combined.jsonl"
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    inputs = []
    for source in metadata["sources"]:
        data = (ROOT/source["file"]).read_bytes()
        assert sha256(data).hexdigest() == source["sha256"]
        assert len(data.splitlines()) == source["records"]
        inputs.append(data)
    assert path.read_bytes() == b"".join(inputs)


def test_original_naive_and_offset_numbers_by_independent_regression():
    records = old_records()
    sizes = np.asarray(sorted({r["n"] for r in records}))
    groups = [np.array([r["mean_distance"] for r in records if r["n"] == n]) for n in sizes]
    means = np.array([g.mean() for g in groups])
    se = np.array([g.std(ddof=1)/np.sqrt(len(g)) for g in groups])
    naive = 1/np.polyfit(np.log(sizes), np.log(means), 1)[0]
    # Different algorithm from production's profile: optimize all three
    # parameters jointly, with no target exponent in the loss.
    fit = least_squares(lambda x: (x[0]*(sizes/sizes[-1])**x[1]+x[2]-means)/se,
                        [30., .2, -2.], bounds=([0., .03, -100.], [1000., .8, 100.]))
    assert abs(naive-4.189502624088149) < 1e-10
    assert abs(1/fit.x[1]-4.83329714) < 2e-5
    assert abs(np.sum(fit.fun**2)-4.95003398) < 1e-6
    assert abs(collapse_histograms(records)["effective_dimension"]-5.07028326) < 1e-7


def test_regenerate_graphs_and_metrics_from_two_archived_runs():
    # Regenerate one realization from each independent original run. This
    # checks actual sample provenance, rather than only repeating seed labels.
    records = old_records()
    for index in [0, 448]:
        old = records[index]
        sample = sample_independent_excursion(old["n"], 3,
                     np.random.default_rng(old["graph_seed"]), law="biased")
        graph = graph_from_excursion(sample, old["n"])
        new = metric_sample(graph.adjacency, np.random.default_rng(old["metric_seed"]), roots=8)
        assert new["roots"] == old["roots"]
        assert new["distance_histogram"] == old["distance_histogram"]
        assert new["degree"] == old["degree"]


def test_biased_midpoint_moment_from_analytical_beta_mixture():
    # T_i=.5/G_i, G_i~Gamma(alpha1), so U~Beta(alpha1,alpha1) and
    # H² | U = 2 U(1-U) Q, Q~Gamma(2 alpha1). Conditional bridge norms
    # give an independently integrated expected squared midpoint radius.
    d, alpha, t = 3, 3.5, .5
    integrand = lambda u: (4*alpha*t*t*(1-u)/u + 3*d*t*(1-t/u))*beta.pdf(u, alpha, alpha)
    expected = 2*quad(integrand, .5, 1, epsabs=1e-11)[0]
    assert abs(expected-2.9814083641853677) < 1e-10
    rng = np.random.default_rng(41415)
    radii = []
    for _ in range(4000):
        sample = sample_independent_excursion(4, d, rng, law="biased")
        midpoint = sample.values[np.searchsorted(sample.times, t)]
        radii.append(float(midpoint@midpoint))
    se = np.std(radii, ddof=1)/np.sqrt(len(radii))
    assert abs(np.mean(radii)-expected) < 5*se
    # The unbiased mean is 3*d*t*(1-t)=2.25, so this also checks that the
    # law switch is substantive rather than merely a metadata distinction.
    assert expected > 3*d*t*(1-t)+.5


def test_coordinate_permutation_changes_provenance_but_not_graph():
    C = from_correlations(-.2, .1, .35)
    alpha, beta_angle, theta = to_angles(C)
    np.testing.assert_allclose(from_angles(alpha, beta_angle, theta), C, atol=1e-14)
    sample = sample_independent_excursion(97, 3, np.random.default_rng(9284), law="biased")
    minima = cell_minima(sample, 97)
    graph = graph_from_minima(minima)
    for permutation in [(0,2,1), (2,0,1), (1,2,0)]:
        permuted_C = C[np.ix_(permutation, permutation)]
        np.testing.assert_allclose(from_angles(*to_angles(permuted_C)), permuted_C, atol=1e-14)
        permuted = graph_from_minima(minima[:, permutation])
        np.testing.assert_array_equal(permuted.edges, graph.edges)
        np.testing.assert_array_equal(permuted.multiplicities, graph.multiplicities)


def test_reference_normalized_eq45_gls_matches_independent_cholesky_objective():
    from src.phase2b_scaling import fit_equation45
    sizes = 2.**np.arange(10, 19)
    x = sizes[:-1]/sizes[-1]
    p, b = .21, .018
    target = x**(-p)*(1-b+b*x**(-p))
    target += np.random.default_rng(132).normal(scale=.003, size=len(x))
    covariance = .0001*np.eye(len(x))+.00003*np.ones((len(x), len(x)))
    result = fit_equation45(sizes, np.r_[target, 1.], covariance, b_bound=.1)
    L = np.linalg.cholesky(covariance)
    def residual(parameters):
        exponent, correction = parameters
        error = x**(-exponent)*(1-correction+correction*x**(-exponent))-target
        return np.linalg.solve(L, error)
    independent = least_squares(residual, [.2, .02], bounds=([.03, -.1], [.8, .1]))
    np.testing.assert_allclose([result["p"], result["b"]], independent.x, atol=2e-6)
    error = np.asarray(result["prediction"][:-1])-target
    assert abs(result["objective"]-error@np.linalg.solve(covariance, error)) < 1e-8
