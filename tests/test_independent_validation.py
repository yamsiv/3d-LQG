"""Independent equation-level oracles, distinct from production algorithms."""

import itertools
import numpy as np
from scipy import sparse

from src.secondary_observables import (
    exact_return_probabilities, monte_carlo_returns,
    effective_resistances, spectral_dimension_pilot,
)
from src.mated_crt import triangulation_fast, graph_from_minima, cell_minima
from src.brownian_excursions import (
    sample_independent_excursion, positive_bridge_minima, ExcursionSample,
)


def brute_polygon_diagonals(minima):
    """Literal §2.5 horizontal-chord condition; cubic time, no stack."""
    n = len(minima)
    edges = set()
    for left in range(n):
        for right in range(left+2, n):
            if left == 0 and right == n-1:
                continue
            if all(minima[k] > max(minima[left], minima[right])
                   for k in range(left+1, right)):
                edges.add((left, right))
    return edges


def noncrossing(diagonals):
    for (a,b), (c,d) in itertools.combinations(diagonals, 2):
        if a < c < b < d or c < a < d < b:
            return False
    return True


def test_independent_hand_checked_hexagons():
    assert brute_polygon_diagonals([0,5,2,7,3,0]) == {(0,2),(2,4),(2,5)}
    assert brute_polygon_diagonals([0,3,7,2,5,0]) == {(0,3),(1,3),(3,5)}


def test_independent_oracle_triangulates_all_small_rank_patterns():
    for n in range(3, 9):
        for interior in itertools.permutations(range(1, n-1)):
            diagonals = brute_polygon_diagonals([0, *interior, 0])
            assert len(diagonals) == n-3
            assert noncrossing(diagonals)
            assert set(map(tuple, triangulation_fast([0, *interior, 0]))) == diagonals


def test_parallel_diagonal_provenance_and_shared_cycle():
    minima = np.array([0,5,2,7,3,0])
    graph = graph_from_minima(np.column_stack([minima,minima]))
    counts = dict(zip(map(tuple,graph.edges), graph.multiplicities))
    assert counts == {(0,1):1,(1,2):1,(2,3):1,(3,4):1,(4,5):1,(0,5):1,
                      (0,2):2,(2,4):2,(2,5):2}
    assert all(set(map(tuple, edges)) == brute_polygon_diagonals(minima)
               for edges in graph.coordinate_edges)


def test_cell_minimum_includes_both_sides_of_internal_junction():
    times = np.array([0,.25,.375,.5,.75,1])
    values = np.array([0,2,4,3,1,0])[:,None]
    minima = np.array([0,1.8,.6,.5,0])[:,None]
    sample = ExcursionSample(times,values,minima,{})
    np.testing.assert_allclose(cell_minima(sample,4).ravel(), [0,.6,.5,0])


def test_positive_bridge_minimum_survival_independently():
    rng = np.random.default_rng(903)
    x,y,dt = .7,1.1,.25
    minima = positive_bridge_minima(np.full(100000,x), y, dt, rng)
    for level in [.05,.2,.4,.6]:
        expected = (1-np.exp(-2*(x-level)*(y-level)/dt))/(1-np.exp(-2*x*y/dt))
        assert abs(np.mean(minima > level)-expected) < .006


def test_unbiased_midpoint_independent_chi_squared_moment():
    rng = np.random.default_rng(1203)
    norms = []
    for _ in range(2500):
        sample = sample_independent_excursion(4,3,rng,law="unbiased")
        norms.append(float(np.sum(sample.values[2]**2)/.25))
    # Chi-square(9): mean 9, variance 18. Six SE tolerance.
    assert abs(np.mean(norms)-9) < 6*np.sqrt(18/len(norms))


def test_biased_hitting_time_junction_direction_laws():
    rng = np.random.default_rng(1204)
    hitting, junction, direction = [],[],[]
    for _ in range(2500):
        sample = sample_independent_excursion(4,3,rng,law="biased")
        hitting.append(sample.metadata["hitting_times"])
        junction.append(sample.metadata["junction_time"])
        direction.append(sample.metadata["junction_direction"])
    # alpha1=7/2. T~InvGamma(alpha1,1/2), U~Beta(alpha1,alpha1).
    assert abs(np.mean(hitting)-.2) < .015
    assert abs(np.mean(junction)-.5) < .015
    np.testing.assert_allclose(np.mean(np.asarray(direction)**2,axis=0), 1/3, atol=.02)


def test_triangle_exact_returns_and_plateau():
    triangle = np.ones((3,3))-np.eye(3)
    result = exact_return_probabilities(triangle, max_steps=50, lazy=.5)
    # Spectrum of lazy triangle walk is (1,1/4,1/4).
    expected = (1+2*.25**np.arange(51))/3
    np.testing.assert_allclose(result["return_probability"], expected, atol=1e-13)


def test_mc_weighted_returns_match_exact_conditional_on_graph():
    matrix = np.array([[0,1,2],[1,0,3],[2,3,0]], dtype=float)
    result = monte_carlo_returns(sparse.csr_matrix(matrix), np.random.default_rng(42),
                                max_steps=16, walkers=100000)
    exact = exact_return_probabilities(matrix, max_steps=16)
    # Six standard errors is a loose deterministic-seed regression tolerance.
    assert np.all(np.abs(result["return_probability"]-exact["return_probability"])
                  <= 6*result["conditional_standard_error"]+1e-10)
    assert result["stationary_plateau_uniform_root"] == 1/3
    assert result["stationary_plateau_stationary_root"] > 1/3


def test_triangle_effective_resistance():
    triangle = np.ones((3,3))-np.eye(3)
    np.testing.assert_allclose(effective_resistances(triangle, [(0,1),(0,2),(1,1)]),
                               [2/3,2/3,0], atol=1e-12)


def test_spectral_pilot_rejects_plateau():
    result = exact_return_probabilities(np.ones((3,3))-np.eye(3), max_steps=64)
    assert spectral_dimension_pilot(result)["estimate"] is None


def test_neighbor_list_secondary_interface():
    result = exact_return_probabilities([np.array([1,2]),np.array([0,2]),np.array([0,1])],
                                        max_steps=4)
    np.testing.assert_allclose(result["return_probability"],
                               (1+2*.25**np.arange(5))/3)
