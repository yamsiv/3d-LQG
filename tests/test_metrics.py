import unittest
import numpy as np

from src.graph_metrics import (bfs_distances, ball_volumes, diameter_bounds,
                               metric_sample, is_connected, sparse_adjacency)
from src.scaling_analysis import bootstrap_scaling, pooled_histogram, collapse_histograms


def cycle(n):
    return [[(v - 1) % n, (v + 1) % n] for v in range(n)]


class MetricTests(unittest.TestCase):
    def test_cycle_distances(self):
        np.testing.assert_array_equal(bfs_distances(cycle(8), 0), [0, 1, 2, 3, 4, 3, 2, 1])
        np.testing.assert_array_equal(ball_volumes(bfs_distances(cycle(8), 0)), [1, 3, 5, 7, 8])

    def test_diameter_exact_and_bounds(self):
        self.assertEqual(diameter_bounds(cycle(8))["lower"], 4)
        result = diameter_bounds(cycle(80), exact_limit=0, sweeps=4)
        self.assertLessEqual(result["lower"], 40)
        self.assertGreaterEqual(result["upper"], 40)
        self.assertFalse(result["exact"])

    def test_complete_graph(self):
        g = [[v for v in range(6) if v != u] for u in range(6)]
        result = metric_sample(g, np.random.default_rng(123), roots=3)
        self.assertAlmostEqual(result["mean_distance"], 5 / 6)
        self.assertEqual(result["distance_histogram"], [3, 15])
        self.assertEqual(result["diameter"]["lower"], 1)

    def test_disconnected(self):
        g = [[1], [0], []]
        self.assertFalse(is_connected(g))
        with self.assertRaises(ValueError):
            diameter_bounds(g)

    def test_graph_cluster_bootstrap_on_known_power(self):
        # Two equal-weight graph distributions of deterministic distance sqrt(n).
        records = []
        for n in (16, 64, 256, 1024):
            d = int(np.sqrt(n))
            hist = [0] * (d + 1)
            hist[d] = n
            records.extend({"n": n, "distance_histogram": hist} for _ in range(4))
        fit = bootstrap_scaling(records, bootstrap=20, seed=14)
        self.assertAlmostEqual(fit["effective_dimension"], 2)
        self.assertAlmostEqual(fit["dimension_ci95"][0], 2)

    def test_equal_graph_weights(self):
        hist = pooled_histogram([{"distance_histogram": [1, 1]},
                                 {"distance_histogram": [0, 100]}])
        np.testing.assert_allclose(hist, [.25, .75])

    def test_sparse_and_list_distances_agree(self):
        g = cycle(101)
        g[0].append(31)
        g[31].append(0)
        np.testing.assert_array_equal(bfs_distances(g, 17), bfs_distances(sparse_adjacency(g), 17))

    def test_bounds_cover_independently_computed_diameter(self):
        rng = np.random.default_rng(713)
        g = cycle(31)
        for _ in range(25):
            u, v = rng.integers(31, size=2)
            if u != v:
                g[u].append(int(v))
                g[v].append(int(u))
        # Independent dense Floyd--Warshall specification.
        d = np.full((31, 31), np.inf)
        np.fill_diagonal(d, 0.)
        for u, row in enumerate(g):
            d[u, row] = 1.
        for k in range(31):
            d = np.minimum(d, d[:, k, None] + d[k, None, :])
        bounds = diameter_bounds(sparse_adjacency(g), exact_limit=0, sweeps=4)
        self.assertLessEqual(bounds["lower"], d.max())
        self.assertGreaterEqual(bounds["upper"], d.max())

    def test_histogram_collapse_recovers_known_scale_law(self):
        records = []
        for size, scale in [(100, 10), (400, 20), (1600, 40)]:
            r = np.arange(3*scale + 1)
            hist = np.maximum(0., 1 - np.abs(r / scale - 2))
            records.append({"n": size, "distance_histogram": hist.tolist()})
        fit = collapse_histograms(records)
        self.assertAlmostEqual(fit["effective_dimension"], 2., delta=.015)


if __name__ == "__main__":
    unittest.main()
