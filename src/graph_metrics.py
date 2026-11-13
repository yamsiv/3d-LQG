"""Intrinsic unweighted graph observables, with explicit finite-volume conventions.

Distances use the underlying simple undirected graph: parallel unit edges do not
change shortest paths. Roots are sampled uniformly with replacement, and every
vertex (including the root) is a target, matching the pair law in Budd--Castro
§4.2. A graph realization, rather than an individual distance, is a statistical
replicate. Diameter bounds below are rigorous for the supplied finite graph.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def sparse_adjacency(graph):
    """Build an unweighted CSR once when SciPy is available, else retain lists."""
    if hasattr(graph, "indptr"):
        return graph
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        return graph
    lengths = np.asarray([len(row) for row in graph], dtype=np.int32)
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    indices = np.concatenate(graph).astype(np.int32) if indptr[-1] else np.array([], dtype=np.int32)
    return csr_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr),
                      shape=(len(graph), len(graph)))


def adjacency_lists(graph):
    """Return a sequence of neighbour arrays from CSR or adjacency lists."""
    if hasattr(graph, "indptr") and hasattr(graph, "indices"):
        return [graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
                for i in range(graph.shape[0])]
    return graph


def bfs_distances(graph, root: int) -> np.ndarray:
    """All distances from one root; disconnected entries are -1.

    Lists use O(n+m) BFS; CSR uses SciPy's compiled unweighted Dijkstra search.
    """
    if hasattr(graph, "indptr"):
        from scipy.sparse.csgraph import shortest_path
        if not 0 <= root < graph.shape[0]:
            raise ValueError("root outside graph")
        # SciPy's compiled unweighted search avoids Python per-edge overhead.
        distance = shortest_path(graph, directed=True, indices=root, unweighted=True,
                                 method="D")
        return np.where(np.isfinite(distance), distance, -1).astype(np.int32)
    adjacency = graph
    n = len(adjacency)
    if not 0 <= root < n:
        raise ValueError("root outside graph")
    distance = np.full(n, -1, dtype=np.int32)
    # A Python integer list avoids NumPy scalar overhead in this hot loop.
    depth = [-1] * n
    depth[root] = 0
    queue = deque([root])
    while queue:
        u = queue.popleft()
        du = depth[u] + 1
        for w in adjacency[u]:
            v = int(w)
            if depth[v] == -1:
                depth[v] = du
                queue.append(v)
    distance[:] = depth
    return distance


def is_connected(graph) -> bool:
    return bool(len(adjacency_lists(graph)) and np.all(bfs_distances(graph, 0) >= 0))


def ball_volumes(distances: np.ndarray) -> np.ndarray:
    """V(r), r=0,...,ecc(root), for one connected graph root."""
    distances = np.asarray(distances)
    if distances.size == 0 or np.any(distances < 0):
        raise ValueError("ball volumes require a nonempty connected graph")
    return np.cumsum(np.bincount(distances.astype(int)))


def diameter_bounds(graph, pivots=None, *, exact_limit=128, sweeps=4):
    """Return certified lower/upper bounds using BFS eccentricities.

    For each pivot p, ecc(p) <= diam <= 2 ecc(p). A tighter upper bound is
    max_v min_p[d(p,v)+ecc(p)]. Farthest-point sweeps improve the lower bound.
    For n <= exact_limit all eccentricities are computed and diameter is exact.
    Complexity is O(k(n+m)), O(n) working memory, with k <= n.
    """
    n = graph.shape[0] if hasattr(graph, "indptr") else len(graph)
    if not n:
        raise ValueError("empty graph")
    targets = list(range(n)) if n <= exact_limit else list(pivots or [0])
    upper_each = np.full(n, n - 1, dtype=np.int64)
    lower = 0
    visited = set()
    count = 0
    index = 0
    while index < len(targets):
        p = int(targets[index])
        index += 1
        if p in visited:
            continue
        visited.add(p)
        d = bfs_distances(graph, p)
        if np.any(d < 0):
            raise ValueError("diameter requires a connected graph")
        ecc = int(d.max())
        lower = max(lower, ecc)
        upper_each = np.minimum(upper_each, d.astype(np.int64) + ecc)
        count += 1
        if n > exact_limit and count < sweeps:
            farthest = int(np.argmax(d))
            if farthest not in visited:
                targets.append(farthest)
    upper = int(upper_each.max())
    return {"lower": lower, "upper": upper, "exact": lower == upper,
            "bfs_count": count, "method": "eccentricity_triangle_bounds"}


def degree_observables(graph):
    """Degree statistics of the supplied adjacency (simple graph expected)."""
    degrees = np.asarray([len(set(map(int, row))) for row in adjacency_lists(graph)], dtype=int)
    return {"mean": float(degrees.mean()), "max": int(degrees.max()),
            "min": int(degrees.min()), "histogram": np.bincount(degrees).tolist(),
            "variance": float(degrees.var())}


def metric_sample(graph, rng: np.random.Generator, *, roots=8,
                  exact_diameter_limit=128, diameter_sweeps=4):
    """One graph's distance/ball/diameter data; JSON-serializable output."""
    if roots < 1:
        raise ValueError("roots must be positive")
    graph = sparse_adjacency(graph)
    n = graph.shape[0] if hasattr(graph, "indptr") else len(graph)
    selected = rng.integers(0, n, size=roots)
    histograms = []
    root_means = []
    for root in selected:
        d = bfs_distances(graph, int(root))
        if np.any(d < 0):
            raise ValueError("metric sample requires a connected graph")
        histograms.append(np.bincount(d))
        root_means.append(float(d.mean()))
    width = max(map(len, histograms))
    root_hist = np.zeros((roots, width), dtype=np.int64)
    for i, hist in enumerate(histograms):
        root_hist[i, :len(hist)] = hist
    pooled = root_hist.sum(axis=0)
    radius = np.arange(width)
    cdf = pooled.cumsum() / pooled.sum()
    quantile = lambda q: int(np.searchsorted(cdf, q))
    return {"n": n, "roots": selected.tolist(), "root_histograms": root_hist.tolist(),
            "distance_histogram": pooled.tolist(), "root_mean_distances": root_means,
            "mean_distance": float(np.dot(radius, pooled) / pooled.sum()),
            "median_distance": quantile(.5), "q25_distance": quantile(.25),
            "q75_distance": quantile(.75),
            "mean_ball_volumes": (root_hist.cumsum(axis=1).mean(axis=0)).tolist(),
            "within_graph_root_mean_variance": float(np.var(root_means, ddof=1)) if roots > 1 else 0.,
            "diameter": diameter_bounds(graph, selected[:2].tolist(),
                                        exact_limit=exact_diameter_limit, sweeps=diameter_sweeps),
            "degree": degree_observables(graph),
            "pair_convention": "uniform_independent_vertices_including_self"}


def apparent_ball_exponent(volumes, n, *, min_radius=2, min_fraction=.02,
                           max_fraction=.25, minimum_points=3):
    """A diagnostic intermediate-radius log slope, never a dimension theorem.

    Excludes the microscopic radii and most finite-volume saturation. Returns
    None if the graph has too little usable intermediate range.
    """
    volume = np.asarray(volumes, dtype=float)
    radius = np.arange(len(volume))
    use = ((radius >= min_radius) & (volume >= n * min_fraction)
           & (volume <= n * max_fraction))
    if np.count_nonzero(use) < minimum_points:
        return None
    slope, intercept = np.polyfit(np.log(radius[use]), np.log(volume[use]), 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "r_min": int(radius[use].min()), "r_max": int(radius[use].max()),
            "points": int(use.sum()), "volume_fraction_window": [min_fraction, max_fraction]}
