"""Finite-graph random-walk diagnostics and small-graph resistance.

These observables never identify an asymptotic spectral dimension on their own.
An adjacency entry is a conductance: passing multiplicities gives the multigraph
walk, while passing a boolean/simple adjacency gives the simple-graph walk.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _adjacency(adjacency):
    if sparse.issparse(adjacency) or isinstance(adjacency, np.ndarray):
        matrix = sparse.csr_matrix(adjacency, dtype=float)
    else:
        # A list/tuple of neighbor arrays is the generator's simple adjacency.
        rows, columns = [], []
        for vertex, neighbors in enumerate(adjacency):
            rows.extend([vertex]*len(neighbors))
            columns.extend(neighbors)
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                                   shape=(len(adjacency), len(adjacency)))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("adjacency must be nonempty and square")
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("conductances must be finite and nonnegative")
    if (matrix != matrix.T).nnz or np.any(matrix.diagonal()):
        raise ValueError("an undirected loopless adjacency is required")
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("walk requires positive degree at every vertex")
    if connected_components(matrix, directed=False, return_labels=False) != 1:
        raise ValueError("observables require a connected graph")
    return matrix, degree


def exact_return_probabilities(adjacency, max_steps=128, lazy=0.5):
    """Exact normalized trace of P^t; roots are uniform over vertices.

    O(max_steps * n * edges) work and O(n^2) storage. This routine is intended
    only for small graphs. A connected lazy graph has limiting trace/n = 1/n,
    even when its stationary measure is nonuniform.
    """
    matrix, degree = _adjacency(adjacency)
    if not 0 <= lazy < 1 or max_steps < 0:
        raise ValueError("invalid holding probability or step count")
    n = matrix.shape[0]
    if n > 2048:
        raise ValueError("use Monte Carlo return probabilities above 2048 vertices")
    transition = (1-lazy) * sparse.diags(1/degree) @ matrix + lazy*sparse.eye(n)
    distribution = np.eye(n)
    returns = np.ones(max_steps+1)
    for time in range(1, max_steps+1):
        distribution = transition @ distribution
        returns[time] = np.trace(distribution)/n
    return {"times": np.arange(max_steps+1), "return_probability": returns,
            "stationary_plateau_uniform_root": 1/n,
            "stationary_plateau_stationary_root": float(np.sum((degree/degree.sum())**2)),
            "root_law": "uniform", "holding_probability": lazy, "method": "exact_trace"}


def monte_carlo_returns(adjacency, rng, max_steps=256, walkers=20000, lazy=0.5):
    """Uniform independent roots and one weighted walk per root.

    Binomial standard errors describe walk/root sampling conditional on THIS
    graph. Independent graph realizations are still needed for ensemble errors.
    The cumulative-conductance search supports parallel-edge multiplicities.
    """
    matrix, degree = _adjacency(adjacency)
    if not 0 <= lazy < 1 or max_steps < 0 or walkers < 1:
        raise ValueError("invalid walk configuration")
    n = matrix.shape[0]
    roots = rng.integers(n, size=walkers)
    positions = roots.copy()
    counts = np.zeros(max_steps+1, dtype=np.int64)
    counts[0] = walkers
    cumulative = np.cumsum(matrix.data)
    starts = np.zeros(n)
    has_predecessor = matrix.indptr[:-1] > 0
    starts[has_predecessor] = cumulative[matrix.indptr[:-1][has_predecessor]-1]
    for time in range(1, max_steps+1):
        move = rng.random(walkers) >= lazy
        current = positions[move]
        targets = starts[current] + rng.random(len(current))*degree[current]
        chosen = np.searchsorted(cumulative, targets, side="right")
        positions[move] = matrix.indices[chosen]
        counts[time] = np.count_nonzero(positions == roots)
    probability = counts/walkers
    return {"times": np.arange(max_steps+1), "return_probability": probability,
            "return_counts": counts, "walkers": walkers,
            "conditional_standard_error": np.sqrt(probability*(1-probability)/walkers),
            "stationary_plateau_uniform_root": 1/n,
            "stationary_plateau_stationary_root": float(np.sum((degree/degree.sum())**2)),
            "root_law": "uniform", "holding_probability": lazy,
            "method": "independent_walks_conditional_on_graph"}


def spectral_dimension_pilot(walk_result, time_min=8, time_max=None, plateau_factor=5):
    """Windowed slope only, with an explicit finite-volume reliability filter.

    No confidence interval is returned because adjacent times and common graphs
    are dependent. Bootstrap independent graph realizations externally.
    """
    times = np.asarray(walk_result["times"])
    probability = np.asarray(walk_result["return_probability"])
    if time_max is None:
        time_max = int(times[-1])
    mask = ((times >= time_min) & (times <= time_max) &
            (probability > plateau_factor*walk_result["stationary_plateau_uniform_root"]))
    if "return_counts" in walk_result:
        mask &= np.asarray(walk_result["return_counts"]) >= 20
    if np.count_nonzero(mask) < 4:
        return {"estimate": None, "reason": "insufficient pre-plateau times", "times_used": []}
    slope, intercept = np.polyfit(np.log(times[mask]), np.log(probability[mask]), 1)
    return {"estimate": float(-2*slope), "times_used": times[mask].tolist(),
            "log_intercept": float(intercept), "status": "finite_graph_window_slope_only"}


def effective_resistances(adjacency, pairs):
    """Exact dense pseudoinverse formula, restricted to connected n<=512 graphs."""
    matrix, degree = _adjacency(adjacency)
    n = matrix.shape[0]
    if n > 512:
        raise ValueError("dense resistance computation is limited to n<=512")
    if connected_components(matrix, directed=False, return_labels=False) != 1:
        raise ValueError("resistance function requires a connected graph")
    laplacian = np.diag(degree)-matrix.toarray()
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    result = []
    for i,j in pairs:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("pair vertex out of range")
        result.append(float(max(0, inverse[i,i]+inverse[j,j]-2*inverse[i,j])))
    return np.asarray(result)
