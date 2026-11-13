"""Finite-grid LFPP metrics with explicit stencil and boundary conventions.

These weighted graphs approximate line integrals of the trilinear interpolant
of exp(xi*h). Fixed directional stencils have a polyhedral xi=0 metric: adding
grid points alone does not remove that anisotropy. No continuum exponent or
gamma--xi identification is built into this module.
"""
from __future__ import annotations

from itertools import product
from math import gcd
import time

import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.csgraph import dijkstra
from scipy.special import exprel


def stencil_offsets(stencil="26"):
    """All directed primitive integer offsets, in a deterministic order."""
    if stencil == "axis":
        limit = 1
        offsets = [x for x in product(range(-1, 2), repeat=3) if sum(abs(v) for v in x) == 1]
    elif stencil in {"26", "primitive2"}:
        limit = 1 if stencil == "26" else 2
        offsets = [x for x in product(range(-limit, limit+1), repeat=3)
                   if any(x) and gcd(gcd(abs(x[0]), abs(x[1])), abs(x[2])) == 1]
    else:
        raise ValueError("stencil must be axis, 26, or primitive2")
    return np.asarray(offsets, dtype=np.int32)


def _interpolate_grid(values, points, periodic):
    """Trilinear interpolation at positions expressed in grid-index units."""
    n = values.shape[0]
    points = np.asarray(points, dtype=float)
    if periodic:
        points = points % n
    elif np.any(points < -1e-12) or np.any(points > n-1+1e-12):
        raise ValueError("Interpolation outside the open grid")
    lower = np.floor(points).astype(np.int32)
    fraction = points-lower
    if periodic:
        lower %= n
        upper = (lower+1) % n
    else:
        lower = np.clip(lower, 0, n-1)
        upper = np.minimum(lower+1, n-1)
    result = np.zeros(len(points))
    for bits in product((0, 1), repeat=3):
        index = tuple(np.where(bits[k], upper[:, k], lower[:, k]) for k in range(3))
        weight = np.prod(np.where(np.asarray(bits), fraction, 1-fraction), axis=1)
        result += weight*values[index]
    return result


def build_lfpp_graph(h, xi, *, stencil="26", boundary="torus", quadrature_per_grid=2,
                     interpolation="weight"):
    """Return (CSR,metadata) for weights integral exp(xi*h) ds.

    Grid points have positions i/N; a cube spans [0,(N-1)/N]^3. For axis
    edges the exact integral of the trilinear *weight* interpolant is endpoint
    trapezoidal. Longer edges use composite trapezoidal quadrature with
    ceil(quadrature_per_grid*|offset|) subdivisions. No GMC variance subtraction
    belongs in these metric weights. Torus wrapping is explicit. Default
    interpolation='weight' uses trilinear exp(xi*h); optional 'field' integrates
    exp(xi*trilinear(h)). These differ at finite resolution. The field option's
    axis integral is exact for a linear field, rather than trapezoidal.
    """
    started = time.perf_counter()
    h = np.asarray(h, dtype=float)
    if h.ndim != 3 or len(set(h.shape)) != 1 or h.shape[0] < 5:
        raise ValueError("h must be a cubic grid of side N>=5")
    if not np.all(np.isfinite(h)) or not np.isfinite(xi):
        raise ValueError("Field and xi must be finite")
    if boundary not in {"torus", "cube"}:
        raise ValueError("boundary must be torus or cube")
    if interpolation not in {"weight", "field"}:
        raise ValueError("interpolation must be weight or field")
    if not np.isfinite(quadrature_per_grid) or quadrature_per_grid <= 0:
        raise ValueError("quadrature_per_grid must be positive")
    n = h.shape[0]
    with np.errstate(over="raise", under="raise", invalid="raise"):
        weight = np.exp(float(xi)*h)
    offsets = stencil_offsets(stencil)
    n_vertices, n_directions = n**3, len(offsets)
    coordinates = np.column_stack(np.unravel_index(np.arange(n_vertices), (n, n, n))).astype(np.int32)
    neighbors = np.full((n_vertices, n_directions), -1, dtype=np.int32)
    costs = np.zeros((n_vertices, n_directions), dtype=float)
    reverse_column = {tuple(offset): i for i, offset in enumerate(offsets)}
    vertices = np.arange(n_vertices, dtype=np.int32)
    for column, offset in enumerate(offsets):
        # Calculate each undirected segment once, preserving exact symmetry.
        first_nonzero = offset[np.flatnonzero(offset)[0]]
        if first_nonzero < 0:
            continue
        target_coordinates = coordinates+offset
        if boundary == "cube":
            valid = np.all((target_coordinates >= 0) & (target_coordinates < n), axis=1)
            source = vertices[valid]
            starts = coordinates[valid]
            targets = target_coordinates[valid]
        else:
            source, starts = vertices, coordinates
            targets = target_coordinates % n
        target = np.ravel_multi_index(targets.T, (n, n, n)).astype(np.int32)
        length = float(np.linalg.norm(offset))
        if length == 1:
            if interpolation == "weight":
                edge_cost = (weight.ravel()[source]+weight.ravel()[target])/(2*n)
            else:
                difference = float(xi)*(h.ravel()[target]-h.ravel()[source])
                edge_cost = weight.ravel()[source]*exprel(difference)/n
        else:
            subdivisions = max(1, int(np.ceil(quadrature_per_grid*length)))
            integral = np.zeros(len(source))
            for step in range(subdivisions+1):
                fraction = step/subdivisions
                values = weight if interpolation == "weight" else h
                value = _interpolate_grid(values, starts+fraction*offset, boundary == "torus")
                if interpolation == "field":
                    value = np.exp(float(xi)*value)
                integral += (.5 if step in (0, subdivisions) else 1.)*value/subdivisions
            edge_cost = length/n*integral
        reverse = reverse_column[tuple(-offset)]
        neighbors[source, column] = target
        costs[source, column] = edge_cost
        neighbors[target, reverse] = source
        costs[target, reverse] = edge_cost
    valid = neighbors >= 0
    lengths = np.sum(valid, axis=1)
    indptr = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    graph = csr_matrix((costs[valid], neighbors[valid], indptr), shape=(n_vertices, n_vertices))
    graph.sort_indices()
    metadata = {
        "grid_side": n, "n_vertices": n_vertices, "xi": float(xi), "boundary": boundary,
        "stencil": stencil, "directions": offsets.tolist(), "directed_edges": graph.nnz,
        "mesh": 1/n, "cube_span": (n-1)/n,
        "edge_rule": "line integral of trilinear exp(xi*h) interpolant" if interpolation == "weight"
                     else "line integral of exp(xi*trilinear(h))",
        "interpolation": interpolation,
        "quadrature_per_grid": float(quadrature_per_grid),
        "axis_rule": "endpoint trapezoid; differs from paper vertex-sum by endpoint halves"
                     if interpolation == "weight" else "exact exponential integral of linear field",
        "gmc_variance_subtraction": False,
        "xi_zero_metric": "L1" if stencil == "axis" else "polyhedral directional-stencil norm",
        "construction_seconds": time.perf_counter()-started,
        "csr_bytes": int(graph.data.nbytes+graph.indices.nbytes+graph.indptr.nbytes),
        "limitations": ["Fixed stencil anisotropy persists under spatial refinement",
                        "Finite-volume torus and cube laws differ",
                        "Trilinear weight interpolation is a finite regularization convention"],
    }
    graph.lfpp_boundary = boundary
    graph.lfpp_grid_side = n
    return graph, metadata


def root_distances(graph, roots):
    """All vertex distances for each requested root, preserving duplicates."""
    roots = np.atleast_1d(np.asarray(roots, dtype=np.int64))
    if np.any(roots < 0) or np.any(roots >= graph.shape[0]):
        raise ValueError("Root outside graph")
    return dijkstra(graph, directed=True, indices=roots)


def cube_crossing(graph, n, axis=0, *, site_weights=None):
    """Shortest distance between opposite grid faces; graph must be open cube.

    With site_weights=exp(xi*h), source/target half-site costs reproduce the
    paper's epsilon*sum_vertex_weights convention exactly on the supplied
    graph. In that usage the graph should use axis edges for CHZ Eq.(1.1).
    Without site_weights, this returns the proper edge-path crossing metric.
    Caller is responsible for supplying a cube graph. A torus graph would
    produce wraparound shortcuts and an invalid cube-crossing observable.
    """
    if axis not in {0, 1, 2} or graph.shape != (n**3, n**3):
        raise ValueError("Invalid cube/axis")
    if getattr(graph, "lfpp_boundary", None) == "torus":
        raise ValueError("Cube crossing cannot use a periodic torus graph")
    indices = np.arange(n**3).reshape((n, n, n))
    start = np.take(indices, 0, axis=axis).ravel()
    end = np.take(indices, n-1, axis=axis).ravel()
    if site_weights is None:
        distances = dijkstra(graph, directed=True, indices=start, min_only=True)
        return float(np.min(distances[end]))
    weight = np.asarray(site_weights, float)
    if weight.shape != (n, n, n) or np.any(~np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("site_weights must be a positive finite cubic grid")
    # A directed auxiliary source allows distinct initial half-site costs.
    connection = csr_matrix((weight.ravel()[start]/(2*n), (np.zeros(len(start), int), start)),
                            shape=(1, n**3))
    augmented = bmat([[graph, None], [connection, csr_matrix((1, 1))]], format="csr")
    # bmat's None column infers zero width without an explicit upper block.
    if augmented.shape != (n**3+1, n**3+1):
        raise RuntimeError("Invalid augmented graph shape")
    distances = dijkstra(augmented, directed=True, indices=n**3)
    return float(np.min(distances[end]+weight.ravel()[end]/(2*n)))


def paper_vertex_distances(graph, roots, site_weights):
    """CHZ vertex-sum path costs from proper trapezoidal axis-edge distances.

    Its finite-grid diagonal is epsilon*weight(root), unlike a proper metric.
    For graph metric-measure comparisons use root_distances instead.
    """
    weight = np.asarray(site_weights, float).ravel()
    roots = np.atleast_1d(np.asarray(roots, int))
    n = round(graph.shape[0]**(1/3))
    if n**3 != graph.shape[0] or len(weight) != n**3 or np.any(weight <= 0) or np.any(~np.isfinite(weight)):
        raise ValueError("Invalid positive site weights for cubic graph")
    return root_distances(graph, roots)+(weight[roots, None]+weight[None, :])/(2*n)


def exact_flat_distances(n, roots, targets=None, boundary="torus", norm="euclidean"):
    """Flat Euclidean/L1 reference on the supplied grid, independent of CSR."""
    roots = np.atleast_1d(roots)
    targets = np.arange(n**3) if targets is None else np.atleast_1d(targets)
    r = np.column_stack(np.unravel_index(roots, (n, n, n)))
    t = np.column_stack(np.unravel_index(targets, (n, n, n)))
    delta = np.abs(r[:, None, :]-t[None, :, :])
    if boundary == "torus":
        delta = np.minimum(delta, n-delta)
    elif boundary != "cube":
        raise ValueError("boundary must be torus or cube")
    return (np.sum(delta, axis=-1) if norm == "L1" else np.linalg.norm(delta, axis=-1))/n
