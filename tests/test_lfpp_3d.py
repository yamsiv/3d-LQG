import numpy as np
import pytest
from scipy.integrate import quad
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.csgraph import dijkstra

from src.lfpp_3d import (build_lfpp_graph, cube_crossing, exact_flat_distances,
                         root_distances, stencil_offsets, _interpolate_grid,
                         paper_vertex_distances)


def test_stencil_sizes_and_primitivity():
    assert [len(stencil_offsets(x)) for x in ("axis", "26", "primitive2")] == [6, 26, 98]
    for name in ("axis", "26", "primitive2"):
        offsets = stencil_offsets(name)
        assert {tuple(x) for x in offsets} == {tuple(-x) for x in offsets}
    with pytest.raises(ValueError):
        stencil_offsets("unknown")


@pytest.mark.parametrize("boundary", ["cube", "torus"])
def test_axis_zero_is_exact_L1_and_cube_faces(boundary):
    n = 7
    graph, meta = build_lfpp_graph(np.zeros((n,)*3), 0, stencil="axis", boundary=boundary)
    roots = np.array([0, 118, n**3-1])
    np.testing.assert_allclose(root_distances(graph, roots),
                               exact_flat_distances(n, roots, boundary=boundary, norm="L1"), atol=1e-14)
    assert meta["gmc_variance_subtraction"] is False
    if boundary == "cube":
        for axis in range(3):
            assert cube_crossing(graph, n, axis) == pytest.approx((n-1)/n)


def test_richer_stencils_reduce_but_do_not_eliminate_flat_anisotropy():
    n = 11
    target = np.ravel_multi_index((4, 2, 1), (n,)*3)
    distances = []
    for stencil in ("axis", "26", "primitive2"):
        graph, _ = build_lfpp_graph(np.zeros((n,)*3), 0, stencil=stencil)
        distances.append(root_distances(graph, [0])[0, target])
    euclidean = np.sqrt(21)/n
    assert distances[0] > distances[1] > distances[2] > euclidean
    # A primitive direction present in radius two is exact when xi=0.
    graph, _ = build_lfpp_graph(np.zeros((n,)*3), 0, stencil="primitive2")
    target = np.ravel_multi_index((4, 2, 0), (n,)*3)
    assert root_distances(graph, [0])[0, target] == pytest.approx(np.sqrt(20)/n)


def test_weights_symmetric_and_constant_shift_scales_distances():
    rng = np.random.default_rng(8392)
    h = rng.normal(size=(5,)*3)
    graph, _ = build_lfpp_graph(h, .3, stencil="26")
    other, _ = build_lfpp_graph(h+2, .3, stencil="26")
    assert np.max(abs((graph-graph.T).data), initial=0) == 0
    np.testing.assert_allclose(other.data, np.exp(.6)*graph.data, rtol=1e-14)
    assert np.all(np.isfinite(root_distances(graph, [0])))


def test_long_edges_do_not_skip_an_intermediate_high_weight():
    n = 7
    h = np.zeros((n,)*3)
    h[1, 0, 0] = 6
    graph, _ = build_lfpp_graph(h, 1, stencil="primitive2", boundary="cube", quadrature_per_grid=8)
    destination = np.ravel_multi_index((2, 1, 0), (n,)*3)
    observed = graph[0, destination]
    endpoint_only = np.sqrt(5)/n
    exact = np.sqrt(5)/n*quad(lambda t: _interpolate_grid(np.exp(h), np.array([[2*t, t, 0]]), False)[0],
                             0, 1, points=[.5], epsabs=1e-11)[0]
    assert observed > 20*endpoint_only
    assert abs(observed/exact-1) < .01


def test_grid_and_overflow_errors_are_explicit():
    with pytest.raises(ValueError):
        build_lfpp_graph(np.zeros((4,)*3), 0)
    with pytest.raises(FloatingPointError):
        build_lfpp_graph(np.full((5,)*3, 1000.), 1)


def test_paper_vertex_sum_endpoints_and_face_crossing():
    n = 5
    graph, _ = build_lfpp_graph(np.zeros((n,)*3), 0, stencil="axis", boundary="cube")
    assert cube_crossing(graph, n, site_weights=np.ones((n,)*3)) == pytest.approx(1.)
    dist = paper_vertex_distances(graph, [0], np.ones((n,)*3))
    assert dist[0, 0] == pytest.approx(1/n)
    assert dist[0, n-1] == pytest.approx(1.)


def test_field_interpolation_integrates_axis_exponential_exactly():
    n = 5
    h = np.zeros((n,)*3)
    h[1, 0, 0] = 2
    field, _ = build_lfpp_graph(h, .5, stencil="axis", interpolation="field")
    weight, _ = build_lfpp_graph(h, .5, stencil="axis", interpolation="weight")
    target = np.ravel_multi_index((1, 0, 0), (n,)*3)
    assert field[0, target] == pytest.approx(np.expm1(1)/n)
    assert field[0, target] < weight[0, target]


def test_nonconstant_paper_crossing_matches_independent_directed_vertex_cost():
    n = 5
    rng = np.random.default_rng(727199)
    weights = np.exp(.4*rng.normal(size=(n,)*3))
    graph, _ = build_lfpp_graph(np.log(weights), 1, stencil="axis", boundary="cube")
    coo = graph.tocoo()
    vertex_graph = csr_matrix((weights.ravel()[coo.col]/n, (coo.row, coo.col)), shape=graph.shape)
    indices = np.arange(n**3).reshape((n,)*3)
    start = indices[0].ravel(); end = indices[-1].ravel()
    source = csr_matrix((weights.ravel()[start]/n, (np.zeros(len(start), int), start)), shape=(1, n**3))
    augmented = bmat([[vertex_graph, None], [source, csr_matrix((1, 1))]], format="csr")
    reference = dijkstra(augmented, directed=True, indices=n**3)[end].min()
    assert cube_crossing(graph, n, site_weights=weights) == pytest.approx(reference, rel=1e-13)
    periodic, _ = build_lfpp_graph(np.zeros((n,)*3), 0, stencil="axis")
    with pytest.raises(ValueError):
        cube_crossing(periodic, n)
