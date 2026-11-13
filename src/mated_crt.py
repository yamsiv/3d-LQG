"""Linear-time horizontal-visibility construction of paper §§2.5,3.1.

Vertices are zero-based cells. Boundary cycle is shared once, while diagonals
retain coordinate provenance and multiplicity. Unweighted metric uses simple
graph because parallel unit edges cannot change shortest path length.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MatedCRTGraph:
    adjacency: list
    edges: np.ndarray
    multiplicities: np.ndarray
    coordinate_edges: tuple
    metadata: dict

    @property
    def n(self):
        return len(self.adjacency)

    def to_sparse(self, multigraph=False):
        from scipy.sparse import csr_matrix
        u,v = self.edges.T
        data = self.multiplicities if multigraph else np.ones(len(u),dtype=np.int32)
        return csr_matrix((np.r_[data,data],(np.r_[u,v],np.r_[v,u])),shape=(self.n,self.n))


def cell_minima(sample, n):
    if int(n)!=n or n<3:
        raise ValueError("n must be an integer at least 3")
    times,values = sample.times,sample.values
    if times[0] != 0 or times[-1] != 1 or np.any(np.diff(times)<=0):
        raise ValueError("Excursion times must increase from zero to one")
    boundaries = np.linspace(0,1,n+1)
    positions = np.searchsorted(times,boundaries)
    positions = np.minimum(positions,len(times)-1)
    nearby = np.maximum(positions-1,0)
    present = (np.isclose(times[positions],boundaries,rtol=0,atol=1e-13)
               | np.isclose(times[nearby],boundaries,rtol=0,atol=1e-13))
    if not np.all(present):
        raise ValueError("The sample grid must contain every graph-cell boundary")
    result = np.full((n,values.shape[1]),np.inf)
    segment_values = sample.segment_minima if sample.segment_minima is not None else np.minimum(values[:-1],values[1:])
    # Each fine segment belongs to exactly one cell; junction U is an extra split.
    owners = np.minimum((0.5*(times[:-1]+times[1:])*n).astype(int),n-1)
    np.minimum.at(result,owners,segment_values)
    result[0] = result[-1] = 0
    if np.any(~np.isfinite(result)):
        raise ValueError("Every graph cell must contain path segments")
    return result


def _rank_minima(minima):
    """PL ties resolved by lexicographic (height,cell-index) infinitesimals.

    Exact continuous minima have no interior ties almost surely. End cells both
    retain rank zero to represent the root. Tie count is exposed to experiments.
    """
    m = np.asarray(minima,float)
    if m.ndim!=1 or len(m)<3 or np.any(~np.isfinite(m)):
        raise ValueError("Need at least three finite cell minima")
    if m[0] != m[-1] or np.any(m[1:-1] <= m[0]):
        raise ValueError("End-cell minima must be equal and strictly below all interior minima")
    interior = m[1:-1]
    order = np.lexsort((np.arange(len(interior)),interior))
    ranks = np.zeros(len(m),dtype=np.int64)
    ranks[order+1] = np.arange(1,len(m)-1)
    ties = len(interior)-len(np.unique(interior))
    return ranks,int(ties)


def triangulation_reference(minima):
    """Quadratic independent specification; maxima endpoints < all interior."""
    m,_ = _rank_minima(minima)
    n = len(m)
    edges=[]
    for i in range(n):
        running = np.inf
        for j in range(i+2,n):
            running = min(running,m[j-1])
            if (i,j)!=(0,n-1) and max(m[i],m[j]) < running:
                edges.append((i,j))
    return np.asarray(edges,dtype=np.int64).reshape(-1,2)


def triangulation_fast(minima):
    """Monotone stack: O(n) after O(n log n) tie-ranking preprocessing.

    With distinct minima the rank pass is unnecessary; comparisons below use
    lexicographic ranks solely to ensure a triangulation for finite PL ties.
    """
    m,_ = _rank_minima(minima)
    stack=[]
    edges=[]
    n=len(m)
    for j in range(n):
        while stack and m[j] < m[stack[-1]]:
            i=stack.pop()
            if j-i>1 and (i,j)!=(0,n-1):
                edges.append((i,j))
        if stack:
            i=stack[-1]
            if j-i>1 and (i,j)!=(0,n-1):
                edges.append((i,j))
        stack.append(j)
    return np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2)


def graph_from_minima(minima, metadata=None, reference=False):
    minima=np.asarray(minima,float)
    if minima.ndim==1:
        minima=minima[:,None]
    if minima.ndim!=2 or minima.shape[1]<1:
        raise ValueError("minima must have shape (n,dimension)")
    n,d=minima.shape
    function=triangulation_reference if reference else triangulation_fast
    coordinate_edges=tuple(function(minima[:,k]) for k in range(d))
    counts={(i,i+1):1 for i in range(n-1)}
    counts[(0,n-1)]=1
    for edges in coordinate_edges:
        if len(edges)!=n-3:
            raise RuntimeError("Coordinate failed triangulation diagonal count")
        for edge in edges:
            pair=tuple(map(int,edge))
            counts[pair]=counts.get(pair,0)+1
    edges=np.asarray(sorted(counts),dtype=np.int64)
    multiplicities=np.array([counts[tuple(edge)] for edge in edges],dtype=np.int32)
    adjacency=[[] for _ in range(n)]
    for i,j in edges:
        adjacency[i].append(int(j)); adjacency[j].append(int(i))
    adjacency=[np.asarray(sorted(neighbors),dtype=np.int32) for neighbors in adjacency]
    meta=dict(metadata or {})
    meta.update({"n":n,"dimension":d,"metric":"underlying_simple_graph_unit_edges",
                 "cycle_multiplicity":1,"diagonal_occurrences":d*(n-3),
                 "symbolic_ties_by_coordinate":[_rank_minima(minima[:,k])[1] for k in range(d)]})
    return MatedCRTGraph(adjacency,edges,multiplicities,coordinate_edges,meta)


def graph_from_excursion(sample,n,reference=False):
    return graph_from_minima(cell_minima(sample,n),sample.metadata,reference)


def validate_graph(graph):
    n=graph.n
    edges=set(map(tuple,graph.edges))
    cycle=all((i,i+1) in edges for i in range(n-1)) and (0,n-1) in edges
    visited={0}; stack=[0]
    while stack:
        for v in graph.adjacency[stack.pop()]:
            v=int(v)
            if v not in visited:
                visited.add(v); stack.append(v)
    return {"connected":len(visited)==n,"hamiltonian_cycle":cycle,
            "coordinate_diagonal_counts":[len(e) for e in graph.coordinate_edges],
            "multigraph_edge_count":int(graph.multiplicities.sum()),
            "simple_edge_count":len(graph.edges)}
