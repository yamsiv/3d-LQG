import unittest
import numpy as np
from src.mated_crt import (triangulation_fast,triangulation_reference,graph_from_minima,
                           graph_from_excursion,validate_graph)
from src.brownian_excursions import sample_independent_excursion

class AdjacencyTests(unittest.TestCase):
    def test_hand_cases(self):
        cases=[([0,5,2,7,3,0],{(0,2),(2,4),(2,5)}),
               ([0,3,7,2,5,0],{(0,3),(1,3),(3,5)})]
        for m,expected in cases:
            self.assertEqual(set(map(tuple,triangulation_fast(m))),expected)
    def test_fast_reference_and_noncrossing(self):
        rng=np.random.default_rng(22)
        for n in range(3,81):
            m=np.r_[0,rng.permutation(np.arange(1,n-1)),0]
            fast=triangulation_fast(m)
            np.testing.assert_array_equal(fast,triangulation_reference(m))
            self.assertEqual(len(fast),n-3)
            for a,b in fast:
                for c,d in fast:
                    self.assertFalse(a<c<b<d or c<a<d<b)
    def test_cycle_multiplicity_connectivity(self):
        m=np.array([0,5,2,7,3,0])
        graph=graph_from_minima(np.column_stack([m,m,m]))
        self.assertEqual(graph.multiplicities.sum(),6+3*3)
        self.assertEqual(sorted(graph.multiplicities.tolist()),[1]*6+[3]*3)
        checks=validate_graph(graph)
        self.assertTrue(checks['connected'] and checks['hamiltonian_cycle'])
    def test_exact_sample(self):
        s=sample_independent_excursion(1024,3,np.random.default_rng(177),'biased')
        graph=graph_from_excursion(s,1024)
        self.assertEqual(graph.metadata['symbolic_ties_by_coordinate'],[0,0,0])
        self.assertTrue(validate_graph(graph)['connected'])
    def test_symbolic_ties_reported(self):
        graph=graph_from_minima([0,2,2,1,0])
        self.assertEqual(graph.metadata['symbolic_ties_by_coordinate'],[1])
    def test_sparse_preserves_parallel_edges(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('scipy unavailable')
        m=np.array([0,5,2,7,3,0])
        graph=graph_from_minima(np.column_stack([m,m,m]))
        sparse=graph.to_sparse()
        multi=graph.to_sparse(multigraph=True)
        self.assertEqual(sparse[0,2],1)
        self.assertEqual(multi[0,2],3)
        self.assertEqual(multi[0,1],1)
        self.assertEqual((multi-multi.T).nnz,0)
        self.assertEqual(int(multi.sum()),2*int(graph.multiplicities.sum()))
