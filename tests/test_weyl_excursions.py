import unittest
import numpy as np
from src.weyl_excursions import (sample_weyl_excursion,weyl_covariance,
                                 weyl_junction_direction,_hermitian_from_coordinates,
                                 _traceless_eigenvalues,thin_weyl_sample)
from src.mated_crt import graph_from_excursion,validate_graph


class WeylExcursionTests(unittest.TestCase):
    def test_whitened_radius_identity(self):
        rng=np.random.default_rng(17)
        eigenvalues=_traceless_eigenvalues(_hermitian_from_coordinates(rng.normal(size=(100,9))))
        gaps=np.diff(eigenvalues,axis=1)/np.sqrt(2)
        whitened=np.linalg.solve(np.linalg.cholesky(weyl_covariance(2)),gaps.T).T
        np.testing.assert_allclose(np.sum(whitened**2,axis=1),np.sum(eigenvalues**2,axis=1),atol=1e-13)
    def test_direction_on_whitened_sphere(self):
        rng=np.random.default_rng(92)
        for d in [2,3]:
            for _ in range(20):
                eigenvalues,height=weyl_junction_direction(d,rng)
                self.assertAlmostEqual(np.sum(eigenvalues**2)+height**2,1)
                self.assertTrue(np.all(np.diff(eigenvalues)>0))
                self.assertAlmostEqual(eigenvalues.sum(),0)
    def test_both_laws_construct_graphs(self):
        for d in [2,3]:
            for law in ['biased','unbiased']:
                s=sample_weyl_excursion(61,d,np.random.default_rng(313),law,oversample=5)
                self.assertTrue(np.all(s.values[1:-1]>0))
                self.assertIsNone(s.segment_minima)
                self.assertTrue(s.metadata['grid_values_exact'])
                self.assertFalse(s.metadata['cell_minima_exact'])
                g=graph_from_excursion(s,61)
                self.assertTrue(validate_graph(g)['connected'])
                self.assertEqual([len(e) for e in g.coordinate_edges],[58]*d)
    def test_unbiased_radial_moment(self):
        rng=np.random.default_rng(711)
        for d,expected in [(2,8),(3,11)]:
            chol=np.linalg.cholesky(weyl_covariance(d))
            radius=[]
            for _ in range(2000):
                s=sample_weyl_excursion(4,d,rng,'unbiased',oversample=1)
                midpoint=np.linalg.solve(chol,s.values[2])
                radius.append(np.sum(midpoint**2)/.25)
            self.assertLess(abs(np.mean(radius)-expected),.28)
    def test_seed_reproducibility(self):
        s=sample_weyl_excursion(17,3,np.random.default_rng(4),'biased',3)
        t=sample_weyl_excursion(17,3,np.random.default_rng(4),'biased',3)
        np.testing.assert_array_equal(s.values,t.values)
        self.assertEqual(s.metadata,t.metadata)
    def test_nested_refinement_preserves_junction(self):
        fine=sample_weyl_excursion(17,3,np.random.default_rng(4),'biased',8)
        coarse=thin_weyl_sample(fine,17,2)
        self.assertIn(fine.metadata['junction_time'],coarse.times)
        self.assertEqual(len(coarse.times),17*2+2)
        self.assertTrue(validate_graph(graph_from_excursion(coarse,17))['connected'])
