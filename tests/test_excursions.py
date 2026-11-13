import unittest
import numpy as np
from src.brownian_excursions import (sample_excursion,sample_independent_excursion,
                                    sample_conditioned_bridge,positive_bridge_minima)
from src.mated_crt import cell_minima

class ExcursionTests(unittest.TestCase):
    def test_exact_laws_and_minima(self):
        for dimension in [2,3]:
            for law in ['biased','unbiased']:
                s=sample_independent_excursion(128,dimension,np.random.default_rng(18),law)
                np.testing.assert_array_equal(s.values[[0,-1]],0)
                self.assertTrue(np.all(s.values[1:-1]>0))
                self.assertTrue(np.all(s.segment_minima<=np.minimum(s.values[:-1],s.values[1:])))
                m=cell_minima(s,128)
                self.assertTrue(np.all(m[1:-1]>0))
                self.assertEqual(len(np.unique(m[:,0])),127)
                self.assertTrue(s.metadata['cell_minima_exact'])
    def test_positive_bridge_minimum_distribution(self):
        rng=np.random.default_rng(819)
        m=positive_bridge_minima(np.full(50000,.7),np.full(50000,1.2),.5,rng)
        z=.3
        expected=(-np.expm1(-2*(.7-z)*(1.2-z)/.5))/(-np.expm1(-2*.7*1.2/.5))
        self.assertLess(abs(np.mean(m>z)-expected),.01)
    def test_unbiased_midpoint(self):
        rng=np.random.default_rng(117)
        values=np.array([sample_independent_excursion(4,3,rng,'unbiased').values[2] for _ in range(2000)])
        self.assertLess(abs(np.mean(np.sum(values**2,axis=1))/.25-9),.25)
    def test_hitting_time_and_junction(self):
        rng=np.random.default_rng(222)
        samples=[sample_independent_excursion(4,3,rng,'biased') for _ in range(2000)]
        inv=np.array([.5/np.array(s.metadata['hitting_times']) for s in samples])
        self.assertLess(abs(inv.mean()-3.5),.08)
        u=np.array([s.metadata['junction_time'] for s in samples])
        self.assertLess(abs(u.mean()-.5),.015)
    def test_mcmc_is_positive_and_explicitly_approximate(self):
        c=np.array([[1,-.2,.1],[-.2,1,.3],[.1,.3,1]])
        s=sample_conditioned_bridge(16,c,np.random.default_rng(13),oversample=2,sweeps=2)
        self.assertTrue(np.all(s.values[1:-1]>0))
        self.assertFalse(s.metadata['cell_minima_exact'])
        self.assertEqual(s.metadata['law'],'unbiased_finite_grid_approximation')
        with self.assertRaises(ValueError): sample_excursion(16,c,law='biased')
    def test_reproducibility(self):
        for law in ['biased','unbiased']:
            s=sample_independent_excursion(32,3,np.random.default_rng(781),law)
            t=sample_independent_excursion(32,3,np.random.default_rng(781),law)
            np.testing.assert_array_equal(s.times,t.times)
            np.testing.assert_array_equal(s.values,t.values)
            np.testing.assert_array_equal(s.segment_minima,t.segment_minima)
        c=np.array([[1,-.2,.1],[-.2,1,.3],[.1,.3,1]])
        s=sample_conditioned_bridge(8,c,np.random.default_rng(12),sweeps=2,initial_scale=3)
        t=sample_conditioned_bridge(8,c,np.random.default_rng(12),sweeps=2,initial_scale=3)
        np.testing.assert_array_equal(s.values,t.values)
        self.assertEqual(s.metadata['initial_scale'],3)
    def test_non_dyadic_grid_roundoff(self):
        for n in [5,7,19,31]:
            for oversample in [3,5,7]:
                sample=sample_independent_excursion(n,3,np.random.default_rng(29),oversample=oversample)
                self.assertEqual(cell_minima(sample,n).shape,(n,3))
