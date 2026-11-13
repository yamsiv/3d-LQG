import unittest
import numpy as np
from src.covariance import (from_correlations,from_angles,to_angles,equicorrelated,
                            cone_geometry,orthant_spectral_data,birectangular_spectral_data)

class CovarianceTests(unittest.TestCase):
    def test_user_convention_and_angles(self):
        c=from_correlations(.2,.1,-.3)
        np.testing.assert_allclose(c,from_angles(*to_angles(c)),atol=1e-15)
        self.assertEqual(c[0,2],.1)
        self.assertAlmostEqual(to_angles(c)[1],np.arccos(.3))
    def test_invalid(self):
        for correlations in [(1,0,0),(-.5,-.5,-.5),(.9,.9,-.9)]:
            with self.assertRaises(ValueError): from_correlations(*correlations)
        for rho in [-.5,1]:
            with self.assertRaises(ValueError): equicorrelated(rho)
    def test_full_admissible_family(self):
        rng=np.random.default_rng(19)
        for _ in range(20):
            a=rng.normal(size=(3,3)); c=a@a.T+.01*np.eye(3)
            c/=np.sqrt(np.diag(c)[:,None]*np.diag(c)[None,:])
            g=cone_geometry(c)
            np.testing.assert_allclose(g['cholesky']@g['cholesky'].T,c)
            np.testing.assert_allclose(np.linalg.norm(g['face_normals'],axis=1),1)
    def test_spectral_benchmarks(self):
        self.assertEqual(orthant_spectral_data(3)['lambda1'],12)
        self.assertEqual(orthant_spectral_data(3)['alpha1'],3.5)
        self.assertAlmostEqual(birectangular_spectral_data(np.pi/2)['lambda1'],12)
