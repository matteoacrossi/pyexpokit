import unittest
import numpy as np

from pyexpokit import expmv
import scipy.sparse as sp

from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
from timeit import default_timer as timer

class TestExpmv(unittest.TestCase):

    def test_real(self):
        n = 700
        A = sp.rand(n, n, .1) #+ 1j * sp.rand(n, n, .1)
        v = np.random.rand(n)
        t = -10.
        
        self.compare_with_scipy(A, v, t)

    def test_complex(self):
        n = 700
        A = sp.rand(n, n, .1) + 1j * sp.rand(n, n, .1)
        v = np.random.rand(n) + 1j * np.random.rand(n)
        t = -10.j
        self.compare_with_scipy(A, v, t)

    def compare_with_scipy(self, A, v, t):
        start = timer()
        result = expmv(t, A, v)
        end = timer()
        print("Expokit: {:.4f}".format(end - start))

        start = timer()
        scipy_result = expm_multiply(t*A, v)
        end = timer()
        print("expm_multiply: {:.4f}".format(end - start))
        
        np.testing.assert_allclose(result, scipy_result)