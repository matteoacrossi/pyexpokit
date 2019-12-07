import unittest
import numpy as np

from expokit import expmv
import scipy.sparse as sp

from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
from timeit import default_timer as timer

class TestExpmv(unittest.TestCase):

    def test_real(self):
        n = 700
        A = sp.rand(n, n, .1) #+ 1j * sp.rand(n, n, .1)
        v = np.random.rand(n)
        t = -1.

        start = timer()
        result = expmv(t, A, v)
        end = timer()
        print("Expokit: {:.4f}".format(end - start))

        start = timer()
        scipy_result = expm_multiply(t*A, v)
        end = timer()
        print("expm_multiply: {:.4f}".format(end - start))
        
        np.testing.assert_almost_equal(result, scipy_result)


    def test_complex(self):
        n = 200
        A = sp.rand(n, n, .1) + 1j * sp.rand(n, n, .1)
        v = np.random.rand(n) + 1j * np.random.rand(n)
        t = -1.j

        start = timer()
        result = expmv(t, A, v)
        end = timer()
        print("Expokit: {:.4f}".format(end - start))

        start = timer()
        scipy_result = expm_multiply(t*A, v)
        end = timer()
        print("expm_multiply: {:.4f}".format(end - start))
        
        np.testing.assert_almost_equal(result, scipy_result)