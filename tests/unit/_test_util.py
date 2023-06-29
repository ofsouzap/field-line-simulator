import unittest
import numpy as np

class ArrayComparingTest(unittest.TestCase):

    def compare_arrs(self, a: np.ndarray, b: np.ndarray):

        self.assertEqual(a.ndim, b.ndim, msg="Arrays have different dimensionality")
        self.assertEqual(a.shape, b.shape, msg="Arrays have different shapes")

        self.assertTrue(np.all(np.isclose(a, b)), msg="Arrays' elements differ")
