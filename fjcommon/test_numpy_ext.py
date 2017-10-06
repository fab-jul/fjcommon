from unittest import TestCase
from . import numpy_ext
import numpy as np


class TestNumpyExt(TestCase):
    def test_reshape_into_tiles(self):
        for C in range(1, 12):
            test = np.stack([np.ones((2, 2)) * c for c in range(C)], 0)
            test = np.transpose(test, (1, 0, 2))
            assert test.shape == (2, C, 2)
            o = numpy_ext.reshape_into_tiles(test, 1)
            print(o)

