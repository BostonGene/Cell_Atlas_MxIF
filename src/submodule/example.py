from mxifpublic.submodule.submodule_c_name import test_sqr_c
import numpy as np


def test_sqrt(array: np.ndarray) -> np.ndarray:
    """
    Test function
    arr = np.random.rand(10000)
    res = test_sqrt(arr)
    :param array: 1D np.float64 array.
    :return: return sqrt of array.
    """
    return test_sqr_c(array)
