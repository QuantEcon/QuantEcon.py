"""
Filename: timing.py
Authors: Pablo Winant
Tests for timing.py
"""
from numpy.testing import assert_allclose


def test_tic_tac_toc():

    from ..timing import tic, tac, toc
    import time

    h = 0.1

    tic()

    time.sleep(h)
    el1 = tac()

    time.sleep(h)
    el2 = tac()

    time.sleep(h)
    el3 = toc()

    rtol = 0.1
    for actual, desired in zip([el1, el2, el3], [h, h, h*3]):
        assert_allclose(actual, desired, rtol=rtol)


if __name__ == "__main__":

    test_tic_tac_toc()
