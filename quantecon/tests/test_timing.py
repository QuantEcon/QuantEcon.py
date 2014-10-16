"""
Filename: timing.py
Authors: Pablo Winant
Tests for timing.py
"""

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

    assert(abs(el1-h)<0.01)
    assert(abs(el2-h)<0.01)
    assert(abs(el3-h*3)<0.01)


if __name__ == "__main__":

    test_tic_tac_toc()
