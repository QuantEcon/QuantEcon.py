"""
Tests for Array Utilities

Functions
---------
searchsorted

"""
import numpy as np
from nose.tools import eq_
from quantecon.util import searchsorted


def test_searchsorted():
    a = np.array([0.2, 0.4, 1.0])
    eq_(searchsorted(a, 0.1), 0)
    eq_(searchsorted(a, 0.4), 2)
    eq_(searchsorted(a, 2), 3)

    a = np.ones(0)
    for (v, i) in zip([0, 1, 2], [0, 0, 0]):
        eq_(searchsorted(a, v), i)

    a = np.ones(1)
    for (v, i) in zip([0, 1, 2], [0, 1, 1]):
        eq_(searchsorted(a, v), i)

    a = np.ones(2)
    for (v, i) in zip([0, 1, 2], [0, 2, 2]):
        eq_(searchsorted(a, v), i)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
