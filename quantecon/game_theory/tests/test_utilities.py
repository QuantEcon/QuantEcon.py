"""
Author: Quentin Batista

Tests for lemke_howson.py

"""
import numpy as np
from math import pi, cos, sin
from numpy.testing import assert_allclose
from quantecon.game_theory import RGUtil


class TestRGUtilites():
    def test_frange(self):
        rng = [0., 1/3, 2/3, 1., 4/3, 5/3, 2., 7/3, 
               8/3, 3., 10/3, 11/3, 4., 13/3, 14/3, 5.]

        start = 0.
        stop = 5.
        step = 1/3
        test_obj = [x for x in RGUtil.frange(start, stop, step)]
        assert_allclose(test_obj, rng)

    def test_unitcircle(self):
        incr = 2*pi/5
        pts = np.array([[cos(0. * incr), sin(0. * incr)],
                        [cos(1. * incr), sin(1. * incr)],
                        [cos(2. * incr), sin(2. * incr)],
                        [cos(3. * incr), sin(3. * incr)],
                        [cos(4. * incr), sin(4. * incr)]])

        npts = 5
        test_obj = RGUtil.unitcircle(npts)
        assert_allclose(test_obj, pts)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
