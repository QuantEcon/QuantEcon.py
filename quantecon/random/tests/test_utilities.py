"""
Tests for util/random.py

Functions
---------
probvec
sample_without_replacement

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from nose.tools import eq_
from quantecon.random import probvec, sample_without_replacement


# probvec #

class TestProbvec:
    def setUp(self):
        self.m, self.k = 2, 3  # m vectors of dimension k
        seed = 1234

        self.out_parallel = probvec(self.m, self.k, random_state=seed)
        self.out_cpu = \
            probvec(self.m, self.k, random_state=seed, parallel=False)

    def test_shape(self):
        for out in [self.out_parallel, self.out_cpu]:
            eq_(out.shape, (self.m, self.k))

    def test_parallel_cpu(self):
        assert_array_equal(self.out_parallel, self.out_cpu)


# sample_without_replacement #

def test_sample_without_replacement_shape():
    assert_array_equal(sample_without_replacement(2, 0).shape, (0,))

    n, k, m = 5, 3, 4
    assert_array_equal(
        sample_without_replacement(n, k).shape,
        (k,)
    )
    assert_array_equal(
        sample_without_replacement(n, k, num_trials=m).shape,
        (m, k)
    )


def test_sample_without_replacement_uniqueness():
    n = 10
    a = sample_without_replacement(n, n)
    b = np.unique(a)
    eq_(len(b), n)


def test_sample_without_replacement_value_error():
    # n <= 0
    assert_raises(ValueError, sample_without_replacement, 0, 2)
    assert_raises(ValueError, sample_without_replacement, -1, -1)

    # k > n
    assert_raises(ValueError, sample_without_replacement, 2, 3)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
