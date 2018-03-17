"""
Tests for markov/utilities.py

"""
from numpy.testing import assert_array_equal
from quantecon.markov import sa_indices


def test_sa_indices():
    num_states, num_actions = 3, 4
    s_expected = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    a_expected = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

    s, a = sa_indices(num_states, num_actions)

    for indices, indices_expected in zip([s, a], [s_expected, a_expected]):
        assert_array_equal(indices, indices_expected)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
