"""
Tests for markov/core.py

Functions
---------
    mc_compute_stationary   [Status: Tested in test_markovchain_pmatrices]
    mc_sample_path          [Status: Tested in test_mc_sample_path]

"""
import numpy as np
from scipy import sparse
import itertools
from numpy.testing import (
    assert_allclose, assert_array_equal, assert_array_less, assert_raises,
    assert_, assert_equal
)

from quantecon.markov import (
    MarkovChain, mc_compute_stationary, mc_sample_path
)


def list_of_array_equal(s, t):
    """
    Compare two lists of ndarrays

    s, t: lists of numpy.ndarrays

    """
    assert_(len(s) == len(t))
    all(assert_array_equal(x, y) for x, y in zip(s, t))


# KMR Function
# Useful because it seems to have 1 unit eigvalue, but a second one that
# approaches unity.  Good test of accuracy.
def KMR_Markov_matrix_sequential(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *sequential* move

    N: number of players
    p: level of p-dominance for action 1
       = the value of p such that action 1 is the BR for (1-q, q) for any q > p,
         where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)
    epsilon: mutation probability

    References:
        KMRMarkovMatrixSequential is contributed from https://github.com/oyamad
    """
    P = np.zeros((N+1, N+1), dtype=float)
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = \
            (n/N) * (epsilon * (1/2) +
                     (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
                     )
        P[n, n+1] = \
            ((N-n)/N) * (epsilon * (1/2) +
                         (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
                         )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


# Tests: methods of MarkovChain, mc_compute_stationary #

def test_markovchain_pmatrices():
    """
    Test the methods of MarkovChain, as well as mc_compute_stationary,
    with P matrix and known solutions
    """
    # Matrix with two recurrent classes [0, 1] and [3, 4, 5],
    # which have periods 2 and 3, respectively
    Q = np.zeros((6, 6))
    Q[0, 1], Q[1, 0] = 1, 1
    Q[2, [0, 3]] = 1/2
    Q[3, 4], Q[4, 5], Q[5, 3] = 1, 1, 1
    Q_stationary_dists = \
        np.array([[1/2, 1/2, 0, 0, 0, 0], [0, 0, 0, 1/3, 1/3, 1/3]])

    testset = [
        {'P': np.array([[0.4, 0.6], [0.2, 0.8]]),  # P matrix
         'stationary_dists': np.array([[0.25, 0.75]]),  # Known solution
         'comm_classes': [np.arange(2)],
         'rec_classes': [np.arange(2)],
         'is_irreducible': True,
         'period': 1,
         'is_aperiodic': True,
         'cyclic_classes': [np.arange(2)],
         },
        {'P': sparse.csr_matrix([[0.4, 0.6], [0.2, 0.8]]),
         'stationary_dists': np.array([[0.25, 0.75]]),
         'comm_classes': [np.arange(2)],
         'rec_classes': [np.arange(2)],
         'is_irreducible': True,
         'period': 1,
         'is_aperiodic': True,
         'cyclic_classes': [np.arange(2)],
         },
        {'P': np.array([[0, 1], [1, 0]]),
         'stationary_dists': np.array([[0.5, 0.5]]),
         'comm_classes': [np.arange(2)],
         'rec_classes': [np.arange(2)],
         'is_irreducible': True,
         'period': 2,
         'is_aperiodic': False,
         'cyclic_classes': [np.array([0]), np.array([1])],
         },
        {'P': np.eye(2),
         'stationary_dists': np.array([[1, 0], [0, 1]]),
         'comm_classes': [np.array([0]), np.array([1])],
         'rec_classes': [np.array([0]), np.array([1])],
         'is_irreducible': False,
         'period': 1,
         'is_aperiodic': True,
         },
        # Reducible mc with a unique recurrent class,
        # where n-1 is a transient state
        {'P': np.array([[1, 0], [1, 0]]),
         'stationary_dists': np.array([[1, 0]]),
         'comm_classes': [np.array([0]), np.array([1])],
         'rec_classes': [np.array([0])],
         'is_irreducible': False,
         'period': 1,
         'is_aperiodic': True,
         },
        {'P': Q,
         'stationary_dists': Q_stationary_dists,
         'comm_classes': [np.array([0, 1]), np.array([2]), np.array([3, 4, 5])],
         'rec_classes': [np.array([0, 1]), np.array([3, 4, 5])],
         'is_irreducible': False,
         'period': 6,
         'is_aperiodic': False,
         },
        {'P': sparse.csr_matrix(Q),
         'stationary_dists': Q_stationary_dists,
         'comm_classes': [np.array([0, 1]), np.array([2]), np.array([3, 4, 5])],
         'rec_classes': [np.array([0, 1]), np.array([3, 4, 5])],
         'is_irreducible': False,
         'period': 6,
         'is_aperiodic': False,
         }
    ]

    # Loop Through TestSet #
    for test_dict in testset:
        mc = MarkovChain(test_dict['P'])
        computed = mc.stationary_distributions
        assert_allclose(computed, test_dict['stationary_dists'])

        assert(mc.num_communication_classes == len(test_dict['comm_classes']))
        assert(mc.is_irreducible == test_dict['is_irreducible'])
        assert(mc.num_recurrent_classes == len(test_dict['rec_classes']))
        list_of_array_equal(
            sorted(mc.communication_classes, key=lambda x: x[0]),
            sorted(test_dict['comm_classes'], key=lambda x: x[0])
        )
        list_of_array_equal(
            sorted(mc.recurrent_classes, key=lambda x: x[0]),
            sorted(test_dict['rec_classes'], key=lambda x: x[0])
        )
        assert(mc.period == test_dict['period'])
        assert(mc.is_aperiodic == test_dict['is_aperiodic'])
        try:
            list_of_array_equal(
                sorted(mc.cyclic_classes, key=lambda x: x[0]),
                sorted(test_dict['cyclic_classes'], key=lambda x: x[0])
            )
        except NotImplementedError:
            assert(mc.is_irreducible is False)

        # Test of mc_compute_stationary
        computed = mc_compute_stationary(test_dict['P'])
        assert_allclose(computed, test_dict['stationary_dists'])


# Basic Class Structure with Setup #
####################################

class Test_markovchain_stationary_distributions_KMRMarkovMatrix2():
    """
    Test Suite for MarkovChain.stationary_distributions using KMR Markov
    Matrix [suitable for nose]
    """

    # Starting Values #

    N = 27
    epsilon = 1e-2
    p = 1/3
    TOL = 1e-2

    def setup_method(self):
        """ Setup a KMRMarkovMatrix and Compute Stationary Values """
        self.P = KMR_Markov_matrix_sequential(self.N, self.p, self.epsilon)
        self.mc = MarkovChain(self.P)
        self.stationary = self.mc.stationary_distributions
        stat_shape = self.stationary.shape

        if len(stat_shape) == 1:
            self.n_stat_dists = 1
        else:
            self.n_stat_dists = stat_shape[0]

    def test_markov_matrix(self):
        "Check that each row of matrix sums to 1"
        mc = self.mc
        assert_allclose(np.sum(mc.P, axis=1), np.ones(mc.n))

    def test_sum_one(self):
        "Check that each stationary distribution sums to 1"
        stationary_distributions = self.stationary
        assert_allclose(np.sum(stationary_distributions, axis=1),
                        np.ones(self.n_stat_dists))

    def test_nonnegative(self):
        "Check that the stationary distributions are non-negative"
        stationary_distributions = self.stationary
        assert(np.all(stationary_distributions > -1e-16))

    def test_left_eigen_vec(self):
        "Check that vP = v for all stationary distributions"
        mc = self.mc
        stationary_distributions = self.stationary

        if self.n_stat_dists == 1:
            assert_allclose(np.dot(stationary_distributions, mc.P),
                            stationary_distributions, atol=self.TOL)
        else:
            for i in range(self.n_stat_dists):
                curr_v = stationary_distributions[i, :]
                assert_allclose(np.dot(curr_v, mc.P), curr_v, atol=self.TOL)


def test_simulate_shape():
    P = [[0.4, 0.6], [0.2, 0.8]]
    mcs = [MarkovChain(P), MarkovChain(sparse.csr_matrix(P))]

    for mc in mcs:
        (ts_length, init, num_reps) = (10, None, None)
        assert_array_equal(mc.simulate(ts_length, init, num_reps).shape,
                           (ts_length,))

        (ts_length, init, num_reps) = (10, [0, 1], None)
        assert_array_equal(mc.simulate(ts_length, init, num_reps).shape,
                           (len(init), ts_length))

        (ts_length, init, num_reps) = (10, [0, 1], 3)
        assert_array_equal(mc.simulate(ts_length, init, num_reps).shape,
                           (len(init)*num_reps, ts_length))

        for (ts_length, init, num_reps) in [(10, None, 3), (10, None, 1)]:
            assert_array_equal(mc.simulate(ts_length, init, num_reps).shape,
                               (num_reps, ts_length))


def test_simulate_init_array_num_reps():
    P = [[0.4, 0.6], [0.2, 0.8]]
    mc = MarkovChain(P)

    ts_length = 10
    init = [0, 1]
    num_reps = 3

    X = mc.simulate(ts_length, init, num_reps)
    assert_array_equal(X[:, 0], init*num_reps)


def test_simulate_init_type():
    P = [[0.4, 0.6], [0.2, 0.8]]
    mc = MarkovChain(P)

    seed = 0
    ts_length = 3
    init = 0  # int
    X = mc.simulate(ts_length, init=init, random_state=seed)

    inits_np_int = [t(init) for t in [np.int32, np.int64]]
    for init in inits_np_int:
        X_np_int = mc.simulate(ts_length, init=init, random_state=seed)
        assert_array_equal(X_np_int, X)


def test_simulate_dense_vs_sparse():
    n = 5
    a = 1/3
    b = (1 - a)/2
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i-1) % n], P[i, i], P[i, (i+1) % n] = b, a, b
    mcs = [MarkovChain(P), MarkovChain(sparse.csr_matrix(P))]

    ts_length = 10
    inits = (None, 0, [0, 1])
    num_repss = (None, 2)

    random_state = 0
    for init, num_reps in itertools.product(inits, num_repss):
        assert_array_equal(*(mc.simulate(ts_length, init, num_reps,
                                         random_state=random_state)
                             for mc in mcs))


def test_simulate_ergodicity():
    P = [[0.4, 0.6], [0.2, 0.8]]
    stationary_dist = [0.25, 0.75]
    init = 0
    mc = MarkovChain(P)

    seed = 4433
    ts_length = 100
    num_reps = 300
    tol = 0.1

    x = mc.simulate(ts_length, init=init, num_reps=num_reps, random_state=seed)
    frequency_1 = x[:, -1].mean()
    assert_(np.abs(frequency_1 - stationary_dist[1]) < tol)


def test_simulate_for_matrices_with_C_F_orders():
    """
    Test MarkovChasin.simulate for matrices with C- and F-orders
    See the issue and fix on Numba:
    github.com/numba/numba/issues/1103
    github.com/numba/numba/issues/1104
    """
    P_C = np.array([[0.5, 0.5], [0, 1]], order='C')
    P_F = np.array([[0.5, 0.5], [0, 1]], order='F')
    init = 1
    ts_length = 10
    sample_path = np.ones(ts_length, dtype=int)

    computed_C_and_F = \
        MarkovChain(np.array([[1.]])).simulate(ts_length, init=0)
    assert_array_equal(computed_C_and_F, np.zeros(ts_length, dtype=int))

    computed_C = MarkovChain(P_C).simulate(ts_length, init)
    computed_F = MarkovChain(P_F).simulate(ts_length, init=init)
    assert_array_equal(computed_C, sample_path)
    assert_array_equal(computed_F, sample_path)


def test_simulate_issue591():
    """
    Test MarkovChasin.simulate for P with dtype=np.float32
    https://github.com/QuantEcon/QuantEcon.py/issues/591
    """
    num_states = 5
    transition_states = 4

    transition_seed = 2
    random_state = np.random.RandomState(transition_seed)
    transitions = random_state.uniform(0., 1., transition_states)
    transitions /= np.sum(transitions)
    P = np.zeros((num_states, num_states), dtype=np.float32)
    P[0, :transition_states] = transitions
    P[1:, 0] = 1.
    mc = MarkovChain(P=P)

    simulate_seed = 22220
    ts_length = 10000
    seq = mc.simulate(
        ts_length=ts_length, init=0, num_reps=1, random_state=simulate_seed
    )
    max_state_in_seq = np.max(seq)

    assert_array_less(max_state_in_seq, num_states)


def test_mc_sample_path():
    P = [[0.4, 0.6], [0.2, 0.8]]
    Ps = [P, sparse.csr_matrix(P)]

    init = 0
    sample_size = 10

    seed = 42

    for P in Ps:
        # init: integer
        expected = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        computed = mc_sample_path(
            P, init=init, sample_size=sample_size, random_state=seed
        )
        assert_array_equal(computed, expected)

        # init: distribution
        distribution = (0.5, 0.5)
        expected = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        computed = mc_sample_path(
            P, init=distribution, sample_size=sample_size, random_state=seed
        )
        assert_array_equal(computed, expected)


def test_mc_sample_path_lln():
    P = [[0.4, 0.6], [0.2, 0.8]]
    stationary_dist = [0.25, 0.75]
    init = 0

    seed = 4433
    sample_size = 10**4
    tol = 0.02

    frequency_1 = mc_sample_path(P, init=init, sample_size=sample_size,
                                 random_state=seed).mean()
    assert_(np.abs(frequency_1 - stationary_dist[1]) < tol)


class TestMCStateValues:
    def setup_method(self):
        state_values = [[0, 1], [2, 3], [4, 5]]  # Pass python list
        self.state_values = np.array(state_values)

        self.mc_reducible_dict = {
            'mc': MarkovChain([[1, 0, 0], [1, 0, 0], [0, 0, 1]],
                              state_values=state_values),
            'coms': [[0], [1], [2]],
            'recs': [[0], [2]],
            'period': 1
        }

        self.mc_periodic_dict = {
            'mc': MarkovChain([[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                              state_values=state_values),
            'coms': [[0, 1, 2]],
            'recs': [[0, 1, 2]],
            'cycs': [[0], [1], [2]],
            'period': 3
        }

    def test_com_rec_classes(self):
        for mc_dict in [self.mc_reducible_dict, self.mc_periodic_dict]:
            mc = mc_dict['mc']
            coms = mc_dict['coms']
            recs = mc_dict['recs']
            properties = ['communication_classes',
                          'recurrent_classes']
            suffix = '_indices'
            for prop0, classes_ind in zip(properties, [coms, recs]):
                for return_indices in [True, False]:
                    if return_indices:
                        classes = classes_ind
                        prop = prop0 + suffix
                        key = lambda x: x[0]
                    else:
                        classes = [self.state_values[i] for i in classes_ind]
                        prop = prop0
                        key = lambda x: x[0, 0]
                    list_of_array_equal(
                        sorted(getattr(mc, prop), key=key),
                        sorted(classes, key=key)
                    )

    def test_cyc_classes(self):
        mc = self.mc_periodic_dict['mc']
        cycs = self.mc_periodic_dict['cycs']
        properties = ['cyclic_classes']
        suffix = '_indices'
        for prop0, classes_ind in zip(properties, [cycs]):
            for return_indices in [True, False]:
                if return_indices:
                    classes = classes_ind
                    prop = prop0 + suffix
                    key = lambda x: x[0]
                else:
                    classes = [self.state_values[i] for i in classes_ind]
                    prop = prop0
                    key = lambda x: x[0, 0]
                list_of_array_equal(
                    sorted(getattr(mc, prop), key=key),
                    sorted(classes, key=key)
                )

    def test_period(self):
        for mc_dict in [self.mc_reducible_dict, self.mc_periodic_dict]:
            mc = mc_dict['mc']
            period = mc_dict['period']
            assert_equal(mc.period, period)

    def test_simulate(self):
        # Deterministic mc
        mc = self.mc_periodic_dict['mc']
        ts_length = 6

        methods = ['simulate_indices', 'simulate']

        init_idx = 0
        inits = [init_idx, self.state_values[init_idx]]
        path = np.arange(init_idx, init_idx+ts_length)%mc.n
        paths = [path, self.state_values[path]]
        for method, init, X_expected in zip(methods, inits, paths):
            X = getattr(mc, method)(ts_length, init)
            assert_array_equal(X, X_expected)

        init_idx = [1, 2]
        inits = [init_idx, self.state_values[init_idx]]
        path = np.array(
            [np.arange(i, i+ts_length)%mc.n for i in init_idx]
        )
        paths = [path, self.state_values[path]]
        for method, init, X_expected in zip(methods, inits, paths):
            X = getattr(mc, method)(ts_length, init)
            assert_array_equal(X, X_expected)

        inits = [None, None]
        seed = 1234  # init will be 2
        init_idx = 2
        path = np.arange(init_idx, init_idx+ts_length)%mc.n
        paths = [path, self.state_values[path]]
        for method, init, X_expected in zip(methods, inits, paths):
            X = getattr(mc, method)(ts_length, init, random_state=seed)
            assert_array_equal(X, X_expected)


def test_mc_stationary_distributions_state_values():
    P = [[0.4, 0.6, 0], [0.2, 0.8, 0], [0, 0, 1]]
    state_values = ['a', 'b', 'c']
    mc = MarkovChain(P, state_values=state_values)
    stationary_dists_expected = [[0.25, 0.75, 0], [0, 0, 1]]
    stationary_dists = mc.stationary_distributions
    assert_allclose(stationary_dists, stationary_dists_expected)


def test_get_index():
    P = [[0.4, 0.6], [0.2, 0.8]]
    mc = MarkovChain(P)

    assert_(mc.get_index(0) == 0)
    assert_(mc.get_index(1) == 1)
    assert_raises(ValueError, mc.get_index, 2)
    assert_array_equal(mc.get_index([1, 0]), [1, 0])
    assert_raises(ValueError, mc.get_index, [[1]])

    mc.state_values = [1, 2]
    assert_(mc.get_index(1) == 0)
    assert_(mc.get_index(2) == 1)
    assert_raises(ValueError, mc.get_index, 0)
    assert_array_equal(mc.get_index([2, 1]), [1, 0])
    assert_raises(ValueError, mc.get_index, [[1]])

    mc.state_values = [[1, 2], [3, 4]]
    assert_(mc.get_index([1, 2]) == 0)
    assert_raises(ValueError, mc.get_index, 1)
    assert_array_equal(mc.get_index([[3, 4], [1, 2]]), [1, 0])


def test_raises_value_error_non_2dim():
    """Test with non 2dim input"""
    assert_raises(ValueError, MarkovChain, np.array([0.4, 0.6]))


def test_raises_value_error_non_sym():
    """Test with non symmetric input"""
    P = np.array([[0.4, 0.6]])
    assert_raises(ValueError, MarkovChain, P)
    assert_raises(ValueError, MarkovChain, sparse.csr_matrix(P))


def test_raises_value_error_non_nonnegative():
    """Test with non nonnegative input"""
    P = np.array([[0.4, 0.6], [-0.2, 1.2]])
    assert_raises(ValueError, MarkovChain, P)
    assert_raises(ValueError, MarkovChain, sparse.csr_matrix(P))


def test_raises_value_error_non_sum_one():
    """Test with input such that some of the rows does not sum to one"""
    P = np.array([[0.4, 0.6], [0.2, 0.9]])
    assert_raises(ValueError, MarkovChain, P)
    assert_raises(ValueError, MarkovChain, sparse.csr_matrix(P))


def test_raises_value_error_simulate_init_out_of_range():
    P = [[0.4, 0.6], [0.2, 0.8]]
    mc = MarkovChain(P)

    n = mc.n
    ts_length = 3
    assert_raises(ValueError, mc.simulate, ts_length, init=n)
    assert_raises(ValueError, mc.simulate, ts_length, init=-(n+1))
    assert_raises(ValueError, mc.simulate, ts_length, init=[0, n])
    assert_raises(ValueError, mc.simulate, ts_length, init=[0, -(n+1)])


def test_raises_non_homogeneous_state_values():
    P = [[0.4, 0.6], [0.2, 0.8]]
    state_values = [(0, 1), 2]
    assert_raises(ValueError, MarkovChain, P, state_values=state_values)
