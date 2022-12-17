"""
Tests for gridtools.py file

"""
import numpy as np
import time
import pytest
from numpy.testing import (
    assert_array_equal, assert_equal, assert_, assert_raises
)

from quantecon._gridtools import (
    cartesian, mlinspace, _repeat_1d, simplex_grid, simplex_index,
    num_compositions, num_compositions_jit, cartesian_nearest_index
)


def test_cartesian_C_order():
    x = np.linspace(0, 9, 10)

    prod = cartesian([x, x, x])

    correct = True
    for i in range(999):
        n = prod[i, 0]*100+prod[i, 1]*10+prod[i, 2]
        correct *= (i == n)

    assert_(correct)


def test_cartesian_C_order_int_float():
    x_int = np.arange(10)
    x_float = np.linspace(0, 9, 10)
    prod_int = cartesian([x_int]*3)
    prod_float = cartesian([x_float]*3)
    assert_(prod_int.dtype == x_int.dtype)
    assert_(prod_float.dtype == x_float.dtype)
    assert_(abs(prod_int-prod_float).max() == 0)


def test_cartesian_C_order_int_float_mixed():
    x_int = [0, 1]
    x_float = [2.3, 4.5]
    prod_expected = np.array(
        [[0., 2.3],
         [0., 4.5],
         [1., 2.3],
         [1., 4.5]]
    )
    prod_computed = cartesian([x_int, x_float])
    assert_array_equal(prod_computed, prod_expected)


def test_cartesian_F_order():
    x = np.linspace(0, 9, 10)

    prod = cartesian([x, x, x], order='F')

    correct = True
    for i in range(999):
        n = prod[i, 2]*100+prod[i, 1]*10+prod[i, 0]
        correct *= (i == n)

    assert_(correct)


@pytest.mark.slow
def test_performance_C():
    N_x = 1000
    N_y = 7777
    x = np.linspace(1, N_x, N_x)
    y = np.linspace(1, N_y, N_y)

    cartesian([x[:10], y[:10]]) # warmup

    t1 = time.time()
    for i in range(100):
        prod = cartesian([x, y])
    t2 = time.time()
    # print(prod.shape)

    # compute the same produce using numpy:

    t3 = time.time()
    for i in range(100):
        prod_numpy = np.column_stack([
            np.repeat(x, N_y),
            np.tile(y, N_x)
        ])
    t4 = time.time()

    print("Timings for 'cartesian' (C order)")
    print("Cartesian: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))
    assert_(abs(prod-prod_numpy).max() == 0)


@pytest.mark.slow
def test_performance_F():
    N_x = 1000
    N_y = 7777
    x = np.linspace(1, N_x, N_x)
    y = np.linspace(1, N_y, N_y)

    cartesian([x[:10], y[:10]]) # warmup

    t1 = time.time()
    for i in range(100):
        prod = cartesian([x, y], order='F')
    t2 = time.time()
    # print(prod.shape)

    # compute the same produce using numpy:

    t3 = time.time()
    for i in range(100):
        prod_numpy = np.column_stack([
            np.tile(x, N_y),
            np.repeat(y, N_x)
        ])
    t4 = time.time()

    print("Timings for 'cartesian'(Fortran order)")
    print("Cartesian: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))
    assert_(abs(prod-prod_numpy).max() == 0)


def test_mlinsplace():
    mlinspace([-1, -1], [2, 3], [30, 50])
    cartesian([np.linspace(-1, 2, 30), np.linspace(-1, 3, 50)])


def test_tile():
    x = np.linspace(1, 100, 100)
    t1 = time.time()
    t_repeat = np.zeros(100*1000)
    _repeat_1d(x, 1, t_repeat)
    t2 = time.time()

    t3 = time.time()
    t_numpy = np.tile(x, 1000)
    t4 = time.time()

    print("Timings for 'tile' operation")
    print("Repeat_1d: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))

    assert_(abs(t_numpy-t_repeat).max())


def test_repeat():
    x = np.linspace(1, 100, 100)
    t1 = time.time()
    t_repeat = np.zeros(100*1000)
    _repeat_1d(x, 1000, t_repeat)
    t2 = time.time()

    t3 = time.time()
    t_numpy = np.repeat(x, 1000)
    t4 = time.time()

    print("Timings for 'repeat' operation")
    print("Repeat_1d: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))

    assert_(abs(t_numpy-t_repeat).max())


class TestCartesianNearestIndex:
    def setup_method(self):
        nums = (5, 6)
        self.nodes = [list(range(nums[0])), np.linspace(0, 1, nums[1])]
        self.orders = ['C', 'F']
        self.prod_dict = \
            {order:cartesian(self.nodes, order=order) for order in self.orders}

    def linear_search(self, x, order='C'):
        x = np.asarray(x)
        return ((self.prod_dict[order] - x)**2).sum(1).argmin()

    def test_1d(self):
        x = (1.2, 0.3)
        for order in self.orders:
            ind_expected = self.linear_search(x, order)
            ind_computed = cartesian_nearest_index(x, self.nodes, order)
            assert_equal(ind_computed, ind_expected)

        assert_raises(
            ValueError, cartesian_nearest_index, x, self.prod_dict['C']
        )

    def test_2d(self):
        T = 10
        rng = np.random.default_rng(1234)
        X = np.column_stack((
            rng.uniform(self.nodes[0][0]-1, self.nodes[0][-1]+1, size=T),
            rng.standard_normal(T) + 0.5
        ))
        ind_expected = np.empty(T, dtype=np.intp)

        for order in self.orders:
            for t in range(T):
                ind_expected[t] = self.linear_search(X[t], order)
            ind_computed = cartesian_nearest_index(X, self.nodes, order)
            assert_array_equal(ind_computed, ind_expected)

        assert_raises(
            ValueError, cartesian_nearest_index, X, self.prod_dict['C']
        )


class TestSimplexGrid:
    def setup_method(self):
        self.simplex_grid_3_4 = np.array([[0, 0, 4],
                                          [0, 1, 3],
                                          [0, 2, 2],
                                          [0, 3, 1],
                                          [0, 4, 0],
                                          [1, 0, 3],
                                          [1, 1, 2],
                                          [1, 2, 1],
                                          [1, 3, 0],
                                          [2, 0, 2],
                                          [2, 1, 1],
                                          [2, 2, 0],
                                          [3, 0, 1],
                                          [3, 1, 0],
                                          [4, 0, 0]])

    def test_simplex_grid(self):
        out = simplex_grid(3, 4)
        assert_array_equal(out, self.simplex_grid_3_4)

        assert_array_equal(simplex_grid(1, 1), [[1]])

    def test_simplex_index(self):
        points = [[0, 0, 4], [1, 1, 2], [4, 0, 0]]
        for point in points:
            idx = simplex_index(point, 3, 4)
            assert_array_equal(self.simplex_grid_3_4[idx], point)

        assert_(simplex_index([1], 1, 1) == 0)

    def test_num_compositions(self):
        num = num_compositions(3, 4)
        assert_(num == len(self.simplex_grid_3_4))

    def test_num_compositions_jit(self):
        num = num_compositions_jit(3, 4)
        assert_(num == len(self.simplex_grid_3_4))

        # Exceed max value of np.intp
        assert_(num_compositions_jit(100, 50) == 0)


def test_simplex_grid_raises_value_error_overflow():
    # Exceed max value of np.intp
    assert_raises(ValueError, simplex_grid, 100, 50)
