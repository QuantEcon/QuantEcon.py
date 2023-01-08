"""
Tests for lcp_lemke

"""
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal

from quantecon.optimize import lcp_lemke


def _assert_ray_termination(res):
    # res: lcp result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report ray termination status")


def _assert_success(res, M, q, desired_z=None, rtol=1e-15, atol=1e-15):
    if not res.success:
        msg = "lcp_lemke status {0}".format(res.status)
        raise AssertionError(msg)

    assert_equal(res.status, 0)
    if desired_z is not None:
        assert_allclose(res.z, desired_z,
                        err_msg="converged to an unexpected solution",
                        rtol=rtol, atol=atol)

    assert_((res.z >= -atol).all())

    w = M @ res.z + q
    assert_((w >= -atol).all())

    assert_allclose(w * res.z, np.zeros_like(res.z), rtol=rtol, atol=atol)


class TestLCPLemke:
    def test_Murty_Ex_2_8(self):
        M = [[1, -1, -1, -1],
             [-1, 1, -1, -1],
             [1, 1, 2, 0],
             [1, 1, 0, 2]]
        q = [3, 5, -9, -5]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)

    def test_Murty_Ex_2_9(self):
        M = [[-1, 0, -3],
             [1, -2, -5],
             [-2, -1, -2]]
        q = [-3, -2, -1]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_ray_termination(res)

    def test_Kostreva_Ex_1(self):
        # Cycling without careful tie breaking
        M = [[1, 2, 0],
             [0, 1, 2],
             [2, 0, 1]]
        q = [-1, -1, -1]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)

    def test_Kostreva_Ex_2(self):
        # Cycling without careful tie breaking
        M = [[1, -1, 3],
             [2, -1, 3],
             [-1, -2, 0]]
        q = [-1, -1, -1]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_ray_termination(res)

    def test_Murty_Ex_2_11(self):
        M = [[-1.5, 2],
             [-4, 4]]
        q = [-5, 17]
        d = [5., 16.]
        M, q, d = map(np.asarray, [M, q, d])
        res = lcp_lemke(M, q, d=d)
        _assert_ray_termination(res)

        res = lcp_lemke(M, q, d=np.ones_like(d))
        _assert_success(res, M, q, atol=1e-13)

    def test_bimatrix_game(self):
        A = [[3, 3],
             [2, 5],
             [0, 6]]
        B = [[3, 2, 3],
             [2, 6, 1]]
        A, B = map(np.asarray, [A, B])
        m, n = A.shape
        I = np.cumsum([0, m, n, m, m, n, n])
        M = np.zeros((3*m+3*n, 3*m+3*n))
        M[I[0]:I[1], I[1]:I[2]] = -A + A.max()
        M[I[0]:I[1], I[2]:I[3]], M[I[0]:I[1], I[3]:I[4]] = 1, -1
        M[I[1]:I[2], I[0]:I[1]] = -B + B.max()
        M[I[1]:I[2], I[4]:I[5]], M[I[1]:I[2], I[5]:I[6]] = 1, -1
        M[I[2]:I[3], I[0]:I[1]], M[I[3]:I[4], I[0]:I[1]] = -1, 1
        M[I[4]:I[5], I[1]:I[2]], M[I[5]:I[6], I[1]:I[2]] = -1, 1
        q = np.zeros(3*m+3*n)
        q[I[2]:I[3]], q[I[3]:I[4]] = 1, -1
        q[I[4]:I[5]], q[I[5]:I[6]] = 1, -1

        res = lcp_lemke(M, q)
        _assert_success(res, M, q)
