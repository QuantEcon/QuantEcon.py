"""
Browser smoke suite for QuantEcon.py on the JupyterLite xeus-python kernel.

One representative function per Numba feature class used in the library.
Run natively with pytest to validate the suite itself; the same file is
consumed by the WASM CI job (issue #933).

Findings from a WASM run should be recorded in issue #928 as a results
table: function name -> works / fails / notes.
"""
import sys
import time
import warnings

import numpy as np
import pytest
from numba import njit

IS_EMSCRIPTEN = sys.platform == "emscripten"

# ---------------------------------------------------------------------------
# Jitted helpers required by optimize tests (must be at module scope)
# ---------------------------------------------------------------------------

@njit
def _rosenbrock(x):
    return -(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


@njit
def _parabola(x):
    return -(x + 2.0) ** 2 + 1.0


@njit
def _cubic(x):
    return x ** 3 - 1.0


@njit
def _cubic_prime(x):
    return 3.0 * x ** 2


@njit
def _linalg_solve(A, b):
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# 1. Import timing — cold vs warm cache (feeds issue #930)
# ---------------------------------------------------------------------------

def test_import_time():
    t0 = time.perf_counter()
    import quantecon  # noqa: F401
    elapsed = time.perf_counter() - t0
    # 30 s is generous for a cold WASM JIT cache; native should be <1 s.
    assert elapsed < 30, f"import took {elapsed:.1f} s"


# ---------------------------------------------------------------------------
# 2. Plain lazy @njit — tauchen and rouwenhorst
# ---------------------------------------------------------------------------

def test_tauchen():
    import quantecon as qe
    mc = qe.tauchen(5, 0.9, 0.1)
    assert mc.P.shape == (5, 5)
    assert np.allclose(mc.P.sum(axis=1), 1.0)


def test_rouwenhorst():
    import quantecon as qe
    mc = qe.rouwenhorst(5, 0.9, 0.1)
    assert mc.P.shape == (5, 5)
    assert np.allclose(mc.P.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 3. MarkovChain.simulate — jitted simulation with NRT-allocated arrays
# ---------------------------------------------------------------------------

def test_markov_simulate():
    import quantecon as qe
    mc = qe.tauchen(5, 0.9, 0.1)
    sim = mc.simulate_indices(ts_length=200, init=0, random_state=42)
    assert len(sim) == 200
    assert np.all((sim >= 0) & (sim < 5))


# ---------------------------------------------------------------------------
# 4. probvec — parallel guvectorize; on Emscripten patch 0007 falls back
#    to 'cpu' target silently, so the result must still be correct
# ---------------------------------------------------------------------------

def test_probvec():
    import quantecon as qe
    result = qe.random.probvec(4, 3, random_state=42)
    assert result.shape == (4, 3)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# 5. sample_without_replacement — eager guvectorize with explicit i8 sig
# ---------------------------------------------------------------------------

def test_sample_without_replacement():
    import quantecon as qe
    result = qe.random.sample_without_replacement(10, 4, random_state=42)
    assert len(result) == 4
    assert len(set(result.tolist())) == 4
    assert np.all((result >= 0) & (result < 10))


# ---------------------------------------------------------------------------
# 6. Optimize: nelder_mead, brent_max, newton
# ---------------------------------------------------------------------------

def test_nelder_mead():
    from quantecon.optimize import nelder_mead
    result = nelder_mead(_rosenbrock, np.array([-1.0, 1.0]))
    assert result.success
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-4)


def test_brent_max():
    from quantecon.optimize import brent_max
    xf, fval, info = brent_max(_parabola, -4.0, 0.0)
    assert abs(xf - (-2.0)) < 1e-4
    assert abs(fval - 1.0) < 1e-4


def test_newton():
    from quantecon.optimize import newton
    result = newton(_cubic, 2.0, _cubic_prime)
    assert abs(result.root - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. game_theory.lemke_howson
# ---------------------------------------------------------------------------

def test_lemke_howson():
    import quantecon as qe
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = qe.game_theory.NormalFormGame(bimatrix)
    NE = qe.game_theory.lemke_howson(g, init_pivot=0)
    assert len(NE) == 2
    assert np.allclose(NE[0].sum(), 1.0, atol=1e-6)
    assert np.allclose(NE[1].sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. game_theory.vertex_enumeration — exercises numba.typed.Dict
# ---------------------------------------------------------------------------

def test_vertex_enumeration():
    import quantecon as qe
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = qe.game_theory.NormalFormGame(bimatrix)
    NEs = qe.game_theory.vertex_enumeration(g)
    assert len(NEs) == 3


# ---------------------------------------------------------------------------
# 9. np.linalg.solve inside @njit — isolates the _LAPACK mechanism (#927)
#    independently of QuantEcon's own overload.
# ---------------------------------------------------------------------------

def test_np_linalg_solve_jit():
    A = np.array([[3.0, 2.0], [1.0, -1.0]])
    b = np.array([8.0, 1.0])
    x = _linalg_solve(A, b)
    assert np.allclose(x, np.linalg.solve(A, b))


# ---------------------------------------------------------------------------
# 10. game_theory.support_enumeration — end-to-end _LAPACK test (#927)
# ---------------------------------------------------------------------------

def test_support_enumeration():
    import quantecon as qe
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = qe.game_theory.NormalFormGame(bimatrix)
    NEs = qe.game_theory.support_enumeration(g)
    assert len(NEs) == 3
    assert np.allclose(NEs[0][0], [1.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# 11. gini_coefficient — @njit(parallel=True) + prange; expected to fail
#     at first call on Emscripten because the ParallelAccelerator pass is
#     not supported (issue #926).
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    IS_EMSCRIPTEN,
    reason="@njit(parallel=True) not supported on Emscripten (#926)",
    strict=True,
)
def test_gini_coefficient():
    import quantecon as qe
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = qe.gini_coefficient(y)
    assert 0.0 < g < 1.0


# ---------------------------------------------------------------------------
# 12. simplex_grid — 32-bit intp boundary behaviour on wasm32 (#929)
# ---------------------------------------------------------------------------

def test_simplex_grid():
    import quantecon as qe
    grid = qe.simplex_grid(3, 4)
    # shape: (L, m) where L = C(4+3-1, 3-1) = 15
    assert grid.shape == (15, 3)
    assert np.all(grid.sum(axis=1) == 4)
    assert np.all(grid >= 0)


# ---------------------------------------------------------------------------
# 13. searchsorted — objmode() shim (deprecated helper)
# ---------------------------------------------------------------------------

def test_searchsorted():
    from quantecon.util.array import searchsorted
    a = np.array([0.2, 0.4, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert searchsorted(a, 0.1) == 0
        assert searchsorted(a, 0.4) == 2
        assert searchsorted(a, 2.0) == 3
