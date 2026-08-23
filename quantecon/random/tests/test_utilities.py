"""
Tests for random/utilities.py

Functions
---------
probvec
sample_without_replacement
draw

"""
import numbers
import numpy as np
import pytest
from numpy.testing import (assert_array_equal, assert_allclose, assert_raises,
                           assert_)
from numba import config, njit, TypingError
from quantecon.random import probvec, sample_without_replacement, draw


# probvec #

class TestProbvec:
    def setup_method(self):
        self.m, self.k = 2, 3  # m vectors of dimension k
        seed = 1234

        self.out_parallel = probvec(self.m, self.k, random_state=seed)
        self.out_cpu = \
            probvec(self.m, self.k, random_state=seed, parallel=False)

    def test_shape(self):
        for out in [self.out_parallel, self.out_cpu]:
            assert_(out.shape == (self.m, self.k))

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
    assert_(len(b) == n)


def test_sample_without_replacement_value_error():
    # n <= 0
    assert_raises(ValueError, sample_without_replacement, 0, 2)
    assert_raises(ValueError, sample_without_replacement, -1, -1)

    # k > n
    assert_raises(ValueError, sample_without_replacement, 2, 3)


# draw #

@njit
def draw_jitted(cdf, size=None):
    return draw(cdf, size)


@njit
def draw_jitted_rng(cdf, size, rng):
    return draw(cdf, size, rng)


@njit
def draw_jitted_rng_kw(cdf, rng):
    return draw(cdf, rng=rng)


@njit
def draw_jitted_explicit_none(cdf, size):
    return draw(cdf, size, None)


@njit
def draw_jitted_fwd(cdf, size=None, rng=None):
    # Forwards its own defaults, which reach the overload as types.none
    return draw(cdf, size, rng)


@njit
def draw_jitted_seeded(cdf, size, seed):
    np.random.seed(seed)
    return draw(cdf, size)


@njit
def draw_jitted_optional_rng(cdf, size, rng, flag):
    rng_or_none = rng if flag else None
    return draw(cdf, size, rng_or_none)


class TestDraw:
    def setup_method(self):
        self.pmf = np.array([0.4, 0.1, 0.5])
        self.cdf = np.cumsum(self.pmf)
        self.n = len(self.pmf)
        self.draw_funcs = [draw, draw_jitted]

    def test_return_types(self):
        for func in self.draw_funcs:
            out = func(self.cdf)
            assert_(isinstance(out, numbers.Integral))

        size = 10
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            assert_(out.shape == (size,))

    def test_numpy_integer_size_returns_array(self):
        size = np.int64(10)
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            assert_(out.shape == (size,))

    def test_bool_size_is_treated_as_scalar(self):
        for func in self.draw_funcs:
            out = func(self.cdf, True)
            assert_(isinstance(out, numbers.Integral))

    def test_return_values(self):
        for func in self.draw_funcs:
            out = func(self.cdf)
            assert_(out in range(self.n))

        size = 10
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            assert_(np.isin(out, range(self.n)).all())

    def test_lln(self):
        size = 1000000
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            hist, bin_edges = np.histogram(out, bins=self.n, density=True)
            pmf_computed = hist * np.diff(bin_edges)
            atol = 1e-2
            assert_allclose(pmf_computed, self.pmf, atol=atol)

    # rng: a Generator behaves the same on both paths #

    def test_python_jitted_agree(self):
        # The contract that keeps the two implementations in step: for
        # the same Generator seed they must return the same values.
        for seed in [0, 1234, 20260801]:
            for size in [1, 10, 1000]:
                out_py = draw(self.cdf, size, np.random.default_rng(seed))
                out_jit = draw_jitted_rng(self.cdf, size,
                                          np.random.default_rng(seed))
                assert_array_equal(out_py, out_jit)

            # Scalar draws are compared as arrays: the Python path
            # returns np.int64 and the jitted path a Python int.
            out_py = draw(self.cdf, None, np.random.default_rng(seed))
            out_jit = draw_jitted_rng(self.cdf, None,
                                      np.random.default_rng(seed))
            assert_array_equal(np.asarray(out_py), np.asarray(out_jit))

    def test_generator_reproducible(self):
        for size in [None, 10]:
            for func in [draw, draw_jitted_rng]:
                out0 = func(self.cdf, size, np.random.default_rng(1234))
                out1 = func(self.cdf, size, np.random.default_rng(1234))
                assert_array_equal(np.asarray(out0), np.asarray(out1))

    def test_generator_state_advances(self):
        # The Generator must be mutated in place by the jitted call, not
        # copied at the boundary, so that successive calls continue the
        # stream rather than restarting it.
        rng = np.random.default_rng(1234)
        parts = [draw_jitted_rng(self.cdf, 5, rng),
                 draw_jitted_rng(self.cdf, 5, rng),
                 draw(self.cdf, 5, rng)]
        expected = draw(self.cdf, 15, np.random.default_rng(1234))
        assert_array_equal(np.concatenate(parts), expected)

    def test_consumes_exactly_one_draw_per_variate(self):
        # An extra variate taken at the *end* of a call is invisible to
        # the value comparisons above, but it desynchronises a shared
        # Generator for every later consumer. Complements
        # test_generator_state_advances, which covers the in-place
        # mutation half of the same contract.
        for func in [draw, draw_jitted_rng]:
            for size in [None, 1, 10]:
                n = 1 if size is None else size
                rng = np.random.default_rng(1234)
                func(self.cdf, size, rng)
                ref = np.random.default_rng(1234)
                ref.random(n)
                assert_array_equal(rng.random(4), ref.random(4))

    def test_generator_keyword_form_in_jit(self):
        out_py = draw(self.cdf, rng=np.random.default_rng(1234))
        out_jit = draw_jitted_rng_kw(self.cdf, np.random.default_rng(1234))
        assert_array_equal(np.asarray(out_py), np.asarray(out_jit))

    # rng=None: unchanged behaviour on both paths #

    def test_none_matches_legacy_global(self):
        for size in [None, 10]:
            np.random.seed(99)
            out = draw(self.cdf, size)
            np.random.seed(99)
            r = np.random.random(size) if size is not None \
                else np.random.random()
            expected = np.searchsorted(self.cdf, r, side='right')
            assert_array_equal(np.asarray(out), np.asarray(expected))

    def test_none_spellings_all_compile(self):
        # Omitted, explicitly None, and forwarded from a jitted caller's
        # own default are distinct numba types, and each must reach the
        # np.random branch of the overload.
        size = 10
        for out in [draw_jitted(self.cdf, size),
                    draw_jitted_explicit_none(self.cdf, size),
                    draw_jitted_fwd(self.cdf, size),
                    draw_jitted_fwd(self.cdf, size, None)]:
            assert_(out.shape == (size,))
            assert_(np.isin(out, range(self.n)).all())

        for out in [draw_jitted(self.cdf), draw_jitted_fwd(self.cdf)]:
            assert_(out in range(self.n))

    def test_jitted_np_random_seed_still_reproducible(self):
        out0 = draw_jitted_seeded(self.cdf, 10, 1234)
        out1 = draw_jitted_seeded(self.cdf, 10, 1234)
        assert_array_equal(out0, out1)

    # rng: compile-time rejections, jitted path only #
    #
    # The None | Generator contract is enforced only by the overload:
    # the pure-Python body does no input checking and its behaviour for
    # other inputs is unspecified, so nothing here calls `draw`
    # directly, and all three tests are skipped when NUMBA_DISABLE_JIT
    # routes the jitted wrappers to the Python body.

    @pytest.mark.skipif(config.DISABLE_JIT,
                        reason='requires nopython compilation')
    def test_int_seed_raises_in_jit(self):
        # Stricter than SPEC 7 by design: nopython mode cannot construct
        # a generator from a seed.
        with pytest.raises(TypingError) as excinfo:
            draw_jitted_rng(self.cdf, 10, 1234)
        # Numba nests the message in its report of the candidate
        # implementations it rejected. Assert on tokens that can
        # only come from draw's own message, not from the caller's
        # source line that numba echoes alongside it.
        msg = str(excinfo.value)
        assert_('quantecon.random.draw' in msg)
        assert_('np.random.default_rng' in msg)

    @pytest.mark.skipif(config.DISABLE_JIT,
                        reason='requires nopython compilation')
    def test_randomstate_raises_in_jit(self):
        # In nopython mode a RandomState cannot be typed at all, so it
        # is rejected during argument typing and the overload never
        # runs. Assert only the exception type, never the message.
        assert_raises(TypingError, draw_jitted_rng, self.cdf, 10,
                      np.random.RandomState(1234))

    @pytest.mark.skipif(config.DISABLE_JIT,
                        reason='requires nopython compilation')
    def test_optional_generator_raises_in_jit(self):
        assert_raises(TypingError, draw_jitted_optional_rng, self.cdf, 10,
                      np.random.default_rng(1234), True)
        try:
            draw_jitted_optional_rng(self.cdf, 10,
                                     np.random.default_rng(1234), True)
        except TypingError as e:
            # A token unique to the Optional hint, so that this branch
            # cannot be satisfied by the int-seed message.
            assert_('could not prove' in str(e))


@njit
def draw_jitted_w_o_size(n):
    cdf = np.linspace(1/n, 1, n)
    return draw(cdf)


def test_draw_jitted_w_o_size():
    n = 3
    assert_(draw_jitted_w_o_size(n) in range(n))
