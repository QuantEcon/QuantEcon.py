"""
Tests for game_theory/game_converters.py

"""
import io
import os
from tempfile import NamedTemporaryFile
from unittest.mock import patch
import numpy as np
from numpy.testing import (
    assert_, assert_array_equal, assert_string_equal, assert_raises
)
from quantecon.game_theory import (
    Player, NormalFormGame, random_game,
    GAMWriter, to_gam, from_gam_string, from_gam_url
)
from quantecon.game_theory.game_converters import GAMPayoffVector


# GAMPayoffVector #

class TestGAMPayoffVector:
    """Golden test for GAMPayoffVector"""

    def setup_method(self):
        nums_actions = (2, 3, 4)
        N = len(nums_actions)
        na = np.prod(nums_actions)

        A0 = np.arange(na).reshape(nums_actions, order='F')
        A1 = np.arange(100, 100+na).reshape(nums_actions, order='F')
        A2 = np.arange(200, 200+na).reshape(nums_actions, order='F')

        self.payoffs1d = np.hstack([A.ravel(order='F') for A in [A0, A1, A2]])
        self.payoffs4d = np.stack([A0, A1, A2], axis=N)

        self.N = N
        self.nums_actions = nums_actions

    def test_init(self):
        p = GAMPayoffVector(self.nums_actions, self.payoffs1d)

        assert_(p.N == self.N)
        assert_(p.nums_actions == self.nums_actions)
        assert_array_equal(p.payoffs, self.payoffs1d)

    def test_from_nfg(self):
        g = NormalFormGame(self.payoffs4d)
        p = GAMPayoffVector.from_nfg(g)

        assert_(p.N == self.N)
        assert_(p.nums_actions == self.nums_actions)
        assert_array_equal(p.payoffs, self.payoffs1d)

    def test_to_nfg(self):
        p = GAMPayoffVector(self.nums_actions, self.payoffs1d)
        g = p.to_nfg()
        assert_array_equal(g.payoff_profile_array, self.payoffs4d)


def test_gampayoffvector_roundtrip():
    for ns in [(4, 3), (2, 2, 3, 2)]:
        N = len(ns)
        seed = 12345
        rng = np.random.default_rng(seed)
        payoffs = rng.integers(low=0, high=100, size=(*ns, N), dtype=np.int64)
        g = NormalFormGame(payoffs)
        p = GAMPayoffVector.from_nfg(g)
        g1 = p.to_nfg()

        p_32 = GAMPayoffVector.from_nfg(g, dtype=np.int32)
        g2 = p_32.to_nfg()
        g3 = p_32.to_nfg(dtype=np.int64)

        assert_(p_32.payoffs.dtype == np.int32)
        assert_(g2.dtype == np.int32)
        assert_(g3.dtype == np.int64)

        for g_new in [g1, g2, g3]:
            assert_(g_new.N == g.N)
            assert_(g_new.nums_actions == g.nums_actions)
            for i in range(N):
                assert_array_equal(g_new.players[i].payoff_array,
                                   g.players[i].payoff_array)


def test_gampayoffvector_1p():
    payoffs = [1., 2., 3.]
    nums_actions = (3,)

    p0 = GAMPayoffVector(nums_actions, payoffs)

    g = NormalFormGame((Player(payoffs),))
    p1 = GAMPayoffVector.from_nfg(g)

    for p in [p0, p1]:
        assert_(p.N == 1)
        assert_(p.nums_actions == nums_actions)
        assert_array_equal(p.payoffs, payoffs)


def test_invalid_inputs():
    assert_raises(ValueError, GAMPayoffVector, (), np.array([], dtype=float))
    assert_raises(TypeError, GAMPayoffVector, (2, 2.0), np.zeros(8))
    assert_raises(ValueError, GAMPayoffVector, (2, 0), np.zeros(0))
    assert_raises(ValueError, GAMPayoffVector, (2, 2), np.zeros((2, 4)))
    assert_raises(ValueError, GAMPayoffVector, (2, 2), np.zeros(7))


# GAMWriter/to_gam #

class TestGAMWriter:
    def setup_method(self):
        nums_actions = (2, 2, 2)
        g = NormalFormGame(nums_actions)
        g[0, 0, 0] = (0, 8, 16)
        g[1, 0, 0] = (1, 9, 17)
        g[0, 1, 0] = (2, 10, 18)
        g[1, 1, 0] = (3, 11, 19)
        g[0, 0, 1] = (4, 12, 20)
        g[1, 0, 1] = (5, 13, 21)
        g[0, 1, 1] = (6, 14, 22)
        g[1, 1, 1] = (7, 15, 23)
        self.g = g

        self.s_desired = """\
3
2 2 2

0. 1. 2. 3. 4. 5. 6. 7. \
8. 9. 10. 11. 12. 13. 14. 15. \
16. 17. 18. 19. 20. 21. 22. 23."""

    def test_to_file(self):
        with NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            GAMWriter.to_file(self.g, temp_path)

        with open(temp_path, 'r') as f:
            s_actual = f.read()
        assert_string_equal(s_actual, self.s_desired + '\n')

        os.remove(temp_path)

    def test_to_string(self):
        s_actual = GAMWriter.to_string(self.g)

        assert_string_equal(s_actual, self.s_desired)

    def test_to_gam(self):
        s_actual = to_gam(self.g)
        assert_string_equal(s_actual, self.s_desired)

        with NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            to_gam(self.g, temp_path)

        with open(temp_path, 'r') as f:
            s_actual = f.read()
        assert_string_equal(s_actual, self.s_desired + '\n')

        os.remove(temp_path)

    def test_from_gam_string(self):
        g2 = from_gam_string(self.s_desired)
        assert_array_equal(g2.payoff_profile_array,
                           self.g.payoff_profile_array)


def test_gam_writer_many_actions():
    n0, n1 = 40, 60
    p0 = Player(np.arange(n0 * n1).reshape(n0, n1))
    p1 = Player((np.arange(n1 * n0) + 10_000).reshape(n1, n0))
    g = NormalFormGame((p0, p1))

    s = to_gam(g)

    # NumPy summary marker should never appear
    assert_('...' not in s)

    # Token count matches N * prod(nums_actions)
    tokens = s.split()
    N = int(tokens[0])
    nums_actions = tuple(int(x) for x in tokens[1:1+N])
    payoff_tokens = tokens[1+N:]

    assert_(N == 2)
    assert_(nums_actions == (n0, n1))

    expected = N * np.prod(nums_actions)
    assert_(len(payoff_tokens) == expected)


# GAMReader/from_gam #

def test_from_gam_string():
    s = """\
2
3 2

3 2 0 3 5 6 3 2 3 2 6 1"""

    g = from_gam_string(s)

    expected = NormalFormGame([
        [(3, 3), (3, 2)],
        [(2, 2), (5, 6)],
        [(0, 3), (6, 1)],
    ])

    assert_array_equal(g.payoff_profile_array, expected.payoff_profile_array)


class _FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()


def test_from_gam_url():
    s = """\
2
3 2

3 2 0 3 5 6 3 2 3 2 6 1"""

    def fake_urlopen(url):
        return _FakeResponse(s.encode("utf-8"))

    with patch("urllib.request.urlopen", fake_urlopen):
        g_url = from_gam_url("http://example.com/game.gam")

    g_str = from_gam_string(s)
    assert_array_equal(g_url.payoff_profile_array, g_str.payoff_profile_array)
