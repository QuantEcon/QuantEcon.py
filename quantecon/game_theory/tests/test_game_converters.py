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
    Player, NormalFormGame, GAMWriter, to_gam, from_gam_string, from_gam_url
)
from quantecon.game_theory.game_converters import (
    PayoffProfileMatrix, _str2num
)


# PayoffProfileMatrix #

class TestPayoffProfileMatrix:
    """Golden test for PayoffProfileMatrix"""

    def setup_method(self):
        nums_actions = (2, 3, 4)
        N = len(nums_actions)
        na = np.prod(nums_actions)

        A0 = np.arange(na).reshape(nums_actions, order='F')
        A1 = np.arange(100, 100+na).reshape(nums_actions, order='F')
        A2 = np.arange(200, 200+na).reshape(nums_actions, order='F')

        self.payoffs4d = np.stack([A0, A1, A2], axis=N)
        self.payoffs2d = self.payoffs4d.reshape((na, N), order='F')
        # Player-major (.gam) and profile-major (.nfg)
        self.payoffs1d_F = np.hstack(
            [A.ravel(order='F') for A in [A0, A1, A2]]
        )
        self.payoffs1d_C = self.payoffs2d.ravel(order='C')

        self.N = N
        self.nums_actions = nums_actions

    def test_init(self):
        for order, payoffs1d in [('F', self.payoffs1d_F),
                                 ('C', self.payoffs1d_C)]:
            p = PayoffProfileMatrix(self.nums_actions, payoffs1d, order=order)

            assert_(p.N == self.N)
            assert_(p.nums_actions == self.nums_actions)
            assert_array_equal(p.payoffs, self.payoffs2d)
            assert_array_equal(p.as_vector(order='F'), self.payoffs1d_F)
            assert_array_equal(p.as_vector(order='C'), self.payoffs1d_C)

    def test_from_normal_form_game(self):
        g = NormalFormGame(self.payoffs4d)
        p = PayoffProfileMatrix.from_normal_form_game(g)

        assert_(p.N == self.N)
        assert_(p.nums_actions == self.nums_actions)
        assert_array_equal(p.payoffs, self.payoffs2d)

    def test_to_normal_form_game(self):
        for order, payoffs1d in [('F', self.payoffs1d_F),
                                 ('C', self.payoffs1d_C)]:
            p = PayoffProfileMatrix(self.nums_actions, payoffs1d, order=order)
            g = p.to_normal_form_game()
            assert_array_equal(g.payoff_profile_array, self.payoffs4d)


def test_payoffprofilematrix_2p():
    # 3x2 game with payoff profiles, in column-major order over the
    # action profiles:
    #   (0,0): (3,2)  (1,0): (0,6)  (2,0): (2,1)
    #   (0,1): (1,3)  (1,1): (4,0)  (2,1): (5,4)
    g = NormalFormGame((Player([[3, 1], [0, 4], [2, 5]]),
                        Player([[2, 6, 1], [3, 0, 4]])))
    payoffs_F = [3, 0, 2, 1, 4, 5, 2, 6, 1, 3, 0, 4]  # player-major
    payoffs_C = [3, 2, 0, 6, 2, 1, 1, 3, 4, 0, 5, 4]  # profile-major

    p = PayoffProfileMatrix.from_normal_form_game(g)
    assert_array_equal(p.as_vector(order='F'), payoffs_F)
    assert_array_equal(p.as_vector(order='C'), payoffs_C)

    for order, payoffs in [('F', payoffs_F), ('C', payoffs_C)]:
        p = PayoffProfileMatrix((3, 2), payoffs, order=order)
        assert_array_equal(p.to_normal_form_game().payoff_profile_array,
                           g.payoff_profile_array)


def test_payoffprofilematrix_roundtrip():
    for ns in [(4, 3), (2, 2, 3, 2)]:
        N = len(ns)
        seed = 12345
        rng = np.random.default_rng(seed)
        payoffs = rng.integers(low=0, high=100, size=(*ns, N), dtype=np.int64)
        g = NormalFormGame(payoffs)
        p = PayoffProfileMatrix.from_normal_form_game(g)
        g1 = p.to_normal_form_game()

        p_32 = PayoffProfileMatrix.from_normal_form_game(g, dtype=np.int32)
        g2 = p_32.to_normal_form_game()
        g3 = p_32.to_normal_form_game(dtype=np.int64)

        assert_(p_32.payoffs.dtype == np.int32)
        assert_(g2.dtype == np.int32)
        assert_(g3.dtype == np.int64)

        for g_new in [g1, g2, g3]:
            assert_(g_new.N == g.N)
            assert_(g_new.nums_actions == g.nums_actions)
            for i in range(N):
                assert_array_equal(g_new.players[i].payoff_array,
                                   g.players[i].payoff_array)

        # Conversion between the orders
        for order in ['F', 'C']:
            p_new = PayoffProfileMatrix(ns, p.as_vector(order=order),
                                        order=order)
            assert_array_equal(p_new.payoffs, p.payoffs)


def test_payoffprofilematrix_1p():
    payoffs = [1., 2., 3.]
    nums_actions = (3,)

    p0 = PayoffProfileMatrix(nums_actions, payoffs, order='F')

    g = NormalFormGame((Player(payoffs),))
    p1 = PayoffProfileMatrix.from_normal_form_game(g)

    for p in [p0, p1]:
        assert_(p.N == 1)
        assert_(p.nums_actions == nums_actions)
        assert_array_equal(p.as_vector(order='F'), payoffs)
        assert_array_equal(p.as_vector(order='C'), payoffs)


def test_payoffprofilematrix_views():
    payoffs = np.arange(12)
    p = PayoffProfileMatrix((3, 2), payoffs, order='F')

    # `payoffs` and the player-major vector are views of the input
    assert_(np.shares_memory(p.payoffs, payoffs))
    assert_(np.shares_memory(p.as_vector(order='F'), payoffs))
    assert_(not np.shares_memory(p.as_vector(order='C'), payoffs))

    # The game does not alias the input, even for one player
    p1 = PayoffProfileMatrix((3,), np.arange(3), order='F')
    g = p1.to_normal_form_game()
    assert_(not np.shares_memory(g.players[0].payoff_array, p1.payoffs))


def test_invalid_inputs():
    assert_raises(ValueError, PayoffProfileMatrix, (), np.array([]),
                  order='F')
    assert_raises(TypeError, PayoffProfileMatrix, (2, 2.0), np.zeros(8),
                  order='F')
    assert_raises(ValueError, PayoffProfileMatrix, (2, 0), np.zeros(0),
                  order='F')
    assert_raises(ValueError, PayoffProfileMatrix, (2, 2), np.zeros(7),
                  order='F')
    # np.prod would overflow and give 0
    assert_raises(ValueError, PayoffProfileMatrix, (2**62, 2**62),
                  np.zeros(2), order='F')
    # order is required
    assert_raises(TypeError, PayoffProfileMatrix, (2, 2), np.zeros(8))


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


def test_gam_writer_float_precision():
    # Values that need more than 8 significant digits, and large and
    # small values
    payoffs = [1/3, np.pi, 0.1 + 0.2, 1e10, 1e-7, -2.5e22, 123456.789, 1.]
    for dtype in [np.float64, np.float32]:
        a = np.array(payoffs, dtype=dtype).reshape(2, 2, 2)
        g = NormalFormGame(a)

        s = to_gam(g)
        payoff_tokens = s.split()[3:]

        # Written without exponent
        for tok in payoff_tokens:
            assert_('e' not in tok.lower())

        # Read back, the original values are recovered when cast to the
        # source dtype
        g2 = from_gam_string(s)
        assert_array_equal(
            g2.payoff_profile_array.astype(dtype), g.payoff_profile_array
        )


def test_gam_writer_float_boundary_values():
    for dtype in [np.float64, np.float32]:
        info = np.finfo(dtype)
        payoffs = np.array([
            0.0,
            -0.0,
            np.nextafter(dtype(0), dtype(1)),  # smallest subnormal
            info.tiny,
            info.max,
            -info.max,
            np.nextafter(dtype(1), dtype(2)),
            1.0,
        ], dtype=dtype).reshape(2, 2, 2)
        g = NormalFormGame(payoffs)

        g2 = from_gam_string(to_gam(g))
        restored = g2.payoff_profile_array.astype(dtype)

        assert_array_equal(restored, g.payoff_profile_array)
        assert_array_equal(
            np.signbit(restored), np.signbit(g.payoff_profile_array)
        )


def test_gam_writer_print_options():
    # The output does not depend on the print options of NumPy
    payoffs = np.array([1/3, np.pi, 1e10, 1e-7, 2., 3., 4., 5.])
    g = NormalFormGame(payoffs.reshape(2, 2, 2))
    s_desired = to_gam(g)

    with np.printoptions(
        precision=2, formatter={'float_kind': lambda x: 'BAD'}
    ):
        assert_string_equal(to_gam(g), s_desired)


def test_gam_writer_bool():
    A = np.array([[True, False], [False, True]])
    g = NormalFormGame((Player(A), Player(A)))

    s = to_gam(g)
    assert_string_equal(s, """\
2
2 2

1 0 0 1 1 0 0 1""")

    g2 = from_gam_string(s)
    assert_array_equal(g2.payoff_profile_array, g.payoff_profile_array)


# GAMReader/from_gam #

def test_str2num():
    for s, x in [('3', 3), ('-3', -3), ('+3', 3)]:
        assert_(_str2num(s) == x)
        assert_(isinstance(_str2num(s), int))

    # Float even if the value is an integer, unless written as an integer
    for s, x in [('0.5', 0.5), ('.5', 0.5), ('3.', 3.), ('1e3', 1000.),
                 ('1E-2', 0.01),
                 ('1/3', 1/3), ('-1/3', -1/3), ('+1/3', 1/3), ('6/4', 1.5),
                 ('2/1', 2.)]:
        assert_(_str2num(s) == x)
        assert_(isinstance(_str2num(s), float))

    for s in ['1/0', '1/2/3', '0.5/2', '/3', '1/', 'abc', '']:
        assert_raises(ValueError, _str2num, s)


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


def test_from_gam_string_number_formats():
    # Exponent with and without decimal point, signs
    s = """\
2
2 2

1e3 1E3 1.0e3 +1.5e+2 -2.5E-1 +3 -4 .5"""

    g = from_gam_string(s)
    payoffs = [1000., 1000., 1000., 150., -0.25, 3., -4., 0.5]

    assert_(g.dtype == np.float64)
    assert_array_equal(
        PayoffProfileMatrix.from_normal_form_game(g).as_vector(order='F'),
        payoffs
    )

    # Integers only, with signs
    s = """\
2
2 2

1 +2 -3 4 5 6 7 8"""

    g = from_gam_string(s)

    assert_(np.issubdtype(g.dtype, np.integer))
    assert_array_equal(
        PayoffProfileMatrix.from_normal_form_game(g).as_vector(order='F'),
        [1, 2, -3, 4, 5, 6, 7, 8]
    )

    assert_raises(ValueError, from_gam_string, "2\n2 2\n\n1 2 3 4 5 6 7 x")


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
