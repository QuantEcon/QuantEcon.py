"""
Utilities for converting between representations of games.

Currently supports reading and writing the GameTracer `.gam` text format
[1]_.

Examples
--------
Create a QuantEcon NormalFormGame from a .gam file storing a 3-player
Minimum Effort Game:

>>> import os
>>> import quantecon.game_theory as gt
>>> filepath = os.path.dirname(gt.__file__)
>>> filepath = os.path.join(filepath, 'tests', 'game_files',
...                         'minimum_effort_game.gam')
>>> nfg = gt.from_gam(filepath)
>>> print(nfg)
3-player NormalFormGame with payoff profile array:
[[[[  1.,   1.,   1.],   [  1.,   1.,  -9.],   [  1.,   1., -19.]],
  [[  1.,  -9.,   1.],   [  1.,  -9.,  -9.],   [  1.,  -9., -19.]],
  [[  1., -19.,   1.],   [  1., -19.,  -9.],   [  1., -19., -19.]]],
<BLANKLINE>
 [[[ -9.,   1.,   1.],   [ -9.,   1.,  -9.],   [ -9.,   1., -19.]],
  [[ -9.,  -9.,   1.],   [  2.,   2.,   2.],   [  2.,   2.,  -8.]],
  [[ -9., -19.,   1.],   [  2.,  -8.,   2.],   [  2.,  -8.,  -8.]]],
<BLANKLINE>
 [[[-19.,   1.,   1.],   [-19.,   1.,  -9.],   [-19.,   1., -19.]],
  [[-19.,  -9.,   1.],   [ -8.,   2.,   2.],   [ -8.,   2.,  -8.]],
  [[-19., -19.,   1.],   [ -8.,  -8.,   2.],   [  3.,   3.,   3.]]]]

References
----------
.. [1] Ben Blum, Daphne Koller, Christian Shelton, "Game Theory:
   GameTracer," http://dags.stanford.edu/Games/gametracer.html

"""
import io
import math
import numbers
from fractions import Fraction
import numpy as np
from .normal_form_game import Player, NormalFormGame


_LAYOUTS = ('player-major', 'profile-major')
_LAYOUT_ERROR = "layout must be 'player-major' or 'profile-major' (got {!r})"


class PayoffVector:
    """
    Intermediate representation that stores the payoffs of an N-player
    game in a single flat 1-dim array, in one of the two orders in which
    game files list them:

    'player-major' (as in the GameTracer .gam format)
        All the payoffs to player 0, then those to player 1, ..., then
        those to player N-1. Within each block, action profiles are
        ordered with player 0 varying fastest, then player 1, ...,
        player N-1 (i.e., column-major order).

    'profile-major' (as in the Gambit .nfg format)
        The payoffs to players 0, ..., N-1 at the first action profile,
        then those at the second action profile, and so on. Action
        profiles are ordered with player 0 varying fastest, then player
        1, ..., player N-1 (i.e., column-major order).

    Parameters
    ----------
    nums_actions : array_like(int, ndim=1)
        Numbers of actions, one for each player.

    payoffs : array_like(ndim=1)
        Payoffs, of length prod(nums_actions) * N, in the order `layout`.

    layout : {'player-major', 'profile-major'}
        Order in which `payoffs` lists the payoffs.

    Attributes
    ----------
    N : scalar(int)
        Number of players.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    payoffs : ndarray(ndim=1)
        Array storing the payoffs in the order `layout`.

    layout : str
        Order in which `payoffs` lists the payoffs.

    """
    def __init__(self, nums_actions, payoffs, *, layout):
        nums_actions = tuple(nums_actions)
        if len(nums_actions) == 0:
            raise ValueError('nums_actions must be a non-empty iterable ' +
                             'of positive integers')

        for n in nums_actions:
            if not isinstance(n, numbers.Integral):
                raise TypeError('nums_actions must contain only integers')
            if n <= 0:
                raise ValueError('all nums_actions must be positive')

        self.nums_actions = tuple(int(n) for n in nums_actions)
        self.N = len(self.nums_actions)

        if layout not in _LAYOUTS:
            raise ValueError(_LAYOUT_ERROR.format(layout))
        self.layout = layout

        payoffs = np.ascontiguousarray(payoffs)
        if payoffs.ndim != 1:
            raise ValueError('payoffs must be a 1-dim array_like')

        expected = math.prod(self.nums_actions) * self.N  # no overflow
        if payoffs.size != expected:
            raise ValueError(
                f'payoffs length mismatch: expected {expected}, ' +
                f'got {payoffs.size}'
            )

        self.payoffs = payoffs

    def _player_block(self, i):
        # The payoffs to player i as an array indexed by the action
        # profile; a view of `payoffs`. This is the only place where the
        # layout matters.
        if self.layout == 'player-major':
            shape = self.nums_actions + (self.N,)
            return self.payoffs.reshape(shape, order='F')[..., i]
        else:
            shape = (self.N,) + self.nums_actions
            return self.payoffs.reshape(shape, order='F')[i, ...]

    @classmethod
    def from_normal_form_game(cls, g, *, layout, dtype=None):
        """
        Construct a PayoffVector from a NormalFormGame `g`.

        Parameters
        ----------
        g : NormalFormGame
            NormalFormGame instance.

        layout : {'player-major', 'profile-major'}
            Order in which the payoffs are stored.

        dtype : data-type, optional(default=None)
            Data type of the payoff array. If None, default to the
            `dtype` attribute of `g`.

        Returns
        -------
        PayoffVector
            The PayoffVector representation of `g`.

        Examples
        --------
        >>> player0 = Player([[0, 3], [1, 4], [2, 5]])
        >>> player1 = Player([[6, 7, 8], [9, 10, 11]])
        >>> g = NormalFormGame((player0, player1))
        >>> print(g)
        2-player NormalFormGame with payoff profile array:
        [[[ 0,  6],  [ 3,  9]],
         [[ 1,  7],  [ 4, 10]],
         [[ 2,  8],  [ 5, 11]]]
        >>> p = PayoffVector.from_normal_form_game(g, layout='player-major')
        >>> p.payoffs
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
        >>> p = PayoffVector.from_normal_form_game(g, layout='profile-major')
        >>> p.payoffs
        array([ 0,  6,  1,  7,  2,  8,  3,  9,  4, 10,  5, 11])

        """
        N = g.N
        if dtype is None:
            dtype = g.dtype

        payoffs = np.empty(math.prod(g.nums_actions) * N, dtype=dtype)
        p = cls(g.nums_actions, payoffs, layout=layout)

        for i, player in enumerate(g.players):
            p._player_block(i)[...] = player.payoff_array.transpose(
                (*range(N-i, N), *range(N-i))
            )

        return p

    def to_layout(self, layout, dtype=None):
        """
        Return a new PayoffVector with the payoffs in the order `layout`.
        The payoffs are copied.

        Parameters
        ----------
        layout : {'player-major', 'profile-major'}
            Order in which the payoffs are stored.

        dtype : data-type, optional(default=None)
            Data type of the payoff array. If None, default to the data
            type of the `payoffs` attribute.

        Returns
        -------
        PayoffVector
            The PayoffVector with the payoffs in the order `layout`.

        Examples
        --------
        >>> p = PayoffVector((3, 2), np.arange(12), layout='player-major')
        >>> p.to_layout('profile-major').payoffs
        array([ 0,  6,  1,  7,  2,  8,  3,  9,  4, 10,  5, 11])

        """
        if dtype is None:
            dtype = self.payoffs.dtype

        payoffs = np.empty(self.payoffs.size, dtype=dtype)
        p = type(self)(self.nums_actions, payoffs, layout=layout)

        for i in range(self.N):
            p._player_block(i)[...] = self._player_block(i)

        return p

    def to_normal_form_game(self, dtype=None):
        """
        Construct a NormalFormGame from self.

        Parameters
        ----------
        dtype : data-type, optional(default=None)
            Data type of the players' payoff arrays. If None, default to
            the data type of the `payoffs` attribute.

        Returns
        -------
        NormalFormGame
            The NormalFormGame represented by self.

        Examples
        --------
        >>> nums_actions = (3, 2)
        >>> payoffs = np.arange(12)
        >>> p = PayoffVector(nums_actions, payoffs, layout='player-major')
        >>> print(p.to_normal_form_game())
        2-player NormalFormGame with payoff profile array:
        [[[ 0,  6],  [ 3,  9]],
         [[ 1,  7],  [ 4, 10]],
         [[ 2,  8],  [ 5, 11]]]
        >>> p = PayoffVector(nums_actions, payoffs, layout='profile-major')
        >>> print(p.to_normal_form_game())
        2-player NormalFormGame with payoff profile array:
        [[[ 0,  1],  [ 6,  7]],
         [[ 2,  3],  [ 8,  9]],
         [[ 4,  5],  [10, 11]]]

        """
        N = self.N
        players = tuple(
            Player(
                np.array(  # always a copy: no aliasing with `payoffs`
                    self._player_block(i).transpose((*range(i, N), *range(i))),
                    dtype=dtype, order='C'
                )
            ) for i in range(N)
        )

        return NormalFormGame(players)


def _str2num(s):
    """
    Convert string to appropriate numeric type.

    Parameters
    ----------
    s : str
        String representation of a number: an integer, a decimal with
        an optional exponent, or a rational `n/d`.

    Returns
    -------
    int or float
        Integer if `s` is written as an integer (digits with an optional
        sign), otherwise float. A rational is converted to the nearest
        float.

    """
    try:
        return int(s)
    except ValueError:
        pass
    if '/' in s:
        try:
            return float(Fraction(s))
        except ZeroDivisionError as err:
            raise ValueError(f'zero denominator: {s!r}') from err
    return float(s)


class GAMReader:
    """
    Parser for the GameTracer .gam format.

    """
    @classmethod
    def from_file(cls, file_path):
        """
        Read from a .gam format file.

        Parameters
        ----------
        file_path : str
            Path to the .gam file.

        Returns
        -------
        NormalFormGame
            The game described by the .gam file.

        """
        with open(file_path, 'r') as f:
            string = f.read()
        return cls._parse(string)

    @classmethod
    def from_url(cls, url):
        """
        Read from a URL.

        Parameters
        ----------
        url : str
            String containing a URL of the .gam file.

        Returns
        -------
        NormalFormGame
            The game described by the .gam file.

        """
        import urllib.request
        with urllib.request.urlopen(url) as response:
            string = response.read().decode()
        return cls._parse(string)

    @classmethod
    def from_string(cls, string):
        """
        Read from a .gam format string.

        Parameters
        ----------
        string : str
            String in .gam format.

        Returns
        -------
        NormalFormGame
            The game described by the .gam string.

        """
        return cls._parse(string)

    @staticmethod
    def _parse(string):
        tokens = string.split()
        if not tokens:
            raise ValueError('empty .gam input')
        pos = 0

        # N
        tok = tokens[pos]
        try:
            N = int(tok)
        except ValueError as err:
            raise ValueError(f'invalid N token: {tok!r}') from err
        pos += 1

        if N <= 0:
            raise ValueError('N must be a positive integer')

        # nums_actions
        if len(tokens) < pos + N:
            got = max(0, len(tokens) - pos)
            raise ValueError(
                f'incomplete header: expected {N} action counts, got {got}'
            )

        try:
            nums_actions = tuple(int(tok) for tok in tokens[pos:pos+N])
        except ValueError as err:
            raise ValueError('invalid action count token in header') from err
        pos += N

        # payoffs
        payoffs = np.array([_str2num(tok) for tok in tokens[pos:]])

        p = PayoffVector(nums_actions, payoffs, layout='player-major')
        return p.to_normal_form_game()


class GAMWriter:
    """
    Serializer for the GameTracer .gam format.

    """
    @classmethod
    def to_file(cls, g, file_path):
        """
        Write `g` to a file in GameTracer .gam format.

        Parameters
        ----------
        g : NormalFormGame
            NormalFormGame instance to write.

        file_path : str
            Path to the file to write to.

        """
        with open(file_path, 'w') as f:
            f.write(cls._dump(g) + '\n')

    @classmethod
    def to_string(cls, g):
        """
        Return the GameTracer .gam string representation of `g`.

        Parameters
        ----------
        g : NormalFormGame
            NormalFormGame instance to convert.

        Returns
        -------
        str
            The .gam format string representation of `g`.

        """
        return cls._dump(g)

    @staticmethod
    def _dump(g):
        p = PayoffVector.from_normal_form_game(g, layout='player-major')

        buf = io.StringIO()

        buf.write(str(p.N))
        buf.write('\n')
        buf.write(' '.join(map(str, p.nums_actions)))
        buf.write('\n\n')

        payoffs = p.payoffs
        if payoffs.dtype == np.bool_:
            # Written as 0 and 1, not True and False
            payoffs = payoffs.astype(int)

        if np.issubdtype(payoffs.dtype, np.floating):
            # Shortest representation that round-trips, without exponent
            def fmt(x):
                return np.format_float_positional(x, trim='.')
        else:
            fmt = str

        buf.write(' '.join(map(fmt, payoffs)))

        return buf.getvalue().rstrip()


def from_gam(filename: str) -> NormalFormGame:
    """
    Read a GameTracer .gam file and return a NormalFormGame.

    Parameters
    ----------
    filename : str
        Path to .gam file.

    Returns
    -------
    NormalFormGame
        The game described by the .gam file.

    Examples
    --------
    Save a .gam format string in a temporary file:

    >>> import tempfile
    >>> fname = tempfile.mkstemp()[1]
    >>> with open(fname, mode='w') as f:
    ...       _ = f.write(\"\"\"\\
    ... 2
    ... 3 2
    ...
    ... 3 2 0 3 5 6 3 2 3 2 6 1\"\"\")

    Read the file:

    >>> g = from_gam(fname)
    >>> print(g)
    2-player NormalFormGame with payoff profile array:
    [[[3, 3],  [3, 2]],
     [[2, 2],  [5, 6]],
     [[0, 3],  [6, 1]]]

    """
    return GAMReader.from_file(filename)


def from_gam_string(string):
    """
    Read a .gam format string and return a NormalFormGame.

    Parameters
    ----------
    string : str
        String in .gam format.

    Returns
    -------
    NormalFormGame
        The game described by the .gam string.

    Examples
    --------
    >>> string = \"\"\"\\
    ... 2
    ... 3 2
    ...
    ... 3 2 0 3 5 6 3 2 3 2 6 1\"\"\"
    >>> g = from_gam_string(string)
    >>> print(g)
    2-player NormalFormGame with payoff profile array:
    [[[3, 3],  [3, 2]],
     [[2, 2],  [5, 6]],
     [[0, 3],  [6, 1]]]

    """
    return GAMReader.from_string(string)


def from_gam_url(url):
    """
    Read a GameTracer .gam file from a URL and return a NormalFormGame.

    Parameters
    ----------
    url : str
        String containing a URL of the .gam file.

    Returns
    -------
    NormalFormGame
        The game described by the .gam file.

    """
    return GAMReader.from_url(url)


def to_gam(g, file_path=None):
    """
    Write a NormalFormGame to a file in .gam format.

    Parameters
    ----------
    g : NormalFormGame

    file_path : str, optional(default=None)
        Path to the file to write to. If None, the result is returned as
        a string.

    Returns
    -------
    None or str

    """
    if file_path is None:
        return GAMWriter.to_string(g)
    return GAMWriter.to_file(g, file_path)
