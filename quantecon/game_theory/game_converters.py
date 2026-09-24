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


class PayoffProfileMatrix:
    """
    Intermediate representation of the payoffs of an N-player game: the
    `payoff_profile_array` of a NormalFormGame with the action-profile
    axes flattened into one, in column-major order (player 0 varying
    fastest, then player 1, ..., player N-1).

    The .gam and .nfg formats store the payoffs as a 1-dim array; see
    `as_vector` for the two orders in which this matrix is flattened.

    Parameters
    ----------
    nums_actions : array_like(int, ndim=1)
        Numbers of actions, one for each player.

    payoffs : array_like(ndim=1)
        Payoffs, of length prod(nums_actions) * N, in the order `order`.

    order : {'C', 'F'}
        Order in which `payoffs` lists the payoffs; see `as_vector`.

    Attributes
    ----------
    N : scalar(int)
        Number of players.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    payoffs : ndarray(ndim=2)
        Array of shape (prod(nums_actions), N), Fortran-contiguous, whose
        [a, i] entry is the payoff to player i at the a-th action
        profile.

    """
    def __init__(self, nums_actions, payoffs, *, order):
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

        payoffs = np.asarray(payoffs)
        na = math.prod(self.nums_actions)  # Python int: no overflow
        expected = na * self.N
        if payoffs.size != expected:
            raise ValueError(
                f'payoffs length mismatch: expected {expected}, ' +
                f'got {payoffs.size}'
            )

        # Fortran-contiguous, so that the payoffs of each player are
        # contiguous; a view of `payoffs` if order='F'
        self.payoffs = np.asfortranarray(
            payoffs.reshape((na, self.N), order=order)
        )

    def as_vector(self, order):
        """
        Return the payoffs as a 1-dim array.

        Parameters
        ----------
        order : {'C', 'F'}
            'F': all the payoffs to player 0, then those to player 1,
            ..., each in the order of the action profiles (player-major,
            as in the .gam format). 'C': the payoffs to players 0, ...,
            N-1 at the first action profile, then those at the second
            action profile, ... (profile-major, as in the .nfg format).

        Returns
        -------
        ndarray(ndim=1)
            A view of `payoffs` if `order` is 'F', otherwise a copy, as
            with `ndarray.ravel`.

        """
        return self.payoffs.ravel(order=order)

    def _player_block(self, i):
        # The payoffs to player i as an array indexed by the action
        # profile; a view of `payoffs`
        return self.payoffs[:, i].reshape(self.nums_actions, order='F')

    @classmethod
    def from_normal_form_game(cls, g, dtype=None):
        """
        Construct a PayoffProfileMatrix from a NormalFormGame `g`.

        Parameters
        ----------
        g : NormalFormGame
            NormalFormGame instance.

        dtype : data-type, optional(default=None)
            Data type of the payoff array. If None, default to the
            `dtype` attribute of `g`.

        Returns
        -------
        PayoffProfileMatrix
            The PayoffProfileMatrix representation of `g`.

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
        >>> p = PayoffProfileMatrix.from_normal_form_game(g)
        >>> p.payoffs
        array([[ 0,  6],
               [ 1,  7],
               [ 2,  8],
               [ 3,  9],
               [ 4, 10],
               [ 5, 11]])
        >>> p.as_vector(order='F')
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
        >>> p.as_vector(order='C')
        array([ 0,  6,  1,  7,  2,  8,  3,  9,  4, 10,  5, 11])

        """
        N = g.N
        if dtype is None:
            dtype = g.dtype

        payoffs = np.empty(math.prod(g.nums_actions) * N, dtype=dtype)
        p = cls(g.nums_actions, payoffs, order='F')  # a view of payoffs

        for i, player in enumerate(g.players):
            p._player_block(i)[...] = player.payoff_array.transpose(
                (*range(N-i, N), *range(N-i))
            )

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
        >>> p = PayoffProfileMatrix(nums_actions, payoffs, order='F')
        >>> g = p.to_normal_form_game()
        >>> print(g)
        2-player NormalFormGame with payoff profile array:
        [[[ 0,  6],  [ 3,  9]],
         [[ 1,  7],  [ 4, 10]],
         [[ 2,  8],  [ 5, 11]]]
        >>> p = PayoffProfileMatrix(nums_actions, payoffs, order='C')
        >>> g = p.to_normal_form_game()
        >>> print(g)
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

        p = PayoffProfileMatrix(nums_actions, payoffs, order='F')
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
        p = PayoffProfileMatrix.from_normal_form_game(g)

        buf = io.StringIO()

        buf.write(str(p.N))
        buf.write('\n')
        buf.write(' '.join(map(str, p.nums_actions)))
        buf.write('\n\n')

        payoffs = p.as_vector(order='F')  # player-major, as in .gam
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
