"""
Functions for converting between ways of storing games.

Examples
--------

Create a QuantEcon NormalFormGame from a gam file storing
a 3-player Minimum Effort Game

>>> import os
>>> import quantecon.game_theory as gt
>>> filepath = os.path.dirname(gt.__file__)
>>> filepath += "/tests/game_files/minimum_effort_game.gam"
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

"""
import numpy as np
from .normal_form_game import Player, NormalFormGame


def str2num(s):
    if '.' in s:
        return float(s)
    return int(s)


class GAMReader:
    """
    Reader object that converts a game in GameTracer gam format into
    a NormalFormGame.

    """
    @classmethod
    def from_file(cls, file_path):
        """
        Read from a gam format file.

        Parameters
        ----------
        file_path : str
            Path to gam file.

        Returns
        -------
        NormalFormGame

        Examples
        --------
        Save a gam format string in a temporary file:

        >>> import tempfile
        >>> fname = tempfile.mkstemp()[1]
        >>> with open(fname, mode='w') as f:
        ...       f.write(\"\"\"\\
        ... 2
        ... 3 2
        ...
        ... 3 2 0 3 5 6 3 2 3 2 6 1\"\"\")

        Read the file:

        >>> g = GAMReader.from_file(fname)
        >>> print(g)
        2-player NormalFormGame with payoff profile array:
        [[[3, 3],  [3, 2]],
         [[2, 2],  [5, 6]],
         [[0, 3],  [6, 1]]]

        """
        with open(file_path, 'r') as f:
            string = f.read()
        return cls._parse(string)

    @classmethod
    def from_url(cls, url):
        """
        Read from a URL.

        """
        import urllib.request
        with urllib.request.urlopen(url) as response:
            string = response.read().decode()
        return cls._parse(string)

    @classmethod
    def from_string(cls, string):
        """
        Read from a gam format string.

        Parameters
        ----------
        string : str
            String in gam format.

        Returns
        -------
        NormalFormGame

        Examples
        --------
        >>> string = \"\"\"\\
        ... 2
        ... 3 2
        ...
        ... 3 2 0 3 5 6 3 2 3 2 6 1\"\"\"
        >>> g = GAMReader.from_string(string)
        >>> print(g)
        2-player NormalFormGame with payoff profile array:
        [[[3, 3],  [3, 2]],
         [[2, 2],  [5, 6]],
         [[0, 3],  [6, 1]]]

        """
        return cls._parse(string)

    @staticmethod
    def _parse(string):
        tokens = string.split()

        N = int(tokens.pop(0))
        nums_actions = tuple(int(tokens.pop(0)) for _ in range(N))
        payoffs = np.array([str2num(s) for s in tokens])

        na = np.prod(nums_actions)
        payoffs2d = payoffs.reshape(N, na)
        players = [
            Player(
                payoffs2d[i, :].reshape(nums_actions, order='F').transpose(
                    (*range(i, N), *range(i))
                )
            ) for i in range(N)
        ]

        return NormalFormGame(players)


class GAMWriter:
    """
    Writer object that converts a NormalFormgame into a game in
    GameTracer gam format.

    """
    @classmethod
    def to_file(cls, g, file_path):
        """
        Save the GameTracer gam format string representation of the
        NormalFormGame `g` to a file.

        Parameters
        ----------
        g : NormalFormGame

        file_path : str
            Path to the file to write to.

        """
        with open(file_path, 'w') as f:
            f.write(cls._dump(g) + '\n')

    @classmethod
    def to_string(cls, g):
        """
        Return a GameTracer gam format string representing the
        NormalFormGame `g`.

        Parameters
        ----------
        g : NormalFormGame

        Returns
        -------
        str
            String representation in gam format.

        """
        return cls._dump(g)

    @staticmethod
    def _dump(g):
        s = str(g.N) + '\n'
        s += ' '.join(map(str, g.nums_actions)) + '\n\n'

        for i, player in enumerate(g.players):
            payoffs = np.array2string(
                player.payoff_array.transpose(
                    (*range(g.N-i, g.N), *range(g.N-i))
                ).ravel(order='F'))[1:-1]
            s += ' '.join(payoffs.split()) + ' '

        return s.rstrip()


def from_gam(filename: str) -> NormalFormGame:
    """
    Makes a QuantEcon Normal Form Game from a gam file.

    Gam files are described by GameTracer [1]_.

    Parameters
    ----------
    filename : str
        path to gam file.

    Returns
    -------
    NormalFormGame
        The QuantEcon Normal Form Game described by the gam file.

    References
    ----------
    .. [1] Bem Blum, Daphne Kohler, Christian Shelton
       http://dags.stanford.edu/Games/gametracer.html

    """
    return GAMReader.from_file(filename)


def to_gam(g, file_path=None):
    """
    Write a NormalFormGame to a file in gam format.

    Parameters
    ----------
    g : NormalFormGame

    file_path : str, optional(default=None)
        Path to the file to write to. If None, the result is returned as
        a string.

    """
    if file_path is None:
        return GAMWriter.to_string(g)
    return GAMWriter.to_file(g, file_path)
