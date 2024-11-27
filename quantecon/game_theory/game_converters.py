"""
Functions for converting between ways of storing games.

Examples
--------

Create a QuantEcon NormalFormGame from a gam file storing
a 3-player Minimum Effort Game

>>> filepath = "./tests/gam_files/minimum_effort_game.gam"
>>> nfg = qe_nfg_from_gam_file(filepath)
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

from .normal_form_game import NormalFormGame
from itertools import product


def qe_nfg_from_gam_file(filename: str) -> NormalFormGame:
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
    with open(filename, 'r') as file:
        lines = file.readlines()
        combined = [
            token
            for line in lines
            for token in line.split()
        ]

        i = iter(combined)
        players = int(next(i))
        actions = [int(next(i)) for _ in range(players)]

        nfg = NormalFormGame(actions)

        entries = [
            {
                tuple(reversed(action_combination)): float(next(i))
                for action_combination in product(
                    *[range(a) for a in actions])
            }
            for _ in range(players)
        ]

        for action_combination in product(*[range(a) for a in actions]):
            nfg[action_combination] = tuple(
                entries[p][action_combination] for p in range(players)
            )

    return nfg
