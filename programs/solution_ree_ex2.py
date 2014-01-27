"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: solution_ree_ex2.py
Authors: Chase Coleman, Spencer Lyon, Thomas Sargent, John Stachurski
Solves an exercise from the rational expectations module
"""
from __future__ import print_function
import numpy as np
from lqcontrol import LQ
from solution_ree_ex1 import beta, R, Q, B

candidates = (
          (94.0886298678, 0.923409232937),
          (93.2119845412, 0.984323478873),
          (95.0818452486, 0.952459076301)
             )

for kappa0, kappa1 in candidates:

    # == Form the associated law of motion == #
    A = np.array([[1, 0, 0], [0, kappa1, kappa0], [0, 0, 1]])

    # == Solve the LQ problem for the firm == #
    lq = LQ(Q, R, A, B, beta=beta)
    P, F, d = lq.stationary_values()
    F = F.flatten()
    h0, h1, h2 = -F[2], 1 - F[0], -F[1]

    # == Test the equilibrium condition == #
    if np.allclose((kappa0, kappa1), (h0, h1 + h2)):
        print('Equilibrium pair =', kappa0, kappa1)
        print('(h0, h1, h2) = ', h0, h1, h2)
        break


