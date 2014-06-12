"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: solution_ree_ex3.py
Authors: Chase Coleman, Spencer Lyon, Thomas Sargent, John Stachurski
Solves an exercise from the rational expectations module
"""

from __future__ import print_function
import numpy as np
from quantecon.lqcontrol import LQ
from solution_ree_ex1 import a0, a1, beta, gamma

# == Formulate the planner's LQ problem == #

A = np.array([[1, 0], [0, 1]])
B = np.array([[1], [0]])
R = -np.array([[a1 / 2, -a0 / 2], [-a0 / 2, 0]])
Q = - gamma / 2

# == Solve for the optimal policy == #

lq = LQ(Q, R, A, B, beta=beta)
P, F, d = lq.stationary_values()

# == Print the results == #

F = F.flatten()
kappa0, kappa1 = -F[1], 1 - F[0]
print(kappa0, kappa1)
