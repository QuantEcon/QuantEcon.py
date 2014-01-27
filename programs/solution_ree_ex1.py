"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: solution_ree_ex1.py
Authors: Chase Coleman, Spencer Lyon, Thomas Sargent, John Stachurski
Solves an exercise from the rational expectations module
"""
from __future__ import print_function
import numpy as np
from lqcontrol import LQ


# == Model parameters == #

a0      = 100
a1      = 0.05
beta    = 0.95
gamma   = 10.0

# == Beliefs == #

kappa0  = 95.5
kappa1  = 0.95

# == Formulate the LQ problem == #

A = np.array([[1, 0, 0], [0, kappa1, kappa0], [0, 0, 1]])
B = np.array([1, 0, 0])
B.shape = 3, 1
R = np.array([[0, -a1/2, a0/2], [-a1/2, 0, 0], [a0/2, 0, 0]])
Q = -0.5 * gamma

# == Solve for the optimal policy == #

lq = LQ(Q, R, A, B, beta=beta)
P, F, d = lq.stationary_values()
F = F.flatten()
out1 = "F = [{0:.3f}, {1:.3f}, {2:.3f}]".format(F[0], F[1], F[2])
h0, h1, h2 = -F[2], 1 - F[0], -F[1]
out2 = "(h0, h1, h2) = ({0:.3f}, {1:.3f}, {2:.3f})".format(h0, h1, h2)

print(out1)
print(out2)
