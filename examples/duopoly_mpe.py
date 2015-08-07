"""
@authors: Chase Coleman, Thomas Sargent, John Stachurski

Markov Perfect Equilibrium for the simple duopoly example.

See the lecture at http://quant-econ.net/py/markov_perf.html for a
description of the model.
"""

from __future__ import division
import numpy as np
import quantecon as qe

# == Parameters == #
a0    = 10.0
a1    = 2.0
beta  = 0.96
gamma = 12.0

# == In LQ form == #

A  = np.eye(3)

B1 = np.array([[0.], [1.], [0.]])
B2 = np.array([[0.], [0.], [1.]])


R1 = [[0.,    -a0/2,  0.],
      [-a0/2., a1,    a1/2.],
      [0,      a1/2., 0.]]

R2 = [[0.,    0.,   -a0/2],
      [0.,    0.,    a1/2.],
      [-a0/2, a1/2., a1]]

Q1 = Q2 = gamma

S1 = S2 = W1 = W2 = M1 = M2 = 0.0

# == Solve using QE's nnash function == #
F1, F2, P1, P2 = qe.nnash(A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2,
                          beta=beta)

# == Display policies == #
print("Computed policies for firm 1 and firm 2:\n")
print("F1 = {}".format(F1))
print("F2 = {}".format(F2))
print("\n")
