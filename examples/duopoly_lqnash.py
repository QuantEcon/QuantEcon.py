"""
Filename: lqnash.py
Authors: Chase Coleman, Thomas Sargent

This file provides an example of a Markov Perfect Equilibrium for a
simple duopoly example.

See the lecture at http://quant-econ.net/py/markov_perf.html for a
description of the model.

"""
from __future__ import division
from numpy import array, eye
from quantecon.lqnash import nnash


# ---------------------------------------------------------------------#
# Set up parameter values and LQ matrices
# Remember state is x_t = [1, y_{1, t}, y_{2, t}] and
# control is u_{i, t} = [y_{i, t+1} - y_{i, t}]
# ---------------------------------------------------------------------#
a0 = 10.
a1 = 1.
beta = 1.
d = .5

a = eye(3)
b1 = array([[0.], [1.], [0.]])
b2 = array([[0.], [0.], [1.]])

r1 = array([[a0, 0., 0.],
            [0., -a1, -a1/2.],
            [0, -a1/2., 0.]])

r2 = array([[a0, 0., 0.],
            [0., 0., -a1/2.],
            [0, -a1/2., -a1]])

q1 = array([[-.5*d]])
q2 = array([[-.5*d]])


# ---------------------------------------------------------------------#
# Solve using QE's nnash function
# ---------------------------------------------------------------------#

f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, 0., 0., 0., 0., 0., 0.,
                       tol=1e-8, max_iter=1000)
