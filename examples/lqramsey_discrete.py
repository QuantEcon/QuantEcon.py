"""
Filename: lqramsey_discrete.py
Authors: Thomas Sargent, Doc-Jin Jang, Jeong-hun Choi, John Stachurski

LQ Ramsey model with discrete exogenous process.

"""
from numpy import array
import lqramsey

# == Parameters == #
beta = 1 / 1.05
P = array([[0.8, 0.2, 0.0],
           [0.0, 0.5, 0.5],
           [0.0, 0.0, 1.0]])
# == Possible states of the world == #
# Each column is a state of the world. The rows are [g d b s 1]
x_vals = array([[0.5, 0.5, 0.25],
                [0.0, 0.0, 0.0],
                [2.2, 2.2, 2.2],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0]])
Sg = array((1, 0, 0, 0, 0)).reshape(1, 5)
Sd = array((0, 1, 0, 0, 0)).reshape(1, 5)
Sb = array((0, 0, 1, 0, 0)).reshape(1, 5)
Ss = array((0, 0, 0, 1, 0)).reshape(1, 5)

economy = lqramsey.Economy(beta=beta,
                           Sg=Sg,
                           Sd=Sd,
                           Sb=Sb,
                           Ss=Ss,
                           discrete=True,
                           proc=(P, x_vals))

T = 15
path = lqramsey.compute_paths(T, economy)
lqramsey.gen_fig_1(path)
