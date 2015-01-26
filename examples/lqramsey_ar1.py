"""
Filename: lqramsey_ar1.py
Authors: Thomas Sargent, Doc-Jin Jang, Jeong-hun Choi, John Stachurski

Example 1: Govt spending is AR(1) and state is (g, 1).

"""

import numpy as np
from numpy import array
import lqramsey

# == Parameters == #
beta = 1 / 1.05
rho, mg = .7, .35
A = np.identity(2)
A[0, :] = rho, mg * (1-rho)
C = np.zeros((2, 1))
C[0, 0] = np.sqrt(1 - rho**2) * mg / 10
Sg = array((1, 0)).reshape(1, 2)
Sd = array((0, 0)).reshape(1, 2)
Sb = array((0, 2.135)).reshape(1, 2)
Ss = array((0, 0)).reshape(1, 2)

economy = lqramsey.Economy(beta=beta,
                           Sg=Sg,
                           Sd=Sd,
                           Sb=Sb,
                           Ss=Ss,
                           discrete=False,
                           proc=(A, C))

T = 50
path = lqramsey.compute_paths(T, economy)
lqramsey.gen_fig_1(path)
